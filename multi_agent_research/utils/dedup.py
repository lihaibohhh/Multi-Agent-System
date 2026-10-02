"""
utils/dedup.py — 信息增量检测模块
====================================

职责：判断最近若干轮搜索是否带来了有效新信息，防止 Supervisor 陷入无效检索循环。

设计原则
--------
1. 解耦：不导入 ResearchState，接受原始 Sequence[dict]，方便单测与跨模块复用。
2. 中文安全：字符级 bigram 替代空格分词，无需外部分词库（jieba 等）。
3. 两层算法，顺序降级：
     Level 0  事件计数快速路径 — 连续 N 轮 new_count == 0 → 信息枯竭（O(n) 精确）
     Level 1  内容 Jaccard 兜底 — bigram 相似度 > threshold → 内容重叠（O(chars)）
4. 利用事件 payload 的 total_count / new_count 分片，不依赖 SearchResult.iteration
   （该字段目前 search_agent.py 未写入，属于已知 Bug，见文末 NOTE）。
5. 可观测：GainCheckResult 携带 reason / detail，调用方无需再拼日志字符串。
6. 保守策略：数据不足或结构异常时一律返回 has_new_info=True，避免误杀有效任务。

TODO Day 12：新增 Level 2 — embedding cosine similarity（需 async 接口）。

NOTE — search_agent.py 配套修复
----------------------------------
search_agent_node 中 new_results 的每条记录缺少 iteration 字段：

    iteration = state.get("iteration_count", 0)
    new_results.append({**r, "iteration": iteration})   # ← 补充此行

本模块当前不依赖该字段，但修复后可为其他下游（Analyst、Writer）提供更丰富的上下文。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 公共配置常量（调用方按需覆盖）
# ─────────────────────────────────────────────

DEFAULT_SIMILARITY_THRESHOLD: float = 0.85  # Jaccard 触发阈值
DEFAULT_MIN_ZERO_ROUNDS: int = 2            # 连续 N 轮 new_count=0 才触发，防网络抖动误判
_NGRAM_SIZE: int = 2                        # 字符级 n-gram 窗口（bigram）
_MAX_CONTENT_CHARS: int = 8_000            # 单轮内容截断上限，防止超长文本拖慢比较


# ─────────────────────────────────────────────
# 返回结构
# ─────────────────────────────────────────────
@dataclass(frozen=True)
class GainCheckResult:
    """
    信息增量检测结果。

    has_new_info=False → Supervisor 应停止派发 search_agent。
    reason / detail    → 直接写入日志，无需调用方二次拼接。

    典型用法::

        result = detect_information_gain(events, results).log()
        if not result:
            # 信息枯竭处理逻辑
            ...
    """
    has_new_info: bool
    reason: str
    detail: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        """令 `if result:` 等价于 `if result.has_new_info:`。"""
        return self.has_new_info

    def log(self, level: int = logging.INFO) -> "GainCheckResult":
        """打印结构化日志并返回自身，支持链式调用。"""
        logger.log(
            level,
            "[InfoGain] has_new_info=%s | %s | detail=%s",
            self.has_new_info,
            self.reason,
            self.detail,
        )
        return self


# ─────────────────────────────────────────────
# 基础算法（可单独单测）
# ─────────────────────────────────────────────
def _char_ngrams(text: str, n: int = _NGRAM_SIZE) -> frozenset[str]:
    """
    字符级 n-gram，中英文通用，无需分词库。

    >>> _char_ngrams("深度学习", 2)
    frozenset({'深度', '度学', '学习'})
    >>> _char_ngrams("ab", 3)   # 文本短于 n，退化为整串
    frozenset({'ab'})
    """
    text = text.strip()[:_MAX_CONTENT_CHARS]
    if not text:
        return frozenset()
    if len(text) < n:
        return frozenset([text])
    return frozenset(text[i: i + n] for i in range(len(text) - n + 1))


def jaccard_similarity(text_a: str, text_b: str) -> float:
    """
    公开接口：计算两段文本的字符 bigram Jaccard 相似度，范围 [0.0, 1.0]。
    空串对空串返回 0.0（保守），而非 1.0。

    可直接用于单测::

        assert jaccard_similarity("abc", "abc") == 1.0
        assert jaccard_similarity("", "abc") == 0.0
    """
    a = _char_ngrams(text_a)
    b = _char_ngrams(text_b)
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


# ─────────────────────────────────────────────
# 内部辅助
# ─────────────────────────────────────────────
def _extract_search_events(events: Sequence[dict]) -> list[dict]:
    """
    从 state["events"] 中提取所有 SearchCompleted 事件，按 iteration 升序。
    iteration 缺失的事件排在最后（防止 sorted 对 None 抛 TypeError）。
    """
    search_events = [
        e for e in events
        if isinstance(e, dict) and e.get("type") == "SearchCompleted"
    ]
    return sorted(
        search_events,
        key=lambda e: (e.get("iteration") is None, e.get("iteration", 0)),
    )


def _safe_payload_counts(event: dict) -> tuple[int, int] | None:
    """
    从单个 SearchCompleted 事件中安全提取 (total_count, new_count)。
    任一字段缺失或类型异常时返回 None（调用方决定如何降级）。
    """
    payload = event.get("payload")
    if not isinstance(payload, dict):
        return None
    total = payload.get("total_count")
    new = payload.get("new_count")
    if total is None or new is None:
        return None
    try:
        return int(total), int(new)
    except (TypeError, ValueError):
        return None


# ─────────────────────────────────────────────
# Level 0 — 事件计数快速路径
# ─────────────────────────────────────────────
def _check_consecutive_zero_counts(
    search_events: list[dict],
    min_rounds: int,
) -> GainCheckResult | None:
    """
    Level 0：最近 min_rounds 轮 new_count 均为 0 → 信息枯竭。

    返回 None 表示"无法判断，交由 Level 1"，原因可能是：
    - 事件数量不足 min_rounds
    - payload 缺字段或类型异常（保守策略：不在此层误判）
    - 存在非零轮次（最近有搜索带来新内容）

    设计说明
    --------
    旧实现只看最后 1 轮，网络抖动导致偶发 new_count=0 时会误触发。
    改为连续 N 轮（默认 2）后，单次异常不会导致误杀任务。
    """
    if len(search_events) < min_rounds:
        logger.debug(
            "[InfoGain/L0] 事件不足 %d 轮（实际 %d），跳过 Level 0",
            min_rounds, len(search_events),
        )
        return None

    recent = search_events[-min_rounds:]
    new_counts: list[int] = []

    for e in recent:
        counts = _safe_payload_counts(e)
        if counts is None:
            logger.debug("[InfoGain/L0] payload 缺字段，跳过 Level 0")
            return None
        new_counts.append(counts[1])  # counts = (total, new)

    if all(c == 0 for c in new_counts):
        iters = [e.get("iteration", "?") for e in recent]
        return GainCheckResult(
            has_new_info=False,
            reason=(
                f"连续 {min_rounds} 轮搜索新增 0 条"
                f"（迭代 {iters}，new_counts={new_counts}）"
            ),
            detail={
                "level": 0,
                "iterations": iters,
                "new_counts": new_counts,
                "min_zero_rounds": min_rounds,
            },
        )

    # 最近有非零轮次 → 交给 Level 1 做内容级验证
    return None


# ─────────────────────────────────────────────
# Level 1 — 内容 Jaccard 相似度
# ─────────────────────────────────────────────
#: 数据不足或结构异常时的保守返回值，避免误杀有效任务。
_CONSERVATIVE_OK = GainCheckResult(
    has_new_info=True,
    reason="数据不足或结构异常，默认有新信息（保守策略）",
    detail={"level": 1, "reason": "fallback_conservative"},
)


def _check_content_similarity(
    search_events: list[dict],
    search_results: Sequence[dict],
    threshold: float,
) -> GainCheckResult:
    """
    Level 1：比较最近两轮新增内容的 bigram Jaccard 相似度。

    关键设计：利用事件 payload 的 total_count / new_count 分片定位每轮结果，
    **完全不依赖 SearchResult.iteration 字段**（该字段目前未写入，是已知 Bug）。

    分片逻辑示意
    ~~~~~~~~~~~~
    search_results（追加有序）：
        [iter0_r0, iter0_r1, | iter1_r0, iter1_r1, iter1_r2, | iter2_r0]
         ←── prev_total=2 ──→   ←────── last_total=5 ──────→

    last_slice = results[last_total - last_new : last_total]
    prev_slice = results[prev_total - prev_new : prev_total]
    """
    if len(search_events) < 2:
        return _CONSERVATIVE_OK

    results_list = list(search_results)
    if len(results_list) < 2:
        return _CONSERVATIVE_OK

    # 最新一轮事件、导数第二轮事件
    last_ev, prev_ev = search_events[-1], search_events[-2]
    last_counts = _safe_payload_counts(last_ev)  # (total_count, new_count)
    prev_counts = _safe_payload_counts(prev_ev)

    if last_counts is None or prev_counts is None:
        return GainCheckResult(
            has_new_info=True,
            reason="事件 payload 缺少 total_count / new_count，无法分片，默认有新信息",
            detail={"level": 1, "reason": "missing_payload_counts"},
        )

    last_total, last_new = last_counts
    prev_total, prev_new = prev_counts

    # ── 边界保护：防止 total_count 与实际列表长度不一致 ──────────
    actual_len = len(results_list)
    if last_total > actual_len:
        logger.debug(
            "[InfoGain/L1] 事件 total_count=%d 超出实际列表长度 %d，截断",
            last_total, actual_len,
        )
        last_total = actual_len

    last_slice = results_list[last_total - last_new: last_total] if last_new > 0 else []
    prev_slice = results_list[prev_total - prev_new: prev_total] if prev_new > 0 else []

    if not last_slice or not prev_slice:
        return GainCheckResult(
            has_new_info=True,
            reason=(
                f"某轮无新增结果（last_new={last_new}, prev_new={prev_new}），"
                "无法比较内容相似度，默认有新信息"
            ),
            detail={
                "level": 1,
                "reason": "empty_slice",
                "last_new": last_new,
                "prev_new": prev_new,
            },
        )

    text_last = " ".join(r.get("content", "") for r in last_slice)
    text_prev = " ".join(r.get("content", "") for r in prev_slice)
    score = jaccard_similarity(text_last, text_prev)

    common_detail = {
        "level": 1,
        "jaccard": round(score, 4),
        "threshold": threshold,
        "last_iter": last_ev.get("iteration"),
        "prev_iter": prev_ev.get("iteration"),
        "last_slice_size": len(last_slice),
        "prev_slice_size": len(prev_slice),
    }

    if score > threshold:
        return GainCheckResult(
            has_new_info=False,
            reason=(
                f"内容高度重叠（bigram Jaccard {score:.3f} > 阈值 {threshold}，"
                f"迭代 {prev_ev.get('iteration')} → {last_ev.get('iteration')}）"
            ),
            detail=common_detail,
        )

    return GainCheckResult(
        has_new_info=True,
        reason=(
            f"内容差异充足（bigram Jaccard {score:.3f} ≤ 阈值 {threshold}，"
            f"迭代 {prev_ev.get('iteration')} → {last_ev.get('iteration')}）"
        ),
        detail=common_detail,
    )


# ─────────────────────────────────────────────
# 公共入口
# ─────────────────────────────────────────────

def detect_information_gain(
    events: Sequence[dict],
    search_results: Sequence[dict],
    *,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    min_consecutive_zero_rounds: int = DEFAULT_MIN_ZERO_ROUNDS,
) -> GainCheckResult:
    """
    信息增量检测主入口。

    算法按层级顺序执行，遇到结论即返回：

    - Level 0（精确 / 快速）：连续 N 轮 new_count=0 → 信息枯竭
    - Level 1（语义 / 兜底）：字符 bigram Jaccard > threshold → 内容重叠

    数据不足或结构异常时一律返回 has_new_info=True（保守策略）。

    Args:
        events: state["events"] 完整列表，仅读取 SearchCompleted 事件。
        search_results: state["search_results"] 完整列表。
        similarity_threshold: Jaccard 触发阈值（默认 0.85）。
        min_consecutive_zero_rounds: 连续零增量轮数阈值（默认 2，防网络抖动）。

    Returns:
        GainCheckResult — 布尔值为 False 表示信息已枯竭，应停止搜索。

    典型用法（supervisor.py）::

        gain = detect_information_gain(
            state.get("events", []),
            state.get("search_results", []),
        ).log()
        if not gain:
            # 信息枯竭，强制推进 writer 或终止
            ...

    单测示例::

        events = [
            {"type": "SearchCompleted", "iteration": 0,
             "agent": "search_agent",
             "payload": {"new_count": 0, "total_count": 3, "queries_used": ["q"]}},
            {"type": "SearchCompleted", "iteration": 1,
             "agent": "search_agent",
             "payload": {"new_count": 0, "total_count": 3, "queries_used": ["q"]}},
        ]
        result = detect_information_gain(events, [])
        assert not result                            # 连续 2 轮零增量
        assert result.detail["level"] == 0
    """
    search_events = _extract_search_events(events)

    # Level 0
    level0 = _check_consecutive_zero_counts(search_events, min_consecutive_zero_rounds)
    if level0 is not None:
        return level0

    # Level 1
    return _check_content_similarity(search_events, search_results, similarity_threshold)


# ─────────────────────────────────────────────
# TODO Day 12 — Level 2: Embedding Cosine Similarity
# ─────────────────────────────────────────────
#
# async def detect_information_gain_async(
#     events: Sequence[dict],
#     search_results: Sequence[dict],
#     *,
#     embed_fn: Callable[[list[str]], Awaitable[list[list[float]]]],
#     similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
#     min_consecutive_zero_rounds: int = DEFAULT_MIN_ZERO_ROUNDS,
# ) -> GainCheckResult:
#     """
#     Level 0 / 1 保持不变（同步快速路径）。
#     Level 2 追加：对 last_slice / prev_slice 内容做 embedding，
#     计算 cosine similarity，阈值独立配置（推荐 0.92）。
#     """
#     # Level 0 → Level 1（同步，不变）
#     sync_result = detect_information_gain(events, search_results, ...)
#     if sync_result.detail.get("level") == 0:
#         return sync_result   # Level 0 已判定，无需 embedding
#
#     # Level 2 ...