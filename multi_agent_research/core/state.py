"""
state.py — 多 Agent 系统的全局状态定义
所有 Agent 共享同一个 ResearchState，私有工作区用独立字段隔离。
"""

from __future__ import annotations
from pydantic import BaseModel, Field, field_validator
from typing import Annotated, Literal, TypedDict
from langgraph.graph.message import add_messages


_DEDUP_KEY_LEN: int = 200  # 去重指纹截取长度（与 search_agent 原逻辑保持一致）


# ─────────────────────────────────────────────────────────────────────────────
# § 1  Reducer 函数
#
# 放在模块顶部，让所有 Annotated 声明可直接引用，也方便单独测试。
# 每个 Reducer 承担一种且只一种合并语义。
# ─────────────────────────────────────────────────────────────────────────────
def _append_events(current: list, incoming: list | None) -> list:
    if not incoming:
        return current
    return current + incoming


def _dedup_append_results(
    current: list[SearchResult],
    incoming: list[SearchResult] | None,
) -> list[SearchResult]:
    """
    SearchResult 去重追加 Reducer。

    【核心设计决策】去重责任从 Agent 下移到状态层：
    新版（Agent 只返新增）：
        return {"search_results": new_results_this_round}   ← 只返本轮产物
        # Reducer 在状态层完成去重追加，Agent 逻辑简化为纯粹的"检索+结果生产"
    """
    if not incoming:
        return current
    seen: set[str] = {r["content"][:_DEDUP_KEY_LEN] for r in current}
    deduplicated = [
        r for r in incoming
        if r["content"][:_DEDUP_KEY_LEN] not in seen
    ]
    return current + deduplicated


def format_parent_context(state: ResearchState, *, max_chars: int = 4_000) -> str:
    """Render the immutable parent snapshot as delimited reference material."""
    context = state.get("parent_context")
    if not context:
        return "（无父任务上下文）"
    excerpt = str(context.get("report_excerpt", ""))[:max_chars]
    references = str(context.get("reference_excerpt", ""))[:2_000]
    return (
        "<parent_run_context>\n"
        f"source_run_id: {context.get('source_run_id', '')}\n"
        f"source_question: {context.get('source_question', '')}\n"
        f"report_excerpt:\n{excerpt}\n"
        f"reference_excerpt:\n{references}\n"
        "</parent_run_context>"
    )


# ─────────────────────────────────────────────────────────────────────────────
# § 2  事件结构（仅追加，审计 / SSE 透传用）
# ─────────────────────────────────────────────────────────────────────────────
class AgentEvent(TypedDict):
    """
    所有 Agent 写入 events 时使用此结构。
    type 值域：
        "SearchCompleted"   — Search Agent 写，payload 含 new_count / total_count / queries_used
        "AnalystCompleted"  — Analyst Agent 写，payload 含 analyst_verdict.model_dump()
        "WriterCompleted"   — Writer Agent 写，payload 含 report_length
        "SupervisorDecided" — Supervisor 写，payload 含 next / reason / iteration
    """
    type: str          # "SearchCompleted" | "AnalystCompleted" | "WriterCompleted"
    iteration: int     # 第几轮
    agent: str         # 哪个 Agent 写的
    payload: dict      # 事件携带的数据


# ─────────────────────────────────────────────────────────────────────────────
# § 3  子结构体（结构化传递，禁止以自然语言字符串替代）
# ─────────────────────────────────────────────────────────────────────────────
class SearchResult(TypedDict):
    """Search Agent 每次检索的结构化结果"""
    query: str                  # 实际执行的检索 query
    source: Literal["knowledge", "web"]  # 来源
    content: str                # 检索到的文本内容
    score: float                # 相关性分数 0-1
    metadata: dict              # 原始文档元信息（title, url, chunk_id 等）
    iteration: int              # ← 新增：记录该结果属于第几轮迭代


class AnalystVerdict(BaseModel):
    """
    Analyst Agent 的结构化审查结果。
    【重要】Supervisor 必须读这个字段做路由决策，绝对不能解析自然语言 messages。
    """
    verdict: Literal["not_reviewed", "pass", "revise", "reject"] = "not_reviewed"
    reason: str = ""                                                      # 人读的解释
    specific_gaps: list[str] = Field(default_factory=list)                # 具体缺口 → Supervisor 据此指导下一次检索
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)           # 0.0-1.0，pass 阈值建议 0.75

    @classmethod
    def empty(cls) -> "AnalystVerdict":
        """空对象工厂：代表 'Analyst 尚未运行'，所有属性安全可访问。"""
        return cls()

    @property
    def is_reviewed(self) -> bool:
        """True = Analyst 真实运行过；False = 初始占位状态。"""
        return self.verdict != "not_reviewed"

    @field_validator("confidence_score", mode="before")
    @classmethod
    def coerce_confidence(cls, v):
        return float(v) if v is not None else 0.0

    model_config = {"extra": "ignore"}


class SupervisorDecision(BaseModel):
    """
    Supervisor 每轮的结构化决策输出。
    使用 with_structured_output 强制 LLM 输出此格式，不做字符串解析。
    """
    next: Literal["search_agent", "analyst_agent", "writer_agent", "FINISH"]
    reason: str = ""  # 决策原因（仅 debug / 日志，不传给任何下游 Agent）
    task_plan: list[str] = Field(default_factory=list)  # 仅 iteration == 0 时填写，后续返回 []
    token_cost: int = 0  # 本次 LLM 调用消耗 token 数（由节点代码填入）


# ─────────────────────────────────────────────────────────────────────────────
# § 4  主状态 Schema
# ─────────────────────────────────────────────────────────────────────────────
class ResearchState(TypedDict):
    # ── 全局只读（写入后不再修改）────────────────
    research_question: str          # 用户原始问题
    parent_context: dict | None     # 父 Run 产物快照，不包含父 Run 内部 State

    workflow_version: int           # 3: evidence/revisions; 2: chapters; missing: legacy
    sections: list[dict]            # persisted artifacts, replaced by the serial coordinator
    active_section: int
    section_policy: dict
    section_step: str
    report_quality: str
    report_review: dict | None
    model_calls: int
    usage_unknown_calls: int

    # ── Supervisor 控制字段 ──────────────────────
    task_plan: list[str]            # Supervisor 分解的子任务列表
    next_agent: str                 # Supervisor 决定的下一跳
    supervisor_reason: str  # 当轮决策原因（调试 / SSE 透传用）
    task_status: dict[str, Literal["pending", "pass", "revise", "reject"]]
    iteration_count: int            # 循环计数（终止条件）
    token_budget_used: int          # 累计 token 消耗（终止条件）

    # ── 消息总线（LangGraph 标准，自动追加）────────
    messages: Annotated[list, add_messages]

    # ── Agent 私有工作区（各 Agent 只写自己的字段）─
    search_results: Annotated[list[SearchResult], _dedup_append_results]      # Search Agent 写入
    analyst_verdict: AnalystVerdict         # 非可选，始终有对象
    writer_status: Literal["not_started", "complete"]

    events: Annotated[list[AgentEvent], _append_events]  # 只追加，永不修改

    # ── 最终输出 ─────────────────────────────────
    final_report: str


# ─────────────────────────────────────────────
# 初始状态工厂（graph.invoke 的入口）
# ─────────────────────────────────────────────

def initial_state(question: str, parent_context: dict | None = None) -> ResearchState:
    return ResearchState(
        research_question=question,
        parent_context=parent_context,
        workflow_version=3,
        sections=[],
        active_section=0,
        section_policy={},
        section_step="plan",
        report_quality="pending",
        report_review=None,
        model_calls=0,
        usage_unknown_calls=0,
        task_plan=[],
        next_agent="supervisor",
        supervisor_reason="",
        task_status={},
        iteration_count=0,
        token_budget_used=0,
        messages=[],
        search_results=[],
        events=[],  # type: ignore[typeddict-item]
        analyst_verdict=AnalystVerdict.empty(),
        writer_status="not_started",
        final_report="",
    )
