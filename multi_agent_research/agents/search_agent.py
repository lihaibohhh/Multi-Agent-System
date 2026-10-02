"""
search_agent.py — 检索 Agent
职责：knowledge-service 检索 + 联网搜索（Tavily），将结果写入 state["search_results"]。
"""

from __future__ import annotations
import logging
import asyncio
from langchain_core.messages import AIMessage

from ..core.config import settings
from ..core.state import AnalystVerdict, ResearchState, SearchResult
from ..knowledge.client import get_knowledge_service_client


logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# § 1  配置常量
# ─────────────────────────────────────────────
tools_con = settings.tools
_DEDUP_KEY_LEN: int = 200       # 去重指纹长度（原 50，研报常见相同开头）
_WEB_MAX_RESULTS: int = 5       # Tavily 单次最多返回条数
_WEB_MAX_RETRIES: int = 2       # Tavily 临时错误（超时/连接）最大重试次数
_WEB_RETRY_DELAY: float = 1.0   # 重试间隔（秒）


# ─────────────────────────────────────────────────────────────────────────────
# § 2  knowledge-service 检索
# ─────────────────────────────────────────────────────────────────────────────
async def _knowledge_search(query: str, iteration: int) -> list[SearchResult]:
    """通过隔壁 knowledge-service 执行只读检索。"""
    q = query.strip()
    if not q:
        return []

    try:
        response = await get_knowledge_service_client().search(
            q,
            top_k=settings.knowledge_service.top_k,
            retrieval_mode=settings.knowledge_service.retrieval_mode,
        )
    except Exception as exc:
        logger.error("[SearchAgent] knowledge-service 检索失败（%s）：%s", q, exc)
        return []

    logger.info(
        "[SearchAgent] knowledge-service 完成：query=%s | stage=%s | cache=%s | 返回 %d 条",
        q, response.stage, response.cache_hit, len(response.chunks),
    )

    max_chars = tools_con.knowledge.max_content_chars
    results: list[SearchResult] = []
    for chunk in response.chunks:
        page = chunk.source_page if chunk.source_page is not None else -1
        prefix = (
            f"[来源：{chunk.source_file}   第 {page} 页]"
            if page >= 1
            else f"[来源：{chunk.source_file}]"
        )
        score = chunk.score if chunk.score is not None else response.top_score
        results.append(SearchResult(
            query=q,
            source="knowledge",
            content=f"{prefix}\n{chunk.content[:max_chars]}",
            score=float(score or 0.0),
            metadata={
                "source": chunk.source_file,
                "page": page,
                "chunk_id": chunk.chunk_id,
                "industry": chunk.industry or "unknown",
                "doc_type": chunk.doc_type,
                "retrieval_stage": response.stage,
                "cache_hit": response.cache_hit,
            },
            iteration=iteration,
        ))
    return results


# ─────────────────────────────────────────────
# § 4  Web 检索（Tavily）
# ─────────────────────────────────────────────
async def _web_search(query: str, iteration: int) -> list[SearchResult]:
    """
    Tavily 联网搜索。

    依赖：pip install langchain-tavily
    配置：TAVILY_API_KEY 环境变量（必须）
    """
    q = query.strip()
    if not q:
        return []

    api_key = settings.tool_secrets.tavily_api_key.get_secret_value()
    if not api_key:
        logger.warning("[SearchAgent] TAVILY_API_KEY 未配置，跳过 Web 检索：%s", q)
        return []

    try:
        from langchain_tavily import TavilySearch
    except ImportError:
        logger.warning(
            "[SearchAgent] langchain-tavily 未安装，跳过 Web 检索。"
            "请执行：pip install langchain-tavily"
        )
        return []

    # ── 带重试的 API 调用 ───────────────────────
    tool = TavilySearch(max_results=_WEB_MAX_RESULTS)
    raw = None

    for attempt in range(1 + _WEB_MAX_RETRIES):
        try:
            # TavilySearch 支持 dict 和 str 两种调用签名，优先用 dict
            try:
                raw = await tool.ainvoke({"query": q})
            except Exception:
                raw = await tool.ainvoke(q)
            break  # 调用成功，退出重试循环

        except (TimeoutError, asyncio.TimeoutError, ConnectionError, OSError) as e:
            # 临时错误：等待后重试
            if attempt < _WEB_MAX_RETRIES:
                logger.warning(
                    "[SearchAgent] Web 检索超时/连接失败，%.1fs 后重试（%d/%d）：%s",
                    _WEB_RETRY_DELAY, attempt + 1, _WEB_MAX_RETRIES, e,
                )
                await asyncio.sleep(_WEB_RETRY_DELAY)
            else:
                logger.error("[SearchAgent] Web 检索重试耗尽（%s）：%s", q, e)
                return []

        except Exception as e:
            # 永久错误（4xx、解析失败等）：直接放弃
            logger.error("[SearchAgent] Web 检索失败（%s）：%s", q, e)
            return []

    if raw is None:
        return []

    # ── 解析 Tavily 响应 ─────────────────────────────────────
    # TavilySearch 不同版本返回格式不同：
    #   新版：list[dict]  每个 dict 含 url / content / title / score
    #   旧版：{"results": [...], "answer": str}
    if isinstance(raw, list):
        items: list[dict] = raw
    elif isinstance(raw, dict):
        items = raw.get("results", [])
    else:
        logger.warning(
            "[SearchAgent] Tavily 返回格式未知（%s），跳过解析：%s",
            type(raw).__name__, q,
        )
        return []

    max_chars: int = tools_con.knowledge.max_content_chars

    results: list[SearchResult] = []
    for item in items[:_WEB_MAX_RESULTS]:
        url = item.get("url", "")
        title = item.get("title", "")
        content_raw = item.get("content", "")[:max_chars]
        published_date = item.get("published_date")

        prefix_parts = [f"来源：{url}"]
        if title:
            prefix_parts.append(f"标题：{title}")
        if published_date:
            prefix_parts.append(f"日期：{published_date}")
        prefix = "[" + " | ".join(prefix_parts) + "]"
        raw_score = float(item.get("score") or 0.5)
        results.append(SearchResult(
            query=q,
            source="web",
            content=f"{prefix}\n{content_raw}",
            score=max(0.0, min(raw_score, 1.0)),
            metadata={
                "url": url,
                "title": title,
                "published_date": published_date,
                "industry": "unknown",
            },
            iteration=iteration,
        ))

    logger.info("[SearchAgent] Web 检索完成：query=%s | 返回 %d 条", q, len(results))
    return results


# ─────────────────────────────────────────────
# § 5  节点函数（供 graph.py 注册）
# ─────────────────────────────────────────────
async def search_agent_node(state: ResearchState) -> dict:
    """
    Search Agent 节点。
    读取：state["research_question"]、最新 messages 中的 Supervisor 指令
    写入：state["search_results"]（追加，不覆盖）
    """
    question = state["research_question"]
    iteration = state.get("iteration_count", 0)
    parent_context = state.get("parent_context")

    # 提取检索 specific_gaps（来自 AnalystVerdict）
    av: AnalystVerdict = state.get("analyst_verdict") or AnalystVerdict.empty()
    gaps = av.specific_gaps
    if parent_context:
        parent_question = str(parent_context.get("source_question", "")).strip()
        primary_query = (
            f"{question} 承接研究背景：{parent_question}"
            if parent_question
            else question
        )[:500]
    else:
        primary_query = question
    queries = [primary_query] + gaps[:2]  # 子问题 + 父任务主题 + 最多 2 个缺口

    # ── 并行触发所有 query 的知识服务 + Web 检索 ───
    knowledge_tasks = [_knowledge_search(q, iteration) for q in queries]
    web_tasks = [_web_search(q, iteration) for q in queries]

    all_batches = await asyncio.gather(
        *knowledge_tasks, *web_tasks,
        return_exceptions=True,  # 单批次异常以对象形式返回，不中断其他
    )

    # ── 轮内去重 ───────────────────────────────
    # 目的：排除同一轮不同 query 召回的重复文档（跨 query 同文档）
    # 历史去重（跨轮次、跨迭代）由 state._dedup_append_results Reducer 统一处理
    # 关键：不读取 state["search_results"]，不持有全量历史 seen set
    seen_this_round: set[str] = set()
    new_results: list[SearchResult] = []


    for batch in all_batches:
        if isinstance(batch, Exception):
            # asyncio.gather 捕获的异常，正常情况不应到这里（各函数内已 try/except）
            logger.error("[SearchAgent] 检索批次异常（已跳过）：%s", batch)
            continue
        for r in sorted(batch, key=lambda x: x["score"], reverse=True):
            key = r["content"][:_DEDUP_KEY_LEN]
            if key not in seen_this_round:
                seen_this_round.add(key)
                new_results.append(r)

    # 追加到已有结果（不覆盖，Analyst 需要看历史变化）
    existing_count = len(state.get("search_results", []))
    # import json
    # print(json.dumps([dict(r) for r in new_results], ensure_ascii=False, default=str))
    logger.info(
        "[SearchAgent] 本轮新增 %d 条（轮内去重后，Reducer 历史去重前）| 近似累计 %d 条 | query：%s",
        len(new_results), existing_count + len(new_results), queries,
    )

    return {
        # 仅返回本轮新增结果，state._dedup_append_results Reducer 负责追加去重
        "search_results": new_results,
        "events": [{
            "type": "SearchCompleted",
            "iteration": iteration,
            "agent": "search_agent",
            "payload": {
                "new_count": len(new_results),  # 准确值（轮内去重后）
                "total_count": existing_count + len(new_results),
                "queries_used": queries
            }
        }],
        "messages": [AIMessage(
            content=f"[SearchAgent] 完成检索，新增 {len(new_results)} 条（知识服务 + Web）"
        )]
    }
