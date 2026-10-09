"""Read-only knowledge-service and web retrieval; this is not an Agent."""

from __future__ import annotations
import logging
import asyncio
from httpx import TimeoutException

from ..core.config import settings
from ..knowledge.client import get_knowledge_service_client
from ..core.budget import CallTimeout, RunControlError, invoke_retrieval
from ..core.retrieval import durable_retrieval, gather_retrievals, retrieval_admitted
from .models import RetrievalRequest, SearchResult
from .normalization import (
    deduplicate_batches,
    normalize_knowledge_response,
    normalize_web_items,
)


logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# § 1  配置常量
# ─────────────────────────────────────────────
tools_con = settings.tools
_WEB_MAX_RESULTS: int = tools_con.search.max_results
_WEB_MAX_RETRIES: int = tools_con.search.max_retries
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
        response = await invoke_retrieval(lambda: get_knowledge_service_client().search(
            q,
            top_k=settings.knowledge_service.top_k,
            retrieval_mode=settings.knowledge_service.retrieval_mode,
        ), label="knowledge")
    except RunControlError:
        raise
    except Exception as exc:
        if retrieval_admitted.get():
            raise
        if isinstance(exc.__cause__, TimeoutException):
            raise CallTimeout("知识服务请求超时；本次检索额度保留，停止当前执行") from exc
        logger.error("[Retrieval] knowledge-service 检索失败（%s）：%s", q, exc)
        return []

    logger.info(
        "[Retrieval] knowledge-service 完成：query=%s | stage=%s | cache=%s | 返回 %d 条",
        q, response.stage, response.cache_hit, len(response.chunks),
    )

    return normalize_knowledge_response(
        response,
        query=q,
        iteration=iteration,
        max_chars=tools_con.knowledge.max_content_chars,
    )


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
        logger.warning("[Retrieval] TAVILY_API_KEY 未配置，跳过 Web 检索：%s", q)
        return []

    try:
        from langchain_tavily import TavilySearch
    except ImportError:
        if retrieval_admitted.get():
            raise
        logger.warning(
            "[Retrieval] langchain-tavily 未安装，跳过 Web 检索。"
            "请执行：pip install langchain-tavily"
        )
        return []

    # ── 带重试的 API 调用 ───────────────────────
    tool = TavilySearch(max_results=_WEB_MAX_RESULTS)
    raw = None

    for attempt in range(1 + _WEB_MAX_RETRIES):
        try:
            raw = await invoke_retrieval(lambda: tool.ainvoke({"query": q}), label="web")
            # Installed Tavily tool returns transport exceptions as data.
            if isinstance(raw, dict) and isinstance(raw.get("error"), Exception):
                raise raw["error"]
            break  # 调用成功，退出重试循环

        except RunControlError:
            raise
        except (TimeoutError, asyncio.TimeoutError, ConnectionError, OSError) as e:
            if retrieval_admitted.get():
                raise
            # 临时错误：等待后重试
            if attempt < _WEB_MAX_RETRIES:
                logger.warning(
                    "[Retrieval] Web 检索超时/连接失败，%.1fs 后重试（%d/%d）：%s",
                    _WEB_RETRY_DELAY, attempt + 1, _WEB_MAX_RETRIES, e,
                )
                await asyncio.sleep(_WEB_RETRY_DELAY)
            else:
                logger.error("[Retrieval] Web 检索重试耗尽（%s）：%s", q, e)
                return []

        except Exception as e:
            if retrieval_admitted.get():
                raise
            # 永久错误（4xx、解析失败等）：直接放弃
            logger.error("[Retrieval] Web 检索失败（%s）：%s", q, e)
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
        if retrieval_admitted.get() and ("error" in raw or not isinstance(raw.get("results"), list)):
            raise ValueError("Tavily returned an error or invalid results")
        items = raw.get("results", [])
    else:
        if retrieval_admitted.get():
            raise ValueError("invalid Tavily response")
        logger.warning(
            "[Retrieval] Tavily 返回格式未知（%s），跳过解析：%s",
            type(raw).__name__, q,
        )
        return []

    results = normalize_web_items(
        items,
        query=q,
        iteration=iteration,
        max_chars=tools_con.knowledge.max_content_chars,
        maximum=_WEB_MAX_RESULTS,
    )

    logger.info("[Retrieval] Web 检索完成：query=%s | 返回 %d 条", q, len(results))
    return results


async def retrieve_evidence(request: RetrievalRequest) -> list[SearchResult]:
    """Execute one durable retrieval round without reading or mutating graph state."""
    primary_query = (
        f"{request.question} 承接研究背景：{request.parent_question}"
        if request.parent_question
        else request.question
    )[:500]
    queries = [primary_query, *request.gaps[:2]]

    # ── 并行触发所有 query 的知识服务 + Web 检索 ───
    scope = request.scope or {
        "section_id": None,
        "round": request.iteration,
        "revision": 0,
    }

    def retrieve(provider, query):
        descriptor = {**scope, "format_version": 1, "provider": provider, "query": query.strip(),
                      "max_content_chars": tools_con.knowledge.max_content_chars}
        if provider == "knowledge":
            descriptor.update(endpoint=settings.knowledge_service.base_url,
                              top_k=settings.knowledge_service.top_k,
                              retrieval_mode=settings.knowledge_service.retrieval_mode,
                              use_query_cache=False)
            async def operation():
                return await _knowledge_search(query, request.iteration)
        else:
            descriptor.update(max_results=_WEB_MAX_RESULTS)
            async def operation():
                return await _web_search(query, request.iteration)
        return durable_retrieval(operation, descriptor)

    knowledge_tasks = [retrieve("knowledge", q) for q in queries if q.strip()]
    # Disabled providers are not successful empty queries and consume no quota.
    web_tasks = ([retrieve("web", q) for q in queries if q.strip()]
                 if settings.tool_secrets.tavily_api_key.get_secret_value() else [])
    all_batches = await gather_retrievals(*knowledge_tasks, *web_tasks)

    new_results = deduplicate_batches(all_batches)
    logger.info(
        "[Retrieval] 本轮新增 %d 条（轮内去重后）| 近似累计 %d 条 | query：%s",
        len(new_results), request.existing_count + len(new_results), queries,
    )
    return new_results
