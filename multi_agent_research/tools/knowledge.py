from __future__ import annotations

import asyncio

from langchain_core.tools import tool

from .support import _err, _ok, _trim_text, with_retry
from ..core.config import settings
from ..knowledge.client import get_knowledge_service_client


@tool(
    description=(
        "【触发条件】用户询问具体公司财务数据、行业研报、政策条款或研报量化指标时，"
        "优先调用本工具。\n"
        "【不触发条件】通用概念解释或用户明确要求互联网最新新闻。\n"
        "【输入】精炼关键词，尽量包含公司名、指标和报告期。\n"
        "【输出】data.results 包含可引用的文档片段、来源和页码；"
        "has_relevant_content=False 时禁止编造。"
    )
)
@with_retry(
    tool_name="query_internal_knowledge",
    max_retries=settings.tools.knowledge.max_retries,
    timeout=settings.tools.knowledge.timeout,
)
async def query_internal_knowledge(query: str) -> dict:
    """通过隔壁 knowledge-service 的只读检索接口查询知识库。"""
    tool_name = "query_internal_knowledge"
    q = (query or "").strip()
    if not q:
        return _err(
            tool_name=tool_name,
            query=q,
            code="BAD_INPUT",
            message="检索词不能为空",
        )

    try:
        response = await get_knowledge_service_client().search(
            q,
            top_k=settings.knowledge_service.top_k,
            retrieval_mode=settings.knowledge_service.retrieval_mode,
        )
        max_chars = settings.tools.knowledge.max_content_chars
        results = []
        for chunk in response.chunks:
            page = chunk.source_page if chunk.source_page is not None else -1
            prefix = (
                f"[来源：{chunk.source_file}  第 {page} 页]"
                if page >= 1
                else f"[来源：{chunk.source_file}]"
            )
            results.append({
                "content": f"{prefix}\n{_trim_text(chunk.content, max_chars)}",
                "source": chunk.source_file,
                "page": page,
                "industry": chunk.industry or "unknown",
                "chunk_id": chunk.chunk_id,
                "score": chunk.score if chunk.score is not None else response.top_score,
            })

        return _ok(
            tool_name=tool_name,
            query=q,
            data={"results": results, "has_relevant_content": bool(results)},
            meta={
                "retrieved_count": len(results),
                "candidates_count": response.candidates_count,
                "stage": response.stage,
                "cache_hit": response.cache_hit,
            },
        )
    except (TimeoutError, asyncio.TimeoutError, ConnectionError, OSError):
        raise
    except Exception as exc:
        return _err(
            tool_name=tool_name,
            query=q,
            code="KNOWLEDGE_SERVICE_FAILED",
            message=f"知识库服务检索失败: {type(exc).__name__}: {exc}",
        )
