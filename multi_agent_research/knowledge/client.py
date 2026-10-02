"""knowledge-service 的窄接口客户端。

本模块只实现就绪检查和检索，不暴露建库、缓存失效或管理写接口。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field

from ..core.config import settings


class KnowledgeServiceError(RuntimeError):
    """knowledge-service 返回了不可重试的错误。"""


class KnowledgeServiceUnavailable(ConnectionError):
    """knowledge-service 暂时不可用，可由上层重试。"""


class RetrievedChunk(BaseModel):
    model_config = ConfigDict(extra="ignore")

    content: str
    source_file: str = "unknown"
    source_page: int | None = None
    chunk_id: str = ""
    score: float | None = None
    doc_type: str | None = None
    industry: str | None = None


class KnowledgeSearchResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    query: str
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    stage: str = "unknown"
    cache_hit: bool = False
    candidates_count: int = 0
    reranked_count: int = 0
    top_score: float | None = None
    timings: dict[str, Any] = Field(default_factory=dict)


class KnowledgeServiceClient:
    """仅允许健康检查与查询的异步客户端。"""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        headers = {"X-Knowledge-Service-Key": api_key} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=timeout,
            transport=transport,
        )

    async def ensure_ready(self) -> dict[str, Any]:
        response = await self._request("GET", "/api/v1/health/ready")
        payload = response.json()
        if not payload.get("ready"):
            raise KnowledgeServiceUnavailable(
                f"knowledge-service 尚未就绪（state={payload.get('state', 'unknown')}）"
            )
        return payload

    async def search(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, Any] | None = None,
        retrieval_mode: Literal["hybrid", "bm25", "vector"] = "hybrid",
    ) -> KnowledgeSearchResponse:
        response = await self._request(
            "POST",
            "/api/v1/retrieval/search",
            json={
                "query": query,
                "top_k": top_k,
                "filters": filters,
                # 强制关闭服务侧查询缓存，避免本项目触发任何缓存写入。
                "use_query_cache": False,
                "retrieval_mode": retrieval_mode,
            },
        )
        return KnowledgeSearchResponse.model_validate(response.json())

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = await self._client.request(method, path, **kwargs)
        except httpx.RequestError as exc:
            raise KnowledgeServiceUnavailable(
                f"无法连接 knowledge-service：{type(exc).__name__}"
            ) from exc

        if response.status_code >= 500:
            raise KnowledgeServiceUnavailable(
                f"knowledge-service 服务端错误（HTTP {response.status_code}）"
            )
        if response.status_code >= 400:
            raise KnowledgeServiceError(
                f"knowledge-service 请求失败（HTTP {response.status_code}）"
            )
        return response

    async def aclose(self) -> None:
        await self._client.aclose()


@lru_cache(maxsize=1)
def get_knowledge_service_client() -> KnowledgeServiceClient:
    config = settings.knowledge_service
    return KnowledgeServiceClient(
        base_url=config.base_url,
        api_key=config.api_key.get_secret_value(),
        timeout=config.timeout,
    )
