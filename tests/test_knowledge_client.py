from __future__ import annotations

import json

import httpx
import pytest

from multi_agent_research.knowledge.client import (
    KnowledgeServiceClient,
    KnowledgeServiceError,
    KnowledgeServiceUnavailable,
)


@pytest.mark.asyncio
async def test_ready_and_search_use_only_read_contract() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/api/v1/health/ready":
            return httpx.Response(200, json={"ready": True, "chunk_count": 43142})
        assert request.url.path == "/api/v1/retrieval/search"
        payload = json.loads(request.content)
        assert payload == {
            "query": "人工智能 教育",
            "top_k": 3,
            "filters": None,
            "use_query_cache": False,
            "retrieval_mode": "hybrid",
        }
        return httpx.Response(200, json={
            "query": payload["query"],
            "chunks": [{
                "content": "测试片段",
                "source_file": "教育/报告.pdf",
                "source_page": 12,
                "chunk_id": "chunk-1",
                "score": None,
                "industry": "教育",
            }],
            "stage": "dual_retrieve+rerank",
            "cache_hit": False,
            "candidates_count": 17,
            "reranked_count": 12,
            "top_score": 0.98,
        })

    client = KnowledgeServiceClient(
        base_url="http://knowledge.test",
        api_key="test-secret",
        timeout=5,
        transport=httpx.MockTransport(handler),
    )
    try:
        ready = await client.ensure_ready()
        result = await client.search("人工智能 教育", top_k=3)
    finally:
        await client.aclose()

    assert ready["chunk_count"] == 43142
    assert result.chunks[0].source_page == 12
    assert result.top_score == pytest.approx(0.98)
    assert all(r.headers["X-Knowledge-Service-Key"] == "test-secret" for r in requests)
    assert [r.url.path for r in requests] == [
        "/api/v1/health/ready",
        "/api/v1/retrieval/search",
    ]


@pytest.mark.asyncio
async def test_client_classifies_http_errors() -> None:
    async def assert_error(status_code: int, expected: type[Exception]) -> None:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(status_code, request=request)
        )
        client = KnowledgeServiceClient(
            base_url="http://knowledge.test",
            api_key="test-secret",
            timeout=5,
            transport=transport,
        )
        try:
            with pytest.raises(expected):
                await client.search("test", top_k=3)
        finally:
            await client.aclose()

    await assert_error(401, KnowledgeServiceError)
    await assert_error(503, KnowledgeServiceUnavailable)
    await assert_error(408, KnowledgeServiceUnavailable)
    await assert_error(429, KnowledgeServiceUnavailable)


@pytest.mark.asyncio
async def test_missing_chunks_is_invalid_not_a_cached_empty_success():
    from pydantic import ValidationError
    client = KnowledgeServiceClient(base_url="http://knowledge.test", api_key="", timeout=5,
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"query": "test"})))
    try:
        with pytest.raises(ValidationError, match="chunks"):
            await client.search("test", top_k=3)
    finally:
        await client.aclose()
