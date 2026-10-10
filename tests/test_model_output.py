import json
from contextlib import asynccontextmanager

import httpx
import pytest
from langchain_openai import ChatOpenAI

from multi_agent_research.sections.model_output import ModelOutputError, attempt_sink, invoke_checked
from multi_agent_research.sections.models import ClaimExtraction, ReportReview, SectionPlan, SectionReview


@asynccontextmanager
async def fake_provider(outputs, *, usage=True, finish="stop"):
    requests = []
    async def handler(request):
        requests.append(json.loads(request.content))
        output = outputs[min(len(requests) - 1, len(outputs) - 1)]
        body = {"id": "test", "object": "chat.completion", "created": 1, "model": "test",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": output},
                             "finish_reason": finish}]}
        if usage:
            body["usage"] = {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}
        return httpx.Response(200, json=body)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = ChatOpenAI(model="test", api_key="test-only", http_async_client=client, max_retries=0)
        yield model, requests


@asynccontextmanager
async def audit():
    records = []
    async def sink(record):
        records.append(record)
    token = attempt_sink.set(sink)
    try:
        yield records
    finally:
        attempt_sink.reset(token)


BAD = json.dumps({"verdict": "revise", "issues": ["缺少反证"], "search_queries": ["A", "B", "C"]})
GOOD = json.dumps({"verdict": "revise", "issues": ["缺少反证"], "search_queries": ["A", "B"]})


@pytest.mark.asyncio
async def test_real_langchain_parser_corrects_three_queries_and_counts_both_attempts():
    async with fake_provider([BAD, GOOD]) as (model, requests), audit() as records:
        result, cost = await invoke_checked(model, "审校返回 JSON", "研究资料", SectionReview)
    assert result.verdict == "revise" and result.issues == ["缺少反证"]
    assert result.search_queries == ["A", "B"]
    assert cost == {"tokens": 20, "unknown": 0, "attempts": 2}
    assert '"maxItems": 2' in requests[0]["messages"][0]["content"]
    assert requests[0]["response_format"] == {"type": "json_object"}
    assert "search_queries" in requests[1]["messages"][-1]["content"]
    assert "at most 2" in requests[1]["messages"][-1]["content"]
    assert records[0]["errors"][0]["field"] == "search_queries"
    assert records[0]["raw"] == BAD
    assert not records[1]["errors"]


@pytest.mark.asyncio
async def test_exhausted_corrections_fail_safely_with_private_diagnostics():
    async with fake_provider([BAD]) as (model, requests), audit() as records:
        with pytest.raises(ModelOutputError, match="search_queries") as caught:
            await invoke_checked(model, "审校 JSON", "PRIVATE RESEARCH CONTENT", SectionReview)
    assert len(requests) == len(records) == 3
    assert sum(r["tokens"] for r in records) == 30
    assert "PRIVATE" not in str(caught.value)
    assert BAD not in str(caught.value)
    assert records[-1]["diagnostic_id"] in str(caught.value)
    assert len({r["diagnostic_id"] for r in records}) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["not json", '{"verdict":"approve"}', '{"verdict":"pass","issues":null}'])
async def test_invalid_json_enum_and_type_use_real_parser(bad):
    async with fake_provider([bad, GOOD]) as (model, _), audit() as records:
        result, _ = await invoke_checked(model, "JSON", "context", SectionReview)
    assert result.verdict == "revise"
    assert records[0]["errors"]


@pytest.mark.asyncio
async def test_truncation_does_not_waste_format_retries():
    async with fake_provider([GOOD], finish="length") as (model, requests), audit() as records:
        with pytest.raises(ModelOutputError, match="长度上限") as caught:
            await invoke_checked(model, "JSON", "context", SectionReview)
    assert len(requests) == 1 and records[0]["errors"][0]["type"] == "truncated"
    assert caught.value.cost == {"tokens": 10, "unknown": 0, "attempts": 1}


@pytest.mark.asyncio
async def test_unknown_usage_and_sensitive_raw_response_are_handled(monkeypatch):
    from pydantic import SecretStr
    from multi_agent_research.core.config import settings
    secret = "sk-unit-test-secret-0123456789"
    monkeypatch.setattr(settings.deepseek, "api_key", SecretStr(secret))
    output = json.dumps({"verdict": "pass", "summary": secret})
    async with fake_provider([output], usage=False) as (model, _), audit() as records:
        _, cost = await invoke_checked(model, "JSON", "context", SectionReview)
    assert cost["unknown"] == 1
    assert secret not in records[0]["raw"]
    assert "REDACTED" in records[0]["raw"]


@pytest.mark.asyncio
@pytest.mark.parametrize("schema,output", [
    (SectionPlan, {"sections": [{"title": "甲", "question": "第一章研究问题"}]}),
    (ClaimExtraction, {"claims": [{"statement": "结论", "draft_quote": "原文", "assessment": "unsupported"}]}),
    (ReportReview, {"verdict": "pass", "issues": []}),
])
async def test_all_structured_schemas_share_format_instructions(schema, output):
    async with fake_provider([json.dumps(output)]) as (model, requests):
        parsed, _ = await invoke_checked(model, "返回 JSON", "context", schema)
    assert isinstance(parsed, schema)
    assert schema.__name__ in requests[0]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_transport_error_is_recorded_but_not_treated_as_format_error():
    class Broken:
        def with_structured_output(self, *args, **kwargs):
            return self
        async def ainvoke(self, messages):
            raise RuntimeError("must not leak credential or prompt")
    async with audit() as records:
        with pytest.raises(ModelOutputError) as caught:
            await invoke_checked(Broken(), "JSON", "context", SectionReview)
    assert len(records) == 1 and records[0]["unknown"] == 1
    assert caught.value.cost == {"tokens": 0, "unknown": 1, "attempts": 1}
    assert "must not leak" not in str(caught.value)
