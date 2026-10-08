"""The real parsing/correction path, with only the HTTP provider replaced."""

import asyncio
import json

import httpx
import pytest
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from multi_agent_research.core import streaming
from multi_agent_research.core.graph import build_graph
from multi_agent_research.runs.models import RunStatus
from multi_agent_research.runs.service import RunService
from multi_agent_research.sections import workflow
from multi_agent_research.sections.artifacts import bind_claims
from multi_agent_research.sections.model_output import ModelOutputError, invoke_checked
from multi_agent_research.sections.models import ClaimExtraction, QuoteSpan, ReportReview, SectionPlan, SectionReview
from multi_agent_research.sections.quotes import locate_quote
from multi_agent_research.sections.validation import (
    BusinessValidationError, ValidationIssue, validate_draft, validate_plan,
    validate_report_review, validate_section_review,
)
from tests.test_model_output import audit, fake_provider
from tests.test_run_service import MemoryRunStore
from tests.test_section_revisions import chapter, extraction
from tests.test_sections import FakeModels


@pytest.mark.asyncio
async def test_schema_then_business_failure_share_three_attempts_and_audit():
    section = chapter()
    bad = extraction()
    bad.claims[0].evidence[0].source_number = 99
    outputs = ['{"claims":[]}', bad.model_dump_json(), extraction().model_dump_json()]
    async with fake_provider(outputs) as (model, requests), audit() as records:
        claims, cost = await invoke_checked(
            model, "JSON", "source material", ClaimExtraction,
            validator=lambda output: bind_claims(section, output), context={"section_id": "section_1"},
        )
    assert len(requests) == 3 and cost["attempts"] == 3 and cost["tokens"] == 30
    assert [r["schema_status"] for r in records] == ["failed", "passed", "passed"]
    assert [r["business_status"] for r in records] == ["not_checked", "failed", "passed"]
    assert [r["accepted"] for r in records] == [False, False, True]
    assert records[1]["errors"][0]["field"] == "claims[0].evidence[0].source_number"
    assert records[1]["errors"][0]["source_number"] == 99
    assert records[1]["context"]["section_id"] == "section_1"
    assert len({r["call_id"] for r in records}) == 1
    assert requests[2]["messages"][-2]["content"] == outputs[1]
    assert len(requests[2]["messages"]) == 4  # Original task + just the last answer/error.
    assert claims[0].evidence[0].quote_span is not None
    assert bad.claims[0].claim_id == ""  # Failed validation never changes the input.


@pytest.mark.asyncio
async def test_business_exhaustion_never_gets_a_second_retry_budget():
    bad = extraction()
    bad.claims[0].evidence[0].quote = "PRIVATE_BAD_QUOTE"
    async with fake_provider([bad.model_dump_json()]) as (model, requests), audit() as records:
        with pytest.raises(ModelOutputError, match=r"claims\[0\].evidence\[0\].quote") as caught:
            await invoke_checked(model, "JSON", "material", ClaimExtraction,
                                 validator=lambda value: bind_claims(chapter(), value),
                                 context={"section_id": "section_1"})
    assert len(records) == len(requests) == 3
    assert all(r["business_status"] == "failed" and not r["accepted"] for r in records)
    assert "PRIVATE_BAD_QUOTE" not in str(caught.value)
    assert records[-1]["diagnostic_id"] in str(caught.value)
    assert records[0]["repair_hints"][0]["original_window"] == chapter().sources[0]["content"]
    assert chapter().sources[0]["content"] in requests[1]["messages"][-1]["content"]
    assert chapter().sources[0]["content"] not in str(caught.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("retryable", [False, "internal"])
async def test_unrecoverable_validation_never_calls_model_again(retryable):
    def invalid(_):
        if retryable == "internal":
            raise KeyError("PRIVATE INTERNAL DATA")
        raise BusinessValidationError([ValidationIssue("$", "state", "State cannot be corrected")],
                                      retryable=False)
    async with fake_provider(['{"verdict":"pass"}']) as (model, requests), audit() as records:
        with pytest.raises(ModelOutputError) as caught:
            await invoke_checked(model, "JSON", "material", SectionReview, validator=invalid)
    assert len(requests) == len(records) == 1
    assert not records[0]["retryable"] and not records[0]["accepted"]
    assert "PRIVATE" not in str(caught.value)


@pytest.mark.parametrize("original,quote", [
    ("人民币 48,333亿元", "人民币48,333亿元"),
    ("零部件供\n应链增长", "零部件供应链增长"),
    ("行业的“马太效应”\n日益凸显", "行业的“马太效应”日益凸显"),
    ("该板块2025 年及26Q1\n营业收入增长", "该板块2025年及26Q1营业收入增长"),
])
def test_layout_matches_restore_original_and_offsets(original, quote):
    text = "前缀：" + original + "。后缀"
    result = locate_quote(text, quote)
    assert result is not None
    canonical, span = result
    assert canonical == original == text[span.start:span.end]
    assert span.match == "layout_whitespace"


@pytest.mark.parametrize("original,quote", [
    ("增长5.0%", "增长6.0%"), ("不支持增长", "支持增长。"),
    ("增长；但不确定", "增长。"), ("预测——尚未实现", "预测。"),
    ("not able", "notable"), ("金额1 000万元", "金额1000万元"),
    ("甲\n\n乙", "甲乙"), ("甲\t乙", "甲乙"),
    ("甲   乙", "甲乙"), ("甲 乙 / 甲\n乙", "甲乙"),
])
def test_matcher_does_not_fuzz_changes_or_ambiguous_layout(original, quote):
    assert locate_quote(original, quote) is None


def test_collect_errors_with_locations_and_do_not_partially_mutate_artifact():
    section, output = chapter(), extraction()
    output.claims[0].draft_quote = "不存在的正文"
    output.claims[0].evidence[0].quote = "不存在的引文"
    before = output.model_dump()
    with pytest.raises(BusinessValidationError) as caught:
        bind_claims(section, output)
    assert [i.field for i in caught.value.issues] == [
        "claims[0].draft_quote", "claims[0].evidence[0].quote",
    ]
    assert all(i.section_id == section.section_id for i in caught.value.issues)
    assert output.model_dump() == before
    assert not section.claims


def test_binding_restores_original_and_overwrites_untrusted_span():
    section, output = chapter(), extraction()
    section.sources[0]["content"] = "成本 的实际证据摘录"
    output.claims[0].evidence[0].quote = "成本的实际证据摘录"
    output.claims[0].evidence[0].quote_span = QuoteSpan(start=999, end=1000, match="exact")
    claims = bind_claims(section, output)
    link = claims[0].evidence[0]
    assert link.quote == section.sources[0]["content"]
    assert link.quote_span.start == 0 and link.quote_span.end == len(link.quote)
    assert link.quote_span.match == "layout_whitespace"
    assert link.evidence_id != "model-invented-id"


@pytest.mark.asyncio
async def test_punctuation_error_repaired_from_hint_not_automatically_accepted():
    section, bad = chapter(), extraction()
    section.draft = "成本的分析结论；仍然存在不确定性[来源1]。"
    bad.claims[0].draft_quote = "成本的分析结论。"
    good = bad.model_copy(deep=True)
    good.claims[0].draft_quote = "成本的分析结论；仍然存在不确定性"
    async with fake_provider([bad.model_dump_json(), good.model_dump_json()]) as (model, requests), audit() as records:
        claims, cost = await invoke_checked(model, "JSON", section.draft, ClaimExtraction,
            validator=lambda value: bind_claims(section, value))
    assert cost["attempts"] == 2 and not records[0]["accepted"]
    assert section.draft in requests[1]["messages"][-1]["content"]
    assert claims[0].draft_quote == good.claims[0].draft_quote
    assert claims[0].draft_span.match == "exact"


@pytest.mark.asyncio
async def test_plan_maximum_and_parent_ids_corrected_inside_real_wrapper(monkeypatch):
    monkeypatch.setattr(workflow.settings.agent, "section_max_count", 1)
    bad = {"sections": [
        {"title": "甲", "question": "第一个研究问题", "parent_section_ids": ["wrong"]},
        {"title": "乙", "question": "第二个研究问题"},
    ]}
    good = {"sections": [{"title": "甲", "question": "第一个研究问题"}]}
    async with fake_provider([json.dumps(bad), json.dumps(good)]) as (model, requests), audit() as records:
        monkeypatch.setattr(workflow, "load_chat_model", lambda _: model)
        result = await workflow.plan_sections({"research_question": "test question", "workflow_version": 3})
    assert len(result["sections"]) == 1 and len(requests) == 2
    assert {i["type"] for i in records[0]["errors"]} == {"chapter_limit", "unknown_parent_section"}


@pytest.mark.asyncio
async def test_report_unknown_chapter_corrected_without_deleting_issue(monkeypatch):
    bad = {"verdict": "revise", "issues": [
        {"kind": "scope", "section_ids": ["section_99"], "detail": "年份口径冲突"},
    ]}
    good = json.loads(json.dumps(bad))
    good["issues"][0]["section_ids"] = ["section_1"]
    async with fake_provider([json.dumps(bad), json.dumps(good)]) as (model, requests), audit() as records:
        monkeypatch.setattr(workflow, "load_chat_model", lambda _: model)
        result = await workflow.review_report({"research_question": "test question", "sections": [chapter().model_dump()]})
    assert len(requests) == 2
    assert result["report_review"]["verdict"] == "revise"
    assert result["report_review"]["issues"][0]["detail"] == "年份口径冲突"
    assert records[0]["errors"][0]["field"] == "issues[0].section_ids"


def test_contradictory_pass_is_downgraded_without_erasing_objections():
    review = SectionReview(verdict="pass", issues=["关键证据不足"], search_queries=["反证"])
    accepted = validate_section_review(review)
    assert review.verdict == "pass" and accepted.verdict == "revise"
    assert accepted.issues == review.issues and accepted.search_queries == review.search_queries
    report = ReportReview(verdict="pass", issues=[{"kind": "scope", "detail": "年份口径冲突"}])
    assert validate_report_review(report, set()).verdict == "revise"


def test_known_plan_and_report_are_accepted_unchanged():
    plan = SectionPlan(sections=[{"title": "甲", "question": "第一个研究问题", "parent_section_ids": ["p1"]}])
    assert validate_plan(plan, 1, {"p1"}) is plan
    report = ReportReview(verdict="pass")
    assert validate_report_review(report, {"section_1"}) is report


@pytest.mark.asyncio
async def test_writer_citations_corrected_before_draft_is_accepted():
    section = chapter()
    async with fake_provider(["缺少引用", "引用越界[来源99]", "结论[来源1]"]) as (model, requests), audit() as records:
        draft, cost = await invoke_checked(model, "写作正文", "资料", validator=lambda value: validate_draft(
            value, section.sources, section.section_id))
    assert draft == "结论[来源1]" and cost["attempts"] == len(requests) == 3
    assert [r["accepted"] for r in records] == [False, False, True]


@pytest.mark.asyncio
async def test_full_graph_schema_parser_business_audit_failure_restart_and_assembly(monkeypatch, tmp_path):
    """Real RunService, graph, parser, correction and SQLite; no paid model/KB calls."""
    fake = FakeModels()
    from multi_agent_research.sections.claim_repair import ClaimRepairs
    from tests.test_claim_repair import repair_from_prompt
    repairs = []
    broken = True
    monkeypatch.setattr(workflow, "search_agent_node", fake.search)
    monkeypatch.setattr(workflow.settings.agent, "section_max_count", 4)
    async def handler(request):
        payload = json.loads(request.content)
        system, prompt = payload["messages"][0]["content"], payload["messages"][1]["content"]
        schema = next((s for s in (ClaimRepairs, ClaimExtraction, SectionPlan, SectionReview, ReportReview)
                       if f'"title": "{s.__name__}"' in system), None)
        if schema is ClaimRepairs:
            repairs.append(prompt)
            value = repair_from_prompt(prompt, broken)
        else:
            value, _ = await fake.raw_model(system, prompt, schema)
        if broken and schema is ClaimExtraction and "本章：渠道\n" in prompt:
            value.claims.append(value.claims[0].model_copy(deep=True))
            value.claims[0].evidence[0].quote = "不存在的原文"
        output = value.model_dump_json() if schema else value
        return httpx.Response(200, json={"id": "test", "object": "chat.completion", "created": 1,
            "model": "test", "choices": [{"index": 0, "message": {"role": "assistant", "content": output},
            "finish_reason": "stop"}], "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}})
    store = MemoryRunStore()
    service = RunService(store)
    await service.create_run(question="分析公司X竞争优势", run_id="business-e2e")
    database = str(tmp_path / "business.sqlite")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = ChatOpenAI(model="test", api_key="test", http_async_client=client, max_retries=0)
        monkeypatch.setattr(workflow, "load_chat_model", lambda _: model)
        async with AsyncSqliteSaver.from_conn_string(database) as saver:
            app = build_graph(saver)
            async def get_app():
                return app
            monkeypatch.setattr(streaming, "_get_app", get_app)
            await service.start_run("business-e2e")
            await asyncio.wait_for(service._tasks["business-e2e"], 30)
            failed = await service.get_run("business-e2e")
            assert failed.status == RunStatus.PAUSED
            assert failed.sections[0].status == "complete"
            assert failed.sections[1].draft and len(failed.sections[1].claims) == 1
            accepted_before = failed.sections[1].claims[0].model_dump()
            assert accepted_before['claim_id'].endswith(':c2')
            assert not failed.final_report
            rejected = [r for r in store.diagnostics if r["business_status"] == "partial"]
            assert len(rejected) == 3 and all(not r["accepted"] for r in rejected)
            assert failed.sections[1].claim_work['attempts'] == 3
            snapshot = await app.aget_state(streaming.checkpoint_config("business-e2e"))
            assert snapshot.next == ("section_claim_gate",)
        broken = False
        async with AsyncSqliteSaver.from_conn_string(database) as saver:
            app = build_graph(saver)
            await service.start_run("business-e2e", resume=True)
            await asyncio.wait_for(service._tasks["business-e2e"], 30)
            result = await service.get_run("business-e2e")
    assert result.status == RunStatus.COMPLETED and "参考来源" in result.final_report
    assert result.report_review.verdict == "pass"
    assert all(section.claims for section in result.sections)
    assert result.sections[1].claims[1].model_dump() == accepted_before
    assert result.budget['known_tokens'] > failed.budget['known_tokens'] > 0
    assert result.sections[1].claim_work['total_attempts'] == 4
    assert fake.calls["write:成本"] == fake.calls["write:渠道"] == 1
    assert fake.calls["claims:渠道"] == 1 and len(repairs) == 3
    assert result.model_usage["attempts"] == len(store.diagnostics)
    assert result.model_usage["tokens"] == 10 * len(store.diagnostics)
    assert store.events[-1].event_type == "done"
    await service.shutdown()
