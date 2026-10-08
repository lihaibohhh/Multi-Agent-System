import asyncio
from copy import deepcopy

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from multi_agent_research.api import server
from multi_agent_research.core import streaming
from multi_agent_research.core.graph import build_graph
from multi_agent_research.core.state import initial_state
from multi_agent_research.runs.models import RunStatus
from multi_agent_research.runs.repository import RunConflictError
from multi_agent_research.runs.service import RunService
from multi_agent_research.sections import workflow
from multi_agent_research.sections.artifacts import (
    bind_claims, dependency_issues, parent_handoff, revision_sections,
)
from multi_agent_research.sections.models import ClaimExtraction, ReportReview, SectionRecord
from multi_agent_research.sections.rendering import evidence_key
from tests.test_run_service import MemoryRunStore
from tests.test_sections import FakeModels, evidence, install


def chapter():
    return SectionRecord(
        section_id="section_1", title="成本", question="公司X的成本优势如何",
        draft="成本的分析结论[来源1]。", sources=[evidence("成本")],
        revision=1, status="complete", artifact_version=2,
    )


async def finish(service, store, run_id):
    # A real graph has asynchronous checkpoint writes; the service unit-test 1s poll is too short.
    await asyncio.wait_for(service._tasks[run_id], timeout=20)
    record = store.runs[run_id]
    assert record.status == RunStatus.COMPLETED, record.error_message
    return record


def extraction():
    return ClaimExtraction(claims=[{
        "statement": "成本的分析结论", "draft_quote": "成本的分析结论",
        "assessment": "supported", "evidence": [{
            "source_number": 1, "quote": "成本 的实际证据摘录", "evidence_id": "model-invented-id",
        }],
    }])


def test_bindings_use_code_identity_and_preserve_contradictions():
    section = chapter()
    data = extraction()
    claims = bind_claims(section, data)
    assert claims[0].claim_id == "section_1:v1:c1"
    assert claims[0].evidence[0].evidence_id == evidence_key(section.sources[0])
    data.claims[0].evidence[0].relation = "contradicts"
    claims = bind_claims(section, data)
    assert claims[0].assessment == "uncertain"
    assert claims[0].evidence[0].relation == "contradicts"
    data.claims[0].evidence = []
    assert bind_claims(section, data)[0].assessment == "unsupported"


@pytest.mark.parametrize("field,value", [
    ("source_number", 99), ("quote", "不存在的原文"), ("quote", "   "),
])
def test_invalid_evidence_cannot_be_persisted_as_verified_binding(field, value):
    data = extraction()
    setattr(data.claims[0].evidence[0], field, value)
    with pytest.raises(ValueError):
        bind_claims(chapter(), data)


def test_claim_must_point_to_actual_draft_excerpt():
    data = extraction()
    data.claims[0].draft_quote = "伪造的正文"
    with pytest.raises(ValueError, match="draft_quote"):
        bind_claims(chapter(), data)


async def completed_parent(monkeypatch):
    fake = FakeModels()
    install(monkeypatch, fake)
    app = build_graph()
    state = initial_state("分析公司X竞争优势")
    result = await app.ainvoke(state, streaming.research_config("parent", state))
    store = MemoryRunStore()
    service = RunService(store)
    parent = await service.create_run(question=state["research_question"], run_id="parent")
    await store.save_sections(parent.run_id, result["sections"])
    await store.save_report_review(parent.run_id, result["report_review"])
    await store.complete_run(parent.run_id, result["final_report"])

    async def get_app():
        return app

    monkeypatch.setattr(streaming, "_get_app", get_app)
    return fake, app, store, service, result


@pytest.mark.asyncio
async def test_revision_only_rebuilds_target_and_synthesis_and_preserves_parent(monkeypatch):
    fake, _, store, service, parent_state = await completed_parent(monkeypatch)
    original = store.runs["parent"].model_dump(mode="json")
    child = await service.create_section_revision(
        "parent", "section_1", instruction="补充成本的反面证据", new_run_id="revision-1",
    )
    seed = child.parent_context.revision_sections
    assert [s.status for s in seed] == ["stale", "complete", "stale"]
    assert child.status == RunStatus.CREATED
    assert child.parent_context.report_excerpt == ""
    assert not child.parent_context.handoff
    await service.start_run(child.run_id)
    completed = await finish(service, store, child.run_id)
    assert completed.sections[1].model_dump(mode="json") == parent_state["sections"][1]
    assert completed.sections[0].revision == 2
    assert completed.sections[2].revision == 2
    assert completed.sections[2].dependency_revisions == {"section_1": 2, "section_2": 1}
    assert fake.calls["plan"] == 1
    assert fake.calls["write:成本"] == 2
    assert fake.calls["write:渠道"] == 1
    assert fake.calls["write:结论"] == 2
    assert completed.sections[0].previous_drafts[0].claims
    assert completed.report_review.verdict == "pass"
    assert store.runs["parent"].model_dump(mode="json") == original
    assert any(e.event_type == "report_review" for e in store.events if e.run_id == child.run_id)

    # A later revision gets fresh per-operation limits, not an exhausted lifetime budget.
    fake.revise = True
    next_child = await service.create_section_revision(
        child.run_id, "section_1", instruction="再次检查反证和口径", new_run_id="revision-2",
    )
    await service.start_run(next_child.run_id)
    again = await finish(service, store, next_child.run_id)
    assert again.sections[0].revision == 4
    assert again.sections[0].search_rounds == 2
    assert again.sections[1] == completed.sections[1]


@pytest.mark.asyncio
async def test_revision_with_no_new_evidence_does_not_reuse_stale_draft(monkeypatch):
    fake, _, store, service, _ = await completed_parent(monkeypatch)
    fake.no_results = True
    child = await service.create_section_revision(
        "parent", "section_1", instruction="重新检索最新年度成本", new_run_id="empty-revision",
    )
    await service.start_run(child.run_id)
    done = await finish(service, store, child.run_id)
    section = done.sections[0]
    assert section.sources == [] and section.claims == []
    assert section.revision == 2 and section.previous_drafts[0].sources
    assert "无法" in section.draft
    assert section.status == "limited"


@pytest.mark.asyncio
async def test_transitive_staleness_is_checked_before_assembly(monkeypatch):
    _, _, _, _, state = await completed_parent(monkeypatch)
    broken = deepcopy(state)
    broken["sections"][0]["revision"] += 1
    with pytest.raises(ValueError, match="dependencies"):
        workflow.assemble_sections(broken)
    sections = [SectionRecord.model_validate(s) for s in state["sections"]]
    assert not dependency_issues(sections)
    revised = revision_sections(sections, "section_2", "补充渠道的证据")
    assert [s.status for s in revised] == ["complete", "stale", "stale"]
    assert sections[1].status == "complete"


@pytest.mark.asyncio
async def test_v2_artifacts_conservatively_invalidate_all_later_chapters(monkeypatch):
    _, _, _, _, state = await completed_parent(monkeypatch)
    sections = [SectionRecord.model_validate(s) for s in state["sections"]]
    for section in sections:
        section.artifact_version = 1
        section.depends_on = []
    revised = revision_sections(sections, "section_1", "补充历史数据证据")
    assert all(s.status == "stale" for s in revised)
    assert revised[1].depends_on == ["section_1"]


@pytest.mark.asyncio
async def test_report_conflicts_are_visible_and_cannot_be_labeled_reviewed(monkeypatch):
    fake = FakeModels()
    install(monkeypatch, fake)

    async def model(system, prompt, schema=None, *, validator=None, context=None):
        if schema is ReportReview:
            output = ReportReview(verdict="pass", issues=[{
                "kind": "scope", "section_ids": ["section_1", "section_2"],
                "detail": "两章年份口径不一致",
            }])
            return (validator(output) if validator else output), {"tokens": 10, "unknown": 0}
        return await fake.model(system, prompt, schema, validator=validator, context=context)

    monkeypatch.setattr(workflow, "call_model", model)
    state = initial_state("分析公司X竞争优势")
    result = await build_graph().ainvoke(state, streaming.research_config("conflict", state))
    assert result["report_review"]["verdict"] == "revise"
    assert result["report_quality"] == "limited"
    assert "两章年份口径不一致" in result["final_report"]


@pytest.mark.asyncio
async def test_selected_parent_handoff_retains_actual_sources_and_unknown_time(monkeypatch):
    _, _, store, service, _ = await completed_parent(monkeypatch)
    child = await service.create_run(
        question="继续检查公司的成本优势", parent_run_id="parent",
        parent_section_ids=["section_1"], run_id="handoff-child",
    )
    handoff = child.parent_context.handoff
    assert [s["section_id"] for s in handoff] == ["section_1"]
    assert handoff[0]["claims"][0]["evidence"][0]["evidence_id"]
    assert handoff[0]["sources"][0]["metadata"]["retrieved_at"]
    assert handoff[0]["reviewed_at"]
    store.runs["parent"].sections[0].sources[0]["content"] = "later changed"
    assert handoff[0]["sources"][0]["content"] != "later changed"
    with pytest.raises(ValueError, match="unknown parent"):
        await service.create_run(question="无效父章节选择", parent_run_id="parent", parent_section_ids=["absent"])
    with pytest.raises(ValueError, match="requires parent"):
        await service.create_run(question="无效的继承设置", parent_section_ids=[])
    old = chapter()
    bundle = parent_handoff([old], None)[0]
    assert bundle["reviewed_at"] is None
    assert "retrieved_at" not in bundle["sources"][0]["metadata"]


@pytest.mark.asyncio
async def test_parent_evidence_is_selected_not_blindly_merged(monkeypatch):
    fake = FakeModels(no_results=True)
    install(monkeypatch, fake)
    old = chapter()
    context = {"schema_version": 2, "source_run_id": "old", "source_question": "旧问题",
               "handoff": parent_handoff([old], None)}
    section = chapter()
    section.sources = []
    section.results = []
    section.parent_section_ids = ["section_1"]
    state = initial_state("研究新的成本问题", context)
    state.update(sections=[section.model_dump(mode="json")], active_section=0)
    result = await workflow.research_section(state)
    source = result["sections"][0]["results"][0]
    assert source["metadata"]["inherited_from_run"] == "old"
    assert "retrieved_at" not in source["metadata"]  # Do not fabricate fresh retrieval time.
    state["sections"][0]["parent_section_ids"] = []
    assert (await workflow.research_section(state))["sections"][0]["results"] == []


@pytest.mark.asyncio
async def test_independent_chapter_does_not_consume_prior_chapter_summary(monkeypatch):
    _, _, _, _, state = await completed_parent(monkeypatch)
    state["active_section"] = 1
    assert workflow._prior_context(state) == "[]"
    state["active_section"] = 2
    assert "成本" in workflow._prior_context(state)
    assert "渠道" in workflow._prior_context(state)


@pytest.mark.asyncio
async def test_fresh_retrieval_replaces_identical_inherited_provenance(monkeypatch):
    fake = FakeModels()
    install(monkeypatch, fake)
    old = chapter()
    old.sources = [evidence(old.question)]
    context = {"schema_version": 2, "source_run_id": "old", "source_question": "旧问题",
               "handoff": parent_handoff([old], None)}
    current = chapter()
    current.sources = []
    current.parent_section_ids = ["section_1"]
    state = initial_state("研究成本优势问题", context)
    state.update(sections=[current.model_dump(mode="json")], active_section=0)
    results = (await workflow.research_section(state))["sections"][0]["results"]
    assert len(results) == 1
    assert results[0]["metadata"]["retrieved_at"]
    assert "inherited_from_run" not in results[0]["metadata"]


@pytest.mark.asyncio
async def test_failed_claim_binding_resumes_without_rewriting_draft(monkeypatch, tmp_path):
    from multi_agent_research.sections import claim_repair
    from tests.test_claim_repair import repair_from_prompt
    fake = FakeModels()
    install(monkeypatch, fake)
    invalid = True
    execution = ['first']
    monkeypatch.setattr(claim_repair, 'epoch', lambda: execution[0])

    async def model(system, prompt, schema=None, *, validator=None, context=None):
        if schema is claim_repair.ClaimRepairs:
            output, cost = repair_from_prompt(prompt, invalid), {'tokens': 10, 'unknown': 0}
        else:
            output, cost = await fake.model(system, prompt, schema)
        if schema is ClaimExtraction and invalid:
            output.claims[0].evidence[0].quote = "不存在的摘录"
        return (validator(output) if validator else output), cost

    monkeypatch.setattr(workflow, "call_model", model)
    state = initial_state("分析公司X竞争优势")
    config = streaming.research_config("claims-recovery", state)
    database = str(tmp_path / "claims.sqlite")
    async with AsyncSqliteSaver.from_conn_string(database) as saver:
        app = build_graph(saver)
        with pytest.raises(claim_repair.ClaimsPending):
            await app.ainvoke(state, config)
        snapshot = await app.aget_state(config)
        assert snapshot.next == ("section_claim_gate",)
        assert snapshot.values["sections"][0]["draft"]
        assert snapshot.values["sections"][0]["claims"] == []
        assert snapshot.values['sections'][0]['claim_work']['attempts'] == 3
    invalid = False
    execution[0] = 'second'
    async with AsyncSqliteSaver.from_conn_string(database) as saver:
        result = await build_graph(saver).ainvoke(None, config)
    assert result["writer_status"] == "complete"
    assert fake.calls["write:成本"] == 1


@pytest.mark.asyncio
async def test_v2_checkpoint_still_skips_new_claims_and_report_nodes(monkeypatch):
    fake = FakeModels()
    install(monkeypatch, fake)
    state = initial_state("分析公司X竞争优势")
    state["workflow_version"] = 2
    result = await build_graph().ainvoke(state, streaming.research_config("v2", state))
    assert result["writer_status"] == "complete"
    assert result["model_calls"] == 10
    assert fake.calls["report_review"] == 0
    assert fake.calls["claims:成本"] == 0


@pytest.mark.asyncio
async def test_revision_api_validates_target_and_preserves_create_start_split(monkeypatch):
    _, _, store, service, _ = await completed_parent(monkeypatch)
    monkeypatch.setattr(server, "run_service", service)
    client = TestClient(server.app)
    url = "/api/runs/parent/sections/section_1/revisions"
    response = client.post(url, json={"instruction": "补充最新成本数据", "run_id": "api-revision"})
    assert response.status_code == 201
    assert response.json()["status"] == "created"
    assert response.json()["parent_context"]["revision_target"] == "section_1"
    assert client.post(url, json={"instruction": "短"}).status_code == 422
    assert client.post(url.replace("section_1", "absent"), json={"instruction": "补充最新成本数据"}).status_code == 422
    assert client.post(url.replace("parent", "missing"), json={"instruction": "补充最新成本数据"}).status_code == 404
    assert store.runs["parent"].status == RunStatus.COMPLETED
    with pytest.raises(RunConflictError):
        await service.create_section_revision("api-revision", "section_1", instruction="未完成任务不能修订")
