import asyncio
from copy import deepcopy

import pytest
from fastapi.testclient import TestClient

from multi_agent_research.api import server
from multi_agent_research.core import streaming
from multi_agent_research.core.budget import RunBudget, current_budget
from multi_agent_research.core.retrieval import durable_retrieval, RetrievalFailed, admit
from multi_agent_research.core.state import initial_state
from multi_agent_research.runs.models import RunStatus, ParentContextSnapshot
from multi_agent_research.runs.repository import RunConflictError
from multi_agent_research.sections.models import SectionRecord
from multi_agent_research.sections.operations import prepare_operation
from tests.test_section_revisions import completed_parent
from tests.test_sections import evidence


async def execute(service, store, child):
    await service.start_run(child.run_id)
    await asyncio.wait_for(service._tasks[child.run_id], 15)
    result = store.runs[child.run_id]
    assert result.status == RunStatus.COMPLETED, result.error_message
    return result


@pytest.mark.asyncio
async def test_supplement_preserves_body_and_claims_then_continue_rebuilds_only_dependents(monkeypatch):
    fake, _, store, service, original = await completed_parent(monkeypatch)
    before = deepcopy(store.runs["parent"].model_dump())
    calls = deepcopy(fake.calls)
    child = await service.create_section_operation("parent", "section_1", mode="supplement",
        instruction="只补充最新成本证据", new_run_id="evidence-child")
    assert child.budget_id == store.runs["parent"].budget_id
    assert child.status == RunStatus.CREATED and fake.calls == calls
    done = await execute(service, store, child)
    target = done.sections[0]
    assert target.draft == original["sections"][0]["draft"]
    assert [c.model_dump(mode="json") for c in target.claims] == original["sections"][0]["claims"]
    assert target.revision == 1 and target.status == "evidence_ready"
    assert done.sections[1].model_dump(mode="json") == original["sections"][1]
    assert done.sections[2].status == "stale"
    assert "不是完整研究报告" in done.final_report
    assert fake.calls["write:成本"] == calls["write:成本"]
    assert fake.calls["write:结论"] == calls["write:结论"]
    assert store.runs["parent"].model_dump() == before
    with pytest.raises(RunConflictError, match="阶段产物"):
        await service.create_run(question="不要继承未完成的旧结论", parent_run_id=done.run_id)
    next_child = await service.create_section_operation(done.run_id, "section_1", mode="continue",
        instruction="使用补充证据继续写作", new_run_id="write-child")
    search_count = fake.calls["search"]
    final = await execute(service, store, next_child)
    assert fake.calls["search"] == search_count  # existing candidates and synthesis handoff
    assert final.sections[0].revision == 2 and final.sections[2].revision == 2
    assert final.sections[1] == done.sections[1]
    assert final.report_review is not None
    assert final.sections[2].dependency_revisions == {"section_1": 2, "section_2": 1}
    await service.shutdown()


@pytest.mark.asyncio
async def test_empty_supplement_waits_and_refresh_never_silently_reuses_stale_evidence(monkeypatch):
    fake, _, store, service, original = await completed_parent(monkeypatch)
    fake.no_results = True
    supplement = await service.create_section_operation("parent", "section_1", mode="supplement",
        instruction="补充最新年度成本来源")
    done = await execute(service, store, supplement)
    assert done.sections[0].status == "waiting_evidence"
    assert done.sections[0].draft == original["sections"][0]["draft"]
    assert "本次检索没有返回" in done.sections[0].limitations[0]
    refreshed = await service.create_section_operation("parent", "section_1", mode="refresh",
        instruction="放弃旧来源重新查询最新情况")
    final = await execute(service, store, refreshed)
    assert final.sections[0].sources == []
    assert final.sections[0].previous_drafts[0].sources
    assert "无法对本章问题作出可靠结论" in final.sections[0].draft
    await service.shutdown()


def test_scope_rejects_unfinished_prerequisite_and_does_not_pull_unrelated_work():
    sections = [SectionRecord(section_id="a", title="A", question="研究成本是什么", artifact_version=2),
                SectionRecord(section_id="b", title="B", question="研究渠道是什么", artifact_version=2),
                SectionRecord(section_id="c", title="C", question="研究综合结论是什么", artifact_version=2,
                              kind="synthesis", depends_on=["a", "b"])]
    with pytest.raises(ValueError, match="前置章节"):
        prepare_operation(sections, "c", "continue", "继续综合结论")
    copied, plan = prepare_operation(sections, "b", "continue", "先完成渠道研究")
    assert plan["work_ids"] == ["b"] and plan["affected_ids"] == ["b", "c"]
    assert copied[0] == sections[0] and copied[2].status == "stale"
    assert sections[2].status == "pending"  # pure transition


@pytest.mark.asyncio
async def test_continue_uses_checkpoint_not_lagging_projection_and_leaves_other_pending_chapter(monkeypatch):
    fake, app, store, service, original = await completed_parent(monkeypatch)
    state = initial_state("分析公司X竞争优势")
    sections = deepcopy(original["sections"])
    sections[0].update(status="pending", draft="", revision=0)
    sections[1].update(status="researching", search_rounds=1, draft="", revision=0)
    sections[2].update(status="pending", draft="", revision=0)
    state.update(sections=sections, active_section=1, section_step="research",
                 section_policy={"max_search_rounds": 1, "max_revisions": 0})
    # A new fixture Run with a checkpoint ahead of its deliberately empty projection.
    parent = await service.create_run(question="分析公司X竞争优势", run_id="unfinished")
    await app.aupdate_state(
        streaming.research_config(parent.run_id, state), state, as_node="plan_sections"
    )
    store.runs[parent.run_id].status = RunStatus.PAUSED
    child = await service.create_section_operation(parent.run_id, "section_2", mode="continue",
        instruction="仅继续渠道章节研究")
    assert child.parent_context.section_operation["section_policy"]["max_search_rounds"] == 1
    with pytest.raises(RunConflictError, match="已有选章继续"):
        await service.create_section_operation(parent.run_id, "section_2", mode="continue", instruction="再次点击同一操作")
    final = await execute(service, store, child)
    assert final.sections[0].status == "pending"
    assert final.sections[1].status == "complete"
    assert final.sections[2].status == "stale"
    assert final.report_review is None and "不是完整研究报告" in final.final_report
    assert store.runs[parent.run_id].sections == []
    await service.shutdown()


def test_new_supplement_sources_are_not_all_hidden_by_old_higher_scores():
    from multi_agent_research.sections.workflow import _select_sources
    from multi_agent_research.sections.rendering import evidence_key
    old = [evidence(str(i), score=0.99) for i in range(20)]
    fresh = evidence("最新反证", score=0.1)
    section = SectionRecord(section_id="s", title="标题", question="研究问题是什么",
                            results=old + [fresh], evidence_update={"source_ids": [evidence_key(fresh)]})
    selected = _select_sources(section)
    assert selected[0] == fresh and len(selected) == 15


@pytest.mark.asyncio
async def test_operation_api_rejects_running_unknown_mode_and_empty_instruction(monkeypatch):
    _, _, store, service, _ = await completed_parent(monkeypatch)
    monkeypatch.setattr(server, "run_service", service)
    client = TestClient(server.app)
    path = "/api/runs/parent/sections/section_1/operations"
    for payload in ({"mode": "surprise", "instruction": "补充最新成本证据"}, {"mode": "refresh", "instruction": "短"}):
        assert client.post(path, json=payload).status_code == 422
    response = client.post(path, json={"mode": "supplement", "instruction": "补充最新成本证据"})
    assert response.status_code == 201 and response.json()["status"] == "created"
    assert not service._tasks
    assert response.json()["budget_id"] == store.runs["parent"].budget_id
    store.runs["parent"].status = RunStatus.RUNNING
    assert client.post(path, json={"mode": "refresh", "instruction": "补充最新成本证据"}).status_code == 409


@pytest.mark.asyncio
async def test_continue_receipts_preserve_success_and_failed_attempt_cap(monkeypatch):
    from multi_agent_research.core.retrieval import operation_key
    fake, _, store, service, _ = await completed_parent(monkeypatch)
    parent = store.runs["parent"]
    descriptor = {"provider": "knowledge", "query": "test"}
    key = operation_key(descriptor)
    data, _ = admit(None, descriptor, "old", "old-request")
    data.update(status="succeeded", results=[evidence("saved")])
    data["attempts"][0]["status"] = "succeeded"
    store.retrievals[parent.run_id, key] = data
    context = ParentContextSnapshot(source_run_id=parent.run_id, source_question=parent.question,
        captured_at=parent.created_at, report_excerpt="", report_truncated=False,
        section_operation={"mode": "continue"})
    child = await store.create_run(run_id="receipt-child", session_id=parent.session_id,
        parent_run_id=parent.run_id, parent_context=context, question=parent.question)
    child = await store.begin_execution(child.run_id, (RunStatus.CREATED,), resume=False)
    token = current_budget.set(RunBudget(store, child, asyncio.Semaphore(2)))
    async def forbidden():
        pytest.fail("saved request should not be sent")
    try:
        before = deepcopy(store.runs[child.run_id].budget)
        assert (await durable_retrieval(forbidden, descriptor))[0]["query"] == "saved"
        assert store.runs[child.run_id].budget == before
        failed_descriptor = {**descriptor, "query": "failed"}
        failed = {**data, "descriptor": failed_descriptor, "status": "retryable_failed",
                  "attempts": [{"execution_id": "old", "status": "retryable_failed"}] * 4}
        store.retrievals[parent.run_id, operation_key(failed_descriptor)] = failed
        with pytest.raises(RetrievalFailed, match="4"):
            await durable_retrieval(forbidden, failed_descriptor)
        assert store.runs[child.run_id].budget == before
    finally:
        current_budget.reset(token)
