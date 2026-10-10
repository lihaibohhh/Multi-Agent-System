"""Budget v2 invariants: continuation changes time, never cumulative expenditure."""
import asyncio
from copy import deepcopy

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from multi_agent_research.core.budget import (
    ExecutionPaused, LegacyBudget, current_budget, invoke_model, migrate_budget, new_budget,
    reserve, settle, start_budget,
)
from multi_agent_research.runs.models import RunStatus
from multi_agent_research.runs.repository import RunConflictError
from multi_agent_research.runs.service import RunService
from tests.test_run_service import MemoryRunStore


def test_cross_day_keeps_ledger_and_pending_reservations(monkeypatch):
    monkeypatch.setattr("multi_agent_research.core.budget.time.time", lambda: 1000)
    first = reserve(start_budget(new_budget()), "known", "model", 200, "write")
    first = settle(first, "known", 150)
    first = reserve(first, "unknown", "model", 300, "review")
    before = deepcopy(first)
    monkeypatch.setattr("multi_agent_research.core.budget.time.time", lambda: 1000 + 7 * 86400)
    after = start_budget(first)
    assert after["deadline"] > before["deadline"]
    assert after["known_tokens"] == 150 and after["charged_tokens"] == 450
    assert after["reservations"] == before["reservations"]
    assert first == before


def test_old_budget_requires_migration_preserves_every_usage_field():
    old = reserve(start_budget(new_budget()), "unknown", "model", 333, "review")
    old.update(version=1, deadline=1)
    before = deepcopy(old)
    with pytest.raises(LegacyBudget):
        start_budget(old)
    migrated = migrate_budget(old)
    assert migrated == {**old, "version": 2, "deadline": None}
    assert old == before


@pytest.mark.asyncio
async def test_every_run_has_an_independent_budget_account():
    store, service = MemoryRunStore(), None
    service = RunService(store)
    parent = await service.create_run(question="parent research", run_id="parent")
    record = await store.begin_execution(parent.run_id, (RunStatus.CREATED,), resume=False)
    await store.reserve_budget("parent", record.execution_id, "one", "model", 200, "write")
    await store.settle_budget("parent", record.execution_id, "one", 123)
    await store.finish_execution("parent", record.execution_id, RunStatus.COMPLETED, {"report": "saved report"})
    child = await service.create_run(question="continue same research", parent_run_id="parent", run_id="child")
    assert child.budget_id != parent.budget_id
    assert child.budget["known_tokens"] == 0
    started = await store.begin_execution("child", (RunStatus.CREATED,), resume=False)
    await store.reserve_budget("child", started.execution_id, "two", "model", 300, "review")
    assert (await service.get_run("parent")).budget["charged_tokens"] == 123
    assert (await service.get_run("child")).budget["charged_tokens"] == 300
    independent = await service.create_run(question="different research", session_id=parent.session_id)
    assert independent.budget_id != parent.budget_id
    assert independent.budget["charged_tokens"] == 0


@pytest.mark.asyncio
async def test_pause_drains_started_model_and_denies_new_requests(monkeypatch):
    store = MemoryRunStore()
    service = RunService(store)
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []
    class Model:
        async def ainvoke(self, messages):
            calls.append(1)
            entered.set()
            await release.wait()
            return AIMessage(content="saved", usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
    async def graph(*args, **kwargs):
        yield "start", {}
        await invoke_model(Model(), [HumanMessage(content="question")])
        # Pausing denies admission but does not prevent settling the first request.
        with pytest.raises(ExecutionPaused):
            await invoke_model(Model(), [HumanMessage(content="second")])
        yield "section_progress", {"sections": [{"section_id": "s", "question": "saved question",
                                                   "title": "Saved", "draft": "kept"}]}
        pytest.fail("must not schedule the next graph step")
    monkeypatch.setattr("multi_agent_research.runs.service.astream_research", graph)
    await service.create_run(question="pause live operation", run_id="r")
    await service.start_run("r")
    task = service._tasks["r"]
    await asyncio.wait_for(entered.wait(), 2)
    await service.pause_run("r")
    assert store.runs["r"].status == RunStatus.RUNNING
    assert store.runs["r"].pause_requested and not task.done()
    release.set()
    await asyncio.wait_for(task, 2)
    assert calls == [1]
    record = store.runs["r"]
    assert record.status == RunStatus.PAUSED and record.sections[0].draft == "kept"
    assert record.budget["known_tokens"] == record.budget["charged_tokens"] == 15
    assert [event.event_type async for event in service.iter_events("r")][-1] == "run_paused"
    assert current_budget.get() is None


@pytest.mark.asyncio
async def test_old_run_migration_is_explicit_idempotent_and_does_not_start():
    store = MemoryRunStore()
    service = RunService(store)
    await service.create_run(question="old research task", run_id="old")
    row = store.runs["old"]
    row.status = RunStatus.FAILED
    row.budget_id = None
    row.budget.update(version=1, deadline=1, known_tokens=99, charged_tokens=99)
    before = deepcopy(row.budget)
    with pytest.raises(RunConflictError, match="迁移"):
        await service.start_run("old", resume=True)
    with pytest.raises(ValueError):
        await service.migrate_run_budget("old", confirm=False, reason="explicit migration")
    assert row.budget == before
    migrated = await service.migrate_run_budget("old", confirm=True, reason="explicit migration")
    again = await service.migrate_run_budget("old", confirm=True, reason="repeat confirmation")
    assert again.budget_id == migrated.budget_id and again.budget["charged_tokens"] == 99
    assert again.status == RunStatus.FAILED and not service._tasks
    assert sum(e.event_type == "budget_migrated" for e in store.events) == 1


@pytest.mark.asyncio
async def test_http_migration_requires_confirmation_and_pause_route(monkeypatch):
    import httpx
    from multi_agent_research.api import server
    store = MemoryRunStore()
    service = RunService(store)
    monkeypatch.setattr(server, "run_service", service)
    await service.create_run(question="legacy migration route", run_id="old")
    row = store.runs["old"]
    row.budget_id = None
    row.budget.update(version=1, deadline=1, known_tokens=88, charged_tokens=88)
    row.status = RunStatus.FAILED
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://test") as client:
        assert (await client.post("/api/runs/old/resume")).status_code == 409
        refused = await client.post("/api/runs/old/budget/migrate", json={"confirm": False, "reason": "explicit test reason"})
        assert refused.status_code == 422 and row.budget_id is None
        migrated = await client.post("/api/runs/old/budget/migrate", json={"confirm": True, "reason": "explicit test reason"})
        assert migrated.status_code == 200
        assert migrated.json()["budget"]["charged_tokens"] == 88
        assert not service._tasks
        assert (await client.post("/api/runs/old/pause")).status_code == 409
        await store.begin_execution("old", (RunStatus.FAILED,), resume=True)
        assert (await client.post("/api/runs/old/pause")).json()["pause_requested"] is True


@pytest.mark.asyncio
async def test_exhausted_budget_rejected_before_new_execution():
    store = MemoryRunStore()
    service = RunService(store)
    await service.create_run(question="budget exhausted", run_id="r")
    row = store.runs["r"]
    row.status = RunStatus.BUDGET_LIMITED
    row.budget["charged_tokens"] = row.budget["policy"]["tokens"]
    before = deepcopy(row)
    for _ in range(3):
        with pytest.raises(RunConflictError, match="Token"):
            await service.start_run("r", resume=True)
    assert store.runs["r"] == before and not service._tasks


@pytest.mark.asyncio
async def test_real_graph_pause_checkpoint_reopen_keeps_completed_chapter(monkeypatch, tmp_path):
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from multi_agent_research.core import streaming
    from multi_agent_research.core.graph import build_graph
    from multi_agent_research.sections import workflow
    from tests.test_sections import FakeModels
    fake = FakeModels()
    class Model:
        def __init__(self, schema=None):
            self.schema = schema
        def with_structured_output(self, schema, **kwargs):
            return Model(schema)
        async def ainvoke(self, messages):
            output, _ = await fake.raw_model(messages[0].content, messages[1].content, self.schema)
            raw = AIMessage(content=output.model_dump_json() if self.schema else output,
                            usage_metadata={"input_tokens": 5, "output_tokens": 5, "total_tokens": 10})
            return {"raw": raw, "parsed": output, "parsing_error": None} if self.schema else raw
    monkeypatch.setattr(workflow, "load_chat_model", lambda _: Model())
    monkeypatch.setattr(workflow, "retrieve_evidence", fake.search)
    store = MemoryRunStore()
    service = RunService(store)
    original_publish = store.publish_execution_event
    async def pause_after_chapter(run_id, execution_id, event_type, payload):
        await original_publish(run_id, execution_id, event_type, payload)
        sections = payload.get("sections", [])
        if sections and sections[0]["status"] == "complete":
            await service.pause_run(run_id)
    monkeypatch.setattr(store, "publish_execution_event", pause_after_chapter)
    async def get_app():
        return app
    monkeypatch.setattr(streaming, "_get_app", get_app)
    await service.create_run(question="分析公司X竞争优势", run_id="pause-reopen")
    database = str(tmp_path / "pause.sqlite")
    async with AsyncSqliteSaver.from_conn_string(database) as saver:
        app = build_graph(saver)
        await service.start_run("pause-reopen")
        await asyncio.wait_for(service._tasks["pause-reopen"], 10)
        row = store.runs["pause-reopen"]
        assert row.status == RunStatus.PAUSED and row.sections[0].status == "complete"
        assert fake.calls["write:成本"] == 1 and fake.calls["write:渠道"] == 0
        before = deepcopy(row.budget)
    monkeypatch.setattr(store, "publish_execution_event", original_publish)
    # New service and reopened persisted graph; ledger never comes from graph state.
    service = RunService(store)
    async with AsyncSqliteSaver.from_conn_string(database) as saver:
        app = build_graph(saver)
        await service.start_run("pause-reopen", resume=True)
        await asyncio.wait_for(service._tasks["pause-reopen"], 10)
    row = store.runs["pause-reopen"]
    assert row.status == RunStatus.COMPLETED
    assert fake.calls["write:成本"] == fake.calls["write:渠道"] == 1
    # Two chapter pipelines plus review, plan/three chapter edits/framing and post-review.
    assert row.budget["model_calls"] == 20 and row.budget["known_tokens"] == 200
    assert before["model_calls"] < row.budget["model_calls"]
    assert all(row.budget["reservations"][key] == entry for key, entry in before["reservations"].items())
