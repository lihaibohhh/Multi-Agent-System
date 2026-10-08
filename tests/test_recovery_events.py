import asyncio

import pytest
from fastapi.testclient import TestClient

from multi_agent_research.api import server
from multi_agent_research.runs.models import RunStatus
from multi_agent_research.runs.service import RunService
from multi_agent_research.sections.model_output import attempt_sink
from tests.test_run_service import MemoryRunStore, _wait_for_status


@pytest.mark.asyncio
async def test_snapshot_resume_cursor_and_execution_generation(monkeypatch):
    store = MemoryRunStore()
    service = RunService(store, poll_interval=0.001)
    async def fail(question, run_id, **kwargs):
        yield "start", {"run_id": run_id}
        raise ValueError("first execution failed")
    monkeypatch.setattr("multi_agent_research.runs.service.astream_research", fail)
    await service.create_run(question="test failed run recovery", run_id="recover")
    await service.start_run("recover")
    failed = await _wait_for_status(store, "recover", RunStatus.FAILED)
    first = await service.get_snapshot("recover")
    assert store.events[-1].event_type == "error"
    assert store.events[-1].sequence == first.cursor
    release = asyncio.Event()
    async def resume(run_id, **kwargs):
        await release.wait()
        yield "done", {"run_id": run_id, "report": "recovered"}
    monkeypatch.setattr("multi_agent_research.runs.service.aresume_research", resume)
    await service.start_run("recover", resume=True)
    snapshot = await service.get_snapshot("recover")
    assert snapshot.run.execution_id != failed.execution_id
    assert snapshot.run.status == RunStatus.RUNNING
    assert snapshot.cursor > first.cursor
    release.set()
    events = [e async for e in service.iter_events("recover", after=snapshot.cursor)]
    assert [e.event_type for e in events] == ["done"]
    assert events[0].payload["execution_id"] == snapshot.run.execution_id
    final = await service.get_snapshot("recover")
    assert final.run.status == RunStatus.COMPLETED and final.run.final_report == "recovered"
    assert final.cursor == events[-1].sequence
    # A late terminal event from the old generation cannot overwrite completion.
    await store.finish_execution("recover", failed.execution_id, RunStatus.FAILED, {"message": "late"})
    assert (await service.get_run("recover")).status == RunStatus.COMPLETED
    monkeypatch.setattr(server, "run_service", service)
    response = TestClient(server.app).get("/api/runs/recover/snapshot")
    assert response.status_code == 200
    assert response.json()["cursor"] == final.cursor


@pytest.mark.asyncio
async def test_failed_node_usage_persists_across_resume_without_exposing_raw(monkeypatch):
    store = MemoryRunStore()
    service = RunService(store)
    async def fail(question, run_id, **kwargs):
        await attempt_sink.get()({"diagnostic_id": "first", "tokens": 10, "unknown": 0, "raw": "private text"})
        raise ValueError("format failed")
        yield
    monkeypatch.setattr("multi_agent_research.runs.service.astream_research", fail)
    await service.create_run(question="usage across node failure", run_id="usage")
    await service.start_run("usage")
    await _wait_for_status(store, "usage", RunStatus.FAILED)
    async def resume(run_id, **kwargs):
        await attempt_sink.get()({"diagnostic_id": "second", "tokens": 12, "unknown": 0, "raw": "second private text"})
        yield "done", {"report": "result"}
    monkeypatch.setattr("multi_agent_research.runs.service.aresume_research", resume)
    await service.start_run("usage", resume=True)
    result = await _wait_for_status(store, "usage", RunStatus.COMPLETED)
    assert result.model_usage == {"attempts": 2, "tokens": 22, "unknown": 0}
    assert "private text" not in result.model_dump_json()
    assert all("private text" not in e.model_dump_json() for e in store.events)
    assert store.events[-1].payload["model_usage"]["tokens"] == 22
    assert attempt_sink.get() is None
