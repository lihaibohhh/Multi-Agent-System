import asyncio
from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.memory import MemorySaver

from multi_agent_research.api import server
from multi_agent_research.core import streaming
from multi_agent_research.core.checkpointer import CheckpointerFactory
from multi_agent_research.core.execution_fence import FencedCheckpointer, execution_fence
from multi_agent_research.core.graph import build_graph
from multi_agent_research.runs.models import RunStatus
from multi_agent_research.runs.repository import StaleExecutionError
from multi_agent_research.runs.runtime import InstanceUnavailableError
from multi_agent_research.runs.service import RunService
from tests.test_run_service import MemoryRunStore
from tests.test_sections import FakeModels, install


@pytest.mark.asyncio
async def test_shutdown_before_runtime_start_does_not_reconcile(monkeypatch):
    store = MemoryRunStore()
    async def unexpected(*args):
        pytest.fail("failed startup must not alter pre-existing runs")
    monkeypatch.setattr(store, "reconcile_running", unexpected)
    await RunService(store).shutdown()


@pytest.mark.asyncio
async def test_checkpoint_write_timeout_releases_execution_guard():
    released = asyncio.Event()
    class Store:
        @asynccontextmanager
        async def guard_execution(self, *args):
            try:
                yield
            finally:
                released.set()
    class SlowSaver(MemorySaver):
        async def aput(self, *args):
            await asyncio.Event().wait()
    token = execution_fence.set((Store(), "r", "e"))
    try:
        saver = FencedCheckpointer(SlowSaver(), write_timeout=0.01)
        with pytest.raises(TimeoutError):
            await saver.aput({"configurable": {"thread_id": "r"}}, {}, {}, {})
        assert released.is_set()
    finally:
        execution_fence.reset(token)


@pytest.mark.asyncio
async def test_start_event_failure_rolls_back_claim(monkeypatch):
    store, service = MemoryRunStore(), None
    service = RunService(store)
    await service.create_run(question="atomic startup failure", run_id="r")
    async def unavailable(*args):
        raise RuntimeError("event storage failed")
    monkeypatch.setattr(store, "append_event", unavailable)
    with pytest.raises(RuntimeError):
        await service.start_run("r")
    assert store.runs["r"].status == RunStatus.CREATED
    assert store.runs["r"].execution_id is None
    assert not service._tasks


@pytest.mark.asyncio
async def test_task_creation_failure_compensates_committed_claim(monkeypatch):
    store = MemoryRunStore()
    service = RunService(store)
    await service.create_run(question="task creation failure", run_id="r")
    def unavailable(*args, **kwargs):
        raise RuntimeError("no background task")
    with monkeypatch.context() as m:
        m.setattr(asyncio, "create_task", unavailable)
        with pytest.raises(RuntimeError, match="background"):
            await service.start_run("r")
    assert store.runs["r"].status == RunStatus.INTERRUPTED
    assert store.events[-1].event_type == "run_interrupted"
    assert await store.can_initialize_missing_checkpoint("r")


@pytest.mark.asyncio
async def test_task_cancelled_before_first_instruction_is_reconciled():
    store = MemoryRunStore()
    service = RunService(store)
    await service.create_run(question="cancel before execution", run_id="r")
    await service.start_run("r")
    service._tasks["r"].cancel()  # No await: the coroutine has never entered its try block.
    await service.shutdown()
    assert store.runs["r"].status == RunStatus.INTERRUPTED
    assert store.events[-1].event_type == "run_interrupted"
    assert not service._tasks


@pytest.mark.asyncio
async def test_shutdown_waits_for_inflight_claim_and_cancels_its_task(monkeypatch):
    store = MemoryRunStore()
    service = RunService(store)
    await service.create_run(question="shutdown claim race", run_id="r")
    claimed, release = asyncio.Event(), asyncio.Event()
    begin = store.begin_execution
    async def delayed(*args, **kwargs):
        record = await begin(*args, **kwargs)
        claimed.set()
        await release.wait()
        return record
    async def waiting(*args, **kwargs):
        await asyncio.Event().wait()
        yield
    monkeypatch.setattr(store, "begin_execution", delayed)
    monkeypatch.setattr("multi_agent_research.runs.service.astream_research", waiting)
    starting = asyncio.create_task(service.start_run("r"))
    await claimed.wait()
    closing = asyncio.create_task(service.shutdown())
    await asyncio.sleep(0)
    release.set()
    await asyncio.wait_for(asyncio.gather(starting, closing), 2)
    assert store.runs["r"].status == RunStatus.INTERRUPTED
    assert not service._tasks


@pytest.mark.asyncio
async def test_terminal_storage_failure_is_reconciled_after_store_recovers(monkeypatch):
    store = MemoryRunStore()
    service = RunService(store)
    await service.create_run(question="storage recovery failure", run_id="r")
    async def broken_stream(*args, **kwargs):
        raise RuntimeError("node failed")
        yield
    async def unavailable(*args):
        raise RuntimeError("database offline")
    monkeypatch.setattr("multi_agent_research.runs.service.astream_research", broken_stream)
    with monkeypatch.context() as m:
        m.setattr(store, "finish_execution", unavailable)
        await service.start_run("r")
        task = service._tasks["r"]
        with pytest.raises(RuntimeError, match="database offline"):
            await task
    assert store.runs["r"].status == RunStatus.RUNNING
    assert await service.recover_stale_runs() == ["r"]
    assert store.runs["r"].status == RunStatus.INTERRUPTED
    count = len(store.events)
    assert await service.recover_stale_runs() == []
    assert len(store.events) == count


@pytest.mark.asyncio
async def test_heartbeat_preserves_active_work_and_stops_on_ownership_loss(monkeypatch):
    store = MemoryRunStore()
    service = RunService(store, heartbeat_interval=0.01)
    entered = asyncio.Event()
    async def waiting(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()
        yield
    monkeypatch.setattr("multi_agent_research.runs.service.astream_research", waiting)
    await service.start_runtime()
    try:
        await service.create_run(question="ownership failure test", run_id="r")
        await service.start_run("r")
        task = service._tasks["r"]
        await entered.wait()
        assert await service.recover_stale_runs() == []
        store.owned = False
        await asyncio.wait_for(service._monitor, 1)
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(InstanceUnavailableError):
            await service.start_run("r", resume=True)
        assert (await service.runtime_health())["ownership"] == "unavailable"
    finally:
        await service.shutdown()


@pytest.mark.asyncio
async def test_old_checkpoint_and_artifact_writes_are_rejected():
    store = MemoryRunStore()
    service = RunService(store)
    await service.create_run(question="old writer isolation", run_id="r")
    old = await store.begin_execution("r", (RunStatus.CREATED,), resume=False)
    saver = FencedCheckpointer(MemorySaver())
    token = execution_fence.set((store, "r", old.execution_id))
    try:
        config = await saver.aput({"configurable": {"thread_id": "r", "checkpoint_ns": ""}},
                                 empty_checkpoint(), {"step": 0, "source": "input", "parents": {}}, {})
        before = await saver.aget_tuple(config)
        await store.interrupt_run("r", "test crash")
        new = await store.begin_execution("r", (RunStatus.INTERRUPTED,), resume=True)
        with pytest.raises(StaleExecutionError):
            await saver.aput(config, empty_checkpoint(), {}, {})
        with pytest.raises(StaleExecutionError):
            await saver.aput_writes(config, [("sections", [])], "old-task")
        with pytest.raises(StaleExecutionError):
            await store.publish_execution_event("r", old.execution_id, "section_progress", {"sections": []})
        with pytest.raises(StaleExecutionError):
            await store.record_model_attempt("r", old.execution_id,
                                             {"diagnostic_id": "old", "tokens": 9, "unknown": 0})
        assert (await saver.aget_tuple(config)).checkpoint == before.checkpoint
        assert store.runs["r"].execution_id == new.execution_id
        assert not store.diagnostics
    finally:
        execution_fence.reset(token)


@pytest.mark.asyncio
async def test_missing_first_checkpoint_reinitializes_without_recreating_run(monkeypatch):
    fake = FakeModels()
    install(monkeypatch, fake)
    app = build_graph()
    async def get_app():
        return app
    monkeypatch.setattr(streaming, "_get_app", get_app)
    store = MemoryRunStore()
    service = RunService(store)
    await service.create_run(question="分析公司X竞争优势", run_id="r")
    await store.begin_execution("r", (RunStatus.CREATED,), resume=False)  # Process exits here.
    await service.recover_stale_runs()
    await service.start_run("r", resume=True)
    await asyncio.wait_for(service._tasks["r"], 20)
    assert store.runs["r"].status == RunStatus.COMPLETED
    assert fake.calls["plan"] == 1
    assert any(e.payload.get("reinitialized") for e in store.events)


@pytest.mark.asyncio
async def test_missing_checkpoint_with_prior_usage_is_not_silently_restarted(monkeypatch):
    app = build_graph()
    async def get_app():
        return app
    monkeypatch.setattr(streaming, "_get_app", get_app)
    store = MemoryRunStore()
    service = RunService(store)
    await service.create_run(question="missing existing checkpoint", run_id="r")
    old = await store.begin_execution("r", (RunStatus.CREATED,), resume=False)
    await store.record_model_attempt("r", old.execution_id, {"diagnostic_id": "d", "tokens": 10, "unknown": 0})
    await store.interrupt_run("r", "checkpoint lost")
    await service.start_run("r", resume=True)
    await service._tasks["r"]
    assert store.runs["r"].status == RunStatus.FAILED
    assert "拒绝静默重做" in store.runs["r"].error_message
    assert len(store.diagnostics) == 1


@pytest.fixture
def isolated_factory(monkeypatch):
    monkeypatch.setattr(CheckpointerFactory, "_instances", {})
    monkeypatch.setattr(CheckpointerFactory, "_effective", {})
    monkeypatch.setattr(CheckpointerFactory, "_lifecycle", {})
    monkeypatch.setattr(CheckpointerFactory, "_lock", None)


@pytest.mark.asyncio
async def test_strict_checkpoint_rejects_memory_and_cached_fallback(monkeypatch, isolated_factory):
    from multi_agent_research.core.config import settings
    monkeypatch.setattr(settings.agent, "checkpoint_backend", "memory")
    with pytest.raises(RuntimeError, match="API"):
        await CheckpointerFactory.create(require_durable=True)
    monkeypatch.setattr(settings.agent, "checkpoint_backend", "postgres")
    async def fallback(cls, backend, cache_key):
        return MemorySaver(), "memory"
    monkeypatch.setattr(CheckpointerFactory, "_create_instance", classmethod(fallback))
    with pytest.raises(RuntimeError, match="拒绝降级"):
        await CheckpointerFactory.create(require_durable=True)
    with pytest.raises(RuntimeError, match="拒绝降级"):
        await CheckpointerFactory.create(require_durable=True)
    health = await CheckpointerFactory.health_check()
    assert not health["persistent"] and health["backend"] == "memory"


@pytest.mark.asyncio
async def test_sqlite_persistence_health_is_verified(monkeypatch, isolated_factory, tmp_path):
    from multi_agent_research.core.config import settings
    monkeypatch.setattr(settings.agent, "checkpoint_backend", "sqlite")
    monkeypatch.setattr(settings.agent, "checkpoint_db_path", str(tmp_path / "health.sqlite"))
    try:
        await CheckpointerFactory.create(require_durable=True)
        result = await CheckpointerFactory.health_check()
        assert result["persistent"] and result["status"] == result["sqlite"] == "ok"
    finally:
        await CheckpointerFactory.close_all()


def test_health_endpoint_reports_unavailable_not_false_green(monkeypatch):
    async def ready():
        return True
    async def memory():
        return {"status": "ok", "backend": "memory", "persistent": False}
    async def runtime():
        return {"ownership": "ok", "monitor": "ok", "reconciliation": "ok"}
    monkeypatch.setattr(server.run_repository, "health_check", ready)
    monkeypatch.setattr(CheckpointerFactory, "health_check", memory)
    monkeypatch.setattr(server.run_service, "runtime_health", runtime)
    response = TestClient(server.app).get("/api/health")
    assert response.status_code == 503 and response.json()["status"] == "unavailable"


def test_start_without_instance_ownership_returns_503(monkeypatch):
    store = MemoryRunStore()
    store.owned = False
    monkeypatch.setattr(server, "run_service", RunService(store))
    assert TestClient(server.app).post("/api/runs/r/start").status_code == 503


@pytest.mark.asyncio
async def test_failed_startup_does_not_reconcile_and_all_cleanup_runs(monkeypatch):
    calls = []
    async def opened():
        calls.append("open")
    async def denied():
        calls.append("acquire")
        raise InstanceUnavailableError("another instance")
    async def unexpected():
        pytest.fail("setup/reconciliation must not run without ownership")
    async def closed():
        calls.append("release")
    async def stop():
        calls.append("stop_workers")
    async def close_saver():
        calls.append("close_saver")
    class Knowledge:
        async def aclose(self):
            calls.append("close_knowledge")
            raise RuntimeError("cleanup failure")
    monkeypatch.setattr(server.run_repository, "open", opened)
    monkeypatch.setattr(server.run_repository, "acquire_instance", denied)
    monkeypatch.setattr(server.run_repository, "setup", unexpected)
    monkeypatch.setattr(server.run_repository, "close", closed)
    monkeypatch.setattr(server.run_service, "start_runtime", unexpected)
    monkeypatch.setattr(server.run_service, "shutdown", stop)
    monkeypatch.setattr(CheckpointerFactory, "close_all", close_saver)
    monkeypatch.setattr(server, "get_knowledge_service_client", lambda: Knowledge())
    monkeypatch.setattr(streaming, "_compiled_app", None)
    with pytest.raises(RuntimeError, match="cleanup failure"):
        async with server.lifespan(server.app):
            pytest.fail("must not serve requests")
    assert calls == ["open", "acquire", "stop_workers", "close_knowledge", "close_saver", "release"]
