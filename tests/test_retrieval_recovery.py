"""Deterministic mixed-success cases; no external network or model calls."""
import asyncio
from collections import Counter
from copy import deepcopy

import pytest
from pydantic import SecretStr

from multi_agent_research.core.budget import BudgetExceeded, RunControlError
from multi_agent_research.core.config import settings
from multi_agent_research.core.retrieval import (
    RetrievalDeferred, RetrievalFailed, durable_retrieval, operation_key,
)
from multi_agent_research.retrieval import RetrievalRequest
from multi_agent_research.retrieval import service as retrieval_service
from multi_agent_research.runs.models import RunStatus
from multi_agent_research.sections.artifacts import stamp_results
from tests.test_run_budget import scope_for
from tests.test_sections import evidence


DESCRIPTOR = {"provider": "knowledge", "query": "q", "section_id": "s1", "round": 0, "revision": 0}


def test_key_separates_context_but_not_execution_and_timestamp_preserved():
    original = operation_key(DESCRIPTOR)
    for key, value in {"provider": "web", "query": "q2", "section_id": "s2", "round": 1, "revision": 1}.items():
        assert operation_key({**DESCRIPTOR, key: value}) != original
    assert operation_key(dict(reversed(list(DESCRIPTOR.items())))) == original
    item = evidence("old")
    item["metadata"]["retrieved_at"] = "2026-10-01"
    assert stamp_results([item])[0]["metadata"]["retrieved_at"] == "2026-10-01"


@pytest.mark.asyncio
async def test_mixed_batch_resume_reuses_success_and_only_retries_failed_query(monkeypatch):
    calls = Counter()
    recovered = False
    async def search(q, iteration):
        calls[q] += 1
        if q == "gap" and not recovered:
            raise TimeoutError("injected dependency timeout")
        # Slow successes must not be cancelled by a faster failed sibling.
        if q == "question":
            await asyncio.sleep(0.03)
        return [evidence(q)]
    monkeypatch.setattr(retrieval_service, "_knowledge_search", search)
    monkeypatch.setattr(settings.tool_secrets, "tavily_api_key", SecretStr(""))
    request = RetrievalRequest(
        question="question",
        gaps=("gap",),
        scope={"section_id": "s1", "round": 0, "revision": 0},
    )
    before = deepcopy(request)
    async with scope_for() as (store, _, record):
        with pytest.raises(RetrievalDeferred):
            await retrieval_service.retrieve_evidence(request)
        assert request == before
        assert calls == {"question": 1, "gap": 2}
        assert store.runs[record.run_id].budget["retrieval_calls"] == 3
        saved = next(v for v in store.retrievals.values() if v["status"] == "succeeded")
        timestamp = saved["results"][0]["metadata"]["retrieved_at"]
        await store.finish_execution(record.run_id, record.execution_id, RunStatus.PAUSED, {})
    recovered = True
    async with scope_for(store) as (_, _, record):
        result = await retrieval_service.retrieve_evidence(request)
        assert calls == {"question": 1, "gap": 3}
        assert len(result) == 2
        assert result[0]["metadata"]["retrieved_at"] == timestamp
        budget = store.runs[record.run_id].budget
        assert budget["retrieval_calls"] == 4
        assert sum(v["status"] == "unknown" for v in budget["reservations"].values()) == 2
        assert any(e.payload.get("reused") for e in store.events)


@pytest.mark.asyncio
async def test_total_attempt_limit_survives_repeated_resume():
    calls = 0
    async def fail():
        nonlocal calls
        calls += 1
        raise ConnectionError("offline")
    store = None
    for expected in (RetrievalDeferred, RetrievalFailed, RetrievalFailed):
        async with scope_for(store) as (store, _, record):
            with pytest.raises(expected):
                await durable_retrieval(fail, DESCRIPTOR)
            await store.finish_execution(record.run_id, record.execution_id, RunStatus.PAUSED, {})
    assert calls == 4
    assert store.runs[record.run_id].budget["retrieval_calls"] == 4


@pytest.mark.asyncio
async def test_empty_success_replayed_at_full_quota_without_another_call(monkeypatch):
    monkeypatch.setattr(settings.agent, "run_max_retrieval_calls", 1)
    calls = 0
    async def empty():
        nonlocal calls
        calls += 1
        return []
    async with scope_for() as (store, _, record):
        assert await durable_retrieval(empty, DESCRIPTOR) == []
        assert await durable_retrieval(empty, DESCRIPTOR) == []
        with pytest.raises(BudgetExceeded):
            await durable_retrieval(empty, {**DESCRIPTOR, "round": 1})
        assert calls == 1 and len(store.retrievals) == 1
        assert store.runs[record.run_id].budget["retrieval_calls"] == 1


@pytest.mark.asyncio
async def test_budget_stops_retry_and_permanent_error_is_not_empty(monkeypatch):
    monkeypatch.setattr(settings.agent, "run_max_retrieval_calls", 1)
    calls = 0
    async def failure():
        nonlocal calls
        calls += 1
        raise ConnectionError("network")
    async with scope_for() as (store, _, record):
        with pytest.raises(BudgetExceeded):
            await durable_retrieval(failure, DESCRIPTOR)
        assert calls == 1
        assert next(iter(store.retrievals.values()))["status"] == "retryable_failed"
    async def invalid():
        raise ValueError("do not expose raw response")
    async with scope_for(run_id="permanent") as (store, _, record):
        with pytest.raises(RetrievalFailed, match="ValueError"):
            await durable_retrieval(invalid, DESCRIPTOR)
        assert store.runs[record.run_id].budget["retrieval_calls"] == 1
        assert "results" not in next(iter(store.retrievals.values()))


@pytest.mark.asyncio
async def test_cancelled_unknown_attempt_can_resume_but_old_worker_cannot_complete():
    entered = asyncio.Event()
    async def hang():
        entered.set()
        await asyncio.Event().wait()
    async with scope_for() as (store, _, old):
        task = asyncio.create_task(durable_retrieval(hang, DESCRIPTOR))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        attempt = deepcopy(next(iter(store.retrievals.values()))["attempts"][0])
        await store.finish_execution(old.run_id, old.execution_id, RunStatus.INTERRUPTED, {})
    async def success():
        return [evidence("new")]
    async with scope_for(store) as (_, _, resumed):
        await durable_retrieval(success, DESCRIPTOR)
        data = next(iter(store.retrievals.values()))
        assert data["attempts"][0]["status"] == "unknown"
        assert store.runs[resumed.run_id].budget["retrieval_calls"] == 2
        with pytest.raises(Exception, match="stale execution"):
            await store.finish_retrieval(old.run_id, old.execution_id, operation_key(DESCRIPTOR),
                                        attempt["reservation_id"], [], None)
        assert data["results"][0]["query"] == "new"


@pytest.mark.asyncio
async def test_storage_failures_fail_closed_and_duplicate_queries_share_attempt(monkeypatch):
    calls = 0
    async def success():
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.01)
        return []
    async with scope_for() as (store, _, record):
        await asyncio.gather(*(durable_retrieval(success, DESCRIPTOR) for _ in range(3)))
        assert calls == store.runs[record.run_id].budget["retrieval_calls"] == 1
        async def offline(*args):
            raise RuntimeError("database offline")
        monkeypatch.setattr(store, "begin_retrieval", offline)
        with pytest.raises(RunControlError, match="持久化"):
            await durable_retrieval(success, DESCRIPTOR)
        assert calls == 1


@pytest.mark.asyncio
async def test_service_marks_dependency_pause_and_can_resume(monkeypatch):
    from multi_agent_research.runs.service import RunService
    from tests.test_run_service import MemoryRunStore
    store = MemoryRunStore()
    async def fail(*args, **kwargs):
        raise RetrievalDeferred("dependency offline, results saved")
        yield
    async def recover(*args, **kwargs):
        yield "done", {"report": "recovered"}
    monkeypatch.setattr("multi_agent_research.runs.service.astream_research", fail)
    monkeypatch.setattr("multi_agent_research.runs.service.aresume_research", recover)
    service = RunService(store)
    await service.create_run(question="test recovery", run_id="r")
    await service.start_run("r")
    await service._tasks["r"]
    assert store.runs["r"].status == RunStatus.PAUSED
    assert store.events[-1].payload["reason"] == "dependency_unavailable"
    await service.start_run("r", resume=True)
    await service._tasks["r"]
    assert store.runs["r"].status == RunStatus.COMPLETED
    await service.shutdown()


@pytest.mark.asyncio
async def test_pause_during_backoff_blocks_next_attempt_and_save_failure_keeps_unknown(monkeypatch):
    from multi_agent_research.core.budget import ExecutionPaused
    async with scope_for() as (store, _, record):
        calls = 0
        async def fail_then_pause():
            nonlocal calls
            calls += 1
            await store.request_pause(record.run_id)
            raise ConnectionError("temporary")
        with pytest.raises(ExecutionPaused):
            await durable_retrieval(fail_then_pause, DESCRIPTOR)
        assert calls == store.runs[record.run_id].budget["retrieval_calls"] == 1
    async with scope_for(run_id="failed-save") as (store, _, record):
        async def empty():
            return []
        async def offline(*args):
            raise RuntimeError("database unavailable")
        monkeypatch.setattr(store, "finish_retrieval", offline)
        with pytest.raises(RunControlError):
            await durable_retrieval(empty, DESCRIPTOR)
        assert next(iter(store.retrievals.values()))["status"] == "inflight"
        assert next(iter(store.runs[record.run_id].budget["reservations"].values()))["status"] == "unknown"


@pytest.mark.asyncio
async def test_sqlite_graph_restart_replays_receipt_and_preserves_completed_chapter(monkeypatch, tmp_path):
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from multi_agent_research.core import streaming
    from multi_agent_research.core.graph import build_graph
    from multi_agent_research.sections import workflow
    from multi_agent_research.sections.models import SectionRecord
    from multi_agent_research.runs.service import RunService
    from tests.test_run_service import MemoryRunStore
    from tests.test_sections import FakeModels

    calls = Counter()
    fail = True
    fake = FakeModels()
    async def search(query, iteration):
        calls[query] += 1
        if query == "渠道补充" and fail:
            raise TimeoutError("controlled second query failure")
        return [evidence("渠道")]
    monkeypatch.setattr(workflow, "call_model", fake.model)
    monkeypatch.setattr(retrieval_service, "_knowledge_search", search)
    monkeypatch.setattr(settings.tool_secrets, "tavily_api_key", SecretStr(""))
    store = MemoryRunStore()
    service = RunService(store)
    await service.create_run(question="分析公司X竞争优势", run_id="receipt-graph")
    first = SectionRecord(section_id="s1", title="成本", question="公司X的成本优势如何",
                          status="complete", revision=1, draft="已完成的成本章节", summary="成本已完成")
    second = SectionRecord(section_id="s2", title="渠道", question="公司X的渠道优势如何", gaps=["渠道补充"])
    from multi_agent_research.core.state import initial_state
    state = initial_state("分析公司X竞争优势")
    state.update(sections=[s.model_dump(mode="json") for s in (first, second)], active_section=1,
                 section_step="research")
    path = str(tmp_path / "receipt.sqlite")
    async def get_app():
        return app
    monkeypatch.setattr(streaming, "_get_app", get_app)
    # Initialize a saved section boundary, then use real resume routing twice.
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        app = build_graph(saver)
        await app.aupdate_state(
            streaming.research_config("receipt-graph", state),
            state,
            as_node="plan_sections",
        )
        await store.finish_execution("receipt-graph", None, RunStatus.PAUSED, {})
        store.runs["receipt-graph"].status = RunStatus.PAUSED
        await service.start_run("receipt-graph", resume=True)
        await service._tasks["receipt-graph"]
        assert store.runs["receipt-graph"].status == RunStatus.PAUSED
        snapshot = await app.aget_state(streaming.research_config("receipt-graph", state))
        assert snapshot.next == ("section_cycle",)
        assert snapshot.values["sections"][0]["draft"] == first.draft
    fail = False
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        app = build_graph(saver)
        await service.start_run("receipt-graph", resume=True)
        await service._tasks["receipt-graph"]
        record = store.runs["receipt-graph"]
        assert record.status == RunStatus.COMPLETED
        assert record.sections[0].draft == first.draft
        assert calls[second.question] == 1 and calls["渠道补充"] == 3
        assert record.budget["retrieval_calls"] == 4
        assert fake.calls["write:成本"] == 0
        assert fake.calls["write:渠道"] == 1
    await service.shutdown()
