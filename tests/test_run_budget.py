import asyncio
import time
from contextlib import asynccontextmanager
import sys
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import SecretStr

from multi_agent_research.core.config import settings
from multi_agent_research.core.budget import (
    BudgetExceeded, CallTimeout, RunBudget, RunControlError, current_budget,
    gather_cancel_on_error, invoke_model, invoke_retrieval, new_budget, reserve, settle, start_budget,
)
from multi_agent_research.runs.models import RunStatus
from multi_agent_research.runs.service import RunService
from multi_agent_research.sections.model_output import invoke_checked
from multi_agent_research.sections.models import SectionReview
from tests.test_model_output import fake_provider, BAD, GOOD
from tests.test_run_service import MemoryRunStore


@asynccontextmanager
async def scope_for(store=None, run_id="budget", semaphore=None):
    store = store or MemoryRunStore()
    service = RunService(store)
    if run_id not in store.runs:
        await service.create_run(question="budgeted research", run_id=run_id)
    record = store.runs[run_id]
    if record.status != RunStatus.RUNNING:
        record = await store.begin_execution(run_id, (record.status,), resume=record.status != RunStatus.CREATED)
    token = current_budget.set(RunBudget(store, record, semaphore or asyncio.Semaphore(2)))
    try:
        yield store, service, record
    finally:
        current_budget.reset(token)


def test_reservation_settlement_and_unknown_usage_are_conservative():
    budget = start_budget(new_budget())
    budget["policy"]["tokens"] = 100
    first = reserve(budget, "a", "model", 80, "write:s1")
    assert budget["model_calls"] == 0  # pure / rollback-safe
    with pytest.raises(BudgetExceeded):
        reserve(first, "b", "model", 30, "review:s1")
    assert settle(first, "a", None) == first
    done = settle(first, "a", 10)
    assert done["known_tokens"] == done["charged_tokens"] == 10
    assert settle(done, "a", 20) == done  # idempotent
    with pytest.raises(RunControlError):
        reserve(done, "a", "model", 10, "duplicate")
    # Actual usage can exceed the admission estimate; keep truth, block further calls.
    over = settle(first, "a", 120)
    assert over["charged_tokens"] == 120
    with pytest.raises(BudgetExceeded):
        reserve(over, "b", "model", 1, "next")


@pytest.mark.asyncio
async def test_schema_correction_consumes_same_durable_model_budget(monkeypatch):
    monkeypatch.setattr(settings.agent, "run_max_model_calls", 1)
    async with scope_for() as (store, _, record), fake_provider([BAD, GOOD]) as (model, requests):
        with pytest.raises(BudgetExceeded, match="model_calls"):
            await invoke_checked(model, "JSON review", "source", SectionReview)
        budget = store.runs[record.run_id].budget
        assert len(requests) == budget["model_calls"] == 1
        assert budget["known_tokens"] == budget["charged_tokens"] == 10


@pytest.mark.asyncio
async def test_resume_keeps_limits_and_unknown_reservation_but_renews_execution_time(monkeypatch):
    monkeypatch.setattr(settings.agent, "run_max_model_calls", 1)
    async with scope_for() as (store, _, record):
        await store.reserve_budget(record.run_id, record.execution_id, "inflight", "model", 999, "write")
        original = store.runs[record.run_id].budget
        await store.interrupt_run(record.run_id, "process died")
    monkeypatch.setattr(settings.agent, "run_max_model_calls", 100)
    # Explicitly advance time; two fast Windows clock reads can otherwise be identical.
    monkeypatch.setattr("multi_agent_research.core.budget.time.time", lambda: original["deadline"] + 86400)
    async with scope_for(store) as (_, _, resumed), fake_provider([GOOD]) as (model, requests):
        with pytest.raises(BudgetExceeded):
            await invoke_model(model, [HumanMessage(content="hello")])
        assert not requests
        assert resumed.budget["deadline"] > original["deadline"]
        assert {k: v for k, v in resumed.budget.items() if k != "deadline"} == {k: v for k, v in original.items() if k != "deadline"}
        assert not await store.can_initialize_missing_checkpoint(record.run_id)
    # An explicitly independent research uses a new policy and account.
    child = await RunService(store).create_run(question="new budgeted task", run_id="new")
    assert child.budget["policy"]["model_calls"] == 100
    assert child.budget["model_calls"] == 0


@pytest.mark.asyncio
async def test_failed_reservation_never_sends_request(monkeypatch):
    async with scope_for() as (store, _, _), fake_provider([GOOD]) as (model, requests):
        async def unavailable(*args):
            raise RuntimeError("database offline")
        monkeypatch.setattr(store, "reserve_budget", unavailable)
        with pytest.raises(RunControlError, match="未发送"):
            await invoke_model(model, [HumanMessage(content="hello")])
        assert not requests


@pytest.mark.asyncio
async def test_missing_usage_keeps_reserved_tokens():
    async with scope_for() as (store, _, record), fake_provider([GOOD], usage=False) as (model, requests):
        await invoke_model(model, [HumanMessage(content="hello")])
        budget = store.runs[record.run_id].budget
        assert len(requests) == 1 and budget["known_tokens"] == 0
        assert budget["charged_tokens"] > 0
        assert next(iter(budget["reservations"].values()))["status"] == "unknown"


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_model_timeout_or_cancel_retains_reservation(monkeypatch, cancel):
    monkeypatch.setattr(settings.agent, "model_call_timeout", 0.02)
    entered, stopped = asyncio.Event(), asyncio.Event()
    class SlowModel:
        async def ainvoke(self, messages):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
    async with scope_for() as (store, _, record):
        task = asyncio.create_task(invoke_model(SlowModel(), [HumanMessage(content="hello")]))
        await entered.wait()
        if cancel:
            task.cancel()
        with pytest.raises(asyncio.CancelledError if cancel else CallTimeout):
            await task
        assert stopped.is_set()
        assert store.runs[record.run_id].budget["charged_tokens"] > 0


@pytest.mark.asyncio
async def test_execution_deadline_pauses_graph_and_resume_restarts(monkeypatch):
    monkeypatch.setattr(settings.agent, "run_timeout", 0.03)
    store, calls, stopped = MemoryRunStore(), [], asyncio.Event()
    service = RunService(store)
    async def slow(*args, **kwargs):
        calls.append(1)
        yield "section_progress", {"sections": [{"section_id": "s", "title": "Saved", "question": "question", "draft": "keep"}]}
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()
    monkeypatch.setattr("multi_agent_research.runs.service.astream_research", slow)
    monkeypatch.setattr("multi_agent_research.runs.service.aresume_research", slow)
    await service.create_run(question="deadline research", run_id="r")
    await service.start_run("r")
    await service._tasks["r"]
    assert stopped.is_set()
    assert store.events[-1].payload["reason"] == "execution_timeout"
    assert store.runs["r"].status == RunStatus.PAUSED
    assert store.runs["r"].sections[0].draft == "keep"
    await service.start_run("r", resume=True)
    await service._tasks["r"]
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_deadline_during_projection_is_not_misclassified_as_shutdown(monkeypatch):
    monkeypatch.setattr(settings.agent, "run_timeout", 0.03)
    store = MemoryRunStore()
    service = RunService(store)
    async def events(*args, **kwargs):
        yield "start", {}
    async def slow(*args):
        await asyncio.Event().wait()
    monkeypatch.setattr("multi_agent_research.runs.service.astream_research", events)
    monkeypatch.setattr(store, "publish_execution_event", slow)
    await service.create_run(question="slow database operation", run_id="r")
    await service.start_run("r")
    await service._tasks["r"]
    assert store.runs["r"].status == RunStatus.PAUSED
    assert store.events[-1].payload["reason"] == "execution_timeout"


@pytest.mark.asyncio
async def test_search_budget_cancels_inflight_siblings(monkeypatch):
    monkeypatch.setattr(settings.agent, "run_max_retrieval_calls", 1)
    entered, stopped = asyncio.Event(), asyncio.Event()
    async def slow():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()
    async def blocked():
        await entered.wait()
        return await invoke_retrieval(slow, label="second")
    async with scope_for() as (store, _, record):
        with pytest.raises(BudgetExceeded):
            await gather_cancel_on_error(invoke_retrieval(slow, label="first"), blocked())
        assert stopped.is_set()
        assert store.runs[record.run_id].budget["retrieval_calls"] == 1


@pytest.mark.asyncio
async def test_search_slots_bound_concurrency_across_runs():
    slots = asyncio.Semaphore(1)
    current, maximum = 0, 0
    async def search():
        nonlocal current, maximum
        current += 1
        maximum = max(maximum, current)
        await asyncio.sleep(0.01)
        current -= 1
        return []
    async def run(name):
        async with scope_for(run_id=name, semaphore=slots):
            await gather_cancel_on_error(*(invoke_retrieval(search, label="test") for _ in range(3)))
    await asyncio.gather(run("a"), run("b"))
    assert maximum == 1


def test_expired_execution_time_is_reset_but_account_is_not():
    value = start_budget(new_budget())
    value["deadline"] = time.time() - 10
    resumed = start_budget(value)
    assert resumed["deadline"] > time.time()
    assert reserve(resumed, "x", "retrieval", 0, "search")["retrieval_calls"] == 1


@pytest.mark.asyncio
async def test_retained_tool_interface_cannot_swallow_budget_stop(monkeypatch):
    from multi_agent_research.tools.knowledge import query_internal_knowledge
    monkeypatch.setattr(settings.agent, "run_max_retrieval_calls", 1)
    async with scope_for() as (store, _, record):
        await store.reserve_budget(record.run_id, record.execution_id, "used", "retrieval", 0, "test")
        with pytest.raises(BudgetExceeded):
            await query_internal_knowledge.ainvoke({"query": "offline test"})
        assert store.runs[record.run_id].budget["retrieval_calls"] == 1


@pytest.mark.asyncio
async def test_web_network_retries_each_need_quota_and_no_signature_fallback(monkeypatch):
    from multi_agent_research.agents import search_agent
    requests = []
    class Tool:
        def __init__(self, **kwargs):
            pass
        async def ainvoke(self, payload):
            requests.append(payload)
            return {"error": ConnectionError("test network fault")}
    monkeypatch.setitem(sys.modules, "langchain_tavily", SimpleNamespace(TavilySearch=Tool))
    monkeypatch.setattr(settings.tool_secrets, "tavily_api_key", SecretStr("test-only"))
    monkeypatch.setattr(settings.agent, "run_max_retrieval_calls", 2)
    monkeypatch.setattr(search_agent, "_WEB_RETRY_DELAY", 0)
    async with scope_for() as (store, _, record):
        with pytest.raises(BudgetExceeded):
            await search_agent._web_search("test", 0)
        assert requests == [{"query": "test"}, {"query": "test"}]
        assert store.runs[record.run_id].budget["retrieval_calls"] == 2


@pytest.mark.asyncio
async def test_knowledge_transport_timeout_is_not_swallowed(monkeypatch):
    from multi_agent_research.core.retrieval import RetrievalDeferred
    from httpx import ReadTimeout
    from multi_agent_research.agents import search_agent
    from multi_agent_research.knowledge.client import KnowledgeServiceUnavailable
    class Client:
        async def search(self, *args, **kwargs):
            raise KnowledgeServiceUnavailable("timeout") from ReadTimeout("test")
    monkeypatch.setattr(search_agent, "get_knowledge_service_client", Client)
    monkeypatch.setattr(settings.tool_secrets, "tavily_api_key", SecretStr(""))
    async with scope_for() as (store, _, record):
        with pytest.raises(RetrievalDeferred):
            await search_agent.search_agent_node({"research_question": "test"})
        assert store.runs[record.run_id].budget["retrieval_calls"] == 2


@pytest.mark.asyncio
async def test_retrieval_timeout_cancels_request(monkeypatch):
    monkeypatch.setattr(settings.agent, "retrieval_call_timeout", 0.01)
    stopped = asyncio.Event()
    async def slow():
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()
    async with scope_for() as (store, _, record):
        with pytest.raises(CallTimeout):
            await invoke_retrieval(slow, label="timeout-test")
        assert stopped.is_set()
        assert store.runs[record.run_id].budget["retrieval_calls"] == 1


@pytest.mark.asyncio
async def test_budget_stop_checkpoint_restart_keeps_first_chapter(monkeypatch, tmp_path):
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
    monkeypatch.setattr(workflow, "search_agent_node", fake.search)
    monkeypatch.setattr(settings.agent, "run_max_model_calls", 6)
    store = MemoryRunStore()
    service = RunService(store)
    await service.create_run(question="分析公司X竞争优势", run_id="quota-e2e")
    database = str(tmp_path / "quota.sqlite")
    async def get_app():
        return app
    monkeypatch.setattr(streaming, "_get_app", get_app)
    for resume in (False, True):
        async with AsyncSqliteSaver.from_conn_string(database) as saver:
            app = build_graph(saver)
            if resume:
                from multi_agent_research.runs.repository import RunConflictError
                with pytest.raises(RunConflictError, match="预算"):
                    await service.start_run("quota-e2e", resume=True)
            else:
                await service.start_run("quota-e2e")
                await asyncio.wait_for(service._tasks["quota-e2e"], 10)
            record = store.runs["quota-e2e"]
            assert record.status == RunStatus.BUDGET_LIMITED
            assert record.sections[0].status == "complete"
            assert record.sections[0].draft
            assert record.budget["model_calls"] == 6
            assert record.budget["known_tokens"] == 60
            assert len(store.diagnostics) == 6
            assert store.events[-1].payload["reason"] == "budget_exhausted"
    assert fake.calls["write:成本"] == 1
    assert fake.calls["write:渠道"] == 0
    await service.shutdown()
