from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from multi_agent_research.agents import (
    AgentContext,
    AgentRunner,
    AgentSpec,
    AgentTurnResult,
    agent_checkpoint_loader,
    agent_event_sink,
    create_resumable_agent_context,
    dispatch_agent_event,
)
from multi_agent_research.agents.events import AgentEvent
from multi_agent_research.runs.models import AgentExecutionStatus, RunStatus
from multi_agent_research.runs.repository import RunConflictError, StaleExecutionError
from multi_agent_research.runs.service import RunService
from tests.test_run_service import MemoryRunStore, _wait_for_status


@dataclass(frozen=True)
class Request:
    question: str


@dataclass(frozen=True)
class Output:
    summary: str


class PersistedTwoTurnAgent:
    spec = AgentSpec(
        name="persisted_two_turn",
        description="exercise durable Agent lifecycle records",
        model_ref="section_model",
        input_type=Request,
        output_type=Output,
        max_turns=2,
    )

    async def run_turn(self, request, *, context, call_model):
        observed, _ = await call_model("system", f"private prompt {context.turn}")
        if context.turn == 1:
            return AgentTurnResult(
                status="continue",
                state_updates={"observation": observed},
                reason="需要汇总",
            )
        return AgentTurnResult(
            status="completed",
            output=Output("done"),
            state_updates={"finished": True},
            handoff={"summary": "可恢复交接"},
            unresolved=("仍需关注时效",),
        )


def _resolver(spec):
    async def call_model(system, prompt, schema=None, *, validator=None, context=None):
        return "observed", {"tokens": 5, "unknown": 0, "attempts": 1}

    return call_model


@pytest.mark.asyncio
async def test_run_service_persists_agent_events_and_latest_checkpoint(monkeypatch) -> None:
    store = MemoryRunStore()
    service = RunService(store, poll_interval=0.001)

    async def stream(question, run_id, **kwargs):
        result = await AgentRunner(
            _resolver,
            event_sink=dispatch_agent_event,
        ).run(
            PersistedTwoTurnAgent(),
            Request(question),
            context=AgentContext.create(run_id),
        )
        assert result.status == "completed"
        yield "done", {"run_id": run_id, "report": "persisted report"}

    monkeypatch.setattr("multi_agent_research.runs.service.astream_research", stream)
    await service.create_run(question="persist Agent runtime lifecycle", run_id="agent-persist")
    await service.start_run("agent-persist")
    await _wait_for_status(store, "agent-persist", RunStatus.COMPLETED)

    executions = await service.list_agent_executions("agent-persist")
    assert len(executions) == 1
    execution = executions[0]
    assert execution.status == AgentExecutionStatus.COMPLETED
    assert execution.turn == 2
    assert execution.usage == {"tokens": 10, "unknown": 0, "attempts": 2}
    assert execution.local_state == {"observation": "observed", "finished": True}
    assert execution.handoff == {"summary": "可恢复交接"}
    assert execution.unresolved == ["仍需关注时效"]

    events = await service.list_agent_events("agent-persist")
    assert [event.event_type for event in events] == [
        "agent_started",
        "agent_turn_started",
        "agent_model_called",
        "agent_retrying",
        "agent_turn_started",
        "agent_model_called",
        "agent_completed",
    ]
    serialized = json.dumps(
        [event.model_dump(mode="json") for event in events],
        ensure_ascii=False,
    )
    assert "private prompt" not in serialized
    public_trace = await service.list_agent_trace("agent-persist")
    public_serialized = json.dumps(
        [event.model_dump(mode="json") for event in public_trace],
        ensure_ascii=False,
    )
    assert "observation" not in public_serialized
    assert "可恢复交接" not in public_serialized
    assert "仍需关注时效" not in public_serialized
    assert public_trace[-1].details["checkpoint"] == {
        "usage": {"tokens": 10, "unknown": 0, "attempts": 2},
        "has_local_state": True,
        "has_handoff": True,
        "unresolved_count": 1,
        "state_discarded": False,
    }
    assert agent_event_sink.get() is None
    assert agent_checkpoint_loader.get() is None


@pytest.mark.asyncio
async def test_run_service_resumes_agent_from_latest_checkpoint(monkeypatch) -> None:
    store = MemoryRunStore()
    service = RunService(store, poll_interval=0.001)
    interrupted = False

    class RecoverableAgent:
        spec = AgentSpec(
            name="recoverable",
            description="resume from a persisted local checkpoint",
            model_ref="section_model",
            input_type=Request,
            output_type=Output,
            max_turns=2,
        )

        async def run_turn(self, request, *, context, call_model):
            nonlocal interrupted
            if context.local_state.get("prepared"):
                if not interrupted:
                    interrupted = True
                    from multi_agent_research.core.budget import ExecutionPaused
                    raise ExecutionPaused("injected pause")
                return AgentTurnResult(
                    status="completed",
                    output=Output("resumed"),
                    state_updates={"finished": True},
                )
            if context.turn == 1:
                return AgentTurnResult(
                    status="continue",
                    state_updates={"prepared": True},
                    reason="checkpoint before interruption",
                )
            raise AssertionError("second turn should see prepared state")

    agent = RecoverableAgent()

    async def execute(run_id):
        context = await create_resumable_agent_context(run_id, agent.spec.name)
        result = await AgentRunner(
            _resolver,
            event_sink=dispatch_agent_event,
        ).run(agent, Request("recover"), context=context)
        assert result.output == Output("resumed")
        yield "done", {"run_id": run_id, "report": "recovered report"}

    async def start_stream(question, run_id, **kwargs):
        async for event in execute(run_id):
            yield event

    async def resume_stream(run_id, **kwargs):
        async for event in execute(run_id):
            yield event

    monkeypatch.setattr("multi_agent_research.runs.service.astream_research", start_stream)
    monkeypatch.setattr("multi_agent_research.runs.service.aresume_research", resume_stream)
    await service.create_run(question="recover Agent checkpoint", run_id="agent-resume")
    await service.start_run("agent-resume")
    await _wait_for_status(store, "agent-resume", RunStatus.PAUSED)
    await service.start_run("agent-resume", resume=True)
    await _wait_for_status(store, "agent-resume", RunStatus.COMPLETED)

    records = await service.list_agent_executions("agent-resume")
    assert len(records) == 2
    first, second = records
    assert first.status == AgentExecutionStatus.PAUSED
    assert first.local_state == {"prepared": True}
    assert second.status == AgentExecutionStatus.COMPLETED
    assert second.parent_agent_run_id == first.agent_run_id
    assert second.local_state == {"prepared": True, "finished": True}


@pytest.mark.asyncio
async def test_tool_trace_events_are_persisted_and_public_view_is_payload_free() -> None:
    store = MemoryRunStore()
    service = RunService(store)
    await service.create_run(question="safe tool trace", run_id="tool-trace")
    running = await store.begin_execution("tool-trace", (RunStatus.CREATED,), resume=False)
    agent_run_id = "tool-agent-run"
    events = (
        AgentEvent(
            event_type="agent_started",
            agent_name="evidence_research",
            agent_version="1",
            run_id="tool-trace",
            agent_run_id=agent_run_id,
            parent_agent_run_id=None,
            section_id="section_1",
            turn=0,
            details={"model_ref": "section_model"},
        ),
        AgentEvent(
            event_type="agent_tool_started",
            agent_name="evidence_research",
            agent_version="1",
            run_id="tool-trace",
            agent_run_id=agent_run_id,
            parent_agent_run_id=None,
            section_id="section_1",
            turn=1,
            details={"tool_name": "retrieve_evidence", "tool_call_id": "call-1"},
        ),
        AgentEvent(
            event_type="agent_tool_completed",
            agent_name="evidence_research",
            agent_version="1",
            run_id="tool-trace",
            agent_run_id=agent_run_id,
            parent_agent_run_id=None,
            section_id="section_1",
            turn=1,
            details={
                "tool_name": "retrieve_evidence",
                "tool_call_id": "call-1",
                "duration_ms": 7,
            },
        ),
    )
    for event in events:
        await store.record_agent_event("tool-trace", running.execution_id, event)

    trace = await service.list_agent_trace("tool-trace")

    assert [item.event_type for item in trace] == [
        "agent_started",
        "agent_tool_started",
        "agent_tool_completed",
    ]
    assert trace[-1].details == {
        "tool_name": "retrieve_evidence",
        "tool_call_id": "call-1",
        "duration_ms": 7,
    }
    assert trace[-1].agent_name == "evidence_research"
    assert trace[-1].section_id == "section_1"


@pytest.mark.asyncio
async def test_agent_event_writes_are_idempotent_fenced_and_reconciled() -> None:
    store = MemoryRunStore()
    service = RunService(store)
    await service.create_run(question="test Agent event fencing", run_id="agent-fence")
    running = await store.begin_execution(
        "agent-fence",
        (RunStatus.CREATED,),
        resume=False,
    )
    event = AgentEvent(
        event_id="agent-event-idempotent",
        event_type="agent_started",
        agent_name="planner",
        agent_version="2",
        run_id="agent-fence",
        agent_run_id="agent-run-fenced",
        parent_agent_run_id=None,
        section_id=None,
        turn=0,
        details={"model_ref": "section_model"},
    )

    first = await store.record_agent_event(
        "agent-fence",
        running.execution_id,
        event,
    )
    second = await store.record_agent_event(
        "agent-fence",
        running.execution_id,
        event,
    )

    assert first == second
    assert len(await store.list_agent_events("agent-fence")) == 1
    await store.record_agent_event("agent-fence", running.execution_id, AgentEvent(
        event_type="agent_retrying",
        agent_name="planner",
        agent_version="2",
        run_id="agent-fence",
        agent_run_id="agent-run-fenced",
        parent_agent_run_id=None,
        section_id=None,
        turn=1,
        details={"checkpoint": {
            "usage": {"tokens": 3, "unknown": 0, "attempts": 1},
            "local_state": {"temporary": True},
            "handoff": {"draft": "temporary"},
            "unresolved": ["temporary"],
        }},
    ))
    cleared = await store.record_agent_event(
        "agent-fence",
        running.execution_id,
        AgentEvent(
            event_type="agent_retrying",
            agent_name="planner",
            agent_version="2",
            run_id="agent-fence",
            agent_run_id="agent-run-fenced",
            parent_agent_run_id=None,
            section_id=None,
            turn=2,
            details={"checkpoint": {
                "usage": {"tokens": 3, "unknown": 0, "attempts": 1},
                "local_state": {},
                "handoff": None,
                "unresolved": [],
            }},
        ),
    )
    assert cleared.local_state == {}
    assert cleared.handoff is None
    assert cleared.unresolved == []
    assert await store.can_initialize_missing_checkpoint("agent-fence") is False
    with pytest.raises(StaleExecutionError):
        await store.record_agent_event("agent-fence", "stale-execution", AgentEvent(
            event_type="agent_turn_started",
            agent_name="planner",
            agent_version="2",
            run_id="agent-fence",
            agent_run_id="agent-run-fenced",
            parent_agent_run_id=None,
            section_id=None,
            turn=1,
        ))

    assert await store.reconcile_running({}) == ["agent-fence"]
    recovered = await store.get_agent_execution("agent-run-fenced")
    assert recovered is not None
    assert recovered.status == AgentExecutionStatus.INTERRUPTED
    assert recovered.error_message == "execution_orphaned"

    resumed = await store.begin_execution(
        "agent-fence",
        (RunStatus.INTERRUPTED,),
        resume=True,
    )
    with pytest.raises(RunConflictError, match="Agent run ID conflict"):
        await store.record_agent_event(
            "agent-fence",
            resumed.execution_id,
            AgentEvent(
                event_type="agent_started",
                agent_name="planner",
                agent_version="2",
                run_id="agent-fence",
                agent_run_id="agent-run-fenced",
                parent_agent_run_id=None,
                section_id=None,
                turn=0,
            ),
        )
    resumed_agent_run_id = "agent-run-resumed"
    for resumed_event in (
        AgentEvent(
            event_type="agent_started",
            agent_name="planner",
            agent_version="2",
            run_id="agent-fence",
            agent_run_id=resumed_agent_run_id,
            parent_agent_run_id="agent-run-fenced",
            section_id=None,
            turn=0,
            details={"model_ref": "section_model"},
        ),
        AgentEvent(
            event_type="agent_completed",
            agent_name="planner",
            agent_version="2",
            run_id="agent-fence",
            agent_run_id=resumed_agent_run_id,
            parent_agent_run_id="agent-run-fenced",
            section_id=None,
            turn=1,
            details={"checkpoint": {
                "usage": {"tokens": 5, "unknown": 0, "attempts": 1},
                "local_state": {},
                "handoff": {"plan": {"sections": []}},
                "unresolved": [],
            }},
        ),
    ):
        await store.record_agent_event(
            "agent-fence",
            resumed.execution_id,
            resumed_event,
        )

    records = await store.list_agent_executions("agent-fence")
    assert {record.status for record in records} == {
        AgentExecutionStatus.INTERRUPTED,
        AgentExecutionStatus.COMPLETED,
    }
    resumed_record = await store.get_agent_execution(resumed_agent_run_id)
    assert resumed_record.parent_agent_run_id == "agent-run-fenced"
    assert resumed_record.handoff == {"plan": {"sections": []}}
    with pytest.raises(RunConflictError, match="Terminal Agent event conflict"):
        await store.record_agent_event(
            "agent-fence",
            resumed.execution_id,
            AgentEvent(
                event_type="agent_turn_started",
                agent_name="planner",
                agent_version="2",
                run_id="agent-fence",
                agent_run_id=resumed_agent_run_id,
                parent_agent_run_id="agent-run-fenced",
                section_id=None,
                turn=2,
            ),
        )
