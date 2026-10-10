from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import datetime, timezone
from typing import Sequence
from uuid import uuid4

import pytest

from multi_agent_research.agents.events import AgentEvent
from multi_agent_research.runs.models import (
    AgentExecutionRecord,
    AgentExecutionStatus,
    AgentLifecycleEventRecord,
    ParentContextSnapshot,
    RunEventRecord,
    RunRecord,
    RunStatus,
    SessionRecord,
    RunSnapshot,
)
from multi_agent_research.runs.repository import RunConflictError, StaleExecutionError
from multi_agent_research.runs.service import RunService
from multi_agent_research.runs.usage import summarize_run_usage
from multi_agent_research.core.budget import (new_budget, start_budget, reserve, settle, budget_summary,
                                            migrate_budget, ExecutionPaused)


class MemoryRunStore:
    def __init__(self) -> None:
        self.runs: dict[str, RunRecord] = {}
        self.sessions: dict[str, SessionRecord] = {}
        self.events: list[RunEventRecord] = []
        self.agent_executions: dict[str, AgentExecutionRecord] = {}
        self.agent_events: list[AgentLifecycleEventRecord] = []
        self.diagnostics: list[dict] = []
        self.retrievals = {}
        self.owned = True

    async def assert_instance_owner(self):
        if not self.owned:
            from multi_agent_research.runs.runtime import InstanceUnavailableError
            raise InstanceUnavailableError("test owner lost")

    async def can_initialize_missing_checkpoint(self, run_id):
        record = self.runs[run_id]
        allowed = {"run_created", "run_started", "run_resumed", "start", "error", "run_interrupted", "section_snapshot",
                   "pause_requested", "run_paused", "budget_limited", "budget_migrated"}
        own_reservations = [v for v in record.budget.get("reservations", {}).values()
                            if not v.get("run_id") or v["run_id"] == run_id]
        return (not record.sections and not record.final_report and not record.report_review
                and not own_reservations and not record.model_usage.get("attempts")
                and not any(d.get("run_id") == run_id for d in self.diagnostics)
                and not any(e.run_id == run_id for e in self.agent_events)
                and not any(e.run_id == run_id and e.event_type not in allowed for e in self.events))

    @asynccontextmanager
    async def guard_execution(self, run_id, execution_id):
        record = self.runs[run_id]
        if record.status != RunStatus.RUNNING or record.execution_id != execution_id:
            raise StaleExecutionError("stale execution")
        yield None

    async def begin_execution(self, run_id, expected, *, resume):
        before = deepcopy(self.runs[run_id])
        events = len(self.events)
        try:
            record = await self.claim_run(run_id, expected)
            await self.append_event(run_id, "run_resumed" if resume else "run_started",
                                    {"run_id": run_id, "execution_id": record.execution_id})
            return record
        except BaseException:
            self.runs[run_id] = before
            del self.events[events:]
            raise

    async def reconcile_running(self, active):
        await self.assert_instance_owner()
        interrupted = []
        for run_id, record in list(self.runs.items()):
            if record.status == RunStatus.RUNNING and active.get(run_id) != record.execution_id:
                await self.finish_execution(run_id, record.execution_id, RunStatus.INTERRUPTED,
                                            {"reason": "execution_orphaned"})
                interrupted.append(run_id)
        return interrupted

    async def publish_execution_event(self, run_id, execution_id, event_type, payload):
        async with self.guard_execution(run_id, execution_id):
            before, events = deepcopy(self.runs[run_id]), len(self.events)
            try:
                if "sections" in payload:
                    await self.save_sections(run_id, payload["sections"])
                if "report_review" in payload:
                    await self.save_report_review(run_id, payload["report_review"])
                await self.append_event(run_id, event_type, {**payload, "budget": budget_summary(self.runs[run_id].budget)})
            except BaseException:
                self.runs[run_id] = before
                del self.events[events:]
                raise

    async def create_session(self, *, session_id: str, title: str) -> SessionRecord:
        now = datetime.now(timezone.utc)
        record = SessionRecord(
            session_id=session_id,
            title=title,
            created_at=now,
            updated_at=now,
        )
        self.sessions[session_id] = record
        return record

    async def get_session(self, session_id: str) -> SessionRecord | None:
        return self.sessions.get(session_id)

    async def list_session_runs(self, session_id: str) -> list[RunRecord]:
        return [run for run in self.runs.values() if run.session_id == session_id]

    async def create_run(
        self,
        *,
        run_id: str,
        session_id: str | None,
        parent_run_id: str | None,
        parent_context: ParentContextSnapshot | None,
        question: str,
        parent_snapshot_cursor: int | None = None,
    ) -> RunRecord:
        if parent_snapshot_cursor is not None:
            current = await self.get_snapshot(parent_run_id)
            if current.run.status == RunStatus.RUNNING or current.cursor != parent_snapshot_cursor:
                raise RunConflictError("父研究已改变")
            operation = parent_context.section_operation if parent_context else None
            if operation and operation["mode"] == "continue":
                for old in self.runs.values():
                    prior = old.parent_context.section_operation if old.parent_context else None
                    if old.parent_run_id == parent_run_id and prior and prior.get("mode") == "continue" and prior.get("target") == operation.get("target") and prior.get("source_cursor") == parent_snapshot_cursor:
                        raise RunConflictError("已有选章继续任务，请打开该任务继续")
        now = datetime.now(timezone.utc)
        record = RunRecord(
            run_id=run_id,
            session_id=session_id,
            parent_run_id=parent_run_id,
            thread_id=run_id,
            question=question,
            status=RunStatus.CREATED,
            budget=new_budget(),
            budget_id="budget_" + run_id,
            parent_context=parent_context,
            created_at=now,
            updated_at=now,
        )
        self.runs[run_id] = record
        return record

    async def get_run(self, run_id: str) -> RunRecord | None:
        return self.runs.get(run_id)

    async def get_run_usage(self, run_id):
        record = self.runs[run_id]
        return summarize_run_usage(
            run_id=run_id,
            budget_id=record.budget_id,
            model_usage=record.model_usage,
            budget=record.budget,
        )

    async def get_snapshot(self, run_id: str) -> RunSnapshot:
        return RunSnapshot(run=self.runs[run_id].model_copy(deep=True),
                           cursor=max((e.sequence for e in self.events if e.run_id == run_id), default=0))

    async def record_model_attempt(self, run_id, execution_id, data):
        async with self.guard_execution(run_id, execution_id):
            pass
        if any(d["diagnostic_id"] == data["diagnostic_id"] for d in self.diagnostics):
            return
        self.diagnostics.append({**data, "run_id": run_id, "execution_id": execution_id})
        usage = self.runs[run_id].model_usage
        updated = {"attempts": usage.get("attempts", 0) + 1,
                   "tokens": usage.get("tokens", 0) + data["tokens"],
                   "unknown": usage.get("unknown", 0) + data["unknown"]}
        self.runs[run_id] = self.runs[run_id].model_copy(update={"model_usage": updated})

    async def record_agent_event(self, run_id, execution_id, event: AgentEvent):
        if event.run_id != run_id:
            raise ValueError("Agent event run_id mismatch")
        async with self.guard_execution(run_id, execution_id):
            duplicate = next(
                (item for item in self.agent_events if item.event_id == event.event_id),
                None,
            )
            if duplicate is not None:
                if duplicate.agent_run_id != event.agent_run_id:
                    raise RunConflictError("Agent event ID conflict")
                return self.agent_executions[event.agent_run_id]
            status = AgentExecutionStatus({
                "agent_started": "running",
                "agent_turn_started": "running",
                "agent_model_called": "running",
                "agent_tool_started": "running",
                "agent_tool_completed": "running",
                "agent_tool_failed": "running",
                "agent_retrying": "running",
                "agent_paused": "paused",
                "agent_completed": "completed",
                "agent_failed": "failed",
            }[event.event_type])
            current = self.agent_executions.get(event.agent_run_id)
            if current and (
                current.run_id != run_id or current.execution_id != execution_id
            ):
                raise RunConflictError("Agent run ID conflict")
            if current and current.status != AgentExecutionStatus.RUNNING:
                raise RunConflictError("Terminal Agent event conflict")
            details = event.details
            checkpoint = details.get("checkpoint") or {}
            has_checkpoint = isinstance(details.get("checkpoint"), dict)
            occurred_at = datetime.fromisoformat(event.occurred_at)
            record = AgentExecutionRecord(
                agent_run_id=event.agent_run_id,
                run_id=run_id,
                execution_id=execution_id,
                parent_agent_run_id=event.parent_agent_run_id,
                agent_name=event.agent_name,
                agent_version=event.agent_version,
                section_id=event.section_id,
                status=status,
                turn=max(event.turn, current.turn if current else 0),
                model_ref=details.get("model_ref") or (current.model_ref if current else None),
                usage=(checkpoint.get("usage", {}) if has_checkpoint
                       else (current.usage if current else {})),
                local_state=(checkpoint.get("local_state", {}) if has_checkpoint
                             else (current.local_state if current else {})),
                handoff=(checkpoint.get("handoff") if has_checkpoint
                         else (current.handoff if current else None)),
                unresolved=(checkpoint.get("unresolved", []) if has_checkpoint
                            else (current.unresolved if current else [])),
                error_type=details.get("error_type") or (current.error_type if current else None),
                error_message=(details.get("error_message") or details.get("reason")
                               or (current.error_message if current else None)),
                started_at=current.started_at if current else occurred_at,
                updated_at=occurred_at,
                completed_at=(current.completed_at if current else None)
                or (occurred_at if status != AgentExecutionStatus.RUNNING else None),
            )
            self.agent_executions[event.agent_run_id] = record
            self.agent_events.append(AgentLifecycleEventRecord(
                sequence=len(self.agent_events) + 1,
                event_id=event.event_id,
                agent_run_id=event.agent_run_id,
                run_id=run_id,
                execution_id=execution_id,
                event_type=event.event_type,
                turn=event.turn,
                details=details,
                occurred_at=occurred_at,
            ))
            return record

    async def get_agent_execution(self, agent_run_id):
        return self.agent_executions.get(agent_run_id)

    async def get_latest_agent_execution(self, run_id, agent_name, section_id):
        matches = [
            record for record in self.agent_executions.values()
            if record.run_id == run_id
            and record.agent_name == agent_name
            and record.section_id == section_id
        ]
        return max(matches, key=lambda item: (item.updated_at, item.agent_run_id), default=None)

    async def list_agent_executions(self, run_id):
        return [record for record in self.agent_executions.values()
                if record.run_id == run_id]

    async def list_agent_events(self, run_id, after=0):
        return [event for event in self.agent_events
                if event.run_id == run_id and event.sequence > after]

    async def reserve_budget(self, run_id, execution_id, reservation_id, kind, tokens, label):
        async with self.guard_execution(run_id, execution_id):
            record = self.runs[run_id]
            if record.pause_requested:
                raise ExecutionPaused("pause requested")
            budget = reserve(record.budget, reservation_id, kind, tokens, label)
            budget["reservations"][reservation_id].update(run_id=run_id, execution_id=execution_id)
            self._save_account(record, budget)

    async def settle_budget(self, run_id, execution_id, reservation_id, tokens):
        async with self.guard_execution(run_id, execution_id):
            record = self.runs[run_id]
            budget = settle(record.budget, reservation_id, tokens)
            self._save_account(record, budget)

    def _save_account(self, record, budget):
        for key, run in self.runs.items():
            if run.budget_id == record.budget_id:
                self.runs[key] = run.model_copy(update={"budget": {**deepcopy(budget), "deadline": run.budget.get("deadline")}})

    async def begin_retrieval(self, run_id, execution_id, operation_id, descriptor, reservation_id, reuse_parent=None):
        from multi_agent_research.core.retrieval import admit, progress
        async with self.guard_execution(run_id, execution_id):
            if self.runs[run_id].pause_requested:
                raise ExecutionPaused("pause requested")
            old = self.retrievals.get((run_id, operation_id))
            if not old and reuse_parent:
                candidate = self.retrievals.get((reuse_parent, operation_id))
                if candidate:
                    old = deepcopy(candidate)
                    self.retrievals[run_id, operation_id] = old
            data, reused = admit(old, descriptor, execution_id, reservation_id)
            if not reused:
                await self.reserve_budget(run_id, execution_id, reservation_id, "retrieval", 0, descriptor["provider"])
                self.retrievals[run_id, operation_id] = data
            await self.append_event(run_id, "retrieval_progress", progress(data, operation_id, reused))
            return deepcopy(data), reused

    async def finish_retrieval(self, run_id, execution_id, operation_id, reservation_id, result, error):
        from multi_agent_research.core.retrieval import complete, progress
        async with self.guard_execution(run_id, execution_id):
            data = complete(self.retrievals[run_id, operation_id], execution_id, reservation_id, result, error)
            if error is None:
                await self.settle_budget(run_id, execution_id, reservation_id, 0)
            self.retrievals[run_id, operation_id] = data
            await self.append_event(run_id, "retrieval_progress", progress(data, operation_id))

    async def request_pause(self, run_id):
        run = self.runs[run_id]
        if run.status != RunStatus.RUNNING:
            raise RunConflictError("not running")
        run.pause_requested = True
        await self.append_event(run_id, "pause_requested", {})
        return run

    async def migrate_run_budget(self, run_id, reason):
        run = self.runs[run_id]
        if run.status == RunStatus.RUNNING:
            raise RunConflictError("running")
        if not run.budget_id:
            old = deepcopy(run.budget)
            run.budget_id = "budget_" + run_id
            run.budget = migrate_budget(old, run.model_usage)
            await self.append_event(run_id, "budget_migrated", {"previous_budget": old, "reason": reason})
        return run

    async def finish_execution(self, run_id, execution_id, status, payload):
        current = self.runs[run_id]
        if current.status != RunStatus.RUNNING or current.execution_id != execution_id:
            return current
        if "sections" in payload:
            await self.save_sections(run_id, payload["sections"])
        if "report_review" in payload:
            await self.save_report_review(run_id, payload["report_review"])
        current = self.runs[run_id]
        update = {"status": status, "error_message": payload.get("message") or payload.get("reason")}
        if status == RunStatus.COMPLETED:
            update["final_report"] = payload["report"]
        self.runs[run_id] = current.model_copy(update=update)
        agent_status = {
            RunStatus.COMPLETED: AgentExecutionStatus.INTERRUPTED,
            RunStatus.FAILED: AgentExecutionStatus.FAILED,
            RunStatus.INTERRUPTED: AgentExecutionStatus.INTERRUPTED,
            RunStatus.PAUSED: AgentExecutionStatus.PAUSED,
            RunStatus.BUDGET_LIMITED: AgentExecutionStatus.PAUSED,
        }[status]
        now = datetime.now(timezone.utc)
        for agent_run_id, agent in list(self.agent_executions.items()):
            if (agent.run_id == run_id and agent.execution_id == execution_id
                    and agent.status == AgentExecutionStatus.RUNNING):
                self.agent_executions[agent_run_id] = agent.model_copy(update={
                    "status": agent_status,
                    "error_type": payload.get("type") or "RunExecutionEnded",
                    "error_message": payload.get("message") or payload.get("reason"),
                    "updated_at": now,
                    "completed_at": agent.completed_at or now,
                })
        kinds = {RunStatus.COMPLETED: "done", RunStatus.FAILED: "error", RunStatus.INTERRUPTED: "run_interrupted",
                 RunStatus.PAUSED: "run_paused", RunStatus.BUDGET_LIMITED: "budget_limited"}
        await self.append_event(run_id, kinds[status], {
            **payload, "execution_id": execution_id, "model_usage": current.model_usage, "budget": current.budget,
        })
        return self.runs[run_id]

    async def save_sections(self, run_id: str, sections: list[dict]) -> None:
        data = self.runs[run_id].model_dump()
        data["sections"] = sections
        self.runs[run_id] = RunRecord.model_validate(data)

    async def save_report_review(self, run_id: str, review: dict | None) -> None:
        data = self.runs[run_id].model_dump()
        data["report_review"] = review
        self.runs[run_id] = RunRecord.model_validate(data)

    async def claim_run(
        self,
        run_id: str,
        expected: Sequence[RunStatus],
    ) -> RunRecord:
        record = self.runs[run_id]
        if record.status not in expected:
            raise RunConflictError("invalid transition")
        if not record.budget_id:
            raise RunConflictError("legacy migration required")
        updated = record.model_copy(
            update={"status": RunStatus.RUNNING, "started_at": datetime.now(timezone.utc),
                    "budget": start_budget(record.budget, record.model_usage),
                    "execution_id": uuid4().hex, "error_message": None, "pause_requested": False}
        )
        self.runs[run_id] = updated
        return updated

    async def complete_run(self, run_id: str, final_report: str) -> RunRecord:
        updated = self.runs[run_id].model_copy(
            update={"status": RunStatus.COMPLETED, "final_report": final_report}
        )
        self.runs[run_id] = updated
        return updated

    async def replace_final_report(
        self,
        run_id: str,
        final_report: str,
        *,
        previous_sha256: str,
        new_sha256: str,
        report_quality: str,
    ) -> RunRecord:
        current = self.runs[run_id]
        if current.status != RunStatus.COMPLETED:
            raise RunConflictError("only a completed run can be reassembled")
        if current.final_report == final_report:
            return current
        updated = current.model_copy(update={"final_report": final_report})
        self.runs[run_id] = updated
        await self.append_event(run_id, "report_reassembled", {
            "previous_sha256": previous_sha256,
            "new_sha256": new_sha256,
            "report_quality": report_quality,
            "model_calls": 0,
            "retrieval_calls": 0,
        })
        return updated

    async def fail_run(self, run_id: str, error_message: str) -> RunRecord:
        updated = self.runs[run_id].model_copy(
            update={"status": RunStatus.FAILED, "error_message": error_message}
        )
        self.runs[run_id] = updated
        return updated

    async def interrupt_run(self, run_id: str, reason: str) -> RunRecord:
        if self.runs[run_id].status != RunStatus.RUNNING:
            return self.runs[run_id]
        updated = self.runs[run_id].model_copy(
            update={"status": RunStatus.INTERRUPTED, "error_message": reason}
        )
        self.runs[run_id] = updated
        return updated

    async def mark_stale_running_interrupted(self) -> list[str]:
        stale = [key for key, value in self.runs.items() if value.status == RunStatus.RUNNING]
        for run_id in stale:
            await self.interrupt_run(run_id, "restart")
        return stale

    async def append_event(
        self,
        run_id: str,
        event_type: str,
        payload: dict,
    ) -> RunEventRecord:
        event = RunEventRecord(
            sequence=len(self.events) + 1,
            run_id=run_id,
            event_type=event_type,
            payload={"execution_id": self.runs[run_id].execution_id, **payload},
            created_at=datetime.now(timezone.utc),
        )
        self.events.append(event)
        return event

    async def list_events(self, run_id: str, after: int = 0) -> list[RunEventRecord]:
        return [
            event
            for event in self.events
            if event.run_id == run_id and event.sequence > after
        ]


async def _wait_for_status(
    store: MemoryRunStore,
    run_id: str,
    expected: RunStatus,
) -> RunRecord:
    for _ in range(100):
        record = store.runs[run_id]
        if record.status == expected:
            return record
        await asyncio.sleep(0.01)
    raise AssertionError(f"run did not reach {expected.value}")


@pytest.mark.asyncio
async def test_create_and_execute_are_separate(monkeypatch: pytest.MonkeyPatch) -> None:
    store = MemoryRunStore()
    service = RunService(store)
    executed = False

    async def fake_stream(question: str, run_id: str, *, parent_context=None):
        nonlocal executed
        executed = True
        yield "done", {"run_id": run_id, "report": f"report: {question}"}

    monkeypatch.setattr("multi_agent_research.runs.service.astream_research", fake_stream)

    created = await service.create_run(question="a valid question", run_id="run-separate")
    assert created.status == RunStatus.CREATED
    assert not executed

    await service.start_run(created.run_id)
    completed = await _wait_for_status(store, created.run_id, RunStatus.COMPLETED)
    assert executed
    assert completed.final_report == "report: a valid question"


@pytest.mark.asyncio
async def test_runs_are_grouped_in_an_automatic_session() -> None:
    store = MemoryRunStore()
    service = RunService(store)

    run = await service.create_run(question="automatic session", run_id="run-auto-session")
    timeline = await service.get_session_timeline(run.session_id or "")

    assert run.session_id is not None
    assert timeline.session.session_id == run.session_id
    assert [item.run_id for item in timeline.runs] == [run.run_id]


@pytest.mark.asyncio
async def test_child_run_gets_immutable_parent_report_snapshot() -> None:
    store = MemoryRunStore()
    service = RunService(store)
    session = await service.create_session(title="Follow-up research", session_id="session-1")
    parent = await service.create_run(
        question="parent question",
        session_id=session.session_id,
        run_id="run-parent",
    )
    await store.complete_run(
        parent.run_id,
        "parent report evidence\n\n## 参考来源\n- [source](https://example.com)",
    )

    child = await service.create_run(
        question="child follow-up",
        session_id=session.session_id,
        parent_run_id=parent.run_id,
        run_id="run-child",
    )

    assert child.parent_context is not None
    assert child.parent_context.source_run_id == parent.run_id
    assert child.parent_context.report_excerpt.startswith("parent report evidence")
    assert "https://example.com" in child.parent_context.reference_excerpt
    assert child.thread_id == child.run_id
    assert child.thread_id != parent.thread_id


@pytest.mark.asyncio
async def test_parent_run_must_be_completed_and_in_same_session() -> None:
    store = MemoryRunStore()
    service = RunService(store)
    await service.create_session(title="One", session_id="session-one")
    await service.create_session(title="Two", session_id="session-two")
    parent = await service.create_run(
        question="unfinished parent",
        session_id="session-one",
        run_id="run-unfinished",
    )

    with pytest.raises(RunConflictError, match="must be completed"):
        await service.create_run(
            question="child question",
            session_id="session-one",
            parent_run_id=parent.run_id,
        )

    await store.complete_run(parent.run_id, "done")
    with pytest.raises(RunConflictError, match="same session"):
        await service.create_run(
            question="child question",
            session_id="session-two",
            parent_run_id=parent.run_id,
        )


@pytest.mark.asyncio
async def test_closing_event_subscription_does_not_cancel_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MemoryRunStore()
    service = RunService(store, poll_interval=0.001)
    release = asyncio.Event()

    async def slow_stream(question: str, run_id: str, *, parent_context=None):
        yield "start", {"run_id": run_id, "question": question}
        await release.wait()
        yield "done", {"run_id": run_id, "report": "finished"}

    monkeypatch.setattr("multi_agent_research.runs.service.astream_research", slow_stream)

    await service.create_run(question="disconnect test", run_id="run-disconnect")
    await service.start_run("run-disconnect")
    subscription = service.iter_events("run-disconnect")
    await anext(subscription)
    await subscription.aclose()

    release.set()
    completed = await _wait_for_status(store, "run-disconnect", RunStatus.COMPLETED)
    assert completed.final_report == "finished"


@pytest.mark.asyncio
async def test_shutdown_marks_active_run_interrupted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MemoryRunStore()
    service = RunService(store)
    entered = asyncio.Event()

    async def never_finishes(question: str, run_id: str, *, parent_context=None):
        entered.set()
        await asyncio.Event().wait()
        yield "done", {"run_id": run_id, "report": "unreachable"}

    monkeypatch.setattr("multi_agent_research.runs.service.astream_research", never_finishes)

    await service.create_run(question="shutdown test", run_id="run-shutdown")
    await service.start_run("run-shutdown")
    await entered.wait()
    await service.shutdown()

    assert store.runs["run-shutdown"].status == RunStatus.INTERRUPTED
    assert store.events[-1].event_type == "run_interrupted"


@pytest.mark.asyncio
async def test_shutdown_cannot_overwrite_completed_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MemoryRunStore()
    service = RunService(store)
    completed_in_store = asyncio.Event()
    release_complete = asyncio.Event()
    original_complete = store.finish_execution

    async def delayed_complete(run_id, execution_id, status, payload) -> RunRecord:
        record = await original_complete(run_id, execution_id, status, payload)
        if status == RunStatus.COMPLETED:
            completed_in_store.set()
            await release_complete.wait()
        return record

    async def stream(question: str, run_id: str, *, parent_context=None):
        yield "done", {"run_id": run_id, "report": "finished"}

    store.finish_execution = delayed_complete  # type: ignore[method-assign]
    monkeypatch.setattr("multi_agent_research.runs.service.astream_research", stream)

    await service.create_run(question="completion race", run_id="run-completion-race")
    await service.start_run("run-completion-race")
    await completed_in_store.wait()
    await service.shutdown()

    assert store.runs["run-completion-race"].status == RunStatus.COMPLETED
    assert store.events[-1].event_type == "done"


@pytest.mark.asyncio
async def test_resume_uses_checkpoint_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    store = MemoryRunStore()
    service = RunService(store)
    resumed = False

    async def fake_resume(run_id: str, **kwargs):
        nonlocal resumed
        resumed = True
        yield "done", {"run_id": run_id, "report": "resumed report"}

    monkeypatch.setattr("multi_agent_research.runs.service.aresume_research", fake_resume)

    await service.create_run(question="resume test", run_id="run-resume")
    await store.claim_run("run-resume", (RunStatus.CREATED,))
    await store.interrupt_run("run-resume", "restart")
    await service.start_run("run-resume", resume=True)
    completed = await _wait_for_status(store, "run-resume", RunStatus.COMPLETED)

    assert resumed
    assert completed.final_report == "resumed report"


@pytest.mark.asyncio
async def test_chapter_drafts_survive_failure_and_resume_refreshes_projection(monkeypatch):
    store = MemoryRunStore()
    service = RunService(store)
    chapter = {
        "section_id": "section_1", "title": "成本", "question": "公司的成本优势如何",
        "status": "drafted", "draft": "已经写好的章节", "revision": 1,
    }

    async def failing_stream(question, run_id, *, parent_context=None):
        yield "section_progress", {"run_id": run_id, "sections": [chapter]}
        raise RuntimeError("review unavailable")

    monkeypatch.setattr("multi_agent_research.runs.service.astream_research", failing_stream)
    await service.create_run(question="研究公司竞争优势", run_id="chapter-recovery")
    await service.start_run("chapter-recovery")
    failed = await _wait_for_status(store, "chapter-recovery", RunStatus.FAILED)
    assert failed.sections[0].draft == chapter["draft"]
    assert failed.sections[0].status == "drafted"
    assert any(event.event_type == "section_progress" for event in store.events)

    # Model a checkpoint ahead of the SQL projection after a crash between writes.
    restored = {**chapter, "draft": "checkpoint 中更新的章节", "status": "complete", "revision": 2}

    async def resume_stream(run_id, **kwargs):
        yield "section_snapshot", {"run_id": run_id, "sections": [restored]}
        yield "done", {"run_id": run_id, "report": "组装的报告", "sections": [restored]}

    monkeypatch.setattr("multi_agent_research.runs.service.aresume_research", resume_stream)
    await service.start_run("chapter-recovery", resume=True)
    completed = await _wait_for_status(store, "chapter-recovery", RunStatus.COMPLETED)
    assert completed.sections[0].draft == restored["draft"]
    assert completed.sections[0].revision == 2
    assert completed.final_report == "组装的报告"


@pytest.mark.asyncio
async def test_chapter_storage_failure_does_not_mark_run_completed(monkeypatch):
    store = MemoryRunStore()
    service = RunService(store)

    async def broken_save(run_id, sections):
        raise RuntimeError("chapter storage unavailable")

    async def stream(question, run_id, *, parent_context=None):
        yield "done", {"run_id": run_id, "report": "report", "sections": []}

    monkeypatch.setattr(store, "save_sections", broken_save)
    monkeypatch.setattr("multi_agent_research.runs.service.astream_research", stream)
    await service.create_run(question="研究公司竞争优势", run_id="chapter-storage-failure")
    await service.start_run("chapter-storage-failure")
    failed = await _wait_for_status(store, "chapter-storage-failure", RunStatus.FAILED)
    assert failed.final_report is None
    assert not any(event.event_type == "done" for event in store.events)
