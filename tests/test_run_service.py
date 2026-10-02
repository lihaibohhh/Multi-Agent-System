from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Sequence

import pytest

from multi_agent_research.runs.models import (
    ParentContextSnapshot,
    RunEventRecord,
    RunRecord,
    RunStatus,
    SessionRecord,
)
from multi_agent_research.runs.repository import RunConflictError
from multi_agent_research.runs.service import RunService


class MemoryRunStore:
    def __init__(self) -> None:
        self.runs: dict[str, RunRecord] = {}
        self.sessions: dict[str, SessionRecord] = {}
        self.events: list[RunEventRecord] = []

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
    ) -> RunRecord:
        now = datetime.now(timezone.utc)
        record = RunRecord(
            run_id=run_id,
            session_id=session_id,
            parent_run_id=parent_run_id,
            thread_id=run_id,
            question=question,
            status=RunStatus.CREATED,
            parent_context=parent_context,
            created_at=now,
            updated_at=now,
        )
        self.runs[run_id] = record
        return record

    async def get_run(self, run_id: str) -> RunRecord | None:
        return self.runs.get(run_id)

    async def save_sections(self, run_id: str, sections: list[dict]) -> None:
        data = self.runs[run_id].model_dump()
        data["sections"] = sections
        self.runs[run_id] = RunRecord.model_validate(data)

    async def claim_run(
        self,
        run_id: str,
        expected: Sequence[RunStatus],
    ) -> RunRecord:
        record = self.runs[run_id]
        if record.status not in expected:
            raise RunConflictError("invalid transition")
        updated = record.model_copy(
            update={"status": RunStatus.RUNNING, "started_at": datetime.now(timezone.utc)}
        )
        self.runs[run_id] = updated
        return updated

    async def complete_run(self, run_id: str, final_report: str) -> RunRecord:
        updated = self.runs[run_id].model_copy(
            update={"status": RunStatus.COMPLETED, "final_report": final_report}
        )
        self.runs[run_id] = updated
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
            payload=payload,
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
    original_complete = store.complete_run

    async def delayed_complete(run_id: str, final_report: str) -> RunRecord:
        record = await original_complete(run_id, final_report)
        completed_in_store.set()
        await release_complete.wait()
        return record

    async def stream(question: str, run_id: str, *, parent_context=None):
        yield "done", {"run_id": run_id, "report": "finished"}

    store.complete_run = delayed_complete  # type: ignore[method-assign]
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

    async def fake_resume(run_id: str):
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

    async def resume_stream(run_id):
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
