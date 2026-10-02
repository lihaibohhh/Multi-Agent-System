"""Run lifecycle orchestration independent from SSE client connections."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from datetime import datetime, timezone

from ..core.run_context import normalize_run_id, normalize_session_id
from ..core.streaming import aresume_research, astream_research
from .models import (
    ParentContextSnapshot,
    RunEventRecord,
    RunRecord,
    RunStatus,
    SessionRecord,
    SessionTimeline,
    TERMINAL_RUN_STATUSES,
)
from .repository import RunConflictError, RunNotFoundError, RunStore


logger = logging.getLogger(__name__)
_PARENT_REPORT_LIMIT = 12_000
_PARENT_REFERENCES_LIMIT = 4_000


def _extract_reference_excerpt(report: str) -> str:
    markers = ("## 参考来源", "## 参考资料", "## References")
    positions = [report.rfind(marker) for marker in markers]
    start = max(positions)
    if start < 0:
        return ""
    return report[start:start + _PARENT_REFERENCES_LIMIT]


class RunService:
    def __init__(self, repository: RunStore, *, poll_interval: float = 0.25) -> None:
        self._repository = repository
        self._poll_interval = poll_interval
        self._tasks: dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()

    async def recover_stale_runs(self) -> list[str]:
        run_ids = await self._repository.mark_stale_running_interrupted()
        for run_id in run_ids:
            await self._repository.append_event(
                run_id,
                "run_interrupted",
                {"run_id": run_id, "reason": "service_restart"},
            )
        return run_ids

    async def create_session(
        self,
        *,
        title: str,
        session_id: str | None = None,
    ) -> SessionRecord:
        resolved_session_id = normalize_session_id(session_id)
        return await self._repository.create_session(
            session_id=resolved_session_id,
            title=title.strip(),
        )

    async def get_session(self, session_id: str) -> SessionRecord:
        resolved_session_id = normalize_session_id(session_id)
        record = await self._repository.get_session(resolved_session_id)
        if record is None:
            raise RunNotFoundError(resolved_session_id)
        return record

    async def get_session_timeline(self, session_id: str) -> SessionTimeline:
        session = await self.get_session(session_id)
        runs = await self._repository.list_session_runs(session.session_id)
        return SessionTimeline(session=session, runs=runs)

    async def create_run(
        self,
        *,
        question: str,
        session_id: str | None = None,
        parent_run_id: str | None = None,
        run_id: str | None = None,
    ) -> RunRecord:
        resolved_run_id = normalize_run_id(run_id)
        resolved_session_id: str
        parent_context: ParentContextSnapshot | None = None

        if parent_run_id:
            parent = await self.get_run(parent_run_id)
            if parent.status != RunStatus.COMPLETED or not parent.final_report:
                raise RunConflictError(
                    f"parent run '{parent_run_id}' must be completed before it can be inherited"
                )
            if not parent.session_id:
                raise RunConflictError(
                    f"parent run '{parent_run_id}' is not attached to a session"
                )
            resolved_session_id = normalize_session_id(session_id or parent.session_id)
            if resolved_session_id != parent.session_id:
                raise RunConflictError("parent and child runs must belong to the same session")
            report = parent.final_report
            parent_context = ParentContextSnapshot(
                source_run_id=parent.run_id,
                source_question=parent.question,
                report_excerpt=report[:_PARENT_REPORT_LIMIT],
                reference_excerpt=_extract_reference_excerpt(report),
                report_truncated=len(report) > _PARENT_REPORT_LIMIT,
                captured_at=datetime.now(timezone.utc),
            )
        else:
            resolved_session_id = normalize_session_id(session_id)

        session = await self._repository.get_session(resolved_session_id)
        if session is None:
            if session_id or parent_run_id:
                raise RunNotFoundError(resolved_session_id)
            session = await self.create_session(
                session_id=resolved_session_id,
                title=question.strip()[:200],
            )

        record = await self._repository.create_run(
            run_id=resolved_run_id,
            session_id=session.session_id,
            parent_run_id=parent_run_id,
            parent_context=parent_context,
            question=question,
        )
        await self._repository.append_event(
            resolved_run_id,
            "run_created",
            {
                "run_id": resolved_run_id,
                "session_id": session.session_id,
                "parent_run_id": parent_run_id,
                "status": RunStatus.CREATED.value,
            },
        )
        return record

    async def get_run(self, run_id: str) -> RunRecord:
        record = await self._repository.get_run(run_id)
        if record is None:
            raise RunNotFoundError(run_id)
        return record

    async def start_run(self, run_id: str, *, resume: bool = False) -> RunRecord:
        async with self._lock:
            existing_task = self._tasks.get(run_id)
            if existing_task is not None and not existing_task.done():
                return await self.get_run(run_id)

            expected = (
                (RunStatus.INTERRUPTED, RunStatus.FAILED)
                if resume
                else (RunStatus.CREATED,)
            )
            record = await self._repository.claim_run(run_id, expected)
            await self._repository.append_event(
                run_id,
                "run_resumed" if resume else "run_started",
                {"run_id": run_id, "status": RunStatus.RUNNING.value},
            )
            task = asyncio.create_task(
                self._execute(record, resume=resume),
                name=f"research-run:{run_id}",
            )
            self._tasks[run_id] = task
            task.add_done_callback(lambda _: self._tasks.pop(run_id, None))
        return record

    async def _execute(self, record: RunRecord, *, resume: bool) -> None:
        run_id = record.run_id
        try:
            stream = (
                aresume_research(run_id)
                if resume
                else astream_research(
                    record.question,
                    run_id,
                    parent_context=(
                        record.parent_context.model_dump(mode="json")
                        if record.parent_context
                        else None
                    ),
                )
            )
            completed_payload: dict | None = None
            async for event_type, payload in stream:
                if "sections" in payload:
                    await self._repository.save_sections(run_id, payload["sections"])
                await self._repository.append_event(run_id, event_type, payload)
                if event_type == "done":
                    completed_payload = payload

            if completed_payload is None:
                raise RuntimeError("graph finished without a done event")

            await self._repository.complete_run(
                run_id,
                str(completed_payload.get("report", "")),
            )
        except asyncio.CancelledError:
            record = await self._repository.interrupt_run(run_id, "service shutdown")
            if record.status == RunStatus.INTERRUPTED:
                await self._repository.append_event(
                    run_id,
                    "run_interrupted",
                    {"run_id": run_id, "reason": "service_shutdown"},
                )
            raise
        except Exception as exc:
            logger.error("[RunService] run %s failed: %s", run_id, exc, exc_info=True)
            await self._repository.fail_run(run_id, str(exc))
            await self._repository.append_event(
                run_id,
                "error",
                {
                    "run_id": run_id,
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            )

    async def iter_events(
        self,
        run_id: str,
        *,
        after: int = 0,
    ) -> AsyncGenerator[RunEventRecord, None]:
        await self.get_run(run_id)
        cursor = after
        while True:
            events = await self._repository.list_events(run_id, after=cursor)
            for event in events:
                cursor = event.sequence
                yield event

            record = await self.get_run(run_id)
            if record.status in TERMINAL_RUN_STATUSES or record.status in {
                RunStatus.FAILED,
                RunStatus.INTERRUPTED,
            }:
                trailing = await self._repository.list_events(run_id, after=cursor)
                for event in trailing:
                    cursor = event.sequence
                    yield event
                return
            await asyncio.sleep(self._poll_interval)

    async def shutdown(self) -> None:
        tasks = [task for task in self._tasks.values() if not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
