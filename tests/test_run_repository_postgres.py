from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

import pytest

from multi_agent_research.runs.models import ParentContextSnapshot, RunStatus
from multi_agent_research.runs.repository import PostgresRunRepository


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_POSTGRES_TESTS") != "1",
    reason="set RUN_POSTGRES_TESTS=1 to run PostgreSQL integration tests",
)


@pytest.mark.asyncio
async def test_postgres_run_lifecycle_roundtrip() -> None:
    repository = PostgresRunRepository()
    run_id = f"run_integration_{uuid.uuid4().hex}"
    child_run_id = f"run_integration_child_{uuid.uuid4().hex}"
    session_id = f"session_integration_{uuid.uuid4().hex}"
    await repository.open()
    try:
        await repository.setup()
        session = await repository.create_session(
            session_id=session_id,
            title="Integration session",
        )
        assert session.session_id == session_id
        created = await repository.create_run(
            run_id=run_id,
            session_id=session_id,
            parent_run_id=None,
            parent_context=None,
            question="PostgreSQL integration check",
        )
        assert created.status == RunStatus.CREATED
        assert created.thread_id == run_id

        await repository.append_event(run_id, "run_created", {"run_id": run_id})
        running = await repository.claim_run(run_id, (RunStatus.CREATED,))
        assert running.status == RunStatus.RUNNING

        await repository.save_sections(run_id, [{
            "section_id": "section_1", "title": "Integration chapter",
            "question": "PostgreSQL chapter roundtrip", "draft": "saved partial draft",
            "status": "drafted", "revision": 1,
        }])
        loaded = await repository.get_run(run_id)
        assert loaded.sections[0].draft == "saved partial draft"

        completed = await repository.complete_run(run_id, "integration report")
        assert completed.status == RunStatus.COMPLETED
        assert completed.final_report == "integration report"
        still_completed = await repository.interrupt_run(run_id, "late shutdown")
        assert still_completed.status == RunStatus.COMPLETED
        assert still_completed.error_message is None

        child = await repository.create_run(
            run_id=child_run_id,
            session_id=session_id,
            parent_run_id=run_id,
            parent_context=ParentContextSnapshot(
                source_run_id=run_id,
                source_question="PostgreSQL integration check",
                report_excerpt="integration report",
                report_truncated=False,
                captured_at=datetime.now(timezone.utc),
            ),
            question="PostgreSQL child check",
        )
        assert child.parent_context is not None
        assert child.parent_context.source_run_id == run_id

        events = await repository.list_events(run_id)
        assert [event.event_type for event in events] == ["run_created"]
        assert {run.run_id for run in await repository.list_session_runs(session_id)} == {
            run_id,
            child_run_id,
        }
    finally:
        pool = repository._require_pool()
        async with pool.connection() as conn:
            await conn.execute(
                "DELETE FROM research_runs WHERE run_id = %s",
                (child_run_id,),
            )
            await conn.execute("DELETE FROM research_runs WHERE run_id = %s", (run_id,))
            await conn.execute(
                "DELETE FROM research_sessions WHERE session_id = %s",
                (session_id,),
            )
        await repository.close()
