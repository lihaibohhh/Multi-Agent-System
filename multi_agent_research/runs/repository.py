"""PostgreSQL repository for business-level run metadata and events."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from psycopg.errors import UniqueViolation
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from ..core.config import settings
from .models import (
    ParentContextSnapshot,
    RunEventRecord,
    RunRecord,
    RunStatus,
    SessionRecord,
)


class RunNotFoundError(LookupError):
    """Raised when a requested run does not exist."""


class RunConflictError(RuntimeError):
    """Raised when a run cannot perform the requested state transition."""


class RunStore(Protocol):
    async def create_session(self, *, session_id: str, title: str) -> SessionRecord: ...
    async def get_session(self, session_id: str) -> SessionRecord | None: ...
    async def list_session_runs(self, session_id: str) -> list[RunRecord]: ...

    async def create_run(
        self,
        *,
        run_id: str,
        session_id: str | None,
        parent_run_id: str | None,
        parent_context: ParentContextSnapshot | None,
        question: str,
    ) -> RunRecord: ...

    async def get_run(self, run_id: str) -> RunRecord | None: ...
    async def save_sections(self, run_id: str, sections: list[dict]) -> None: ...
    async def claim_run(self, run_id: str, expected: Sequence[RunStatus]) -> RunRecord: ...
    async def complete_run(self, run_id: str, final_report: str) -> RunRecord: ...
    async def fail_run(self, run_id: str, error_message: str) -> RunRecord: ...
    async def interrupt_run(self, run_id: str, reason: str) -> RunRecord: ...
    async def mark_stale_running_interrupted(self) -> list[str]: ...

    async def append_event(
        self,
        run_id: str,
        event_type: str,
        payload: dict,
    ) -> RunEventRecord: ...

    async def list_events(self, run_id: str, after: int = 0) -> list[RunEventRecord]: ...


class PostgresRunRepository:
    """Owns a separate pool and tables from LangGraph's checkpoint storage."""

    def __init__(self) -> None:
        self._pool: AsyncConnectionPool | None = None

    async def open(self) -> None:
        if self._pool is not None:
            return
        self._pool = AsyncConnectionPool(
            settings.database.url,
            min_size=1,
            max_size=settings.database.pool_size,
            kwargs={"autocommit": True, "row_factory": dict_row},
            open=False,
        )
        await self._pool.open(wait=True)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    def _require_pool(self) -> AsyncConnectionPool:
        if self._pool is None:
            raise RuntimeError("run repository is not open")
        return self._pool

    async def setup(self) -> None:
        pool = self._require_pool()
        async with pool.connection() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS research_sessions (
                    session_id VARCHAR(128) PRIMARY KEY,
                    title VARCHAR(200) NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS research_runs (
                    run_id VARCHAR(128) PRIMARY KEY,
                    session_id VARCHAR(128),
                    parent_run_id VARCHAR(128),
                    thread_id VARCHAR(128) NOT NULL UNIQUE,
                    question TEXT NOT NULL,
                    status VARCHAR(32) NOT NULL,
                    parent_context JSONB,
                    final_report TEXT,
                    error_message TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    started_at TIMESTAMPTZ,
                    completed_at TIMESTAMPTZ,
                    CONSTRAINT research_runs_status_check CHECK (
                        status IN (
                            'created', 'running', 'completed',
                            'failed', 'interrupted', 'cancelled'
                        )
                    )
                )
                """
            )
            await conn.execute(
                """
                ALTER TABLE research_runs
                ADD COLUMN IF NOT EXISTS parent_context JSONB
                """
            )
            await conn.execute(
                "ALTER TABLE research_runs ADD COLUMN IF NOT EXISTS sections "
                "JSONB NOT NULL DEFAULT '[]'::jsonb"
            )
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS research_run_events (
                    sequence BIGSERIAL PRIMARY KEY,
                    run_id VARCHAR(128) NOT NULL
                        REFERENCES research_runs(run_id) ON DELETE CASCADE,
                    event_type VARCHAR(64) NOT NULL,
                    payload JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS research_runs_session_created_idx
                ON research_runs (session_id, created_at DESC)
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS research_runs_parent_idx
                ON research_runs (parent_run_id)
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS research_run_events_run_sequence_idx
                ON research_run_events (run_id, sequence)
                """
            )

    async def health_check(self) -> bool:
        pool = self._require_pool()
        async with pool.connection() as conn:
            row = await (await conn.execute("SELECT 1 AS ok")).fetchone()
        return bool(row and row["ok"] == 1)

    async def create_session(self, *, session_id: str, title: str) -> SessionRecord:
        pool = self._require_pool()
        try:
            async with pool.connection() as conn:
                row = await (
                    await conn.execute(
                        """
                        INSERT INTO research_sessions (session_id, title)
                        VALUES (%s, %s)
                        RETURNING *
                        """,
                        (session_id, title),
                    )
                ).fetchone()
        except UniqueViolation as exc:
            raise RunConflictError(f"session_id '{session_id}' already exists") from exc
        return SessionRecord.model_validate(row)

    async def get_session(self, session_id: str) -> SessionRecord | None:
        pool = self._require_pool()
        async with pool.connection() as conn:
            row = await (
                await conn.execute(
                    "SELECT * FROM research_sessions WHERE session_id = %s",
                    (session_id,),
                )
            ).fetchone()
        return SessionRecord.model_validate(row) if row else None

    async def list_session_runs(self, session_id: str) -> list[RunRecord]:
        pool = self._require_pool()
        async with pool.connection() as conn:
            rows = await (
                await conn.execute(
                    """
                    SELECT * FROM research_runs
                    WHERE session_id = %s
                    ORDER BY created_at ASC, run_id ASC
                    """,
                    (session_id,),
                )
            ).fetchall()
        return [RunRecord.model_validate(row) for row in rows]

    async def create_run(
        self,
        *,
        run_id: str,
        session_id: str | None,
        parent_run_id: str | None,
        parent_context: ParentContextSnapshot | None,
        question: str,
    ) -> RunRecord:
        pool = self._require_pool()
        try:
            async with pool.connection() as conn:
                row = await (
                    await conn.execute(
                        """
                        INSERT INTO research_runs (
                            run_id, session_id, parent_run_id, thread_id,
                            question, status, parent_context
                        ) VALUES (%s, %s, %s, %s, %s, 'created', %s)
                        RETURNING *
                        """,
                        (
                            run_id,
                            session_id,
                            parent_run_id,
                            run_id,
                            question,
                            Jsonb(parent_context.model_dump(mode="json"))
                            if parent_context
                            else None,
                        ),
                    )
                ).fetchone()
                if session_id:
                    await conn.execute(
                        """
                        UPDATE research_sessions SET updated_at = NOW()
                        WHERE session_id = %s
                        """,
                        (session_id,),
                    )
        except UniqueViolation as exc:
            raise RunConflictError(f"run_id '{run_id}' already exists") from exc
        return RunRecord.model_validate(row)

    async def get_run(self, run_id: str) -> RunRecord | None:
        pool = self._require_pool()
        async with pool.connection() as conn:
            row = await (
                await conn.execute(
                    "SELECT * FROM research_runs WHERE run_id = %s",
                    (run_id,),
                )
            ).fetchone()
        return RunRecord.model_validate(row) if row else None

    async def save_sections(self, run_id: str, sections: list[dict]) -> None:
        """Queryable projection of chapter checkpoints, including unfinished drafts."""
        from ..sections.models import SectionRecord

        normalized = [SectionRecord.model_validate(s).model_dump(mode="json") for s in sections]
        async with self._require_pool().connection() as conn:
            cursor = await conn.execute(
                "UPDATE research_runs SET sections = %s, updated_at = NOW() WHERE run_id = %s",
                (Jsonb(normalized), run_id),
            )
        if cursor.rowcount != 1:
            raise RunNotFoundError(run_id)

    async def claim_run(self, run_id: str, expected: Sequence[RunStatus]) -> RunRecord:
        pool = self._require_pool()
        expected_values = [status.value for status in expected]
        async with pool.connection() as conn:
            row = await (
                await conn.execute(
                    """
                    UPDATE research_runs
                    SET status = 'running', started_at = COALESCE(started_at, NOW()),
                        updated_at = NOW(), error_message = NULL
                    WHERE run_id = %s AND status = ANY(%s)
                    RETURNING *
                    """,
                    (run_id, expected_values),
                )
            ).fetchone()
        if row:
            return RunRecord.model_validate(row)
        current = await self.get_run(run_id)
        if current is None:
            raise RunNotFoundError(run_id)
        raise RunConflictError(
            f"run '{run_id}' cannot start from status '{current.status.value}'"
        )

    async def complete_run(self, run_id: str, final_report: str) -> RunRecord:
        return await self._finish_run(
            run_id,
            status=RunStatus.COMPLETED,
            final_report=final_report,
            error_message=None,
        )

    async def fail_run(self, run_id: str, error_message: str) -> RunRecord:
        return await self._finish_run(
            run_id,
            status=RunStatus.FAILED,
            final_report=None,
            error_message=error_message,
        )

    async def interrupt_run(self, run_id: str, reason: str) -> RunRecord:
        pool = self._require_pool()
        async with pool.connection() as conn:
            row = await (
                await conn.execute(
                    """
                    UPDATE research_runs
                    SET status = 'interrupted', error_message = %s, updated_at = NOW()
                    WHERE run_id = %s AND status = 'running'
                    RETURNING *
                    """,
                    (reason, run_id),
                )
            ).fetchone()
        if row:
            return RunRecord.model_validate(row)
        current = await self.get_run(run_id)
        if current is None:
            raise RunNotFoundError(run_id)
        return current

    async def _finish_run(
        self,
        run_id: str,
        *,
        status: RunStatus,
        final_report: str | None,
        error_message: str | None,
    ) -> RunRecord:
        pool = self._require_pool()
        async with pool.connection() as conn:
            row = await (
                await conn.execute(
                    """
                    UPDATE research_runs
                    SET status = %s, final_report = COALESCE(%s, final_report),
                        error_message = %s, updated_at = NOW(),
                        completed_at = CASE
                            WHEN %s = 'completed' THEN NOW()
                            ELSE completed_at
                        END
                    WHERE run_id = %s
                    RETURNING *
                    """,
                    (
                        status.value,
                        final_report,
                        error_message,
                        status.value,
                        run_id,
                    ),
                )
            ).fetchone()
            if row and row.get("session_id"):
                await conn.execute(
                    """
                    UPDATE research_sessions SET updated_at = NOW()
                    WHERE session_id = %s
                    """,
                    (row["session_id"],),
                )
        if row is None:
            raise RunNotFoundError(run_id)
        return RunRecord.model_validate(row)

    async def mark_stale_running_interrupted(self) -> list[str]:
        pool = self._require_pool()
        async with pool.connection() as conn:
            rows = await (
                await conn.execute(
                    """
                    UPDATE research_runs
                    SET status = 'interrupted',
                        error_message = 'service restarted while run was active',
                        updated_at = NOW()
                    WHERE status = 'running'
                    RETURNING run_id
                    """
                )
            ).fetchall()
        return [row["run_id"] for row in rows]

    async def append_event(
        self,
        run_id: str,
        event_type: str,
        payload: dict,
    ) -> RunEventRecord:
        pool = self._require_pool()
        async with pool.connection() as conn:
            row = await (
                await conn.execute(
                    """
                    INSERT INTO research_run_events (run_id, event_type, payload)
                    VALUES (%s, %s, %s)
                    RETURNING sequence, run_id, event_type, payload, created_at
                    """,
                    (run_id, event_type, Jsonb(payload)),
                )
            ).fetchone()
        return RunEventRecord.model_validate(row)

    async def list_events(self, run_id: str, after: int = 0) -> list[RunEventRecord]:
        pool = self._require_pool()
        async with pool.connection() as conn:
            rows = await (
                await conn.execute(
                    """
                    SELECT sequence, run_id, event_type, payload, created_at
                    FROM research_run_events
                    WHERE run_id = %s AND sequence > %s
                    ORDER BY sequence ASC
                    """,
                    (run_id, after),
                )
            ).fetchall()
        return [RunEventRecord.model_validate(row) for row in rows]
