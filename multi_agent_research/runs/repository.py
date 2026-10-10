"""PostgreSQL repository for business-level run metadata and events."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from contextlib import asynccontextmanager
from copy import deepcopy
from typing import Protocol
from uuid import uuid4

from psycopg.errors import UniqueViolation
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from ..agents.events import AgentEvent
from ..core.config import settings
from .runtime import InstanceLock
from ..core.budget import new_budget, start_budget, reserve, settle, budget_summary, migrate_budget, ExecutionPaused
from .models import (
    AgentExecutionRecord,
    AgentLifecycleEventRecord,
    BudgetIncreaseRequest,
    ParentContextSnapshot,
    RunEventRecord,
    RunRecord,
    RunUsageSummary,
    RunStatus,
    RunSnapshot,
    SessionRecord,
)
from .usage import summarize_run_usage
from ..coordination.models import CoordinationSnapshot, CoordinationUnitWrite


class RunNotFoundError(LookupError):
    """Raised when a requested run does not exist."""


class RunConflictError(RuntimeError):
    """Raised when a run cannot perform the requested state transition."""


class StaleExecutionError(RunConflictError):
    """An obsolete worker attempted to write a replaced or terminal execution."""


_AGENT_EVENT_STATUS = {
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
}


class RunStore(Protocol):
    async def increase_run_budget(self, run_id: str, request: BudgetIncreaseRequest) -> RunRecord: ...
    async def begin_retrieval(self, run_id, execution_id, operation_id, descriptor, reservation_id, reuse_parent=None): ...
    async def finish_retrieval(self, run_id, execution_id, operation_id, reservation_id, result, error): ...
    async def request_pause(self, run_id): ...
    async def migrate_run_budget(self, run_id, reason): ...
    async def reserve_budget(self, run_id, execution_id, reservation_id, kind, tokens, label): ...
    async def settle_budget(self, run_id, execution_id, reservation_id, tokens): ...
    async def assert_instance_owner(self) -> None: ...
    async def begin_execution(self, run_id: str, expected: Sequence[RunStatus], *, resume: bool) -> RunRecord: ...
    async def reconcile_running(self, active: dict[str, str]) -> list[str]: ...
    async def publish_execution_event(self, run_id: str, execution_id: str, event_type: str, payload: dict) -> None: ...
    def guard_execution(self, run_id: str, execution_id: str): ...
    async def can_initialize_missing_checkpoint(self, run_id: str) -> bool: ...
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
        parent_snapshot_cursor: int | None = None,
    ) -> RunRecord: ...

    async def get_run(self, run_id: str) -> RunRecord | None: ...
    async def get_run_usage(self, run_id: str) -> RunUsageSummary: ...
    async def get_snapshot(self, run_id: str) -> RunSnapshot: ...
    async def record_model_attempt(self, run_id: str, execution_id: str | None, data: dict) -> None: ...
    async def record_agent_event(self, run_id: str, execution_id: str,
                                 event: AgentEvent) -> AgentExecutionRecord: ...
    async def get_agent_execution(self, agent_run_id: str) -> AgentExecutionRecord | None: ...
    async def get_latest_agent_execution(self, run_id: str, agent_name: str,
                                         section_id: str | None) -> AgentExecutionRecord | None: ...
    async def list_agent_executions(self, run_id: str) -> list[AgentExecutionRecord]: ...
    async def list_agent_events(self, run_id: str, after: int = 0) -> list[AgentLifecycleEventRecord]: ...
    async def finish_execution(self, run_id: str, execution_id: str | None,
                               status: RunStatus, payload: dict) -> RunRecord: ...
    async def save_sections(self, run_id: str, sections: list[dict]) -> None: ...
    async def save_report_review(self, run_id: str, review: dict | None) -> None: ...
    async def save_coordination_unit(self, unit: CoordinationUnitWrite | dict) -> int: ...
    async def get_coordination_snapshot(self, run_id: str) -> CoordinationSnapshot | None: ...
    async def claim_run(self, run_id: str, expected: Sequence[RunStatus]) -> RunRecord: ...
    async def complete_run(self, run_id: str, final_report: str) -> RunRecord: ...
    async def replace_final_report(
        self,
        run_id: str,
        final_report: str,
        *,
        previous_sha256: str,
        new_sha256: str,
        report_quality: str,
    ) -> RunRecord: ...
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
        self._instance = InstanceLock()

    async def acquire_instance(self):
        await self._instance.acquire(settings.database.url)

    async def assert_instance_owner(self):
        await self._instance.check()

    async def release_instance(self):
        await self._instance.release()

    async def open(self) -> None:
        if self._pool is not None:
            return
        self._pool = AsyncConnectionPool(
            settings.database.url,
            min_size=1,
            max_size=settings.database.pool_size,
            kwargs={"autocommit": True, "row_factory": dict_row,
                    "options": "-c statement_timeout=15000 -c lock_timeout=10000"},
            open=False,
        )
        await self._pool.open(wait=True)

    async def close(self) -> None:
        await self.release_instance()
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
                "ALTER TABLE research_runs ADD COLUMN IF NOT EXISTS report_review JSONB"
            )
            await conn.execute("ALTER TABLE research_runs ADD COLUMN IF NOT EXISTS execution_id TEXT")
            await conn.execute("ALTER TABLE research_runs ADD COLUMN IF NOT EXISTS budget JSONB NOT NULL DEFAULT '{}'::jsonb")
            await conn.execute("CREATE TABLE IF NOT EXISTS research_budget_accounts ("
                               "budget_id TEXT PRIMARY KEY, budget JSONB NOT NULL)")
            await conn.execute("ALTER TABLE research_runs ADD COLUMN IF NOT EXISTS budget_id TEXT "
                               "REFERENCES research_budget_accounts(budget_id)")
            await conn.execute("ALTER TABLE research_runs ADD COLUMN IF NOT EXISTS execution_deadline DOUBLE PRECISION")
            await conn.execute("ALTER TABLE research_runs ADD COLUMN IF NOT EXISTS pause_requested BOOLEAN NOT NULL DEFAULT FALSE")
            await conn.execute("CREATE INDEX IF NOT EXISTS research_runs_budget_idx ON research_runs (budget_id)")
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS research_retrieval_operations (
                    run_id VARCHAR(128) NOT NULL REFERENCES research_runs(run_id) ON DELETE CASCADE,
                    operation_id TEXT NOT NULL,
                    data JSONB NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (run_id, operation_id)
                )
            """)
            async with conn.transaction():
                await conn.execute("ALTER TABLE research_runs DROP CONSTRAINT IF EXISTS research_runs_status_check")
                await conn.execute("ALTER TABLE research_runs ADD CONSTRAINT research_runs_status_check CHECK "
                                   "(status IN ('created','running','completed','failed','interrupted','cancelled',"
                                   "'paused','budget_limited'))")
            await conn.execute(
                "ALTER TABLE research_runs ADD COLUMN IF NOT EXISTS model_usage "
                "JSONB NOT NULL DEFAULT '{}'::jsonb"
            )
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS research_model_attempts (
                    diagnostic_id TEXT PRIMARY KEY,
                    run_id VARCHAR(128) NOT NULL REFERENCES research_runs(run_id) ON DELETE CASCADE,
                    execution_id TEXT,
                    data JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS research_model_attempts_run_idx "
                "ON research_model_attempts(run_id, created_at)"
            )
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS research_agent_executions (
                    agent_run_id TEXT PRIMARY KEY,
                    run_id VARCHAR(128) NOT NULL REFERENCES research_runs(run_id) ON DELETE CASCADE,
                    execution_id TEXT NOT NULL,
                    parent_agent_run_id TEXT,
                    agent_name TEXT NOT NULL,
                    agent_version TEXT NOT NULL,
                    section_id TEXT,
                    status TEXT NOT NULL CHECK (
                        status IN ('running','completed','paused','failed','interrupted')
                    ),
                    turn INTEGER NOT NULL DEFAULT 0 CHECK (turn >= 0),
                    model_ref TEXT,
                    usage JSONB NOT NULL DEFAULT '{}'::jsonb,
                    local_state JSONB NOT NULL DEFAULT '{}'::jsonb,
                    handoff JSONB,
                    unresolved JSONB NOT NULL DEFAULT '[]'::jsonb,
                    error_type TEXT,
                    error_message TEXT,
                    started_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL,
                    completed_at TIMESTAMPTZ
                )
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS research_agent_executions_run_idx
                ON research_agent_executions(run_id, started_at, agent_run_id)
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS research_agent_events (
                    sequence BIGSERIAL PRIMARY KEY,
                    event_id TEXT NOT NULL UNIQUE,
                    agent_run_id TEXT NOT NULL REFERENCES research_agent_executions(agent_run_id)
                        ON DELETE CASCADE,
                    run_id VARCHAR(128) NOT NULL REFERENCES research_runs(run_id) ON DELETE CASCADE,
                    execution_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    turn INTEGER NOT NULL CHECK (turn >= 0),
                    details JSONB NOT NULL DEFAULT '{}'::jsonb,
                    occurred_at TIMESTAMPTZ NOT NULL
                )
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS research_agent_events_run_sequence_idx
                ON research_agent_events(run_id, sequence)
            """)
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
            from ..coordination.schema import setup_coordination_schema

            await setup_coordination_schema(conn)
            await self._isolate_shared_budget_accounts(conn)

    async def _isolate_shared_budget_accounts(self, conn) -> None:
        """One-time, idempotent split of legacy shared accounts into per-Run ledgers."""
        groups = await (
            await conn.execute(
                """
                SELECT budget_id FROM research_runs
                WHERE budget_id IS NOT NULL
                GROUP BY budget_id HAVING COUNT(*) > 1
                """
            )
        ).fetchall()
        for group in groups:
            old_budget_id = group["budget_id"]
            async with conn.transaction():
                account = await (
                    await conn.execute(
                        "SELECT budget FROM research_budget_accounts "
                        "WHERE budget_id = %s FOR UPDATE",
                        (old_budget_id,),
                    )
                ).fetchone()
                runs = await (
                    await conn.execute(
                        """
                        SELECT run_id, parent_run_id, model_usage, final_report, created_at
                        FROM research_runs WHERE budget_id = %s
                        ORDER BY created_at, run_id FOR UPDATE
                        """,
                        (old_budget_id,),
                    )
                ).fetchall()
                if account is None or len(runs) < 2:
                    continue
                old = account["budget"]
                run_ids = {run["run_id"] for run in runs}
                owner_id = runs[0]["run_id"]
                partitioned: dict[str, dict] = {run_id: {} for run_id in run_ids}
                for reservation_id, raw in (old.get("reservations") or {}).items():
                    entry = deepcopy(raw) if isinstance(raw, dict) else {}
                    target = entry.get("run_id")
                    if target not in run_ids:
                        target = owner_id
                        entry["run_id"] = target
                    partitioned[target][reservation_id] = entry

                increases = await (
                    await conn.execute(
                        """
                        SELECT run_id, sequence, payload
                        FROM research_run_events
                        WHERE run_id = ANY(%s) AND event_type = 'budget_increased'
                        ORDER BY sequence
                        """,
                        (list(run_ids),),
                    )
                ).fetchall()
                first_expected = next(
                    (
                        int(event["payload"]["expected_tokens"])
                        for event in increases
                        if event["payload"].get("expected_tokens") is not None
                    ),
                    int((old.get("policy") or new_budget()["policy"])["tokens"]),
                )
                increases_by_run: dict[str, list[dict]] = {run_id: [] for run_id in run_ids}
                for event in increases:
                    increases_by_run[event["run_id"]].append(event["payload"])

                retrieval_rows = await (
                    await conn.execute(
                        """
                        SELECT run_id, COUNT(*) AS calls
                        FROM research_retrieval_operations
                        WHERE run_id = ANY(%s) GROUP BY run_id
                        """,
                        (list(run_ids),),
                    )
                ).fetchall()
                retrieval_counts = {row["run_id"]: int(row["calls"]) for row in retrieval_rows}
                budgets: dict[str, dict] = {}
                for run in runs:
                    run_id = run["run_id"]
                    entries = partitioned[run_id]
                    model_entries = [entry for entry in entries.values() if entry.get("kind") == "model"]
                    retrieval_entries = [entry for entry in entries.values() if entry.get("kind") == "retrieval"]
                    actual = sum(
                        int(entry.get("actual_tokens") or 0)
                        for entry in model_entries
                        if entry.get("status") == "settled"
                    )
                    charged = sum(
                        int(
                            entry.get("actual_tokens")
                            if entry.get("status") == "settled"
                            and entry.get("actual_tokens") is not None
                            else entry.get("reserved_tokens", 0)
                        )
                        for entry in model_entries
                    )
                    usage = run.get("model_usage") or {}
                    known_tokens = max(actual, int(usage.get("tokens") or 0))
                    charged += max(0, known_tokens - actual)
                    policy = deepcopy(old.get("policy") or new_budget()["policy"])
                    policy["tokens"] = first_expected
                    for increase in increases_by_run[run_id]:
                        policy["tokens"] = int(increase["new_tokens"])
                    saved_increases = {
                        key: deepcopy(value)
                        for key, value in (old.get("increases") or {}).items()
                        if (value.get("request") or {}).get("run_id") == run_id
                    }
                    exact_model_ledger = (
                        len(model_entries) == int(usage.get("attempts") or 0)
                        and actual == int(usage.get("tokens") or 0)
                    )
                    budgets[run_id] = {
                        "version": 2,
                        "policy": policy,
                        "deadline": None,
                        "model_calls": max(len(model_entries), int(usage.get("attempts") or 0)),
                        "retrieval_calls": max(
                            len(retrieval_entries), retrieval_counts.get(run_id, 0)
                        ),
                        "known_tokens": known_tokens,
                        "charged_tokens": charged,
                        "legacy_unknown_calls": max(
                            sum(entry.get("status") != "settled" for entry in model_entries),
                            int(usage.get("unknown") or 0),
                        ),
                        "legacy_history_incomplete": bool(
                            not exact_model_ledger
                            or (old.get("legacy_history_incomplete") and run.get("final_report") and not usage)
                        ),
                        "reservations": entries,
                        "increases": saved_increases,
                    }

                # Preserve any unattributed legacy aggregate on the oldest Run only.
                for field in ("model_calls", "retrieval_calls", "known_tokens", "charged_tokens"):
                    remainder = max(
                        0,
                        int(old.get(field) or 0)
                        - sum(int(budget[field]) for budget in budgets.values()),
                    )
                    budgets[owner_id][field] += remainder
                    if remainder:
                        budgets[owner_id]["legacy_history_incomplete"] = True

                for run in runs:
                    run_id = run["run_id"]
                    new_budget_id = "budget_run_" + hashlib.sha256(run_id.encode()).hexdigest()[:32]
                    await conn.execute(
                        "INSERT INTO research_budget_accounts (budget_id, budget) VALUES (%s, %s) "
                        "ON CONFLICT (budget_id) DO UPDATE SET budget = EXCLUDED.budget",
                        (new_budget_id, Jsonb(budgets[run_id])),
                    )
                    await conn.execute(
                        "UPDATE research_runs SET budget_id = %s, updated_at = NOW() WHERE run_id = %s",
                        (new_budget_id, run_id),
                    )
                    await conn.execute(
                        """
                        INSERT INTO research_run_events (run_id, event_type, payload)
                        VALUES (%s, 'budget_account_isolated', %s)
                        """,
                        (
                            run_id,
                            Jsonb(
                                {
                                    "run_id": run_id,
                                    "previous_budget_id": old_budget_id,
                                    "budget_id": new_budget_id,
                                    "scope": "run",
                                    "preserved_usage": budget_summary(budgets[run_id]),
                                }
                            ),
                        ),
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
                    SELECT r.*, CASE WHEN a.budget_id IS NULL THEN r.budget ELSE a.budget ||
                           jsonb_build_object('deadline', r.execution_deadline) END AS budget
                    FROM research_runs r LEFT JOIN research_budget_accounts a USING (budget_id)
                    WHERE r.session_id = %s
                    ORDER BY r.created_at ASC, r.run_id ASC
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
        parent_snapshot_cursor: int | None = None,
    ) -> RunRecord:
        pool = self._require_pool()
        try:
            async with pool.connection() as conn, conn.transaction():
                budget_id = "budget_" + uuid4().hex
                parent_workspace_summary = ""
                if parent_run_id:
                    parent = await (await conn.execute("SELECT status,sections FROM research_runs WHERE run_id=%s FOR UPDATE",
                                                      (parent_run_id,))).fetchone()
                    if not parent:
                        raise RunNotFoundError(parent_run_id)
                    if parent_snapshot_cursor is not None:
                        cursor = await (await conn.execute("SELECT COALESCE(MAX(sequence),0) AS n FROM research_run_events WHERE run_id=%s",
                                                           (parent_run_id,))).fetchone()
                        if parent["status"] == "running" or cursor["n"] != parent_snapshot_cursor:
                            raise RunConflictError("父研究已改变，请重新加载章节后再创建操作")
                        operation = parent_context.section_operation if parent_context else None
                        if operation and operation["mode"] == "continue":
                            duplicate = await (await conn.execute("""
                                SELECT run_id FROM research_runs WHERE parent_run_id=%s
                                  AND parent_context->'section_operation'->>'mode'='continue'
                                  AND parent_context->'section_operation'->>'target'=%s
                                  AND parent_context->'section_operation'->>'source_cursor'=%s
                                LIMIT 1
                            """, (parent_run_id, operation.get("target"), str(parent_snapshot_cursor)))).fetchone()
                            if duplicate:
                                raise RunConflictError(f"该快照已有选章继续任务 {duplicate['run_id']}，请打开该任务继续，避免重复消耗")
                    workspace = await (
                        await conn.execute(
                            "SELECT summary FROM research_workspaces WHERE run_id = %s",
                            (parent_run_id,),
                        )
                    ).fetchone()
                    selected_ids = {
                        item.section_id if hasattr(item, "section_id") else item.get("section_id")
                        for item in (parent_context.handoff if parent_context else [])
                    }
                    all_ids = {
                        item.get("section_id") for item in (parent.get("sections") or [])
                    }
                    parent_workspace_summary = (
                        workspace["summary"]
                        if workspace and selected_ids and selected_ids == all_ids
                        else ""
                    )
                await conn.execute(
                    "INSERT INTO research_budget_accounts VALUES (%s, %s)",
                    (budget_id, Jsonb(new_budget())),
                )
                await (
                    await conn.execute(
                        """
                        INSERT INTO research_runs (
                            run_id, session_id, parent_run_id, thread_id,
                            question, status, parent_context, budget, budget_id
                        ) VALUES (%s, %s, %s, %s, %s, 'created', %s, %s, %s)
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
                            Jsonb(new_budget()),
                            budget_id,
                        ),
                    )
                ).fetchone()
                from ..coordination.repository import initialize_coordination_workspace

                await initialize_coordination_workspace(
                    conn,
                    run_id,
                    parent_context.model_dump(mode="json") if parent_context else None,
                    parent_workspace_summary=parent_workspace_summary,
                )
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
        return await self.get_run(run_id)

    async def get_run(self, run_id: str) -> RunRecord | None:
        pool = self._require_pool()
        async with pool.connection() as conn:
            row = await (
                await conn.execute(
                    "SELECT r.*, CASE WHEN a.budget_id IS NULL THEN r.budget ELSE a.budget || "
                    "jsonb_build_object('deadline', r.execution_deadline) END AS budget "
                    "FROM research_runs r LEFT JOIN research_budget_accounts a USING (budget_id) WHERE r.run_id = %s",
                    (run_id,),
                )
            ).fetchone()
        return RunRecord.model_validate(row) if row else None

    async def get_run_usage(self, run_id: str) -> RunUsageSummary:
        pool = self._require_pool()
        async with pool.connection() as conn:
            row = await (
                await conn.execute(
                    """
                    SELECT r.run_id, r.budget_id, r.model_usage,
                           CASE WHEN a.budget_id IS NULL THEN r.budget ELSE a.budget END AS budget
                    FROM research_runs r
                    LEFT JOIN research_budget_accounts a USING (budget_id)
                    WHERE r.run_id = %s
                    """,
                    (run_id,),
                )
            ).fetchone()
        if row is None:
            raise RunNotFoundError(run_id)
        return summarize_run_usage(
            run_id=row["run_id"],
            budget_id=row["budget_id"],
            model_usage=row["model_usage"],
            budget=row["budget"],
        )

    async def get_snapshot(self, run_id: str) -> RunSnapshot:
        # One MVCC statement: never fetch the cursor after a separately-read stale state.
        async with self._require_pool().connection() as conn:
            row = await (await conn.execute("""
                SELECT r.*, CASE WHEN a.budget_id IS NULL THEN r.budget ELSE a.budget ||
                       jsonb_build_object('deadline', r.execution_deadline) END AS budget,
                       COALESCE((SELECT MAX(e.sequence) FROM research_run_events e
                       WHERE e.run_id = r.run_id), 0) AS event_cursor
                FROM research_runs r LEFT JOIN research_budget_accounts a USING (budget_id) WHERE r.run_id = %s
            """, (run_id,))).fetchone()
        if row is None:
            raise RunNotFoundError(run_id)
        return RunSnapshot(run=RunRecord.model_validate(row), cursor=row["event_cursor"])

    async def record_model_attempt(self, run_id: str, execution_id: str | None, data: dict) -> None:
        """Private diagnostics, never included in Run/SSE responses. Count failed calls too."""
        async with self.guard_execution(run_id, execution_id) as conn:
            async with conn.transaction():
                inserted = await (await conn.execute("""
                    INSERT INTO research_model_attempts (diagnostic_id, run_id, execution_id, data)
                    VALUES (%s, %s, %s, %s) ON CONFLICT (diagnostic_id) DO NOTHING
                    RETURNING diagnostic_id
                """, (data["diagnostic_id"], run_id, execution_id, Jsonb(data)))).fetchone()
                if inserted:
                    await conn.execute("""
                        UPDATE research_runs SET model_usage = jsonb_build_object(
                          'attempts', COALESCE((model_usage->>'attempts')::bigint, 0) + 1,
                          'tokens', COALESCE((model_usage->>'tokens')::bigint, 0) + %s,
                          'unknown', COALESCE((model_usage->>'unknown')::bigint, 0) + %s)
                        WHERE run_id = %s
                    """, (data["tokens"], data["unknown"], run_id))

    async def record_agent_event(
        self,
        run_id: str,
        execution_id: str,
        event: AgentEvent,
    ) -> AgentExecutionRecord:
        """Idempotently append a lifecycle event and advance its Agent snapshot."""
        if event.run_id != run_id:
            raise ValueError("Agent event run_id does not match the current execution")
        status = _AGENT_EVENT_STATUS[event.event_type]
        details = event.details
        checkpoint = details.get("checkpoint") or {}
        has_checkpoint = isinstance(details.get("checkpoint"), dict)
        terminal = status in {"completed", "paused", "failed"}
        async with self.guard_execution(run_id, execution_id) as conn:
            duplicate = await (await conn.execute(
                "SELECT agent_run_id FROM research_agent_events WHERE event_id = %s",
                (event.event_id,),
            )).fetchone()
            if duplicate is not None:
                if duplicate["agent_run_id"] != event.agent_run_id:
                    raise RunConflictError("Agent event_id 已绑定到其他 Agent 执行")
                row = await (await conn.execute(
                    "SELECT * FROM research_agent_executions WHERE agent_run_id = %s",
                    (event.agent_run_id,),
                )).fetchone()
                return AgentExecutionRecord.model_validate(row)
            owner = await (await conn.execute("""
                SELECT run_id, execution_id, status FROM research_agent_executions
                WHERE agent_run_id = %s
            """, (event.agent_run_id,))).fetchone()
            if owner and (
                owner["run_id"] != run_id or owner["execution_id"] != execution_id
            ):
                raise RunConflictError("agent_run_id 已属于其他 Run 或执行批次")
            if owner and owner["status"] != "running":
                raise RunConflictError("终态 Agent 执行不能追加新的生命周期事件")
            await conn.execute("""
                INSERT INTO research_agent_executions (
                    agent_run_id, run_id, execution_id, parent_agent_run_id,
                    agent_name, agent_version, section_id, status, turn, model_ref,
                    usage, local_state, handoff, unresolved, error_type, error_message,
                    started_at, updated_at, completed_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (agent_run_id) DO UPDATE SET
                    status = CASE
                        WHEN research_agent_executions.status IN
                            ('completed','paused','failed','interrupted')
                        THEN research_agent_executions.status
                        ELSE EXCLUDED.status
                    END,
                    turn = GREATEST(research_agent_executions.turn, EXCLUDED.turn),
                    model_ref = COALESCE(EXCLUDED.model_ref, research_agent_executions.model_ref),
                    usage = CASE WHEN %s
                        THEN EXCLUDED.usage ELSE research_agent_executions.usage END,
                    local_state = CASE WHEN %s
                        THEN EXCLUDED.local_state ELSE research_agent_executions.local_state END,
                    handoff = CASE WHEN %s
                        THEN EXCLUDED.handoff ELSE research_agent_executions.handoff END,
                    unresolved = CASE WHEN %s
                        THEN EXCLUDED.unresolved ELSE research_agent_executions.unresolved END,
                    error_type = COALESCE(EXCLUDED.error_type, research_agent_executions.error_type),
                    error_message = COALESCE(EXCLUDED.error_message, research_agent_executions.error_message),
                    updated_at = EXCLUDED.updated_at,
                    completed_at = COALESCE(
                        research_agent_executions.completed_at,
                        EXCLUDED.completed_at
                    )
            """, (
                event.agent_run_id,
                run_id,
                execution_id,
                event.parent_agent_run_id,
                event.agent_name,
                event.agent_version,
                event.section_id,
                status,
                event.turn,
                details.get("model_ref"),
                Jsonb(checkpoint.get("usage", {})),
                Jsonb(checkpoint.get("local_state", {})),
                Jsonb(checkpoint.get("handoff")) if checkpoint.get("handoff") is not None else None,
                Jsonb(checkpoint.get("unresolved", [])),
                details.get("error_type"),
                details.get("error_message") or details.get("reason"),
                event.occurred_at,
                event.occurred_at,
                event.occurred_at if terminal else None,
                has_checkpoint,
                has_checkpoint,
                has_checkpoint,
                has_checkpoint,
            ))
            await conn.execute("""
                INSERT INTO research_agent_events (
                    event_id, agent_run_id, run_id, execution_id,
                    event_type, turn, details, occurred_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                event.event_id,
                event.agent_run_id,
                run_id,
                execution_id,
                event.event_type,
                event.turn,
                Jsonb(details),
                event.occurred_at,
            ))
            row = await (await conn.execute(
                "SELECT * FROM research_agent_executions WHERE agent_run_id = %s",
                (event.agent_run_id,),
            )).fetchone()
        return AgentExecutionRecord.model_validate(row)

    async def get_agent_execution(
        self,
        agent_run_id: str,
    ) -> AgentExecutionRecord | None:
        async with self._require_pool().connection() as conn:
            row = await (await conn.execute(
                "SELECT * FROM research_agent_executions WHERE agent_run_id = %s",
                (agent_run_id,),
            )).fetchone()
        return AgentExecutionRecord.model_validate(row) if row else None

    async def get_latest_agent_execution(
        self,
        run_id: str,
        agent_name: str,
        section_id: str | None,
    ) -> AgentExecutionRecord | None:
        """Return the newest checkpoint candidate for one logical Agent scope."""
        async with self._require_pool().connection() as conn:
            row = await (await conn.execute("""
                SELECT * FROM research_agent_executions
                WHERE run_id = %s AND agent_name = %s
                  AND section_id IS NOT DISTINCT FROM %s
                ORDER BY updated_at DESC, agent_run_id DESC
                LIMIT 1
            """, (run_id, agent_name, section_id))).fetchone()
        return AgentExecutionRecord.model_validate(row) if row else None

    async def list_agent_executions(self, run_id: str) -> list[AgentExecutionRecord]:
        async with self._require_pool().connection() as conn:
            rows = await (await conn.execute("""
                SELECT * FROM research_agent_executions
                WHERE run_id = %s ORDER BY started_at, agent_run_id
            """, (run_id,))).fetchall()
        return [AgentExecutionRecord.model_validate(row) for row in rows]

    async def list_agent_events(
        self,
        run_id: str,
        after: int = 0,
    ) -> list[AgentLifecycleEventRecord]:
        async with self._require_pool().connection() as conn:
            rows = await (await conn.execute("""
                SELECT * FROM research_agent_events
                WHERE run_id = %s AND sequence > %s ORDER BY sequence
            """, (run_id, after))).fetchall()
        return [AgentLifecycleEventRecord.model_validate(row) for row in rows]

    async def reserve_budget(self, run_id, execution_id, reservation_id, kind, tokens, label):
        async with self.guard_execution(run_id, execution_id) as conn:
            row = await (await conn.execute("SELECT * FROM research_runs WHERE run_id = %s", (run_id,))).fetchone()
            if row["pause_requested"]:
                raise ExecutionPaused("用户请求暂停；不再发送新请求")
            budget = await self._account_budget(conn, row, lock=True)
            budget = reserve(budget, reservation_id, kind, tokens, label)
            budget["reservations"][reservation_id].update(run_id=run_id, execution_id=execution_id)
            await self._save_account(conn, row, budget)

    async def settle_budget(self, run_id, execution_id, reservation_id, tokens):
        async with self.guard_execution(run_id, execution_id) as conn:
            row = await (await conn.execute("SELECT * FROM research_runs WHERE run_id = %s", (run_id,))).fetchone()
            budget = await self._account_budget(conn, row, lock=True)
            entry = budget["reservations"][reservation_id]
            if entry.get("execution_id") != execution_id or entry.get("run_id") != run_id:
                raise StaleExecutionError("不能结算其他执行批次的调用")
            await self._save_account(conn, row, settle(budget, reservation_id, tokens))

    async def _account_budget(self, conn, row, *, lock=False):
        if not row.get("budget_id"):
            raise RunConflictError("旧版预算需显式迁移后继续；历史费用保留")
        account = await (await conn.execute("SELECT budget FROM research_budget_accounts WHERE budget_id=%s" +
                                            (" FOR UPDATE" if lock else ""), (row["budget_id"],))).fetchone()
        return {**account["budget"], "deadline": row.get("execution_deadline")}

    async def _save_account(self, conn, row, budget):
        await conn.execute("UPDATE research_budget_accounts SET budget=%s WHERE budget_id=%s",
                           (Jsonb({**budget, "deadline": None}), row["budget_id"]))

    async def begin_retrieval(self, run_id, execution_id, operation_id, descriptor, reservation_id, reuse_parent=None):
        from ..core.retrieval import admit, progress

        async with self.guard_execution(run_id, execution_id) as conn:
            row = await (await conn.execute("SELECT * FROM research_runs WHERE run_id=%s", (run_id,))).fetchone()
            if row["pause_requested"]:
                raise ExecutionPaused("正在暂停，未发起新的检索请求")
            old = await (await conn.execute(
                "SELECT data FROM research_retrieval_operations WHERE run_id=%s AND operation_id=%s",
                (run_id, operation_id))).fetchone()
            if not old and reuse_parent:
                context = row.get("parent_context") or {}
                operation = context.get("section_operation") or {}
                if row["parent_run_id"] != reuse_parent or operation.get("mode") != "continue":
                    raise RunConflictError("仅明确的选章继续可复用父查询成果")
                old = await (await conn.execute("""
                    SELECT o.data FROM research_retrieval_operations o
                    WHERE o.run_id=%s AND o.operation_id=%s
                """, (reuse_parent, operation_id))).fetchone()
                if old:
                    # Continue inherits failed attempts too: branching cannot reset
                    # the operation's lifetime cap. New explicit refresh is distinct.
                    await conn.execute("INSERT INTO research_retrieval_operations (run_id,operation_id,data) VALUES (%s,%s,%s)",
                                       (run_id, operation_id, Jsonb({**old["data"], "reused_from_run": reuse_parent})))
            data, reused = admit(old["data"] if old else None, descriptor, execution_id, reservation_id)
            if not reused:
                budget = reserve(await self._account_budget(conn, row, lock=True), reservation_id,
                                 "retrieval", 0, descriptor["provider"])
                budget["reservations"][reservation_id].update(run_id=run_id, execution_id=execution_id,
                                                            operation_id=operation_id)
                await self._save_account(conn, row, budget)
                await conn.execute("""
                    INSERT INTO research_retrieval_operations (run_id,operation_id,data) VALUES (%s,%s,%s)
                    ON CONFLICT (run_id,operation_id) DO UPDATE SET data=EXCLUDED.data,updated_at=NOW()
                """, (run_id, operation_id, Jsonb(data)))
            await conn.execute("INSERT INTO research_run_events (run_id,event_type,payload) VALUES (%s,%s,%s)",
                (run_id, "retrieval_progress", Jsonb({**progress(data, operation_id, reused), "execution_id": execution_id})))
            return data, reused

    async def finish_retrieval(self, run_id, execution_id, operation_id, reservation_id, result, error):
        from ..core.retrieval import complete, progress

        async with self.guard_execution(run_id, execution_id) as conn:
            row = await (await conn.execute("SELECT * FROM research_runs WHERE run_id=%s", (run_id,))).fetchone()
            old = await (await conn.execute(
                "SELECT data FROM research_retrieval_operations WHERE run_id=%s AND operation_id=%s",
                (run_id, operation_id))).fetchone()
            data = complete(old["data"], execution_id, reservation_id, result, error)
            if error is None:
                budget = settle(await self._account_budget(conn, row, lock=True), reservation_id, 0)
                await self._save_account(conn, row, budget)
            # Errors/cancellation do not refund an admitted request. Unknown
            # external outcomes remain reserved; a retry is a new paid attempt.
            await conn.execute("UPDATE research_retrieval_operations SET data=%s,updated_at=NOW() "
                               "WHERE run_id=%s AND operation_id=%s", (Jsonb(data), run_id, operation_id))
            await conn.execute("INSERT INTO research_run_events (run_id,event_type,payload) VALUES (%s,%s,%s)",
                (run_id, "retrieval_progress", Jsonb({**progress(data, operation_id), "execution_id": execution_id})))

    async def request_pause(self, run_id):
        async with self._require_pool().connection() as conn:
            async with conn.transaction():
                row = await (await conn.execute("SELECT * FROM research_runs WHERE run_id=%s FOR UPDATE",
                                                (run_id,))).fetchone()
                if row is None:
                    raise RunNotFoundError(run_id)
                if row["status"] != "running":
                    raise RunConflictError("仅执行中的任务可以请求暂停")
                if not row["pause_requested"]:
                    await conn.execute("UPDATE research_runs SET pause_requested=TRUE, updated_at=NOW() WHERE run_id=%s", (run_id,))
                    await conn.execute("INSERT INTO research_run_events (run_id,event_type,payload) VALUES (%s,%s,%s)",
                        (run_id, "pause_requested", Jsonb({"execution_id": row["execution_id"], "message": "正在暂停，等待当前操作保存"})))
        return await self.get_run(run_id)

    async def migrate_run_budget(self, run_id, reason):
        """Explicit, idempotent migration; untouched old rows are never silently re-budgeted."""
        async with self._require_pool().connection() as conn:
            async with conn.transaction():
                row = await (await conn.execute("SELECT * FROM research_runs WHERE run_id=%s FOR UPDATE", (run_id,))).fetchone()
                if row is None:
                    raise RunNotFoundError(run_id)
                if row["status"] == "running":
                    raise RunConflictError("执行中的任务不能迁移预算")
                if not row["budget_id"]:
                    budget_id = "budget_" + uuid4().hex
                    budget = migrate_budget(row["budget"], row["model_usage"])
                    # Historical parent/child Runs had independent allowances; never merge them implicitly.
                    await conn.execute("INSERT INTO research_budget_accounts VALUES (%s,%s)", (budget_id, Jsonb(budget)))
                    await conn.execute("UPDATE research_runs SET budget_id=%s, execution_deadline=NULL, updated_at=NOW() WHERE run_id=%s",
                                       (budget_id, run_id))
                    await conn.execute("INSERT INTO research_run_events (run_id,event_type,payload) VALUES (%s,%s,%s)",
                        (run_id, "budget_migrated", Jsonb({"reason": reason, "budget_id": budget_id,
                         "previous_budget": budget_summary(row["budget"]), "new_budget": budget_summary(budget),
                         "historical_scope": "this_run_only", "execution_id": row["execution_id"]})))
        return await self.get_run(run_id)

    async def increase_run_budget(self, run_id: str, request: BudgetIncreaseRequest) -> RunRecord:
        """CAS + account lock + idempotency; ledger and audit commit together."""
        request = BudgetIncreaseRequest.model_validate(request)
        if not request.confirm or len(request.reason.strip()) < 5:
            raise ValueError("追加需明确确认并提供原因；不清空消耗、不自动执行")
        if request.new_tokens <= request.expected_tokens:
            raise ValueError("新 Token 上限必须高于原上限")
        async with self._require_pool().connection() as conn:
            async with conn.transaction():
                row = await (await conn.execute("SELECT * FROM research_runs WHERE run_id=%s FOR UPDATE", (run_id,))).fetchone()
                if row is None:
                    raise RunNotFoundError(run_id)
                if not row['budget_id']:
                    raise RunConflictError("旧预算需先明确迁移，不能借追加重建账本")
                budget = await self._account_budget(conn, row, lock=True)
                if budget.get('version') != 2:
                    raise RunConflictError("旧预算需先迁移")
                fingerprint = {**request.model_dump(exclude={'confirm'}), 'run_id': run_id}
                history = budget.setdefault('increases', {})
                previous = history.get(request.request_id)
                if previous is not None:
                    if previous['request'] != fingerprint:
                        raise RunConflictError("追加请求 ID 已用于不同内容")
                    # A retry after response loss never increments again or changes status.
                else:
                    if row['status'] == 'running':
                        raise RunConflictError("请先暂停当前 Run，再追加预算")
                    if budget['policy']['tokens'] != request.expected_tokens:
                        raise RunConflictError("当前 Run 的独立额度已变化，请刷新后重新确认")
                    at = await (await conn.execute("SELECT NOW() AS at")).fetchone()
                    history[request.request_id] = {'request': fingerprint, 'at': at['at'].isoformat()}
                    budget['policy']['tokens'] = request.new_tokens
                    await self._save_account(conn, row, budget)
                    # No execution starts; only release the target's budget-limited UI gate.
                    await conn.execute("UPDATE research_runs SET status=CASE WHEN status='budget_limited' THEN 'paused' ELSE status END, "
                                       "error_message=CASE WHEN status='budget_limited' THEN NULL ELSE error_message END, "
                                       "updated_at=NOW() WHERE run_id=%s", (run_id,))
                    await conn.execute("INSERT INTO research_run_events (run_id,event_type,payload) VALUES (%s,%s,%s)",
                        (run_id, 'budget_increased', Jsonb({**fingerprint, 'budget_id': row['budget_id'],
                         'delta_tokens': request.new_tokens - request.expected_tokens,
                         'previous_status': row['status'], 'execution_id': row['execution_id'],
                         'budget': budget_summary(budget), 'message': '当前 Run 的独立 Token 额度已追加，消耗保留，未启动执行'})))
        return await self.get_run(run_id)

    async def save_sections(self, run_id: str, sections: list[dict]) -> None:
        """Queryable projection of chapter checkpoints, including unfinished drafts."""
        from ..sections.models import SectionRecord

        normalized = [SectionRecord.model_validate(s).model_dump(mode="json") for s in sections]
        async with self._require_pool().connection() as conn, conn.transaction():
            cursor = await conn.execute(
                "UPDATE research_runs SET sections = %s, updated_at = NOW() WHERE run_id = %s",
                (Jsonb(normalized), run_id),
            )
            if cursor.rowcount == 1:
                row = await (
                    await conn.execute(
                        "SELECT parent_context FROM research_runs WHERE run_id = %s",
                        (run_id,),
                    )
                ).fetchone()
                from ..coordination.repository import sync_section_artifacts

                await sync_section_artifacts(
                    conn, run_id, normalized, row["parent_context"]
                )
        if cursor.rowcount != 1:
            raise RunNotFoundError(run_id)

    async def save_report_review(self, run_id: str, review: dict | None) -> None:
        from ..sections.models import ReportReview

        normalized = ReportReview.model_validate(review).model_dump(mode="json") if review else None
        async with self._require_pool().connection() as conn:
            cursor = await conn.execute(
                "UPDATE research_runs SET report_review = %s, updated_at = NOW() WHERE run_id = %s",
                (Jsonb(normalized) if normalized else None, run_id),
            )
        if cursor.rowcount != 1:
            raise RunNotFoundError(run_id)

    async def save_coordination_unit(
        self,
        unit: CoordinationUnitWrite | dict,
    ) -> int:
        """Atomically replace one parent/section unit using workspace CAS."""
        from ..coordination.repository import replace_coordination_unit

        normalized = CoordinationUnitWrite.model_validate(unit)
        async with self._require_pool().connection() as conn, conn.transaction():
            return await replace_coordination_unit(conn, normalized)

    async def get_coordination_snapshot(
        self,
        run_id: str,
    ) -> CoordinationSnapshot | None:
        """Return the read-only logical table consumed by coordination Agents."""
        from ..coordination.repository import load_coordination_snapshot

        async with self._require_pool().connection() as conn:
            return await load_coordination_snapshot(conn, run_id)

    async def claim_run(self, run_id: str, expected: Sequence[RunStatus], *, event_type: str | None = None) -> RunRecord:
        pool = self._require_pool()
        expected_values = [status.value for status in expected]
        async with pool.connection() as conn:
            async with conn.transaction():
                row = await (await conn.execute("""
                    UPDATE research_runs
                    SET status = 'running', started_at = COALESCE(started_at, NOW()),
                        updated_at = NOW(), error_message = NULL, execution_id = %s, pause_requested=FALSE
                    WHERE run_id = %s AND status = ANY(%s) RETURNING *
                """, (uuid4().hex, run_id, expected_values))).fetchone()
                if row:
                    row["budget"] = start_budget(await self._account_budget(conn, row))
                    row["execution_deadline"] = row["budget"]["deadline"]
                    await conn.execute("UPDATE research_runs SET execution_deadline = %s WHERE run_id = %s",
                                       (row["execution_deadline"], run_id))
                if row and event_type:
                    await conn.execute("INSERT INTO research_run_events (run_id, event_type, payload) VALUES (%s, %s, %s)",
                        (run_id, event_type, Jsonb({"run_id": run_id, "status": "running", "execution_id": row["execution_id"],
                                                  "execution_deadline": row["execution_deadline"], "budget_id": row["budget_id"],
                                                  "budget": budget_summary(row["budget"])})))
        if row:
            return RunRecord.model_validate(row)
        current = await self.get_run(run_id)
        if current is None:
            raise RunNotFoundError(run_id)
        raise RunConflictError(
            f"run '{run_id}' cannot start from status '{current.status.value}'"
        )

    async def begin_execution(self, run_id: str, expected: Sequence[RunStatus], *, resume: bool) -> RunRecord:
        return await self.claim_run(run_id, expected, event_type="run_resumed" if resume else "run_started")

    async def can_initialize_missing_checkpoint(self, run_id: str) -> bool:
        async with self._require_pool().connection() as conn:
            row = await (await conn.execute("""
                SELECT (r.sections = '[]'::jsonb AND r.final_report IS NULL AND r.report_review IS NULL
                    AND COALESCE((r.budget->>'model_calls')::bigint, 0) = 0
                    AND COALESCE((r.budget->>'retrieval_calls')::bigint, 0) = 0
                    AND COALESCE((r.model_usage->>'attempts')::bigint, 0) = 0
                    AND NOT EXISTS (SELECT 1 FROM research_budget_accounts a,
                        jsonb_each(a.budget->'reservations') e WHERE a.budget_id=r.budget_id
                        AND (e.value->>'run_id'=r.run_id OR NOT e.value ? 'run_id'))
                    AND NOT EXISTS(SELECT 1 FROM research_model_attempts a WHERE a.run_id = r.run_id)
                    AND NOT EXISTS(SELECT 1 FROM research_agent_events a WHERE a.run_id = r.run_id)
                    AND NOT EXISTS(SELECT 1 FROM research_run_events e WHERE e.run_id = r.run_id
                        AND e.event_type NOT IN ('run_created', 'run_started', 'run_resumed', 'start',
                                               'error', 'run_interrupted', 'section_snapshot', 'pause_requested',
                                               'run_paused', 'budget_limited', 'budget_migrated', 'budget_increased'))) AS safe
                FROM research_runs r WHERE r.run_id = %s
            """, (run_id,))).fetchone()
        return bool(row and row["safe"])

    @asynccontextmanager
    async def guard_execution(self, run_id: str, execution_id: str | None):
        async with self._require_pool().connection() as conn:
            async with conn.transaction():
                row = await (await conn.execute(
                    "SELECT status, execution_id FROM research_runs WHERE run_id = %s FOR UPDATE", (run_id,)
                )).fetchone()
                if row is None:
                    raise RunNotFoundError(run_id)
                if row["status"] != "running" or not execution_id or row["execution_id"] != execution_id:
                    raise StaleExecutionError("执行批次已过期，拒绝写入")
                yield conn

    async def _apply_artifacts(self, conn, run_id: str, payload: dict):
        from ..sections.models import ReportReview, SectionRecord
        if "sections" in payload:
            sections = [SectionRecord.model_validate(s).model_dump(mode="json") for s in payload["sections"]]
            await conn.execute("UPDATE research_runs SET sections = %s, updated_at = NOW() WHERE run_id = %s",
                               (Jsonb(sections), run_id))
            row = await (
                await conn.execute(
                    "SELECT parent_context FROM research_runs WHERE run_id = %s",
                    (run_id,),
                )
            ).fetchone()
            from ..coordination.repository import sync_section_artifacts

            await sync_section_artifacts(
                conn, run_id, sections, row["parent_context"] if row else None
            )
        if "report_review" in payload:
            review = ReportReview.model_validate(payload["report_review"]).model_dump(mode="json") if payload["report_review"] else None
            await conn.execute("UPDATE research_runs SET report_review = %s, updated_at = NOW() WHERE run_id = %s",
                               (Jsonb(review) if review else None, run_id))

    async def _terminalize_agent_executions(
        self,
        conn,
        run_id: str,
        execution_id: str | None,
        run_status: RunStatus,
        payload: dict,
    ) -> None:
        status = {
            RunStatus.FAILED: "failed",
            RunStatus.PAUSED: "paused",
            RunStatus.BUDGET_LIMITED: "paused",
            RunStatus.INTERRUPTED: "interrupted",
            RunStatus.COMPLETED: "interrupted",
        }[run_status]
        await conn.execute("""
            UPDATE research_agent_executions
            SET status = %s,
                error_type = COALESCE(error_type, %s),
                error_message = COALESCE(error_message, %s),
                updated_at = NOW(), completed_at = COALESCE(completed_at, NOW())
            WHERE run_id = %s AND execution_id IS NOT DISTINCT FROM %s
              AND status = 'running'
        """, (
            status,
            payload.get("type") or "RunExecutionEnded",
            payload.get("message") or payload.get("reason"),
            run_id,
            execution_id,
        ))

    async def publish_execution_event(self, run_id: str, execution_id: str, event_type: str, payload: dict):
        async with self.guard_execution(run_id, execution_id) as conn:
            await self._apply_artifacts(conn, run_id, payload)
            row = await (await conn.execute("SELECT * FROM research_runs WHERE run_id = %s", (run_id,))).fetchone()
            await conn.execute("INSERT INTO research_run_events (run_id, event_type, payload) VALUES (%s, %s, %s)",
                (run_id, event_type, Jsonb({**payload, "run_id": run_id, "execution_id": execution_id,
                                         "budget": budget_summary(await self._account_budget(conn, row))})))

    async def reconcile_running(self, active: dict[str, str]) -> list[str]:
        # Caller holds instance ownership and the service start/reconcile mutex.
        await self.assert_instance_owner()
        interrupted = []
        async with self._require_pool().connection() as conn:
            async with conn.transaction():
                rows = await (await conn.execute(
                    "SELECT run_id, execution_id FROM research_runs WHERE status = 'running' FOR UPDATE"
                )).fetchall()
                for row in rows:
                    if active.get(row["run_id"]) == row["execution_id"] and row["execution_id"]:
                        continue
                    run_id = row["run_id"]
                    await conn.execute("UPDATE research_runs SET status = 'interrupted', error_message = %s, "
                                       "updated_at = NOW() WHERE run_id = %s", ("执行进程退出或后台任务失联，可从 Checkpoint 恢复", run_id))
                    await self._terminalize_agent_executions(
                        conn,
                        run_id,
                        row["execution_id"],
                        RunStatus.INTERRUPTED,
                        {"reason": "execution_orphaned"},
                    )
                    await conn.execute("INSERT INTO research_run_events (run_id, event_type, payload) VALUES (%s, %s, %s)",
                        (run_id, "run_interrupted", Jsonb({"run_id": run_id, "execution_id": row["execution_id"], "reason": "execution_orphaned"})))
                    interrupted.append(run_id)
        return interrupted

    async def finish_execution(self, run_id: str, execution_id: str | None,
                               status: RunStatus, payload: dict) -> RunRecord:
        """Publish terminal state and its event atomically, only for the current execution."""
        kinds = {RunStatus.COMPLETED: "done", RunStatus.FAILED: "error",
                 RunStatus.INTERRUPTED: "run_interrupted", RunStatus.PAUSED: "run_paused",
                 RunStatus.BUDGET_LIMITED: "budget_limited"}
        if status not in kinds:
            raise ValueError("invalid execution terminal status")
        report = str(payload.get("report", "")) if status == RunStatus.COMPLETED else None
        error = payload.get("message") or payload.get("reason") if status != RunStatus.COMPLETED else None
        async with self._require_pool().connection() as conn:
            async with conn.transaction():
                row = await (await conn.execute("""
                    UPDATE research_runs SET status = %s,
                        final_report = COALESCE(%s, final_report), error_message = %s,
                        updated_at = NOW(),
                        completed_at = CASE WHEN %s = 'completed' THEN NOW() ELSE completed_at END
                    WHERE run_id = %s AND status = 'running'
                          AND execution_id IS NOT DISTINCT FROM %s RETURNING *
                """, (status.value, report, error, status.value, run_id, execution_id))).fetchone()
                if row:
                    await self._terminalize_agent_executions(
                        conn,
                        run_id,
                        execution_id,
                        status,
                        payload,
                    )
                    await self._apply_artifacts(conn, run_id, payload)
                    if status == RunStatus.COMPLETED:
                        from ..coordination.repository import finalize_coordination_workspace

                        await finalize_coordination_workspace(
                            conn,
                            run_id,
                            report_quality=str(payload.get("report_quality", "reviewed")),
                        )
                    row = await (await conn.execute("SELECT * FROM research_runs WHERE run_id = %s", (run_id,))).fetchone()
                    row["budget"] = await self._account_budget(conn, row)
                    event_payload = {**payload, "run_id": run_id, "execution_id": execution_id,
                                     "model_usage": row["model_usage"], "budget": budget_summary(row["budget"])}
                    await conn.execute("""
                        INSERT INTO research_run_events (run_id, event_type, payload)
                        VALUES (%s, %s, %s)
                    """, (run_id, kinds[status], Jsonb(event_payload)))
                    if row.get("session_id"):
                        await conn.execute("UPDATE research_sessions SET updated_at = NOW() "
                                           "WHERE session_id = %s", (row["session_id"],))
        if row:
            return RunRecord.model_validate(row)
        current = await self.get_run(run_id)
        if current is None:
            raise RunNotFoundError(run_id)
        return current

    async def complete_run(self, run_id: str, final_report: str) -> RunRecord:
        return await self._finish_run(
            run_id,
            status=RunStatus.COMPLETED,
            final_report=final_report,
            error_message=None,
        )

    async def replace_final_report(
        self,
        run_id: str,
        final_report: str,
        *,
        previous_sha256: str,
        new_sha256: str,
        report_quality: str,
    ) -> RunRecord:
        """Replace only a completed Run's presentation artifact, atomically and idempotently."""
        pool = self._require_pool()
        async with pool.connection() as conn, conn.transaction():
            current = await (
                await conn.execute(
                    "SELECT status, final_report, execution_id, session_id "
                    "FROM research_runs WHERE run_id = %s FOR UPDATE",
                    (run_id,),
                )
            ).fetchone()
            if current is None:
                raise RunNotFoundError(run_id)
            if current["status"] != RunStatus.COMPLETED.value:
                raise RunConflictError("only a completed run can be reassembled")
            if current["final_report"] != final_report:
                await conn.execute(
                    "UPDATE research_runs SET final_report = %s, updated_at = NOW() "
                    "WHERE run_id = %s",
                    (final_report, run_id),
                )
                await conn.execute(
                    """
                    INSERT INTO research_run_events (run_id, event_type, payload)
                    VALUES (%s, 'report_reassembled', %s)
                    """,
                    (
                        run_id,
                        Jsonb(
                            {
                                "run_id": run_id,
                                "execution_id": current["execution_id"],
                                "previous_sha256": previous_sha256,
                                "new_sha256": new_sha256,
                                "report_quality": report_quality,
                                "model_calls": 0,
                                "retrieval_calls": 0,
                            }
                        ),
                    ),
                )
                if current.get("session_id"):
                    await conn.execute(
                        "UPDATE research_sessions SET updated_at = NOW() WHERE session_id = %s",
                        (current["session_id"],),
                    )
        updated = await self.get_run(run_id)
        if updated is None:  # pragma: no cover - protected by the row lock above
            raise RunNotFoundError(run_id)
        return updated

    async def fail_run(self, run_id: str, error_message: str) -> RunRecord:
        return await self._finish_run(
            run_id,
            status=RunStatus.FAILED,
            final_report=None,
            error_message=error_message,
        )

    async def interrupt_run(self, run_id: str, reason: str) -> RunRecord:
        pool = self._require_pool()
        async with pool.connection() as conn, conn.transaction():
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
            if row and status == RunStatus.COMPLETED:
                from ..coordination.repository import finalize_coordination_workspace

                await finalize_coordination_workspace(conn, run_id)
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
        # Compatibility entry point; startup still requires exclusive ownership.
        return await self.reconcile_running({})

    async def append_event(
        self,
        run_id: str,
        event_type: str,
        payload: dict,
    ) -> RunEventRecord:
        pool = self._require_pool()
        async with pool.connection() as conn:
            async with conn.transaction():
                # Serialize sequence allocation/commit per Run so snapshots cannot skip
                # an earlier allocated event that commits after their cursor was read.
                current = await (await conn.execute(
                    "SELECT execution_id FROM research_runs WHERE run_id = %s FOR UPDATE",
                    (run_id,),
                )).fetchone()
                if current is None:
                    raise RunNotFoundError(run_id)
                tagged = {"execution_id": current["execution_id"], **payload}
                row = await (await conn.execute("""
                    INSERT INTO research_run_events (run_id, event_type, payload)
                    VALUES (%s, %s, %s)
                    RETURNING sequence, run_id, event_type, payload, created_at
                """, (run_id, event_type, Jsonb(tagged)))).fetchone()
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
