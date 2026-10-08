"""Opt-in real DB faults: transactional private schema and ephemeral test locks."""

import asyncio
import os
import secrets
import subprocess
import sys
from contextlib import asynccontextmanager
from uuid import uuid4

import pytest
from psycopg import AsyncConnection, sql
from psycopg.errors import StringDataRightTruncation
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from multi_agent_research.core.config import settings
from multi_agent_research.core.budget import BudgetExceeded
from multi_agent_research.runs.models import RunStatus
from multi_agent_research.runs.repository import PostgresRunRepository, StaleExecutionError
from multi_agent_research.runs.runtime import InstanceLock, InstanceUnavailableError


pytestmark = pytest.mark.skipif(os.getenv("RUN_POSTGRES_TESTS") != "1", reason="opt-in PostgreSQL runtime faults")


@pytest.mark.asyncio
async def test_real_start_event_rollback_fencing_and_reconciliation(monkeypatch):
    async with await AsyncConnection.connect(settings.database.url, autocommit=True, row_factory=dict_row) as conn:
        schema = "runtime_test_" + uuid4().hex
        async with conn.transaction(force_rollback=True):
            await conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
            await conn.execute(sql.SQL("SET LOCAL search_path TO {}").format(sql.Identifier(schema)))
            class Pool:
                @asynccontextmanager
                async def connection(self):
                    yield conn
            repo = PostgresRunRepository()
            repo._pool = Pool()
            async def owned():
                pass  # Lock acquisition itself is verified below with separate real processes.
            monkeypatch.setattr(repo, "assert_instance_owner", owned)
            await repo.setup()
            await repo.create_run(run_id="r", session_id=None, parent_run_id=None, parent_context=None, question="runtime faults")
            assert await repo.can_initialize_missing_checkpoint("r")
            # The event insert fails AFTER the status update; neither may commit.
            with pytest.raises(StringDataRightTruncation):
                await repo.claim_run("r", (RunStatus.CREATED,), event_type="x" * 65)
            assert (await repo.get_run("r")).status == RunStatus.CREATED
            assert not await repo.list_events("r")
            old = await repo.begin_execution("r", (RunStatus.CREATED,), resume=False)
            assert [e.event_type for e in await repo.list_events("r")] == ["run_started"]
            assert await repo.reconcile_running({"r": old.execution_id}) == []
            chapter = {"section_id": "s1", "title": "Test", "question": "test question", "draft": "saved"}
            with pytest.raises(TypeError):
                await repo.publish_execution_event("r", old.execution_id, "section_progress",
                                                    {"sections": [chapter], "bad": object()})
            assert not (await repo.get_run("r")).sections
            assert len(await repo.list_events("r")) == 1
            assert await repo.reconcile_running({}) == ["r"]
            after = await repo.get_snapshot("r")
            assert after.run.status == RunStatus.INTERRUPTED
            assert (await repo.list_events("r"))[-1].sequence == after.cursor
            assert await repo.reconcile_running({}) == []
            new = await repo.begin_execution("r", (RunStatus.INTERRUPTED,), resume=True)
            assert new.execution_id != old.execution_id
            with pytest.raises(StaleExecutionError):
                await repo.publish_execution_event("r", old.execution_id, "section_progress", {"sections": [chapter]})
            with pytest.raises(StaleExecutionError):
                async with repo.guard_execution("r", old.execution_id):
                    pytest.fail("obsolete checkpoint write was allowed")
            with pytest.raises(StaleExecutionError):
                await repo.record_model_attempt("r", old.execution_id, {"diagnostic_id": "bad", "tokens": 9, "unknown": 0})
            assert not (await repo.get_run("r")).model_usage
            await repo.publish_execution_event("r", new.execution_id, "section_progress", {"sections": [chapter]})
            assert not await repo.can_initialize_missing_checkpoint("r")
            await repo.finish_execution("r", new.execution_id, RunStatus.COMPLETED,
                                        {"report": "finished", "sections": [chapter]})
            await repo.finish_execution("r", old.execution_id, RunStatus.FAILED, {"message": "late"})
            assert (await repo.get_run("r")).status == RunStatus.COMPLETED

            # Durable quota survives new repository/service objects and execution IDs.
            await repo.create_run(run_id="budget", session_id=None, parent_run_id=None, parent_context=None,
                                  question="budget faults")
            record = await repo.begin_execution("budget", (RunStatus.CREATED,), resume=False)
            value = record.budget
            value["policy"]["model_calls"] = 1
            await conn.execute("UPDATE research_budget_accounts SET budget = %s WHERE budget_id = %s", (Jsonb(value), record.budget_id))
            await repo.reserve_budget("budget", record.execution_id, "one", "model", 500, "write:s1")
            assert not await repo.can_initialize_missing_checkpoint("budget")
            with pytest.raises(BudgetExceeded):
                await repo.reserve_budget("budget", record.execution_id, "two", "model", 500, "write:s1")
            await repo.finish_execution("budget", record.execution_id, RunStatus.INTERRUPTED, {"reason": "test crash"})
            replacement = PostgresRunRepository()
            replacement._pool = repo._pool
            resumed = await replacement.begin_execution("budget", (RunStatus.INTERRUPTED,), resume=True)
            assert resumed.budget["deadline"] > record.budget["deadline"]
            assert resumed.budget["model_calls"] == 1 and resumed.budget["charged_tokens"] == 500
            with pytest.raises(StaleExecutionError):
                await repo.settle_budget("budget", record.execution_id, "one", 10)
            with pytest.raises(BudgetExceeded):
                await replacement.reserve_budget("budget", resumed.execution_id, "three", "model", 10, "again")
            assert (await replacement.get_run("budget")).budget == resumed.budget

            # Upgraded legacy rows retain known history; missing historical usage is explicit.
            await conn.execute("UPDATE research_runs SET budget_id=NULL, budget = '{}'::jsonb, model_usage = %s WHERE run_id = 'r'",
                               (Jsonb({"attempts": 7, "tokens": 321, "unknown": 2}),))
            from multi_agent_research.runs.repository import RunConflictError
            with pytest.raises(RunConflictError, match="迁移"):
                await repo.claim_run("r", (RunStatus.COMPLETED,))
            migrated = await repo.migrate_run_budget("r", "explicit test migration")
            assert migrated.budget["model_calls"] == 7 and migrated.budget["known_tokens"] == 321
            assert migrated.budget["legacy_history_incomplete"]
        assert await (await conn.execute("SELECT 1 FROM pg_namespace WHERE nspname = %s", (schema,))).fetchone() is None


@pytest.mark.asyncio
async def test_real_single_instance_lock_released_when_child_process_exits():
    # Different from the API's fixed key: do not interfere with the user's server.
    key = (1296126534, 100000000 + secrets.randbelow(100000000))
    child = f'''
import asyncio,sys
from multi_agent_research.core.config import settings
from multi_agent_research.runs.runtime import InstanceLock
async def main():
    lock = InstanceLock({key!r})
    await lock.acquire(settings.database.url)
    await lock.check()
    print("OWNED", flush=True)
    await asyncio.Event().wait()
if sys.platform == "win32": asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
asyncio.run(main())
'''
    process = subprocess.Popen([sys.executable, "-u", "-c", child],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
    challenger = InstanceLock(key)
    try:
        assert (await asyncio.wait_for(asyncio.to_thread(process.stdout.readline), 20)).strip() == b"OWNED"
        with pytest.raises(InstanceUnavailableError, match="已有"):
            await challenger.acquire(settings.database.url)
        process.terminate()  # Only this test's child, not any user process.
        await asyncio.wait_for(asyncio.to_thread(process.wait), 10)
        await challenger.acquire(settings.database.url)
        await challenger.check()
        await challenger.connection.close()
        with pytest.raises(InstanceUnavailableError):
            await challenger.check()
    finally:
        if process.returncode is None:
            process.terminate()
            await asyncio.wait_for(asyncio.to_thread(process.wait), 10)
        process.stdout.close()
        process.stderr.close()
        await challenger.release()
