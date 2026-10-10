"""Receipt + accounting transactions, exclusively inside a rollback-only schema."""
import os
from contextlib import asynccontextmanager
from uuid import uuid4

import pytest
from psycopg import AsyncConnection, sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from multi_agent_research.core.config import settings
from multi_agent_research.core.budget import BudgetExceeded
from multi_agent_research.core.retrieval import operation_key, RetrievalFailed
from multi_agent_research.runs.models import RunStatus, ParentContextSnapshot
from multi_agent_research.runs.repository import PostgresRunRepository, StaleExecutionError, RunConflictError
from tests.test_retrieval_recovery import DESCRIPTOR

pytestmark = pytest.mark.skipif(os.getenv("RUN_POSTGRES_TESTS") != "1", reason="opt-in isolated PostgreSQL")


@pytest.mark.asyncio
async def test_atomic_receipts_replay_rollback_restart_and_fencing():
    async with await AsyncConnection.connect(settings.database.url, autocommit=True, row_factory=dict_row) as conn:
        async with conn.transaction(force_rollback=True):
            schema = "retrieval_v2_test_" + uuid4().hex
            await conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
            await conn.execute(sql.SQL("SET LOCAL search_path TO {}").format(sql.Identifier(schema)))
            class Pool:
                @asynccontextmanager
                async def connection(self):
                    yield conn
            repo = PostgresRunRepository()
            repo._pool = Pool()
            await repo.setup()
            await repo.setup()
            await repo.create_run(run_id="r", session_id=None, parent_run_id=None, parent_context=None, question="test")
            old = await repo.begin_execution("r", (RunStatus.CREATED,), resume=False)
            key = operation_key(DESCRIPTOR)
            data, cached = await repo.begin_retrieval("r", old.execution_id, key, DESCRIPTOR, "a")
            assert not cached and len(data["attempts"]) == 1
            # Force the transaction to fail AFTER it has written the receipt.
            await conn.execute(sql.SQL("""CREATE FUNCTION {}.reject_receipt_event() RETURNS trigger LANGUAGE plpgsql AS $$
                BEGIN IF NEW.event_type='retrieval_progress' THEN RAISE EXCEPTION 'injected event failure';
                END IF; RETURN NEW; END $$""").format(sql.Identifier(schema)))
            await conn.execute("CREATE TRIGGER reject_receipt BEFORE INSERT ON research_run_events "
                               "FOR EACH ROW EXECUTE FUNCTION reject_receipt_event()")
            with pytest.raises(Exception, match="injected event failure"):
                await repo.finish_retrieval("r", old.execution_id, key, "a", [], None)
            saved = await (await conn.execute("SELECT data FROM research_retrieval_operations")).fetchone()
            assert saved["data"]["status"] == "inflight"
            assert (await repo.get_run("r")).budget["reservations"]["a"]["status"] == "unknown"
            other = {**DESCRIPTOR, "round": 1}
            with pytest.raises(Exception, match="injected event failure"):
                await repo.begin_retrieval("r", old.execution_id, operation_key(other), other, "rolled_back")
            assert (await repo.get_run("r")).budget["retrieval_calls"] == 1
            count = await (await conn.execute("SELECT COUNT(*) AS n FROM research_retrieval_operations")).fetchone()
            assert count["n"] == 1
            await conn.execute("DROP TRIGGER reject_receipt ON research_run_events")
            await repo.finish_retrieval("r", old.execution_id, key, "a", [], None)
            await repo.finish_execution("r", old.execution_id, RunStatus.PAUSED, {})
            replacement = PostgresRunRepository()
            replacement._pool = repo._pool
            new = await replacement.begin_execution("r", (RunStatus.PAUSED,), resume=True)
            data, cached = await replacement.begin_retrieval("r", new.execution_id, key, DESCRIPTOR, "unused")
            assert cached and data["results"] == []
            budget = (await repo.get_run("r")).budget
            assert budget["retrieval_calls"] == 1 and "unused" not in budget["reservations"]
            with pytest.raises(StaleExecutionError):
                await repo.finish_retrieval("r", old.execution_id, key, "a", [{"bad": True}], None)
            budget["policy"]["retrieval_calls"] = 1
            await conn.execute("UPDATE research_budget_accounts SET budget=%s WHERE budget_id=%s", (Jsonb(budget), new.budget_id))
            with pytest.raises(BudgetExceeded):
                await replacement.begin_retrieval("r", new.execution_id, operation_key(other), other, "over_quota")
            assert (await repo.get_run("r")).budget["retrieval_calls"] == 1
            assert (await replacement.begin_retrieval("r", new.execution_id, key, DESCRIPTOR, "still_unused"))[1]
            await repo.reserve_budget("r", new.execution_id, "known-model", "model", 80, "write")
            await repo.settle_budget("r", new.execution_id, "known-model", 50)
            await repo.reserve_budget("r", new.execution_id, "unknown-model", "model", 20, "cancelled")
            await repo.finish_execution("r", new.execution_id, RunStatus.PAUSED, {})
            source = await repo.get_snapshot("r")
            context = ParentContextSnapshot(source_run_id="r", source_question="test", report_excerpt="",
                report_truncated=False, captured_at=source.run.created_at, section_operation={"mode": "continue"})
            await repo.append_event("r", "changed", {})
            with pytest.raises(RunConflictError, match="已改变"):
                await repo.create_run(run_id="stale-child", session_id=None, parent_run_id="r", parent_context=context,
                                      question="continue", parent_snapshot_cursor=source.cursor)
            assert await repo.get_run("stale-child") is None
            source = await repo.get_snapshot("r")
            context.section_operation.update(target="s1", source_cursor=source.cursor)
            child = await repo.create_run(run_id="child", session_id=None, parent_run_id="r", parent_context=context,
                                          question="continue", parent_snapshot_cursor=source.cursor)
            with pytest.raises(RunConflictError, match="已有选章继续"):
                await repo.create_run(run_id="duplicate", session_id=None, parent_run_id="r", parent_context=context,
                                      question="continue", parent_snapshot_cursor=source.cursor)
            assert await repo.get_run("duplicate") is None
            assert child.budget_id != new.budget_id
            assert child.budget["known_tokens"] == 0 and child.budget["charged_tokens"] == 0
            assert child.budget["reservations"] == {}
            assert (await repo.get_run("child")).parent_context.section_operation["mode"] == "continue"
            child = await repo.begin_execution("child", (RunStatus.CREATED,), resume=False)
            copied, cached = await repo.begin_retrieval("child", child.execution_id, key, DESCRIPTOR, "unused-child", "r")
            assert cached and copied["results"] == []
            # Replaying a completed parent receipt performs no external call and
            # therefore does not consume the child's independent retrieval quota.
            assert (await repo.get_run("child")).budget["retrieval_calls"] == 0
            assert "unused-child" not in (await repo.get_run("child")).budget["reservations"]
            await repo.reserve_budget("child", child.execution_id, "child-model", "model", 20, "write")
            await repo.settle_budget("child", child.execution_id, "child-model", 10)
            assert (await repo.get_run("child")).budget["known_tokens"] == 10
            assert (await repo.get_run("child")).budget["charged_tokens"] == 10
            assert (await repo.get_run("r")).budget["known_tokens"] == 50
            assert (await repo.get_run("r")).budget["charged_tokens"] == 70
            assert (await repo.get_run("r")).budget["reservations"]["unknown-model"]["status"] == "unknown"
            exhausted = {**copied, "descriptor": other, "status": "retryable_failed",
                         "attempts": [{"execution_id": "old", "status": "retryable_failed"}] * 4}
            await conn.execute("INSERT INTO research_retrieval_operations (run_id,operation_id,data) VALUES (%s,%s,%s)",
                               ("r", operation_key(other), Jsonb(exhausted)))
            with pytest.raises(RetrievalFailed):
                await repo.begin_retrieval("child", child.execution_id, operation_key(other), other, "no-reset", "r")
            assert (await repo.get_run("r")).budget["retrieval_calls"] == 1
