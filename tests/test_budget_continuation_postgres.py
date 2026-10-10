"""Real budget/account tests confined to a rollback-only private schema."""
import os
import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
from uuid import uuid4

import pytest
from psycopg import AsyncConnection, sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from multi_agent_research.core.config import settings
from multi_agent_research.core.budget import BudgetExceeded, ExecutionPaused
from multi_agent_research.runs.models import BudgetIncreaseRequest, RunStatus
from multi_agent_research.runs.repository import PostgresRunRepository, RunConflictError, StaleExecutionError

pytestmark = pytest.mark.skipif(os.getenv("RUN_POSTGRES_TESTS") != "1", reason="opt-in isolated PostgreSQL")


@pytest.mark.asyncio
async def test_setup_splits_a_legacy_shared_account_by_run_without_resetting_usage():
    """The production migration must preserve each Run's usage and own increases."""
    async with await AsyncConnection.connect(
        settings.database.url,
        autocommit=True,
        row_factory=dict_row,
    ) as conn:
        schema = "budget_split_test_" + uuid4().hex
        async with conn.transaction(force_rollback=True):
            await conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
            await conn.execute(sql.SQL("SET LOCAL search_path TO {}").format(sql.Identifier(schema)))

            class Pool:
                @asynccontextmanager
                async def connection(self):
                    yield conn

            repo = PostgresRunRepository()
            repo._pool = Pool()
            await repo.setup()
            root = await repo.create_run(
                run_id="root",
                session_id=None,
                parent_run_id=None,
                parent_context=None,
                question="root",
            )
            root = await repo.begin_execution("root", (RunStatus.CREATED,), resume=False)
            await repo.reserve_budget("root", root.execution_id, "root-model", "model", 120, "plan")
            await repo.settle_budget("root", root.execution_id, "root-model", 100)
            await repo.finish_execution("root", root.execution_id, RunStatus.PAUSED, {})

            child = await repo.create_run(
                run_id="child",
                session_id=None,
                parent_run_id="root",
                parent_context=None,
                question="child",
            )
            child = await repo.begin_execution("child", (RunStatus.CREATED,), resume=False)
            await repo.reserve_budget("child", child.execution_id, "child-model", "model", 240, "write")
            await repo.settle_budget("child", child.execution_id, "child-model", 200)
            await repo.finish_execution("child", child.execution_id, RunStatus.PAUSED, {})
            original_limit = child.budget["policy"]["tokens"]
            increased_limit = original_limit + 100_000
            child = await repo.increase_run_budget(
                "child",
                BudgetIncreaseRequest(
                    confirm=True,
                    request_id="child-only-increase",
                    expected_tokens=original_limit,
                    new_tokens=increased_limit,
                    reason="test child-only independent increase",
                ),
            )

            # Recreate the legacy defect: both Runs point at one aggregate account.
            root_now = await repo.get_run("root")
            child_now = await repo.get_run("child")
            legacy_shared = deepcopy(child_now.budget)
            legacy_shared["reservations"] = {
                **root_now.budget["reservations"],
                **child_now.budget["reservations"],
            }
            legacy_shared.update(
                model_calls=2,
                retrieval_calls=2,
                known_tokens=300,
                charged_tokens=300,
            )
            old_budget_id = root_now.budget_id
            await conn.execute(
                "UPDATE research_budget_accounts SET budget=%s WHERE budget_id=%s",
                (Jsonb({**legacy_shared, "deadline": None}), old_budget_id),
            )
            await conn.execute(
                "UPDATE research_runs SET budget_id=%s WHERE run_id='child'",
                (old_budget_id,),
            )
            for run_id in ("root", "child"):
                await conn.execute(
                    "INSERT INTO research_retrieval_operations (run_id,operation_id,data) "
                    "VALUES (%s,%s,%s)",
                    (run_id, f"{run_id}-search", Jsonb({"status": "completed"})),
                )

            await repo.setup()
            split_root = await repo.get_run("root")
            split_child = await repo.get_run("child")
            assert split_root.budget_id != split_child.budget_id
            assert split_root.budget_id != old_budget_id
            assert split_child.budget_id != old_budget_id
            assert split_root.budget["policy"]["tokens"] == original_limit
            assert split_child.budget["policy"]["tokens"] == increased_limit
            assert split_root.budget["known_tokens"] == 100
            assert split_child.budget["known_tokens"] == 200
            assert split_root.budget["model_calls"] == 1
            assert split_child.budget["model_calls"] == 1
            assert split_root.budget["retrieval_calls"] == 1
            assert split_child.budget["retrieval_calls"] == 1
            assert set(split_root.budget["reservations"]) == {"root-model"}
            assert set(split_child.budget["reservations"]) == {"child-model"}
            assert "child-only-increase" not in split_root.budget.get("increases", {})
            assert "child-only-increase" in split_child.budget["increases"]
            for run_id in ("root", "child"):
                assert "budget_account_isolated" in {
                    event.event_type for event in await repo.list_events(run_id)
                }


@pytest.mark.asyncio
async def test_independent_accounts_migration_pause_and_restart(monkeypatch):
    async with await AsyncConnection.connect(settings.database.url, autocommit=True, row_factory=dict_row) as conn:
        schema = "budget_v2_test_" + uuid4().hex
        async with conn.transaction(force_rollback=True):
            await conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
            await conn.execute(sql.SQL("SET LOCAL search_path TO {}").format(sql.Identifier(schema)))
            class Pool:
                @asynccontextmanager
                async def connection(self):
                    yield conn
            repo = PostgresRunRepository()
            repo._pool = Pool()
            await repo.setup()
            await repo.setup()  # schema upgrade idempotent, no account reset
            root = await repo.create_run(run_id="root", session_id=None, parent_run_id=None, parent_context=None, question="research")
            a = await repo.begin_execution("root", (RunStatus.CREATED,), resume=False)
            await repo.reserve_budget("root", a.execution_id, "settled", "model", 100, "write")
            await repo.settle_budget("root", a.execution_id, "settled", 70)
            await repo.settle_budget("root", a.execution_id, "settled", 999)
            await repo.reserve_budget("root", a.execution_id, "pending", "model", 80, "review")
            await repo.request_pause("root")
            with pytest.raises(ExecutionPaused):
                await repo.reserve_budget("root", a.execution_id, "forbidden", "retrieval", 0, "search")
            # In-flight settlement still accepted after a pause request.
            await repo.settle_budget("root", a.execution_id, "pending", None)
            await repo.finish_execution("root", a.execution_id, RunStatus.PAUSED, {"reason": "user_pause"})
            before = (await repo.get_run("root")).budget
            monkeypatch.setattr("multi_agent_research.core.budget.time.time", lambda: a.budget["deadline"] + 86400 * 5)
            replacement = PostgresRunRepository()
            replacement._pool = repo._pool
            b = await replacement.begin_execution("root", (RunStatus.PAUSED,), resume=True)
            assert b.budget["deadline"] > a.budget["deadline"] + 86400 * 5
            assert b.budget["known_tokens"] == 70 and b.budget["charged_tokens"] == 150
            assert b.budget["reservations"] == before["reservations"]
            with pytest.raises(StaleExecutionError):
                await repo.settle_budget("root", a.execution_id, "pending", 10)
            await repo.finish_execution("root", b.execution_id, RunStatus.COMPLETED, {"report": "saved"})
            child = await repo.create_run(run_id="child", session_id=None, parent_run_id="root", parent_context=None, question="revision")
            assert child.budget_id != root.budget_id and child.budget["charged_tokens"] == 0
            assert await repo.can_initialize_missing_checkpoint("child")
            c = await repo.begin_execution("child", (RunStatus.CREATED,), resume=False)
            # Set a test quota, do not alter any real Run.
            account = deepcopy(c.budget)
            account["policy"]["tokens"] = 50
            await conn.execute("UPDATE research_budget_accounts SET budget=%s WHERE budget_id=%s", (Jsonb(account), c.budget_id))
            with pytest.raises(BudgetExceeded):
                await repo.reserve_budget("child", c.execution_id, "too_much", "model", 51, "write")
            await repo.reserve_budget("child", c.execution_id, "fits", "model", 50, "write")
            assert not await repo.can_initialize_missing_checkpoint("child")
            assert (await repo.get_snapshot("root")).run.budget["charged_tokens"] == 150
            # Replacing graph/business projection cannot roll back the account.
            await repo.save_sections("child", [])
            await conn.execute("UPDATE research_runs SET budget='{}'::jsonb WHERE run_id='child'")
            assert (await repo.get_run("child")).budget["charged_tokens"] == 50

            legacy = deepcopy(before)
            legacy.update(version=1, deadline=1)
            await conn.execute("INSERT INTO research_runs (run_id,thread_id,question,status,budget) VALUES ('old','old','legacy','failed',%s)", (Jsonb(legacy),))
            with pytest.raises(RunConflictError, match="迁移"):
                await repo.begin_execution("old", (RunStatus.FAILED,), resume=True)
            old = await repo.get_run("old")
            assert old.budget == legacy and old.status == RunStatus.FAILED and old.execution_id is None
            migrated = await repo.migrate_run_budget("old", "explicit confirmed migration")
            again = await repo.migrate_run_budget("old", "repeat confirmation")
            assert migrated.budget_id == again.budget_id
            assert migrated.budget == {**legacy, "version": 2, "deadline": None}
            assert [e.event_type for e in await repo.list_events("old")] == ["budget_migrated"]
            raw = await (await conn.execute("SELECT budget FROM research_runs WHERE run_id='old'")).fetchone()
            assert raw["budget"] == legacy  # preserved original policy/ledger for audit
            assert migrated.status == RunStatus.FAILED  # migration never executes research
            assert not await repo.can_initialize_missing_checkpoint("old")
            await conn.execute("INSERT INTO research_runs (run_id,thread_id,question,status,model_usage) "
                               "VALUES ('ancient','ancient','old history','failed',%s)",
                               (Jsonb({"attempts": 1, "tokens": 10, "unknown": 0}),))
            await repo.migrate_run_budget("ancient", "confirmed test migration")
            assert not await repo.can_initialize_missing_checkpoint("ancient")
        assert await (await conn.execute("SELECT 1 FROM pg_namespace WHERE nspname=%s", (schema,))).fetchone() is None


@pytest.mark.asyncio
async def test_concurrent_children_spend_only_their_independent_accounts():
    # Separate connections require a committed private schema, deleted in finally.
    schema = "budget_concurrency_test_" + uuid4().hex
    async with await AsyncConnection.connect(settings.database.url, autocommit=True) as admin:
        await admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        try:
            async with AsyncConnectionPool(settings.database.url, min_size=1, max_size=4,
                    kwargs={"autocommit": True, "row_factory": dict_row,
                            "options": f"-c search_path={schema} -c statement_timeout=15000"}, open=False) as pool:
                repo = PostgresRunRepository()
                repo._pool = pool
                await repo.setup()
                root = await repo.create_run(run_id="root", session_id=None, parent_run_id=None, parent_context=None, question="root")
                value = deepcopy(root.budget)
                value["policy"]["tokens"] = 100
                async with pool.connection() as conn:
                    await conn.execute("UPDATE research_budget_accounts SET budget=%s WHERE budget_id=%s", (Jsonb(value), root.budget_id))
                records = []
                for name in ("child1", "child2"):
                    await repo.create_run(run_id=name, session_id=None, parent_run_id="root", parent_context=None, question="child")
                    records.append(await repo.begin_execution(name, (RunStatus.CREATED,), resume=False))
                result = await asyncio.gather(*(repo.reserve_budget(r.run_id, r.execution_id, r.run_id,
                                          "model", 80, "write") for r in records), return_exceptions=True)
                assert sum(isinstance(x, BudgetExceeded) for x in result) == 0
                assert sum(x is None for x in result) == 2
                budget = (await repo.get_run("root")).budget
                assert budget["charged_tokens"] == 0 and budget["model_calls"] == 0
                for record in records:
                    assert (await repo.get_run(record.run_id)).budget["charged_tokens"] == 80
                from multi_agent_research.runs.models import BudgetIncreaseRequest
                requests = [BudgetIncreaseRequest(confirm=True, request_id='concurrent-' + str(i),
                    expected_tokens=100, new_tokens=200, reason='并发追加独立预算测试') for i in range(2)]
                increases = await asyncio.gather(*(repo.increase_run_budget('root', r) for r in requests),
                                                 return_exceptions=True)
                assert sum(isinstance(r, RunConflictError) for r in increases) == 1
                final = (await repo.get_run('root')).budget
                assert final['policy']['tokens'] == 200 and final['charged_tokens'] == 0
                assert final['reservations'] == budget['reservations']
                assert len([e for e in await repo.list_events('root') if e.event_type == 'budget_increased']) == 1
        finally:
            assert schema.startswith("budget_concurrency_test_") and len(schema) == len("budget_concurrency_test_") + 32
            await admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
