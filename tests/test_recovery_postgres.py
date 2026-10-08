"""Real PostgreSQL checks in a private schema inside a force-rollback transaction."""

import os
from contextlib import asynccontextmanager
from uuid import uuid4

import pytest
from psycopg import AsyncConnection, sql
from psycopg.rows import dict_row

from multi_agent_research.core.config import settings
from multi_agent_research.runs.models import RunStatus
from multi_agent_research.runs.repository import PostgresRunRepository


pytestmark = pytest.mark.skipif(os.getenv("RUN_POSTGRES_TESTS") != "1", reason="opt-in PostgreSQL test")


@pytest.mark.asyncio
async def test_snapshot_terminal_event_and_private_attempts_are_transactional():
    async with await AsyncConnection.connect(settings.database.url, autocommit=True, row_factory=dict_row) as conn:
        async with conn.transaction(force_rollback=True):
            schema = "recovery_test_" + uuid4().hex
            await conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
            await conn.execute(sql.SQL("SET LOCAL search_path TO {}").format(sql.Identifier(schema)))
            class Pool:
                @asynccontextmanager
                async def connection(self):
                    yield conn
            repo = PostgresRunRepository()
            repo._pool = Pool()
            await repo.setup()
            await repo.setup()  # Migration idempotency in an isolated schema.
            await repo.create_session(session_id="s", title="snapshot recovery test")
            await repo.create_run(run_id="r", session_id="s", parent_run_id=None,
                                  parent_context=None, question="recovery integration test")
            first = await repo.claim_run("r", (RunStatus.CREATED,))
            from tests.test_claim_repair import example
            from multi_agent_research.sections.claim_repair import split_extraction
            from multi_agent_research.sections.models import Claim
            section, bad, good = example()
            section.status = 'claims_pending'
            section.claim_work = split_extraction(section, [good, bad])
            section.claims = [Claim.model_validate(section.claim_work['accepted']['1'])]
            await repo.save_sections('r', [section.model_dump(mode='json')])
            roundtrip = (await repo.get_run('r')).sections[0]
            assert roundtrip.model_dump() == section.model_dump()
            await repo.append_event("r", "run_started", {"execution_id": first.execution_id})
            audit = {"diagnostic_id": "d1", "tokens": 17, "unknown": 0,
                     "raw": "private test response", "errors": [{"field": "search_queries"}]}
            await repo.record_model_attempt("r", first.execution_id, audit)
            await repo.record_model_attempt("r", first.execution_id, audit)
            before = await repo.get_snapshot("r")
            assert before.run.model_usage == {"attempts": 1, "tokens": 17, "unknown": 0}
            # Force event serialization to fail after UPDATE; the whole finish must roll back.
            with pytest.raises(TypeError):
                await repo.finish_execution("r", first.execution_id, RunStatus.FAILED,
                                            {"message": "fail", "unserializable": object()})
            after = await repo.get_snapshot("r")
            assert after.run.status == RunStatus.RUNNING
            assert after.cursor == before.cursor
            await repo.finish_execution("r", first.execution_id, RunStatus.FAILED, {"message": "format failed"})
            failed = await repo.get_snapshot("r")
            assert failed.run.status == RunStatus.FAILED
            assert (await repo.list_events("r"))[-1].sequence == failed.cursor
            second = await repo.claim_run("r", (RunStatus.FAILED,))
            assert second.execution_id != first.execution_id
            await repo.append_event("r", "run_resumed", {"execution_id": second.execution_id})
            snapshot = await repo.get_snapshot("r")
            assert snapshot.run.sections[0].claim_work == section.claim_work
            await repo.finish_execution("r", first.execution_id, RunStatus.FAILED, {"message": "old late failure"})
            assert (await repo.get_snapshot("r")).run.status == RunStatus.RUNNING
            await repo.finish_execution("r", second.execution_id, RunStatus.COMPLETED, {"report": "done"})
            events = await repo.list_events("r", after=snapshot.cursor)
            assert [e.event_type for e in events] == ["done"]
            final = await repo.get_snapshot("r")
            assert final.run.status == RunStatus.COMPLETED
            assert final.cursor == events[0].sequence
            assert "private test response" not in final.model_dump_json()
            assert "private test response" not in events[0].model_dump_json()
        # The outer transaction rolled back schema creation, all DDL, and all test rows.
        assert await (await conn.execute("SELECT 1 FROM pg_namespace WHERE nspname = %s", (schema,))).fetchone() is None
