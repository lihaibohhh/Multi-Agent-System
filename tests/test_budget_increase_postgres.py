"""Budget adjustment tests in a private schema, rolled back including DDL."""
import os
from contextlib import asynccontextmanager
from copy import deepcopy
from uuid import uuid4

import pytest
from psycopg import AsyncConnection, sql
from psycopg.errors import CheckViolation
from psycopg.rows import dict_row

from multi_agent_research.core.config import settings
from multi_agent_research.runs.repository import PostgresRunRepository, RunConflictError
from multi_agent_research.runs.models import RunStatus
from tests.test_budget_increase import request

pytestmark = pytest.mark.skipif(os.getenv('RUN_POSTGRES_TESTS') != '1', reason='opt-in isolated PostgreSQL')


@pytest.mark.asyncio
async def test_increase_is_atomic_idempotent_run_scoped_and_preserves_usage():
    async with await AsyncConnection.connect(settings.database.url, autocommit=True, row_factory=dict_row) as conn:
        schema = 'increase_test_' + uuid4().hex
        async with conn.transaction(force_rollback=True):
            await conn.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(schema)))
            await conn.execute(sql.SQL('SET LOCAL search_path TO {}').format(sql.Identifier(schema)))
            class Pool:
                @asynccontextmanager
                async def connection(self):
                    yield conn
            repo = PostgresRunRepository()
            repo._pool = Pool()
            await repo.setup()
            await repo.create_run(run_id='root', session_id=None, parent_run_id=None, parent_context=None, question='research')
            active = await repo.begin_execution('root', (RunStatus.CREATED,), resume=False)
            with pytest.raises(RunConflictError):
                await repo.increase_run_budget('root', request())
            await repo.reserve_budget('root', active.execution_id, 'known', 'model', 300, 'write')
            await repo.settle_budget('root', active.execution_id, 'known', 123)
            await repo.reserve_budget('root', active.execution_id, 'unknown', 'model', 400, 'review')
            await repo.finish_execution('root', active.execution_id, RunStatus.BUDGET_LIMITED, {'message':'limited'})
            await repo.create_run(run_id='child', session_id=None, parent_run_id='root', parent_context=None, question='child')
            before = await repo.get_run('root')
            payload = request().model_copy(update={'expected_tokens':before.budget['policy']['tokens'],
                                                   'new_tokens':before.budget['policy']['tokens']+100000})
            after = await repo.increase_run_budget('root', payload)
            repeat = await repo.increase_run_budget('root', payload)
            assert after.budget == repeat.budget
            expected = deepcopy(before.budget)
            expected['policy']['tokens'] += 100000
            assert {k:v for k,v in after.budget.items() if k != 'increases'} == expected
            assert after.budget['known_tokens'] == 123 and after.budget['charged_tokens'] == 523
            assert after.status == RunStatus.PAUSED and after.error_message is None
            assert after.execution_id == before.execution_id
            child = await repo.get_run('child')
            assert child.budget['policy']['tokens'] != after.budget['policy']['tokens']
            assert len([e for e in await repo.list_events('root') if e.event_type == 'budget_increased']) == 1
            with pytest.raises(RunConflictError):
                await repo.increase_run_budget('root', payload.model_copy(update={'request_id':'another-request'}))
            with pytest.raises(RunConflictError):
                await repo.increase_run_budget('root', payload.model_copy(update={'reason':'不同的追加请求'}))
            with pytest.raises(ValueError):
                await repo.increase_run_budget('root', payload.model_copy(update={'confirm':False}))
            with pytest.raises(ValueError):
                await repo.increase_run_budget('root', payload.model_copy(update={'new_tokens':1}))
            # Reject a new audit insert: account/status must roll back with it.
            await conn.execute("ALTER TABLE research_run_events ADD CONSTRAINT block_new_audit CHECK (event_type <> 'budget_increased') NOT VALID")
            with pytest.raises(CheckViolation):
                await repo.increase_run_budget('root', payload.model_copy(update={'request_id':'rollback-test',
                    'expected_tokens':payload.new_tokens,'new_tokens':payload.new_tokens+100000}))
            assert (await repo.get_run('root')).budget == after.budget
        assert await (await conn.execute('SELECT 1 FROM pg_namespace WHERE nspname=%s', (schema,))).fetchone() is None
