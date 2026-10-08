"""Explicit audited quota maintenance; never starts/restarts research or runs DDL."""
import argparse
import asyncio
import json

from psycopg import AsyncConnection
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from multi_agent_research.core.config import settings
from multi_agent_research.core.run_context import checkpoint_config
from multi_agent_research.runs.models import BudgetIncreaseRequest
from multi_agent_research.runs.repository import PostgresRunRepository


async def checkpoint_id(run_id):
    async with await AsyncConnection.connect(settings.database.url, autocommit=True, row_factory=dict_row,
        connect_timeout=5, options='-c default_transaction_read_only=on -c statement_timeout=15000') as conn:
        value = await AsyncPostgresSaver(conn).aget_tuple(checkpoint_config(run_id))
        return value.checkpoint['id'] if value else None


async def main(args):
    repo = PostgresRunRepository()
    await repo.open()
    try:
        before = await repo.get_run(args.run_id)
        if before is None or before.budget_id != args.budget_id:
            raise ValueError('Run/account identity mismatch; no change made')
        before_checkpoint = await checkpoint_id(args.run_id)
        request = BudgetIncreaseRequest(confirm=args.confirm, request_id=args.request_id,
            expected_tokens=args.expected_tokens, new_tokens=args.new_tokens, reason=args.reason)
        after = await repo.increase_run_budget(args.run_id, request)
        after_checkpoint = await checkpoint_id(args.run_id)
        print(json.dumps({'run_id':after.run_id, 'budget_id':after.budget_id,
            'previous_limit':before.budget['policy']['tokens'], 'new_limit':after.budget['policy']['tokens'],
            'known_tokens':after.budget['known_tokens'], 'charged_tokens':after.budget['charged_tokens'],
            'reservations_unchanged':before.budget['reservations'] == after.budget['reservations'],
            'ledger_unchanged':{k:v for k,v in before.budget.items() if k not in {'policy','increases'}}
                               == {k:v for k,v in after.budget.items() if k not in {'policy','increases'}},
            'artifacts_unchanged':before.sections == after.sections and before.final_report == after.final_report,
            'checkpoint_unchanged':before_checkpoint == after_checkpoint,
            'execution_id_unchanged':before.execution_id == after.execution_id,
            'status':after.status.value, 'request_id':request.request_id}, ensure_ascii=False))
    finally:
        await repo.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--budget-id', required=True)
    parser.add_argument('--request-id', required=True)
    parser.add_argument('--expected-tokens', type=int, required=True)
    parser.add_argument('--new-tokens', type=int, required=True)
    parser.add_argument('--reason', required=True)
    parser.add_argument('--confirm', action='store_true')
    args = parser.parse_args()
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main(args))
