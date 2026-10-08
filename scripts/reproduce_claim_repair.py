"""Replay the reported Claim failure read-only; --live permits bounded isolated LLM calls.

Run: python -m scripts.reproduce_claim_repair [--live]
No production Run/Checkpoint/budget writes; no knowledge-service or KB file access.
"""
import argparse
import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace

from psycopg import AsyncConnection
from psycopg.rows import dict_row

from scripts.reproduce_retrieval_case import Ledger, RUN, emit, read_case
from multi_agent_research.core.budget import RunBudget, current_budget, budget_summary
from multi_agent_research.core.config import settings
from multi_agent_research.sections import claim_repair, workflow
from multi_agent_research.sections.model_output import attempt_sink
from multi_agent_research.sections.models import Claim, SectionRecord

DIAGNOSTIC = 'c4d1ce66972b4d06878c57bf929b0103'


async def main(live):
    state, before = await read_case()
    state = deepcopy(state)
    async with await AsyncConnection.connect(settings.database.url, autocommit=True, row_factory=dict_row,
        connect_timeout=5, options='-c default_transaction_read_only=on -c statement_timeout=15000') as conn:
        row = await (await conn.execute(
            'SELECT data FROM research_model_attempts WHERE run_id=%s AND diagnostic_id=%s',
            (RUN, DIAGNOSTIC))).fetchone()
    if not row:
        raise RuntimeError('historical diagnostic unavailable')
    section = SectionRecord.model_validate(next(s for s in state['sections'] if s['section_id'] == 'section_2'))
    work = claim_repair.split_extraction(section, json.loads(row['data']['raw'])['claims'])
    original = deepcopy(work['accepted'])
    emit('historical_replay', accepted=len(original), pending=[p['slot'] for p in work['pending']],
         error_types=[e['type'] for p in work['pending'] for e in p['errors']])
    if not live:
        return
    ledger = Ledger()
    ledger.budget['policy']['model_calls'] = 3
    record = SimpleNamespace(run_id='isolated-claim-repair', execution_id='diagnostic', budget=ledger.budget)
    token = current_budget.set(RunBudget(ledger, record, asyncio.Semaphore(1)))
    async def sink(data):
        emit('model_attempt', schema=data['schema'], accepted=data['accepted'], tokens=data['tokens'],
             accepted_slots=data.get('accepted_slots'), pending_slots=data.get('pending_slots'),
             error_types=[e['type'] for e in data['errors']])
    audit = attempt_sink.set(sink)
    work.update(epoch=claim_repair.epoch(), attempts=0)
    section.claim_work = work
    section.claims = [Claim.model_validate(v) for v in original.values()]
    section.status = 'claims_pending'
    index = next(i for i, s in enumerate(state['sections']) if s['section_id'] == section.section_id)
    state['sections'][index] = section.model_dump(mode='json')
    state.update(active_section=index, section_step='claims')
    try:
        for _ in range(3):
            state.update(await workflow.extract_claims(state))
            if state['section_step'] == 'advance':
                break
            state.update(workflow.claim_gate(state))
        repaired = state['sections'][index]['claim_work']
        emit('result', accepted=len(repaired['accepted']), pending=len(repaired['pending']),
             accepted_siblings_unchanged=all(repaired['accepted'][k] == v for k, v in original.items()),
             next_step=state['section_step'])
    finally:
        emit('isolated_budget', budget=budget_summary(ledger.budget))
        attempt_sink.reset(audit)
        current_budget.reset(token)
        _, after = await read_case()
        emit('original_unchanged', value=before == after)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main(args.live))
