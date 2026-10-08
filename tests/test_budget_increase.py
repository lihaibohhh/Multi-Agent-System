import pytest
from pydantic import ValidationError

from multi_agent_research.core.budget import BudgetExceeded, new_budget, reserve, start_budget
from multi_agent_research.runs.models import BudgetIncreaseRequest


def request(**changes):
    return BudgetIncreaseRequest.model_validate(dict(confirm=True, request_id='increase-test-1',
        expected_tokens=500000, new_tokens=600000, reason='明确追加研究额度', **changes))


def test_budget_block_explains_shortfall_without_admitting_call():
    value = start_budget(new_budget())
    value['policy']['tokens'] = 500000
    value['charged_tokens'] = value['known_tokens'] = 435910
    with pytest.raises(BudgetExceeded) as caught:
        reserve(value, 'denied', 'model', 64908, 'report_review')
    assert caught.value.details['shortfall_tokens'] == 818
    assert caught.value.details['remaining_tokens'] == 64090
    assert '64,908' in str(caught.value)
    assert 'denied' not in value['reservations'] and value['model_calls'] == 0


@pytest.mark.parametrize('change', [{'new_tokens':True}, {'new_tokens':600000.5},
    {'expected_tokens':0}, {'request_id':'bad id!'}, {'reason':'短'}])
def test_increase_request_rejects_ambiguous_inputs(change):
    raw = request().model_dump()
    raw.update(change)
    with pytest.raises(ValidationError):
        BudgetIncreaseRequest.model_validate(raw)


@pytest.mark.asyncio
async def test_increase_http_passes_explicit_contract_without_starting(monkeypatch):
    import httpx
    from multi_agent_research.api import server
    from multi_agent_research.runs.repository import RunConflictError
    from tests.test_run_service import MemoryRunStore
    from multi_agent_research.runs.service import RunService
    store = MemoryRunStore()
    service = RunService(store)
    await service.create_run(question='test budget increase', run_id='r')
    calls = []
    async def increase(run_id, payload):
        calls.append(payload)
        if not payload.confirm:
            raise ValueError('需明确确认')
        if payload.expected_tokens != 500000:
            raise RunConflictError('额度已变化')
        return await store.get_run(run_id)
    store.increase_run_budget = increase
    monkeypatch.setattr(server, 'run_service', service)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url='http://test') as client:
        payload = request().model_dump()
        assert (await client.post('/api/runs/r/budget/increase', json=payload)).status_code == 200
        assert (await client.post('/api/runs/r/budget/increase', json={**payload, 'confirm':False})).status_code == 422
        assert (await client.post('/api/runs/r/budget/increase', json={**payload, 'expected_tokens':400000})).status_code == 409
        assert (await client.post('/api/runs/r/budget/increase', json={**payload, 'new_tokens':True})).status_code == 422
    assert len(calls) == 3 and not service._tasks
