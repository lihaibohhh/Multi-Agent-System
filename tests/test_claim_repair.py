import json
from copy import deepcopy

import pytest
from pydantic import ValidationError

from multi_agent_research.sections import claim_repair as repair
from multi_agent_research.sections.claim_candidates import ClaimRepairCandidate
from multi_agent_research.sections.models import SectionRecord
from multi_agent_research.sections.segments import build_segment_catalog


def repair_from_prompt(prompt, invalid=False):
    view = json.loads(prompt)
    draft = next(k for k in view['segments'] if k.startswith('D'))
    source = next(k for k in view['segments'] if k.startswith('E'))
    return repair.ClaimRepairs(repairs=[{
        'slot': item['slot'],
        'draft_segment_id': 'missing' if invalid else draft, 'assessment': 'supported',
        'evidence': [{'segment_id': source, 'relation': 'supports'}],
    } for item in view['pending']])


def example():
    section = SectionRecord(section_id='section_2', title='行业', question='行业趋势如何？',
        draft='头部企业的马太效应凸显。其他已经通过的结论。', revision=2,
        sources=[{'content': '细分行业进入头部企业强者恒强阶段。', 'title': '来源',
                  'source': 'knowledge', 'metadata': {}, 'score': 0.9, 'query': '行业趋势'}])
    raw = {'statement': '头部企业优势加强', 'draft_quote': '细分行业进入头部企业强者恒强阶段',
           'assessment': 'supported', 'evidence': [{'source_number': 1,
           'quote': '细分行业进入头部企业强者恒强阶段', 'relation': 'supports'}]}
    good = dict(raw, statement='不能被重写的通过项', draft_quote='其他已经通过的结论')
    return section, raw, good


def test_partial_acceptance_source_confusion_and_local_id_repair():
    section, bad, good = example()
    work = repair.split_extraction(section, [good] * 7 + [bad])
    original = deepcopy(work['accepted'])
    assert len(original) == 7 and work['pending'][0]['slot'] == 8
    assert 'claim_id' not in work['pending'][0]['candidate']
    prompt = repair.repair_prompt(section, work)
    assert '不能被重写的通过项' not in prompt
    fixed = repair.apply_repairs(section, work, repair_from_prompt(prompt))
    assert fixed['pending'] == [] and len(fixed['accepted']) == 8
    assert all(fixed['accepted'][k] == v for k, v in original.items())
    assert fixed['accepted']['8']['claim_id'] == 'section_2:v2:c8'
    assert fixed['accepted']['8']['draft_quote'] in section.draft
    assert work['pending']  # Pure validator does not mutate its input.


@pytest.mark.parametrize('kind', ['wrong_kind', 'unknown', 'duplicate', 'dropped_counter'])
def test_invalid_patches_keep_original_pending(kind):
    section, bad, good = example()
    if kind == 'dropped_counter':
        bad['evidence'][0]['relation'] = 'contradicts'
    work = repair.split_extraction(section, [good, bad])
    patches = repair_from_prompt(repair.repair_prompt(section, work))
    patch = patches.repairs[0]
    if kind == 'wrong_kind':
        patch.draft_segment_id = patch.evidence[0].segment_id
    elif kind == 'unknown':
        patch.slot = 1  # Cannot overwrite an accepted slot.
    elif kind == 'duplicate':
        patches.repairs.append(patch.model_copy(deep=True))
    updated = repair.apply_repairs(section, work, patches)
    assert updated['accepted'] == work['accepted']
    assert updated['pending'][0]['candidate']['statement'] == bad['statement']


def test_repair_schema_forbids_program_owned_or_statement_fields():
    with pytest.raises(ValidationError, match='extra_forbidden'):
        ClaimRepairCandidate.model_validate({
            'slot': 1,
            'statement': '不得由修复模型改写',
            'draft_segment_id': 'D0001',
            'assessment': 'unsupported',
            'evidence': [],
        })


def test_malformed_siblings_are_salvaged_individually():
    section, bad, good = example()
    work = repair.split_extraction(section, [good, bad, dict(bad, assessment='invalid')])
    assert len(work['accepted']) == 1 and len(work['pending']) == 2
    patches = repair_from_prompt(repair.repair_prompt(section, work)).model_dump()
    patches['repairs'][1]['assessment'] = 'invalid'
    updated = repair.salvage_patches(section, work, patches)
    assert set(updated['accepted']) == {'1', '2'}
    assert [p['slot'] for p in updated['pending']] == [3]


def test_repair_cannot_upgrade_uncertainty_or_lose_caveat():
    section, bad, _ = example()
    bad.update(assessment='uncertain', caveat='原有局限')
    work = repair.split_extraction(section, [bad])
    patches = repair_from_prompt(repair.repair_prompt(section, work))
    fixed = repair.apply_repairs(section, work, patches)
    assert fixed['accepted']['1']['assessment'] == 'uncertain'
    assert '原有局限' in fixed['accepted']['1']['caveat']


def test_source_or_draft_change_invalidates_fingerprint():
    section, _, _ = example()
    original = repair.fingerprint(section)
    section.draft += '新内容'
    assert repair.fingerprint(section) != original
    original = repair.fingerprint(section)
    section.sources[0]['content'] += '新来源'
    assert repair.fingerprint(section) != original


@pytest.mark.asyncio
async def test_real_parser_failure_keeps_good_siblings_and_charges_one_call(monkeypatch):
    from multi_agent_research.sections import workflow
    from multi_agent_research.core.state import initial_state
    from tests.test_model_output import fake_provider, audit
    section, _, good = example()
    catalog = build_segment_catalog(section)
    draft = next(
        key for key, segment in catalog.segments.items()
        if segment.kind == 'draft' and '其他已经通过的结论' in segment.text
    )
    evidence = next(
        key for key, segment in catalog.segments.items() if segment.kind == 'evidence'
    )
    valid = {
        'statement': good['statement'],
        'draft_segment_id': draft,
        'assessment': 'supported',
        'evidence': [{'segment_id': evidence, 'relation': 'supports'}],
    }
    bad_candidate = dict(valid, assessment='invalid')
    state = initial_state('行业趋势如何？')
    state.update(sections=[section.model_dump()], active_section=0)
    async with fake_provider([json.dumps({'claims': [valid, bad_candidate]})]) as (model, requests), audit() as records:
        monkeypatch.setattr(workflow, 'load_chat_model', lambda _: model)
        output = await workflow.extract_claims(state)
    saved = output['sections'][0]
    assert len(requests) == len(records) == 1
    assert output['model_calls'] == 1 and output['token_budget_used'] == 10
    assert output['section_step'] == 'claim_gate'
    assert len(saved['claims']) == 1
    assert saved['claim_work']['pending'][0]['slot'] == 2
    assert records[0]['schema_status'] == 'failed'
    assert saved['claim_work']['diagnostic_id'] == records[0]['diagnostic_id']


def test_selected_chapter_continue_keeps_partial_work():
    from multi_agent_research.sections.operations import prepare_operation, continue_step
    from multi_agent_research.sections.models import SectionPolicy
    section, bad, good = example()
    section.claim_work = repair.split_extraction(section, [good, bad])
    section.status = 'claims_pending'
    copied, _ = prepare_operation([section.model_dump()], section.section_id, 'continue', '继续局部修复')
    assert continue_step(copied[0], SectionPolicy()) == 'claims'
    assert copied[0].claim_work == section.claim_work
    copied[0].claim_work['pending'].clear()
    assert section.claim_work['pending']  # Parent snapshot remains independent.
