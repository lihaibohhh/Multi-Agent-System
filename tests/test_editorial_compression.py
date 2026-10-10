from __future__ import annotations

import json

import pytest

from multi_agent_research.agents import AgentRunner
from multi_agent_research.sections.editorial import (
    editorial_length_bounds,
    editorial_visible_length,
    stable_editorial_sections,
)
from multi_agent_research.sections.models import (
    EditedSectionArtifact,
    EditorialBlueprint,
    ReportReview,
    SectionRecord,
)
from multi_agent_research.sections.model_output import ModelOutputError
from multi_agent_research.sections.nodes.editorial import (
    compress_report_section,
    edit_report_section,
)
from multi_agent_research.sections.rendering import evidence_key
from multi_agent_research.sections.validation import validate_editorial_blueprint


def _section() -> SectionRecord:
    source = {
        "query": "成本",
        "source": "knowledge",
        "content": "成本持续下降。",
        "score": 0.9,
        "metadata": {"source": "成本.pdf", "chunk_id": "cost::1", "page": 1},
    }
    return SectionRecord(
        section_id="section_1",
        title="成本",
        question="成本变化意味着什么？",
        status="complete",
        revision=1,
        draft="成本持续下降[来源1]。",
        sources=[source],
        claims=[{
            "claim_id": "section_1:v1:c1",
            "statement": "成本持续下降",
            "draft_quote": "成本持续下降",
            "assessment": "supported",
            "evidence": [{
                "source_number": 1,
                "quote": "成本持续下降。",
                "evidence_id": evidence_key(source),
            }],
        }],
    )


def _blueprint(section: SectionRecord) -> EditorialBlueprint:
    evidence_id = section.claims[0].evidence[0].evidence_id
    return EditorialBlueprint(
        verdict="ready",
        report_title="成本研究",
        thesis="成本改善经营弹性。",
        audience="外部读者",
        style_rules=["简洁"],
        section_plans=[{
            "source_section_id": section.section_id,
            "title": "成本趋势",
            "purpose": "解释成本变化",
            "claim_ids": [section.claims[0].claim_id],
            "evidence_ids": [evidence_id],
            "target_chars": 400,
        }],
    )


def _state(section: SectionRecord, blueprint: EditorialBlueprint) -> dict:
    return {
        "research_question": "成本变化意味着什么？",
        "sections": [section.model_dump(mode="json")],
        "report_review": ReportReview(verdict="pass").model_dump(mode="json"),
        "editorial_blueprint": blueprint.model_dump(mode="json"),
        "editorial_sections": [],
        "editorial_active_index": 0,
        "editorial_candidate": None,
        "editorial_compression": {},
        "editorial_warnings": [],
        "token_budget_used": 0,
        "model_calls": 0,
        "usage_unknown_calls": 0,
    }


@pytest.mark.asyncio
async def test_oversized_candidate_checkpoints_then_stalled_compression_falls_back() -> None:
    section = _section()
    blueprint = _blueprint(section)
    stable, _ = stable_editorial_sections([section])
    token = stable[0]["draft"].split("成本持续下降", 1)[1].split("。", 1)[0]
    oversized = EditedSectionArtifact(
        section={
            "title": "成本趋势",
            "body": "重复分析" * 350 + token + "。",
            "source_section_ids": ["section_1"],
            "claim_ids": ["section_1:v1:c1"],
        },
        summary="成本下降。",
        handoff="后文讨论影响。",
    )
    calls = []

    async def model(_system, prompt, schema=None, *, validator=None, context=None):
        payload = json.loads(prompt)
        calls.append(payload)
        value = oversized
        return validator(value), {"tokens": 10, "unknown": 0, "attempts": 1}

    runner = AgentRunner(lambda _spec: model)
    state = _state(section, blueprint)
    first = await edit_report_section(state, runner=runner)

    assert first["section_step"] == "chief_compress_section"
    assert first["editorial_candidate"]["section"]["body"] == oversized.section.body
    assert first["editorial_compression"]["attempts"] == 0
    state.update(first)

    second = await compress_report_section(state, runner=runner)

    assert len(calls) == 2  # One edit plus one stagnant compression; no third paid retry.
    assert calls[0]["phase"] == "section"
    assert "blueprint" not in calls[0]
    assert calls[1]["phase"] == "compress"
    assert calls[1]["required_length"]["maximum_characters"] < editorial_visible_length(
        oversized.section.body
    )
    assert second["section_step"] == "chief_write_framing"
    assert second["editorial_sections"][0]["section"]["body"] == stable[0]["draft"]
    assert "回退到审校通过的原章节正文" in second["editorial_warnings"][0]


@pytest.mark.asyncio
async def test_invalid_compression_output_falls_back_and_accounts_for_cost() -> None:
    section = _section()
    blueprint = _blueprint(section)
    stable, _ = stable_editorial_sections([section])
    token = stable[0]["draft"].split("成本持续下降", 1)[1].split("。", 1)[0]
    oversized = EditedSectionArtifact(
        section={
            "title": "成本趋势",
            "body": "重复分析" * 350 + token + "。",
            "source_section_ids": ["section_1"],
            "claim_ids": ["section_1:v1:c1"],
        },
        summary="成本下降。",
    )

    async def model(_system, _prompt, schema=None, *, validator=None, context=None):
        return validator(oversized), {"tokens": 10, "unknown": 0, "attempts": 1}

    state = _state(section, blueprint)
    first = await edit_report_section(state, runner=AgentRunner(lambda _spec: model))
    state.update(first)

    class InvalidOutputRunner:
        async def run(self, *_args, **_kwargs):
            raise ModelOutputError(
                "invalid compression",
                record={
                    "schema_status": "failed",
                    "business_status": "not_checked",
                    "finish_reason": None,
                },
                cost={"tokens": 7, "unknown": 0, "attempts": 1},
            )

    result = await compress_report_section(state, runner=InvalidOutputRunner())

    assert result["section_step"] == "chief_write_framing"
    assert result["editorial_sections"][0]["section"]["body"] == stable[0]["draft"]
    assert result["token_budget_used"] == 17
    assert result["model_calls"] == 2


def test_visible_length_does_not_charge_internal_evidence_id_width() -> None:
    token = "[[evidence:" + "a" * 64 + "]]"
    assert editorial_visible_length("结论" + token) == len("结论[来源]")


def test_blueprint_capacity_is_normalized_before_section_editing() -> None:
    section = _section()
    blueprint = _blueprint(section)
    normalized = validate_editorial_blueprint(
        blueprint,
        [section],
        ReportReview(verdict="pass"),
        {section.claims[0].evidence[0].evidence_id},
    )
    plan = normalized.section_plans[0]
    lower, upper = editorial_length_bounds(plan, section.draft)
    assert plan.target_chars == 630
    assert lower < upper
