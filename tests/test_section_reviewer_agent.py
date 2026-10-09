from __future__ import annotations

import pytest

from multi_agent_research.agents.contracts import SectionReviewRequest
from multi_agent_research.agents.section_reviewer_agent import SectionReviewerAgent
from multi_agent_research.sections.models import SectionReview


@pytest.mark.asyncio
async def test_section_reviewer_owns_prompt_schema_and_usage() -> None:
    captured = {}

    async def call_model(system, prompt, schema=None, *, validator=None, context=None):
        captured.update(system=system, prompt=prompt, schema=schema, context=context)
        value = SectionReview(
            verdict="pass",
            summary="正文结论有来源支撑",
        )
        return validator(value), {"tokens": 13, "unknown": 0, "attempts": 1}

    result = await SectionReviewerAgent().run(
        SectionReviewRequest(
            section_id="section_1",
            section_context="本章：成本\n本章必须回答：成本优势能否持续\n",
            evidence_text="[来源1] 成本材料",
            draft="成本优势得到证据支持[来源1]。",
        ),
        call_model=call_model,
    )

    assert result.review.verdict == "pass"
    assert result.cost["tokens"] == 13
    assert captured["schema"] is SectionReview
    assert captured["context"] == {
        "agent": "section_reviewer",
        "section_id": "section_1",
    }
    assert "是否忽略反证" in captured["system"]
    assert "成本优势得到证据支持" in captured["prompt"]


@pytest.mark.asyncio
async def test_section_reviewer_preserves_issues_and_downgrades_pass() -> None:
    async def call_model(system, prompt, schema=None, *, validator=None, context=None):
        value = SectionReview(
            verdict="pass",
            issues=["缺少成本反例"],
            search_queries=["公司 成本 反例"],
        )
        return validator(value), {"tokens": 1, "unknown": 0}

    result = await SectionReviewerAgent().run(
        SectionReviewRequest(
            section_id="section_1",
            section_context="本章：成本\n",
            evidence_text="[来源1] 正面材料",
            draft="成本优势明显[来源1]。",
        ),
        call_model=call_model,
    )

    assert result.review.verdict == "revise"
    assert result.review.issues == ["缺少成本反例"]
    assert result.review.search_queries == ["公司 成本 反例"]
