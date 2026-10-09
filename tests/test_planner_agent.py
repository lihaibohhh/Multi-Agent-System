from __future__ import annotations

import pytest

from multi_agent_research.agents.contracts import SectionPlanningRequest
from multi_agent_research.agents.planner_agent import PlannerAgent
from multi_agent_research.sections.models import SectionPlan
from multi_agent_research.sections.validation import BusinessValidationError


@pytest.mark.asyncio
async def test_planner_owns_prompt_schema_and_business_validation() -> None:
    captured = {}

    async def call_model(system, prompt, schema=None, *, validator=None, context=None):
        captured.update(system=system, prompt=prompt, schema=schema, context=context)
        value = SectionPlan(sections=[{
            "title": "成本",
            "question": "公司成本优势是否可以持续",
            "parent_section_ids": ["parent_1"],
        }])
        return validator(value), {"tokens": 7, "unknown": 0, "attempts": 1}

    plan, cost = await PlannerAgent().run(
        SectionPlanningRequest(
            research_question="分析公司竞争优势",
            maximum_sections=3,
            parent_view='{"sections": []}',
            available_parent_section_ids=frozenset({"parent_1"}),
        ),
        call_model=call_model,
    )

    assert plan.sections[0].parent_section_ids == ["parent_1"]
    assert cost["tokens"] == 7
    assert captured["schema"] is SectionPlan
    assert captured["context"] == {"agent": "planner"}
    assert "最多 3 章" in captured["prompt"]
    assert "专题研究编辑" in captured["system"]


@pytest.mark.asyncio
async def test_planner_rejects_unknown_parent_section() -> None:
    async def call_model(system, prompt, schema=None, *, validator=None, context=None):
        value = SectionPlan(sections=[{
            "title": "成本",
            "question": "公司成本优势是否可以持续",
            "parent_section_ids": ["unknown"],
        }])
        return validator(value), {"tokens": 1, "unknown": 0}

    with pytest.raises(BusinessValidationError):
        await PlannerAgent().run(
            SectionPlanningRequest(
                research_question="分析公司竞争优势",
                maximum_sections=2,
                parent_view="{}",
                available_parent_section_ids=frozenset({"parent_1"}),
            ),
            call_model=call_model,
        )
