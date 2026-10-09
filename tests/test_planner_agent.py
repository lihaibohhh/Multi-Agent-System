from __future__ import annotations

import pytest

from multi_agent_research.agents import AgentContext, AgentRunner
from multi_agent_research.agents.contracts import SectionPlanningRequest
from multi_agent_research.agents.planner_agent import PlannerAgent
from multi_agent_research.sections import workflow
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

    lifecycle = []
    runner = AgentRunner(lambda spec: call_model, event_sink=lifecycle.append)
    result = await runner.run(
        PlannerAgent(),
        SectionPlanningRequest(
            research_question="分析公司竞争优势",
            maximum_sections=3,
            parent_view='{"sections": []}',
            available_parent_section_ids=frozenset({"parent_1"}),
        ),
        context=AgentContext.create("run-planner"),
    )
    plan = result.output

    assert plan is not None
    assert plan.sections[0].parent_section_ids == ["parent_1"]
    assert result.usage["tokens"] == 7
    assert result.turns == 1
    assert result.handoff == {"plan": plan.model_dump(mode="json")}
    assert PlannerAgent.spec.model_ref == "section_model"
    assert PlannerAgent.spec.allowed_tools == frozenset()
    assert captured["schema"] is SectionPlan
    assert captured["context"]["agent"] == "planner"
    assert captured["context"]["agent_version"] == "2"
    assert captured["context"]["agent_run_id"] == lifecycle[0].agent_run_id
    assert "最多 3 章" in captured["prompt"]
    assert "专题研究编辑" in captured["system"]
    assert [event.event_type for event in lifecycle] == [
        "agent_started",
        "agent_turn_started",
        "agent_model_called",
        "agent_completed",
    ]
    assert all(event.run_id == "run-planner" for event in lifecycle)


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
        await AgentRunner(lambda spec: call_model).run(
            PlannerAgent(),
            SectionPlanningRequest(
                research_question="分析公司竞争优势",
                maximum_sections=2,
                parent_view="{}",
                available_parent_section_ids=frozenset({"parent_1"}),
            ),
            context=AgentContext.create("run-planner"),
        )


@pytest.mark.asyncio
async def test_workflow_runs_planner_with_langgraph_run_identity(monkeypatch) -> None:
    lifecycle = []

    async def call_model(system, prompt, schema=None, *, validator=None, context=None):
        value = SectionPlan(sections=[{
            "title": "市场",
            "question": "目标市场的竞争格局是什么",
        }])
        return validator(value), {"tokens": 11, "unknown": 0, "attempts": 1}

    monkeypatch.setattr(workflow, "call_model", call_model)
    monkeypatch.setattr(
        workflow,
        "agent_runner",
        AgentRunner(workflow.resolve_agent_model, event_sink=lifecycle.append),
    )

    result = await workflow.plan_sections(
        {"research_question": "分析目标市场"},
        config={"configurable": {"thread_id": "run-workflow-planner"}},
    )

    assert result["sections"][0]["section_id"] == "section_1"
    assert result["token_budget_used"] == 11
    assert result["model_calls"] == 1
    assert [event.event_type for event in lifecycle] == [
        "agent_started",
        "agent_turn_started",
        "agent_model_called",
        "agent_completed",
    ]
    assert all(event.run_id == "run-workflow-planner" for event in lifecycle)
