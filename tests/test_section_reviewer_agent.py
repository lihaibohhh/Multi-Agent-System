from __future__ import annotations

import pytest

from multi_agent_research.agents import AgentContext, AgentRunner
from multi_agent_research.agents.contracts import SectionReviewRequest
from multi_agent_research.agents.section_reviewer_agent import SectionReviewerAgent
from multi_agent_research.sections import workflow
from multi_agent_research.sections.models import SectionRecord, SectionReview


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

    lifecycle = []
    runner = AgentRunner(lambda spec: call_model, event_sink=lifecycle.append)
    result = await runner.run(
        SectionReviewerAgent(),
        SectionReviewRequest(
            section_id="section_1",
            section_context="本章：成本\n本章必须回答：成本优势能否持续\n",
            evidence_text="[来源1] 成本材料",
            draft="成本优势得到证据支持[来源1]。",
        ),
        context=AgentContext.create("run-review", section_id="section_1"),
    )
    review = result.output

    assert review is not None
    assert review.verdict == "pass"
    assert result.usage["tokens"] == 13
    assert result.turns == 1
    assert result.handoff == {"review": review.model_dump(mode="json")}
    assert SectionReviewerAgent.spec.model_ref == "section_model"
    assert SectionReviewerAgent.spec.allowed_tools == frozenset()
    assert captured["schema"] is SectionReview
    assert captured["context"]["agent"] == "section_reviewer"
    assert captured["context"]["agent_version"] == "2"
    assert captured["context"]["agent_run_id"] == lifecycle[0].agent_run_id
    assert captured["context"]["section_id"] == "section_1"
    assert "是否忽略反证" in captured["system"]
    assert "成本优势得到证据支持" in captured["prompt"]
    assert [event.event_type for event in lifecycle] == [
        "agent_started",
        "agent_turn_started",
        "agent_model_called",
        "agent_completed",
    ]
    assert all(event.section_id == "section_1" for event in lifecycle)


@pytest.mark.asyncio
async def test_section_reviewer_preserves_issues_and_downgrades_pass() -> None:
    async def call_model(system, prompt, schema=None, *, validator=None, context=None):
        value = SectionReview(
            verdict="pass",
            issues=["缺少成本反例"],
            search_queries=["公司 成本 反例"],
        )
        return validator(value), {"tokens": 1, "unknown": 0}

    result = await AgentRunner(lambda spec: call_model).run(
        SectionReviewerAgent(),
        SectionReviewRequest(
            section_id="section_1",
            section_context="本章：成本\n",
            evidence_text="[来源1] 正面材料",
            draft="成本优势明显[来源1]。",
        ),
        context=AgentContext.create("run-review", section_id="section_1"),
    )
    review = result.output

    assert review is not None
    assert review.verdict == "revise"
    assert review.issues == ["缺少成本反例"]
    assert review.search_queries == ["公司 成本 反例"]


@pytest.mark.asyncio
async def test_workflow_runs_section_reviewer_with_scoped_context(monkeypatch) -> None:
    lifecycle = []

    async def call_model(system, prompt, schema=None, *, validator=None, context=None):
        value = SectionReview(verdict="pass", summary="章节证据与结论一致")
        return validator(value), {"tokens": 17, "unknown": 0, "attempts": 1}

    source = {
        "query": "成本优势",
        "source": "knowledge",
        "content": "公司成本优势来自规模效应",
        "score": 0.9,
        "iteration": 0,
        "metadata": {"source": "cost.pdf", "page": 2, "chunk_id": "cost::2"},
    }
    section = SectionRecord(
        section_id="section_1",
        title="成本",
        question="公司的成本优势能否持续",
        status="drafted",
        revision=1,
        sources=[source],
        draft="公司的成本优势来自规模效应[来源1]。",
    )
    state = {
        "research_question": "分析公司竞争优势",
        "parent_context": None,
        "sections": [section.model_dump(mode="json")],
        "active_section": 0,
        "section_policy": {"max_search_rounds": 2, "max_revisions": 1},
    }
    monkeypatch.setattr(workflow, "call_model", call_model)
    monkeypatch.setattr(
        workflow,
        "agent_runner",
        AgentRunner(workflow.resolve_agent_model, event_sink=lifecycle.append),
    )

    result = await workflow.review_section(
        state,
        config={"configurable": {"thread_id": "run-workflow-review"}},
    )

    assert result["section_step"] == "claims"
    assert result["sections"][0]["review"]["verdict"] == "pass"
    assert result["token_budget_used"] == 17
    assert result["model_calls"] == 1
    assert all(event.run_id == "run-workflow-review" for event in lifecycle)
    assert all(event.section_id == "section_1" for event in lifecycle)
