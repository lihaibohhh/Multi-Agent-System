from __future__ import annotations

import json

import pytest

from multi_agent_research.agents import AgentContext, AgentRunner
from multi_agent_research.agents.contracts import ReportReviewRequest
from multi_agent_research.agents.report_reviewer_agent import ReportReviewerAgent
from multi_agent_research.sections import workflow
from multi_agent_research.sections.models import ReportReview, SectionRecord


def _chapter(section_id: str = "section_1") -> SectionRecord:
    return SectionRecord(
        section_id=section_id,
        title="成本",
        question="公司的成本优势是否持续？",
        status="complete",
        draft="公司的单位成本下降，但长期趋势仍有不确定性。[来源1]",
        claims=[
            {
                "claim_id": f"{section_id}:v1:c1",
                "statement": "公司的单位成本下降",
                "draft_quote": "公司的单位成本下降",
                "assessment": "uncertain",
                "caveat": "长期趋势仍不确定",
            }
        ],
        limitations=["缺少更长时间序列"],
        sources=[{"content": "不应进入全篇审校视图的来源原文"}],
        results=[{"content": "不应进入全篇审校视图的检索结果"}],
    )


@pytest.mark.asyncio
async def test_report_reviewer_owns_prompt_schema_and_bounded_view() -> None:
    captured = {}

    async def call_model(system, prompt, schema=None, *, validator=None, context=None):
        captured.update(system=system, prompt=prompt, schema=schema, context=context)
        review = ReportReview(verdict="pass", summary="章节之间未发现口径冲突")
        return validator(review), {"tokens": 21, "unknown": 0, "attempts": 1}

    lifecycle = []
    runner = AgentRunner(lambda spec: call_model, event_sink=lifecycle.append)
    result = await runner.run(
        ReportReviewerAgent(),
        ReportReviewRequest(
            research_question="公司的竞争优势是否持续？",
            sections=(_chapter(),),
        ),
        context=AgentContext.create("run-report-review"),
    )
    review = result.output

    assert review is not None
    assert review.verdict == "pass"
    assert result.usage["tokens"] == 21
    assert result.turns == 1
    assert result.handoff == {"review": review.model_dump(mode="json")}
    assert ReportReviewerAgent.spec.model_ref == "section_model"
    assert ReportReviewerAgent.spec.allowed_tools == frozenset()
    assert captured["schema"] is ReportReview
    assert captured["context"]["agent"] == "report_reviewer"
    assert captured["context"]["agent_version"] == "2"
    assert captured["context"]["agent_run_id"] == lifecycle[0].agent_run_id
    assert captured["context"]["section_ids"] == ["section_1"]
    assert "相反结论" in captured["system"]
    prompt = captured["prompt"]
    assert "公司的竞争优势是否持续" in prompt
    assert "长期趋势仍不确定" in prompt
    assert "不应进入全篇审校视图" not in prompt
    view = json.loads(prompt.split("章节：", 1)[1])
    assert set(view[0]) == {
        "section_id",
        "question",
        "draft",
        "claims",
        "limitations",
    }
    assert [event.event_type for event in lifecycle] == [
        "agent_started",
        "agent_turn_started",
        "agent_model_called",
        "agent_completed",
    ]
    assert all(event.section_id is None for event in lifecycle)


@pytest.mark.asyncio
async def test_report_reviewer_preserves_issues_and_downgrades_pass() -> None:
    async def call_model(system, prompt, schema=None, *, validator=None, context=None):
        review = ReportReview(
            verdict="pass",
            summary="存在年份口径冲突",
            issues=[
                {
                    "kind": "scope",
                    "section_ids": ["section_1"],
                    "detail": "两个结论使用了不同年份口径",
                }
            ],
        )
        return validator(review), {"tokens": 3, "unknown": 0}

    result = await AgentRunner(lambda spec: call_model).run(
        ReportReviewerAgent(),
        ReportReviewRequest(
            research_question="公司的竞争优势是否持续？",
            sections=(_chapter(),),
        ),
        context=AgentContext.create("run-report-review"),
    )
    review = result.output

    assert review is not None
    assert review.verdict == "revise"
    assert review.issues[0].detail == "两个结论使用了不同年份口径"


@pytest.mark.asyncio
async def test_workflow_runs_report_reviewer_with_run_context(monkeypatch) -> None:
    lifecycle = []

    async def call_model(system, prompt, schema=None, *, validator=None, context=None):
        review = ReportReview(verdict="pass", summary="全篇章节口径一致")
        return validator(review), {"tokens": 23, "unknown": 0, "attempts": 1}

    monkeypatch.setattr(workflow, "call_model", call_model)
    monkeypatch.setattr(
        workflow,
        "agent_runner",
        AgentRunner(workflow.resolve_agent_model, event_sink=lifecycle.append),
    )

    result = await workflow.review_report(
        {
            "research_question": "公司的竞争优势是否持续？",
            "sections": [_chapter().model_dump(mode="json")],
        },
        config={"configurable": {"thread_id": "run-workflow-report-review"}},
    )

    assert result["section_step"] == "chief_edit"
    assert result["report_review"]["verdict"] == "pass"
    assert result["token_budget_used"] == 23
    assert result["model_calls"] == 1
    assert all(event.run_id == "run-workflow-report-review" for event in lifecycle)
    assert all(event.section_id is None for event in lifecycle)
