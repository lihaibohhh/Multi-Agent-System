from __future__ import annotations

import json

import pytest

from multi_agent_research.agents.contracts import ReportReviewRequest
from multi_agent_research.agents.report_reviewer_agent import ReportReviewerAgent
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

    result = await ReportReviewerAgent().run(
        ReportReviewRequest(
            research_question="公司的竞争优势是否持续？",
            sections=(_chapter(),),
        ),
        call_model=call_model,
    )

    assert result.review.verdict == "pass"
    assert result.cost["tokens"] == 21
    assert captured["schema"] is ReportReview
    assert captured["context"] == {
        "agent": "report_reviewer",
        "section_ids": ["section_1"],
    }
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

    result = await ReportReviewerAgent().run(
        ReportReviewRequest(
            research_question="公司的竞争优势是否持续？",
            sections=(_chapter(),),
        ),
        call_model=call_model,
    )

    assert result.review.verdict == "revise"
    assert result.review.issues[0].detail == "两个结论使用了不同年份口径"
