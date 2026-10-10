from __future__ import annotations

import pytest

from multi_agent_research.agents import AgentContext, AgentRunner
from multi_agent_research.agents.chief_editor_agent import ChiefEditorAgent
from multi_agent_research.agents.contracts import ChiefEditorRequest
from multi_agent_research.sections.editorial import (
    render_edited_report,
    stable_editorial_sections,
)
from multi_agent_research.sections.models import (
    ChiefEditorResult,
    ReportReview,
    SectionRecord,
)
from multi_agent_research.sections.rendering import evidence_key
from multi_agent_research.sections.validation import (
    BusinessValidationError,
    validate_chief_editor_result,
)


def _source(name: str) -> dict:
    return {
        "query": name,
        "source": "knowledge",
        "content": f"{name}的证据摘录",
        "score": 0.9,
        "metadata": {
            "source": f"{name}.pdf",
            "title": f"{name}报告",
            "page": 3,
            "chunk_id": f"{name}::3",
        },
    }


def _section() -> SectionRecord:
    source = _source("成本")
    return SectionRecord(
        section_id="section_1",
        title="成本变化",
        question="公司的成本变化意味着什么？",
        status="complete",
        revision=1,
        draft="公司的成本正在下降[来源1]。",
        sources=[source],
        claims=[{
            "claim_id": "section_1:v1:c1",
            "statement": "公司的成本正在下降",
            "draft_quote": "公司的成本正在下降",
            "assessment": "supported",
            "evidence": [{
                "source_number": 1,
                "quote": "成本的证据摘录",
                "evidence_id": evidence_key(source),
            }],
        }],
    )


@pytest.mark.asyncio
async def test_chief_editor_uses_stable_evidence_and_runtime_contract() -> None:
    section = _section()
    stable, registry = stable_editorial_sections([section])
    review = ReportReview(verdict="pass", summary="全篇一致")
    captured = {}

    async def call_model(system, prompt, schema=None, *, validator=None, context=None):
        captured.update(system=system, prompt=prompt, schema=schema, context=context)
        token = stable[0]["draft"].split("。", 1)[0].split("下降", 1)[1]
        result = ChiefEditorResult(
            verdict="ready",
            report_title="公司成本研究报告",
            executive_summary=f"成本呈下降趋势{token}。",
            sections=[{
                "title": "成本趋势及含义",
                "body": stable[0]["draft"],
                "source_section_ids": ["section_1"],
                "claim_ids": ["section_1:v1:c1"],
            }],
            conclusion="现有证据支持成本下降这一判断。",
            used_claim_ids=["section_1:v1:c1"],
        )
        return validator(result), {"tokens": 13, "unknown": 0, "attempts": 1}

    lifecycle = []
    result = await AgentRunner(
        lambda _spec: call_model,
        event_sink=lifecycle.append,
    ).run(
        ChiefEditorAgent(),
        ChiefEditorRequest(
            research_question="公司的成本变化意味着什么？",
            sections=(section,),
            report_review=review,
            coordination_context="",
            stable_sections=stable,
            evidence_ids=frozenset(registry),
        ),
        context=AgentContext.create("run-chief-editor"),
    )

    assert result.output is not None
    assert result.output.verdict == "ready"
    assert result.usage["tokens"] == 13
    assert ChiefEditorAgent.spec.allowed_tools == frozenset()
    assert captured["schema"] is ChiefEditorResult
    assert captured["context"]["agent"] == "chief_editor"
    report = render_edited_report(result.output, registry, limited=False)
    assert "成本呈下降趋势[来源1]" in report
    assert "成本报告 | p.3" in report
    assert "[[evidence:" not in report
    assert [event.event_type for event in lifecycle] == [
        "agent_started",
        "agent_turn_started",
        "agent_model_called",
        "agent_completed",
    ]


def test_chief_editor_rejects_unknown_evidence_and_missing_chapter() -> None:
    section = _section()
    review = ReportReview(verdict="pass")
    result = ChiefEditorResult(
        verdict="ready",
        report_title="错误报告",
        executive_summary="错误引用[[evidence:" + "f" * 64 + "]]。",
        sections=[{
            "title": "错误章节",
            "body": "错误引用[[evidence:" + "f" * 64 + "]]。",
            "source_section_ids": ["unknown"],
            "claim_ids": [],
        }],
        conclusion="结论",
    )
    with pytest.raises(BusinessValidationError) as exc:
        validate_chief_editor_result(
            result,
            [section],
            review,
            {evidence_key(section.sources[0])},
        )
    kinds = {issue.type for issue in exc.value.issues}
    assert {"unknown_section", "missing_section", "unknown_evidence"} <= kinds
