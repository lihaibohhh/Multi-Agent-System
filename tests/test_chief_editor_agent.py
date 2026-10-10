from __future__ import annotations

import json

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
    EditedSectionArtifact,
    EditorialBlueprint,
    EditorialFraming,
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
async def test_chief_editor_runs_bounded_plan_section_and_framing_phases() -> None:
    section = _section()
    stable, registry = stable_editorial_sections([section])
    review = ReportReview(verdict="pass", summary="全篇一致")
    calls = []

    async def call_model(system, prompt, schema=None, *, validator=None, context=None):
        payload = json.loads(prompt)
        calls.append((payload["phase"], payload, schema, context))
        if payload["phase"] == "plan":
            value = EditorialBlueprint(**{
                    "verdict": "ready",
                    "report_title": "公司成本研究报告",
                    "thesis": "成本下降改善了经营弹性。",
                    "audience": "外部决策者",
                    "style_rules": ["先证据后结论"],
                    "section_plans": [{
                        "source_section_id": "section_1",
                        "title": "成本趋势及含义",
                        "purpose": "解释成本变化及其含义",
                        "claim_ids": ["section_1:v1:c1"],
                        "evidence_ids": list(registry),
                    }],
            })
        elif payload["phase"] == "section":
            value = EditedSectionArtifact(**{
                    "section": {
                        "title": "成本趋势及含义",
                        "body": stable[0]["draft"],
                        "source_section_ids": ["section_1"],
                        "claim_ids": ["section_1:v1:c1"],
                    },
                    "summary": "成本呈下降趋势。",
                    "handoff": "下一部分讨论经营含义。",
            })
        else:
            token = stable[0]["draft"].split("下降", 1)[1].split("。", 1)[0]
            value = EditorialFraming(**{
                    "executive_summary": f"成本呈下降趋势{token}。",
                    "conclusion": "现有证据支持成本下降这一判断。",
            })
        return validator(value), {"tokens": 13, "unknown": 0, "attempts": 1}

    runner = AgentRunner(lambda _spec: call_model)
    common = {
        "research_question": "公司的成本变化意味着什么？",
        "sections": (section,),
        "report_review": review,
        "coordination_context": "",
        "stable_sections": stable,
        "evidence_ids": frozenset(registry),
    }
    planned = await runner.run(
        ChiefEditorAgent(),
        ChiefEditorRequest(phase="plan", **common),
        context=AgentContext.create("run-chief-editor", section_id="__plan__"),
    )
    blueprint = planned.output.blueprint
    edited = await runner.run(
        ChiefEditorAgent(),
        ChiefEditorRequest(
            phase="section",
            blueprint=blueprint,
            target_section_id="section_1",
            **common,
        ),
        context=AgentContext.create("run-chief-editor", section_id="section_1"),
    )
    artifact = edited.output.section_artifact
    framed = await runner.run(
        ChiefEditorAgent(),
        ChiefEditorRequest(
            phase="framing",
            blueprint=blueprint,
            edited_sections=(artifact,),
            **common,
        ),
        context=AgentContext.create("run-chief-editor", section_id="__framing__"),
    )

    final = ChiefEditorResult(
        verdict=blueprint.verdict,
        report_title=blueprint.report_title,
        executive_summary=framed.output.framing.executive_summary,
        sections=[artifact.section],
        conclusion=framed.output.framing.conclusion,
        issue_resolutions=blueprint.issue_resolutions,
        used_claim_ids=artifact.section.claim_ids,
        unresolved_issues=blueprint.unresolved_issues,
    )
    validate_chief_editor_result(final, [section], review, set(registry))
    report = render_edited_report(final, registry, limited=False)
    assert [phase for phase, *_ in calls] == ["plan", "section", "framing"]
    assert [schema for _, _, schema, _ in calls] == [
        EditorialBlueprint,
        EditedSectionArtifact,
        EditorialFraming,
    ]
    assert calls[0][1]["chapters"][0].get("draft") is None
    assert calls[1][1]["target_chapter"]["draft"] == stable[0]["draft"]
    assert "blueprint" not in calls[1][1]
    assert calls[1][1]["editorial_brief"]["target_plan"]["source_section_id"] == "section_1"
    assert "edited_chapters" in calls[2][1]
    assert ChiefEditorAgent.spec.allowed_tools == frozenset()
    assert "成本呈下降趋势[来源1]" in report
    assert "成本报告 | p.3" in report
    assert "[[evidence:" not in report


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


def test_editorial_models_bound_each_checkpoint_artifact() -> None:
    blueprint = EditorialBlueprint(
        verdict="ready",
        report_title="标题",
        thesis="中心论点",
        audience="外部读者",
        style_rules=["清晰"],
        section_plans=[{
            "source_section_id": "section_1",
            "title": "章节",
            "purpose": "回答问题",
        }],
    )
    artifact = EditedSectionArtifact(
        section={
            "title": "章节",
            "body": "正文",
            "source_section_ids": ["section_1"],
        },
        summary="摘要",
    )
    assert blueprint.section_plans[0].target_chars == 2000
    assert artifact.section.source_section_ids == ["section_1"]
