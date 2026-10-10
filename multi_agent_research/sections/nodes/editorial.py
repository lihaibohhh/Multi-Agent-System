"""Whole-report editing and independent post-edit review nodes."""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from ...agents.bootstrap import agent_context, agent_runner, resumable_agent_context
from ...agents.registry import agent_registry
from ...coordination.briefing import current_editorial_context
from ..editorial import render_edited_report, stable_editorial_sections
from ..models import ChiefEditorResult, SectionRecord
from ..request_factory import chief_editor_request, report_review_request
from ..transitions import usage_delta


MAX_EDITORIAL_ATTEMPTS = 2


async def edit_report(
    state: dict,
    config: RunnableConfig | None = None,
    *,
    runner=agent_runner,
) -> dict:
    sections = [SectionRecord.model_validate(raw) for raw in state["sections"]]
    if any(section.status not in {"complete", "limited"} for section in sections):
        raise ValueError("cannot edit unfinished or stale chapters")
    if not state.get("report_review"):
        raise ValueError("whole-report review is required before editing")
    request, _ = chief_editor_request(
        state,
        sections,
        await current_editorial_context(),
    )
    result = await runner.run(
        agent_registry.chief_editor,
        request,
        context=await resumable_agent_context(
            agent_registry.chief_editor.spec.name,
            config,
        ),
    )
    if result.output is None:
        raise RuntimeError("ChiefEditorAgent 未返回编辑结果")
    return {
        "edited_report": result.output.model_dump(mode="json"),
        "editorial_attempts": int(state.get("editorial_attempts", 0)) + 1,
        "section_step": "edited_report_review",
        **usage_delta(state, result.usage),
    }


async def review_edited_report(
    state: dict,
    config: RunnableConfig | None = None,
    *,
    runner=agent_runner,
) -> dict:
    sections = [SectionRecord.model_validate(raw) for raw in state["sections"]]
    edited = ChiefEditorResult.model_validate(state["edited_report"])
    _, evidence_registry = stable_editorial_sections(sections)
    candidate = render_edited_report(
        edited,
        evidence_registry,
        limited=edited.verdict == "limited" or any(
            section.status == "limited" for section in sections
        ),
    )
    result = await runner.run(
        agent_registry.report_reviewer,
        report_review_request(state, sections, candidate_report=candidate),
        context=agent_context(config),
    )
    if result.output is None:
        raise RuntimeError("ReportReviewerAgent 未返回编辑后复审结果")
    attempts = int(state.get("editorial_attempts", 0))
    next_step = (
        "chief_edit"
        if result.output.verdict == "revise" and attempts < MAX_EDITORIAL_ATTEMPTS
        else "assemble"
    )
    return {
        "report_review": result.output.model_dump(mode="json"),
        "section_step": next_step,
        **usage_delta(state, result.usage),
    }


__all__ = ["MAX_EDITORIAL_ATTEMPTS", "edit_report", "review_edited_report"]
