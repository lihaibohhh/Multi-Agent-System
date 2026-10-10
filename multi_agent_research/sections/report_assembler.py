"""Deterministic final report and partial-operation assembly."""

from __future__ import annotations

from .artifacts import dependency_issues
from .models import SectionRecord
from .models import ChiefEditorResult
from .operations import FINISHED, operation_summary
from .editorial import render_edited_report, stable_editorial_sections


def assemble_sections(state: dict) -> dict:
    operation = (state.get("parent_context") or {}).get("section_operation")
    unfinished = any(section["status"] not in FINISHED for section in state["sections"])
    if operation and (operation["mode"] == "supplement" or unfinished):
        return operation_summary(state)

    sections = [SectionRecord.model_validate(raw) for raw in state["sections"]]
    review = state.get("report_review")
    edited = state.get("edited_report")
    if not review or not edited or dependency_issues(sections):
        raise ValueError(
            "report requires semantic editing, consistency review and current dependencies"
        )

    limited = any(section.status == "limited" for section in sections)
    limited = limited or review["verdict"] != "pass"
    edited_result = ChiefEditorResult.model_validate(edited)
    limited = limited or edited_result.verdict == "limited"
    _, evidence_registry = stable_editorial_sections(sections)
    report = render_edited_report(
        edited_result,
        evidence_registry,
        limited=limited,
    )
    return {
        "final_report": report,
        "writer_status": "complete",
        "section_step": "done",
        "report_quality": "limited" if limited else "reviewed",
    }


__all__ = ["assemble_sections"]
