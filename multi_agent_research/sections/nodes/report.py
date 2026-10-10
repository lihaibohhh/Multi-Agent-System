"""Whole-report semantic review node."""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from ...agents.bootstrap import agent_context, agent_runner
from ...agents.registry import agent_registry
from ..artifacts import dependency_issues
from ..models import SectionRecord
from ..request_factory import report_review_request
from ..transitions import usage_delta


async def review_report(
    state: dict,
    config: RunnableConfig | None = None,
    *,
    runner=agent_runner,
) -> dict:
    sections = [SectionRecord.model_validate(raw) for raw in state["sections"]]
    stale = dependency_issues(sections)
    if stale:
        raise ValueError("; ".join(stale))
    if any(section.status not in {"complete", "limited"} for section in sections):
        raise ValueError("cannot review unfinished or stale chapters")
    result = await runner.run(
        agent_registry.report_reviewer,
        report_review_request(state, sections),
        context=agent_context(config),
    )
    if result.output is None:
        raise RuntimeError("ReportReviewerAgent 未返回全篇审校结果")
    return {
        "report_review": result.output.model_dump(mode="json"),
        "section_step": "chief_edit",
        **usage_delta(state, result.usage),
    }


__all__ = ["review_report"]
