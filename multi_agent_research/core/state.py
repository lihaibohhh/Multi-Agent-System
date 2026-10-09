"""Current chapter-workflow state; legacy Agent-loop fields are intentionally absent."""

from __future__ import annotations
from typing import Literal, TypedDict


CURRENT_WORKFLOW_VERSION = 5


def format_parent_context(state: ResearchState, *, max_chars: int = 4_000) -> str:
    """Render the immutable parent snapshot as delimited reference material."""
    context = state.get("parent_context")
    if not context:
        return "（无父任务上下文）"
    excerpt = str(context.get("report_excerpt", ""))[:max_chars]
    references = str(context.get("reference_excerpt", ""))[:2_000]
    return (
        "<parent_run_context>\n"
        f"source_run_id: {context.get('source_run_id', '')}\n"
        f"source_question: {context.get('source_question', '')}\n"
        f"report_excerpt:\n{excerpt}\n"
        f"reference_excerpt:\n{references}\n"
        "</parent_run_context>"
    )


class ResearchState(TypedDict):
    # ── 全局只读（写入后不再修改）────────────────
    research_question: str          # 用户原始问题
    parent_context: dict | None     # 父 Run 产物快照，不包含父 Run 内部 State

    workflow_version: int           # Exact schema marker; older checkpoints are rejected.
    sections: list[dict]            # persisted artifacts, replaced by the serial coordinator
    active_section: int
    section_policy: dict
    section_step: str
    report_quality: str
    report_review: dict | None
    model_calls: int
    usage_unknown_calls: int

    iteration_count: int            # Completed chapter retrieval rounds.
    token_budget_used: int
    writer_status: Literal["not_started", "complete"]

    # ── 最终输出 ─────────────────────────────────
    final_report: str


# ─────────────────────────────────────────────
# 初始状态工厂（graph.invoke 的入口）
# ─────────────────────────────────────────────

def initial_state(question: str, parent_context: dict | None = None) -> ResearchState:
    return ResearchState(
        research_question=question,
        parent_context=parent_context,
        workflow_version=CURRENT_WORKFLOW_VERSION,
        sections=[],
        active_section=0,
        section_policy={},
        section_step="plan",
        report_quality="pending",
        report_review=None,
        model_calls=0,
        usage_unknown_calls=0,
        iteration_count=0,
        token_budget_used=0,
        writer_status="not_started",
        final_report="",
    )
