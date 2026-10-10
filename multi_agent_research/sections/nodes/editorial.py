"""Checkpointed whole-report editing and independent post-edit review nodes."""

from __future__ import annotations

import hashlib

from langchain_core.runnables import RunnableConfig

from ...agents.bootstrap import agent_context, agent_runner, resumable_agent_context
from ...agents.registry import agent_registry
from ...coordination.briefing import current_editorial_context
from ..editorial import (
    editorial_length_bounds,
    editorial_visible_length,
    render_edited_report,
    stable_editorial_sections,
)
from ..models import (
    ChiefEditorResult,
    EditedSectionArtifact,
    EditorialBlueprint,
    SectionRecord,
)
from ..model_output import ModelOutputError
from ..request_factory import chief_editor_request, report_review_request
from ..transitions import usage_delta
from ..validation import validate_chief_editor_result


MAX_EDITORIAL_ATTEMPTS = 2
MAX_EDITORIAL_COMPRESSION_ATTEMPTS = 2


def _sections(state: dict) -> list[SectionRecord]:
    sections = [SectionRecord.model_validate(raw) for raw in state["sections"]]
    if any(section.status not in {"complete", "limited"} for section in sections):
        raise ValueError("cannot edit unfinished or stale chapters")
    if not state.get("report_review"):
        raise ValueError("whole-report review is required before editing")
    return sections


def _target_source(sections: list[SectionRecord], section_id: str) -> SectionRecord:
    return next(section for section in sections if section.section_id == section_id)


def _candidate_signature(artifact: EditedSectionArtifact) -> dict:
    body = artifact.section.body
    return {
        "sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "visible_chars": editorial_visible_length(body),
        "raw_chars": len(body),
    }


def _compression_stalled(previous: dict, current: dict) -> bool:
    return (
        previous.get("sha256") == current.get("sha256")
        or current["visible_chars"] >= round(previous["visible_chars"] * 0.95)
    )


def _recoverable_compression_output_failure(error: ModelOutputError) -> bool:
    """Distinguish unusable model output from transport or validator defects."""

    record = error.record or {}
    if record.get("business_status") == "internal_error":
        return False
    return (
        record.get("finish_reason") in {"length", "max_tokens"}
        or record.get("schema_status") == "failed"
        or record.get("business_status") == "failed"
    )


def _complete_section_edit(
    state: dict,
    blueprint: EditorialBlueprint,
    artifact: EditedSectionArtifact,
    *,
    usage: dict,
    warnings: list[str] | None = None,
) -> dict:
    artifacts = [
        EditedSectionArtifact.model_validate(item)
        for item in state.get("editorial_sections", [])
    ]
    artifacts.append(artifact)
    next_index = int(state.get("editorial_active_index", 0)) + 1
    return {
        "editorial_sections": [item.model_dump(mode="json") for item in artifacts],
        "editorial_active_index": next_index,
        "editorial_candidate": None,
        "editorial_compression": {},
        "editorial_warnings": [
            *state.get("editorial_warnings", []),
            *(warnings or []),
        ],
        "section_step": (
            "chief_edit_section"
            if next_index < len(blueprint.section_plans)
            else "chief_write_framing"
        ),
        **usage_delta(state, usage),
    }


def _source_fallback(
    sections: list[SectionRecord],
    blueprint: EditorialBlueprint,
    index: int,
    candidate: EditedSectionArtifact,
) -> EditedSectionArtifact:
    plan = blueprint.section_plans[index]
    source = _target_source(sections, plan.source_section_id)
    stable, _ = stable_editorial_sections([source])
    return EditedSectionArtifact(
        section={
            "title": plan.title,
            "body": stable[0]["draft"],
            "source_section_ids": [source.section_id],
            "claim_ids": plan.claim_ids,
        },
        summary=candidate.summary,
        handoff=plan.transition_out or candidate.handoff,
    )


async def edit_report(
    state: dict,
    config: RunnableConfig | None = None,
    *,
    runner=agent_runner,
) -> dict:
    """Stage 1: create the shared whole-report editorial blueprint."""

    sections = _sections(state)
    request, _ = chief_editor_request(
        state,
        sections,
        await current_editorial_context(),
        phase="plan",
    )
    result = await runner.run(
        agent_registry.chief_editor,
        request,
        context=await resumable_agent_context(
            agent_registry.chief_editor.spec.name,
            config,
            section_id="__editorial_plan__",
        ),
    )
    if result.output is None or result.output.blueprint is None:
        raise RuntimeError("ChiefEditorAgent 未返回编辑蓝图")
    return {
        "editorial_blueprint": result.output.blueprint.model_dump(mode="json"),
        "editorial_sections": [],
        "editorial_active_index": 0,
        "editorial_candidate": None,
        "editorial_compression": {},
        "editorial_warnings": [],
        "section_step": "chief_edit_section",
        **usage_delta(state, result.usage),
    }


async def edit_report_section(
    state: dict,
    config: RunnableConfig | None = None,
    *,
    runner=agent_runner,
) -> dict:
    """Stage 2: edit exactly one chapter under the persisted shared blueprint."""

    sections = _sections(state)
    blueprint = EditorialBlueprint.model_validate(state["editorial_blueprint"])
    index = int(state.get("editorial_active_index", 0))
    if not 0 <= index < len(blueprint.section_plans):
        raise ValueError("editorial_active_index 超出编辑蓝图范围")
    target_id = blueprint.section_plans[index].source_section_id
    source = _target_source(sections, target_id)
    lower, upper = editorial_length_bounds(blueprint.section_plans[index], source.draft)
    request, _ = chief_editor_request(
        state,
        sections,
        "",
        phase="section",
        target_section_id=target_id,
        target_min_chars=lower,
        target_max_chars=upper,
    )
    result = await runner.run(
        agent_registry.chief_editor,
        request,
        context=await resumable_agent_context(
            agent_registry.chief_editor.spec.name,
            config,
            section_id=target_id,
        ),
    )
    if result.output is None or result.output.section_artifact is None:
        raise RuntimeError("ChiefEditorAgent 未返回逐章编辑结果")
    candidate = result.output.section_artifact
    signature = _candidate_signature(candidate)
    if signature["visible_chars"] <= upper:
        return _complete_section_edit(
            state,
            blueprint,
            candidate,
            usage=result.usage,
        )
    return {
        "editorial_candidate": candidate.model_dump(mode="json"),
        "editorial_compression": {
            "section_id": target_id,
            "minimum_chars": lower,
            "maximum_chars": upper,
            "attempts": 0,
            "history": [signature],
        },
        "section_step": "chief_compress_section",
        **usage_delta(state, result.usage),
    }


async def compress_report_section(
    state: dict,
    config: RunnableConfig | None = None,
    *,
    runner=agent_runner,
) -> dict:
    """Compress one persisted candidate with explicit bounds and progress fencing."""

    sections = _sections(state)
    blueprint = EditorialBlueprint.model_validate(state["editorial_blueprint"])
    index = int(state.get("editorial_active_index", 0))
    plan = blueprint.section_plans[index]
    work = dict(state.get("editorial_compression") or {})
    candidate = EditedSectionArtifact.model_validate(state["editorial_candidate"])
    if work.get("section_id") != plan.source_section_id:
        raise ValueError("压缩状态与当前编辑章节不一致")
    request, _ = chief_editor_request(
        state,
        sections,
        "",
        phase="compress",
        target_section_id=plan.source_section_id,
        section_candidate=candidate,
        target_min_chars=int(work["minimum_chars"]),
        target_max_chars=int(work["maximum_chars"]),
    )
    try:
        result = await runner.run(
            agent_registry.chief_editor,
            request,
            context=await resumable_agent_context(
                agent_registry.chief_editor.spec.name,
                config,
                section_id=plan.source_section_id,
            ),
        )
    except ModelOutputError as exc:
        if not _recoverable_compression_output_failure(exc):
            raise
        fallback = _source_fallback(sections, blueprint, index, candidate)
        warning = (
            f"章节 {plan.source_section_id} 的压缩输出未通过结构或保真校验；"
            "已回退到审校通过的原章节正文并保留稳定引用。"
        )
        return _complete_section_edit(
            state,
            blueprint,
            fallback,
            usage=exc.cost or {"tokens": 0, "unknown": 1, "attempts": 1},
            warnings=[warning],
        )
    if result.output is None or result.output.section_artifact is None:
        raise RuntimeError("ChiefEditorAgent 未返回章节压缩结果")
    compressed = result.output.section_artifact
    signature = _candidate_signature(compressed)
    attempts = int(work.get("attempts", 0)) + 1
    history = [*work.get("history", []), signature]
    if signature["visible_chars"] <= int(work["maximum_chars"]):
        return _complete_section_edit(
            state,
            blueprint,
            compressed,
            usage=result.usage,
        )

    stalled = _compression_stalled(history[-2], history[-1])
    if stalled or attempts >= MAX_EDITORIAL_COMPRESSION_ATTEMPTS:
        fallback = _source_fallback(sections, blueprint, index, compressed)
        reason = "连续压缩没有实质进展" if stalled else "压缩次数已达上限"
        warning = (
            f"章节 {plan.source_section_id} 的主编候选稿超出动态篇幅上限，{reason}；"
            "已回退到审校通过的原章节正文并保留稳定引用。"
        )
        return _complete_section_edit(
            state,
            blueprint,
            fallback,
            usage=result.usage,
            warnings=[warning],
        )
    return {
        "editorial_candidate": compressed.model_dump(mode="json"),
        "editorial_compression": {
            **work,
            "attempts": attempts,
            "history": history,
        },
        "section_step": "chief_compress_section",
        **usage_delta(state, result.usage),
    }


async def write_report_framing(
    state: dict,
    config: RunnableConfig | None = None,
    *,
    runner=agent_runner,
) -> dict:
    """Stage 3: write summary/conclusion and deterministically assemble final JSON."""

    sections = _sections(state)
    blueprint = EditorialBlueprint.model_validate(state["editorial_blueprint"])
    artifacts = [
        EditedSectionArtifact.model_validate(item)
        for item in state.get("editorial_sections", [])
    ]
    if len(artifacts) != len(blueprint.section_plans):
        raise ValueError("全部章节完成编辑后才能生成摘要与结论")
    request, evidence_registry = chief_editor_request(
        state,
        sections,
        "",
        phase="framing",
    )
    result = await runner.run(
        agent_registry.chief_editor,
        request,
        context=await resumable_agent_context(
            agent_registry.chief_editor.spec.name,
            config,
            section_id="__editorial_framing__",
        ),
    )
    if result.output is None or result.output.framing is None:
        raise RuntimeError("ChiefEditorAgent 未返回摘要与结论")

    used_claim_ids = list(dict.fromkeys(
        claim_id
        for artifact in artifacts
        for claim_id in artifact.section.claim_ids
    ))
    editorial_warnings = list(state.get("editorial_warnings", []))
    edited = ChiefEditorResult(
        verdict="limited" if editorial_warnings else blueprint.verdict,
        report_title=blueprint.report_title,
        executive_summary=result.output.framing.executive_summary,
        sections=[artifact.section for artifact in artifacts],
        conclusion=result.output.framing.conclusion,
        issue_resolutions=blueprint.issue_resolutions,
        used_claim_ids=used_claim_ids,
        unresolved_issues=[*blueprint.unresolved_issues, *editorial_warnings],
    )
    edited = validate_chief_editor_result(
        edited,
        sections,
        request.report_review,
        set(evidence_registry),
    )
    return {
        "edited_report": edited.model_dump(mode="json"),
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


__all__ = [
    "MAX_EDITORIAL_ATTEMPTS",
    "MAX_EDITORIAL_COMPRESSION_ATTEMPTS",
    "compress_report_section",
    "edit_report",
    "edit_report_section",
    "review_edited_report",
    "write_report_framing",
]
