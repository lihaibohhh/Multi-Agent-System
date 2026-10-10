"""Runtime-managed Agent for checkpointed, evidence-safe report editing."""

from __future__ import annotations

import hashlib
import json
from functools import partial

from .context import AgentContext
from .contracts import ChiefEditorRequest, ModelCall
from .runtime import AgentTurnResult
from .spec import AgentSpec
from ..sections.editorial import (
    editorial_framing_view,
    editorial_planning_view,
    editorial_section_brief,
    editorial_visible_length,
    evidence_tokens,
)
from ..sections.models import (
    ChiefEditorStepResult,
    EditedSectionArtifact,
    EditorialBlueprint,
    EditorialFraming,
)
from ..sections.validation import (
    validate_edited_section_artifact,
    validate_editorial_blueprint,
    validate_editorial_framing,
)


_BASE_PROMPT = (
    "你是研究报告的 ChiefEditorAgent。所有事实必须来自输入中已审校并完成 Claim 绑定的章节；"
    "不得新增事实、数字、Claim 或 Evidence，不得把 uncertain/disputed 结论写成确定事实，"
    "不得删除重要限制。引用只能原样使用 [[evidence:64位ID]]，禁止生成 [来源N] 或参考来源列表。"
    "只完成本次 phase 指定的工作，并只返回该阶段要求的 JSON 对象。"
)

CHIEF_EDITOR_PLAN_PROMPT = _BASE_PROMPT + (
    "phase=plan：先审视全篇论点、受众、章节职责、顺序、术语、重复、冲突和衔接，"
    "产出一份全篇共享 EditorialBlueprint。section_plans 必须恰好覆盖输入章节一次；"
    "claim_ids/evidence_ids 只能从对应章节选择。逐项处理 report_review issue；无法编辑解决的"
    "问题保留为限制并将 verdict 设为 limited。此阶段不写报告正文。"
)

CHIEF_EDITOR_SECTION_PROMPT = _BASE_PROMPT + (
    "phase=section：只编辑 target_chapter，严格执行共享蓝图中的标题、职责、术语、篇幅和前后衔接。"
    "不得复制其他章节形成重复段落；source_section_ids 只能包含当前来源章节。"
    "summary 要概括本章实际交付的关键发现，handoff 说明下一章应如何承接。"
)

CHIEF_EDITOR_COMPRESSION_PROMPT = _BASE_PROMPT + (
    "phase=compress：将输入的单章候选稿压缩到明确给出的字符区间。必须保留 required_claim_ids、"
    "required_evidence_ids、关键数字、限定条件和反证，不得用删除限制或证据来换取篇幅。"
    "只返回完整压缩稿，不解释压缩过程。summary 与 handoff 同步精简。"
)

CHIEF_EDITOR_FRAMING_PROMPT = _BASE_PROMPT + (
    "phase=framing：所有正文已定稿。只依据蓝图和各章 summary/handoff 写执行摘要与结论，"
    "不要再次输出章节正文。摘要回答研究问题、交代最重要证据与限制；结论综合含义和边界，"
    "避免逐章机械复述。"
)

# Backward-compatible export name for callers that inspect the role prompt.
CHIEF_EDITOR_SYSTEM_PROMPT = CHIEF_EDITOR_PLAN_PROMPT


def _previous_candidate_view(candidate: dict | None) -> dict | None:
    if not candidate:
        return None
    sections = []
    for section in candidate.get("sections", []):
        body = str(section.get("body", ""))
        sections.append({
            "title": section.get("title", ""),
            "source_section_ids": section.get("source_section_ids", []),
            "claim_ids": section.get("claim_ids", []),
            "body_excerpt": body[:500],
        })
    return {
        "verdict": candidate.get("verdict"),
        "report_title": candidate.get("report_title"),
        "sections": sections,
        "unresolved_issues": candidate.get("unresolved_issues", []),
    }


def _target_plan(request: ChiefEditorRequest):
    if request.blueprint is None or request.target_section_id is None:
        raise ValueError("section phase requires blueprint and target_section_id")
    return next(
        plan
        for plan in request.blueprint.section_plans
        if plan.source_section_id == request.target_section_id
    )


class ChiefEditorAgent:
    """One Agent identity executing plan, per-section edit and framing phases."""

    spec = AgentSpec(
        name="chief_editor",
        description="以共享编辑蓝图逐章生成连贯、可追溯的完整研究报告",
        model_ref="section_model",
        input_type=ChiefEditorRequest,
        output_type=ChiefEditorStepResult,
        version="2",
        max_turns=1,
        timeout_seconds=600,
    )

    async def run_turn(
        self,
        request: ChiefEditorRequest,
        *,
        context: AgentContext,
        call_model: ModelCall,
    ) -> AgentTurnResult[ChiefEditorStepResult]:
        if request.phase == "plan":
            system = CHIEF_EDITOR_PLAN_PROMPT
            output_schema = EditorialBlueprint
            validate_phase_output = partial(
                validate_editorial_blueprint,
                sections=list(request.sections),
                report_review=request.report_review,
                known_evidence_ids=set(request.evidence_ids),
            )
            payload = {
                "phase": "plan",
                "research_question": request.research_question,
                "coordination_context": request.coordination_context,
                "chapters": editorial_planning_view(request.stable_sections),
                "report_review": request.report_review.model_dump(mode="json"),
                "previous_candidate": _previous_candidate_view(request.previous_candidate),
            }
        elif request.phase == "section":
            system = CHIEF_EDITOR_SECTION_PROMPT
            output_schema = EditedSectionArtifact
            validate_phase_output = partial(
                validate_edited_section_artifact,
                plan=_target_plan(request),
            )
            target = next(
                section for section in request.stable_sections
                if section["section_id"] == request.target_section_id
            )
            previous_section = None
            if request.previous_candidate:
                previous_section = next((
                    section for section in request.previous_candidate.get("sections", [])
                    if request.target_section_id in section.get("source_section_ids", [])
                ), None)
            payload = {
                "phase": "section",
                "research_question": request.research_question,
                "editorial_brief": editorial_section_brief(
                    request.blueprint,
                    request.report_review,
                    request.target_section_id,
                ),
                "target_chapter": target,
                "required_length": {
                    "measurement": "将内部 [[evidence:...]] 按一个 [来源] 标记计数",
                    "minimum_characters": request.target_min_chars,
                    "maximum_characters": request.target_max_chars,
                },
                "prior_editor_handoffs": editorial_framing_view(request.edited_sections),
                "previous_candidate_section": previous_section,
            }
        elif request.phase == "compress":
            if (
                request.blueprint is None
                or request.section_candidate is None
                or request.target_min_chars is None
                or request.target_max_chars is None
            ):
                raise ValueError("compress phase requires blueprint, candidate and length bounds")
            system = CHIEF_EDITOR_COMPRESSION_PROMPT
            output_schema = EditedSectionArtifact
            validate_phase_output = partial(
                validate_edited_section_artifact,
                plan=_target_plan(request),
                required_claim_ids=set(request.section_candidate.section.claim_ids),
                required_evidence_ids=evidence_tokens(request.section_candidate.section.body),
            )
            candidate = request.section_candidate
            payload = {
                "phase": "compress",
                "research_question": request.research_question,
                "editorial_brief": editorial_section_brief(
                    request.blueprint,
                    request.report_review,
                    request.target_section_id,
                ),
                "required_length": {
                    "measurement": "将内部 [[evidence:...]] 按一个 [来源] 标记计数",
                    "minimum_characters": request.target_min_chars,
                    "maximum_characters": request.target_max_chars,
                    "current_characters": editorial_visible_length(candidate.section.body),
                },
                "required_claim_ids": candidate.section.claim_ids,
                "required_evidence_ids": sorted(evidence_tokens(candidate.section.body)),
                "candidate": candidate.model_dump(mode="json"),
            }
        else:
            if request.blueprint is None:
                raise ValueError("framing phase requires blueprint")
            system = CHIEF_EDITOR_FRAMING_PROMPT
            output_schema = EditorialFraming
            validate_phase_output = partial(
                validate_editorial_framing,
                known_evidence_ids=set(request.evidence_ids),
            )
            payload = {
                "phase": "framing",
                "research_question": request.research_question,
                "blueprint": request.blueprint.model_dump(mode="json"),
                "edited_chapters": editorial_framing_view(request.edited_sections),
            }

        fingerprint = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        saved = context.local_state
        if saved.get("fingerprint") == fingerprint and saved.get("completed"):
            restored = ChiefEditorStepResult.model_validate(saved.get("result"))
            return AgentTurnResult(status="completed", output=restored)

        phase_result, _ = await call_model(
            system,
            json.dumps(payload, ensure_ascii=False),
            output_schema,
            validator=validate_phase_output,
            context={
                "agent": self.spec.name,
                "agent_version": self.spec.version,
                "agent_run_id": context.agent_run_id,
                "editorial_phase": request.phase,
                "section_id": request.target_section_id,
                "single_attempt": request.phase == "compress",
            },
        )
        result = ChiefEditorStepResult(
            phase=request.phase,
            **{
                {
                    "plan": "blueprint",
                    "section": "section_artifact",
                    "compress": "section_artifact",
                    "framing": "framing",
                }[request.phase]: phase_result
            },
        )
        return AgentTurnResult(
            status="completed",
            output=result,
            state_updates={
                "fingerprint": fingerprint,
                "completed": True,
                "result": result.model_dump(mode="json"),
            },
            handoff={
                "phase": request.phase,
                "section_id": request.target_section_id,
            },
        )


__all__ = [
    "CHIEF_EDITOR_FRAMING_PROMPT",
    "CHIEF_EDITOR_COMPRESSION_PROMPT",
    "CHIEF_EDITOR_PLAN_PROMPT",
    "CHIEF_EDITOR_SECTION_PROMPT",
    "CHIEF_EDITOR_SYSTEM_PROMPT",
    "ChiefEditorAgent",
]
