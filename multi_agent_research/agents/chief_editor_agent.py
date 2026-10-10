"""Runtime-managed Agent for evidence-safe whole-report editing."""

from __future__ import annotations

import hashlib
import json

from .context import AgentContext
from .contracts import ChiefEditorRequest, ModelCall
from .runtime import AgentTurnResult
from .spec import AgentSpec
from ..sections.models import ChiefEditorResult
from ..sections.validation import validate_chief_editor_result


CHIEF_EDITOR_SYSTEM_PROMPT = (
    "你是研究报告主编。把已经完成审校和 Claim 绑定的章节编辑成一篇自然连贯、"
    "符合人类阅读习惯的完整报告。返回 ChiefEditorResult JSON。可以重排论述、合并重复、"
    "统一术语、补写非事实性的过渡句，并重写执行摘要和结论；不得新增输入中不存在的事实、"
    "数字、Claim 或 Evidence，不得把 uncertain/disputed 结论写成确定事实，不得删除重要限制。"
    "正文中的引用只能原样使用 [[evidence:64位ID]]，禁止生成 [来源N] 或参考来源列表。"
    "每个输出章节必须列出其 source_section_ids 和使用的 claim_ids；used_claim_ids 汇总全文。"
    "对每个全篇审校 issue 按零基 issue_index 恰好返回一个 resolution。无法通过编辑解决的问题"
    "必须 action=preserved_as_limitation、写入 unresolved_issues，并将 verdict 设为 limited。"
)


class ChiefEditorAgent:
    """Edit reviewed chapter artifacts without owning retrieval or graph routing."""

    spec = AgentSpec(
        name="chief_editor",
        description="将已审校章节编辑为连贯、可追溯的完整研究报告",
        model_ref="section_model",
        input_type=ChiefEditorRequest,
        output_type=ChiefEditorResult,
        version="1",
        max_turns=1,
        timeout_seconds=600,
    )

    async def run_turn(
        self,
        request: ChiefEditorRequest,
        *,
        context: AgentContext,
        call_model: ModelCall,
    ) -> AgentTurnResult[ChiefEditorResult]:
        payload = {
            "research_question": request.research_question,
            "coordination_context": request.coordination_context,
            "chapters": request.stable_sections,
            "report_review": request.report_review.model_dump(mode="json"),
            "previous_candidate": request.previous_candidate,
            "allowed_evidence_ids": sorted(request.evidence_ids),
        }
        fingerprint = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        saved = context.local_state
        if saved.get("fingerprint") == fingerprint and saved.get("completed"):
            restored = ChiefEditorResult.model_validate(saved.get("result"))
            return AgentTurnResult(
                status="completed",
                output=restored,
                handoff={
                    "verdict": restored.verdict,
                    "used_claim_ids": restored.used_claim_ids,
                    "unresolved_count": len(restored.unresolved_issues),
                },
            )

        prompt = json.dumps(payload, ensure_ascii=False)
        result, _ = await call_model(
            CHIEF_EDITOR_SYSTEM_PROMPT,
            prompt,
            ChiefEditorResult,
            validator=lambda value: validate_chief_editor_result(
                value,
                list(request.sections),
                request.report_review,
                set(request.evidence_ids),
            ),
            context={
                "agent": self.spec.name,
                "agent_version": self.spec.version,
                "agent_run_id": context.agent_run_id,
                "section_ids": [section.section_id for section in request.sections],
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
                "verdict": result.verdict,
                "used_claim_ids": result.used_claim_ids,
                "unresolved_count": len(result.unresolved_issues),
            },
        )


__all__ = ["CHIEF_EDITOR_SYSTEM_PROMPT", "ChiefEditorAgent"]
