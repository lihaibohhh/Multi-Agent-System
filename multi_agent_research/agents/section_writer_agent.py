"""Bounded Agent for drafting, deterministic validation, and local repair."""

from __future__ import annotations

import json
import hashlib

from .context import AgentContext
from .contracts import ModelCall, SectionWritingRequest, SectionWritingResult
from .runtime import AgentContractError, AgentTurnResult
from .spec import AgentSpec
from ..sections.rendering import evidence_key
from ..sections.validation import BusinessValidationError, validate_draft


SECTION_WRITER_SYSTEM_PROMPT = (
    "仅写当前章节正文，不写报告标题、章节总标题、执行摘要或参考来源列表；"
    "综合章可以写综合判断。使用 ### 作为小节标题。每个事实性判断标注 [来源N]。"
    "根据原文处理矛盾和不确定性；父报告/前章摘要不是证据。"
    "证据缺口必须明确披露，不能用推断补成事实。将正文控制在约1500个中文字符。"
)


class SectionWriterAgent:
    """Own chapter generation and a bounded deterministic repair loop."""

    spec = AgentSpec(
        name="section_writer",
        description="撰写、校验并有限修订单章正文",
        model_ref="section_model",
        input_type=SectionWritingRequest,
        output_type=SectionWritingResult,
        version="2",
        max_turns=3,
        timeout_seconds=420,
    )

    async def run_turn(
        self,
        request: SectionWritingRequest,
        *,
        context: AgentContext,
        call_model: ModelCall,
    ) -> AgentTurnResult[SectionWritingResult]:
        fingerprint = self._fingerprint(request)
        saved = context.local_state if context.local_state.get("fingerprint") == fingerprint else {}
        attempts = int(saved.get("attempts", 0))
        if saved.get("completed") and isinstance(saved.get("draft"), str):
            output = SectionWritingResult(draft=saved["draft"], attempts=attempts)
            return AgentTurnResult(
                status="completed",
                output=output,
                handoff={"revision": request.next_revision, "attempts": attempts},
            )
        if attempts >= self.spec.max_turns:
            raise AgentContractError(
                f"SectionWriterAgent 已用完 {self.spec.max_turns} 次写作校验机会"
            )

        prompt = (
            request.section_context
            + "可引用来源（只允许本次列表的编号）：\n"
            + request.evidence_text
            + f"\n审查缺口：{list(request.limitations)}\n"
        )
        if request.current_draft:
            old_numbering = [
                {
                    "old_number": index,
                    "metadata": source.get("metadata", {}),
                    "content": source["content"][:300],
                }
                for index, source in enumerate(request.previous_sources, 1)
            ]
            prompt += (
                f"\n待修改草稿（旧编号需按新来源表重新核对）：\n{request.current_draft}\n"
                f"旧编号对应资料：{json.dumps(old_numbering, ensure_ascii=False)}\n"
                f"修改要求：{request.review}\n"
            )
        if saved.get("draft"):
            prompt += (
                "\n上一次候选正文未通过确定性校验，请完整重写，不要只返回补丁：\n"
                f"{saved['draft']}\n"
                "校验问题："
                + json.dumps(saved.get("validation_issues", []), ensure_ascii=False)
                + "\n"
            )
        draft, _ = await call_model(
            SECTION_WRITER_SYSTEM_PROMPT,
            prompt,
            context={
                "agent": self.spec.name,
                "section_id": request.section_id,
                "revision": request.next_revision,
                "writer_attempt": attempts + 1,
                "single_attempt": True,
            },
        )
        attempts += 1
        try:
            validated = validate_draft(draft, list(request.sources), request.section_id)
        except BusinessValidationError as exc:
            return AgentTurnResult(
                status="continue",
                state_updates={
                    "fingerprint": fingerprint,
                    "attempts": attempts,
                    "draft": draft,
                    "validation_issues": exc.details(),
                },
                unresolved=tuple(issue["message"] for issue in exc.details()),
                reason="候选正文未通过引用和正文完整性校验",
            )

        output = SectionWritingResult(draft=validated, attempts=attempts)
        return AgentTurnResult(
            status="completed",
            output=output,
            state_updates={
                "fingerprint": fingerprint,
                "attempts": attempts,
                "draft": validated,
                "validation_issues": [],
                "completed": True,
            },
            handoff={"revision": request.next_revision, "attempts": attempts},
        )

    @staticmethod
    def _fingerprint(request: SectionWritingRequest) -> str:
        value = {
            "section_id": request.section_id,
            "next_revision": request.next_revision,
            "section_context": request.section_context,
            "sources": [evidence_key(source) for source in request.sources],
            "limitations": request.limitations,
            "current_draft": request.current_draft,
            "review": request.review.model_dump(mode="json") if request.review else None,
        }
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
