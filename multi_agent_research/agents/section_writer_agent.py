"""Agent responsible only for drafting or revising one report chapter."""

from __future__ import annotations

import json

from .contracts import ModelCall, SectionWritingRequest, SectionWritingResult
from ..sections.validation import validate_draft


SECTION_WRITER_SYSTEM_PROMPT = (
    "仅写当前章节正文，不写报告标题、章节总标题、执行摘要或参考来源列表；"
    "综合章可以写综合判断。使用 ### 作为小节标题。每个事实性判断标注 [来源N]。"
    "根据原文处理矛盾和不确定性；父报告/前章摘要不是证据。"
    "证据缺口必须明确披露，不能用推断补成事实。将正文控制在约1500个中文字符。"
)


class SectionWriterAgent:
    """Generate a chapter body without mutating section or graph state."""

    name = "section_writer"

    async def run(
        self,
        request: SectionWritingRequest,
        *,
        call_model: ModelCall,
    ) -> SectionWritingResult:
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
        draft, cost = await call_model(
            SECTION_WRITER_SYSTEM_PROMPT,
            prompt,
            validator=lambda value: validate_draft(
                value,
                list(request.sources),
                request.section_id,
            ),
            context={
                "agent": self.name,
                "section_id": request.section_id,
                "revision": request.next_revision,
            },
        )
        return SectionWritingResult(draft=draft, cost=cost)
