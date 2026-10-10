"""Evidence-safe preparation and rendering for whole-report semantic editing."""

from __future__ import annotations

import re
from typing import Iterable

from .models import ChiefEditorResult, SectionRecord
from .rendering import CITATION, document_key, evidence_key, extract_doc_title


EVIDENCE_TOKEN = re.compile(r"\[\[evidence:([0-9a-f]{64})\]\]")


def evidence_tokens(text: str) -> set[str]:
    return set(EVIDENCE_TOKEN.findall(text or ""))


def stable_editorial_sections(
    sections: Iterable[SectionRecord],
) -> tuple[tuple[dict, ...], dict[str, dict]]:
    """Namespace chapter-local citations with stable evidence identifiers."""

    prepared = []
    registry: dict[str, dict] = {}
    for section in sections:
        for source in section.sources:
            registry[evidence_key(source)] = source

        def replace(match: re.Match) -> str:
            number = int(match.group(1))
            if not 1 <= number <= len(section.sources):
                raise ValueError(
                    f"section {section.section_id} references unknown source {number}"
                )
            identifier = evidence_key(section.sources[number - 1])
            return f"[[evidence:{identifier}]]"

        prepared.append({
            "section_id": section.section_id,
            "title": section.title,
            "question": section.question,
            "draft": CITATION.sub(replace, section.draft),
            "claims": [
                {
                    "claim_id": claim.claim_id,
                    "statement": claim.statement,
                    "assessment": claim.assessment,
                    "caveat": claim.caveat,
                    "evidence_ids": [link.evidence_id for link in claim.evidence],
                }
                for claim in section.claims
                if claim.assessment != "unsupported"
            ],
            "limitations": section.limitations,
        })
    return tuple(prepared), registry


def render_edited_report(
    result: ChiefEditorResult,
    evidence_registry: dict[str, dict],
    *,
    limited: bool,
) -> str:
    """Render stable evidence tokens into one deduplicated public bibliography."""

    references: list[dict] = []
    document_numbers: dict[str, int] = {}

    def replace(match: re.Match) -> str:
        identifier = match.group(1)
        source = evidence_registry.get(identifier)
        if source is None:
            raise ValueError(f"edited report references unknown evidence {identifier}")
        key = document_key(source)
        if key not in document_numbers:
            references.append({"source": source, "pages": set()})
            document_numbers[key] = len(references)
        page = (source.get("metadata") or {}).get("page")
        if isinstance(page, int) and page >= 1:
            references[document_numbers[key] - 1]["pages"].add(page)
        return f"[来源{document_numbers[key]}]"

    def public_text(value: str) -> str:
        return EVIDENCE_TOKEN.sub(replace, value.strip())

    parts = [f"# {result.report_title}"]
    if limited or result.verdict == "limited":
        parts.append(
            "> 阅读提示：部分结论的证据强度或跨章节一致性有限。"
            "正文已保留适用范围、时效和不确定性说明。"
        )
    parts.append("## 执行摘要\n\n" + public_text(result.executive_summary))
    for section in result.sections:
        parts.append(f"## {section.title}\n\n{public_text(section.body)}")
    parts.append("## 结论\n\n" + public_text(result.conclusion))

    lines = ["## 参考来源"]
    for index, reference in enumerate(references, 1):
        source = reference["source"]
        metadata = source.get("metadata") or {}
        locator = (
            metadata.get("title")
            or extract_doc_title(source)
            or metadata.get("publisher")
            or "来源名称未提供"
        )
        pages = sorted(reference["pages"])
        if pages:
            locator += " | " + ", ".join(f"p.{page}" for page in pages)
        if metadata.get("url"):
            locator += f" | <{metadata['url']}>"
        lines.append(f"[来源{index}] {locator}")
    parts.append("\n".join(lines))
    return "\n\n".join(parts)


__all__ = [
    "EVIDENCE_TOKEN",
    "evidence_tokens",
    "render_edited_report",
    "stable_editorial_sections",
]
