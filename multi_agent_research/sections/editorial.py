"""Evidence-safe preparation and rendering for whole-report semantic editing."""

from __future__ import annotations

import re
from typing import Iterable

from .models import (
    ChiefEditorResult,
    EditedSectionArtifact,
    EditorialBlueprint,
    EditorialSectionPlan,
    ReportReview,
    SectionRecord,
)
from .rendering import CITATION, document_key, evidence_key, extract_doc_title


EVIDENCE_TOKEN = re.compile(r"\[\[evidence:([0-9a-f]{64})\]\]")


def evidence_tokens(text: str) -> set[str]:
    return set(EVIDENCE_TOKEN.findall(text or ""))


def editorial_visible_length(text: str) -> int:
    """Count reader-visible characters, not 64-character internal evidence IDs."""

    return len(EVIDENCE_TOKEN.sub("[来源]", text or ""))


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


def editorial_planning_view(stable_sections: Iterable[dict]) -> tuple[dict, ...]:
    """Return a compact whole-report view for planning, not full-report rewriting."""

    view = []
    for section in stable_sections:
        draft = str(section.get("draft", ""))
        if len(draft) <= 1600:
            excerpt = draft
        else:
            excerpt = draft[:1200].rstrip() + "\n…\n" + draft[-400:].lstrip()
        view.append({
            "section_id": section["section_id"],
            "title": section["title"],
            "question": section["question"],
            "draft_excerpt": excerpt,
            "claims": section.get("claims", []),
            "limitations": section.get("limitations", []),
        })
    return tuple(view)


def editorial_framing_view(
    artifacts: Iterable[EditedSectionArtifact],
) -> tuple[dict, ...]:
    """Compact chapter handoffs used to write summary and conclusion."""

    return tuple({
        "title": artifact.section.title,
        "source_section_ids": artifact.section.source_section_ids,
        "claim_ids": artifact.section.claim_ids,
        "summary": artifact.summary,
        "handoff": artifact.handoff,
    } for artifact in artifacts)


def editorial_capacity_floor(plan: EditorialSectionPlan) -> int:
    """Conservative space needed to state allocated claims and cite evidence."""

    return 450 + 150 * len(plan.claim_ids) + 30 * len(plan.evidence_ids)


def editorial_length_bounds(
    plan: EditorialSectionPlan,
    source_draft: str,
) -> tuple[int, int]:
    """Derive a chapter-specific publication range from plan and source size."""

    target = max(plan.target_chars, editorial_capacity_floor(plan))
    lower = max(400, round(target * 0.75))
    upper = max(
        target + 400,
        round(target * 1.25),
        min(editorial_visible_length(source_draft) + 250, 6000),
    )
    return lower, min(6000, upper)


def editorial_section_brief(
    blueprint: EditorialBlueprint,
    report_review: ReportReview,
    section_id: str,
) -> dict:
    """Project the whole-report plan to only what the current chapter needs."""

    plan = next(
        item for item in blueprint.section_plans
        if item.source_section_id == section_id
    )
    relevant_issue_indexes = [
        index
        for index, issue in enumerate(report_review.issues)
        if not issue.section_ids or section_id in issue.section_ids
    ]
    resolutions = {
        item.issue_index: item
        for item in blueprint.issue_resolutions
        if item.issue_index in relevant_issue_indexes
    }
    return {
        "report_title": blueprint.report_title,
        "thesis": blueprint.thesis,
        "audience": blueprint.audience,
        "style_rules": blueprint.style_rules,
        "terminology": [item.model_dump(mode="json") for item in blueprint.terminology],
        "target_plan": plan.model_dump(mode="json"),
        "review_issues": [
            {
                "issue_index": index,
                **report_review.issues[index].model_dump(mode="json"),
                "resolution": (
                    resolutions[index].model_dump(mode="json")
                    if index in resolutions else None
                ),
            }
            for index in relevant_issue_indexes
        ],
        "report_limitations": blueprint.unresolved_issues,
    }


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
    "editorial_framing_view",
    "editorial_capacity_floor",
    "editorial_length_bounds",
    "editorial_planning_view",
    "editorial_section_brief",
    "editorial_visible_length",
    "evidence_tokens",
    "render_edited_report",
    "stable_editorial_sections",
]
