"""Pure artifact operations: exact evidence bindings and conservative invalidation."""

from copy import deepcopy
from datetime import datetime, timezone

from .models import ClaimExtraction, SectionRecord
from .rendering import evidence_key
from .quotes import locate_quote, quote_repair_hint
from .validation import BusinessValidationError, ValidationIssue


def bind_claims(section: SectionRecord, extraction: ClaimExtraction):
    """Validate locators/exact excerpts; semantic support remains a model judgment."""
    claims = deepcopy(extraction.claims)
    issues = []
    repair_hints = []
    for index, claim in enumerate(claims, 1):
        path = f"claims[{index - 1}]"
        draft_match = locate_quote(section.draft, claim.draft_quote)
        if draft_match is None:
            issues.append(ValidationIssue(
                f"{path}.draft_quote", "draft_quote_not_found",
                "draft_quote is not an exact excerpt of the chapter；复制正文连续原文，不改标点或措辞",
                section_id=section.section_id,
            ))
            repair_hints.append(quote_repair_hint(section.draft, claim.draft_quote, f"{path}.draft_quote"))
        else:
            claim.draft_quote, claim.draft_span = draft_match
        claim.claim_id = f"{section.section_id}:v{section.revision}:c{index}"
        for link_index, link in enumerate(claim.evidence):
            link_path = f"{path}.evidence[{link_index}]"
            if not 1 <= link.source_number <= len(section.sources):
                issues.append(ValidationIssue(
                    f"{link_path}.source_number", "unknown_source",
                    f"source_number 必须在 1–{len(section.sources)} 内",
                    section_id=section.section_id, source_number=link.source_number,
                ))
                continue
            source = section.sources[link.source_number - 1]
            evidence_match = locate_quote(source["content"][:1000], link.quote)
            if evidence_match is None:
                issues.append(ValidationIssue(
                    f"{link_path}.quote", "evidence_quote_not_found",
                    "evidence quote is not an exact excerpt of the shown source；复制该来源展示片段的连续原文",
                    section_id=section.section_id, source_number=link.source_number,
                ))
                repair_hints.append(quote_repair_hint(source["content"][:1000], link.quote, f"{link_path}.quote"))
                continue
            link.quote, link.quote_span = evidence_match
            link.evidence_id = evidence_key(source)
        supports = any(link.relation == "supports" for link in claim.evidence)
        contradicts = any(link.relation == "contradicts" for link in claim.evidence)
        if claim.assessment == "supported" and (not supports or contradicts):
            claim.assessment = "uncertain" if contradicts else "unsupported"
            claim.caveat = claim.caveat or "支持证据缺失或存在反证，不能确认为充分支持"
    if issues:
        raise BusinessValidationError(issues, repair_hints=repair_hints[:10])
    return claims


def stamp_results(results: list[dict]) -> list[dict]:
    timestamp = datetime.now(timezone.utc).isoformat()
    stamped = deepcopy(results)
    for item in stamped:
        item.setdefault("metadata", {}).setdefault("retrieved_at", timestamp)
    return stamped


def validate_dependencies(sections: list[SectionRecord]) -> None:
    seen = set()
    for section in sections:
        if section.section_id in seen:
            raise ValueError("duplicate section_id")
        if not set(section.depends_on) <= seen:
            raise ValueError("chapter dependencies must refer to earlier sections")
        seen.add(section.section_id)


def revision_sections(sections: list[SectionRecord], target: str, instruction: str):
    """Copy an immutable parent report, invalidate target and transitive dependents."""
    copied = [section.model_copy(deep=True) for section in sections]
    if not copied or len(copied) > 4 or target not in {s.section_id for s in copied}:
        raise ValueError("unknown chapter or unsupported chapter plan")
    if any(s.status not in {"complete", "limited"} for s in copied):
        raise ValueError("only finished chapter artifacts can be revised")
    # V2 prompts exposed ALL earlier summaries, so their dependencies are conservative.
    for i, section in enumerate(copied):
        if section.artifact_version < 2:
            section.depends_on = [s.section_id for s in copied[:i]]
            section.dependency_revisions = {s.section_id: s.revision for s in copied[:i]}
    validate_dependencies(copied)
    affected = {target}
    for section in copied:
        if section.section_id in affected or affected.intersection(section.depends_on):
            affected.add(section.section_id)
            section.status = "stale"
            section.invalidated_by = [target]
            section.revision_instruction = (
                instruction if section.section_id == target
                else f"依赖章节 {target} 已修订；按最新证据和结论重新检查并重写本章。"
            )
            section.revision_base = section.revision
            section.search_rounds = 0
            section.results = []
            section.gaps = [instruction[:500]] if section.section_id == target else []
            section.limitations = []
            section.analyst = {}
            section.review = None
            section.dependency_revisions = {}
            section.artifact_version = 2
    return copied


def dependency_issues(sections: list[SectionRecord]) -> list[str]:
    validate_dependencies(sections)
    by_id = {s.section_id: s for s in sections}
    return [
        f"{s.section_id} 的依赖 {dep} 版本已变化，必须重建"
        for s in sections for dep in s.depends_on
        if s.dependency_revisions.get(dep) != by_id[dep].revision
    ]


def parent_handoff(sections: list[SectionRecord], selected: list[str] | None) -> list[dict]:
    """Selected artifacts, not scraped references or unbounded draft history."""
    wanted = set(selected) if selected is not None else {s.section_id for s in sections}
    if wanted - {s.section_id for s in sections}:
        raise ValueError("unknown parent_section_ids")
    handoff = []
    for section in sections:
        if section.section_id not in wanted:
            continue
        sources = section.sources[:15]
        handoff.append({
            "section_id": section.section_id, "title": section.title,
            "question": section.question, "revision": section.revision,
            "claims": [c.model_dump(mode="json") for c in section.claims],
            "summary": section.review.summary if section.review else "",
            "unresolved": section.limitations,
            "reviewed_at": section.reviewed_at.isoformat() if section.reviewed_at else None,
            "sources": deepcopy(sources),
            "sources_truncated": len(section.sources) > len(sources),
            "trust": "inherited_not_revalidated",
        })
    return handoff
