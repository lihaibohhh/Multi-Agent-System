"""Deterministic projections from persisted Run/Section artifacts.

The projection never summarizes raw report text and never calls a model.  It
reuses the model-produced section review summary, binds it to stable Claim IDs,
and builds a traceable parent summary from accepted parent Claims.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime

from ..sections.models import Claim, SectionRecord
from ..sections.rendering import evidence_key, extract_doc_title
from .models import (
    ClaimArtifact,
    CoordinationUnitWrite,
    DocumentArtifact,
    EvidenceArtifact,
    EvidenceBindingArtifact,
    EvidenceQuoteArtifact,
)


def workspace_claim_id(origin_run_id: str, claim_id: str) -> str:
    """Namespace chapter-local Claim IDs by their immutable origin Run."""

    return f"{origin_run_id}:{claim_id}"[:256]


def _parse_datetime(value) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _document_id(source: dict) -> str:
    metadata = source.get("metadata") or {}
    identity = (
        metadata.get("url")
        or metadata.get("source")
        or metadata.get("title")
        or str(metadata.get("chunk_id", "")).split("::")[0]
        or extract_doc_title(source)
        or "unknown"
    )
    raw = json.dumps(
        [source.get("source", "unknown"), identity],
        ensure_ascii=False,
        sort_keys=True,
    )
    return "doc_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _locator(metadata: dict) -> str | None:
    parts = []
    if metadata.get("page") not in (None, -1):
        parts.append(f"page={metadata['page']}")
    if metadata.get("chunk_id"):
        parts.append(f"chunk_id={metadata['chunk_id']}")
    return "; ".join(parts) or None


def _source_artifacts(source: dict) -> tuple[DocumentArtifact, EvidenceArtifact]:
    metadata = source.get("metadata") or {}
    content = str(source.get("content", "")).strip()
    document_id = _document_id(source)
    return (
        DocumentArtifact(
            document_id=document_id,
            canonical_url=metadata.get("url"),
            title=(
                metadata.get("title")
                or extract_doc_title(source)
                or metadata.get("source")
                or "来源名称未提供"
            ),
            publisher=metadata.get("publisher"),
            author=metadata.get("author"),
            published_at=_parse_datetime(
                metadata.get("published_at") or metadata.get("published_date")
            ),
            source_type=str(source.get("source") or "unknown")[:64],
            content_hash=metadata.get("content_hash"),
            retrieved_at=_parse_datetime(metadata.get("retrieved_at")),
            metadata=metadata,
        ),
        EvidenceArtifact(
            evidence_id=evidence_key(source),
            document_id=document_id,
            excerpt=content,
            excerpt_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            locator=_locator(metadata),
            retrieval_query=source.get("query"),
        ),
    )


def _collect_sources(sources: list[dict]):
    documents = {}
    evidence = {}
    for source in sources:
        if not str(source.get("content", "")).strip():
            continue
        document, excerpt = _source_artifacts(source)
        documents[document.document_id] = document
        evidence[excerpt.evidence_id] = excerpt
    return documents, evidence


def _claim_status(assessment: str) -> str:
    return {
        "supported": "accepted",
        "uncertain": "disputed",
        "unsupported": "rejected",
    }[assessment]


def _project_claim(
    claim: Claim,
    *,
    origin_run_id: str,
    origin_section_id: str | None,
    revision: int,
    sources: list[dict],
) -> ClaimArtifact:
    grouped: dict[str, list[tuple[str, EvidenceQuoteArtifact]]] = {}
    for link in claim.evidence:
        evidence_id = link.evidence_id
        if not evidence_id and 1 <= link.source_number <= len(sources):
            evidence_id = evidence_key(sources[link.source_number - 1])
        if not evidence_id:
            continue
        status = {
            "supports": "supports",
            "contradicts": "contradicts",
            "context": "insufficient",
        }[link.relation]
        quote_ref = EvidenceQuoteArtifact(
            source_number=link.source_number,
            quote=link.quote,
            relation=link.relation,
            quote_span=(
                link.quote_span.model_dump(mode="json") if link.quote_span else None
            ),
        )
        values = grouped.setdefault(evidence_id, [])
        quote_key = (
            quote_ref.source_number,
            quote_ref.quote,
            quote_ref.relation,
            json.dumps(
                quote_ref.quote_span.model_dump(mode="json")
                if quote_ref.quote_span
                else None,
                sort_keys=True,
            ),
        )
        if not any(
            (
                existing.source_number,
                existing.quote,
                existing.relation,
                json.dumps(
                    existing.quote_span.model_dump(mode="json")
                    if existing.quote_span
                    else None,
                    sort_keys=True,
                ),
            )
            == quote_key
            for _, existing in values
        ):
            values.append((status, quote_ref))

    bindings = []
    for evidence_id, values in grouped.items():
        statuses = {status for status, _ in values}
        if "supports" in statuses and "contradicts" in statuses:
            support_status = "pending"
            reason = "同一证据中同时存在支持与反向引文，需协调审查"
        elif "supports" in statuses:
            support_status = "supports"
            reason = None
        elif "contradicts" in statuses:
            support_status = "contradicts"
            reason = claim.caveat or "该摘录与结论存在反向关系"
        else:
            support_status = "insufficient"
            reason = "该摘录仅提供背景，不能单独证明结论"
        bindings.append(
            EvidenceBindingArtifact(
                evidence_id=evidence_id,
                support_status=support_status,
                reason=reason,
                required_supplement=(
                    (claim.caveat or reason) if support_status != "supports" else None
                ),
                quote_refs=[quote for _, quote in values],
            )
        )
    return ClaimArtifact(
        claim_id=workspace_claim_id(origin_run_id, claim.claim_id),
        statement=claim.statement,
        origin_run_id=origin_run_id,
        origin_section_id=origin_section_id,
        claim_type="inference",
        status=_claim_status(claim.assessment),
        revision=max(1, revision),
        caveat=claim.caveat,
        evidence_bindings=bindings,
    )


def _summary_from_claims(claims: list[ClaimArtifact], maximum: int = 12):
    selected = [claim for claim in claims if claim.status == "accepted"][:maximum]
    if not selected:
        selected = [claim for claim in claims if claim.status == "disputed"][:maximum]
    return (
        "\n".join(f"- {claim.statement}" for claim in selected),
        [claim.claim_id for claim in selected],
    )


def project_parent_context(
    run_id: str,
    parent_context: dict | None,
    *,
    expected_workspace_version: int,
    parent_workspace_summary: str = "",
) -> CoordinationUnitWrite | None:
    """Build one parent scope from the bounded, persisted handoff artifact."""

    context = parent_context or {}
    handoff = context.get("handoff") or []
    source_run_id = str(context.get("source_run_id") or "")
    if not source_run_id or not handoff:
        return None

    all_sources = [source for item in handoff for source in item.get("sources", [])]
    documents, evidence = _collect_sources(all_sources)
    claims = []
    open_questions = []
    revision = 1
    for item in handoff:
        section_id = item.get("section_id")
        revision = max(revision, int(item.get("revision") or 1))
        sources = item.get("sources") or []
        for raw_claim in item.get("claims") or []:
            claims.append(
                _project_claim(
                    Claim.model_validate(raw_claim),
                    origin_run_id=source_run_id,
                    origin_section_id=section_id,
                    revision=int(item.get("revision") or 1),
                    sources=sources,
                )
            )
        open_questions.extend(item.get("unresolved") or [])

    generated_summary, summary_claim_ids = _summary_from_claims(claims)
    return CoordinationUnitWrite(
        run_id=run_id,
        scope_type="parent_run",
        scope_id=source_run_id,
        source_run_id=source_run_id,
        revision=revision,
        summary=parent_workspace_summary.strip() or generated_summary,
        summary_claim_ids=summary_claim_ids,
        status="provisional",
        open_questions=list(dict.fromkeys(str(item) for item in open_questions if item)),
        expected_workspace_version=expected_workspace_version,
        documents=list(documents.values()),
        evidence=list(evidence.values()),
        claims=claims,
    )


def _dependency_claim_ids(
    run_id: str,
    section: SectionRecord,
    all_sections: list[SectionRecord],
    parent_context: dict | None,
) -> list[str]:
    dependencies = []
    by_id = {item.section_id: item for item in all_sections}
    source_sections = set(section.depends_on)
    source_sections.update(
        source.get("metadata", {}).get("shared_from_section")
        for source in section.sources
        if source.get("metadata", {}).get("shared_from_section")
    )
    for source_section_id in source_sections:
        source_section = by_id.get(source_section_id)
        if source_section:
            dependencies.extend(
                workspace_claim_id(run_id, claim.claim_id)
                for claim in source_section.claims
                if claim.assessment != "unsupported"
            )

    context = parent_context or {}
    parent_run_id = str(context.get("source_run_id") or "")
    parent_ids = set(section.parent_section_ids)
    parent_ids.update(
        source.get("metadata", {}).get("inherited_from_section")
        for source in section.sources
        if source.get("metadata", {}).get("inherited_from_section")
    )
    if parent_run_id:
        for item in context.get("handoff") or []:
            if item.get("section_id") not in parent_ids:
                continue
            dependencies.extend(
                workspace_claim_id(parent_run_id, raw["claim_id"])
                for raw in item.get("claims") or []
                if raw.get("assessment") != "unsupported" and raw.get("claim_id")
            )
    return list(dict.fromkeys(dependencies))


def project_section(
    run_id: str,
    raw_section: SectionRecord | dict,
    all_sections: list[SectionRecord | dict],
    parent_context: dict | None,
    *,
    expected_workspace_version: int,
) -> CoordinationUnitWrite:
    """Project one persisted chapter without inventing a new summary or metric."""

    section = SectionRecord.model_validate(
        raw_section.model_dump(mode="json")
        if isinstance(raw_section, SectionRecord)
        else raw_section
    )
    sections = [
        SectionRecord.model_validate(
            item.model_dump(mode="json") if isinstance(item, SectionRecord) else item
        )
        for item in all_sections
    ]
    documents, evidence = _collect_sources(section.sources)
    claims = [
        _project_claim(
            claim,
            origin_run_id=run_id,
            origin_section_id=section.section_id,
            revision=section.revision,
            sources=section.sources,
        )
        for claim in section.claims
    ]
    fallback_summary, fallback_claim_ids = _summary_from_claims(claims, maximum=8)
    summary_claim_ids = [
        claim.claim_id for claim in claims if claim.status in {"accepted", "disputed"}
    ][:8]
    return CoordinationUnitWrite(
        run_id=run_id,
        scope_type="section",
        scope_id=section.section_id,
        source_run_id=run_id,
        revision=max(1, section.revision),
        summary=(section.review.summary.strip() if section.review else "") or fallback_summary,
        summary_claim_ids=summary_claim_ids or fallback_claim_ids,
        status="stale" if section.status == "stale" else "provisional",
        dependency_claim_ids=_dependency_claim_ids(
            run_id, section, sections, parent_context
        ),
        open_questions=list(
            dict.fromkeys(item for item in [*section.gaps, *section.limitations] if item)
        ),
        expected_workspace_version=expected_workspace_version,
        documents=list(documents.values()),
        evidence=list(evidence.values()),
        claims=claims,
    )
