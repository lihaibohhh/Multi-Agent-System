"""Bounded, read-only research briefs for chapter Agents."""

from __future__ import annotations

import json

from ..sections.models import SectionRecord
from .models import CoordinationSnapshot, CoordinationUnit
from .runtime import load_coordination_snapshot


def _unit_view(unit: CoordinationUnit, section: SectionRecord) -> dict | None:
    if unit.status == "stale" or (
        unit.scope_type == "section" and unit.scope_id == section.section_id
    ):
        return None
    claims = [
        claim
        for claim in unit.claims
        if claim.get("status") in {"accepted", "disputed"}
        and (
            unit.scope_type != "parent_run"
            or not section.parent_section_ids
            or claim.get("origin_section_id") in section.parent_section_ids
        )
    ][:6]
    metrics = [
        metric
        for metric in unit.metrics
        if metric.get("status") in {"accepted", "disputed"}
    ][:6]
    if not unit.summary and not claims and not metrics:
        return None
    return {
        "scope_type": unit.scope_type,
        "scope_id": unit.scope_id,
        "revision": unit.revision,
        "summary": unit.summary[:1500],
        "summary_claim_ids": unit.summary_claim_ids,
        "summary_metric_ids": unit.summary_metric_ids,
        "claims": [
            {
                "claim_id": claim.get("claim_id"),
                "statement": str(claim.get("statement", ""))[:600],
                "status": claim.get("status"),
                "caveat": str(claim.get("caveat", ""))[:300],
                "evidence_ids": [
                    binding.get("evidence_id")
                    for binding in claim.get("evidence_bindings", [])
                    if binding.get("support_status") == "supports"
                ],
            }
            for claim in claims
        ],
        "metrics": [
            {
                key: metric.get(key)
                for key in (
                    "metric_id",
                    "metric_name",
                    "value_text",
                    "unit",
                    "period",
                    "geography",
                    "sample_scope",
                    "denominator_definition",
                    "status",
                    "evidence_ids",
                )
            }
            for metric in metrics
        ],
        "open_questions": unit.open_questions[:6],
    }


def build_research_brief(
    snapshot: CoordinationSnapshot | None,
    section: SectionRecord,
) -> str:
    """Select traceable summaries, not full drafts or raw Agent histories."""

    if snapshot is None:
        return ""
    units = ([snapshot.parent_run] if snapshot.parent_run else []) + snapshot.sections
    selected = [view for unit in units if (view := _unit_view(unit, section))]
    if not selected:
        return ""
    return json.dumps(
        {
            "workspace_version": snapshot.workspace_version,
            "trust": (
                "共享结论仅是研究线索；只有纳入本章来源表并通过 Claim 绑定的证据"
                "才能作为本章引用。disputed 结论必须保留不确定性。"
            ),
            "units": selected,
        },
        ensure_ascii=False,
    )


def build_editorial_context(snapshot: CoordinationSnapshot | None) -> str:
    """Build a bounded whole-Run view without raw evidence excerpts or histories."""

    if snapshot is None:
        return ""
    units = ([snapshot.parent_run] if snapshot.parent_run else []) + snapshot.sections
    selected = []
    for unit in units:
        if unit.status == "stale":
            continue
        selected.append({
            "scope_type": unit.scope_type,
            "scope_id": unit.scope_id,
            "revision": unit.revision,
            "summary": unit.summary[:1500],
            "claims": [
                {
                    "claim_id": claim.get("claim_id"),
                    "statement": str(claim.get("statement", ""))[:600],
                    "status": claim.get("status"),
                    "caveat": str(claim.get("caveat", ""))[:300],
                }
                for claim in unit.claims
                if claim.get("status") in {"accepted", "disputed"}
            ][:12],
            "metrics": [
                {
                    key: metric.get(key)
                    for key in (
                        "metric_id",
                        "metric_name",
                        "value_text",
                        "unit",
                        "period",
                        "geography",
                        "sample_scope",
                        "denominator_definition",
                        "status",
                    )
                }
                for metric in unit.metrics
                if metric.get("status") in {"accepted", "disputed"}
            ][:8],
            "open_questions": unit.open_questions[:8],
        })
    return json.dumps({
        "workspace_version": snapshot.workspace_version,
        "summary": snapshot.summary[:2000],
        "units": selected,
    }, ensure_ascii=False)


async def current_research_brief(section: SectionRecord) -> str:
    return build_research_brief(await load_coordination_snapshot(), section)


async def current_editorial_context() -> str:
    return build_editorial_context(await load_coordination_snapshot())
