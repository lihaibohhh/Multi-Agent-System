"""Strict Claim candidate binding, opaque-segment repair, and retry fencing."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from copy import deepcopy

from pydantic import ValidationError

from ..core.budget import ExecutionPaused, current_budget
from .artifacts import bind_claims
from .claim_candidates import ClaimCandidate, ClaimRepairBatch, ClaimRepairCandidate
from .models import Claim, ClaimExtraction, EvidenceLink, QuoteSpan
from .rendering import evidence_key
from .segments import SegmentCatalog, build_segment_catalog
from .validation import BusinessValidationError


# Compatibility imports for callers and persisted diagnostic tooling. Their
# shape follows the current protocol even though the public names predate it.
ClaimPatch = ClaimRepairCandidate
ClaimRepairs = ClaimRepairBatch
CLAIM_PROTOCOL_VERSION = 3


class ClaimsPending(ExecutionPaused):
    def __init__(self, section):
        self.section_id = section.section_id
        pending = len(section.claim_work.get("pending", []))
        reason = f"{pending} 条待修复" if pending else "抽取格式仍待修复"
        super().__init__(
            f"章节 {section.section_id}：已保存 {len(section.claims)} 条通过校验的关联，"
            f"{reason}；自动修复已停止，请补充证据、重写章节或人工处置"
        )


def epoch():
    scope = current_budget.get()
    return scope.execution_id if scope else "standalone"


def fingerprint(section):
    return hashlib.sha256(
        json.dumps(
            [section.revision, section.draft, section.sources],
            ensure_ascii=False,
            sort_keys=True,
        ).encode()
    ).hexdigest()


def new_work(section):
    catalog = build_segment_catalog(section)
    return {
        "fingerprint": fingerprint(section),
        "protocol_version": CLAIM_PROTOCOL_VERSION,
        "catalog_fingerprint": catalog.fingerprint,
        "segmentation_version": catalog.segmentation_version,
        "accepted": {},
        "pending": [],
        "batch_errors": [],
        "attempts": 0,
        "total_attempts": 0,
        "epoch": epoch(),
        "last_failure_fingerprint": "",
        "repeated_failures": 0,
    }


def _safe_legacy_candidate(raw) -> dict:
    """Drop model-owned internal locators while retaining repair semantics."""

    if not isinstance(raw, dict):
        return {}
    evidence = []
    for item in raw.get("evidence", []) if isinstance(raw.get("evidence"), list) else []:
        if not isinstance(item, dict):
            continue
        cleaned = {
            key: item[key]
            for key in ("source_number", "relation")
            if key in item
        }
        evidence.append(cleaned)
    return {
        key: value
        for key, value in {
            "statement": raw.get("statement"),
            "assessment": raw.get("assessment"),
            "caveat": raw.get("caveat", ""),
            "evidence": evidence,
            "draft_segment_id": raw.get("draft_segment_id"),
        }.items()
        if value is not None
    }


def prepare_work(section, raw_work: dict | None) -> dict:
    """Load current work or migrate an older checkpoint without changing Claims."""

    if not raw_work or raw_work.get("fingerprint") != fingerprint(section):
        return new_work(section)
    work = deepcopy(raw_work)
    catalog = build_segment_catalog(section)
    migrated = work.get("protocol_version") != CLAIM_PROTOCOL_VERSION
    catalog_changed = work.get("catalog_fingerprint") != catalog.fingerprint
    if migrated:
        work["pending"] = [
            {
                "slot": item.get("slot"),
                "candidate": _safe_legacy_candidate(item.get("candidate")),
                "errors": deepcopy(item.get("errors") or []),
            }
            for item in work.get("pending", [])
            if isinstance(item, dict) and isinstance(item.get("slot"), int)
        ]
    work.update(
        protocol_version=CLAIM_PROTOCOL_VERSION,
        catalog_fingerprint=catalog.fingerprint,
        segmentation_version=catalog.segmentation_version,
        epoch=epoch(),
    )
    work.setdefault("accepted", {})
    work.setdefault("pending", [])
    work.setdefault("batch_errors", [])
    work.setdefault("total_attempts", 0)
    if migrated or catalog_changed:
        # A new protocol/catalog is a materially new repair strategy. Preserve
        # lifetime usage while opening one bounded new-protocol attempt window.
        work.update(
            attempts=0,
            last_failure_fingerprint="",
            repeated_failures=0,
        )
    else:
        work.setdefault("attempts", 0)
        work.setdefault("last_failure_fingerprint", "")
        work.setdefault("repeated_failures", 0)
    return work


def _validation_errors(exc: ValidationError) -> list[dict]:
    return [
        {
            "field": ".".join(map(str, error["loc"])),
            "type": error["type"],
            "message": error["msg"],
        }
        for error in exc.errors(include_input=False, include_context=False)
    ]


def _legacy_accept_one(section, raw, slot):
    """V1 compatibility used only for old checkpoints/tests, never model Schema."""

    try:
        candidate = Claim.model_validate(raw)
        bound = bind_claims(section, ClaimExtraction(claims=[candidate]))[0]
        bound.claim_id = f"{section.section_id}:v{section.revision}:c{slot}"
        return bound.model_dump(mode="json"), None
    except (ValidationError, BusinessValidationError) as exc:
        errors = exc.details() if isinstance(exc, BusinessValidationError) else _validation_errors(exc)
        return None, {"slot": slot, "candidate": _safe_legacy_candidate(raw), "errors": errors}


def split_extraction(section, candidates, work=None):
    """Safely import legacy literal-quote candidates into current repair state."""

    updated = deepcopy(work or new_work(section))
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 12:
        updated["batch_errors"] = [
            {"type": "invalid_batch", "message": "需返回 1–12 条 Claim，不可隐式截断结果"}
        ]
        return updated
    updated["batch_errors"] = []
    updated["accepted"], updated["pending"] = {}, []
    for slot, raw in enumerate(candidates, 1):
        bound, pending = _legacy_accept_one(section, raw, slot)
        if pending:
            updated["pending"].append(pending)
        else:
            updated["accepted"][str(slot)] = bound
    return updated


def _candidate_to_claim(
    section,
    catalog: SegmentCatalog,
    candidate: ClaimCandidate,
    slot: int,
    *,
    original: dict | None = None,
) -> Claim:
    draft = catalog.segments.get(candidate.draft_segment_id)
    if draft is None or draft.kind != "draft":
        raise ValueError("draft_segment_id 必须是当前目录中的正文片段")

    links: list[EvidenceLink] = []
    selected_ids: set[str] = set()
    for selection in candidate.evidence:
        segment_id = selection.segment_id
        if segment_id in selected_ids:
            raise ValueError("同一 Claim 不能重复选择证据片段")
        selected_ids.add(segment_id)
        segment = catalog.segments.get(segment_id)
        if segment is None or segment.kind != "evidence" or segment.source_number is None:
            raise ValueError("evidence.segment_id 必须来自当前目录中的来源片段")
        source = section.sources[segment.source_number - 1]
        links.append(
            EvidenceLink(
                source_number=segment.source_number,
                quote=segment.text,
                relation=selection.relation,
                evidence_id=evidence_key(source),
                quote_span=QuoteSpan(
                    start=segment.start,
                    end=segment.end,
                    match="exact",
                ),
            )
        )

    assessment = candidate.assessment
    caveats = []
    if original:
        old_assessment = original.get("assessment")
        if old_assessment in {"uncertain", "unsupported"} and assessment == "supported":
            assessment = old_assessment
        old_caveat = original.get("caveat", "")
        if isinstance(old_caveat, str) and old_caveat:
            caveats.append(old_caveat)

        counter_sources = set()
        had_counter = False
        for item in original.get("evidence", []):
            if not isinstance(item, dict) or item.get("relation") != "contradicts":
                continue
            had_counter = True
            if isinstance(item.get("source_number"), int):
                counter_sources.add(item["source_number"])
            legacy_ids = item.get("segment_ids", [])
            if isinstance(item.get("segment_id"), str):
                legacy_ids = [item["segment_id"]]
            for segment_id in legacy_ids:
                segment = catalog.segments.get(segment_id)
                counter_sources.add(segment.source_number if segment else None)
        selected_counter_sources = {
            link.source_number for link in links if link.relation == "contradicts"
        }
        if (
            (had_counter and not selected_counter_sources)
            or None in counter_sources
            or not counter_sources <= selected_counter_sources
        ):
            raise ValueError("不得删除原反证；无法定位则保留待修复")
    if candidate.caveat:
        caveats.append(candidate.caveat)
    caveat = "；".join(dict.fromkeys(caveats))
    if len(caveat) > 1000:
        raise ValueError("原有限制与新 caveat 合计超过上限")

    supports = any(link.relation == "supports" for link in links)
    contradicts = any(link.relation == "contradicts" for link in links)
    if assessment == "supported" and (not supports or contradicts):
        assessment = "uncertain" if contradicts else "unsupported"
        caveat = caveat or "支持证据缺失或存在反证，不能确认为充分支持"

    return Claim(
        claim_id=f"{section.section_id}:v{section.revision}:c{slot}",
        statement=candidate.statement,
        draft_quote=draft.text,
        draft_span=QuoteSpan(start=draft.start, end=draft.end, match="exact"),
        assessment=assessment,
        evidence=links,
        caveat=caveat,
    )


def _accept_candidate(section, raw, slot, catalog, *, original=None):
    try:
        candidate = ClaimCandidate.model_validate(raw)
        claim = _candidate_to_claim(section, catalog, candidate, slot, original=original)
        return claim.model_dump(mode="json"), None
    except ValidationError as exc:
        errors = _validation_errors(exc)
    except ValueError as exc:
        errors = [{"field": "$", "type": "invalid_selection", "message": str(exc)}]
    candidate = raw if isinstance(raw, dict) else {}
    return None, {"slot": slot, "candidate": deepcopy(candidate), "errors": errors}


def split_candidates(section, candidates, work=None, catalog=None):
    """Validate initial Candidate siblings independently and preserve successes."""

    updated = deepcopy(work or new_work(section))
    catalog = catalog or build_segment_catalog(section)
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 12:
        updated["batch_errors"] = [
            {"type": "invalid_batch", "message": "需返回 1–12 条 Claim，不可隐式截断结果"}
        ]
        return updated
    updated["batch_errors"] = []
    updated["accepted"], updated["pending"] = {}, []
    for slot, raw in enumerate(candidates, 1):
        bound, pending = _accept_candidate(section, raw, slot, catalog)
        if pending:
            updated["pending"].append(pending)
        else:
            updated["accepted"][str(slot)] = bound
    return updated


def _pending_source_numbers(pending: list[dict]) -> set[int] | None:
    numbers: set[int] = set()
    unrestricted = False
    for item in pending:
        candidate = item.get("candidate") if isinstance(item, dict) else None
        evidence = candidate.get("evidence", []) if isinstance(candidate, dict) else []
        if not isinstance(evidence, list) or not evidence:
            unrestricted = True
            continue
        found = False
        for selection in evidence:
            if not isinstance(selection, dict):
                continue
            number = selection.get("source_number")
            if isinstance(number, int):
                numbers.add(number)
                found = True
        if not found:
            unrestricted = True
    return None if unrestricted or not numbers else numbers


def _pending_prompt_item(item: dict) -> dict:
    candidate = item.get("candidate") if isinstance(item, dict) else {}
    candidate = candidate if isinstance(candidate, dict) else {}
    return {
        "slot": item.get("slot"),
        "statement": candidate.get("statement", ""),
        "assessment": candidate.get("assessment", "unsupported"),
        "caveat": candidate.get("caveat", ""),
        "errors": item.get("errors", []),
    }


def repair_prompt(section, work):
    catalog = build_segment_catalog(section)
    source_numbers = _pending_source_numbers(work.get("pending", []))
    return json.dumps(
        {
            "section_id": section.section_id,
            "catalog_fingerprint": catalog.fingerprint,
            "pending": [_pending_prompt_item(item) for item in work.get("pending", [])],
            "previous_patch_errors": work.get("batch_errors", []),
            "segments": catalog.prompt_view(source_numbers),
        },
        ensure_ascii=False,
    )


def apply_repairs(section, work, patches):
    updated = deepcopy(work)
    catalog = build_segment_catalog(section)
    wanted = {item["slot"] for item in work.get("pending", [])}
    counts = Counter(patch.slot for patch in patches.repairs)
    updated["ignored_slots"] = [patch.slot for patch in patches.repairs if patch.slot not in wanted]
    updated["batch_errors"] = []
    by_slot = {
        patch.slot: patch
        for patch in patches.repairs
        if patch.slot in wanted and counts[patch.slot] == 1
    }
    remaining = []
    for pending in work.get("pending", []):
        item = deepcopy(pending)
        slot = item["slot"]
        patch = by_slot.get(slot)
        if patch is None:
            item["errors"] = [{
                "field": "$",
                "type": "missing_patch",
                "message": "必须恰好修复一次此 slot，不能删除未解决项",
            }]
            remaining.append(item)
            continue
        original = item.get("candidate") if isinstance(item.get("candidate"), dict) else {}
        statement = original.get("statement")
        if not isinstance(statement, str) or not statement.strip() or len(statement) > 1000:
            item["errors"] = [{
                "field": "statement",
                "type": "invalid_original_statement",
                "message": "原 Claim statement 无法安全修复",
            }]
            remaining.append(item)
            continue
        raw = {
            "statement": statement,
            "draft_segment_id": patch.draft_segment_id,
            "assessment": patch.assessment,
            "evidence": [selection.model_dump(mode="json") for selection in patch.evidence],
            "caveat": patch.caveat,
        }
        bound, failed = _accept_candidate(
            section,
            raw,
            slot,
            catalog,
            original=original,
        )
        if failed:
            item["errors"] = failed["errors"]
            item["last_patch"] = patch.model_dump(mode="json")
            remaining.append(item)
        else:
            updated["accepted"][str(slot)] = bound
    updated["pending"] = remaining
    return updated


def salvage_candidates(section, work, raw):
    if not isinstance(raw, dict) or not isinstance(raw.get("claims"), list):
        return work
    return split_candidates(section, raw["claims"], work)


def salvage_patches(section, work, raw):
    if not isinstance(raw, dict) or not isinstance(raw.get("repairs"), list):
        return work
    patches = []
    slots = [item.get("slot") for item in raw["repairs"] if isinstance(item, dict)]
    for value in raw["repairs"]:
        try:
            patch = ClaimRepairCandidate.model_validate(value)
            if slots.count(patch.slot) == 1:
                patches.append(patch)
        except ValidationError:
            pass
    return apply_repairs(section, work, ClaimRepairBatch(repairs=patches)) if patches else work


def update_failure_state(work: dict) -> dict:
    """Persist a safe stagnation signature across process restarts."""

    updated = deepcopy(work)
    if not updated.get("pending") and not updated.get("batch_errors"):
        updated.update(last_failure_fingerprint="", repeated_failures=0)
        return updated
    payload = {
        "catalog": updated.get("catalog_fingerprint"),
        "pending": [
            {
                "slot": item.get("slot"),
                "errors": [
                    {"field": error.get("field"), "type": error.get("type")}
                    for error in item.get("errors", [])
                    if isinstance(error, dict)
                ],
                "last_patch": item.get("last_patch"),
            }
            for item in updated.get("pending", [])
            if isinstance(item, dict)
        ],
        "batch_errors": [
            {"field": error.get("field"), "type": error.get("type")}
            for error in updated.get("batch_errors", [])
            if isinstance(error, dict)
        ],
    }
    signature = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    previous = updated.get("last_failure_fingerprint")
    updated["repeated_failures"] = (
        int(updated.get("repeated_failures", 0)) + 1 if previous == signature else 1
    )
    updated["last_failure_fingerprint"] = signature
    return updated


def retry_blocked(work: dict) -> bool:
    return int(work.get("attempts", 0)) >= 3 or int(work.get("repeated_failures", 0)) >= 2


__all__ = [
    "CLAIM_PROTOCOL_VERSION",
    "ClaimPatch",
    "ClaimRepairs",
    "ClaimsPending",
    "apply_repairs",
    "epoch",
    "fingerprint",
    "new_work",
    "prepare_work",
    "repair_prompt",
    "retry_blocked",
    "salvage_candidates",
    "salvage_patches",
    "split_candidates",
    "split_extraction",
    "update_failure_state",
]
