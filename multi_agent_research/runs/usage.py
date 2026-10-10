"""Deterministic, payload-free usage views for the API and UI."""

from __future__ import annotations

from collections import defaultdict

from .models import RunUsageSummary, StageUsage, UsageTotals


_STAGE_ORDER = {
    "plan_sections": 0,
    "section_search": 1,
    "section_write": 2,
    "section_review": 3,
    "section_claims": 4,
    "report_review": 5,
    "chief_edit": 6,
    "chief_edit_section": 7,
    "chief_compress_section": 8,
    "chief_write_framing": 9,
    "edited_report_review": 10,
    "retrieval": 11,
    "unattributed": 12,
}


def _non_negative(value) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _charged(entry: dict) -> int:
    if entry.get("status") == "settled" and entry.get("actual_tokens") is not None:
        return _non_negative(entry.get("actual_tokens"))
    return _non_negative(entry.get("reserved_tokens"))


def summarize_run_usage(
    *,
    run_id: str,
    budget_id: str | None,
    model_usage: dict | None,
    budget: dict | None,
) -> RunUsageSummary:
    """Build settled usage and account occupancy for one independent Run."""

    usage = model_usage or {}
    ledger = budget or {}
    reservations = ledger.get("reservations") or {}
    own_entries = [
        entry
        for entry in reservations.values()
        if isinstance(entry, dict) and entry.get("run_id") == run_id
    ]
    grouped: dict[tuple[str, str, str], dict[str, int]] = defaultdict(
        lambda: {
            "calls": 0,
            "known_tokens": 0,
            "charged_tokens": 0,
            "unknown_calls": 0,
        }
    )
    for entry in own_entries:
        kind = entry.get("kind") if entry.get("kind") in {"model", "retrieval"} else "unattributed"
        label = str(entry.get("label") or kind)[:229]
        if kind == "model":
            stage, _, scope = label.partition(":")
        elif kind == "retrieval":
            stage, scope = "retrieval", label
        else:
            stage, scope = "unattributed", label
        key = (kind, stage[:100] or kind, scope[:128])
        row = grouped[key]
        row["calls"] += 1
        if entry.get("status") == "settled" and entry.get("actual_tokens") is not None:
            row["known_tokens"] += _non_negative(entry.get("actual_tokens"))
        elif kind == "model":
            row["unknown_calls"] += 1
        row["charged_tokens"] += _charged(entry)

    attributed_model_calls = sum(
        row["calls"] for (kind, _, _), row in grouped.items() if kind == "model"
    )
    attributed_known_tokens = sum(
        row["known_tokens"] for (kind, _, _), row in grouped.items() if kind == "model"
    )
    reported_model_calls = _non_negative(usage.get("attempts"))
    reported_known_tokens = _non_negative(usage.get("tokens"))
    missing_calls = max(0, reported_model_calls - attributed_model_calls)
    missing_tokens = max(0, reported_known_tokens - attributed_known_tokens)
    if missing_calls or missing_tokens:
        grouped[("unattributed", "unattributed", "legacy_or_missing_ledger")].update(
            calls=missing_calls,
            known_tokens=missing_tokens,
            charged_tokens=missing_tokens,
            unknown_calls=_non_negative(usage.get("unknown")),
        )

    stages = [
        StageUsage(kind=kind, stage=stage, scope=scope, **values)
        for (kind, stage, scope), values in grouped.items()
    ]
    stages.sort(
        key=lambda item: (
            _STAGE_ORDER.get(item.stage, 99),
            item.stage,
            item.scope,
            item.kind,
        )
    )
    own_model_entries = [entry for entry in own_entries if entry.get("kind") == "model"]
    own_retrieval_entries = [entry for entry in own_entries if entry.get("kind") == "retrieval"]
    attributed_charged = sum(_charged(entry) for entry in own_model_entries)
    current_known = max(reported_known_tokens, attributed_known_tokens)
    current = UsageTotals(
        model_calls=max(reported_model_calls, len(own_model_entries)),
        retrieval_calls=len(own_retrieval_entries),
        known_tokens=current_known,
        charged_tokens=attributed_charged + missing_tokens,
        unknown_model_calls=max(
            _non_negative(usage.get("unknown")),
            sum(
                entry.get("status") != "settled" for entry in own_model_entries
            ),
        ),
    )
    budget_totals = UsageTotals(
        model_calls=_non_negative(ledger.get("model_calls")),
        retrieval_calls=_non_negative(ledger.get("retrieval_calls")),
        known_tokens=_non_negative(ledger.get("known_tokens")),
        charged_tokens=_non_negative(ledger.get("charged_tokens")),
        unknown_model_calls=_non_negative(ledger.get("unknown_model_calls"))
        or sum(
            entry.get("kind") == "model" and entry.get("status") != "settled"
            for entry in reservations.values()
            if isinstance(entry, dict)
        ),
    )
    return RunUsageSummary(
        run_id=run_id,
        budget_id=budget_id,
        current=current,
        budget=budget_totals,
        stages=stages,
        stage_attribution_complete=(
            missing_calls == 0
            and missing_tokens == 0
            and len(own_model_entries) == reported_model_calls
        ),
        history_incomplete=bool(ledger.get("legacy_history_incomplete")),
    )
