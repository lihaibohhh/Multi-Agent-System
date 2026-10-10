from multi_agent_research.runs.usage import summarize_run_usage


def _entry(run_id: str, kind: str, label: str, tokens: int = 0) -> dict:
    return {
        "run_id": run_id,
        "execution_id": "execution",
        "kind": kind,
        "label": label,
        "status": "settled",
        "reserved_tokens": tokens + 1000 if kind == "model" else 0,
        "actual_tokens": tokens,
    }


def test_usage_separates_confirmed_usage_from_independent_budget_occupancy() -> None:
    current_run = "run_current"
    amounts = [2223, 6261, 9200, 9784, 15424, 10210, 10404, 5449, 12495]
    labels = [
        "plan_sections:",
        "section_search:section_1",
        "section_search:section_1",
        "section_write:section_1",
        "section_write:section_1",
        "section_review:section_1",
        "section_review:section_1",
        "section_claims:section_1",
        "section_claims:section_1",
    ]
    reservations = {
        f"current_model_{index}": _entry(current_run, "model", label, tokens)
        for index, (label, tokens) in enumerate(zip(labels, amounts, strict=True))
    }
    reservations.update({
        f"current_retrieval_{index}": _entry(
            current_run,
            "retrieval",
            "knowledge" if index < 4 else "web",
        )
        for index in range(8)
    })
    summary = summarize_run_usage(
        run_id=current_run,
        budget_id="budget_current",
        model_usage={"attempts": 9, "tokens": 81_450, "unknown": 0},
        budget={
            "model_calls": 9,
            "retrieval_calls": 8,
            "known_tokens": 81_450,
            "charged_tokens": 81_450,
            "legacy_history_incomplete": False,
            "reservations": reservations,
        },
    )

    assert summary.current.model_calls == 9
    assert summary.current.retrieval_calls == 8
    assert summary.current.known_tokens == 81_450
    assert summary.current.charged_tokens == 81_450
    assert summary.budget.model_calls == 9
    assert summary.budget.retrieval_calls == 8
    assert summary.budget.charged_tokens == 81_450
    assert summary.budget_scope == "run"
    assert summary.stage_attribution_complete is True
    by_stage = {stage.stage: stage for stage in summary.stages if stage.kind == "model"}
    assert by_stage["section_write"].known_tokens == 25_208
    assert by_stage["section_review"].known_tokens == 20_614
    assert by_stage["section_claims"].known_tokens == 17_944


def test_unknown_current_call_uses_reservation_and_is_visibly_unsettled() -> None:
    summary = summarize_run_usage(
        run_id="run_current",
        budget_id="budget_current",
        model_usage={"attempts": 1, "tokens": 0, "unknown": 1},
        budget={
            "model_calls": 1,
            "retrieval_calls": 0,
            "known_tokens": 0,
            "charged_tokens": 5000,
            "reservations": {
                "pending": {
                    "run_id": "run_current",
                    "kind": "model",
                    "label": "section_write:section_1",
                    "status": "unknown",
                    "reserved_tokens": 5000,
                }
            },
        },
    )

    assert summary.current.known_tokens == 0
    assert summary.current.charged_tokens == 5000
    assert summary.current.unknown_model_calls == 1
    assert summary.stages[0].unknown_calls == 1
