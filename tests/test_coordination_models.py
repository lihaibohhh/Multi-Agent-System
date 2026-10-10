from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from multi_agent_research.coordination.models import CoordinationUnitWrite


def _unit(**overrides) -> dict:
    value = {
        "run_id": "run_child",
        "scope_type": "section",
        "scope_id": "section_1",
        "source_run_id": "run_child",
        "revision": 1,
        "expected_workspace_version": 0,
        "documents": [{
            "document_id": "doc_1",
            "title": "Industry report",
            "source_type": "knowledge_service",
            "retrieved_at": datetime.now(timezone.utc),
        }],
        "evidence": [{
            "evidence_id": "ev_1",
            "document_id": "doc_1",
            "excerpt": "The measured value was 53.47%.",
            "excerpt_hash": "hash_1",
        }],
        "claims": [{
            "claim_id": "section_1:v1:c1",
            "statement": "The measured value was 53.47%.",
            "origin_run_id": "run_child",
            "origin_section_id": "section_1",
            "claim_type": "metric",
            "status": "accepted",
            "revision": 1,
            "evidence_bindings": [{
                "evidence_id": "ev_1",
                "support_status": "supports",
            }],
        }],
        "metrics": [{
            "metric_id": "metric_1",
            "metric_name": "penetration rate",
            "value_text": "53.47%",
            "value_numeric": "53.47",
            "unit": "%",
            "period": "2025",
            "claim_id": "section_1:v1:c1",
            "evidence_ids": ["ev_1"],
        }],
    }
    value.update(overrides)
    return value


def test_coordination_unit_validates_local_bindings() -> None:
    unit = CoordinationUnitWrite.model_validate(_unit())

    assert unit.claims[0].evidence_bindings[0].support_status == "supports"
    assert unit.metrics[0].evidence_ids == ["ev_1"]


def test_non_support_binding_requires_reason() -> None:
    value = _unit()
    value["claims"][0]["evidence_bindings"][0]["support_status"] = "insufficient"

    with pytest.raises(ValidationError, match="require a reason"):
        CoordinationUnitWrite.model_validate(value)


def test_unknown_evidence_is_rejected() -> None:
    value = _unit()
    value["claims"][0]["evidence_bindings"][0]["evidence_id"] = "missing"

    with pytest.raises(ValidationError, match="unknown evidence"):
        CoordinationUnitWrite.model_validate(value)


def test_other_scope_evidence_can_be_referenced_explicitly() -> None:
    value = _unit(
        documents=[],
        evidence=[],
        referenced_evidence_ids=["parent_ev_1"],
        metrics=[],
    )
    value["claims"][0]["evidence_bindings"][0]["evidence_id"] = "parent_ev_1"

    unit = CoordinationUnitWrite.model_validate(value)

    assert unit.referenced_evidence_ids == ["parent_ev_1"]


def test_duplicate_binding_for_one_claim_is_rejected_before_persistence() -> None:
    value = _unit()
    value["claims"][0]["evidence_bindings"].append({
        "evidence_id": "ev_1",
        "support_status": "supports",
    })

    with pytest.raises(ValidationError, match="duplicate evidence binding"):
        CoordinationUnitWrite.model_validate(value)
