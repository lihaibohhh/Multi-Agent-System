from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from multi_agent_research.sections import claim_repair
from multi_agent_research.sections.claim_candidates import ClaimCandidateBatch
from multi_agent_research.sections.models import SectionRecord
from multi_agent_research.sections.policies import claim_gate
from multi_agent_research.sections.segments import build_segment_catalog


def _section() -> SectionRecord:
    return SectionRecord(
        section_id="section_1",
        title="市场",
        question="市场如何？",
        draft="市场保持增长。但增速仍有不确定性。",
        revision=3,
        sources=[
            {
                "content": "2025年市场同比增长。行业口径存在差异。",
                "title": "来源",
                "source": "knowledge",
                "metadata": {},
                "score": 0.9,
                "query": "市场",
            }
        ],
    )


def test_model_schema_has_no_program_owned_identity_or_offsets() -> None:
    schema = ClaimCandidateBatch.model_json_schema()
    candidate_fields = schema["$defs"]["ClaimCandidate"]["properties"]
    evidence_fields = schema["$defs"]["EvidenceSelection"]["properties"]
    assert not {"claim_id", "draft_span", "draft_quote"} & candidate_fields.keys()
    assert not {"evidence_id", "quote_span", "quote", "source_number"} & evidence_fields.keys()
    assert set(evidence_fields) == {"segment_id", "relation"}
    assert schema["$defs"]["ClaimCandidate"]["properties"]["evidence"]["maxItems"] == 8

    with pytest.raises(ValidationError, match="extra_forbidden"):
        ClaimCandidateBatch.model_validate(
            {
                "claims": [
                    {
                        "statement": "结论",
                        "claim_id": "forged",
                        "draft_segment_id": "D0001",
                        "assessment": "supported",
                        "evidence": [],
                    }
                ]
            }
        )


def test_prompt_catalog_hides_offsets_but_keeps_exact_internal_mapping() -> None:
    section = _section()
    catalog = build_segment_catalog(section)
    prompt_view = catalog.prompt_view()
    assert list(prompt_view) == ["D0001", "D0002", "E0001", "E0002"]
    assert all("start" not in value and "end" not in value for value in prompt_view.values())
    for segment in catalog.segments.values():
        owner = section.draft if segment.kind == "draft" else section.sources[0]["content"]
        assert owner[segment.start : segment.end] == segment.text


def test_single_pdf_line_break_is_displayed_as_soft_wrap_not_a_new_segment() -> None:
    section = _section()
    section.sources[0]["content"] = (
        "AI推动先进封装演进，\n"
        "晶圆厂将其视为延续摩尔定律的关键手段。下一句。"
    )
    catalog = build_segment_catalog(section)
    evidence = [segment for segment in catalog.segments.values() if segment.kind == "evidence"]
    assert len(evidence) == 2
    assert "\n" in evidence[0].text
    assert "\n" not in catalog.prompt_view()["E0001"]["text"]
    assert section.sources[0]["content"][evidence[0].start : evidence[0].end] == evidence[0].text


def test_multiple_opaque_evidence_segments_bind_to_exact_persisted_links() -> None:
    section = _section()
    catalog = build_segment_catalog(section)
    work = claim_repair.split_candidates(
        section,
        [
            {
                "statement": "市场增长但口径有差异",
                "draft_segment_id": "D0001",
                "assessment": "uncertain",
                "evidence": [
                    {
                        "segment_id": "E0001",
                        "relation": "supports",
                    },
                    {"segment_id": "E0002", "relation": "supports"},
                ],
                "caveat": "统计口径需核对",
            }
        ],
        catalog=catalog,
    )
    claim = work["accepted"]["1"]
    assert [link["quote"] for link in claim["evidence"]] == [
        "2025年市场同比增长。",
        "行业口径存在差异。",
    ]
    assert all(link["quote_span"]["match"] == "exact" for link in claim["evidence"])


def test_wrong_kind_or_unknown_segment_stays_pending() -> None:
    section = _section()
    for bad_id in ("E0001", "D9999"):
        work = claim_repair.split_candidates(
            section,
            [
                {
                    "statement": "市场结论",
                    "draft_segment_id": bad_id,
                    "assessment": "unsupported",
                    "evidence": [],
                }
            ],
        )
        assert not work["accepted"]
        assert work["pending"][0]["errors"][0]["type"] == "invalid_selection"


def test_legacy_checkpoint_migrates_once_and_resume_does_not_reset_attempts(monkeypatch) -> None:
    section = _section()
    legacy = claim_repair.split_extraction(
        section,
        [
            {
                "statement": "不存在的结论",
                "draft_quote": "不存在的正文",
                "assessment": "supported",
                "claim_id": "forged",
                "draft_span": {"start": 0, "end": 1, "match": "exact"},
                "evidence": [],
            }
        ],
    )
    legacy.pop("protocol_version")
    legacy.pop("catalog_fingerprint")
    legacy.pop("segmentation_version")
    legacy.update(attempts=3, total_attempts=6, epoch="old")
    legacy["pending"][0]["candidate"]["claim_id"] = "forged"
    legacy["pending"][0]["candidate"]["draft_span"] = {"start": 0, "end": 1}

    monkeypatch.setattr(claim_repair, "epoch", lambda: "first-v3")
    migrated = claim_repair.prepare_work(section, legacy)
    assert migrated["protocol_version"] == claim_repair.CLAIM_PROTOCOL_VERSION
    assert migrated["attempts"] == 0 and migrated["total_attempts"] == 6
    assert "claim_id" not in migrated["pending"][0]["candidate"]
    assert "draft_span" not in migrated["pending"][0]["candidate"]

    migrated["attempts"] = 2
    monkeypatch.setattr(claim_repair, "epoch", lambda: "second-v3")
    resumed = claim_repair.prepare_work(section, deepcopy(migrated))
    assert resumed["epoch"] == "second-v3"
    assert resumed["attempts"] == 2 and resumed["total_attempts"] == 6


def test_v2_over_segmented_pending_migrates_without_losing_accepted_claims() -> None:
    section = _section()
    old = claim_repair.new_work(section)
    old.update(
        protocol_version=2,
        segmentation_version=2,
        catalog_fingerprint="old-catalog",
        attempts=3,
        total_attempts=3,
        accepted={"9": {"claim_id": "kept"}},
        pending=[
            {
                "slot": 1,
                "candidate": {
                    "statement": "AI推动技术演进",
                    "assessment": "supported",
                    "caveat": "旧版片段过碎",
                    "evidence": [
                        {
                            "segment_ids": [f"E{index:04d}" for index in range(1, 9)],
                            "relation": "supports",
                        },
                        {"segment_ids": ["E0009"], "relation": "supports"},
                    ],
                },
                "errors": [{"type": "value_error", "message": "at most 8"}],
            }
        ],
    )

    migrated = claim_repair.prepare_work(section, old)
    assert migrated["protocol_version"] == claim_repair.CLAIM_PROTOCOL_VERSION
    assert migrated["segmentation_version"] == 3
    assert migrated["attempts"] == 0 and migrated["total_attempts"] == 3
    assert migrated["accepted"] == {"9": {"claim_id": "kept"}}
    assert migrated["pending"][0]["candidate"]["evidence"] == [
        {"relation": "supports"},
        {"relation": "supports"},
    ]


def test_claim_gate_allows_old_protocol_to_reach_migration() -> None:
    section = _section()
    section.claim_work = claim_repair.new_work(section)
    section.claim_work.update(protocol_version=2, attempts=3, repeated_failures=2)
    assert claim_gate({"sections": [section.model_dump()], "active_section": 0}) == {
        "section_step": "claims"
    }


def test_repeated_identical_failure_stops_before_wasting_all_attempts() -> None:
    work = claim_repair.new_work(_section())
    work["pending"] = [
        {
            "slot": 1,
            "candidate": {"statement": "结论", "assessment": "unsupported"},
            "errors": [{"field": "$", "type": "invalid_selection", "message": "bad"}],
        }
    ]
    first = claim_repair.update_failure_state(work)
    second = claim_repair.update_failure_state(first)
    assert first["repeated_failures"] == 1
    assert second["repeated_failures"] == 2
    assert claim_repair.retry_blocked(second)
