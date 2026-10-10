from datetime import datetime, timezone

from multi_agent_research.coordination.briefing import build_research_brief
from multi_agent_research.coordination.models import CoordinationSnapshot, CoordinationUnit
from multi_agent_research.coordination.projection import (
    project_parent_context,
    project_section,
    workspace_claim_id,
)
from multi_agent_research.sections.models import EvidenceLink, SectionRecord
from multi_agent_research.sections.nodes.research import _inherit_shared_section_sources
from multi_agent_research.sections.rendering import evidence_key


def _source(name: str, **metadata) -> dict:
    return {
        "query": name,
        "source": "knowledge",
        "content": f"{name}的证据正文",
        "score": 0.9,
        "iteration": 0,
        "metadata": {
            "source": f"{name}.pdf",
            "chunk_id": f"{name}::1",
            "retrieved_at": "2026-10-09T10:00:00+00:00",
            **metadata,
        },
    }


def _completed_section(section_id: str, name: str) -> SectionRecord:
    source = _source(name)
    return SectionRecord(
        section_id=section_id,
        title=name,
        question=f"{name}情况如何？",
        status="complete",
        revision=1,
        sources=[source],
        results=[source],
        draft=f"{name}结论[来源1]。",
        reviewed_at=datetime.now(timezone.utc),
        review={"verdict": "pass", "summary": f"{name}的审校摘要"},
        claims=[{
            "claim_id": f"{section_id}:v1:c1",
            "statement": f"{name}结论",
            "draft_quote": f"{name}结论",
            "assessment": "supported",
            "evidence": [{
                "source_number": 1,
                "quote": f"{name}的证据正文",
                "relation": "supports",
                "evidence_id": evidence_key(source),
            }],
        }],
    )


def test_section_projection_reuses_review_summary_and_binds_claim_ids() -> None:
    section = _completed_section("section_1", "成本")

    unit = project_section(
        "run_current",
        section,
        [section],
        None,
        expected_workspace_version=0,
    )

    claim_id = workspace_claim_id("run_current", "section_1:v1:c1")
    assert unit.summary == "成本的审校摘要"
    assert unit.summary_claim_ids == [claim_id]
    assert unit.claims[0].claim_id == claim_id
    assert unit.claims[0].evidence_bindings[0].support_status == "supports"
    assert unit.evidence[0].document_id == unit.documents[0].document_id


def test_projection_merges_multiple_quotes_from_the_same_evidence() -> None:
    section = _completed_section("section_1", "集成化")
    evidence_id = section.claims[0].evidence[0].evidence_id
    section.claims[0].evidence = [
        EvidenceLink(
            source_number=1,
            quote="第一段支持性引文",
            relation="supports",
            evidence_id=evidence_id,
            quote_span={"start": 0, "end": 8, "match": "exact"},
        ),
        EvidenceLink(
            source_number=1,
            quote="第二段支持性引文",
            relation="supports",
            evidence_id=evidence_id,
            quote_span={"start": 9, "end": 17, "match": "exact"},
        ),
    ]

    unit = project_section(
        "run_current",
        section,
        [section],
        None,
        expected_workspace_version=0,
    )

    bindings = unit.claims[0].evidence_bindings
    assert len(bindings) == 1
    assert bindings[0].evidence_id == evidence_id
    assert bindings[0].support_status == "supports"
    assert [item.quote for item in bindings[0].quote_refs] == [
        "第一段支持性引文",
        "第二段支持性引文",
    ]


def test_parent_summary_is_built_from_claims_not_report_excerpt() -> None:
    section = _completed_section("section_1", "市场")
    context = {
        "source_run_id": "run_parent",
        "report_excerpt": "THIS TRUNCATED REPORT MUST NOT BECOME THE SUMMARY",
        "handoff": [{
            "section_id": section.section_id,
            "revision": section.revision,
            "summary": section.review.summary,
            "claims": [claim.model_dump(mode="json") for claim in section.claims],
            "sources": section.sources,
            "unresolved": [],
        }],
    }

    unit = project_parent_context(
        "run_child",
        context,
        expected_workspace_version=0,
    )

    assert unit is not None
    assert unit.summary == "- 市场结论"
    assert "TRUNCATED" not in unit.summary
    assert unit.summary_claim_ids == [
        workspace_claim_id("run_parent", "section_1:v1:c1")
    ]


def test_shared_evidence_creates_a_traceable_cross_section_dependency() -> None:
    first = _completed_section("section_1", "成本")
    second = SectionRecord(
        section_id="section_2",
        title="竞争",
        question="竞争情况如何？",
    )
    state = {
        "sections": [first.model_dump(mode="json"), second.model_dump(mode="json")],
        "active_section": 1,
    }

    _inherit_shared_section_sources(state, second)

    assert second.results[0]["metadata"]["shared_from_section"] == "section_1"
    shared_source = second.results[0]
    second = SectionRecord.model_validate({
        **second.model_dump(mode="json"),
        "status": "complete",
        "revision": 1,
        "sources": second.results,
        "draft": "竞争结论[来源1]。",
        "review": {"verdict": "pass", "summary": "竞争摘要"},
        "claims": [{
            "claim_id": "section_2:v1:c1",
            "statement": "竞争结论",
            "draft_quote": "竞争结论",
            "assessment": "supported",
            "evidence": [{
                "source_number": 1,
                "quote": "成本的证据正文",
                "relation": "supports",
                "evidence_id": evidence_key(shared_source),
            }],
        }],
    })
    unit = project_section(
        "run_current",
        second,
        [first, second],
        None,
        expected_workspace_version=1,
    )

    assert unit.dependency_claim_ids == [
        workspace_claim_id("run_current", "section_1:v1:c1")
    ]


def test_research_brief_excludes_stale_units_and_raw_excerpts() -> None:
    now = datetime.now(timezone.utc)
    parent = CoordinationUnit(
        run_id="run_child",
        scope_type="parent_run",
        scope_id="run_parent",
        source_run_id="run_parent",
        revision=1,
        summary="父任务摘要",
        summary_claim_ids=["run_parent:c1"],
        status="provisional",
        claims=[{
            "claim_id": "run_parent:c1",
            "statement": "父任务结论",
            "status": "accepted",
            "origin_section_id": "parent_section_1",
            "evidence_bindings": [{
                "evidence_id": "parent_evidence",
                "support_status": "supports",
            }],
        }],
        evidence=[{"evidence_id": "parent_evidence", "excerpt": "不应进入简报的原文"}],
        updated_at=now,
    )
    stale = CoordinationUnit(
        run_id="run_child",
        scope_type="section",
        scope_id="section_old",
        source_run_id="run_child",
        revision=1,
        summary="过期摘要",
        status="stale",
        updated_at=now,
    )
    snapshot = CoordinationSnapshot(
        run_id="run_child",
        workspace_version=2,
        status="collecting",
        parent_run=parent,
        sections=[stale],
        updated_at=now,
    )
    target = SectionRecord(
        section_id="section_1",
        title="目标",
        question="目标问题是什么？",
        parent_section_ids=["parent_section_1"],
    )

    brief = build_research_brief(snapshot, target)

    assert "父任务结论" in brief
    assert "过期摘要" not in brief
    assert "不应进入简报的原文" not in brief
