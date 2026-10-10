from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

import pytest

from multi_agent_research.coordination.repository import WorkspaceVersionConflict
from multi_agent_research.runs.models import ParentContextSnapshot
from multi_agent_research.runs.repository import PostgresRunRepository
from multi_agent_research.sections.artifacts import parent_handoff
from multi_agent_research.sections.models import SectionRecord
from multi_agent_research.sections.rendering import evidence_key


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_POSTGRES_TESTS") != "1",
    reason="set RUN_POSTGRES_TESTS=1 to run PostgreSQL integration tests",
)


@pytest.mark.asyncio
async def test_parent_evidence_is_shared_with_a_section_by_stable_id() -> None:
    repository = PostgresRunRepository()
    suffix = uuid.uuid4().hex
    session_id = f"session_coordination_{suffix}"
    run_id = f"run_coordination_{suffix}"
    now = datetime.now(timezone.utc)
    await repository.open()
    try:
        await repository.setup()
        await repository.create_session(session_id=session_id, title="Coordination test")
        await repository.create_run(
            run_id=run_id,
            session_id=session_id,
            parent_run_id=None,
            parent_context=None,
            question="How should shared research artifacts work?",
        )

        version = await repository.save_coordination_unit({
            "run_id": run_id,
            "scope_type": "parent_run",
            "scope_id": "parent_seed",
            "source_run_id": "parent_seed",
            "summary": "Inherited findings",
            "expected_workspace_version": 0,
            "documents": [{
                "document_id": "doc_parent",
                "title": "Parent source",
                "source_type": "knowledge_service",
                "retrieved_at": now,
            }],
            "evidence": [{
                "evidence_id": "evidence_parent",
                "document_id": "doc_parent",
                "excerpt": "A parent finding with an exact source excerpt.",
                "excerpt_hash": "parent_hash",
            }],
            "claims": [{
                "claim_id": "parent:c1",
                "statement": "A parent finding can be inherited.",
                "origin_run_id": "parent_seed",
                "claim_type": "fact",
                "status": "accepted",
                "evidence_bindings": [{
                    "evidence_id": "evidence_parent",
                    "support_status": "supports",
                }],
            }],
        })
        assert version == 1

        version = await repository.save_coordination_unit({
            "run_id": run_id,
            "scope_type": "section",
            "scope_id": "section_1",
            "source_run_id": run_id,
            "summary": "Section reuses the inherited evidence.",
            "expected_workspace_version": 1,
            "referenced_evidence_ids": ["evidence_parent"],
            "dependency_claim_ids": ["parent:c1"],
            "claims": [{
                "claim_id": "section_1:v1:c1",
                "statement": "The inherited finding remains relevant.",
                "origin_run_id": run_id,
                "origin_section_id": "section_1",
                "claim_type": "inference",
                "status": "accepted",
                "evidence_bindings": [{
                    "evidence_id": "evidence_parent",
                    "support_status": "supports",
                }],
            }],
        })
        assert version == 2

        snapshot = await repository.get_coordination_snapshot(run_id)
        assert snapshot is not None
        assert snapshot.workspace_version == 2
        assert snapshot.parent_run.claims[0]["claim_id"] == "parent:c1"
        assert snapshot.sections[0].dependency_claim_ids == ["parent:c1"]
        assert snapshot.sections[0].claims[0]["evidence_bindings"][0][
            "evidence_id"
        ] == "evidence_parent"

        with pytest.raises(WorkspaceVersionConflict):
            await repository.save_coordination_unit({
                "run_id": run_id,
                "scope_type": "section",
                "scope_id": "section_2",
                "source_run_id": run_id,
                "expected_workspace_version": 1,
            })
    finally:
        pool = repository._require_pool()
        async with pool.connection() as conn:
            await conn.execute("DELETE FROM research_runs WHERE run_id = %s", (run_id,))
            await conn.execute(
                "DELETE FROM research_sessions WHERE session_id = %s",
                (session_id,),
            )
        await repository.close()


@pytest.mark.asyncio
async def test_section_projection_reuses_evidence_and_propagates_stale() -> None:
    repository = PostgresRunRepository()
    suffix = uuid.uuid4().hex
    session_id = f"session_projection_{suffix}"
    run_id = f"run_projection_{suffix}"
    await repository.open()
    try:
        await repository.setup()
        await repository.create_session(session_id=session_id, title="Projection test")
        await repository.create_run(
            run_id=run_id,
            session_id=session_id,
            parent_run_id=None,
            parent_context=None,
            question="How should chapter evidence be coordinated?",
        )
        source = {
            "query": "shared fact",
            "source": "knowledge",
            "content": "A shared exact evidence excerpt.",
            "score": 0.9,
            "iteration": 0,
            "metadata": {
                "source": "shared.pdf",
                "chunk_id": "shared::1",
                "retrieved_at": "2026-10-09T10:00:00+00:00",
            },
        }
        first = SectionRecord(
            section_id="section_1",
            title="First",
            question="What is the first finding?",
            status="complete",
            revision=1,
            draft="The first finding is supported[来源1].",
            sources=[source],
            review={"verdict": "pass", "summary": "First summary"},
            claims=[{
                "claim_id": "section_1:v1:c1",
                "statement": "The first finding is supported.",
                "draft_quote": "The first finding is supported",
                "assessment": "supported",
                "evidence": [{
                    "source_number": 1,
                    "quote": "A shared exact evidence excerpt.",
                    "relation": "supports",
                    "evidence_id": evidence_key(source),
                }],
            }],
        )
        await repository.save_sections(run_id, [first.model_dump(mode="json")])

        shared_source = {
            **source,
            "metadata": {**source["metadata"], "shared_from_section": "section_1"},
        }
        second = SectionRecord(
            section_id="section_2",
            title="Second",
            question="What follows from the first finding?",
            status="complete",
            revision=1,
            draft="The second finding reuses evidence[来源1].",
            sources=[shared_source],
            review={"verdict": "pass", "summary": "Second summary"},
            claims=[{
                "claim_id": "section_2:v1:c1",
                "statement": "The second finding reuses evidence.",
                "draft_quote": "The second finding reuses evidence",
                "assessment": "supported",
                "evidence": [{
                    "source_number": 1,
                    "quote": "A shared exact evidence excerpt.",
                    "relation": "supports",
                    "evidence_id": evidence_key(shared_source),
                }],
            }],
        )
        await repository.save_sections(
            run_id,
            [first.model_dump(mode="json"), second.model_dump(mode="json")],
        )
        snapshot = await repository.get_coordination_snapshot(run_id)
        assert snapshot.workspace_version == 2
        assert snapshot.sections[1].dependency_claim_ids == [
            f"{run_id}:section_1:v1:c1"
        ]

        first.claims[0].statement = "The corrected first finding is supported."
        first.revision = 2
        await repository.save_sections(
            run_id,
            [first.model_dump(mode="json"), second.model_dump(mode="json")],
        )
        snapshot = await repository.get_coordination_snapshot(run_id)
        assert snapshot.status == "stale"
        assert snapshot.sections[1].status == "stale"
        assert snapshot.sections[1].claims[0]["evidence_bindings"][0][
            "evidence_id"
        ] == evidence_key(shared_source)

        first.revision = 3
        first.sources = []
        first.claims = []
        first.review.summary = "The old evidence was withdrawn."
        await repository.save_sections(
            run_id,
            [first.model_dump(mode="json"), second.model_dump(mode="json")],
        )
        snapshot = await repository.get_coordination_snapshot(run_id)
        first_evidence = snapshot.sections[0].evidence
        assert first_evidence[0]["status"] == "stale"
        assert snapshot.sections[1].claims[0]["evidence_bindings"][0][
            "evidence_id"
        ] == evidence_key(shared_source)
    finally:
        pool = repository._require_pool()
        async with pool.connection() as conn:
            await conn.execute("DELETE FROM research_runs WHERE run_id = %s", (run_id,))
            await conn.execute(
                "DELETE FROM research_sessions WHERE session_id = %s",
                (session_id,),
            )
        await repository.close()


@pytest.mark.asyncio
async def test_child_workspace_imports_parent_claim_summary_without_report_excerpt() -> None:
    repository = PostgresRunRepository()
    suffix = uuid.uuid4().hex
    session_id = f"session_parent_projection_{suffix}"
    parent_run_id = f"run_parent_projection_{suffix}"
    child_run_id = f"run_child_projection_{suffix}"
    await repository.open()
    try:
        await repository.setup()
        await repository.create_session(session_id=session_id, title="Parent projection")
        await repository.create_run(
            run_id=parent_run_id,
            session_id=session_id,
            parent_run_id=None,
            parent_context=None,
            question="What did the parent research establish?",
        )
        source = {
            "query": "parent evidence",
            "source": "knowledge",
            "content": "The parent source directly supports the finding.",
            "score": 0.9,
            "iteration": 0,
            "metadata": {"source": "parent.pdf", "chunk_id": "parent::1"},
        }
        section = SectionRecord(
            section_id="section_1",
            title="Parent chapter",
            question="What is the parent finding?",
            status="complete",
            revision=1,
            draft="The parent finding is supported[来源1].",
            sources=[source],
            review={"verdict": "pass", "summary": "Model-written section summary"},
            claims=[{
                "claim_id": "section_1:v1:c1",
                "statement": "The parent finding is supported.",
                "draft_quote": "The parent finding is supported",
                "assessment": "supported",
                "evidence": [{
                    "source_number": 1,
                    "quote": "The parent source directly supports the finding.",
                    "relation": "supports",
                    "evidence_id": evidence_key(source),
                }],
            }],
        )
        await repository.save_sections(parent_run_id, [section.model_dump(mode="json")])
        await repository.complete_run(parent_run_id, "TRUNCATED REPORT TEXT")
        parent_snapshot = await repository.get_coordination_snapshot(parent_run_id)
        context = ParentContextSnapshot(
            schema_version=2,
            source_run_id=parent_run_id,
            source_question="What did the parent research establish?",
            report_excerpt="TRUNCATED REPORT TEXT",
            report_truncated=False,
            captured_at=datetime.now(timezone.utc),
            handoff=parent_handoff([section], None),
        )
        await repository.create_run(
            run_id=child_run_id,
            session_id=session_id,
            parent_run_id=parent_run_id,
            parent_context=context,
            question="How does the parent finding affect the child question?",
        )

        child_snapshot = await repository.get_coordination_snapshot(child_run_id)
        assert child_snapshot.parent_run is not None
        assert child_snapshot.parent_run.summary == parent_snapshot.summary
        assert "TRUNCATED REPORT TEXT" not in child_snapshot.parent_run.summary
        assert child_snapshot.parent_run.summary_claim_ids == [
            f"{parent_run_id}:section_1:v1:c1"
        ]
    finally:
        pool = repository._require_pool()
        async with pool.connection() as conn:
            await conn.execute(
                "DELETE FROM research_runs WHERE run_id = %s", (child_run_id,)
            )
            await conn.execute(
                "DELETE FROM research_runs WHERE run_id = %s", (parent_run_id,)
            )
            await conn.execute(
                "DELETE FROM research_sessions WHERE session_id = %s", (session_id,)
            )
        await repository.close()


@pytest.mark.asyncio
async def test_multiple_quotes_for_one_evidence_persist_as_one_binding() -> None:
    repository = PostgresRunRepository()
    suffix = uuid.uuid4().hex
    session_id = f"session_multi_quote_{suffix}"
    run_id = f"run_multi_quote_{suffix}"
    await repository.open()
    try:
        await repository.setup()
        await repository.create_session(session_id=session_id, title="Multi quote")
        await repository.create_run(
            run_id=run_id,
            session_id=session_id,
            parent_run_id=None,
            parent_context=None,
            question="Can one source support a claim with multiple quotes?",
        )
        source = {
            "query": "integrated drive systems",
            "source": "knowledge",
            "content": "First exact quote. Second exact quote.",
            "score": 0.9,
            "iteration": 0,
            "metadata": {"source": "integration.pdf", "chunk_id": "integration::1"},
        }
        section = SectionRecord(
            section_id="section_1",
            title="Integration",
            question="How are drive systems integrated?",
            status="complete",
            revision=1,
            draft="Integration is accelerating[来源1].",
            sources=[source],
            review={"verdict": "pass", "summary": "Integration summary"},
            claims=[{
                "claim_id": "section_1:v1:c1",
                "statement": "Integration is accelerating.",
                "draft_quote": "Integration is accelerating",
                "assessment": "supported",
                "evidence": [{
                    "source_number": 1,
                    "quote": "First exact quote.",
                    "relation": "supports",
                    "evidence_id": evidence_key(source),
                    "quote_span": {"start": 0, "end": 18, "match": "exact"},
                }, {
                    "source_number": 1,
                    "quote": "Second exact quote.",
                    "relation": "supports",
                    "evidence_id": evidence_key(source),
                    "quote_span": {"start": 19, "end": 38, "match": "exact"},
                }],
            }],
        )

        await repository.save_sections(run_id, [section.model_dump(mode="json")])
        snapshot = await repository.get_coordination_snapshot(run_id)
        bindings = snapshot.sections[0].claims[0]["evidence_bindings"]

        assert len(bindings) == 1
        assert len(bindings[0]["quote_refs"]) == 2
        assert bindings[0]["quote_refs"][1]["quote"] == "Second exact quote."
    finally:
        pool = repository._require_pool()
        async with pool.connection() as conn:
            await conn.execute("DELETE FROM research_runs WHERE run_id = %s", (run_id,))
            await conn.execute(
                "DELETE FROM research_sessions WHERE session_id = %s",
                (session_id,),
            )
        await repository.close()
