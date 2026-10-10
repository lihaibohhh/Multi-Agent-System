"""Atomic persistence for the run-level coordination workspace."""

from __future__ import annotations

import hashlib
import json

from psycopg.types.json import Jsonb

from .models import CoordinationSnapshot, CoordinationUnit, CoordinationUnitWrite
from .projection import project_parent_context, project_section


class WorkspaceVersionConflict(RuntimeError):
    """A section tried to commit against an obsolete workspace snapshot."""


def _unit_fingerprint(unit: CoordinationUnitWrite) -> str:
    payload = unit.model_dump(mode="json", exclude={"expected_workspace_version"})
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


async def replace_coordination_unit(conn, raw_unit: CoordinationUnitWrite | dict) -> int:
    """Replace one scope atomically and return the new workspace version.

    The caller must provide a transaction.  Locally-owned evidence is replaced
    with the scope; references may target evidence owned by another scope in the
    same Run workspace.
    """

    unit = CoordinationUnitWrite.model_validate(raw_unit)
    await conn.execute(
        """
        INSERT INTO research_workspaces (run_id)
        VALUES (%s) ON CONFLICT (run_id) DO NOTHING
        """,
        (unit.run_id,),
    )
    workspace = await (
        await conn.execute(
            """
            SELECT workspace_version FROM research_workspaces
            WHERE run_id = %s FOR UPDATE
            """,
            (unit.run_id,),
        )
    ).fetchone()
    current_version = workspace["workspace_version"]
    fingerprint = _unit_fingerprint(unit)
    previous_scope = await (
        await conn.execute(
            """
            SELECT artifact_fingerprint FROM research_coordination_scopes
            WHERE run_id = %s AND scope_type = %s AND scope_id = %s
            """,
            (unit.run_id, unit.scope_type, unit.scope_id),
        )
    ).fetchone()
    if previous_scope and previous_scope["artifact_fingerprint"] == fingerprint:
        return current_version
    if current_version != unit.expected_workspace_version:
        raise WorkspaceVersionConflict(
            f"workspace version changed: expected {unit.expected_workspace_version}, "
            f"found {current_version}"
        )

    if unit.referenced_evidence_ids:
        rows = await (
            await conn.execute(
                """
                SELECT evidence_id FROM research_evidence
                WHERE run_id = %s AND evidence_id = ANY(%s)
                  AND NOT (scope_type = %s AND scope_id = %s)
                """,
                (
                    unit.run_id,
                    unit.referenced_evidence_ids,
                    unit.scope_type,
                    unit.scope_id,
                ),
            )
        ).fetchall()
        found = {row["evidence_id"] for row in rows}
        missing = set(unit.referenced_evidence_ids) - found
        if missing:
            raise ValueError(
                f"referenced evidence is absent or owned by the replaced scope: {sorted(missing)}"
            )

    previous_claim_rows = await (
        await conn.execute(
            """
            SELECT claim_id, statement, status, revision FROM research_claims
            WHERE run_id = %s AND scope_type = %s AND scope_id = %s
            """,
            (unit.run_id, unit.scope_type, unit.scope_id),
        )
    ).fetchall()
    previous_claims = {
        row["claim_id"]: (row["statement"], row["status"], row["revision"])
        for row in previous_claim_rows
    }
    new_claims = {
        claim.claim_id: (claim.statement, claim.status, claim.revision)
        for claim in unit.claims
    }
    changed_claim_ids = sorted(
        claim_id
        for claim_id in set(previous_claims) | set(new_claims)
        if previous_claims.get(claim_id) != new_claims.get(claim_id)
    )
    previous_evidence_rows = await (
        await conn.execute(
            """
            SELECT evidence_id FROM research_evidence
            WHERE run_id = %s AND scope_type = %s AND scope_id = %s
            """,
            (unit.run_id, unit.scope_type, unit.scope_id),
        )
    ).fetchall()
    previous_evidence_ids = [row["evidence_id"] for row in previous_evidence_rows]
    external_claim_bindings = []
    external_metric_bindings = []
    if previous_evidence_ids:
        external_claim_bindings = await (
            await conn.execute(
                """
                SELECT b.* FROM research_claim_evidence_bindings b
                JOIN research_claims c
                  ON c.run_id = b.run_id AND c.claim_id = b.claim_id
                WHERE b.run_id = %s AND b.evidence_id = ANY(%s)
                  AND NOT (c.scope_type = %s AND c.scope_id = %s)
                """,
                (
                    unit.run_id,
                    previous_evidence_ids,
                    unit.scope_type,
                    unit.scope_id,
                ),
            )
        ).fetchall()
        external_metric_bindings = await (
            await conn.execute(
                """
                SELECT b.* FROM research_metric_evidence_bindings b
                JOIN research_metrics m
                  ON m.run_id = b.run_id AND m.metric_id = b.metric_id
                WHERE b.run_id = %s AND b.evidence_id = ANY(%s)
                  AND NOT (m.scope_type = %s AND m.scope_id = %s)
                """,
                (
                    unit.run_id,
                    previous_evidence_ids,
                    unit.scope_type,
                    unit.scope_id,
                ),
            )
        ).fetchall()

    await conn.execute(
        """
        DELETE FROM research_coordination_scopes
        WHERE run_id = %s AND scope_type = %s AND scope_id = %s
        """,
        (unit.run_id, unit.scope_type, unit.scope_id),
    )
    await conn.execute(
        """
        INSERT INTO research_coordination_scopes (
            run_id, scope_type, scope_id, source_run_id, revision, summary,
            summary_claim_ids, summary_metric_ids, artifact_fingerprint, status,
            dependency_claim_ids, open_questions
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """,
        (
            unit.run_id,
            unit.scope_type,
            unit.scope_id,
            unit.source_run_id,
            unit.revision,
            unit.summary,
            Jsonb(unit.summary_claim_ids),
            Jsonb(unit.summary_metric_ids),
            fingerprint,
            unit.status,
            Jsonb(unit.dependency_claim_ids),
            Jsonb(unit.open_questions),
        ),
    )

    for document in unit.documents:
        await conn.execute(
            """
            INSERT INTO research_workspace_documents (
                run_id, document_id, canonical_url, title, publisher, author,
                published_at, source_type, content_hash, retrieved_at, metadata
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (run_id, document_id) DO UPDATE SET
                canonical_url = EXCLUDED.canonical_url,
                title = EXCLUDED.title,
                publisher = EXCLUDED.publisher,
                author = EXCLUDED.author,
                published_at = EXCLUDED.published_at,
                source_type = EXCLUDED.source_type,
                content_hash = EXCLUDED.content_hash,
                retrieved_at = EXCLUDED.retrieved_at,
                metadata = EXCLUDED.metadata
            """,
            (
                unit.run_id,
                document.document_id,
                document.canonical_url,
                document.title,
                document.publisher,
                document.author,
                document.published_at,
                document.source_type,
                document.content_hash,
                document.retrieved_at,
                Jsonb(document.metadata),
            ),
        )

    for evidence in unit.evidence:
        await conn.execute(
            """
            INSERT INTO research_evidence (
                run_id, evidence_id, scope_type, scope_id, document_id,
                excerpt, excerpt_hash, locator, retrieval_query, status
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (run_id, evidence_id) DO UPDATE SET
                scope_type = EXCLUDED.scope_type,
                scope_id = EXCLUDED.scope_id,
                document_id = EXCLUDED.document_id,
                excerpt = EXCLUDED.excerpt,
                excerpt_hash = EXCLUDED.excerpt_hash,
                locator = EXCLUDED.locator,
                retrieval_query = EXCLUDED.retrieval_query,
                status = EXCLUDED.status
            """,
            (
                unit.run_id,
                evidence.evidence_id,
                unit.scope_type,
                unit.scope_id,
                evidence.document_id,
                evidence.excerpt,
                evidence.excerpt_hash,
                evidence.locator,
                evidence.retrieval_query,
                evidence.status,
            ),
        )

    removed_evidence_ids = set(previous_evidence_ids) - {
        item.evidence_id for item in unit.evidence
    }
    if removed_evidence_ids:
        await conn.execute(
            """
            UPDATE research_evidence SET status = 'stale'
            WHERE run_id = %s AND evidence_id = ANY(%s)
            """,
            (unit.run_id, sorted(removed_evidence_ids)),
        )

    for claim in unit.claims:
        await conn.execute(
            """
            INSERT INTO research_claims (
                run_id, claim_id, scope_type, scope_id, statement, origin_run_id,
                origin_section_id, origin_agent_run_id, claim_type, status,
                confidence, revision, subject, time_scope, geography_scope,
                industry_scope, visibility, caveat
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                unit.run_id,
                claim.claim_id,
                unit.scope_type,
                unit.scope_id,
                claim.statement,
                claim.origin_run_id,
                claim.origin_section_id,
                claim.origin_agent_run_id,
                claim.claim_type,
                claim.status,
                claim.confidence,
                claim.revision,
                claim.subject,
                claim.time_scope,
                claim.geography_scope,
                claim.industry_scope,
                claim.visibility,
                claim.caveat,
            ),
        )
        for binding in claim.evidence_bindings:
            await conn.execute(
                """
                INSERT INTO research_claim_evidence_bindings (
                    run_id, claim_id, evidence_id, support_status, reason,
                    required_supplement, created_by_agent_run_id, quote_refs
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (run_id, claim_id, evidence_id) DO UPDATE SET
                    support_status = EXCLUDED.support_status,
                    reason = EXCLUDED.reason,
                    required_supplement = EXCLUDED.required_supplement,
                    created_by_agent_run_id = EXCLUDED.created_by_agent_run_id,
                    quote_refs = EXCLUDED.quote_refs
                """,
                (
                    unit.run_id,
                    claim.claim_id,
                    binding.evidence_id,
                    binding.support_status,
                    binding.reason,
                    binding.required_supplement,
                    binding.created_by_agent_run_id,
                    Jsonb([
                        quote.model_dump(mode="json") for quote in binding.quote_refs
                    ]),
                ),
            )

    for metric in unit.metrics:
        await conn.execute(
            """
            INSERT INTO research_metrics (
                run_id, metric_id, scope_type, scope_id, claim_id, metric_name,
                value_numeric, value_text, unit, period, geography, population,
                sample_scope, numerator_definition, denominator_definition,
                methodology, status
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                unit.run_id,
                metric.metric_id,
                unit.scope_type,
                unit.scope_id,
                metric.claim_id,
                metric.metric_name,
                metric.value_numeric,
                metric.value_text,
                metric.unit,
                metric.period,
                metric.geography,
                metric.population,
                metric.sample_scope,
                metric.numerator_definition,
                metric.denominator_definition,
                metric.methodology,
                metric.status,
            ),
        )
        for evidence_id in metric.evidence_ids:
            await conn.execute(
                """
                INSERT INTO research_metric_evidence_bindings (
                    run_id, metric_id, evidence_id
                ) VALUES (%s,%s,%s)
                """,
                (unit.run_id, metric.metric_id, evidence_id),
            )

    current_evidence_ids = {item.evidence_id for item in unit.evidence}
    for binding in external_claim_bindings:
        if binding["evidence_id"] not in current_evidence_ids:
            continue
        await conn.execute(
            """
            INSERT INTO research_claim_evidence_bindings (
                run_id, claim_id, evidence_id, support_status, reason,
                required_supplement, created_by_agent_run_id, quote_refs, created_at
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (run_id, claim_id, evidence_id) DO UPDATE SET
                support_status = EXCLUDED.support_status,
                reason = EXCLUDED.reason,
                required_supplement = EXCLUDED.required_supplement,
                created_by_agent_run_id = EXCLUDED.created_by_agent_run_id,
                quote_refs = EXCLUDED.quote_refs
            """,
            (
                binding["run_id"],
                binding["claim_id"],
                binding["evidence_id"],
                binding["support_status"],
                binding["reason"],
                binding["required_supplement"],
                binding["created_by_agent_run_id"],
                Jsonb(binding.get("quote_refs") or []),
                binding["created_at"],
            ),
        )
    for binding in external_metric_bindings:
        if binding["evidence_id"] not in current_evidence_ids:
            continue
        await conn.execute(
            """
            INSERT INTO research_metric_evidence_bindings (
                run_id, metric_id, evidence_id
            ) VALUES (%s,%s,%s) ON CONFLICT DO NOTHING
            """,
            (binding["run_id"], binding["metric_id"], binding["evidence_id"]),
        )

    # Only remove documents no longer used by any scope in this workspace.
    await conn.execute(
        """
        DELETE FROM research_workspace_documents d
        WHERE d.run_id = %s AND NOT EXISTS (
            SELECT 1 FROM research_evidence e
            WHERE e.run_id = d.run_id AND e.document_id = d.document_id
        )
        """,
        (unit.run_id,),
    )
    stale_count = 0
    if changed_claim_ids:
        stale = await (
            await conn.execute(
                """
                UPDATE research_coordination_scopes dependent
                SET status = 'stale', updated_at = NOW()
                WHERE dependent.run_id = %s
                  AND NOT (
                    dependent.scope_type = %s AND dependent.scope_id = %s
                  )
                  AND EXISTS (
                    SELECT 1
                    FROM jsonb_array_elements_text(
                        dependent.dependency_claim_ids
                    ) AS dependency(claim_id)
                    WHERE dependency.claim_id = ANY(%s)
                  )
                  AND dependent.status <> 'stale'
                RETURNING scope_id
                """,
                (
                    unit.run_id,
                    unit.scope_type,
                    unit.scope_id,
                    changed_claim_ids,
                ),
            )
        ).fetchall()
        stale_count = len(stale)

    row = await (
        await conn.execute(
            """
            UPDATE research_workspaces
            SET workspace_version = workspace_version + 1,
                status = %s, updated_at = NOW()
            WHERE run_id = %s
            RETURNING workspace_version
            """,
            ("stale" if stale_count else "collecting", unit.run_id),
        )
    ).fetchone()
    return row["workspace_version"]


async def initialize_coordination_workspace(
    conn,
    run_id: str,
    parent_context: dict | None = None,
    *,
    parent_workspace_summary: str = "",
) -> int:
    """Create one empty workspace and optionally import its immutable parent handoff."""

    await conn.execute(
        """
        INSERT INTO research_workspaces (run_id)
        VALUES (%s) ON CONFLICT (run_id) DO NOTHING
        """,
        (run_id,),
    )
    row = await (
        await conn.execute(
            "SELECT workspace_version FROM research_workspaces WHERE run_id = %s",
            (run_id,),
        )
    ).fetchone()
    existing_parent = await (
        await conn.execute(
            """
            SELECT 1 FROM research_coordination_scopes
            WHERE run_id = %s AND scope_type = 'parent_run' LIMIT 1
            """,
            (run_id,),
        )
    ).fetchone()
    if existing_parent:
        return row["workspace_version"]
    unit = project_parent_context(
        run_id,
        parent_context,
        expected_workspace_version=row["workspace_version"],
        parent_workspace_summary=parent_workspace_summary,
    )
    if unit is None:
        return row["workspace_version"]
    return await replace_coordination_unit(conn, unit)


async def _reuse_existing_evidence(conn, unit: CoordinationUnitWrite):
    evidence_ids = [item.evidence_id for item in unit.evidence]
    if not evidence_ids:
        return unit
    rows = await (
        await conn.execute(
            """
            SELECT evidence_id FROM research_evidence
            WHERE run_id = %s AND evidence_id = ANY(%s)
              AND NOT (scope_type = %s AND scope_id = %s)
            """,
            (unit.run_id, evidence_ids, unit.scope_type, unit.scope_id),
        )
    ).fetchall()
    referenced = {row["evidence_id"] for row in rows}
    if not referenced:
        return unit
    owned_evidence = [
        item for item in unit.evidence if item.evidence_id not in referenced
    ]
    owned_document_ids = {item.document_id for item in owned_evidence}
    return unit.model_copy(
        update={
            "documents": [
                item for item in unit.documents
                if item.document_id in owned_document_ids
            ],
            "evidence": owned_evidence,
            "referenced_evidence_ids": sorted(
                set(unit.referenced_evidence_ids) | referenced
            ),
        }
    )


async def sync_section_artifacts(
    conn,
    run_id: str,
    sections: list[dict],
    parent_context: dict | None,
) -> int:
    """Idempotently project finished/stale chapter snapshots into the workspace."""

    await initialize_coordination_workspace(conn, run_id, parent_context)
    version_row = await (
        await conn.execute(
            "SELECT workspace_version FROM research_workspaces WHERE run_id = %s",
            (run_id,),
        )
    ).fetchone()
    version = version_row["workspace_version"]
    for raw in sections:
        if raw.get("status") not in {"complete", "limited", "stale"}:
            continue
        unit = project_section(
            run_id,
            raw,
            sections,
            parent_context,
            expected_workspace_version=version,
        )
        unit = await _reuse_existing_evidence(conn, unit)
        version = await replace_coordination_unit(conn, unit)
    return version


async def finalize_coordination_workspace(
    conn,
    run_id: str,
    *,
    report_quality: str = "reviewed",
) -> None:
    """Build a deterministic Run summary from accepted, non-stale chapter Claims."""

    rows = await (
        await conn.execute(
            """
            SELECT c.statement
            FROM research_claims c
            JOIN research_coordination_scopes s
              ON s.run_id = c.run_id AND s.scope_type = c.scope_type
             AND s.scope_id = c.scope_id
            WHERE c.run_id = %s AND c.scope_type = 'section'
              AND c.status = 'accepted' AND s.status <> 'stale'
            ORDER BY c.scope_id, c.claim_id
            LIMIT 12
            """,
            (run_id,),
        )
    ).fetchall()
    stale = await (
        await conn.execute(
            """
            SELECT 1 FROM research_coordination_scopes
            WHERE run_id = %s AND status = 'stale' LIMIT 1
            """,
            (run_id,),
        )
    ).fetchone()
    summary = "\n".join(f"- {row['statement']}" for row in rows)
    await conn.execute(
        """
        UPDATE research_workspaces
        SET summary = %s, status = %s, updated_at = NOW()
        WHERE run_id = %s
        """,
        (
            summary,
            "stale" if stale else "ready" if report_quality == "reviewed" else "coordinating",
            run_id,
        ),
    )


async def load_coordination_snapshot(conn, run_id: str) -> CoordinationSnapshot | None:
    workspace = await (
        await conn.execute(
            "SELECT * FROM research_workspaces WHERE run_id = %s",
            (run_id,),
        )
    ).fetchone()
    if workspace is None:
        return None
    rows = await (
        await conn.execute(
            """
            SELECT * FROM research_coordination_units
            WHERE run_id = %s
            ORDER BY CASE WHEN scope_type = 'parent_run' THEN 0 ELSE 1 END, scope_id
            """,
            (run_id,),
        )
    ).fetchall()
    units = [CoordinationUnit.model_validate(row) for row in rows]
    parents = [unit for unit in units if unit.scope_type == "parent_run"]
    return CoordinationSnapshot(
        run_id=run_id,
        schema_version=workspace["schema_version"],
        workspace_version=workspace["workspace_version"],
        status=workspace["status"],
        summary=workspace["summary"],
        parent_run=parents[0] if parents else None,
        sections=[unit for unit in units if unit.scope_type == "section"],
        updated_at=workspace["updated_at"],
    )
