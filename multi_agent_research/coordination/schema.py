"""Idempotent PostgreSQL schema for versioned cross-section research artifacts."""

from __future__ import annotations


async def setup_coordination_schema(conn) -> None:
    """Create normalized tables and the read-only Agent coordination view."""

    statements = (
        """
        CREATE TABLE IF NOT EXISTS research_workspaces (
            run_id VARCHAR(128) PRIMARY KEY
                REFERENCES research_runs(run_id) ON DELETE CASCADE,
            schema_version INTEGER NOT NULL DEFAULT 1 CHECK (schema_version = 1),
            workspace_version BIGINT NOT NULL DEFAULT 0 CHECK (workspace_version >= 0),
            status VARCHAR(32) NOT NULL DEFAULT 'collecting' CHECK (
                status IN ('collecting','coordinating','ready','stale')
            ),
            summary TEXT NOT NULL DEFAULT '',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS research_coordination_scopes (
            run_id VARCHAR(128) NOT NULL
                REFERENCES research_workspaces(run_id) ON DELETE CASCADE,
            scope_type VARCHAR(32) NOT NULL CHECK (scope_type IN ('parent_run','section')),
            scope_id VARCHAR(128) NOT NULL,
            source_run_id VARCHAR(128) NOT NULL,
            revision INTEGER NOT NULL CHECK (revision >= 1),
            summary TEXT NOT NULL DEFAULT '',
            summary_claim_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
            summary_metric_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
            artifact_fingerprint VARCHAR(64),
            status VARCHAR(32) NOT NULL DEFAULT 'provisional' CHECK (
                status IN ('provisional','coordinated','accepted','stale')
            ),
            dependency_claim_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
            open_questions JSONB NOT NULL DEFAULT '[]'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (run_id, scope_type, scope_id),
            CHECK (jsonb_typeof(summary_claim_ids) = 'array'),
            CHECK (jsonb_typeof(summary_metric_ids) = 'array'),
            CHECK (jsonb_typeof(dependency_claim_ids) = 'array'),
            CHECK (jsonb_typeof(open_questions) = 'array')
        )
        """,
        """
        CREATE UNIQUE INDEX IF NOT EXISTS research_coordination_one_parent_idx
        ON research_coordination_scopes(run_id)
        WHERE scope_type = 'parent_run'
        """,
        """
        CREATE TABLE IF NOT EXISTS research_workspace_documents (
            run_id VARCHAR(128) NOT NULL
                REFERENCES research_workspaces(run_id) ON DELETE CASCADE,
            document_id VARCHAR(256) NOT NULL,
            canonical_url TEXT,
            title TEXT NOT NULL,
            publisher TEXT,
            author TEXT,
            published_at TIMESTAMPTZ,
            source_type VARCHAR(64) NOT NULL,
            content_hash VARCHAR(128),
            retrieved_at TIMESTAMPTZ NOT NULL,
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            PRIMARY KEY (run_id, document_id),
            CHECK (jsonb_typeof(metadata) = 'object')
        )
        """,
        """
        ALTER TABLE research_workspace_documents
        ALTER COLUMN retrieved_at DROP NOT NULL
        """,
        """
        ALTER TABLE research_coordination_scopes
        ADD COLUMN IF NOT EXISTS summary_claim_ids JSONB NOT NULL DEFAULT '[]'::jsonb
        """,
        """
        ALTER TABLE research_coordination_scopes
        ADD COLUMN IF NOT EXISTS summary_metric_ids JSONB NOT NULL DEFAULT '[]'::jsonb
        """,
        """
        ALTER TABLE research_coordination_scopes
        ADD COLUMN IF NOT EXISTS artifact_fingerprint VARCHAR(64)
        """,
        """
        CREATE INDEX IF NOT EXISTS research_workspace_documents_hash_idx
        ON research_workspace_documents(run_id, content_hash)
        """,
        """
        CREATE TABLE IF NOT EXISTS research_evidence (
            run_id VARCHAR(128) NOT NULL,
            evidence_id VARCHAR(256) NOT NULL,
            scope_type VARCHAR(32) NOT NULL,
            scope_id VARCHAR(128) NOT NULL,
            document_id VARCHAR(256) NOT NULL,
            excerpt TEXT NOT NULL,
            excerpt_hash VARCHAR(128) NOT NULL,
            locator TEXT,
            retrieval_query TEXT,
            status VARCHAR(32) NOT NULL DEFAULT 'active' CHECK (
                status IN ('active','stale','unavailable')
            ),
            PRIMARY KEY (run_id, evidence_id),
            FOREIGN KEY (run_id)
                REFERENCES research_workspaces(run_id) ON DELETE CASCADE,
            FOREIGN KEY (run_id, document_id)
                REFERENCES research_workspace_documents(run_id, document_id)
                ON DELETE RESTRICT
        )
        """,
        """
        ALTER TABLE research_evidence
        DROP CONSTRAINT IF EXISTS research_evidence_run_id_scope_type_scope_id_fkey
        """,
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conname = 'research_evidence_workspace_fkey'
            ) THEN
                ALTER TABLE research_evidence
                ADD CONSTRAINT research_evidence_workspace_fkey
                FOREIGN KEY (run_id) REFERENCES research_workspaces(run_id)
                ON DELETE CASCADE;
            END IF;
        END $$
        """,
        """
        CREATE INDEX IF NOT EXISTS research_evidence_scope_idx
        ON research_evidence(run_id, scope_type, scope_id)
        """,
        """
        CREATE TABLE IF NOT EXISTS research_claims (
            run_id VARCHAR(128) NOT NULL,
            claim_id VARCHAR(256) NOT NULL,
            scope_type VARCHAR(32) NOT NULL,
            scope_id VARCHAR(128) NOT NULL,
            statement TEXT NOT NULL,
            origin_run_id VARCHAR(128) NOT NULL,
            origin_section_id VARCHAR(128),
            origin_agent_run_id VARCHAR(256),
            claim_type VARCHAR(32) NOT NULL CHECK (
                claim_type IN ('fact','metric','inference','forecast','recommendation')
            ),
            status VARCHAR(32) NOT NULL DEFAULT 'pending' CHECK (
                status IN ('pending','accepted','disputed','rejected','superseded')
            ),
            confidence DOUBLE PRECISION CHECK (
                confidence IS NULL OR (confidence >= 0 AND confidence <= 1)
            ),
            revision INTEGER NOT NULL CHECK (revision >= 1),
            subject TEXT,
            time_scope TEXT,
            geography_scope TEXT,
            industry_scope TEXT,
            visibility VARCHAR(16) NOT NULL DEFAULT 'public' CHECK (
                visibility IN ('public','internal')
            ),
            caveat TEXT NOT NULL DEFAULT '',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (run_id, claim_id),
            FOREIGN KEY (run_id, scope_type, scope_id)
                REFERENCES research_coordination_scopes(run_id, scope_type, scope_id)
                ON DELETE CASCADE
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS research_claims_scope_status_idx
        ON research_claims(run_id, scope_type, scope_id, status)
        """,
        """
        CREATE TABLE IF NOT EXISTS research_claim_evidence_bindings (
            run_id VARCHAR(128) NOT NULL,
            claim_id VARCHAR(256) NOT NULL,
            evidence_id VARCHAR(256) NOT NULL,
            support_status VARCHAR(32) NOT NULL CHECK (
                support_status IN ('supports','contradicts','insufficient','pending')
            ),
            reason TEXT,
            required_supplement TEXT,
            created_by_agent_run_id VARCHAR(256),
            quote_refs JSONB NOT NULL DEFAULT '[]'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (run_id, claim_id, evidence_id),
            FOREIGN KEY (run_id, claim_id)
                REFERENCES research_claims(run_id, claim_id) ON DELETE CASCADE,
            FOREIGN KEY (run_id, evidence_id)
                REFERENCES research_evidence(run_id, evidence_id) ON DELETE CASCADE,
            CHECK (support_status = 'supports' OR NULLIF(BTRIM(reason), '') IS NOT NULL),
            CHECK (jsonb_typeof(quote_refs) = 'array')
        )
        """,
        """
        ALTER TABLE research_claim_evidence_bindings
        ADD COLUMN IF NOT EXISTS quote_refs JSONB NOT NULL DEFAULT '[]'::jsonb
        """,
        """
        CREATE TABLE IF NOT EXISTS research_metrics (
            run_id VARCHAR(128) NOT NULL,
            metric_id VARCHAR(256) NOT NULL,
            scope_type VARCHAR(32) NOT NULL,
            scope_id VARCHAR(128) NOT NULL,
            claim_id VARCHAR(256),
            metric_name TEXT NOT NULL,
            value_numeric NUMERIC,
            value_text TEXT NOT NULL,
            unit VARCHAR(128),
            period VARCHAR(256),
            geography VARCHAR(256),
            population TEXT,
            sample_scope TEXT,
            numerator_definition TEXT,
            denominator_definition TEXT,
            methodology TEXT,
            status VARCHAR(32) NOT NULL DEFAULT 'pending' CHECK (
                status IN ('pending','accepted','disputed','rejected','superseded')
            ),
            PRIMARY KEY (run_id, metric_id),
            FOREIGN KEY (run_id, scope_type, scope_id)
                REFERENCES research_coordination_scopes(run_id, scope_type, scope_id)
                ON DELETE CASCADE,
            FOREIGN KEY (run_id, claim_id)
                REFERENCES research_claims(run_id, claim_id) ON DELETE CASCADE
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS research_metrics_name_scope_idx
        ON research_metrics(run_id, metric_name, period, geography)
        """,
        """
        CREATE TABLE IF NOT EXISTS research_metric_evidence_bindings (
            run_id VARCHAR(128) NOT NULL,
            metric_id VARCHAR(256) NOT NULL,
            evidence_id VARCHAR(256) NOT NULL,
            PRIMARY KEY (run_id, metric_id, evidence_id),
            FOREIGN KEY (run_id, metric_id)
                REFERENCES research_metrics(run_id, metric_id) ON DELETE CASCADE,
            FOREIGN KEY (run_id, evidence_id)
                REFERENCES research_evidence(run_id, evidence_id) ON DELETE CASCADE
        )
        """,
        """
        CREATE OR REPLACE VIEW research_coordination_units AS
        SELECT
            s.run_id,
            s.scope_type,
            s.scope_id,
            s.source_run_id,
            s.revision,
            s.summary,
            s.status,
            s.dependency_claim_ids,
            s.open_questions,
            COALESCE((
                SELECT jsonb_agg(
                    (to_jsonb(e) - 'run_id' - 'scope_type' - 'scope_id') ||
                    jsonb_build_object('document', to_jsonb(d) - 'run_id')
                    ORDER BY e.evidence_id
                )
                FROM research_evidence e
                JOIN research_workspace_documents d
                  ON d.run_id = e.run_id AND d.document_id = e.document_id
                WHERE e.run_id = s.run_id AND e.scope_type = s.scope_type
                  AND e.scope_id = s.scope_id
            ), '[]'::jsonb) AS evidence,
            COALESCE((
                SELECT jsonb_agg(
                    (to_jsonb(c) - 'run_id' - 'scope_type' - 'scope_id') ||
                    jsonb_build_object('evidence_bindings', COALESCE((
                        SELECT jsonb_agg(to_jsonb(b) - 'run_id' - 'claim_id'
                                         ORDER BY b.evidence_id)
                        FROM research_claim_evidence_bindings b
                        WHERE b.run_id = c.run_id AND b.claim_id = c.claim_id
                    ), '[]'::jsonb))
                    ORDER BY c.claim_id
                )
                FROM research_claims c
                WHERE c.run_id = s.run_id AND c.scope_type = s.scope_type
                  AND c.scope_id = s.scope_id
            ), '[]'::jsonb) AS claims,
            COALESCE((
                SELECT jsonb_agg(
                    (to_jsonb(m) - 'run_id' - 'scope_type' - 'scope_id') ||
                    jsonb_build_object('evidence_ids', COALESCE((
                        SELECT jsonb_agg(me.evidence_id ORDER BY me.evidence_id)
                        FROM research_metric_evidence_bindings me
                        WHERE me.run_id = m.run_id AND me.metric_id = m.metric_id
                    ), '[]'::jsonb))
                    ORDER BY m.metric_id
                )
                FROM research_metrics m
                WHERE m.run_id = s.run_id AND m.scope_type = s.scope_type
                  AND m.scope_id = s.scope_id
            ), '[]'::jsonb) AS metrics,
            s.updated_at,
            s.summary_claim_ids,
            s.summary_metric_ids
        FROM research_coordination_scopes s
        """,
    )
    for statement in statements:
        await conn.execute(statement)
