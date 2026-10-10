"""Preview or apply the idempotent legacy shared-budget isolation migration."""

from __future__ import annotations

import argparse
import asyncio
import json

from multi_agent_research.runs.repository import PostgresRunRepository


async def shared_groups(repo: PostgresRunRepository) -> list[dict]:
    async with repo._require_pool().connection() as conn:
        rows = await (
            await conn.execute(
                """
                SELECT budget_id, array_agg(run_id ORDER BY created_at, run_id) AS run_ids
                FROM research_runs
                WHERE budget_id IS NOT NULL
                GROUP BY budget_id
                HAVING COUNT(*) > 1
                ORDER BY budget_id
                """
            )
        ).fetchall()
    return [dict(row) for row in rows]


async def main(*, apply: bool, inspect_run_ids: list[str]) -> None:
    repo = PostgresRunRepository()
    await repo.open()
    try:
        before = await shared_groups(repo)
        affected_run_ids = list(
            dict.fromkeys(
                [run_id for group in before for run_id in group["run_ids"]]
                + inspect_run_ids
            )
        )
        if apply:
            # setup() owns the transactional, idempotent migration. It does not
            # start/resume a Run or make model/retrieval calls.
            await repo.setup()
        after = await shared_groups(repo)
        runs = []
        for run_id in affected_run_ids:
            record = await repo.get_run(run_id)
            runs.append(
                {
                    "run_id": run_id,
                    "budget_id": record.budget_id,
                    "token_limit": record.budget["policy"]["tokens"],
                    "known_tokens": record.budget["known_tokens"],
                    "charged_tokens": record.budget["charged_tokens"],
                    "model_calls": record.budget["model_calls"],
                    "retrieval_calls": record.budget["retrieval_calls"],
                    "history_incomplete": record.budget.get(
                        "legacy_history_incomplete", False
                    ),
                }
            )
        print(
            json.dumps(
                {
                    "applied": apply,
                    "shared_groups_before": before,
                    "shared_groups_after": after,
                    "affected_runs": runs,
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )
    finally:
        await repo.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="apply the migration; without this flag the command is read-only",
    )
    parser.add_argument(
        "--inspect-run",
        action="append",
        default=[],
        help="include a Run's independent account summary in the output",
    )
    arguments = parser.parse_args()
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main(apply=arguments.apply, inspect_run_ids=arguments.inspect_run))
