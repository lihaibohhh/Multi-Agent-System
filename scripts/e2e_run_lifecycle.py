"""Run one real research task through the public lifecycle service."""

from __future__ import annotations

import argparse
import asyncio
import json
import time

from multi_agent_research.core.checkpointer import CheckpointerFactory
from multi_agent_research.knowledge.client import get_knowledge_service_client
from multi_agent_research.runs.models import RunStatus
from multi_agent_research.runs.repository import PostgresRunRepository
from multi_agent_research.runs.service import RunService


async def run(
    question: str | None,
    *,
    parent_run_id: str | None,
    resume_run_id: str | None,
    timeout: float,
) -> None:
    repository = PostgresRunRepository()
    service = RunService(repository)
    knowledge_client = get_knowledge_service_client()
    await repository.open()
    try:
        await repository.setup()
        await knowledge_client.ensure_ready()
        await service.recover_stale_runs()
        if resume_run_id:
            created = await service.get_run(resume_run_id)
            await service.start_run(created.run_id, resume=True)
        else:
            if not question:
                raise ValueError("question is required when --resume-run-id is omitted")
            created = await service.create_run(
                question=question,
                parent_run_id=parent_run_id,
            )
            await service.start_run(created.run_id)

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            record = await service.get_run(created.run_id)
            if record.status in {
                RunStatus.COMPLETED,
                RunStatus.FAILED,
                RunStatus.INTERRUPTED,
                RunStatus.CANCELLED,
            }:
                events = await repository.list_events(record.run_id)
                print(json.dumps({
                    "session_id": record.session_id,
                    "run_id": record.run_id,
                    "parent_run_id": record.parent_run_id,
                    "status": record.status.value,
                    "report_chars": len(record.final_report or ""),
                    "event_types": [event.event_type for event in events],
                    "error_type": (
                        events[-1].payload.get("type")
                        if events and events[-1].event_type == "error"
                        else None
                    ),
                }, ensure_ascii=False, indent=2))
                if record.status != RunStatus.COMPLETED:
                    raise SystemExit(1)
                return
            await asyncio.sleep(0.5)
        raise TimeoutError(f"run did not finish within {timeout:.0f}s")
    finally:
        await service.shutdown()
        await repository.close()
        await knowledge_client.aclose()
        await CheckpointerFactory.close_all()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?")
    parser.add_argument("--parent-run-id")
    parser.add_argument("--resume-run-id")
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()
    asyncio.run(
        run(
            args.question,
            parent_run_id=args.parent_run_id,
            resume_run_id=args.resume_run_id,
            timeout=args.timeout,
        )
    )


if __name__ == "__main__":
    main()
