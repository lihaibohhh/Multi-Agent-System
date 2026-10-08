"""Explicit live two-query recovery check. Private PG schema is rolled back.

Run with --live. Reads the historical Run/checkpoint twice, never resumes it.
Only knowledge-service search (use_query_cache=false); no model/KB writes.
"""
import argparse
import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from uuid import uuid4

from psycopg import AsyncConnection, sql
from psycopg.rows import dict_row

from multi_agent_research.agents.search_agent import _knowledge_search
from multi_agent_research.core.budget import RunBudget, current_budget
from multi_agent_research.core.config import settings
from multi_agent_research.core.retrieval import RetrievalDeferred, durable_retrieval, gather_retrievals
from multi_agent_research.knowledge.client import get_knowledge_service_client
from multi_agent_research.runs.models import RunStatus
from multi_agent_research.runs.repository import PostgresRunRepository
from scripts.reproduce_retrieval_case import read_case, queries_for


def emit(**data):
    print(json.dumps(data, ensure_ascii=False), flush=True)


async def main():
    state, signature = await read_case()
    queries = queries_for(state)[:2]
    assert len(queries) == 2
    calls = {"real_primary": 0, "injected_timeout": 0, "real_secondary": 0}
    primary_done = asyncio.Event()
    started = time.monotonic()
    async def primary():
        calls["real_primary"] += 1
        result = await _knowledge_search(queries[0], 0)
        emit(event="real_primary_done", chunks=len(result))
        primary_done.set()
        return result
    async def injected():
        await primary_done.wait()
        calls["injected_timeout"] += 1
        raise TimeoutError("INJECTED sibling timeout, not a natural service outage")
    async def secondary():
        calls["real_secondary"] += 1
        result = await _knowledge_search(queries[1], 0)
        emit(event="real_secondary_done", chunks=len(result))
        return result
    descriptors = [{"provider": "knowledge", "query": q, "section_id": "case-s2", "round": 1,
                    "revision": 0, "format_version": 1, "use_query_cache": False} for q in queries]
    try:
        async with await AsyncConnection.connect(settings.database.url, autocommit=True,
                                                  row_factory=dict_row, connect_timeout=5) as conn:
            async with conn.transaction(force_rollback=True):
                schema = "retrieval_live_test_" + uuid4().hex
                await conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
                await conn.execute(sql.SQL("SET LOCAL search_path TO {}").format(sql.Identifier(schema)))
                lock = asyncio.Lock()
                class Pool:
                    @asynccontextmanager
                    async def connection(self):
                        async with lock:
                            yield conn
                pool = Pool()
                repo = PostgresRunRepository()
                repo._pool = pool
                await repo.setup()
                await repo.create_run(run_id="isolated-retrieval-case", session_id=None, parent_run_id=None,
                                      parent_context=None, question="Live receipt replay verification")
                first = await repo.begin_execution("isolated-retrieval-case", (RunStatus.CREATED,), resume=False)
                token = current_budget.set(RunBudget(repo, first, asyncio.Semaphore(2)))
                try:
                    try:
                        await gather_retrievals(durable_retrieval(primary, descriptors[0]),
                                                durable_retrieval(injected, descriptors[1]))
                    except RetrievalDeferred:
                        pass
                    else:
                        raise AssertionError("expected controlled dependency pause")
                finally:
                    current_budget.reset(token)
                assert calls == {"real_primary": 1, "injected_timeout": 2, "real_secondary": 0}
                await repo.finish_execution(first.run_id, first.execution_id, RunStatus.PAUSED,
                                            {"reason": "dependency_unavailable"})
                # Reconstruct repository and budget scope, as after a service restart.
                replacement = PostgresRunRepository()
                replacement._pool = pool
                resumed = await replacement.begin_execution(first.run_id, (RunStatus.PAUSED,), resume=True)
                token = current_budget.set(RunBudget(replacement, resumed, asyncio.Semaphore(2)))
                try:
                    batches = await gather_retrievals(durable_retrieval(primary, descriptors[0]),
                                                      durable_retrieval(secondary, descriptors[1]))
                finally:
                    current_budget.reset(token)
                budget = (await replacement.get_run(first.run_id)).budget
                assert calls == {"real_primary": 1, "injected_timeout": 2, "real_secondary": 1}
                assert budget["retrieval_calls"] == 4
                emit(event="verified", calls=calls, batch_sizes=[len(b) for b in batches],
                     charged_retrieval_attempts=budget["retrieval_calls"],
                     unknown_reservations=sum(v["status"] == "unknown" for v in budget["reservations"].values()),
                     seconds=round(time.monotonic()-started, 2))
            emit(event="private_schema_rolled_back")
    finally:
        await get_knowledge_service_client().aclose()
        _, after = await read_case()
        assert signature == after, "historical case changed during diagnostics"
        emit(event="original_run_unchanged", signature=after)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", required=True)
    parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(main())
