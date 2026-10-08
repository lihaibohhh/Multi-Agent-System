"""Explicit live diagnostics. Original Run and Checkpoint are read-only.

No service setup, production resume, KB file access or query cache writes.
Only chapter mode calls the configured LLM, with isolated in-memory quotas.
"""
import argparse
import asyncio
import hashlib
import json
import logging
import time
from copy import deepcopy
from types import SimpleNamespace

import httpx
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from multi_agent_research.core.config import settings
from multi_agent_research.core.budget import (
    RunBudget, current_budget, new_budget, start_budget, reserve, settle, budget_summary, CallTimeout,
)
from multi_agent_research.core.run_context import checkpoint_config
from multi_agent_research.knowledge.client import KnowledgeServiceClient

RUN = "run_7c73d441e9ac4d1b9207db1eb87261b8"


def emit(event, **data):
    print(json.dumps({"event": event, **data}, ensure_ascii=False, default=str), flush=True)


async def read_case():
    conn = await AsyncConnection.connect(settings.database.url, autocommit=True, row_factory=dict_row,
        connect_timeout=5, options="-c default_transaction_read_only=on -c statement_timeout=15000")
    async with conn:
        row = await (await conn.execute("SELECT * FROM research_runs WHERE run_id=%s", (RUN,))).fetchone()
        cp = await AsyncPostgresSaver(conn).aget_tuple(checkpoint_config(RUN))
        if not row or not cp:
            raise RuntimeError("original Run/checkpoint unavailable")
        signature = hashlib.sha256(json.dumps({"row": row, "checkpoint_id": cp.checkpoint["id"]},
                                             sort_keys=True, default=str).encode()).hexdigest()
        return cp.checkpoint["channel_values"], signature


def queries_for(state):
    section = state["sections"][state["active_section"]]
    parent = (state.get("parent_context") or {}).get("source_question", "")
    question = section["question"]
    primary = f"{question} 承接研究背景：{parent}"[:500] if parent else question
    return [primary, *section.get("gaps", [])[:2]]


class Ledger:
    def __init__(self):
        self.retrievals = {}
        self.budget = start_budget(new_budget())
        self.budget["policy"].update(model_calls=12, retrieval_calls=24, tokens=500000,
                                     wall_seconds=1800, model_timeout=120, retrieval_timeout=60)
        self.budget["deadline"] = time.time() + 1800

    async def reserve_budget(self, run_id, execution_id, key, kind, tokens, label):
        self.budget = reserve(self.budget, key, kind, tokens, label)

    async def settle_budget(self, run_id, execution_id, key, tokens):
        self.budget = settle(self.budget, key, tokens)

    async def begin_retrieval(self, run_id, execution_id, operation_id, descriptor, reservation_id, reuse_parent=None):
        from multi_agent_research.core.retrieval import admit
        data, reused = admit(self.retrievals.get(operation_id), descriptor, execution_id, reservation_id)
        if not reused:
            await self.reserve_budget(run_id, execution_id, reservation_id, "retrieval", 0, descriptor["provider"])
            self.retrievals[operation_id] = data
        return deepcopy(data), reused

    async def finish_retrieval(self, run_id, execution_id, operation_id, reservation_id, result, error):
        from multi_agent_research.core.retrieval import complete
        data = complete(self.retrievals[operation_id], execution_id, reservation_id, result, error)
        if error is None:
            await self.settle_budget(run_id, execution_id, reservation_id, 0)
        self.retrievals[operation_id] = data


async def health():
    async with httpx.AsyncClient(timeout=8) as client:
        for port, path in ((8000, "/api/health"), (8001, "/api/v1/health/live"), (8001, "/api/v1/health/ready")):
            started = time.monotonic()
            try:
                r = await client.get(f"http://127.0.0.1:{port}{path}")
                emit("health", port=port, path=path, elapsed=round(time.monotonic()-started, 3),
                     status=r.status_code, body=r.json())
            except Exception as exc:
                emit("health_error", port=port, path=path, error=type(exc).__name__)


async def probes(state, timeout):
    cfg = settings.knowledge_service
    client = KnowledgeServiceClient(base_url=cfg.base_url, api_key=cfg.api_key.get_secret_value(), timeout=timeout)
    async def query(q, phase, index):
        started = time.monotonic()
        emit("query_start", phase=phase, index=index, query=q)
        try:
            async with asyncio.timeout(timeout):
                r = await client.search(q, top_k=cfg.top_k, retrieval_mode=cfg.retrieval_mode)
            emit("query_done", phase=phase, index=index, elapsed=round(time.monotonic()-started, 3),
                 chunks=len(r.chunks), cache_hit=r.cache_hit, timings=r.timings,
                 candidates=r.candidates_count, reranked=r.reranked_count,
                 chunk_ids=[c.chunk_id for c in r.chunks])
        except Exception as exc:
            emit("query_error", phase=phase, index=index, elapsed=round(time.monotonic()-started, 3),
                 error=type(exc).__name__, cause=type(exc.__cause__).__name__)
    try:
        for i, q in enumerate(queries_for(state)):
            await query(q, "serial", i)
        await health()
        await asyncio.gather(*(query(q, "concurrent", i) for i, q in enumerate(queries_for(state))))
    finally:
        await client.aclose()


async def chapter(state):
    from multi_agent_research.sections import workflow
    from multi_agent_research.sections.model_output import attempt_sink
    from multi_agent_research.knowledge.client import get_knowledge_service_client
    ledger = Ledger()
    record = SimpleNamespace(run_id="isolated-live-case", execution_id="diagnostic", budget=ledger.budget)
    token = current_budget.set(RunBudget(ledger, record, asyncio.Semaphore(4)))
    async def sink(data):
        emit("model_attempt", node=data["node"], schema=data["schema"], attempt=data["attempt"],
             accepted=data["accepted"], tokens=data["tokens"],
             errors=[{k: e.get(k) for k in ("field", "type")} for e in data["errors"]])
    audit_token = attempt_sink.set(sink)
    nodes = {"research": workflow.research_section, "analyze": workflow.analyze_section,
             "write": workflow.write_section, "review": workflow.review_section, "claims": workflow.extract_claims}
    try:
        async with asyncio.timeout(900):
            for _ in range(12):
                step = state["section_step"]
                if step == "advance":
                    break
                emit("node_start", step=step)
                started = time.monotonic()
                state.update(await nodes[step](state))
                s = state["sections"][state["active_section"]]
                emit("node_done", step=step, elapsed=round(time.monotonic()-started, 3),
                     next=state["section_step"], status=s["status"], results=len(s["results"]),
                     draft_chars=len(s["draft"]), claims=len(s["claims"]), limitations=s["limitations"])
    except Exception as exc:
        emit("chapter_error", error=type(exc).__name__, message=str(exc)[:1500], step=state["section_step"])
    finally:
        emit("isolated_budget", budget=budget_summary(ledger.budget))
        attempt_sink.reset(audit_token)
        current_budget.reset(token)
        await get_knowledge_service_client().aclose()


async def partial_failure(state):
    """Real successful query plus a clearly injected sibling failure, repeated twice."""
    from multi_agent_research.agents import search_agent
    from multi_agent_research.sections import workflow
    from multi_agent_research.knowledge.client import get_knowledge_service_client
    ready = asyncio.Event()
    counts = {"real_queries": 0, "successful_chunks": 0}
    original_knowledge, original_web = search_agent._knowledge_search, search_agent._web_search
    target = queries_for(state)[0]
    async def knowledge(query, iteration):
        if query == target:
            counts["real_queries"] += 1
            result = await original_knowledge(query, iteration)
            counts["successful_chunks"] += len(result)
            emit("real_partial_success", chunks=len(result))
            ready.set()
            return result
        await ready.wait()
        raise CallTimeout("INJECTED: sibling timeout after a real query completed")
    async def no_web(*args):
        return []
    search_agent._knowledge_search, search_agent._web_search = knowledge, no_web
    try:
        for attempt in (1, 2):
            ready.clear()
            before = deepcopy(state)
            try:
                async with asyncio.timeout(90):
                    state.update(await workflow.research_section(state))
            except Exception as exc:
                emit("partial_failure", attempt=attempt, error=type(exc).__name__, state_unchanged=state == before,
                     **counts)
    finally:
        search_agent._knowledge_search, search_agent._web_search = original_knowledge, original_web
        await get_knowledge_service_client().aclose()


async def main(args):
    state, before = await read_case()
    state = deepcopy(state)
    emit("case", section=state["active_section"], step=state["section_step"], queries=queries_for(state),
         original_signature=before, mode=args.mode)
    await health()
    if args.mode == "probes":
        await probes(state, args.timeout)
    elif args.mode == "chapter":
        await chapter(state)
    else:
        await partial_failure(state)
    _, after = await read_case()
    emit("original_unchanged", value=before == after)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("probes", "chapter", "partial-failure"))
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main(args))
