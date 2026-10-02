"""FastAPI endpoints for sessions, research runs, and persisted SSE events."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from ..core.checkpointer import CheckpointerFactory
from ..core.run_context import normalize_run_id
from ..core.streaming import _get_app
from ..knowledge.client import get_knowledge_service_client
from ..runs.models import (
    RunCreateRequest,
    RunRecord,
    SessionCreateRequest,
    SessionRecord,
    SessionTimeline,
)
from ..runs.repository import (
    PostgresRunRepository,
    RunConflictError,
    RunNotFoundError,
)
from ..runs.service import RunService
from .demo import register_demo


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

run_repository = PostgresRunRepository()
run_service = RunService(run_repository)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[Server] 正在初始化图并检查 knowledge-service...")
    knowledge_client = get_knowledge_service_client()
    try:
        await run_repository.open()
        await run_repository.setup()
        interrupted = await run_service.recover_stale_runs()
        if interrupted:
            logger.warning(
                "[Server] 已将 %d 个遗留 running 任务标记为 interrupted",
                len(interrupted),
            )
        await asyncio.gather(_get_app(), knowledge_client.ensure_ready())
        logger.info("[Server] knowledge-service 已就绪，当前服务可用 ✅")
        yield
    finally:
        await run_service.shutdown()
        await run_repository.close()
        await knowledge_client.aclose()
        await CheckpointerFactory.close_all()


app = FastAPI(
    title="Multi-Agent Research API",
    description="基于 LangGraph 的多 Agent 研究系统 — 实时进度流接口",
    version="1.1.0",
    docs_url="/api/docs",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def _sse(event: str, data: dict, *, event_id: int | None = None) -> str:
    """Format one W3C Server-Sent Event."""
    id_line = f"id: {event_id}\n" if event_id is not None else ""
    return f"{id_line}event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _run_http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, RunNotFoundError):
        return HTTPException(status_code=404, detail=f"resource '{exc.args[0]}' not found")
    if isinstance(exc, RunConflictError):
        return HTTPException(status_code=409, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=422, detail=str(exc))
    return HTTPException(status_code=500, detail="run operation failed")


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "service": "multi-agent-research",
        "run_store": "ok" if await run_repository.health_check() else "error",
        "checkpointer": await CheckpointerFactory.health_check(),
    }


@app.post(
    "/api/sessions",
    response_model=SessionRecord,
    status_code=status.HTTP_201_CREATED,
    summary="创建研究 Session",
)
async def create_session(request: SessionCreateRequest):
    try:
        return await run_service.create_session(**request.model_dump())
    except (RunConflictError, ValueError) as exc:
        raise _run_http_error(exc) from exc


@app.get(
    "/api/sessions/{session_id}",
    response_model=SessionTimeline,
    summary="读取 Session 及 Run 时间线",
)
async def get_session_timeline(session_id: str):
    try:
        return await run_service.get_session_timeline(session_id)
    except (RunNotFoundError, ValueError) as exc:
        raise _run_http_error(exc) from exc


@app.get(
    "/api/sessions/{session_id}/runs",
    response_model=list[RunRecord],
    summary="按创建时间列出 Session 内的 Run",
)
async def list_session_runs(session_id: str):
    try:
        timeline = await run_service.get_session_timeline(session_id)
        return timeline.runs
    except (RunNotFoundError, ValueError) as exc:
        raise _run_http_error(exc) from exc


@app.post(
    "/api/runs",
    response_model=RunRecord,
    status_code=status.HTTP_201_CREATED,
    summary="创建研究任务（不执行）",
)
async def create_run(request: RunCreateRequest):
    try:
        return await run_service.create_run(**request.model_dump())
    except (RunConflictError, RunNotFoundError, ValueError) as exc:
        raise _run_http_error(exc) from exc


@app.get("/api/runs/{run_id}", response_model=RunRecord, summary="查询任务状态与结果")
async def get_run(run_id: str):
    try:
        return await run_service.get_run(run_id)
    except RunNotFoundError as exc:
        raise _run_http_error(exc) from exc


@app.post(
    "/api/runs/{run_id}/start",
    response_model=RunRecord,
    status_code=status.HTTP_202_ACCEPTED,
    summary="执行已创建的任务",
)
async def start_run(run_id: str):
    try:
        return await run_service.start_run(run_id)
    except (RunConflictError, RunNotFoundError) as exc:
        raise _run_http_error(exc) from exc


@app.post(
    "/api/runs/{run_id}/resume",
    response_model=RunRecord,
    status_code=status.HTTP_202_ACCEPTED,
    summary="从 Checkpoint 恢复失败或中断的任务",
)
async def resume_run(run_id: str):
    try:
        return await run_service.start_run(run_id, resume=True)
    except (RunConflictError, RunNotFoundError) as exc:
        raise _run_http_error(exc) from exc


@app.get("/api/runs/{run_id}/stream", summary="读取持久化的任务事件流")
async def run_event_stream(
    run_id: str,
    after: int = Query(default=0, ge=0, description="仅读取该 sequence 之后的事件"),
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
):
    try:
        await run_service.get_run(run_id)
    except RunNotFoundError as exc:
        raise _run_http_error(exc) from exc

    try:
        resume_after = max(after, int(last_event_id or 0))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Last-Event-ID must be an integer") from exc

    async def generate() -> AsyncGenerator[str, None]:
        async for event in run_service.iter_events(run_id, after=resume_after):
            yield _sse(event.event_type, event.payload, event_id=event.sequence)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get(
    "/api/research/stream",
    summary="研究进度 SSE 流（兼容接口）",
    deprecated=True,
)
async def research_stream(
    question: str = Query(..., description="研究问题", min_length=5, max_length=500),
    run_id: str = Query(default="", description="研究任务 ID，空则自动生成新任务"),
):
    """Deprecated create/start/stream compatibility endpoint."""
    try:
        resolved_run_id = normalize_run_id(run_id)
        await run_service.create_run(question=question, run_id=resolved_run_id)
        await run_service.start_run(resolved_run_id)
    except (RunConflictError, RunNotFoundError, ValueError) as exc:
        raise _run_http_error(exc) from exc

    async def generate() -> AsyncGenerator[str, None]:
        async for event in run_service.iter_events(resolved_run_id):
            yield _sse(event.event_type, event.payload, event_id=event.sequence)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


register_demo(app)
