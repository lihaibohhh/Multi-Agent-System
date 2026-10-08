"""FastAPI endpoints for sessions, research runs, and persisted SSE events."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from ..core.checkpointer import CheckpointerFactory
from ..runs.runtime import InstanceUnavailableError
from ..core.run_context import normalize_run_id
from ..core.streaming import _get_app
from ..knowledge.client import get_knowledge_service_client
from ..runs.models import (
    RunCreateRequest,
    RunRecord,
    RunSnapshot,
    SessionCreateRequest,
    SessionRecord,
    SessionTimeline,
    SectionRevisionRequest,
    SectionOperationRequest,
    BudgetMigrationRequest,
    BudgetIncreaseRequest,
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
    from ..core.streaming import reset_app
    async with AsyncExitStack() as cleanup:
        # All callbacks run even when another cleanup fails. Ownership is last.
        cleanup.push_async_callback(run_repository.close)
        cleanup.push_async_callback(reset_app)
        cleanup.push_async_callback(CheckpointerFactory.close_all)
        cleanup.push_async_callback(knowledge_client.aclose)
        cleanup.push_async_callback(run_service.shutdown)
        await run_repository.open()
        await run_repository.acquire_instance()
        await run_repository.setup()
        await _get_app()  # Durable checkpoint initialization must succeed first.
        await knowledge_client.ensure_ready()
        await run_service.start_runtime()
        logger.info("[Server] knowledge-service 已就绪，当前服务可用 ✅")
        yield


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
    async def probe(operation, fallback):
        try:
            async with asyncio.timeout(8):
                return await operation()
        except Exception:
            return fallback
    store, checkpoint, runtime = await asyncio.gather(
        probe(run_repository.health_check, False),
        probe(CheckpointerFactory.health_check, {"status": "error", "persistent": False}),
        probe(run_service.runtime_health, {"ownership": "unavailable", "monitor": "stopped"}),
    )
    ok = (store and checkpoint.get("status") == "ok" and checkpoint.get("persistent")
          and runtime.get("ownership") == "ok" and runtime.get("monitor") == "ok"
          and runtime.get("reconciliation") == "ok")
    return JSONResponse(status_code=200 if ok else 503, content={
        "status": "ok" if ok else "unavailable",
        "service": "multi-agent-research",
        "run_store": "ok" if store else "error",
        "checkpointer": checkpoint, "runtime": runtime,
    })


@app.exception_handler(InstanceUnavailableError)
async def instance_unavailable(request, exc):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


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


@app.post("/api/runs/{run_id}/budget/increase", response_model=RunRecord,
          summary="明确追加共享 Token 上限；累计消耗不变、不自动恢复")
async def increase_run_budget(run_id: str, request: BudgetIncreaseRequest):
    try:
        return await run_service.increase_run_budget(run_id, **request.model_dump())
    except (RunConflictError, RunNotFoundError, ValueError) as exc:
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
    "/api/runs/{run_id}/sections/{section_id}/revisions",
    response_model=RunRecord,
    status_code=status.HTTP_201_CREATED,
    summary="创建章节修订 Run（保留原报告，需另行 start）",
)
async def create_section_revision(run_id: str, section_id: str, request: SectionRevisionRequest):
    try:
        return await run_service.create_section_revision(
            run_id, section_id, instruction=request.instruction, new_run_id=request.run_id,
        )
    except (RunConflictError, RunNotFoundError, ValueError) as exc:
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


@app.post("/api/runs/{run_id}/sections/{section_id}/operations", response_model=RunRecord,
          status_code=status.HTTP_201_CREATED, summary="创建选章继续/仅补证据/刷新来源操作（共预算，另行启动）")
async def create_section_operation(run_id: str, section_id: str, request: SectionOperationRequest):
    try:
        return await run_service.create_section_operation(run_id, section_id, mode=request.mode,
            instruction=request.instruction, new_run_id=request.run_id)
    except (RunConflictError, RunNotFoundError, ValueError) as exc:
        raise _run_http_error(exc) from exc


@app.get("/api/runs/{run_id}/snapshot", response_model=RunSnapshot,
         summary="读取同一数据库快照内的 Run 状态与 SSE 游标")
async def get_run_snapshot(run_id: str):
    try:
        return await run_service.get_snapshot(run_id)
    except RunNotFoundError as exc:
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


@app.post("/api/runs/{run_id}/pause", response_model=RunRecord, status_code=202,
          summary="请求安全暂停；当前调用在超时范围内完成，不再调度新请求")
async def pause_run(run_id: str):
    try:
        return await run_service.pause_run(run_id)
    except (RunConflictError, RunNotFoundError) as exc:
        raise _run_http_error(exc) from exc


@app.post("/api/runs/{run_id}/budget/migrate", response_model=RunRecord,
          summary="显式迁移旧版时间策略，保留历史消耗；不自动执行")
async def migrate_run_budget(run_id: str, request: BudgetMigrationRequest):
    try:
        return await run_service.migrate_run_budget(run_id, **request.model_dump())
    except (RunConflictError, RunNotFoundError, ValueError) as exc:
        raise _run_http_error(exc) from exc


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
