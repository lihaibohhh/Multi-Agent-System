"""Run-local retrieval receipts: replay results, never reset attempt/cost history."""

import asyncio
import hashlib
import json
from contextvars import ContextVar
from copy import deepcopy
from datetime import datetime, timezone
from uuid import uuid4

import httpx

from .budget import (CallTimeout, ExecutionPaused, RunControlError, current_budget,
                     gather_cancel_on_error)


class RetrievalDeferred(ExecutionPaused):
    """Temporary dependency failure; the user may resume with remaining attempts."""


class RetrievalFailed(RunControlError):
    """Permanent response error or exhausted per-operation lifetime attempts."""


# Adapters still serve CLI/tool callers. Within a durable attempt their inner
# invoke_retrieval must not reserve twice or start another retry loop.
retrieval_admitted = ContextVar("retrieval_admitted", default=False)


def operation_key(descriptor):
    return hashlib.sha256(json.dumps(descriptor, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def admit(value, descriptor, execution_id, reservation_id):
    """Pure transition; repository commits this AND budget reservation atomically."""
    data = deepcopy(value) if value else {
        "descriptor": descriptor, "attempts": [], "max_attempts": 4,
        "attempts_per_execution": 2, "status": "pending",
    }
    if data["descriptor"] != descriptor:
        raise RunControlError("检索标识与已保存参数不一致，停止执行")
    if data["status"] == "succeeded":
        return data, True
    label = descriptor["provider"]
    attempts = data["attempts"]
    if len(attempts) >= data["max_attempts"]:
        raise RetrievalFailed(f"{label} 查询累计尝试已达 {data['max_attempts']} 次；停止重试，需排查依赖服务")
    current = [a for a in attempts if a["execution_id"] == execution_id]
    if current and current[-1]["status"] == "inflight":
        raise RunControlError("同一查询仍在执行，拒绝重复请求")
    if current and current[-1]["status"] == "permanent_failed":
        raise RetrievalFailed(f"{label} 返回不可自动重试的错误；修正配置或响应后再继续")
    if len(current) >= data["attempts_per_execution"]:
        raise RetrievalDeferred(f"{label} 暂时不可用，已保存成功查询；稍后可继续，累计尝试不会清零")
    for attempt in attempts:
        if attempt["status"] == "inflight":
            attempt.update(status="unknown", error_type="execution_interrupted")
    attempts.append({"execution_id": execution_id, "reservation_id": reservation_id,
                     "status": "inflight", "started_at": datetime.now(timezone.utc).isoformat()})
    data["status"] = "inflight"
    return data, False


def complete(value, execution_id, reservation_id, result, error):
    data = deepcopy(value)
    last = data["attempts"][-1]
    if (last["execution_id"] != execution_id or last["reservation_id"] != reservation_id
            or last["status"] != "inflight"):
        raise RunControlError("检索尝试已过期，拒绝覆盖成果")
    status = error["status"] if error else "succeeded"
    last.update(status=status, finished_at=datetime.now(timezone.utc).isoformat())
    if error:
        last.update(error)
    else:
        data["results"] = deepcopy(result)
    data["status"] = status
    return data


def progress(data, operation_id, reused=False):
    descriptor = data["descriptor"]
    return {"operation_id": operation_id, "provider": descriptor["provider"],
            "section_id": descriptor.get("section_id"), "round": descriptor.get("round"),
            "status": data["status"], "attempts": len(data["attempts"]),
            "max_attempts": data["max_attempts"], "reused": reused,
            "result_count": len(data.get("results", [])),
            "error_type": (data["attempts"][-1].get("error_type") if data["attempts"] else None),
            "message": ("复用已保存检索结果" if reused else "检索操作：" + data["status"])}


async def _store(method, *args):
    try:
        return await method(*args)
    except RunControlError:
        raise
    except Exception as exc:
        raise RunControlError("检索账本无法持久化或执行批次已过期；停止外部调用，保留已记额度") from exc


async def durable_retrieval(operation, descriptor):
    scope = current_budget.get()
    if scope is None:
        return await operation()
    key = operation_key(descriptor)
    # Identical queries in one batch share a local lock; DB fencing remains
    # authoritative across process restart. Dictionary lives only one execution.
    lock = scope.retrieval_locks.setdefault(key, asyncio.Lock())
    async with lock:
        while True:
            async with scope.search_slots:
                reservation_id = uuid4().hex
                data, reused = await _store(scope.repository.begin_retrieval, scope.run_id,
                    scope.execution_id, key, descriptor, reservation_id, scope.retrieval_parent)
                if reused:
                    return deepcopy(data["results"])
                token = retrieval_admitted.set(True)
                try:
                    async with asyncio.timeout(scope.policy.retrieval_timeout):
                        result = await operation()
                    if not isinstance(result, list) or any(not isinstance(r, dict) for r in result):
                        raise ValueError("invalid normalized retrieval results")
                    timestamp = datetime.now(timezone.utc).isoformat()
                    for item in result:
                        item.setdefault("metadata", {}).setdefault("retrieved_at", timestamp)
                except (TimeoutError, ConnectionError, httpx.TransportError, CallTimeout) as exc:
                    error = {"status": "retryable_failed", "error_type": type(exc).__name__}
                except RunControlError:
                    raise  # Leave admitted work unknown; never hide budget/fence/pause errors.
                except Exception as exc:
                    error = {"status": "permanent_failed", "error_type": type(exc).__name__}
                else:
                    error = None
                finally:
                    retrieval_admitted.reset(token)
                await _store(scope.repository.finish_retrieval, scope.run_id, scope.execution_id,
                             key, reservation_id, result if error is None else None, error)
                if error is None:
                    return result
                if error["status"] == "permanent_failed":
                    raise RetrievalFailed(f"{descriptor['provider']} 返回不可自动重试的错误（{error['error_type']}）；已保存其他成功查询")
                # Decide without another request; admission also enforces these
                # limits on restart, so clicks/replayed nodes cannot reset them.
                if len(data["attempts"]) >= data["max_attempts"]:
                    raise RetrievalFailed(f"{descriptor['provider']} 查询累计尝试已达 {data['max_attempts']} 次；停止重试，需排查依赖服务")
                used = sum(a["execution_id"] == scope.execution_id for a in data["attempts"])
                if used >= data["attempts_per_execution"]:
                    raise RetrievalDeferred(f"{descriptor['provider']} 暂时不可用，已保存成功查询；稍后可继续，累计尝试不会清零")
            # Release the concurrency permit while backing off; next admission
            # rechecks persisted pause, time and cumulative budget.
            await asyncio.sleep(1)


async def gather_retrievals(*operations):
    async def capture(operation):
        try:
            return await operation
        except (RetrievalDeferred, RetrievalFailed) as exc:
            return exc
    results = await gather_cancel_on_error(*(capture(op) for op in operations))
    # Ordinary dependency failures must not discard slower successful siblings.
    for kind in (RetrievalFailed, RetrievalDeferred):
        for result in results:
            if isinstance(result, kind):
                raise result
    return results
