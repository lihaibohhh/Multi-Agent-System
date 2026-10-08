"""Durable admission accounting. Unknown work keeps its reservation across resumes."""

import asyncio
import time
from contextvars import ContextVar
from copy import deepcopy
from uuid import uuid4

from pydantic import BaseModel, Field

from .config import settings


class RunControlError(RuntimeError):
    """Must propagate through agent fallbacks, never become an empty search/report."""


class BudgetExceeded(RunControlError):
    def __init__(self, message, *, details=None):
        super().__init__(message)
        self.details = details or {}


class CallTimeout(RunControlError):
    pass


class ExecutionPaused(RunControlError):
    pass


class ExecutionTimeLimit(ExecutionPaused):
    pass


class LegacyBudget(RunControlError):
    pass


class BudgetPolicy(BaseModel):
    model_calls: int = Field(gt=0)
    tokens: int = Field(gt=0)
    retrieval_calls: int = Field(gt=0)
    wall_seconds: float = Field(gt=0)
    model_timeout: float = Field(gt=0)
    retrieval_timeout: float = Field(gt=0)

    @classmethod
    def configured(cls):
        cfg = settings.agent
        return cls(model_calls=cfg.run_max_model_calls, tokens=cfg.run_max_tokens,
                   retrieval_calls=cfg.run_max_retrieval_calls, wall_seconds=cfg.run_timeout,
                   model_timeout=cfg.model_call_timeout, retrieval_timeout=cfg.retrieval_call_timeout)


def new_budget(usage=None):
    usage = usage or {}
    return {"version": 2, "policy": BudgetPolicy.configured().model_dump(), "deadline": None,
            "model_calls": usage.get("attempts", 0), "retrieval_calls": 0,
            "known_tokens": usage.get("tokens", 0), "charged_tokens": usage.get("tokens", 0),
            "legacy_unknown_calls": usage.get("unknown", 0), "legacy_history_incomplete": bool(usage),
            "reservations": {}}


def start_budget(value, usage=None):
    budget = deepcopy(value) if value else new_budget(usage)
    if budget.get("version", 1) != 2:
        raise LegacyBudget("旧版时间预算需显式迁移；累计消耗和未知预留不会清零")
    budget["deadline"] = time.time() + BudgetPolicy.model_validate(budget["policy"]).wall_seconds
    return budget


def migrate_budget(value, usage=None):
    """Explicit policy migration only; never infer historical execution duration."""
    budget = deepcopy(value) if value else new_budget(usage)
    if not value:
        budget["legacy_history_incomplete"] = True
    budget.update(version=2, deadline=None)
    return budget


def check_available(budget):
    policy = BudgetPolicy.model_validate(budget["policy"])
    for field in ("model_calls",):
        if budget[field] >= getattr(policy, field):
            raise BudgetExceeded(f"累计 {field} 预算已耗尽；继续不会重置，需明确追加额度")
    if budget["charged_tokens"] >= policy.tokens:
        raise BudgetExceeded("累计 Token 预算已耗尽；继续不会重置，需明确追加额度")


def remaining_seconds(budget):
    remaining = budget["deadline"] - time.time()
    if remaining <= 0:
        raise ExecutionTimeLimit("本次执行时限已到，可继续；累计费用与未知预留保留")
    return remaining


def reserve(value, reservation_id, kind, tokens, label):
    budget = deepcopy(value)
    policy = BudgetPolicy.model_validate(budget["policy"])
    remaining_seconds(budget)
    if kind not in {"model", "retrieval"} or tokens < 0:
        raise ValueError("invalid reservation")
    if reservation_id in budget["reservations"]:
        raise RunControlError("重复的调用预算申请，拒绝再次发起请求")
    counter = "model_calls" if kind == "model" else "retrieval_calls"
    if budget[counter] >= getattr(policy, counter):
        raise BudgetExceeded(f"Run {counter} 调用预算已耗尽；已保存章节保留，恢复不会重置预算")
    if kind == "model" and budget["charged_tokens"] + tokens > policy.tokens:
        remaining = max(0, policy.tokens - budget['charged_tokens'])
        shortfall = budget['charged_tokens'] + tokens - policy.tokens
        raise BudgetExceeded(
            f"Run Token 预算不足以预留本次调用：剩余 {remaining:,}，本次预留 {tokens:,}，缺口 {shortfall:,}；"
            "预留为保守估算，不是实际用量；请求未发送，已保存章节保留，恢复不会重置预算",
            details={'kind': 'token_reservation', 'limit': policy.tokens, 'charged_tokens': budget['charged_tokens'],
                     'remaining_tokens': remaining, 'requested_tokens': tokens, 'shortfall_tokens': shortfall,
                     'label': label[:100], 'estimate_method': 'utf8_bytes_plus_output_and_margin'})
    budget[counter] += 1
    budget["charged_tokens"] += tokens
    budget["reservations"][reservation_id] = {
        "kind": kind, "reserved_tokens": tokens, "status": "unknown", "label": label[:100],
    }
    return budget


def settle(value, reservation_id, actual_tokens):
    budget = deepcopy(value)
    entry = budget["reservations"][reservation_id]
    if entry["status"] == "settled":
        return budget
    if actual_tokens is None:
        return budget  # Includes transport failure, cancellation and process death.
    if actual_tokens < 0:
        raise ValueError("negative token usage")
    budget["charged_tokens"] += actual_tokens - entry["reserved_tokens"]
    budget["known_tokens"] += actual_tokens
    entry.update(status="settled", actual_tokens=actual_tokens)
    return budget


def budget_summary(value):
    """Public progress view without repeating the per-call ledger in every event."""
    return {**{key: item for key, item in value.items() if key not in {"reservations", "increases"}},
            "unknown_model_calls": sum(entry["kind"] == "model" and entry["status"] == "unknown"
                                       for entry in value.get("reservations", {}).values())}


current_budget: ContextVar = ContextVar("run_budget", default=None)


class RunBudget:
    def __init__(self, repository, record, search_slots):
        self.repository, self.run_id, self.execution_id = repository, record.run_id, record.execution_id
        self.policy = BudgetPolicy.model_validate(record.budget["policy"])
        self.deadline = record.budget["deadline"]
        self.search_slots = search_slots
        self.retrieval_locks = {}
        context = getattr(record, "parent_context", None)
        operation = context.section_operation if context else None
        self.retrieval_parent = context.source_run_id if operation and operation["mode"] == "continue" else None

    async def reserve(self, kind, tokens, label):
        reservation_id = uuid4().hex
        try:
            await self.repository.reserve_budget(self.run_id, self.execution_id, reservation_id, kind, tokens, label)
        except RunControlError:
            raise
        except Exception as exc:
            raise RunControlError("调用预算无法持久化，未发送外部请求") from exc
        return reservation_id

    async def settle(self, reservation_id, tokens):
        try:
            await self.repository.settle_budget(self.run_id, self.execution_id, reservation_id, tokens)
        except Exception as exc:
            raise RunControlError("调用用量无法持久化；保留预留额度并停止执行") from exc


def _usage(response):
    raw = response.get("raw") if isinstance(response, dict) else response
    meta = getattr(raw, "response_metadata", {}) or {}
    usage = getattr(raw, "usage_metadata", None) or meta.get("token_usage") or {}
    value = usage.get("total_tokens")
    return int(value) if isinstance(value, (int, float)) and value >= 0 else None


async def invoke_model(runnable, messages, *, model_ref="", label="model"):
    scope = current_budget.get()
    timeout = scope.policy.model_timeout if scope else settings.agent.model_call_timeout
    reservation_id = None
    if scope:
        provider = model_ref.split("/", 1)[0]
        provider = {"ds": "deepseek", "qwen-local": "local", "openai-compatible": "local"}.get(provider, provider)
        cfg = getattr(settings, provider, None)
        output_limit = getattr(cfg, "max_tokens", 4096)
        # Deliberately conservative admission estimate, not a provider tokenizer guarantee.
        tokens = sum(len(str(m.content).encode("utf-8")) + 128 for m in messages) + output_limit + 1024
        reservation_id = await scope.reserve("model", tokens, label)
    try:
        async with asyncio.timeout(timeout):
            response = await runnable.ainvoke(messages)
    except TimeoutError as exc:
        raise CallTimeout("模型调用超时；请求可能已计费，预留预算不退回") from exc
    except Exception as exc:
        # SDK may raise with a completed response (e.g. length finish reason).
        usage = getattr(getattr(exc, "completion", None), "usage", None)
        if scope and usage is not None:
            await scope.settle(reservation_id, usage.total_tokens)
        raise
    if scope:
        await scope.settle(reservation_id, _usage(response))
    return response


async def invoke_retrieval(operation, *, label):
    from .retrieval import retrieval_admitted
    if retrieval_admitted.get():
        return await operation()
    scope = current_budget.get()
    timeout = scope.policy.retrieval_timeout if scope else settings.agent.retrieval_call_timeout
    async def call():
        reservation_id = await scope.reserve("retrieval", 0, label) if scope else None
        try:
            async with asyncio.timeout(timeout):
                result = await operation()
        except TimeoutError as exc:
            raise CallTimeout("检索调用超时；本次请求额度保留，停止当前执行") from exc
        if scope:
            await scope.settle(reservation_id, 0)
        return result
    if scope:
        async with scope.search_slots:
            return await call()
    return await call()


async def gather_cancel_on_error(*operations):
    tasks = [asyncio.create_task(operation) for operation in operations]
    try:
        return await asyncio.gather(*tasks)
    except ExecutionPaused:
        # Already-admitted requests get their bounded chance to finish/settle.
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
