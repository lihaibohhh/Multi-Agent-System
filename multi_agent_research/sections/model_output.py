"""Bounded structured-output correction and private, per-attempt diagnostics."""

import json
import logging
import re
from dataclasses import dataclass
from collections.abc import Callable
from contextvars import ContextVar
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.config import get_config
from pydantic import BaseModel, SecretStr
from openai import LengthFinishReasonError

from ..core.config import settings
from .validation import BusinessValidationError
from ..core.budget import RunControlError, invoke_model


logger = logging.getLogger(__name__)
# RunService supplies a durable DB sink; standalone callers retain ordinary cost output.
attempt_sink: ContextVar = ContextVar("model_attempt_sink", default=None)
MAX_CORRECTIONS = 2
RAW_LIMIT = 16_000


@dataclass
class PartialResult:
    value: dict
    errors: list[dict]


class ModelOutputError(ValueError):
    """Safe public message; the raw response exists only in private diagnostics."""

    def __init__(self, message, *, record=None, cost=None):
        super().__init__(message)
        self.record = record
        self.cost = cost


def _redact(text: str) -> str:
    def secrets(value):
        if isinstance(value, SecretStr):
            yield value.get_secret_value()
        elif isinstance(value, BaseModel):
            for field in type(value).model_fields:
                yield from secrets(getattr(value, field))
    for secret in secrets(settings):
        if len(secret) >= 6:
            text = text.replace(secret, "[REDACTED]")
    return re.sub(r"\b(?:sk-|tvly-|ghp_|github_pat_)[A-Za-z0-9_-]{12,}", "[REDACTED]", text)


def _errors(error) -> list[dict]:
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if hasattr(error, "errors"):
            return [{"field": ".".join(map(str, e["loc"])), "type": e["type"],
                     "message": _redact(e["msg"])[:400]}
                    for e in error.errors(include_input=False, include_context=False, include_url=False)[:10]]
        error = error.__cause__
    return [{"field": "$", "type": "invalid_json", "message": "Response is not valid schema-conforming JSON"}]


async def invoke_checked(model, system: str, prompt: str, schema=None, *,
                         validator: Callable | None = None, context: dict | None = None):
    """Validate schema and business rules with one correction budget.

    Validators are synchronous, pure, and must not mutate persistent state.
    Only BusinessValidationError is retryable, not arbitrary programming errors.
    """
    model_ref = settings.agent.section_model
    try:
        node = get_config().get("metadata", {}).get("langgraph_node", "standalone")
    except RuntimeError:
        node = "standalone"
    if schema is not None:
        system += "\n只输出满足以下 JSON Schema 的对象；不得省略类型、枚举、长度限制：\n"
        system += json.dumps(schema.model_json_schema(), ensure_ascii=False)
        runnable = model.with_structured_output(schema, method="json_mode", include_raw=True)
    else:
        runnable = model
    cost = {"tokens": 0, "unknown": 0, "attempts": 0}
    call_id = uuid4().hex
    correction = ""
    previous_output = ""
    limit = 1 + MAX_CORRECTIONS if schema or validator else 1
    if (context or {}).get("single_attempt"):
        limit = 1  # Claim coordinator checkpoints between attempts; no nested retries.
    for attempt in range(1, limit + 1):
        diagnostic_id = uuid4().hex
        record = {"diagnostic_id": diagnostic_id, "node": node, "model": model_ref,
                  "call_id": call_id,
                  "schema": schema.__name__ if schema else None, "attempt": attempt,
                  "tokens": 0, "unknown": 1, "errors": [], "raw": "", "raw_truncated": False,
                  "context": context or {}, "schema_status": "not_checked",
                  "business_status": "not_checked" if validator else "not_applicable",
                  "accepted": False, "retryable": False}
        repair_hints = []
        cost["attempts"] += 1
        try:
            messages = [SystemMessage(content=system), HumanMessage(content=prompt)]
            if correction:
                messages.append(AIMessage(content=previous_output))
                messages.append(HumanMessage(content=correction))
            response = await invoke_model(runnable, messages, model_ref=model_ref,
                                          label=f"{node}:{(context or {}).get('section_id', '')}")
        except RunControlError:
            raise
        except LengthFinishReasonError as exc:
            # The SDK's parse API can raise before LangChain returns include_raw.
            completion = exc.completion
            usage = completion.usage
            total = usage.total_tokens if usage else None
            text = (completion.choices[0].message.content or "") if completion.choices else ""
            record.update(tokens=int(total or 0), unknown=int(total is None),
                          raw=_redact(text)[:RAW_LIMIT], raw_truncated=len(text) > RAW_LIMIT,
                          finish_reason="length",
                          errors=[{"field": "$", "type": "truncated", "message": "模型输出达到长度上限"}])
            await _save_attempt(record)
            raise ModelOutputError(f"模型输出达到长度上限；诊断 ID {diagnostic_id}") from None
        except Exception as exc:
            # Transport/auth failures are not JSON failures; do not retry them here.
            record["errors"] = [{"field": "$", "type": type(exc).__name__,
                                 "message": "Model request failed before a usable response"}]
            await _save_attempt(record)
            raise ModelOutputError(f"模型请求失败：{type(exc).__name__}；诊断 ID {diagnostic_id}") from None

        raw = response["raw"] if schema else response
        metadata = getattr(raw, "response_metadata", {}) or {}
        usage = getattr(raw, "usage_metadata", None) or metadata.get("token_usage") or {}
        total = usage.get("total_tokens")
        record["tokens"], record["unknown"] = int(total or 0), int(total is None)
        cost["tokens"] += record["tokens"]
        cost["unknown"] += record["unknown"]
        content = getattr(raw, "content", "")
        text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        record["raw"] = _redact(text)[:RAW_LIMIT]
        record["raw_truncated"] = len(text) > RAW_LIMIT
        truncated = (metadata.get("finish_reason") in {"length", "max_tokens"}
                     or metadata.get("stop_reason") == "max_tokens")
        record["finish_reason"] = metadata.get("finish_reason") or metadata.get("stop_reason")
        parsed = None
        if truncated:
            record["errors"] = [{"field": "$", "type": "truncated", "message": "模型输出达到长度上限"}]
        elif schema:
            error = response.get("parsing_error")
            if not error and response.get("parsed") is not None:
                try:
                    parsed = schema.model_validate(response["parsed"])
                except ValueError as exc:
                    error = exc
            if parsed is None:
                record["errors"] = _errors(error)
        elif not isinstance(content, str) or not content.strip():
            record["errors"] = [{"field": "$", "type": "empty_output", "message": "模型正文为空或非文本"}]
        record["schema_status"] = "failed" if record["errors"] else "passed"
        record["retryable"] = bool(record["errors"]) and not truncated
        accepted = parsed if schema else content
        partial = False
        if not record["errors"] and validator is not None:
            try:
                accepted = validator(accepted)
                record["business_status"] = "passed"
                if isinstance(accepted, PartialResult):
                    partial = True
                    record["errors"] = accepted.errors
                    record["business_status"] = "partial" if accepted.errors else "passed"
                    record["retryable"] = bool(accepted.errors)
                    record["accepted_slots"] = list(accepted.value.get("accepted", {}))
                    record["pending_slots"] = [p["slot"] for p in accepted.value.get("pending", [])]
                    accepted = accepted.value
            except BusinessValidationError as exc:
                record["business_status"] = "failed"
                record["errors"] = exc.details()
                record["retryable"] = exc.retryable
                repair_hints = exc.repair_hints
                record["repair_hints"] = json.loads(_redact(json.dumps(repair_hints, ensure_ascii=False)))
            except Exception as exc:
                record["business_status"] = "internal_error"
                record["errors"] = [{"field": "$", "type": "validator_internal_error",
                                     "message": f"业务校验器异常：{type(exc).__name__}"}]
                record["retryable"] = False
        record["accepted"] = not record["errors"]
        await _save_attempt(record)
        if partial:
            return accepted, cost  # Coordinator checkpoints before local repair.
        if not record["errors"]:
            return accepted, cost
        details = "; ".join(f"{e['field']}: {_redact(e['message'])[:400]}" for e in record["errors"][:5])
        if len(record["errors"]) > 5:
            details += f"; 另有 {len(record['errors']) - 5} 项错误，见诊断记录"
        logger.warning("Model output rejected node=%s schema=%s attempt=%s id=%s errors=%s",
                       node, record["schema"], attempt, diagnostic_id, details)
        # More calls cannot fix an insufficient output budget. Preserve the checkpoint.
        if not record["retryable"] or attempt == limit:
            location = f"章节 {context['section_id']}：" if context and context.get("section_id") else ""
            raise ModelOutputError(
                f"{location}{record['schema'] or '正文'} 输出校验失败（{attempt} 次）：{details}；诊断 ID {diagnostic_id}",
                record=record, cost=cost,
            ) from None
        # Keep only the last answer; the original task/source view stays intact.
        previous_output = text[:RAW_LIMIT]
        if len(text) > RAW_LIMIT:
            previous_output += "\n[上次输出过长，此处仅提供前缀]"
        correction = (
            "上次输出未通过校验，具体错误如下：" + json.dumps(record["errors"][:20], ensure_ascii=False)
            + f"\n共 {len(record['errors'])} 项错误，上面最多展示20项；请一并检查同类问题。"
            "\n请重新生成完整结果，遵守系统给出的 Schema 和业务约束；超量时选择最重要的允许条数。"
            "引文必须复制对应资料的连续原文，不改数字、否定词或标点；不能修复的证据不得伪造。"
            "保持真实审校意见，不得为通过校验改成 pass，也不得静默删除反证或尚未解决的问题。"
            + "\n出错字段的原文定位提示（仅供重新选择，不能自动视为支持证据）："
            + json.dumps(repair_hints, ensure_ascii=False)
        )
    raise AssertionError("unreachable")


async def _save_attempt(record: dict) -> None:
    sink = attempt_sink.get()
    if sink is not None:
        await sink(record)
