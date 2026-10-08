"""Tool adapters shared result envelopes, trimming, and retry helpers."""

from __future__ import annotations

import asyncio
import functools
import inspect
import random
import time
from collections.abc import Callable, Iterable
from typing import Any, TypeVar
from ..core.budget import RunControlError, current_budget


ToolCallable = TypeVar("ToolCallable", bound=Callable[..., Any])


def _ok(
    *,
    tool_name: str,
    query: str,
    data: Any,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a stable success envelope for LangChain tools."""
    return {
        "ok": True,
        "tool": tool_name,
        "query": query,
        "data": data,
        "error": None,
        "meta": meta or {},
    }


def _err(
    *,
    tool_name: str,
    query: str,
    message: str,
    code: str = "TOOL_ERROR",
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a stable error envelope for LangChain tools."""
    return {
        "ok": False,
        "tool": tool_name,
        "query": query,
        "data": None,
        "error": {"code": code, "message": message},
        "meta": meta or {},
    }


def _trim_text(text: str, max_chars: int) -> str:
    """Trim text without returning a string longer than ``max_chars``."""
    value = (text or "").strip()
    if len(value) <= max_chars:
        return value
    if max_chars <= 3:
        return value[:max_chars]
    return value[: max_chars - 3] + "..."


def _shrink_search_results(
    raw: Any,
    *,
    max_items: int = 5,
    max_chars_per_item: int = 800,
) -> dict[str, Any]:
    """Normalize Tavily response variants into a bounded result dictionary."""
    if isinstance(raw, dict):
        raw_items = raw.get("results", [])
        answer = _trim_text(str(raw.get("answer", "")), 1200)
    elif isinstance(raw, list):
        raw_items = raw
        answer = ""
    elif isinstance(raw, str):
        return {"results": [], "answer": _trim_text(raw, 1200)}
    else:
        return {"results": [], "answer": ""}

    results: list[dict[str, Any]] = []
    for item in raw_items[:max_items]:
        if not isinstance(item, dict):
            continue
        results.append(
            {
                "title": _trim_text(str(item.get("title", "")), 200),
                "url": str(item.get("url", "")),
                "content": _trim_text(
                    str(item.get("content", "")),
                    max_chars_per_item,
                ),
                "score": item.get("score"),
                "published_date": (
                    item.get("published_date") or item.get("published_time")
                ),
            }
        )
    return {"results": results, "answer": answer}


def with_retry(
    *,
    tool_name: str,
    max_retries: int = 2,
    timeout: float = 15.0,
    base_delay: float = 0.5,
    max_delay: float = 6.0,
    retry_on: Iterable[type[BaseException]] = (
        TimeoutError,
        ConnectionError,
        OSError,
    ),
) -> Callable[[ToolCallable], ToolCallable]:
    """Add timeout, exponential backoff, and a stable failure envelope."""
    retryable = tuple(retry_on) + (asyncio.TimeoutError,)

    def decorator(function: ToolCallable) -> ToolCallable:
        is_async = inspect.iscoroutinefunction(function)

        async def invoke(*args: Any, **kwargs: Any) -> dict[str, Any]:
            query = str(kwargs.get("query") or (args[0] if args else "")).strip()
            last_exception: BaseException | None = None
            attempt = 0

            for attempt in range(max_retries + 1):
                started_at = time.perf_counter()
                try:
                    if is_async and current_budget.get() is not None:
                        # External operation owns the snapshotted Run timeout.
                        result = await function(*args, **kwargs)
                    elif is_async:
                        result = await asyncio.wait_for(
                            function(*args, **kwargs),
                            timeout=timeout,
                        )
                    else:
                        result = await asyncio.wait_for(
                            asyncio.to_thread(function, *args, **kwargs),
                            timeout=timeout,
                        )
                    if not isinstance(result, dict):
                        return _err(
                            tool_name=tool_name,
                            query=query,
                            code="BAD_TOOL_RETURN",
                            message=(
                                "Tool return type must be dict, got "
                                f"{type(result).__name__}"
                            ),
                        )
                    meta = result.setdefault("meta", {})
                    if not isinstance(meta, dict):
                        meta = result["meta"] = {}
                    meta.update(
                        attempt=attempt,
                        timeout=timeout,
                        elapsed=round(time.perf_counter() - started_at, 3),
                    )
                    return result
                except RunControlError:
                    raise
                except Exception as exc:
                    last_exception = exc
                    if not isinstance(exc, retryable):
                        break
                if attempt < max_retries:
                    delay = min(max_delay, base_delay * (2**attempt))
                    await asyncio.sleep(delay * (0.8 + 0.4 * random.random()))

            is_timeout = isinstance(
                last_exception,
                (TimeoutError, asyncio.TimeoutError),
            )
            return _err(
                tool_name=tool_name,
                query=query,
                code="TOOL_TIMEOUT" if is_timeout else "TOOL_FAILED",
                message=(
                    f"Tool call exceeded {timeout:g} seconds"
                    if is_timeout
                    else f"Tool call failed: {type(last_exception).__name__}: {last_exception}"
                ),
                meta={"retries": attempt, "timeout": timeout},
            )

        if is_async:

            @functools.wraps(function)
            async def async_wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
                return await invoke(*args, **kwargs)

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(function)
        def sync_wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
            return asyncio.run(invoke(*args, **kwargs))

        return sync_wrapper  # type: ignore[return-value]

    return decorator


__all__ = [
    "_err",
    "_ok",
    "_shrink_search_results",
    "_trim_text",
    "with_retry",
]
