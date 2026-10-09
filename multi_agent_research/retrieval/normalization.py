"""Pure normalization and within-round deduplication for retrieval responses."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .models import SearchResult


DEDUP_KEY_LEN = 200


def normalize_knowledge_response(
    response: Any,
    *,
    query: str,
    iteration: int,
    max_chars: int,
) -> list[SearchResult]:
    results: list[SearchResult] = []
    for chunk in response.chunks:
        page = chunk.source_page if chunk.source_page is not None else -1
        prefix = (
            f"[来源：{chunk.source_file}   第 {page} 页]"
            if page >= 1
            else f"[来源：{chunk.source_file}]"
        )
        score = chunk.score if chunk.score is not None else response.top_score
        results.append(
            SearchResult(
                query=query,
                source="knowledge",
                content=f"{prefix}\n{chunk.content[:max_chars]}",
                score=float(score or 0.0),
                metadata={
                    "source": chunk.source_file,
                    "page": page,
                    "chunk_id": chunk.chunk_id,
                    "industry": chunk.industry or "unknown",
                    "doc_type": chunk.doc_type,
                    "retrieval_stage": response.stage,
                    "cache_hit": response.cache_hit,
                },
                iteration=iteration,
            )
        )
    return results


def normalize_web_items(
    items: list[dict],
    *,
    query: str,
    iteration: int,
    max_chars: int,
    maximum: int,
) -> list[SearchResult]:
    results: list[SearchResult] = []
    for item in items[:maximum]:
        url = item.get("url", "")
        title = item.get("title", "")
        published_date = item.get("published_date")
        prefix_parts = [f"来源：{url}"]
        if title:
            prefix_parts.append(f"标题：{title}")
        if published_date:
            prefix_parts.append(f"日期：{published_date}")
        prefix = "[" + " | ".join(prefix_parts) + "]"
        score = max(0.0, min(float(item.get("score") or 0.5), 1.0))
        results.append(
            SearchResult(
                query=query,
                source="web",
                content=f"{prefix}\n{item.get('content', '')[:max_chars]}",
                score=score,
                metadata={
                    "url": url,
                    "title": title,
                    "published_date": published_date,
                    "industry": "unknown",
                },
                iteration=iteration,
            )
        )
    return results


def deduplicate_batches(
    batches: Iterable[list[SearchResult]],
) -> list[SearchResult]:
    seen: set[str] = set()
    results: list[SearchResult] = []
    for batch in batches:
        for result in sorted(batch, key=lambda item: item["score"], reverse=True):
            key = result["content"][:DEDUP_KEY_LEN]
            if key not in seen:
                seen.add(key)
                results.append(result)
    return results
