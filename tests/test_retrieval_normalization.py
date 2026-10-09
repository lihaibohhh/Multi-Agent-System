from __future__ import annotations

from multi_agent_research.retrieval.normalization import (
    deduplicate_batches,
    normalize_web_items,
)


def test_web_results_are_normalized_and_bounded() -> None:
    normalized = normalize_web_items(
        [
            {
                "title": "title",
                "url": "https://example.com",
                "content": "abcdef",
                "published_date": "2026-09-28",
                "score": 2,
            },
            {"url": "https://ignored.example.com", "content": "ignored"},
        ],
        query="q",
        iteration=2,
        max_chars=5,
        maximum=1,
    )

    assert len(normalized) == 1
    assert normalized[0]["content"] == (
        "[来源：https://example.com | 标题：title | 日期：2026-09-28]\nabcde"
    )
    assert normalized[0]["metadata"]["published_date"] == "2026-09-28"
    assert normalized[0]["score"] == 1.0
    assert normalized[0]["iteration"] == 2


def test_deduplicate_batches_keeps_highest_scored_duplicate() -> None:
    low = {
        "query": "q1",
        "source": "web",
        "content": "same content",
        "score": 0.2,
        "metadata": {},
        "iteration": 0,
    }
    high = {**low, "query": "q2", "score": 0.9}

    results = deduplicate_batches([[low, high]])

    assert results == [high]
