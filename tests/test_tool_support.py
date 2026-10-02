from __future__ import annotations

import asyncio
import unittest

from multi_agent_research.tools.support import (
    _shrink_search_results,
    _trim_text,
    with_retry,
)


class ToolSupportTests(unittest.TestCase):
    def test_trim_text_respects_limit(self) -> None:
        self.assertEqual(_trim_text("abcdef", 5), "ab...")
        self.assertEqual(_trim_text("abc", 5), "abc")

    def test_search_results_are_normalized_and_bounded(self) -> None:
        normalized = _shrink_search_results(
            {
                "answer": "summary",
                "results": [
                    {
                        "title": "title",
                        "url": "https://example.com",
                        "content": "abcdef",
                        "published_time": "2026-09-28",
                    }
                ],
            },
            max_items=1,
            max_chars_per_item=5,
        )
        self.assertEqual(normalized["answer"], "summary")
        self.assertEqual(normalized["results"][0]["content"], "ab...")
        self.assertEqual(
            normalized["results"][0]["published_date"],
            "2026-09-28",
        )

    def test_retry_returns_stable_envelope(self) -> None:
        attempts = 0

        @with_retry(tool_name="demo", max_retries=1, timeout=1, base_delay=0)
        async def flaky(query: str) -> dict:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ConnectionError("temporary")
            return {
                "ok": True,
                "tool": "demo",
                "query": query,
                "data": {"value": 1},
                "error": None,
                "meta": {},
            }

        result = asyncio.run(flaky("q"))
        self.assertTrue(result["ok"])
        self.assertEqual(result["meta"]["attempt"], 1)


if __name__ == "__main__":
    unittest.main()
