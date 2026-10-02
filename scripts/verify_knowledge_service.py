"""验证 knowledge-service 就绪状态与只读检索契约。"""

from __future__ import annotations

import argparse
import asyncio

from multi_agent_research.knowledge.client import get_knowledge_service_client


async def _verify(query: str) -> None:
    client = get_knowledge_service_client()
    try:
        ready = await client.ensure_ready()
        result = await client.search(query, top_k=3)
        first = result.chunks[0] if result.chunks else None
        print({
            "ready": ready.get("ready"),
            "chunk_count": ready.get("chunk_count"),
            "results": len(result.chunks),
            "stage": result.stage,
            "first_source": first.source_file if first else None,
            "first_page": first.source_page if first else None,
        })
    finally:
        await client.aclose()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", default="人工智能 教育")
    args = parser.parse_args()
    asyncio.run(_verify(args.query))


if __name__ == "__main__":
    main()
