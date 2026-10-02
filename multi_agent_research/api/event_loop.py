"""Event-loop factory compatible with psycopg async connections on Windows."""

from __future__ import annotations

import asyncio
import sys


def selector_loop_factory() -> asyncio.AbstractEventLoop:
    """Return a selector loop on Windows and the platform default elsewhere."""
    if sys.platform == "win32":
        return asyncio.SelectorEventLoop()
    return asyncio.new_event_loop()
