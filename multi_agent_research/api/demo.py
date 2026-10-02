"""Serve the standalone browser demo assets."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


STATIC_DIR = Path(__file__).with_name("static")
INDEX_FILE = STATIC_DIR / "index.html"


def register_demo(app: FastAPI) -> None:
    """Mount static assets and register the demo landing page."""
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    async def demo_page() -> FileResponse:
        return FileResponse(INDEX_FILE, media_type="text/html")

    app.add_api_route("/", demo_page, methods=["GET"], include_in_schema=False)
