"""Versioned, opaque excerpt catalogs backed by exact source offsets."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Literal

from .models import SectionRecord


SEGMENTATION_VERSION = 3
_MAX_SOURCE_CHARS = 1000
_MAX_SEGMENT_CHARS = 450


@dataclass(frozen=True, slots=True)
class ExcerptSegment:
    segment_id: str
    kind: Literal["draft", "evidence"]
    start: int
    end: int
    text: str
    source_number: int | None = None


@dataclass(frozen=True, slots=True)
class SegmentCatalog:
    fingerprint: str
    segmentation_version: int
    segments: dict[str, ExcerptSegment]

    def prompt_view(self, source_numbers: set[int] | None = None) -> dict:
        return {
            segment_id: {
                "kind": segment.kind,
                "source_number": segment.source_number,
                "text": _display_text(segment.text),
            }
            for segment_id, segment in self.segments.items()
            if (
                segment.kind == "draft"
                or source_numbers is None
                or segment.source_number in source_numbers
            )
        }


def _display_text(text: str) -> str:
    """Hide PDF line wrapping from the model without changing stored offsets."""

    return re.sub(r"(?<!\n)[ \t]*\n[ \t]*(?!\n)", " ", text)


def _bounded_spans(text: str, start: int, end: int):
    while end - start > _MAX_SEGMENT_CHARS:
        limit = start + _MAX_SEGMENT_CHARS
        window = text[start:limit]
        cut = max(window.rfind(mark) for mark in ("\n", "；", ";", "，", ",", " "))
        split = start + cut + 1 if cut >= _MAX_SEGMENT_CHARS // 2 else limit
        if text[start:split].strip():
            yield start, split
        start = split
    if text[start:end].strip():
        yield start, end


def _spans(text: str):
    """Yield exact spans while treating single PDF line breaks as soft wraps."""

    start = 0
    boundary = re.compile(
        r"[。！？!?]+[”’」』】)]*|(?<=[.!?])(?=\s+[A-Z0-9])|\n[ \t]*\n+"
    )
    for match in boundary.finditer(text):
        end = match.end()
        yield from _bounded_spans(text, start, end)
        start = end
    if start < len(text):
        yield from _bounded_spans(text, start, len(text))


def _catalog_fingerprint(section: SectionRecord) -> str:
    payload = {
        "segmentation_version": SEGMENTATION_VERSION,
        "section_id": section.section_id,
        "revision": section.revision,
        "draft": section.draft,
        "sources": [source.get("content", "")[:_MAX_SOURCE_CHARS] for source in section.sources],
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def build_segment_catalog(section: SectionRecord) -> SegmentCatalog:
    """Build stable handles scoped to one exact chapter/source snapshot."""

    segments: dict[str, ExcerptSegment] = {}
    for index, (start, end) in enumerate(_spans(section.draft), 1):
        segment_id = f"D{index:04d}"
        segments[segment_id] = ExcerptSegment(
            segment_id=segment_id,
            kind="draft",
            start=start,
            end=end,
            text=section.draft[start:end],
        )

    evidence_index = 0
    for source_number, source in enumerate(section.sources, 1):
        text = source.get("content", "")[:_MAX_SOURCE_CHARS]
        for start, end in _spans(text):
            evidence_index += 1
            segment_id = f"E{evidence_index:04d}"
            segments[segment_id] = ExcerptSegment(
                segment_id=segment_id,
                kind="evidence",
                source_number=source_number,
                start=start,
                end=end,
                text=text[start:end],
            )

    return SegmentCatalog(
        fingerprint=_catalog_fingerprint(section),
        segmentation_version=SEGMENTATION_VERSION,
        segments=segments,
    )


__all__ = [
    "ExcerptSegment",
    "SEGMENTATION_VERSION",
    "SegmentCatalog",
    "build_segment_catalog",
]
