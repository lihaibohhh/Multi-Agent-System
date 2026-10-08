"""Conservative layout matching with offsets back into untouched source text."""

import re
from difflib import SequenceMatcher

from .models import QuoteSpan


def _cjk(char: str) -> bool:
    return bool(char) and ("\u3400" <= char <= "\u9fff" or "\U00020000" <= char <= "\U0002ffff")


def _word(char: str) -> bool:
    return (_cjk(char) or (bool(char) and char.isascii() and char.isalnum())
            or char in set("，。；：！？、（）“”‘’《》【】「」『』,.;:!?()\"'"))


def _layout_view(text: str):
    chars, starts, ends = [], [], []
    for token in re.finditer(r"\s+|\S", text):
        value = token.group()
        if value.isspace():
            left = text[token.start() - 1] if token.start() else ""
            right = text[token.end()] if token.end() < len(text) else ""
            # Do not join English words, digits, table-like gaps or paragraphs.
            # Only one ordinary space / one wrapped line next to CJK is removable.
            gap = value.replace("\r\n", "\n")
            simple_gap = gap in {" ", "\u00a0", "\u3000", "\n"}
            if simple_gap and _word(left) and _word(right) and (_cjk(left) or _cjk(right)):
                continue
            # Retain all other whitespace verbatim: no global whitespace stripping.
        for offset, char in enumerate(value):
            chars.append(char)
            starts.append(token.start() + offset)
            ends.append(token.start() + offset + 1)
    return "".join(chars), starts, ends


def locate_quote(text: str, quote: str) -> tuple[str, QuoteSpan] | None:
    if not quote.strip():
        return None
    start = text.find(quote)
    if start >= 0:
        return quote, QuoteSpan(start=start, end=start + len(quote), match="exact")
    view, starts, ends = _layout_view(text)
    target, _, _ = _layout_view(quote)
    start = view.find(target)
    if not target or start < 0 or view.find(target, start + 1) >= 0:
        return None  # Normalization must identify one unambiguous span.
    left, right = starts[start], ends[start + len(target) - 1]
    original = text[left:right]
    if len(original) > 1000:
        return None  # Ask the model for a shorter excerpt; preserve the schema bound.
    return original, QuoteSpan(start=left, end=right, match="layout_whitespace")


def quote_repair_hint(text: str, quote: str, field: str) -> dict:
    """Suggest a source window to re-select from; NEVER use fuzzy text to accept a quote."""
    match = SequenceMatcher(None, quote, text, autojunk=False).find_longest_match()
    start = max(0, match.b - 80) if match.size >= 4 else 0
    end = min(len(text), start + 600)
    return {"field": field, "original_window": text[start:end], "window_start": start,
            "instruction": "这是未经改写的原文窗口，不代表已证实支持结论。请从完整输入中重新选择连续短摘录，"
                           "不拼接，不补句号；保留结论的不确定性与反证。"}
