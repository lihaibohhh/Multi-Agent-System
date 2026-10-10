"""Deterministic evidence identity, local citation checks and report assembly."""

import hashlib
import json
import re

from .models import SectionRecord


CITATION = re.compile(r"\[来源(\d+)\]")

SOURCE_LABEL: dict[str, str] = {
    "knowledge": "内部知识库服务",
    "web": "联网检索",
    "cache": "语义缓存",
}


def extract_doc_title(result: dict) -> str:
    """Return a human-readable source title from retrieval metadata."""
    metadata = result.get("metadata") or {}
    file_path = metadata.get("source", "")
    if file_path:
        filename = file_path.replace("\\", "/").split("/")[-1]
        return filename.removesuffix(".pdf")[:60]
    chunk_id = metadata.get("chunk_id", "")
    if chunk_id:
        raw = chunk_id.split("::")[0]
        filename = raw.replace("\\", "/").split("/")[-1]
        return filename.removesuffix(".pdf")[:60]
    url = metadata.get("url", "")
    return url[:80] if url else ""


def format_results_for_prompt(results: list[dict], max_content: int = 600) -> str:
    """Format retrieval artifacts for model prompts with stable citation labels."""
    parts: list[str] = []
    for index, result in enumerate(results, 1):
        source = result["source"]
        source_label = SOURCE_LABEL.get(source, source)
        doc_title = extract_doc_title(result)
        title_part = f" · {doc_title}" if doc_title else ""
        parts.append(
            f"[来源{index}] {source_label}{title_part} | 相关性 {result['score']:.2f}\n"
            f"检索词：{result['query']}\n"
            f"内容：{result['content'][:max_content]}"
        )
    return "\n\n".join(parts)


def evidence_key(result: dict) -> str:
    meta = result.get("metadata") or {}
    # Retain different excerpts/versions of the same chunk or URL.
    identity = [
        result.get("source"), meta.get("chunk_id"), meta.get("source"),
        meta.get("page"), meta.get("url"), result.get("content", ""),
    ]
    return hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode()).hexdigest()


def document_key(result: dict) -> str:
    """Identify a public reference document while keeping excerpt identity separate."""
    meta = result.get("metadata") or {}
    identity = (
        ("url", meta["url"])
        if meta.get("url")
        else ("source", meta["source"])
        if meta.get("source")
        else ("title", meta["title"])
        if meta.get("title")
        else ("evidence", evidence_key(result))
    )
    return hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode()).hexdigest()


def merge_results(existing: list[dict], incoming: list[dict]) -> list[dict]:
    merged = {evidence_key(item): item for item in existing}
    for item in incoming:
        merged.setdefault(evidence_key(item), item)
    return list(merged.values())


def citation_issues(draft: str, sources: list[dict]) -> list[str]:
    citations = [int(value) for value in CITATION.findall(draft)]
    invalid = sorted({value for value in citations if not 1 <= value <= len(sources)})
    issues = [f"引用不存在的来源编号：{invalid}"] if invalid else []
    if sources and not citations:
        issues.append("正文没有任何 [来源N] 引用，不能作为已审校章节")
    if not draft.strip():
        issues.append("章节正文为空")
    return issues


def assemble_report(
    question: str,
    sections: list[SectionRecord],
    *,
    limited: bool | None = None,
) -> str:
    """Build the consumer report without leaking internal review artifacts."""
    references: list[dict] = []
    registry: dict[str, int] = {}
    parts = [f"# 研究报告：{question}"]
    is_limited = (
        any(section.status == "limited" for section in sections)
        if limited is None
        else limited
    )
    if is_limited:
        parts.append(
            "> 阅读提示：部分结论的证据强度有限。正文已保留适用范围、"
            "时效和不确定性说明，建议结合参考来源审慎使用。"
        )
    for section in sections:
        if section.status not in {"complete", "limited"}:
            raise ValueError(f"section {section.section_id} is unfinished")
        issues = citation_issues(section.draft, section.sources)
        if issues:
            raise ValueError("; ".join(issues))

        def replace(match: re.Match) -> str:
            source = section.sources[int(match.group(1)) - 1]
            key = document_key(source)
            if key not in registry:
                references.append({"source": source, "pages": set()})
                registry[key] = len(references)
            page = (source.get("metadata") or {}).get("page")
            if isinstance(page, int) and page >= 1:
                references[registry[key] - 1]["pages"].add(page)
            return f"[来源{registry[key]}]"

        body = CITATION.sub(replace, section.draft)
        parts.append(f"## {section.title}\n\n{body}")
    lines = ["## 参考来源"]
    for index, reference in enumerate(references, 1):
        source = reference["source"]
        meta = source.get("metadata") or {}
        locator = (
            meta.get("title")
            or extract_doc_title(source)
            or meta.get("publisher")
            or "来源名称未提供"
        )
        pages = sorted(reference["pages"])
        if pages:
            locator += " | " + ", ".join(f"p.{page}" for page in pages)
        if meta.get("url"):
            locator += f" | <{meta['url']}>"
        lines.append(f"[来源{index}] {locator}")
    parts.append("\n\n".join(lines))
    return "\n\n".join(parts)
