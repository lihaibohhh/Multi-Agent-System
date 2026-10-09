"""Deterministic, non-Agent retrieval capabilities."""

from .models import RetrievalRequest, SearchResult
from .service import retrieve_evidence

__all__ = ["RetrievalRequest", "SearchResult", "retrieve_evidence"]
