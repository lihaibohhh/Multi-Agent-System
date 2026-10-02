"""隔壁 knowledge-service 的只读 HTTP 客户端。"""

from .client import (
    KnowledgeSearchResponse,
    KnowledgeServiceClient,
    KnowledgeServiceError,
    KnowledgeServiceUnavailable,
    RetrievedChunk,
    get_knowledge_service_client,
)

__all__ = [
    "KnowledgeSearchResponse",
    "KnowledgeServiceClient",
    "KnowledgeServiceError",
    "KnowledgeServiceUnavailable",
    "RetrievedChunk",
    "get_knowledge_service_client",
]
