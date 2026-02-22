"""Vector Store Abstraction Layer

Unified interface for multiple vector database backends with automatic
backend selection based on environment configuration.

Supported backends:
- Chroma: Local, file-based storage (development default)
- Milvus: Scalable, distributed storage (production option)

Key Operations:
- Store document embeddings with metadata
- Similarity search to retrieve top-k relevant documents
- Document management (add, delete, clear)
- Backend health checks

Usage:
    from atlasrag.src.retrieval import get_vector_store

    vector_store = get_vector_store()

    # Store embeddings
    response = vector_store.add_documents(chunks, embeddings, metadata)

    # Similarity search
    results = vector_store.similarity_search(query_embedding, top_k=5)

    # Delete documents
    delete_resp = vector_store.delete_documents(doc_ids)

    # Health check
    if vector_store.is_available():
        print("Vector store is ready")

Configuration:
    Set VECTOR_STORE environment variable to choose:
    - "chroma": Local Chroma backend (default)
    - "milvus": Milvus backend

Design Pattern:
    Factory pattern with @lru_cache() singleton. All backends implement
    the BaseVectorStore interface, enabling seamless backend switching
    without code changes.
"""

from .factory import get_vector_store
from .base import BaseVectorStore
from .chroma_store import ChromaVectorStore
from .milvus_store import MilvusVectorStore
from .models import (
    SearchResult,
    SearchResults,
    DocumentMetadata,
    StorageResponse,
    DeleteResponse,
)
from .exceptions import (
    VectorStoreException,
    VectorStoreUnavailableError,
    DocumentNotFoundError,
    SearchError,
    StorageError,
    DeletionError,
    QueryError,
)

__all__ = [
    "get_vector_store",
    "BaseVectorStore",
    "ChromaVectorStore",
    "MilvusVectorStore",
    "SearchResult",
    "SearchResults",
    "DocumentMetadata",
    "StorageResponse",
    "DeleteResponse",
    "VectorStoreException",
    "VectorStoreUnavailableError",
    "DocumentNotFoundError",
    "SearchError",
    "StorageError",
    "DeletionError",
    "QueryError",
]
