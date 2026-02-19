"""Vector Store Configuration Models

Frozen BaseModel classes for vector database configurations.
"""

from typing import Literal
from pydantic import BaseModel, SecretStr


class ChromaConfig(BaseModel, frozen=True):
    """Configuration for Chroma vector database (local)."""

    persist_directory: str
    collection_name: str


class MilvusConfig(BaseModel, frozen=True):
    """Configuration for Milvus vector database (scalable)."""

    host: str
    port: int
    user: str
    password: SecretStr
    collection_name: str


class VectorStoreConfig(BaseModel, frozen=True):
    """Unified vector store configuration with backend-specific sub-configs."""

    backend: Literal["chroma", "milvus"]
    chroma: ChromaConfig
    milvus: MilvusConfig
    retrieval_top_k: int
    similarity_threshold: float
    enable_reranking: bool
