"""Vector Store Factory

Singleton factory function that instantiates the correct vector store
based on configuration. Uses @lru_cache() for singleton pattern.
"""

from functools import lru_cache

from atlasrag.src.config import get_settings
from .base import BaseVectorStore
from .exceptions import VectorStoreUnavailableError


@lru_cache()
def get_vector_store() -> BaseVectorStore:
    """
    Get vector store instance based on settings.

    Returns the appropriate vector store (Chroma or Milvus)
    based on settings.vector_store. Uses @lru_cache() singleton pattern.

    Returns:
        BaseVectorStore: Instantiated vector store matching current settings

    Raises:
        VectorStoreUnavailableError: If backend initialization fails

    Example:
        from atlasrag.src.retrieval import get_vector_store

        vector_store = get_vector_store()
        results = vector_store.similarity_search(embedding, top_k=5)
    """
    settings = get_settings()
    vs_config = settings.get_vector_store_config()

    backend = vs_config.backend

    if backend == "chroma":
        from .chroma_store import ChromaVectorStore

        return ChromaVectorStore(vs_config.chroma)

    elif backend == "milvus":
        from .milvus_store import MilvusVectorStore

        return MilvusVectorStore(vs_config.milvus)

    else:
        raise VectorStoreUnavailableError(
            f"Vector store backend '{backend}' is not supported. "
            f"Choose from: chroma, milvus"
        )
