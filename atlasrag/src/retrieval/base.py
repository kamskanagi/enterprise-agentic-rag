"""Abstract Base Class for Vector Stores

All concrete vector store implementations must inherit from BaseVectorStore
and implement the abstract methods.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from .models import SearchResult, SearchResults, StorageResponse, DeleteResponse


class BaseVectorStore(ABC):
    """
    Abstract interface for vector store backends.

    All vector stores (Chroma, Milvus, etc.) must implement these methods
    to ensure a consistent interface across different vector database backends.
    """

    @abstractmethod
    def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> StorageResponse:
        """
        Store document chunks with embeddings.

        Args:
            documents: List of document texts/chunks
            embeddings: List of embedding vectors (one per document)
            metadata: Optional list of metadata dicts (one per document)

        Returns:
            StorageResponse: Response with document count and status

        Raises:
            StorageError: If storage fails
            QueryError: If input validation fails
        """
        pass

    @abstractmethod
    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        similarity_threshold: float = 0.0,
    ) -> SearchResults:
        """
        Find documents most similar to a query embedding.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return (default 5)
            similarity_threshold: Minimum similarity score (0.0-1.0)

        Returns:
            SearchResults: List of most similar documents

        Raises:
            SearchError: If search fails
            QueryError: If query parameters are invalid
        """
        pass

    @abstractmethod
    def delete_documents(self, document_ids: List[str]) -> DeleteResponse:
        """
        Delete documents by ID.

        Args:
            document_ids: List of document IDs to delete

        Returns:
            DeleteResponse: Response with deletion count and status

        Raises:
            DeletionError: If deletion fails
            DocumentNotFoundError: If documents don't exist
        """
        pass

    @abstractmethod
    def clear_collection(self) -> None:
        """
        Clear all documents from the collection.

        Raises:
            DeletionError: If clearing fails
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the vector store backend is accessible.

        Returns:
            bool: True if backend is reachable, False otherwise
        """
        pass

    @abstractmethod
    def get_document_count(self) -> int:
        """
        Get total number of documents in the store.

        Returns:
            int: Number of documents
        """
        pass
