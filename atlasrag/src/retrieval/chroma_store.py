"""Chroma Vector Store Implementation

Local, file-based vector database for development and small-scale deployments.
"""

from typing import List, Dict, Any, Optional
import chromadb

from atlasrag.src.config.vector_store_config import ChromaConfig
from .base import BaseVectorStore
from .models import (
    SearchResult,
    SearchResults,
    StorageResponse,
    DeleteResponse,
)
from .exceptions import (
    SearchError,
    StorageError,
    DeletionError,
    DocumentNotFoundError,
    VectorStoreUnavailableError,
)


class ChromaVectorStore(BaseVectorStore):
    """
    Vector store implementation using Chroma.

    Chroma is a local, file-based vector database ideal for development
    and small-to-medium scale deployments.
    """

    def __init__(self, config: ChromaConfig):
        """Initialize Chroma vector store.

        Args:
            config: ChromaConfig with persist_directory and collection_name
        """
        self.config = config
        self.collection_name = config.collection_name

        try:
            # Initialize Chroma client with persistent storage
            self.client = chromadb.PersistentClient(
                path=config.persist_directory,
            )
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            raise VectorStoreUnavailableError(
                f"Failed to initialize Chroma: {str(e)}"
            )

    def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> StorageResponse:
        """Store document chunks with embeddings.

        Args:
            documents: List of document texts
            embeddings: List of embedding vectors
            metadata: Optional list of metadata dicts

        Returns:
            StorageResponse: Storage confirmation

        Raises:
            StorageError: If storage fails
        """
        try:
            if not documents or not embeddings:
                raise ValueError("Documents and embeddings cannot be empty")

            if len(documents) != len(embeddings):
                raise ValueError("Documents and embeddings must have same length")

            if metadata is None:
                metadata = [{"index": i} for i in range(len(documents))]
            elif len(metadata) != len(documents):
                raise ValueError("Metadata must have same length as documents")

            # Generate unique IDs using existing count as offset
            offset = self.collection.count()
            ids = [f"doc_{offset + i}" for i in range(len(documents))]

            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadata,
            )

            return StorageResponse(
                document_count=len(documents),
                vector_store="chroma",
                status="success",
            )
        except Exception as e:
            raise StorageError(f"Chroma storage failed: {str(e)}")

    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        similarity_threshold: float = 0.0,
    ) -> SearchResults:
        """Find similar documents.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score

        Returns:
            SearchResults: Similar documents

        Raises:
            SearchError: If search fails
        """
        try:
            if not query_embedding:
                raise ValueError("Query embedding cannot be empty")

            if top_k < 1:
                raise ValueError("top_k must be at least 1")

            if not (0.0 <= similarity_threshold <= 1.0):
                raise ValueError("similarity_threshold must be between 0.0 and 1.0")

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )

            # Convert distances to similarity scores (Chroma returns distances)
            search_results = []
            if results["documents"] and results["documents"][0]:
                for doc, metadata, distance in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                ):
                    # Convert distance to similarity (1 - distance for cosine)
                    similarity_score = 1 - distance

                    if similarity_score >= similarity_threshold:
                        search_results.append(
                            SearchResult(
                                document_id=metadata.get("index", "unknown"),
                                content=doc,
                                similarity_score=float(similarity_score),
                                metadata=metadata,
                            )
                        )

            return SearchResults(
                query=None,
                results=search_results,
                total_results=len(search_results),
                vector_store="chroma",
            )
        except Exception as e:
            raise SearchError(f"Chroma search failed: {str(e)}")

    def delete_documents(self, document_ids: List[str]) -> DeleteResponse:
        """Delete documents by ID.

        Args:
            document_ids: List of document IDs to delete

        Returns:
            DeleteResponse: Deletion confirmation

        Raises:
            DeletionError: If deletion fails
        """
        try:
            if not document_ids:
                return DeleteResponse(
                    deleted_count=0,
                    vector_store="chroma",
                    status="no documents to delete",
                )

            self.collection.delete(ids=document_ids)

            return DeleteResponse(
                deleted_count=len(document_ids),
                vector_store="chroma",
                status="success",
            )
        except Exception as e:
            raise DeletionError(f"Chroma deletion failed: {str(e)}")

    def clear_collection(self) -> None:
        """Clear all documents from collection.

        Raises:
            DeletionError: If clearing fails
        """
        try:
            # Delete the old collection and create a new one
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            raise DeletionError(f"Chroma clear failed: {str(e)}")

    def is_available(self) -> bool:
        """Check if Chroma is accessible.

        Returns:
            bool: True if Chroma is accessible
        """
        try:
            # Try to get collection count
            self.client.list_collections()
            return True
        except Exception:
            return False

    def get_document_count(self) -> int:
        """Get total document count.

        Returns:
            int: Number of documents
        """
        try:
            return self.collection.count()
        except Exception:
            return 0
