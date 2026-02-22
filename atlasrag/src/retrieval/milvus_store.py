"""Milvus Vector Store Implementation

Scalable, distributed vector database for production deployments.
"""

from typing import List, Dict, Any, Optional
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
)

from atlasrag.src.config.vector_store_config import MilvusConfig
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
    VectorStoreUnavailableError,
)


class MilvusVectorStore(BaseVectorStore):
    """
    Vector store implementation using Milvus.

    Milvus is a scalable, open-source vector database ideal for
    large-scale production deployments.
    """

    def __init__(self, config: MilvusConfig):
        """Initialize Milvus vector store.

        Args:
            config: MilvusConfig with host, port, credentials, collection_name

        Raises:
            VectorStoreUnavailableError: If connection fails
        """
        self.config = config
        self.collection_name = config.collection_name

        try:
            # Connect to Milvus
            alias = "default"
            connections.connect(
                alias=alias,
                host=config.host,
                port=config.port,
                user=config.user if config.user else "default",
                password=config.password.get_secret_value()
                if config.password.get_secret_value()
                else "Milvus",
            )

            # Get or create collection
            if self.collection_name not in connections.list_collections(alias):
                self._create_collection()

            self.collection = Collection(self.collection_name, using=alias)
            self.collection.load()

        except Exception as e:
            raise VectorStoreUnavailableError(
                f"Failed to initialize Milvus: {str(e)}"
            )

    def _create_collection(self) -> None:
        """Create collection schema in Milvus."""
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True,
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=384,  # Default dimension, can be overridden
            ),
            FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="metadata", dtype=DataType.JSON),
        ]

        schema = CollectionSchema(
            fields=fields,
            description="Vector collection for RAG documents",
        )

        Collection.construct_from_dataframe(
            name=self.collection_name,
            schema=schema,
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

            # Prepare data
            document_ids = [f"doc_{i}" for i in range(len(documents))]

            data = [
                document_ids,
                embeddings,
                documents,
                metadata,
            ]

            # Insert into collection
            self.collection.insert(data)
            self.collection.flush()

            return StorageResponse(
                document_count=len(documents),
                vector_store="milvus",
                status="success",
            )
        except Exception as e:
            raise StorageError(f"Milvus storage failed: {str(e)}")

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

            # Perform search
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"nprobe": 32}},
                limit=top_k,
                output_fields=["document_id", "content", "metadata"],
            )

            search_results = []
            for hits in results:
                for hit in hits:
                    similarity_score = float(hit.distance)

                    if similarity_score >= similarity_threshold:
                        entity = hit.entity

                        search_results.append(
                            SearchResult(
                                document_id=entity.get("document_id", "unknown"),
                                content=entity.get("content", ""),
                                similarity_score=similarity_score,
                                metadata=entity.get("metadata", {}),
                            )
                        )

            return SearchResults(
                query=None,
                results=search_results,
                total_results=len(search_results),
                vector_store="milvus",
            )
        except Exception as e:
            raise SearchError(f"Milvus search failed: {str(e)}")

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
                    vector_store="milvus",
                    status="no documents to delete",
                )

            # Build filter expression
            filter_expr = " or ".join(
                [f'document_id == "{doc_id}"' for doc_id in document_ids]
            )

            # Delete matching documents
            self.collection.delete(filter_expr)
            self.collection.flush()

            return DeleteResponse(
                deleted_count=len(document_ids),
                vector_store="milvus",
                status="success",
            )
        except Exception as e:
            raise DeletionError(f"Milvus deletion failed: {str(e)}")

    def clear_collection(self) -> None:
        """Clear all documents from collection.

        Raises:
            DeletionError: If clearing fails
        """
        try:
            # Delete all entities
            self.collection.delete("id != 0")
            self.collection.flush()
        except Exception as e:
            raise DeletionError(f"Milvus clear failed: {str(e)}")

    def is_available(self) -> bool:
        """Check if Milvus is accessible.

        Returns:
            bool: True if Milvus is accessible
        """
        try:
            # Try to get collection info
            connections.get_connection_addr("default")
            return True
        except Exception:
            return False

    def get_document_count(self) -> int:
        """Get total document count.

        Returns:
            int: Number of documents
        """
        try:
            return self.collection.num_entities
        except Exception:
            return 0
