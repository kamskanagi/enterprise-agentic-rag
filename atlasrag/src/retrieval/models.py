"""Pydantic Models for Vector Store Responses

Type-safe response models for vector store operations with immutable (frozen) BaseModels.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class SearchResult(BaseModel, frozen=True):
    """
    Result from a similarity search.

    Attributes:
        document_id: Unique identifier for the document
        content: The text content of the document chunk
        similarity_score: Similarity score (0.0-1.0, higher is more similar)
        metadata: Associated metadata (source, page, etc.)
    """

    document_id: str
    content: str
    similarity_score: float
    metadata: Dict[str, Any]


class SearchResults(BaseModel, frozen=True):
    """
    Collection of search results from a query.

    Attributes:
        query: The query embedding or text that was searched
        results: List of SearchResult objects
        total_results: Total number of results found (before limit)
        vector_store: Name of the vector store backend used
    """

    query: Optional[str]
    results: List[SearchResult]
    total_results: int
    vector_store: str


class DocumentMetadata(BaseModel, frozen=True):
    """
    Metadata associated with a stored document.

    Attributes:
        source: Source file or identifier
        page: Page number if applicable
        chunk_index: Index of the chunk within the document
        timestamp: When the document was added
        custom_fields: Any additional custom metadata
    """

    source: str
    page: Optional[int] = None
    chunk_index: Optional[int] = None
    timestamp: Optional[str] = None
    custom_fields: Dict[str, Any] = {}


class StorageResponse(BaseModel, frozen=True):
    """
    Response from document storage/insertion.

    Attributes:
        document_count: Number of documents stored
        vector_store: Name of the vector store backend
        status: Status message
    """

    document_count: int
    vector_store: str
    status: str


class DeleteResponse(BaseModel, frozen=True):
    """
    Response from document deletion.

    Attributes:
        deleted_count: Number of documents deleted
        vector_store: Name of the vector store backend
        status: Status message
    """

    deleted_count: int
    vector_store: str
    status: str
