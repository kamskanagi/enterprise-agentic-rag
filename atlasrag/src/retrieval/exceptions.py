"""Custom Exceptions for Vector Store Layer

Exception hierarchy for the vector store abstraction layer.
All exceptions inherit from VectorStoreException for easy catching.
"""


class VectorStoreException(Exception):
    """Base exception for all vector store operations."""

    pass


class VectorStoreUnavailableError(VectorStoreException):
    """Raised when the vector store backend is not accessible."""

    pass


class DocumentNotFoundError(VectorStoreException):
    """Raised when a document is not found in the vector store."""

    pass


class SearchError(VectorStoreException):
    """Raised when similarity search fails."""

    pass


class StorageError(VectorStoreException):
    """Raised when document storage/insertion fails."""

    pass


class DeletionError(VectorStoreException):
    """Raised when document deletion fails."""

    pass


class QueryError(VectorStoreException):
    """Raised when query validation or execution fails."""

    pass
