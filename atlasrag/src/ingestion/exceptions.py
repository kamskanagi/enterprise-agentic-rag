"""Custom Exceptions for Document Ingestion Layer

Exception hierarchy for the document ingestion pipeline.
All exceptions inherit from IngestionException for easy catching.
"""


class IngestionException(Exception):
    """Base exception for all ingestion operations."""

    pass


class FileNotFoundError(IngestionException):
    """Raised when a document file is not found."""

    pass


class UnsupportedFileTypeError(IngestionException):
    """Raised when file type is not supported."""

    pass


class FileSizeExceededError(IngestionException):
    """Raised when file size exceeds maximum allowed."""

    pass


class ExtractionError(IngestionException):
    """Raised when text extraction from document fails."""

    pass


class CleaningError(IngestionException):
    """Raised when text cleaning fails."""

    pass


class ChunkingError(IngestionException):
    """Raised when document chunking fails."""

    pass


class EmbeddingError(IngestionException):
    """Raised when embedding generation fails."""

    pass


class StorageError(IngestionException):
    """Raised when storing ingested data fails."""

    pass


class ValidationError(IngestionException):
    """Raised when ingestion data validation fails."""

    pass
