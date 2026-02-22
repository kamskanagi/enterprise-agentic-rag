"""Pydantic Models for Document Ingestion

Type-safe models for document ingestion pipeline operations.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel, frozen=True):
    """A single chunk of a document with metadata.

    Attributes:
        content: The text content of the chunk
        chunk_index: Sequential index of chunk within document
        source_file: Original document filename
        source_url: Optional URL if document came from web
        page_number: Page number (for PDFs, etc.)
        start_offset: Character offset in original document
        end_offset: Character offset in original document
        custom_metadata: Any additional metadata
    """

    content: str
    chunk_index: int
    source_file: str
    source_url: Optional[str] = None
    page_number: Optional[int] = None
    start_offset: int = 0
    end_offset: int = 0
    custom_metadata: Dict[str, Any] = Field(default_factory=dict)


class ProcessedDocument(BaseModel, frozen=True):
    """A document after processing through the ingestion pipeline.

    Attributes:
        document_id: Unique identifier for the document
        original_filename: Original filename
        total_chunks: Number of chunks created
        chunks: List of DocumentChunk objects
        extraction_status: Whether extraction was successful
        cleaning_status: Whether cleaning was successful
        chunking_status: Whether chunking was successful
        processed_at: Timestamp of processing
        metadata: Document-level metadata
    """

    document_id: str
    original_filename: str
    total_chunks: int
    chunks: List[DocumentChunk]
    extraction_status: str = "success"  # success, failed, partial
    cleaning_status: str = "success"
    chunking_status: str = "success"
    processed_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestionJob(BaseModel, frozen=True):
    """Metadata for an ingestion job.

    Attributes:
        job_id: Unique job identifier
        document_id: Document being processed
        status: Job status (pending, processing, completed, failed)
        total_chunks: Number of chunks created
        stored_chunks: Number of chunks stored in vector DB
        error_message: Error message if job failed
        started_at: Job start time
        completed_at: Job completion time
        progress: Progress percentage (0-100)
    """

    job_id: str
    document_id: str
    status: str  # pending, processing, completed, failed
    total_chunks: int = 0
    stored_chunks: int = 0
    error_message: Optional[str] = None
    started_at: datetime
    completed_at: Optional[datetime] = None
    progress: int = Field(default=0, ge=0, le=100)


class ChunkingConfig(BaseModel):
    """Configuration for document chunking.

    Attributes:
        chunk_size: Number of characters per chunk
        chunk_overlap: Number of overlapping characters between chunks
        separator: Text separator to use for chunking
        keep_separator: Whether to keep separator in chunks
    """

    chunk_size: int = Field(default=512, gt=0)
    chunk_overlap: int = Field(default=50, ge=0)
    separator: str = "\n"
    keep_separator: bool = True


class CleaningConfig(BaseModel):
    """Configuration for text cleaning.

    Attributes:
        remove_extra_whitespace: Whether to collapse multiple spaces
        remove_special_characters: Whether to remove special characters
        lowercase: Whether to convert to lowercase
        normalize_unicode: Whether to normalize Unicode characters
    """

    remove_extra_whitespace: bool = True
    remove_special_characters: bool = False
    lowercase: bool = False
    normalize_unicode: bool = True
