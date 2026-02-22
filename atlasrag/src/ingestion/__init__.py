"""Document Ingestion Pipeline

Process raw documents into searchable knowledge through a 5-stage pipeline:
1. Load: Extract text from various file formats
2. Clean: Remove formatting artifacts and normalize text
3. Chunk: Split into context-preserving segments (with overlap)
4. Embed: Convert chunks to vector embeddings (via LLM provider)
5. Store: Save embeddings to vector store

Supported File Formats:
- PDF (.pdf) - Using pypdf
- Word Documents (.docx) - Using python-docx
- Plain Text (.txt)
- HTML (.html) - Using beautifulsoup4
- Markdown (.md)

Usage:
    from atlasrag.src.ingestion import IngestionPipeline

    pipeline = IngestionPipeline()

    # Single file
    job_id = pipeline.ingest_file("path/to/document.pdf")
    status = pipeline.get_status(job_id)

    # Multiple files
    job_ids = pipeline.ingest_files(["file1.pdf", "file2.docx"])

    # Check if ingestion is complete
    if status and status.status == "completed":
        print(f"Stored {status.stored_chunks} chunks")

Configuration (from .env):
    CHUNK_SIZE=512              # Characters per chunk
    CHUNK_OVERLAP=50            # Overlap between chunks
    MAX_DOCUMENT_SIZE_MB=50     # Max file size to ingest
    SUPPORTED_FILE_TYPES=...    # Allowed extensions

Design:
- Uses Phase 3 (get_llm_client) for embeddings
- Uses Phase 4 (get_vector_store) for storage
- No backend-specific code needed!
"""

from .pipeline import IngestionPipeline
from .loaders import (
    UniversalLoader,
    TextFileLoader,
    PDFLoader,
    DocxLoader,
    MarkdownLoader,
    HTMLLoader,
)
from .cleaners import TextCleaner, DocumentCleaner
from .chunkers import DocumentChunker, RecursiveChunker, SentenceChunker
from .models import (
    DocumentChunk,
    ProcessedDocument,
    IngestionJob,
    ChunkingConfig,
    CleaningConfig,
)
from .exceptions import (
    IngestionException,
    FileNotFoundError,
    UnsupportedFileTypeError,
    FileSizeExceededError,
    ExtractionError,
    CleaningError,
    ChunkingError,
    EmbeddingError,
    StorageError,
    ValidationError,
)

__all__ = [
    "IngestionPipeline",
    "UniversalLoader",
    "TextFileLoader",
    "PDFLoader",
    "DocxLoader",
    "MarkdownLoader",
    "HTMLLoader",
    "TextCleaner",
    "DocumentCleaner",
    "DocumentChunker",
    "RecursiveChunker",
    "SentenceChunker",
    "DocumentChunk",
    "ProcessedDocument",
    "IngestionJob",
    "ChunkingConfig",
    "CleaningConfig",
    "IngestionException",
    "FileNotFoundError",
    "UnsupportedFileTypeError",
    "FileSizeExceededError",
    "ExtractionError",
    "CleaningError",
    "ChunkingError",
    "EmbeddingError",
    "StorageError",
    "ValidationError",
]
