"""
Document Ingestion Pipeline

TODO: Phase 5 - Process raw documents into searchable knowledge

Pipeline Stages:
1. Load: Extract text from various file formats
2. Clean: Remove formatting artifacts and normalize text
3. Chunk: Split into context-preserving segments (with overlap)
4. Embed: Convert chunks to vector embeddings
5. Store: Save embeddings to vector DB + metadata to PostgreSQL

Supported File Formats:
- PDF (.pdf) - Using pypdf
- Word Documents (.docx) - Using python-docx
- Plain Text (.txt)
- HTML (.html) - Using beautifulsoup4
- Markdown (.md)

Key files to be implemented:
- loaders.py: Document loaders for each file type
- cleaners.py: Text preprocessing and normalization
- chunkers.py: Smart text splitting strategies
- pipeline.py: Orchestrates the full ingestion flow
- models.py: Database models for tracking ingestion jobs

Usage example (Phase 5+):
    from src.ingestion.pipeline import IngestionPipeline
    pipeline = IngestionPipeline()

    job_id = pipeline.ingest_file("path/to/document.pdf")
    status = pipeline.get_status(job_id)

Configuration (from .env):
- CHUNK_SIZE: Characters per chunk (default: 1000)
- CHUNK_OVERLAP: Overlap between chunks (default: 200)
- MAX_DOCUMENT_SIZE_MB: Max file size to ingest
- SUPPORTED_FILE_TYPES: Allowed extensions
"""
