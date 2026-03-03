"""Document Ingestion Pipeline

Orchestrates the full ingestion workflow: load → clean → chunk → embed → store.
"""

from typing import List, Optional
from datetime import datetime
import logging
import uuid

from atlasrag.src.config import get_settings
from atlasrag.src.ingestion.loaders import UniversalLoader
from atlasrag.src.ingestion.cleaners import DocumentCleaner
from atlasrag.src.ingestion.chunkers import RecursiveChunker
from atlasrag.src.ingestion.models import (
    DocumentChunk,
    ProcessedDocument,
    IngestionJob,
    ChunkingConfig,
    CleaningConfig,
)
from atlasrag.src.ingestion.exceptions import (
    IngestionException,
    FileSizeExceededError,
    ExtractionError,
    ValidationError,
)
from atlasrag.src.llm import get_llm_client
from atlasrag.src.retrieval import get_vector_store

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Full document ingestion pipeline."""

    def __init__(
        self,
        chunking_config: Optional[ChunkingConfig] = None,
        cleaning_config: Optional[CleaningConfig] = None,
    ):
        """Initialize the ingestion pipeline.

        Args:
            chunking_config: Configuration for chunking
            cleaning_config: Configuration for text cleaning
        """
        self.settings = get_settings()
        self.chunking_config = chunking_config or ChunkingConfig(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        self.cleaning_config = cleaning_config or CleaningConfig()
        self.jobs = {}  # Simple in-memory job tracking

    def ingest_file(self, file_path: str, metadata: dict = None) -> str:
        """Ingest a single document file.

        Args:
            file_path: Path to the document file
            metadata: Optional custom metadata

        Returns:
            Job ID for tracking progress

        Raises:
            IngestionException: If ingestion fails
        """
        job_id = str(uuid.uuid4())
        document_id = str(uuid.uuid4())

        try:
            # Check file size
            import os

            file_size = os.path.getsize(file_path)
            max_size_bytes = self.settings.max_document_size_mb * 1024 * 1024

            if file_size > max_size_bytes:
                raise FileSizeExceededError(
                    f"File size {file_size} exceeds maximum {max_size_bytes}"
                )

            # Create job record
            job = IngestionJob(
                job_id=job_id,
                document_id=document_id,
                status="processing",
                started_at=datetime.now(),
                progress=0,
            )
            self.jobs[job_id] = job

            # Stage 1: Load
            logger.info(f"Loading document: {file_path}")
            text, doc_metadata = UniversalLoader.load(file_path)
            self._update_job(job_id, progress=20)

            # Stage 2: Clean
            logger.info(f"Cleaning document: {file_path}")
            text = DocumentCleaner.clean_for_chunking(
                text,
                remove_headers=True,
                remove_urls=False,
                remove_emails=False,
            )
            self._update_job(job_id, progress=40)

            # Stage 3: Chunk
            logger.info(f"Chunking document: {file_path}")
            chunker = RecursiveChunker(self.chunking_config)
            chunks = chunker.chunk(
                text=text,
                source_file=os.path.basename(file_path),
                custom_metadata=metadata or {},
            )
            self._update_job(job_id, total_chunks=len(chunks), progress=60)

            # Stage 4: Embed
            logger.info(f"Generating embeddings: {len(chunks)} chunks")
            llm = get_llm_client()
            embeddings = []
            for chunk in chunks:
                try:
                    embedding = llm.embed(chunk.content)
                    embeddings.append(embedding.embedding)
                except Exception as e:
                    logger.warning(f"Failed to embed chunk {chunk.chunk_index}: {e}")
                    # Use zero vector as fallback
                    embeddings.append([0.0] * 384)  # Default dimension
            self._update_job(job_id, progress=80)

            # Stage 5: Store
            logger.info(f"Storing in vector database: {len(chunks)} chunks")
            vector_store = get_vector_store()

            # Prepare metadata for storage (filter out None values — Chroma rejects them)
            metadatas = []
            for chunk in chunks:
                meta = {
                    **{k: v for k, v in chunk.custom_metadata.items() if v is not None},
                    "source": chunk.source_file,
                    "chunk_index": chunk.chunk_index,
                    "document_id": document_id,
                }
                if chunk.page_number is not None:
                    meta["page"] = chunk.page_number
                metadatas.append(meta)

            # Store embeddings
            response = vector_store.add_documents(
                documents=[chunk.content for chunk in chunks],
                embeddings=embeddings,
                metadata=metadatas,
            )

            self._update_job(
                job_id,
                stored_chunks=response.document_count,
                progress=100,
                status="completed",
            )

            logger.info(f"Successfully ingested: {file_path}")
            return job_id

        except IngestionException as e:
            logger.error(f"Ingestion error for {file_path}: {e}")
            self._update_job(
                job_id,
                status="failed",
                error_message=str(e),
            )
            raise
        except Exception as e:
            logger.error(f"Unexpected error during ingestion: {e}")
            self._update_job(
                job_id,
                status="failed",
                error_message=f"Unexpected error: {str(e)}",
            )
            raise IngestionException(f"Ingestion failed: {str(e)}")

    def ingest_files(self, file_paths: List[str], metadata: dict = None) -> List[str]:
        """Ingest multiple document files.

        Args:
            file_paths: List of paths to document files
            metadata: Optional custom metadata for all documents

        Returns:
            List of job IDs
        """
        job_ids = []
        for file_path in file_paths:
            try:
                job_id = self.ingest_file(file_path, metadata)
                job_ids.append(job_id)
            except Exception as e:
                logger.error(f"Failed to ingest {file_path}: {e}")

        return job_ids

    def get_status(self, job_id: str) -> Optional[IngestionJob]:
        """Get the status of an ingestion job.

        Args:
            job_id: The job ID

        Returns:
            IngestionJob with current status, or None if not found
        """
        return self.jobs.get(job_id)

    def _update_job(
        self,
        job_id: str,
        status: str = None,
        total_chunks: int = None,
        stored_chunks: int = None,
        progress: int = None,
        error_message: str = None,
    ) -> None:
        """Update job status in memory.

        Args:
            job_id: The job ID
            status: New status
            total_chunks: Total chunks count
            stored_chunks: Stored chunks count
            progress: Progress percentage
            error_message: Error message if failed
        """
        if job_id not in self.jobs:
            return

        old_job = self.jobs[job_id]

        self.jobs[job_id] = IngestionJob(
            job_id=old_job.job_id,
            document_id=old_job.document_id,
            status=status or old_job.status,
            total_chunks=total_chunks or old_job.total_chunks,
            stored_chunks=stored_chunks or old_job.stored_chunks,
            error_message=error_message or old_job.error_message,
            started_at=old_job.started_at,
            completed_at=datetime.now() if status == "completed" else None,
            progress=progress or old_job.progress,
        )
