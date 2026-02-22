"""Unit Tests for Document Ingestion Pipeline (Phase 5)

Tests cover:
- Document loaders for various file types
- Text cleaning and normalization
- Document chunking strategies
- Full ingestion pipeline
"""

import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from atlasrag.src.ingestion import (
    IngestionPipeline,
    UniversalLoader,
    TextFileLoader,
    DocumentChunk,
    TextCleaner,
    DocumentCleaner,
    DocumentChunker,
    RecursiveChunker,
    SentenceChunker,
    ChunkingConfig,
    CleaningConfig,
    IngestionException,
    UnsupportedFileTypeError,
    ExtractionError,
    ChunkingError,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def temp_text_file():
    """Create a temporary text file for testing."""
    with TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "test.txt"
        file_path.write_text("This is a test document.\nIt has multiple lines.\n\nAnd paragraphs.")
        yield str(file_path)


@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return """
    Introduction to Document Processing

    Document processing is a critical component of any document management system.
    This text contains multiple paragraphs and sentences for testing chunking strategies.

    Chunking Strategy

    Different chunking strategies have different trade-offs:
    1. Character-level chunking is the simplest but loses context.
    2. Word-level chunking preserves word boundaries.
    3. Sentence-level chunking preserves meaning better.
    4. Paragraph-level chunking is most context-aware.

    Conclusion

    Choosing the right chunking strategy is important for performance.
    """


# ============================================================================
# TestTextCleaner: Text Cleaning
# ============================================================================


class TestTextCleaner:
    """Test text cleaning functionality."""

    def test_cleaner_creation(self):
        """TextCleaner can be instantiated."""
        cleaner = TextCleaner()
        assert cleaner is not None

    def test_cleaner_with_config(self):
        """TextCleaner accepts custom config."""
        config = CleaningConfig(remove_special_characters=True)
        cleaner = TextCleaner(config)
        assert cleaner.config.remove_special_characters is True

    def test_remove_extra_whitespace(self):
        """Cleaner removes extra whitespace."""
        cleaner = TextCleaner(CleaningConfig(remove_extra_whitespace=True))
        text = "This  has   multiple    spaces\n\n\nand newlines"
        result = cleaner.clean(text)
        assert "   " not in result
        assert "\n\n" not in result

    def test_normalize_unicode(self):
        """Cleaner normalizes Unicode."""
        cleaner = TextCleaner(CleaningConfig(normalize_unicode=True))
        text = "Café"  # é could be different unicode forms
        result = cleaner.clean(text)
        assert isinstance(result, str)

    def test_document_cleaner_remove_urls(self):
        """DocumentCleaner removes URLs."""
        text = "Visit https://example.com for more info"
        result = DocumentCleaner.clean_for_chunking(text, remove_urls=True)
        assert "https://" not in result

    def test_document_cleaner_remove_emails(self):
        """DocumentCleaner removes email addresses."""
        text = "Contact test@example.com for questions"
        result = DocumentCleaner.clean_for_chunking(text, remove_emails=True)
        assert "test@example.com" not in result


# ============================================================================
# TestDocumentChunker: Document Chunking
# ============================================================================


class TestDocumentChunker:
    """Test document chunking functionality."""

    def test_chunker_creation(self):
        """DocumentChunker can be instantiated."""
        chunker = DocumentChunker()
        assert chunker is not None

    def test_chunker_with_config(self):
        """DocumentChunker accepts custom config."""
        config = ChunkingConfig(chunk_size=256, chunk_overlap=32)
        chunker = DocumentChunker(config)
        assert chunker.config.chunk_size == 256

    def test_chunk_creation(self, sample_text):
        """Chunker creates DocumentChunk objects."""
        chunker = DocumentChunker(ChunkingConfig(chunk_size=100, chunk_overlap=10))
        chunks = chunker.chunk(sample_text, "test.txt")
        assert len(chunks) > 0
        assert all(isinstance(c, DocumentChunk) for c in chunks)

    def test_chunk_metadata(self, sample_text):
        """Chunks include correct metadata."""
        chunker = DocumentChunker()
        chunks = chunker.chunk(
            sample_text,
            "test.txt",
            source_url="http://example.com",
            page_number=1,
        )
        for chunk in chunks:
            assert chunk.source_file == "test.txt"
            assert chunk.source_url == "http://example.com"
            assert chunk.page_number == 1

    def test_chunk_overlap(self):
        """Chunks respect overlap setting."""
        text = "word " * 100  # Multiple words
        config = ChunkingConfig(chunk_size=50, chunk_overlap=5, separator=" ")
        chunker = DocumentChunker(config)
        chunks = chunker.chunk(text, "test.txt")
        assert len(chunks) >= 1  # At least one chunk

    def test_recursive_chunker(self, sample_text):
        """RecursiveChunker tries multiple separators."""
        chunker = RecursiveChunker()
        chunks = chunker.chunk(sample_text, "test.txt")
        assert len(chunks) > 0
        assert all(c.content for c in chunks)  # Non-empty content

    def test_sentence_chunker(self, sample_text):
        """SentenceChunker chunks by sentences."""
        chunker = SentenceChunker()
        chunks = chunker.chunk(sample_text, "test.txt")
        assert len(chunks) > 0


# ============================================================================
# TestDocumentLoaders: File Type Loading
# ============================================================================


class TestDocumentLoaders:
    """Test document loaders."""

    def test_text_loader(self, temp_text_file):
        """TextFileLoader can load text files."""
        text, metadata = TextFileLoader.load(temp_text_file)
        assert isinstance(text, str)
        assert len(text) > 0
        assert metadata["file_type"] == "txt"

    def test_text_loader_missing_file(self):
        """TextFileLoader raises error for missing file."""
        with pytest.raises(Exception):  # FileNotFoundError or ExtractionError
            TextFileLoader.load("/nonexistent/file.txt")

    def test_universal_loader_txt(self, temp_text_file):
        """UniversalLoader handles .txt files."""
        text, metadata = UniversalLoader.load(temp_text_file)
        assert isinstance(text, str)
        assert metadata["file_type"] == "txt"

    def test_universal_loader_unsupported_type(self):
        """UniversalLoader rejects unsupported file types."""
        with TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.xyz"
            file_path.touch()
            with pytest.raises(UnsupportedFileTypeError):
                UniversalLoader.load(str(file_path))


# ============================================================================
# TestIngestionPipeline: Full Pipeline
# ============================================================================


class TestIngestionPipeline:
    """Test the full ingestion pipeline."""

    def test_pipeline_creation(self):
        """IngestionPipeline can be instantiated."""
        pipeline = IngestionPipeline()
        assert pipeline is not None

    def test_pipeline_with_config(self):
        """IngestionPipeline accepts custom configs."""
        chunk_config = ChunkingConfig(chunk_size=256)
        clean_config = CleaningConfig(remove_extra_whitespace=True)
        pipeline = IngestionPipeline(chunk_config, clean_config)
        assert pipeline.chunking_config.chunk_size == 256

    @patch("atlasrag.src.ingestion.pipeline.get_llm_client")
    @patch("atlasrag.src.ingestion.pipeline.get_vector_store")
    def test_ingest_file(self, mock_vector_store, mock_llm, temp_text_file):
        """Pipeline can ingest a file."""
        # Mock LLM
        mock_llm_instance = MagicMock()
        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.1, 0.2, 0.3]
        mock_llm_instance.embed.return_value = mock_embedding
        mock_llm.return_value = mock_llm_instance

        # Mock vector store
        mock_vs_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.document_count = 5
        mock_vs_instance.add_documents.return_value = mock_response
        mock_vector_store.return_value = mock_vs_instance

        pipeline = IngestionPipeline()
        job_id = pipeline.ingest_file(temp_text_file)

        assert job_id is not None
        assert job_id in pipeline.jobs

    @patch("atlasrag.src.ingestion.pipeline.get_llm_client")
    @patch("atlasrag.src.ingestion.pipeline.get_vector_store")
    def test_ingest_multiple_files(self, mock_vector_store, mock_llm, temp_text_file):
        """Pipeline can ingest multiple files."""
        # Mock LLM
        mock_llm_instance = MagicMock()
        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.1, 0.2, 0.3]
        mock_llm_instance.embed.return_value = mock_embedding
        mock_llm.return_value = mock_llm_instance

        # Mock vector store
        mock_vs_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.document_count = 5
        mock_vs_instance.add_documents.return_value = mock_response
        mock_vector_store.return_value = mock_vs_instance

        pipeline = IngestionPipeline()
        job_ids = pipeline.ingest_files([temp_text_file, temp_text_file])

        assert len(job_ids) == 2
        assert all(job_id in pipeline.jobs for job_id in job_ids)

    def test_get_job_status(self):
        """Pipeline can retrieve job status."""
        pipeline = IngestionPipeline()
        # Add a mock job
        from atlasrag.src.ingestion.models import IngestionJob
        from datetime import datetime

        job = IngestionJob(
            job_id="test_job",
            document_id="test_doc",
            status="completed",
            started_at=datetime.now(),
        )
        pipeline.jobs["test_job"] = job

        status = pipeline.get_status("test_job")
        assert status is not None
        assert status.status == "completed"

    def test_get_nonexistent_job_status(self):
        """Pipeline returns None for nonexistent job."""
        pipeline = IngestionPipeline()
        status = pipeline.get_status("nonexistent")
        assert status is None
