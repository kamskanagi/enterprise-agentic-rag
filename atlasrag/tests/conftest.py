"""
Pytest Configuration and Shared Fixtures

This file contains pytest configuration and fixtures that are shared across
all test files in the atlasrag/tests/ directory.

TODO: Phases 2-11 - Add fixtures as modules are implemented
Each phase will add fixtures for testing its module.
"""

import pytest
import os
from pathlib import Path


# ============================================================================
# Configuration
# ============================================================================

pytest_plugins = []

# Set test environment
os.environ.setdefault("DEBUG", "true")
os.environ.setdefault("LOG_LEVEL", "DEBUG")


# ============================================================================
# Phase 2 - Configuration Fixtures
# ============================================================================

@pytest.fixture
def test_settings(monkeypatch):
    """Provide test-specific settings with defaults.

    Clears the lru_cache before and after to ensure each test gets fresh settings.
    Uses monkeypatch to set environment variables without writing to .env file.
    """
    from atlasrag.src.config.settings import get_settings

    # Clear cache before test
    get_settings.cache_clear()

    # Set default test environment variables
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("VECTOR_STORE", "chroma")
    monkeypatch.setenv("DEBUG", "false")

    # Yield the settings instance
    settings = get_settings()
    yield settings

    # Clear cache after test
    get_settings.cache_clear()


# ============================================================================
# Phase 3 - LLM Provider Fixtures
# ============================================================================


@pytest.fixture
def mock_llm_provider():
    """Provide mock LLM provider for testing without API calls.

    Returns a MagicMock that behaves like a BaseLLMProvider.
    """
    from unittest.mock import MagicMock
    from atlasrag.src.llm import LLMResponse, EmbeddingResponse

    mock = MagicMock()
    mock.generate.return_value = LLMResponse(
        content="Mock response",
        tokens_used=10,
        model="mock-model",
        provider="mock",
    )
    mock.embed.return_value = EmbeddingResponse(
        embedding=[0.1, 0.2, 0.3],
        dimensions=3,
        model="mock-embedding-model",
        provider="mock",
    )
    mock.is_available.return_value = True
    return mock


@pytest.fixture
def llm_client(monkeypatch, mock_llm_provider):
    """Provide LLM client with mocked provider.

    Clears the factory cache and patches get_llm_client to return mock.
    Useful for testing code that uses LLM without making actual API calls.
    """
    from atlasrag.src.llm.factory import get_llm_client

    get_llm_client.cache_clear()
    monkeypatch.setattr(
        "atlasrag.src.llm.factory.get_llm_client",
        lambda: mock_llm_provider,
    )
    yield mock_llm_provider
    get_llm_client.cache_clear()


# ============================================================================
# Phase 4 - Vector Store Fixtures
# ============================================================================


@pytest.fixture
def mock_vector_store():
    """Provide mock vector store for testing without backend.

    Returns a MagicMock that behaves like a BaseVectorStore.
    """
    from unittest.mock import MagicMock
    from atlasrag.src.retrieval import SearchResults, StorageResponse, DeleteResponse

    mock = MagicMock()
    mock.add_documents.return_value = StorageResponse(
        document_count=2,
        vector_store="mock",
        status="success",
    )
    mock.similarity_search.return_value = SearchResults(
        query=None,
        results=[],
        total_results=0,
        vector_store="mock",
    )
    mock.delete_documents.return_value = DeleteResponse(
        deleted_count=1,
        vector_store="mock",
        status="success",
    )
    mock.is_available.return_value = True
    mock.get_document_count.return_value = 2
    return mock


@pytest.fixture
def vector_store(monkeypatch, mock_vector_store):
    """Provide vector store with mocked backend.

    Clears the factory cache and patches get_vector_store to return mock.
    Useful for testing code that uses vector store without backend.
    """
    from atlasrag.src.retrieval.factory import get_vector_store

    get_vector_store.cache_clear()
    monkeypatch.setattr(
        "atlasrag.src.retrieval.factory.get_vector_store",
        lambda: mock_vector_store,
    )
    yield mock_vector_store
    get_vector_store.cache_clear()


# ============================================================================
# TODO: Phase 5 - Document Ingestion Fixtures
# ============================================================================
# These fixtures will be implemented when Phase 5 (Ingestion) is done.
#
# @pytest.fixture
# def sample_pdf(tmp_path):
#     """Create a sample PDF for testing"""
#     # Create PDF
#     pdf_path = tmp_path / "sample.pdf"
#     # Write PDF content
#     return pdf_path
#
# @pytest.fixture
# def sample_docx(tmp_path):
#     """Create a sample DOCX for testing"""
#     docx_path = tmp_path / "sample.docx"
#     # Write DOCX content
#     return docx_path


# ============================================================================
# TODO: Phase 9 - FastAPI Test Client Fixture
# ============================================================================
# These fixtures will be implemented when Phase 9 (API) is done.
#
# @pytest.fixture
# def test_client():
#     """Provide FastAPI test client"""
#     from fastapi.testclient import TestClient
#     from src.api.main import app
#     return TestClient(app)
#
# @pytest.fixture
# def authenticated_client(test_client):
#     """Test client with authentication headers"""
#     # Add API key header
#     pass


# ============================================================================
# TODO: Phase 11 - Evaluation Fixtures
# ============================================================================
# These fixtures will be implemented when Phase 11 (Evaluation) is done.
#
# @pytest.fixture
# def sample_golden_dataset():
#     """Provide sample evaluation dataset"""
#     return [
#         {
#             "query": "What's our vacation policy?",
#             "reference_answer": "20 days annually",
#             "source_documents": [...],
#             "expected_citations": ["doc1"]
#         }
#     ]


# ============================================================================
# Pytest Hooks (for all tests)
# ============================================================================

def pytest_configure(config):
    """Configure pytest before tests run"""
    # Add custom markers
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    for item in items:
        # Add markers based on file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)


# ============================================================================
# Shared Utilities
# ============================================================================

@pytest.fixture
def test_data_dir():
    """Provide path to test data directory"""
    return Path(__file__).parent / "test_data"


@pytest.fixture(autouse=True)
def cleanup_temp_files(tmp_path):
    """Auto-cleanup temporary files after each test"""
    yield
    # Cleanup happens automatically with tmp_path


# ============================================================================
# Session-Scoped Fixtures (run once per test session)
# ============================================================================

@pytest.fixture(scope="session")
def session_id():
    """Provide unique session ID for test run"""
    import uuid
    return str(uuid.uuid4())
