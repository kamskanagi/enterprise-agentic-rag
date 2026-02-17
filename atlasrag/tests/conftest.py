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
# TODO: Phase 2 - Configuration Fixtures
# ============================================================================
# These fixtures will be implemented when Phase 2 (Configuration) is done.
#
# @pytest.fixture
# def test_settings():
#     """Provide test-specific settings instead of .env values"""
#     from src.config.settings import Settings
#     return Settings(
#         llm_provider="ollama",
#         vector_store="chroma",
#         database_url="postgresql://test:test@localhost/test"
#     )
#
# @pytest.fixture(scope="session")
# def test_database(test_settings):
#     """Create test database and teardown after all tests"""
#     # Create database
#     # Yield for tests
#     # Drop database
#     pass


# ============================================================================
# TODO: Phase 3 - LLM Provider Fixtures
# ============================================================================
# These fixtures will be implemented when Phase 3 (LLM) is done.
#
# @pytest.fixture
# def mock_llm_provider():
#     """Provide mock LLM provider for testing without API calls"""
#     from unittest.mock import MagicMock
#     mock = MagicMock()
#     mock.generate.return_value = "Test response"
#     mock.embed.return_value = [0.1, 0.2, 0.3]
#     return mock
#
# @pytest.fixture
# def llm_client(mock_llm_provider):
#     """Provide LLM client with mocked provider"""
#     from src.llm.factory import get_llm_client
#     # Patch the factory to return mock
#     pass


# ============================================================================
# TODO: Phase 4 - Vector Store Fixtures
# ============================================================================
# These fixtures will be implemented when Phase 4 (Retrieval) is done.
#
# @pytest.fixture
# def test_vector_store(tmp_path):
#     """Provide in-memory vector store for testing"""
#     from src.retrieval.chroma_store import ChromaVectorStore
#     # Use temp directory for Chroma
#     return ChromaVectorStore(persist_directory=str(tmp_path))
#
# @pytest.fixture
# def vector_store_with_data(test_vector_store):
#     """Vector store pre-populated with test data"""
#     test_vector_store.add(
#         texts=["Test document 1", "Test document 2"],
#         embeddings=[[0.1, 0.2], [0.15, 0.25]],
#         metadata=[{"source": "test1"}, {"source": "test2"}]
#     )
#     return test_vector_store


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
