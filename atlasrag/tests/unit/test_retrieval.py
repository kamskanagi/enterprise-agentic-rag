"""Unit Tests for Vector Store Abstraction Layer (Phase 4)

Tests cover:
- Abstract base class contract
- Vector store implementations (Chroma, Milvus)
- Factory selection logic
- Custom exception hierarchy
- Response model validation
- Backend availability checks
"""

import pytest
from unittest.mock import MagicMock, patch

from atlasrag.src.retrieval import (
    get_vector_store,
    BaseVectorStore,
    ChromaVectorStore,
    MilvusVectorStore,
    SearchResult,
    SearchResults,
    StorageResponse,
    DeleteResponse,
    DocumentMetadata,
    VectorStoreException,
    VectorStoreUnavailableError,
    SearchError,
    StorageError,
    DeletionError,
    QueryError,
)
from atlasrag.src.config.vector_store_config import ChromaConfig, MilvusConfig
from pydantic import SecretStr


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def chroma_config():
    """Chroma configuration for testing."""
    return ChromaConfig(
        persist_directory="./test_chroma_data",
        collection_name="test_collection",
    )


@pytest.fixture
def milvus_config():
    """Milvus configuration for testing."""
    return MilvusConfig(
        host="localhost",
        port=19530,
        user="test_user",
        password=SecretStr("test_password"),
        collection_name="test_collection",
    )


# ============================================================================
# TestBaseVectorStore: Abstract Interface Contract
# ============================================================================


class TestBaseVectorStore:
    """Test the abstract base class interface."""

    def test_cannot_instantiate_abstract_class(self):
        """Cannot directly instantiate BaseVectorStore."""
        with pytest.raises(TypeError):
            BaseVectorStore()

    def test_abstract_methods_required(self):
        """Subclasses must implement all abstract methods."""

        class IncompleteStore(BaseVectorStore):
            pass

        with pytest.raises(TypeError):
            IncompleteStore()

    def test_all_methods_documented(self):
        """Abstract methods have docstrings."""
        assert BaseVectorStore.add_documents.__doc__ is not None
        assert BaseVectorStore.similarity_search.__doc__ is not None
        assert BaseVectorStore.is_available.__doc__ is not None


# ============================================================================
# TestResponseModels: Pydantic Response Models
# ============================================================================


class TestResponseModels:
    """Test response models for vector store operations."""

    def test_search_result_creation(self):
        """SearchResult can be created with valid fields."""
        result = SearchResult(
            document_id="doc_1",
            content="test content",
            similarity_score=0.95,
            metadata={"source": "test.pdf"},
        )
        assert result.document_id == "doc_1"
        assert result.content == "test content"
        assert result.similarity_score == 0.95

    def test_search_result_is_frozen(self):
        """SearchResult is immutable (frozen)."""
        result = SearchResult(
            document_id="doc_1",
            content="test",
            similarity_score=0.9,
            metadata={},
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            result.content = "modified"

    def test_search_results_creation(self):
        """SearchResults can be created with multiple results."""
        results = SearchResults(
            query="test query",
            results=[],
            total_results=0,
            vector_store="chroma",
        )
        assert results.vector_store == "chroma"
        assert results.total_results == 0

    def test_storage_response_creation(self):
        """StorageResponse can be created."""
        response = StorageResponse(
            document_count=10,
            vector_store="chroma",
            status="success",
        )
        assert response.document_count == 10
        assert response.status == "success"

    def test_delete_response_creation(self):
        """DeleteResponse can be created."""
        response = DeleteResponse(
            deleted_count=5,
            vector_store="milvus",
            status="success",
        )
        assert response.deleted_count == 5
        assert response.vector_store == "milvus"

    def test_document_metadata_creation(self):
        """DocumentMetadata can be created."""
        metadata = DocumentMetadata(
            source="document.pdf",
            page=1,
            chunk_index=0,
            custom_fields={"author": "test"},
        )
        assert metadata.source == "document.pdf"
        assert metadata.page == 1


# ============================================================================
# TestVectorStoreExceptions: Custom Exception Hierarchy
# ============================================================================


class TestVectorStoreExceptions:
    """Test custom exception classes."""

    def test_vector_store_exception_inheritance(self):
        """VectorStoreException is base exception."""
        exc = VectorStoreException("test error")
        assert isinstance(exc, Exception)

    def test_all_exceptions_inherit_from_base(self):
        """All exceptions inherit from VectorStoreException."""
        exceptions = [
            VectorStoreUnavailableError("test"),
            SearchError("test"),
            StorageError("test"),
            DeletionError("test"),
            QueryError("test"),
        ]
        for exc in exceptions:
            assert isinstance(exc, VectorStoreException)

    def test_catch_all_exceptions(self):
        """Can catch all vector store exceptions with VectorStoreException."""
        exceptions = [
            SearchError("search failed"),
            StorageError("storage failed"),
        ]
        for exc in exceptions:
            try:
                raise exc
            except VectorStoreException:
                pass  # Successfully caught


# ============================================================================
# TestChromaVectorStore: Chroma Implementation
# ============================================================================


class TestChromaVectorStore:
    """Test Chroma vector store implementation."""

    def test_chroma_store_creation(self, chroma_config):
        """ChromaVectorStore can be instantiated."""
        with patch("atlasrag.src.retrieval.chroma_store.chromadb.Client"):
            store = ChromaVectorStore(chroma_config)
            assert store.collection_name == "test_collection"

    def test_chroma_store_is_base_vector_store(self, chroma_config):
        """ChromaVectorStore is a BaseVectorStore."""
        with patch("atlasrag.src.retrieval.chroma_store.chromadb.Client"):
            store = ChromaVectorStore(chroma_config)
            assert isinstance(store, BaseVectorStore)

    def test_chroma_add_documents_validation(self, chroma_config):
        """ChromaVectorStore validates documents and embeddings."""
        with patch("atlasrag.src.retrieval.chroma_store.chromadb.Client"):
            store = ChromaVectorStore(chroma_config)

            # Test empty documents
            with pytest.raises(StorageError):
                store.add_documents([], [])

            # Test mismatched lengths
            with pytest.raises(StorageError):
                store.add_documents(["doc1"], [[0.1, 0.2], [0.3, 0.4]])

    def test_chroma_similarity_search_validation(self, chroma_config):
        """ChromaVectorStore validates search parameters."""
        with patch("atlasrag.src.retrieval.chroma_store.chromadb.Client"):
            store = ChromaVectorStore(chroma_config)

            # Test invalid top_k
            with pytest.raises(SearchError):
                store.similarity_search([0.1, 0.2], top_k=0)

            # Test invalid threshold
            with pytest.raises(SearchError):
                store.similarity_search([0.1, 0.2], similarity_threshold=1.5)

    def test_chroma_is_available(self, chroma_config):
        """ChromaVectorStore availability check."""
        with patch("atlasrag.src.retrieval.chroma_store.chromadb.Client"):
            store = ChromaVectorStore(chroma_config)
            store.client.list_collections = MagicMock()
            assert store.is_available() is True

    def test_chroma_get_document_count(self, chroma_config):
        """ChromaVectorStore returns document count."""
        with patch("atlasrag.src.retrieval.chroma_store.chromadb.Client"):
            store = ChromaVectorStore(chroma_config)
            store.collection.count = MagicMock(return_value=42)
            assert store.get_document_count() == 42


# ============================================================================
# TestMilvusVectorStore: Milvus Implementation
# ============================================================================


class TestMilvusVectorStore:
    """Test Milvus vector store implementation."""

    @patch("atlasrag.src.retrieval.milvus_store.connections")
    def test_milvus_store_creation(self, mock_connections, milvus_config):
        """MilvusVectorStore can be instantiated."""
        mock_connections.list_collections.return_value = []

        with patch("atlasrag.src.retrieval.milvus_store.Collection"):
            store = MilvusVectorStore(milvus_config)
            assert store.collection_name == "test_collection"

    @patch("atlasrag.src.retrieval.milvus_store.connections")
    def test_milvus_store_is_base_vector_store(
        self, mock_connections, milvus_config
    ):
        """MilvusVectorStore is a BaseVectorStore."""
        mock_connections.list_collections.return_value = []

        with patch("atlasrag.src.retrieval.milvus_store.Collection"):
            store = MilvusVectorStore(milvus_config)
            assert isinstance(store, BaseVectorStore)

    @patch("atlasrag.src.retrieval.milvus_store.connections")
    def test_milvus_add_documents_validation(
        self, mock_connections, milvus_config
    ):
        """MilvusVectorStore validates documents and embeddings."""
        mock_connections.list_collections.return_value = []

        with patch("atlasrag.src.retrieval.milvus_store.Collection"):
            store = MilvusVectorStore(milvus_config)

            # Test empty documents
            with pytest.raises(StorageError):
                store.add_documents([], [])

            # Test mismatched lengths
            with pytest.raises(StorageError):
                store.add_documents(
                    ["doc1"], [[0.1, 0.2], [0.3, 0.4]]
                )

    @patch("atlasrag.src.retrieval.milvus_store.connections")
    def test_milvus_similarity_search_validation(
        self, mock_connections, milvus_config
    ):
        """MilvusVectorStore validates search parameters."""
        mock_connections.list_collections.return_value = []

        with patch("atlasrag.src.retrieval.milvus_store.Collection"):
            store = MilvusVectorStore(milvus_config)

            # Test invalid top_k
            with pytest.raises(SearchError):
                store.similarity_search([0.1, 0.2], top_k=0)

            # Test invalid threshold
            with pytest.raises(SearchError):
                store.similarity_search([0.1, 0.2], similarity_threshold=1.5)

    @patch("atlasrag.src.retrieval.milvus_store.connections")
    def test_milvus_is_available(self, mock_connections, milvus_config):
        """MilvusVectorStore availability check."""
        mock_connections.list_collections.return_value = []
        mock_connections.get_connection_addr.return_value = "localhost:19530"

        with patch("atlasrag.src.retrieval.milvus_store.Collection"):
            store = MilvusVectorStore(milvus_config)
            assert store.is_available() is True

    @patch("atlasrag.src.retrieval.milvus_store.connections")
    def test_milvus_get_document_count(self, mock_connections, milvus_config):
        """MilvusVectorStore returns document count."""
        mock_connections.list_collections.return_value = []

        with patch("atlasrag.src.retrieval.milvus_store.Collection") as mock_col:
            store = MilvusVectorStore(milvus_config)
            store.collection.num_entities = 100
            assert store.get_document_count() == 100


# ============================================================================
# TestVectorStoreFactory: Provider Selection Factory
# ============================================================================


class TestVectorStoreFactory:
    """Test vector store factory and singleton pattern."""

    def test_get_vector_store_returns_base_store(self):
        """get_vector_store returns BaseVectorStore instance."""
        get_vector_store.cache_clear()
        with patch("atlasrag.src.retrieval.chroma_store.chromadb.Client"):
            store = get_vector_store()
            assert isinstance(store, BaseVectorStore)
            get_vector_store.cache_clear()

    def test_factory_chroma_default(self, test_settings, monkeypatch):
        """Factory returns ChromaVectorStore by default."""
        get_vector_store.cache_clear()
        monkeypatch.setenv("VECTOR_STORE", "chroma")
        from atlasrag.src.config import get_settings

        get_settings.cache_clear()
        with patch("atlasrag.src.retrieval.chroma_store.chromadb.Client"):
            store = get_vector_store()
            assert isinstance(store, ChromaVectorStore)
            get_vector_store.cache_clear()
            get_settings.cache_clear()

    def test_factory_milvus(self, monkeypatch):
        """Factory returns MilvusVectorStore when configured."""
        get_vector_store.cache_clear()
        monkeypatch.setenv("VECTOR_STORE", "milvus")
        from atlasrag.src.config import get_settings

        get_settings.cache_clear()
        with patch("atlasrag.src.retrieval.milvus_store.connections"):
            with patch("atlasrag.src.retrieval.milvus_store.Collection"):
                store = get_vector_store()
                assert isinstance(store, MilvusVectorStore)
                get_vector_store.cache_clear()
                get_settings.cache_clear()

    def test_factory_singleton_lru_cache(self):
        """Factory uses @lru_cache singleton pattern."""
        get_vector_store.cache_clear()
        with patch("atlasrag.src.retrieval.chroma_store.chromadb.Client"):
            store1 = get_vector_store()
            store2 = get_vector_store()
            assert store1 is store2
            get_vector_store.cache_clear()

    def test_factory_cache_clear_resets_singleton(self, monkeypatch):
        """cache_clear() resets the singleton."""
        get_vector_store.cache_clear()
        monkeypatch.setenv("VECTOR_STORE", "chroma")
        from atlasrag.src.config import get_settings

        get_settings.cache_clear()
        with patch("atlasrag.src.retrieval.chroma_store.chromadb.Client"):
            store1 = get_vector_store()

            get_vector_store.cache_clear()
            get_settings.cache_clear()
            monkeypatch.setenv("VECTOR_STORE", "chroma")
            store2 = get_vector_store()

            assert store1 is not store2
            get_vector_store.cache_clear()
            get_settings.cache_clear()


# ============================================================================
# TestVectorStoreBackendAvailability: Health Checks
# ============================================================================


class TestVectorStoreBackendAvailability:
    """Test vector store backend availability checks."""

    def test_chroma_availability_uses_list_collections(self, chroma_config):
        """Chroma availability check calls list_collections."""
        with patch("atlasrag.src.retrieval.chroma_store.chromadb.Client"):
            store = ChromaVectorStore(chroma_config)
            store.client.list_collections = MagicMock()

            store.is_available()

            store.client.list_collections.assert_called_once()

    def test_chroma_availability_timeout(self, chroma_config):
        """Chroma availability check handles exception."""
        with patch("atlasrag.src.retrieval.chroma_store.chromadb.Client"):
            store = ChromaVectorStore(chroma_config)
            store.client.list_collections = MagicMock(
                side_effect=Exception("timeout")
            )

            assert store.is_available() is False

    @patch("atlasrag.src.retrieval.milvus_store.connections")
    def test_milvus_availability_check(self, mock_connections, milvus_config):
        """Milvus availability check verifies connection."""
        mock_connections.list_collections.return_value = []
        mock_connections.get_connection_addr.return_value = "localhost:19530"

        with patch("atlasrag.src.retrieval.milvus_store.Collection"):
            store = MilvusVectorStore(milvus_config)
            assert store.is_available() is True

    def test_all_stores_have_availability_check(
        self, chroma_config, milvus_config
    ):
        """All stores implement is_available()."""
        with patch("atlasrag.src.retrieval.chroma_store.chromadb.Client"):
            chroma = ChromaVectorStore(chroma_config)
            assert hasattr(chroma, "is_available")
            assert callable(chroma.is_available)

        with patch("atlasrag.src.retrieval.milvus_store.connections"):
            with patch("atlasrag.src.retrieval.milvus_store.Collection"):
                milvus = MilvusVectorStore(milvus_config)
                assert hasattr(milvus, "is_available")
                assert callable(milvus.is_available)
