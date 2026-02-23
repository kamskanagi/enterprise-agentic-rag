"""Unit tests for the Basic RAG Pipeline (Phase 6)."""

import pytest
from unittest.mock import MagicMock

from atlasrag.src.llm.models import EmbeddingResponse, LLMResponse
from atlasrag.src.rag.exceptions import (
    ContextBuildError,
    NoDocumentsFoundError,
    RAGException,
)
from atlasrag.src.rag.models import RAGConfig, RAGResponse, SourceReference
from atlasrag.src.rag.pipeline import BasicRAGPipeline
from atlasrag.src.rag.prompts import DEFAULT_SYSTEM_PROMPT, build_rag_prompt
from atlasrag.src.retrieval.models import SearchResult, SearchResults


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_sources():
    """Provide sample source references for testing."""
    return [
        SourceReference(
            content="The company was founded in 2020.",
            source="company_history.pdf",
            page=1,
            chunk_index=0,
            similarity_score=0.95,
        ),
        SourceReference(
            content="Annual revenue reached $10M in 2023.",
            source="financials.pdf",
            page=5,
            chunk_index=3,
            similarity_score=0.88,
        ),
    ]


@pytest.fixture
def search_results_with_hits():
    """Provide SearchResults with actual results."""
    return SearchResults(
        query=None,
        results=[
            SearchResult(
                document_id="doc-1",
                content="The company was founded in 2020.",
                similarity_score=0.95,
                metadata={
                    "source": "company_history.pdf",
                    "page": 1,
                    "chunk_index": 0,
                },
            ),
            SearchResult(
                document_id="doc-2",
                content="Annual revenue reached $10M in 2023.",
                similarity_score=0.88,
                metadata={
                    "source": "financials.pdf",
                    "page": 5,
                    "chunk_index": 3,
                },
            ),
        ],
        total_results=2,
        vector_store="mock",
    )


@pytest.fixture
def empty_search_results():
    """Provide SearchResults with no results."""
    return SearchResults(
        query=None,
        results=[],
        total_results=0,
        vector_store="mock",
    )


@pytest.fixture
def rag_pipeline(mock_llm_provider, mock_vector_store):
    """Provide a BasicRAGPipeline with mocked dependencies."""
    return BasicRAGPipeline(
        llm=mock_llm_provider,
        vector_store=mock_vector_store,
    )


# ============================================================================
# Model Tests
# ============================================================================


class TestSourceReference:
    """Tests for SourceReference model."""

    def test_create_with_all_fields(self):
        """SourceReference accepts all fields."""
        ref = SourceReference(
            content="Some text",
            source="doc.pdf",
            page=3,
            chunk_index=1,
            similarity_score=0.92,
        )
        assert ref.content == "Some text"
        assert ref.source == "doc.pdf"
        assert ref.page == 3
        assert ref.chunk_index == 1
        assert ref.similarity_score == 0.92

    def test_create_with_optional_fields_none(self):
        """SourceReference works with optional fields as None."""
        ref = SourceReference(
            content="text",
            source="doc.txt",
            similarity_score=0.8,
        )
        assert ref.page is None
        assert ref.chunk_index is None

    def test_is_frozen(self):
        """SourceReference is immutable."""
        ref = SourceReference(
            content="text", source="doc.txt", similarity_score=0.8
        )
        with pytest.raises(Exception):
            ref.content = "new text"


class TestRAGResponse:
    """Tests for RAGResponse model."""

    def test_create_response(self, sample_sources):
        """RAGResponse accepts all required fields."""
        response = RAGResponse(
            answer="The company was founded in 2020.",
            sources=sample_sources,
            query="When was the company founded?",
            model="llama2",
            provider="ollama",
            retrieval_count=2,
        )
        assert response.answer == "The company was founded in 2020."
        assert len(response.sources) == 2
        assert response.query == "When was the company founded?"
        assert response.model == "llama2"
        assert response.provider == "ollama"
        assert response.retrieval_count == 2

    def test_is_frozen(self, sample_sources):
        """RAGResponse is immutable."""
        response = RAGResponse(
            answer="answer",
            sources=sample_sources,
            query="query",
            model="m",
            provider="p",
            retrieval_count=2,
        )
        with pytest.raises(Exception):
            response.answer = "new answer"


class TestRAGConfig:
    """Tests for RAGConfig model."""

    def test_defaults(self):
        """RAGConfig has sensible defaults."""
        config = RAGConfig()
        assert config.top_k == 5
        assert config.similarity_threshold == 0.7
        assert config.max_tokens == 2000
        assert config.temperature == 0.3
        assert config.system_prompt is None

    def test_custom_values(self):
        """RAGConfig accepts custom values."""
        config = RAGConfig(
            top_k=10,
            similarity_threshold=0.5,
            max_tokens=1000,
            temperature=0.1,
            system_prompt="Custom prompt",
        )
        assert config.top_k == 10
        assert config.similarity_threshold == 0.5
        assert config.max_tokens == 1000
        assert config.temperature == 0.1
        assert config.system_prompt == "Custom prompt"


# ============================================================================
# Prompt Tests
# ============================================================================


class TestBuildRagPrompt:
    """Tests for build_rag_prompt function."""

    def test_builds_prompt_with_numbered_sources(self, sample_sources):
        """build_rag_prompt includes numbered source references."""
        prompt = build_rag_prompt("What year was the company founded?", sample_sources)
        assert "[1]" in prompt
        assert "[2]" in prompt
        assert "company_history.pdf" in prompt
        assert "financials.pdf" in prompt

    def test_includes_query(self, sample_sources):
        """build_rag_prompt includes the original question."""
        query = "What year was the company founded?"
        prompt = build_rag_prompt(query, sample_sources)
        assert query in prompt

    def test_includes_page_numbers(self, sample_sources):
        """build_rag_prompt shows page numbers when available."""
        prompt = build_rag_prompt("question", sample_sources)
        assert "page 1" in prompt
        assert "page 5" in prompt

    def test_handles_no_page_number(self):
        """build_rag_prompt works when page is None."""
        sources = [
            SourceReference(
                content="text",
                source="doc.txt",
                similarity_score=0.9,
            )
        ]
        prompt = build_rag_prompt("question", sources)
        assert "doc.txt" in prompt
        assert "page" not in prompt

    def test_includes_chunk_content(self, sample_sources):
        """build_rag_prompt includes the actual chunk text."""
        prompt = build_rag_prompt("question", sample_sources)
        assert "The company was founded in 2020." in prompt
        assert "Annual revenue reached $10M in 2023." in prompt

    def test_default_system_prompt_exists(self):
        """DEFAULT_SYSTEM_PROMPT is a non-empty string."""
        assert isinstance(DEFAULT_SYSTEM_PROMPT, str)
        assert len(DEFAULT_SYSTEM_PROMPT) > 0
        assert "context" in DEFAULT_SYSTEM_PROMPT.lower()


# ============================================================================
# Exception Tests
# ============================================================================


class TestExceptions:
    """Tests for RAG exception hierarchy."""

    def test_no_documents_found_is_rag_exception(self):
        """NoDocumentsFoundError inherits from RAGException."""
        assert issubclass(NoDocumentsFoundError, RAGException)

    def test_context_build_error_is_rag_exception(self):
        """ContextBuildError inherits from RAGException."""
        assert issubclass(ContextBuildError, RAGException)

    def test_rag_exception_is_exception(self):
        """RAGException inherits from Exception."""
        assert issubclass(RAGException, Exception)


# ============================================================================
# Pipeline Tests
# ============================================================================


class TestBasicRAGPipeline:
    """Tests for BasicRAGPipeline."""

    def test_query_returns_rag_response(
        self, rag_pipeline, mock_llm_provider, mock_vector_store, search_results_with_hits
    ):
        """Pipeline.query returns a RAGResponse with answer and sources."""
        mock_vector_store.similarity_search.return_value = search_results_with_hits
        mock_llm_provider.generate.return_value = LLMResponse(
            content="The company was founded in 2020 [1].",
            tokens_used=20,
            model="llama2",
            provider="ollama",
        )

        response = rag_pipeline.query("When was the company founded?")

        assert isinstance(response, RAGResponse)
        assert response.answer == "The company was founded in 2020 [1]."
        assert response.query == "When was the company founded?"
        assert response.model == "llama2"
        assert response.provider == "ollama"
        assert response.retrieval_count == 2
        assert len(response.sources) == 2

    def test_query_calls_embed(
        self, rag_pipeline, mock_llm_provider, mock_vector_store, search_results_with_hits
    ):
        """Pipeline.query embeds the question via LLM."""
        mock_vector_store.similarity_search.return_value = search_results_with_hits

        rag_pipeline.query("test question")

        mock_llm_provider.embed.assert_called_once_with("test question")

    def test_query_calls_similarity_search(
        self, rag_pipeline, mock_llm_provider, mock_vector_store, search_results_with_hits
    ):
        """Pipeline.query searches vector store with embedding."""
        mock_vector_store.similarity_search.return_value = search_results_with_hits

        rag_pipeline.query("test question")

        mock_vector_store.similarity_search.assert_called_once()
        call_kwargs = mock_vector_store.similarity_search.call_args
        assert call_kwargs.kwargs["query_embedding"] == [0.1, 0.2, 0.3]
        assert call_kwargs.kwargs["top_k"] == 5

    def test_query_calls_generate_with_context(
        self, rag_pipeline, mock_llm_provider, mock_vector_store, search_results_with_hits
    ):
        """Pipeline.query passes retrieved context to LLM generate."""
        mock_vector_store.similarity_search.return_value = search_results_with_hits

        rag_pipeline.query("test question")

        mock_llm_provider.generate.assert_called_once()
        call_kwargs = mock_llm_provider.generate.call_args
        prompt = call_kwargs.kwargs["prompt"]
        assert "company_history.pdf" in prompt
        assert "test question" in prompt

    def test_query_no_results_raises(
        self, rag_pipeline, mock_vector_store, empty_search_results
    ):
        """Pipeline.query raises NoDocumentsFoundError when no results."""
        mock_vector_store.similarity_search.return_value = empty_search_results

        with pytest.raises(NoDocumentsFoundError):
            rag_pipeline.query("question with no matching docs")

    def test_query_uses_config_temperature(
        self, mock_llm_provider, mock_vector_store, search_results_with_hits
    ):
        """Pipeline.query uses configured temperature."""
        mock_vector_store.similarity_search.return_value = search_results_with_hits
        config = RAGConfig(temperature=0.1)
        pipeline = BasicRAGPipeline(
            llm=mock_llm_provider, vector_store=mock_vector_store, config=config
        )

        pipeline.query("test")

        call_kwargs = mock_llm_provider.generate.call_args
        assert call_kwargs.kwargs["temperature"] == 0.1

    def test_query_uses_custom_system_prompt(
        self, mock_llm_provider, mock_vector_store, search_results_with_hits
    ):
        """Pipeline.query uses custom system prompt when configured."""
        mock_vector_store.similarity_search.return_value = search_results_with_hits
        config = RAGConfig(system_prompt="You are a legal expert.")
        pipeline = BasicRAGPipeline(
            llm=mock_llm_provider, vector_store=mock_vector_store, config=config
        )

        pipeline.query("test")

        call_kwargs = mock_llm_provider.generate.call_args
        assert "You are a legal expert." in call_kwargs.kwargs["prompt"]

    def test_sources_map_metadata_correctly(
        self, rag_pipeline, mock_llm_provider, mock_vector_store, search_results_with_hits
    ):
        """Pipeline.query maps search result metadata to SourceReference."""
        mock_vector_store.similarity_search.return_value = search_results_with_hits

        response = rag_pipeline.query("test")

        src = response.sources[0]
        assert src.source == "company_history.pdf"
        assert src.page == 1
        assert src.chunk_index == 0
        assert src.similarity_score == 0.95

    def test_is_ready_true(self, rag_pipeline):
        """is_ready returns True when LLM and vector store are available."""
        assert rag_pipeline.is_ready() is True

    def test_is_ready_false_when_llm_unavailable(
        self, mock_llm_provider, mock_vector_store
    ):
        """is_ready returns False when LLM is unavailable."""
        mock_llm_provider.is_available.return_value = False
        pipeline = BasicRAGPipeline(
            llm=mock_llm_provider, vector_store=mock_vector_store
        )
        assert pipeline.is_ready() is False

    def test_is_ready_false_when_vector_store_unavailable(
        self, mock_llm_provider, mock_vector_store
    ):
        """is_ready returns False when vector store is unavailable."""
        mock_vector_store.is_available.return_value = False
        pipeline = BasicRAGPipeline(
            llm=mock_llm_provider, vector_store=mock_vector_store
        )
        assert pipeline.is_ready() is False

    def test_is_ready_false_on_exception(
        self, mock_llm_provider, mock_vector_store
    ):
        """is_ready returns False when a check raises an exception."""
        mock_llm_provider.is_available.side_effect = Exception("connection error")
        pipeline = BasicRAGPipeline(
            llm=mock_llm_provider, vector_store=mock_vector_store
        )
        assert pipeline.is_ready() is False
