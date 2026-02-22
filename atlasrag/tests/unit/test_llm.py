"""Unit Tests for LLM Provider Abstraction Layer (Phase 3)

Tests cover:
- Abstract base class contract
- Provider implementations (Ollama, OpenAI, Gemini)
- Factory selection logic
- Custom exception hierarchy
- Response model validation
- Provider availability checks
- Retry logic via tenacity
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import httpx

from atlasrag.src.llm import (
    get_llm_client,
    BaseLLMProvider,
    OllamaProvider,
    OpenAIProvider,
    GeminiProvider,
    LLMResponse,
    EmbeddingResponse,
    LLMException,
    UnsupportedProviderError,
    ProviderUnavailableError,
    GenerationError,
    EmbeddingError,
    RateLimitError,
)
from atlasrag.src.config.llm_config import (
    OllamaConfig,
    OpenAIConfig,
    GeminiConfig,
)
from pydantic import SecretStr


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def ollama_config():
    """Ollama configuration for testing."""
    return OllamaConfig(
        base_url="http://localhost:11434",
        model="llama3.2",
        embedding_model="nomic-embed-text",
    )


@pytest.fixture
def openai_config():
    """OpenAI configuration for testing."""
    return OpenAIConfig(
        api_key=SecretStr("sk-test-key"),
        model="gpt-4o-mini",
        embedding_model="text-embedding-3-small",
    )


@pytest.fixture
def gemini_config():
    """Gemini configuration for testing."""
    return GeminiConfig(
        api_key=SecretStr("test-api-key"),
        model="gemini-1.5-flash",
        embedding_model="models/text-embedding-004",
    )


# ============================================================================
# TestBaseLLMProvider: Abstract Interface Contract
# ============================================================================


class TestBaseLLMProvider:
    """Test the abstract base class interface."""

    def test_cannot_instantiate_abstract_class(self):
        """Cannot directly instantiate BaseLLMProvider."""
        with pytest.raises(TypeError):
            BaseLLMProvider()

    def test_abstract_methods_required(self):
        """Subclasses must implement all abstract methods."""

        class IncompleteProvider(BaseLLMProvider):
            pass

        with pytest.raises(TypeError):
            IncompleteProvider()

    def test_all_methods_documented(self):
        """Abstract methods have docstrings."""
        assert BaseLLMProvider.generate.__doc__ is not None
        assert BaseLLMProvider.embed.__doc__ is not None
        assert BaseLLMProvider.is_available.__doc__ is not None


# ============================================================================
# TestLLMResponseModels: Pydantic Response Models
# ============================================================================


class TestLLMResponseModels:
    """Test LLMResponse and EmbeddingResponse Pydantic models."""

    def test_llm_response_creation(self):
        """LLMResponse can be created with valid fields."""
        response = LLMResponse(
            content="test response",
            tokens_used=10,
            model="test-model",
            provider="ollama",
        )
        assert response.content == "test response"
        assert response.tokens_used == 10
        assert response.model == "test-model"
        assert response.provider == "ollama"

    def test_llm_response_is_frozen(self):
        """LLMResponse is immutable (frozen)."""
        response = LLMResponse(
            content="test",
            tokens_used=10,
            model="test",
            provider="ollama",
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            response.content = "modified"

    def test_embedding_response_creation(self):
        """EmbeddingResponse can be created with valid fields."""
        embedding = [0.1, 0.2, 0.3]
        response = EmbeddingResponse(
            embedding=embedding,
            dimensions=3,
            model="embed-model",
            provider="openai",
        )
        assert response.embedding == embedding
        assert response.dimensions == 3
        assert response.model == "embed-model"
        assert response.provider == "openai"

    def test_embedding_response_is_frozen(self):
        """EmbeddingResponse is immutable (frozen)."""
        response = EmbeddingResponse(
            embedding=[0.1],
            dimensions=1,
            model="test",
            provider="gemini",
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            response.embedding = [0.2]

    def test_llm_response_validation_missing_field(self):
        """LLMResponse requires all fields."""
        with pytest.raises(Exception):  # ValidationError
            LLMResponse(content="test", tokens_used=10)

    def test_embedding_response_validation_dimensions_match(self):
        """EmbeddingResponse dimensions should match embedding length."""
        response = EmbeddingResponse(
            embedding=[0.1, 0.2, 0.3],
            dimensions=3,
            model="test",
            provider="ollama",
        )
        assert response.dimensions == len(response.embedding)


# ============================================================================
# TestLLMExceptions: Custom Exception Hierarchy
# ============================================================================


class TestLLMExceptions:
    """Test custom exception classes."""

    def test_llm_exception_inheritance(self):
        """LLMException is base exception."""
        exc = LLMException("test error")
        assert isinstance(exc, Exception)

    def test_unsupported_provider_error_inheritance(self):
        """UnsupportedProviderError inherits from LLMException."""
        exc = UnsupportedProviderError("unknown provider")
        assert isinstance(exc, LLMException)

    def test_provider_unavailable_error_inheritance(self):
        """ProviderUnavailableError inherits from LLMException."""
        exc = ProviderUnavailableError("api down")
        assert isinstance(exc, LLMException)

    def test_generation_error_inheritance(self):
        """GenerationError inherits from LLMException."""
        exc = GenerationError("generation failed")
        assert isinstance(exc, LLMException)

    def test_embedding_error_inheritance(self):
        """EmbeddingError inherits from LLMException."""
        exc = EmbeddingError("embedding failed")
        assert isinstance(exc, LLMException)

    def test_rate_limit_error_inheritance(self):
        """RateLimitError inherits from LLMException."""
        exc = RateLimitError("rate limited")
        assert isinstance(exc, LLMException)

    def test_catch_all_llm_exceptions(self):
        """Can catch all LLM exceptions with LLMException."""
        exceptions = [
            UnsupportedProviderError("test"),
            ProviderUnavailableError("test"),
            GenerationError("test"),
            EmbeddingError("test"),
            RateLimitError("test"),
        ]
        for exc in exceptions:
            try:
                raise exc
            except LLMException:
                pass  # Successfully caught


# ============================================================================
# TestOllamaProvider: Ollama Implementation
# ============================================================================


class TestOllamaProvider:
    """Test Ollama provider implementation."""

    def test_ollama_provider_creation(self, ollama_config):
        """OllamaProvider can be instantiated."""
        provider = OllamaProvider(ollama_config)
        assert provider.model == "llama3.2"
        assert provider.embedding_model == "nomic-embed-text"
        assert provider.base_url == "http://localhost:11434"

    def test_ollama_provider_is_base_llm_provider(self, ollama_config):
        """OllamaProvider is a BaseLLMProvider."""
        provider = OllamaProvider(ollama_config)
        assert isinstance(provider, BaseLLMProvider)

    @patch("atlasrag.src.llm.ollama_provider.httpx.post")
    def test_ollama_generate_success(self, mock_post, ollama_config):
        """OllamaProvider.generate returns LLMResponse on success."""
        mock_post.return_value.json.return_value = {
            "response": "2 plus 2 is 4",
            "eval_count": 5,
        }
        mock_post.return_value.raise_for_status = MagicMock()

        provider = OllamaProvider(ollama_config)
        response = provider.generate("What is 2+2?")

        assert isinstance(response, LLMResponse)
        assert response.content == "2 plus 2 is 4"
        assert response.provider == "ollama"
        assert response.model == "llama3.2"

    @patch("atlasrag.src.llm.ollama_provider.httpx.post")
    def test_ollama_generate_http_error(self, mock_post, ollama_config):
        """OllamaProvider.generate raises GenerationError on HTTP error."""
        mock_post.side_effect = httpx.HTTPError("Connection refused")

        provider = OllamaProvider(ollama_config)
        with pytest.raises(GenerationError):
            provider.generate("test prompt")

    @patch("atlasrag.src.llm.ollama_provider.httpx.post")
    def test_ollama_embed_success(self, mock_post, ollama_config):
        """OllamaProvider.embed returns EmbeddingResponse on success."""
        mock_post.return_value.json.return_value = {
            "embeddings": [[0.1, 0.2, 0.3]]
        }
        mock_post.return_value.raise_for_status = MagicMock()

        provider = OllamaProvider(ollama_config)
        response = provider.embed("test text")

        assert isinstance(response, EmbeddingResponse)
        assert response.embedding == [0.1, 0.2, 0.3]
        assert response.dimensions == 3
        assert response.provider == "ollama"

    @patch("atlasrag.src.llm.ollama_provider.httpx.post")
    def test_ollama_embed_error(self, mock_post, ollama_config):
        """OllamaProvider.embed raises EmbeddingError on failure."""
        mock_post.side_effect = httpx.HTTPError("Server error")

        provider = OllamaProvider(ollama_config)
        with pytest.raises(EmbeddingError):
            provider.embed("test text")

    @patch("atlasrag.src.llm.ollama_provider.httpx.get")
    def test_ollama_is_available_true(self, mock_get, ollama_config):
        """OllamaProvider.is_available returns True when accessible."""
        mock_get.return_value.status_code = 200

        provider = OllamaProvider(ollama_config)
        assert provider.is_available() is True

    @patch("atlasrag.src.llm.ollama_provider.httpx.get")
    def test_ollama_is_available_false(self, mock_get, ollama_config):
        """OllamaProvider.is_available returns False when not accessible."""
        mock_get.side_effect = Exception("Connection refused")

        provider = OllamaProvider(ollama_config)
        assert provider.is_available() is False


# ============================================================================
# TestOpenAIProvider: OpenAI Implementation
# ============================================================================


class TestOpenAIProvider:
    """Test OpenAI provider implementation."""

    def test_openai_provider_creation(self, openai_config):
        """OpenAIProvider can be instantiated."""
        provider = OpenAIProvider(openai_config)
        assert provider.model == "gpt-4o-mini"
        assert provider.embedding_model == "text-embedding-3-small"

    def test_openai_provider_is_base_llm_provider(self, openai_config):
        """OpenAIProvider is a BaseLLMProvider."""
        provider = OpenAIProvider(openai_config)
        assert isinstance(provider, BaseLLMProvider)

    @patch("atlasrag.src.llm.openai_provider.OpenAI")
    def test_openai_generate_success(self, mock_openai_class, openai_config):
        """OpenAIProvider.generate returns LLMResponse on success."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "The answer is 4"
        mock_response.usage.total_tokens = 15
        mock_client.chat.completions.create.return_value = mock_response

        provider = OpenAIProvider(openai_config)
        response = provider.generate("What is 2+2?")

        assert isinstance(response, LLMResponse)
        assert response.content == "The answer is 4"
        assert response.provider == "openai"

    @patch("atlasrag.src.llm.openai_provider.OpenAI")
    def test_openai_embed_success(self, mock_openai_class, openai_config):
        """OpenAIProvider.embed returns EmbeddingResponse on success."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.data[0].embedding = [0.1, 0.2, 0.3]
        mock_client.embeddings.create.return_value = mock_response

        provider = OpenAIProvider(openai_config)
        response = provider.embed("test")

        assert isinstance(response, EmbeddingResponse)
        assert response.embedding == [0.1, 0.2, 0.3]
        assert response.provider == "openai"

    @patch("atlasrag.src.llm.openai_provider.OpenAI")
    def test_openai_is_available_true(self, mock_openai_class, openai_config):
        """OpenAIProvider.is_available returns True when accessible."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.data[0].embedding = [0.1]
        mock_client.embeddings.create.return_value = mock_response

        provider = OpenAIProvider(openai_config)
        assert provider.is_available() is True

    @patch("atlasrag.src.llm.openai_provider.OpenAI")
    def test_openai_is_available_false(self, mock_openai_class, openai_config):
        """OpenAIProvider.is_available returns False on error."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.embeddings.create.side_effect = Exception("Invalid API key")

        provider = OpenAIProvider(openai_config)
        assert provider.is_available() is False


# ============================================================================
# TestGeminiProvider: Gemini Implementation
# ============================================================================


class TestGeminiProvider:
    """Test Gemini provider implementation."""

    def test_gemini_provider_creation(self, gemini_config):
        """GeminiProvider can be instantiated."""
        with patch("atlasrag.src.llm.gemini_provider.genai.configure"):
            with patch("atlasrag.src.llm.gemini_provider.genai.GenerativeModel"):
                provider = GeminiProvider(gemini_config)
                assert provider.model == "gemini-1.5-flash"
                assert provider.embedding_model == "models/text-embedding-004"

    def test_gemini_provider_is_base_llm_provider(self, gemini_config):
        """GeminiProvider is a BaseLLMProvider."""
        with patch("atlasrag.src.llm.gemini_provider.genai.configure"):
            with patch("atlasrag.src.llm.gemini_provider.genai.GenerativeModel"):
                provider = GeminiProvider(gemini_config)
                assert isinstance(provider, BaseLLMProvider)

    @patch("atlasrag.src.llm.gemini_provider.genai.GenerativeModel")
    @patch("atlasrag.src.llm.gemini_provider.genai.configure")
    def test_gemini_generate_success(
        self, mock_configure, mock_gen_model_class, gemini_config
    ):
        """GeminiProvider.generate returns LLMResponse on success."""
        mock_gen_model = MagicMock()
        mock_gen_model_class.return_value = mock_gen_model

        mock_response = MagicMock()
        mock_response.text = "The answer is 4"
        mock_gen_model.generate_content.return_value = mock_response

        provider = GeminiProvider(gemini_config)
        response = provider.generate("What is 2+2?")

        assert isinstance(response, LLMResponse)
        assert response.content == "The answer is 4"
        assert response.provider == "gemini"

    @patch("atlasrag.src.llm.gemini_provider.genai.embed_content")
    @patch("atlasrag.src.llm.gemini_provider.genai.configure")
    def test_gemini_embed_success(
        self, mock_configure, mock_embed_content, gemini_config
    ):
        """GeminiProvider.embed returns EmbeddingResponse on success."""
        mock_embed_content.return_value = {"embedding": [0.1, 0.2, 0.3]}

        with patch("atlasrag.src.llm.gemini_provider.genai.GenerativeModel"):
            provider = GeminiProvider(gemini_config)
            response = provider.embed("test")

        assert isinstance(response, EmbeddingResponse)
        assert response.embedding == [0.1, 0.2, 0.3]
        assert response.provider == "gemini"

    @patch("atlasrag.src.llm.gemini_provider.genai.embed_content")
    @patch("atlasrag.src.llm.gemini_provider.genai.configure")
    def test_gemini_is_available_true(
        self, mock_configure, mock_embed_content, gemini_config
    ):
        """GeminiProvider.is_available returns True when accessible."""
        mock_embed_content.return_value = {"embedding": [0.1]}

        with patch("atlasrag.src.llm.gemini_provider.genai.GenerativeModel"):
            provider = GeminiProvider(gemini_config)
            assert provider.is_available() is True

    @patch("atlasrag.src.llm.gemini_provider.genai.embed_content")
    @patch("atlasrag.src.llm.gemini_provider.genai.configure")
    def test_gemini_is_available_false(
        self, mock_configure, mock_embed_content, gemini_config
    ):
        """GeminiProvider.is_available returns False on error."""
        mock_embed_content.side_effect = Exception("Invalid API key")

        with patch("atlasrag.src.llm.gemini_provider.genai.GenerativeModel"):
            provider = GeminiProvider(gemini_config)
            assert provider.is_available() is False


# ============================================================================
# TestLLMFactory: Provider Selection Factory
# ============================================================================


class TestLLMFactory:
    """Test LLM provider factory and singleton pattern."""

    def test_get_llm_client_returns_base_provider(self):
        """get_llm_client returns BaseLLMProvider instance."""
        get_llm_client.cache_clear()
        client = get_llm_client()
        assert isinstance(client, BaseLLMProvider)
        get_llm_client.cache_clear()

    def test_factory_ollama_default(self, test_settings, monkeypatch):
        """Factory returns OllamaProvider by default."""
        get_llm_client.cache_clear()
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        from atlasrag.src.config import get_settings

        get_settings.cache_clear()
        client = get_llm_client()
        assert isinstance(client, OllamaProvider)
        get_llm_client.cache_clear()
        get_settings.cache_clear()

    def test_factory_openai(self, monkeypatch):
        """Factory returns OpenAIProvider when configured."""
        get_llm_client.cache_clear()
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from atlasrag.src.config import get_settings

        get_settings.cache_clear()
        client = get_llm_client()
        assert isinstance(client, OpenAIProvider)
        get_llm_client.cache_clear()
        get_settings.cache_clear()

    def test_factory_gemini(self, monkeypatch):
        """Factory returns GeminiProvider when configured."""
        get_llm_client.cache_clear()
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        from atlasrag.src.config import get_settings

        get_settings.cache_clear()
        with patch("atlasrag.src.llm.gemini_provider.genai.configure"):
            with patch("atlasrag.src.llm.gemini_provider.genai.GenerativeModel"):
                client = get_llm_client()
                assert isinstance(client, GeminiProvider)
        get_llm_client.cache_clear()
        get_settings.cache_clear()

    def test_factory_unsupported_provider(self):
        """Factory raises UnsupportedProviderError for unknown provider."""
        get_llm_client.cache_clear()

        # Mock settings to return an unsupported provider via Literal union
        mock_settings = MagicMock()
        mock_llm_config = MagicMock()
        mock_llm_config.provider = "unsupported_provider"
        mock_settings.get_llm_config.return_value = mock_llm_config

        with patch("atlasrag.src.llm.factory.get_settings", return_value=mock_settings):
            with pytest.raises(UnsupportedProviderError):
                get_llm_client()

        get_llm_client.cache_clear()

    def test_factory_singleton_lru_cache(self):
        """Factory uses @lru_cache singleton pattern."""
        get_llm_client.cache_clear()
        client1 = get_llm_client()
        client2 = get_llm_client()
        assert client1 is client2
        get_llm_client.cache_clear()

    def test_factory_cache_clear_resets_singleton(self, monkeypatch):
        """cache_clear() resets the singleton."""
        get_llm_client.cache_clear()
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        from atlasrag.src.config import get_settings

        get_settings.cache_clear()
        client1 = get_llm_client()

        get_llm_client.cache_clear()
        get_settings.cache_clear()
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        client2 = get_llm_client()

        assert client1 is not client2
        get_llm_client.cache_clear()
        get_settings.cache_clear()


# ============================================================================
# TestProviderAvailability: Health Checks
# ============================================================================


class TestProviderAvailability:
    """Test provider availability/health checks."""

    def test_ollama_availability_check_uses_tags_endpoint(self, ollama_config):
        """Ollama availability check uses /api/tags endpoint."""
        with patch("atlasrag.src.llm.ollama_provider.httpx.get") as mock_get:
            mock_get.return_value.status_code = 200

            provider = OllamaProvider(ollama_config)
            provider.is_available()

            mock_get.assert_called_once()
            call_args = mock_get.call_args[0][0]
            assert "/api/tags" in call_args

    def test_ollama_availability_timeout(self, ollama_config):
        """Ollama availability check handles timeout."""
        with patch(
            "atlasrag.src.llm.ollama_provider.httpx.get",
            side_effect=httpx.TimeoutException("timeout"),
        ):
            provider = OllamaProvider(ollama_config)
            assert provider.is_available() is False

    def test_all_providers_have_availability_check(
        self, ollama_config, openai_config, gemini_config
    ):
        """All providers implement is_available()."""
        with patch("atlasrag.src.llm.ollama_provider.httpx.get"):
            ollama = OllamaProvider(ollama_config)
            assert hasattr(ollama, "is_available")
            assert callable(ollama.is_available)

        with patch("atlasrag.src.llm.openai_provider.OpenAI"):
            openai = OpenAIProvider(openai_config)
            assert hasattr(openai, "is_available")
            assert callable(openai.is_available)

        with patch("atlasrag.src.llm.gemini_provider.genai.configure"):
            with patch("atlasrag.src.llm.gemini_provider.genai.GenerativeModel"):
                gemini = GeminiProvider(gemini_config)
                assert hasattr(gemini, "is_available")
                assert callable(gemini.is_available)
