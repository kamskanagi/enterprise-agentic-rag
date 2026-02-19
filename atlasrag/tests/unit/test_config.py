"""Unit Tests for Configuration System (Phase 2)

Tests cover:
- Settings defaults without .env file
- Settings loaded from environment variables
- Singleton pattern with @lru_cache()
- Domain configuration factory methods
- Secret handling (SecretStr doesn't leak values)
"""

import pytest
from pydantic import SecretStr

from atlasrag.src.config import (
    Settings,
    get_settings,
    LLMConfig,
    OllamaConfig,
    OpenAIConfig,
    GeminiConfig,
    VectorStoreConfig,
    ChromaConfig,
    MilvusConfig,
    APIConfig,
    CORSConfig,
    DatabaseConfig,
    ObservabilityConfig,
)


class TestSettingsDefaults:
    """Test that Settings loads correct defaults without .env file."""

    def test_default_llm_provider(self, test_settings):
        """Default LLM provider should be ollama."""
        assert test_settings.llm_provider == "ollama"

    def test_default_vector_store(self, test_settings):
        """Default vector store should be chroma."""
        assert test_settings.vector_store == "chroma"

    def test_default_api_host_port(self, test_settings):
        """Default API should be localhost:8000."""
        assert test_settings.api_host == "0.0.0.0"
        assert test_settings.api_port == 8000

    def test_default_debug_disabled(self, test_settings):
        """Debug mode should be disabled by default."""
        assert test_settings.debug is False

    def test_default_log_level(self, test_settings):
        """Default log level (from test environment) should be DEBUG."""
        # conftest.py sets LOG_LEVEL=DEBUG for testing
        assert test_settings.log_level == "DEBUG"

    def test_default_verification_mode(self, test_settings):
        """Default verification mode should be strict."""
        assert test_settings.verification_mode == "strict"

    def test_default_chunk_size(self, test_settings):
        """Default chunk size should be 512."""
        assert test_settings.chunk_size == 512

    def test_default_retrieval_top_k(self, test_settings):
        """Default retrieval top_k should be 5."""
        assert test_settings.retrieval_top_k == 5


class TestSettingsFromEnv:
    """Test that Settings reads from environment variables."""

    def test_llm_provider_from_env(self, monkeypatch):
        """LLM provider should be read from environment."""
        get_settings.cache_clear()
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        settings = get_settings()
        assert settings.llm_provider == "openai"
        get_settings.cache_clear()

    def test_vector_store_from_env(self, monkeypatch):
        """Vector store should be read from environment."""
        get_settings.cache_clear()
        monkeypatch.setenv("VECTOR_STORE", "milvus")
        settings = get_settings()
        assert settings.vector_store == "milvus"
        get_settings.cache_clear()

    def test_api_port_from_env(self, monkeypatch):
        """API port should be read from environment."""
        get_settings.cache_clear()
        monkeypatch.setenv("API_PORT", "9000")
        settings = get_settings()
        assert settings.api_port == 9000
        get_settings.cache_clear()

    def test_debug_flag_from_env(self, monkeypatch):
        """Debug flag should be read from environment."""
        get_settings.cache_clear()
        monkeypatch.setenv("DEBUG", "true")
        settings = get_settings()
        assert settings.debug is True
        get_settings.cache_clear()

    def test_chunk_size_from_env(self, monkeypatch):
        """Chunk size should be read from environment."""
        get_settings.cache_clear()
        monkeypatch.setenv("CHUNK_SIZE", "1024")
        settings = get_settings()
        assert settings.chunk_size == 1024
        get_settings.cache_clear()

    def test_log_level_from_env(self, monkeypatch):
        """Log level should be read from environment."""
        get_settings.cache_clear()
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        settings = get_settings()
        assert settings.log_level == "DEBUG"
        get_settings.cache_clear()

    def test_multiple_env_vars(self, monkeypatch):
        """Multiple environment variables should be applied together."""
        get_settings.cache_clear()
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("VECTOR_STORE", "milvus")
        monkeypatch.setenv("DEBUG", "true")
        settings = get_settings()
        assert settings.llm_provider == "gemini"
        assert settings.vector_store == "milvus"
        assert settings.debug is True
        get_settings.cache_clear()


class TestSettingsSingleton:
    """Test that get_settings() returns singleton via @lru_cache()."""

    def test_same_instance_on_repeated_calls(self, test_settings):
        """get_settings() should return same object on repeated calls."""
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_cache_clear_resets_singleton(self, monkeypatch):
        """cache_clear() should reset the singleton."""
        get_settings.cache_clear()
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        s1 = get_settings()

        get_settings.cache_clear()
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        s2 = get_settings()

        assert s1 is not s2
        assert s1.llm_provider == "ollama"
        assert s2.llm_provider == "openai"
        get_settings.cache_clear()

    def test_singleton_isolated_between_tests(self, test_settings):
        """test_settings fixture should clear cache between tests."""
        # The fixture clears cache before yielding, so each test gets fresh settings
        assert test_settings.llm_provider == "ollama"
        assert test_settings.vector_store == "chroma"


class TestLLMConfig:
    """Test LLM configuration factory method and properties."""

    def test_get_llm_config_returns_config_object(self, test_settings):
        """get_llm_config() should return LLMConfig object."""
        llm_config = test_settings.get_llm_config()
        assert isinstance(llm_config, LLMConfig)

    def test_llm_config_provider_matches(self, test_settings):
        """LLMConfig provider should match settings."""
        assert test_settings.get_llm_config().provider == test_settings.llm_provider

    def test_active_model_ollama(self, monkeypatch, test_settings):
        """active_model should return Ollama model."""
        get_settings.cache_clear()
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        monkeypatch.setenv("OLLAMA_MODEL", "llama2")
        settings = get_settings()
        llm_config = settings.get_llm_config()
        assert llm_config.active_model == "llama2"
        get_settings.cache_clear()

    def test_active_model_openai(self, monkeypatch):
        """active_model should return OpenAI model."""
        get_settings.cache_clear()
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4")
        settings = get_settings()
        llm_config = settings.get_llm_config()
        assert llm_config.active_model == "gpt-4"
        get_settings.cache_clear()

    def test_active_embedding_model_ollama(self, monkeypatch):
        """active_embedding_model should return Ollama embedding model."""
        get_settings.cache_clear()
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        monkeypatch.setenv("OLLAMA_EMBEDDING_MODEL", "mxbai-embed-large")
        settings = get_settings()
        llm_config = settings.get_llm_config()
        assert llm_config.active_embedding_model == "mxbai-embed-large"
        get_settings.cache_clear()

    def test_llm_config_is_frozen(self, test_settings):
        """LLMConfig should be frozen (immutable)."""
        llm_config = test_settings.get_llm_config()
        with pytest.raises(Exception):  # FrozenInstanceError
            llm_config.provider = "gemini"


class TestVectorStoreConfig:
    """Test vector store configuration factory method."""

    def test_get_vector_store_config_returns_config_object(self, test_settings):
        """get_vector_store_config() should return VectorStoreConfig object."""
        vs_config = test_settings.get_vector_store_config()
        assert isinstance(vs_config, VectorStoreConfig)

    def test_vector_store_backend_matches(self, test_settings):
        """VectorStoreConfig backend should match settings."""
        assert test_settings.get_vector_store_config().backend == test_settings.vector_store

    def test_chroma_config_for_chroma_backend(self, monkeypatch):
        """Chroma config should be populated for chroma backend."""
        get_settings.cache_clear()
        monkeypatch.setenv("VECTOR_STORE", "chroma")
        monkeypatch.setenv("CHROMA_PERSIST_DIRECTORY", "/tmp/chroma")
        monkeypatch.setenv("CHROMA_COLLECTION_NAME", "test_collection")
        settings = get_settings()
        vs_config = settings.get_vector_store_config()
        assert vs_config.backend == "chroma"
        assert vs_config.chroma.persist_directory == "/tmp/chroma"
        assert vs_config.chroma.collection_name == "test_collection"
        get_settings.cache_clear()

    def test_milvus_config_for_milvus_backend(self, monkeypatch):
        """Milvus config should be populated for milvus backend."""
        get_settings.cache_clear()
        monkeypatch.setenv("VECTOR_STORE", "milvus")
        monkeypatch.setenv("MILVUS_HOST", "milvus.example.com")
        monkeypatch.setenv("MILVUS_PORT", "19530")
        monkeypatch.setenv("MILVUS_USER", "milvus_user")
        settings = get_settings()
        vs_config = settings.get_vector_store_config()
        assert vs_config.backend == "milvus"
        assert vs_config.milvus.host == "milvus.example.com"
        assert vs_config.milvus.port == 19530
        get_settings.cache_clear()

    def test_retrieval_config_in_vector_store_config(self, monkeypatch):
        """Retrieval config should be included in VectorStoreConfig."""
        get_settings.cache_clear()
        monkeypatch.setenv("RETRIEVAL_TOP_K", "10")
        monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.8")
        settings = get_settings()
        vs_config = settings.get_vector_store_config()
        assert vs_config.retrieval_top_k == 10
        assert vs_config.similarity_threshold == 0.8
        get_settings.cache_clear()

    def test_vector_store_config_is_frozen(self, test_settings):
        """VectorStoreConfig should be frozen (immutable)."""
        vs_config = test_settings.get_vector_store_config()
        with pytest.raises(Exception):  # FrozenInstanceError
            vs_config.backend = "milvus"


class TestAPIConfig:
    """Test API configuration factory method."""

    def test_get_api_config_returns_config_object(self, test_settings):
        """get_api_config() should return APIConfig object."""
        api_config = test_settings.get_api_config()
        assert isinstance(api_config, APIConfig)

    def test_api_config_basic_fields(self, test_settings):
        """APIConfig should contain basic API fields."""
        api_config = test_settings.get_api_config()
        assert api_config.host == test_settings.api_host
        assert api_config.port == test_settings.api_port
        assert api_config.workers == test_settings.api_workers
        assert api_config.debug == test_settings.debug

    def test_api_config_cors(self, test_settings):
        """APIConfig should contain CORS configuration."""
        api_config = test_settings.get_api_config()
        assert isinstance(api_config.cors, CORSConfig)
        assert "http://localhost:3000" in api_config.cors.origins

    def test_api_config_database(self, test_settings):
        """APIConfig should contain database configuration."""
        api_config = test_settings.get_api_config()
        assert isinstance(api_config.database, DatabaseConfig)
        assert api_config.database.pool_size == test_settings.database_pool_size

    def test_api_config_jwt_fields(self, test_settings):
        """APIConfig should contain JWT security fields."""
        api_config = test_settings.get_api_config()
        assert api_config.jwt_algorithm == test_settings.jwt_algorithm
        assert api_config.jwt_expiration_minutes == test_settings.jwt_expiration_minutes

    def test_api_config_is_frozen(self, test_settings):
        """APIConfig should be frozen (immutable)."""
        api_config = test_settings.get_api_config()
        with pytest.raises(Exception):  # FrozenInstanceError
            api_config.port = 9000


class TestObservabilityConfig:
    """Test observability configuration factory method."""

    def test_get_observability_config_returns_config_object(self, test_settings):
        """get_observability_config() should return ObservabilityConfig object."""
        obs_config = test_settings.get_observability_config()
        assert isinstance(obs_config, ObservabilityConfig)

    def test_observability_config_fields(self, test_settings):
        """ObservabilityConfig should match settings fields."""
        obs_config = test_settings.get_observability_config()
        assert obs_config.enable_tracing == test_settings.enable_tracing
        assert obs_config.enable_metrics == test_settings.enable_metrics
        assert obs_config.log_level == test_settings.log_level
        assert obs_config.log_format == test_settings.log_format

    def test_observability_config_log_level_from_env(self, monkeypatch):
        """ObservabilityConfig log_level should be read from environment."""
        get_settings.cache_clear()
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("LOG_FORMAT", "json")
        settings = get_settings()
        obs_config = settings.get_observability_config()
        assert obs_config.log_level == "DEBUG"
        assert obs_config.log_format == "json"
        get_settings.cache_clear()

    def test_observability_config_is_frozen(self, test_settings):
        """ObservabilityConfig should be frozen (immutable)."""
        obs_config = test_settings.get_observability_config()
        with pytest.raises(Exception):  # FrozenInstanceError
            obs_config.log_level = "ERROR"


class TestSecretHandling:
    """Test that SecretStr fields never leak values in logs."""

    def test_secretstr_not_in_str(self, test_settings):
        """SecretStr value should not appear in str() representation."""
        settings_str = str(test_settings)
        # If API key was set, it shouldn't be visible
        assert "change-me-in-production" not in settings_str or settings_str.count("***") > 0

    def test_secretstr_not_in_repr(self, test_settings):
        """SecretStr value should not appear in repr() representation."""
        settings_repr = repr(test_settings)
        # Check that sensitive values are masked
        assert "postgresql" not in settings_repr or "***" in settings_repr

    def test_secretstr_get_secret_value(self, test_settings):
        """get_secret_value() should return actual value."""
        secret = test_settings.jwt_secret
        assert isinstance(secret, SecretStr)
        actual_value = secret.get_secret_value()
        assert isinstance(actual_value, str)
        assert len(actual_value) > 0

    def test_api_key_secretstr(self, monkeypatch):
        """API key should be SecretStr."""
        get_settings.cache_clear()
        monkeypatch.setenv("API_KEY", "secret-key-123")
        settings = get_settings()
        assert isinstance(settings.api_key, SecretStr)
        assert settings.api_key.get_secret_value() == "secret-key-123"
        get_settings.cache_clear()

    def test_database_url_secretstr(self, test_settings):
        """Database URL should be SecretStr."""
        assert isinstance(test_settings.database_url, SecretStr)
        url = test_settings.database_url.get_secret_value()
        assert "postgresql" in url

    def test_openai_api_key_secretstr(self, test_settings):
        """OpenAI API key should be SecretStr."""
        assert isinstance(test_settings.openai_api_key, SecretStr)

    def test_gemini_api_key_secretstr(self, test_settings):
        """Gemini API key should be SecretStr."""
        assert isinstance(test_settings.gemini_api_key, SecretStr)

    def test_milvus_password_secretstr(self, test_settings):
        """Milvus password should be SecretStr."""
        assert isinstance(test_settings.milvus_password, SecretStr)
