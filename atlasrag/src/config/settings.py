"""Central Settings Module

Pydantic v2 BaseSettings class that reads all environment variables from .env
and provides factory methods to construct typed domain configuration objects.

Usage:
    from atlasrag.src.config import get_settings

    settings = get_settings()
    llm_config = settings.get_llm_config()
    print(f"Using LLM provider: {llm_config.provider}")
"""

from __future__ import annotations
from functools import lru_cache
from typing import Literal, Any
from pydantic import SecretStr, field_validator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.sources import DotEnvSettingsSource, EnvSettingsSource

from .llm_config import LLMConfig, OllamaConfig, OpenAIConfig, GeminiConfig
from .vector_store_config import VectorStoreConfig, ChromaConfig, MilvusConfig
from .api_config import APIConfig, CORSConfig, DatabaseConfig
from .observability_config import ObservabilityConfig


class CommaSeparatedListSource(EnvSettingsSource):
    """Custom env source that handles comma-separated lists without JSON parsing."""

    def decode_complex_value(self, field_name: str, field, value: str) -> Any:
        """Override to skip JSON parsing for comma-separated list fields."""
        if field_name in ("supported_file_types", "cors_origins"):
            # Return the string as-is, let the field validator handle parsing
            return value
        return super().decode_complex_value(field_name, field, value)


class CommaSeparatedDotEnvSource(DotEnvSettingsSource):
    """Custom dotenv source that handles comma-separated lists without JSON parsing."""

    def decode_complex_value(self, field_name: str, field, value: str) -> Any:
        """Override to skip JSON parsing for comma-separated list fields."""
        if field_name in ("supported_file_types", "cors_origins"):
            # Return the string as-is, let the field validator handle parsing
            return value
        return super().decode_complex_value(field_name, field, value)


class Settings(BaseSettings):
    """
    Central settings class that loads all environment variables from .env file.

    All environment variables are flat fields; domain-specific configurations
    are accessed via factory methods (get_llm_config(), get_api_config(), etc.)
    which construct frozen BaseModel views.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
        settings_sources=[
            CommaSeparatedDotEnvSource,
            CommaSeparatedListSource,
        ],
    )

    # =========================================================================
    # LLM Provider Configuration
    # =========================================================================
    llm_provider: Literal["ollama", "openai", "gemini"] = "ollama"

    # Ollama (local)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    ollama_embedding_model: str = "nomic-embed-text"

    # OpenAI (cloud)
    openai_api_key: SecretStr = SecretStr("")
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    # Gemini (cloud)
    gemini_api_key: SecretStr = SecretStr("")
    gemini_model: str = "gemini-1.5-flash"
    gemini_embedding_model: str = "models/text-embedding-004"

    # =========================================================================
    # Vector Store Configuration
    # =========================================================================
    vector_store: Literal["chroma", "milvus"] = "chroma"

    # Chroma (local)
    chroma_persist_directory: str = "./chroma_data"
    chroma_collection_name: str = "atlasrag"

    # Milvus (cloud/scalable)
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_user: str = ""
    milvus_password: SecretStr = SecretStr("")
    milvus_collection_name: str = "atlasrag"

    # =========================================================================
    # Database Configuration
    # =========================================================================
    database_url: SecretStr = SecretStr(
        "postgresql://atlasrag:atlasrag@localhost:5432/atlasrag"
    )
    database_pool_size: int = 5
    database_max_overflow: int = 10

    # =========================================================================
    # API Server Configuration
    # =========================================================================
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    api_reload: bool = True
    debug: bool = False
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:8080"]
    api_key: SecretStr = SecretStr("")

    # JWT / Security
    jwt_secret: SecretStr = SecretStr("change-me-in-production")
    jwt_algorithm: str = "HS256"
    jwt_expiration_minutes: int = 60

    # =========================================================================
    # Document Ingestion Configuration
    # =========================================================================
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_document_size_mb: int = 50
    supported_file_types: list[str] = [".pdf", ".docx", ".txt", ".md"]

    # =========================================================================
    # Retrieval Configuration
    # =========================================================================
    retrieval_top_k: int = 5
    similarity_threshold: float = 0.7
    enable_reranking: bool = False

    # =========================================================================
    # Agent Configuration
    # =========================================================================
    max_repair_iterations: int = 3
    min_citation_coverage: float = 0.8
    verification_mode: Literal["strict", "lenient", "disabled"] = "strict"
    enable_contradiction_detection: bool = True
    paragraph_citation_required: bool = True

    # =========================================================================
    # Observability Configuration
    # =========================================================================
    enable_tracing: bool = False
    enable_metrics: bool = True
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: Literal["json", "text"] = "text"

    # =========================================================================
    # Evaluation Configuration
    # =========================================================================
    eval_dataset_path: str = "atlasrag/eval/golden_dataset.jsonl"
    eval_batch_size: int = 10

    # =========================================================================
    # Feature Flags
    # =========================================================================
    enable_query_planning: bool = True
    enable_self_reflection: bool = True
    enable_multi_query: bool = False

    # =========================================================================
    # Validators (handle comma-separated strings from .env)
    # =========================================================================

    @field_validator("supported_file_types", mode="wrap")
    @classmethod
    def parse_supported_file_types(cls, v: Any, handler: Any) -> list[str]:
        """Parse comma-separated file types from .env string."""
        if isinstance(v, str):
            return [ft.strip() for ft in v.split(",")]
        if isinstance(v, list):
            return v
        return handler(v)

    @field_validator("cors_origins", mode="wrap")
    @classmethod
    def parse_cors_origins(cls, v: Any, handler: Any) -> list[str]:
        """Parse comma-separated CORS origins from .env string."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        if isinstance(v, list):
            return v
        return handler(v)

    # =========================================================================
    # Factory Methods (Domain Configuration Views)
    # =========================================================================

    def get_llm_config(self) -> LLMConfig:
        """Construct and return LLM configuration object."""
        return LLMConfig(
            provider=self.llm_provider,
            ollama=OllamaConfig(
                base_url=self.ollama_base_url,
                model=self.ollama_model,
                embedding_model=self.ollama_embedding_model,
            ),
            openai=OpenAIConfig(
                api_key=self.openai_api_key,
                model=self.openai_model,
                embedding_model=self.openai_embedding_model,
            ),
            gemini=GeminiConfig(
                api_key=self.gemini_api_key,
                model=self.gemini_model,
                embedding_model=self.gemini_embedding_model,
            ),
        )

    def get_vector_store_config(self) -> VectorStoreConfig:
        """Construct and return vector store configuration object."""
        return VectorStoreConfig(
            backend=self.vector_store,
            chroma=ChromaConfig(
                persist_directory=self.chroma_persist_directory,
                collection_name=self.chroma_collection_name,
            ),
            milvus=MilvusConfig(
                host=self.milvus_host,
                port=self.milvus_port,
                user=self.milvus_user,
                password=self.milvus_password,
                collection_name=self.milvus_collection_name,
            ),
            retrieval_top_k=self.retrieval_top_k,
            similarity_threshold=self.similarity_threshold,
            enable_reranking=self.enable_reranking,
        )

    def get_api_config(self) -> APIConfig:
        """Construct and return API configuration object."""
        return APIConfig(
            host=self.api_host,
            port=self.api_port,
            workers=self.api_workers,
            reload=self.api_reload,
            debug=self.debug,
            cors=CORSConfig(origins=self.cors_origins),
            database=DatabaseConfig(
                url=self.database_url,
                pool_size=self.database_pool_size,
                max_overflow=self.database_max_overflow,
            ),
            api_key=self.api_key,
            jwt_secret=self.jwt_secret,
            jwt_algorithm=self.jwt_algorithm,
            jwt_expiration_minutes=self.jwt_expiration_minutes,
        )

    def get_observability_config(self) -> ObservabilityConfig:
        """Construct and return observability configuration object."""
        return ObservabilityConfig(
            enable_tracing=self.enable_tracing,
            enable_metrics=self.enable_metrics,
            log_level=self.log_level,
            log_format=self.log_format,
        )


@lru_cache()
def get_settings() -> Settings:
    """
    Get singleton Settings instance.

    Uses @lru_cache() to ensure the same Settings object is returned on
    every call, avoiding multiple .env reads. Clear cache with
    get_settings.cache_clear() (useful for testing).
    """
    return Settings()
