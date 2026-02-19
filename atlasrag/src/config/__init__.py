"""
Configuration Module

Centralized settings management using Pydantic v2 BaseSettings.

This module handles:
- Loading environment variables from .env file
- Validating configuration with type safety
- Providing domain-specific configuration objects
- Supporting multiple LLM providers (Ollama, OpenAI, Gemini)
- Supporting multiple vector stores (Chroma, Milvus)

Key components:
- settings.py: Main Settings class + get_settings() singleton
- llm_config.py: LLM provider configurations
- vector_store_config.py: Vector database configurations
- api_config.py: API server and database configurations
- observability_config.py: Logging and tracing configurations

Usage:
    from atlasrag.src.config import get_settings, LLMConfig

    settings = get_settings()
    print(f"LLM provider: {settings.llm_provider}")

    llm_config = settings.get_llm_config()
    print(f"Using model: {llm_config.active_model}")
"""

from .settings import Settings, get_settings
from .llm_config import LLMConfig, OllamaConfig, OpenAIConfig, GeminiConfig
from .vector_store_config import VectorStoreConfig, ChromaConfig, MilvusConfig
from .api_config import APIConfig, CORSConfig, DatabaseConfig
from .observability_config import ObservabilityConfig

__all__ = [
    # Main entry point
    "Settings",
    "get_settings",
    # LLM Configs
    "LLMConfig",
    "OllamaConfig",
    "OpenAIConfig",
    "GeminiConfig",
    # Vector Store Configs
    "VectorStoreConfig",
    "ChromaConfig",
    "MilvusConfig",
    # API Configs
    "APIConfig",
    "CORSConfig",
    "DatabaseConfig",
    # Observability Config
    "ObservabilityConfig",
]
