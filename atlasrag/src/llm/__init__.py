"""LLM Provider Abstraction Layer

Unified interface for multiple LLM providers with automatic provider selection
based on environment configuration.

Supported providers:
- Ollama: Local, privacy-preserving LLM runtime
- OpenAI: Cloud-based GPT models
- Gemini: Google's generative AI models

Usage:
    from atlasrag.src.llm import get_llm_client

    llm = get_llm_client()
    response = llm.generate("What is 2+2?")
    embeddings = llm.embed("Hello world")

Configuration:
    Set LLM_PROVIDER environment variable to choose:
    - "ollama": Local Ollama runtime (default)
    - "openai": OpenAI cloud API
    - "gemini": Google Gemini API

Design Pattern:
    Factory pattern with @lru_cache() singleton. All providers implement
    the BaseLLMProvider interface, enabling seamless provider switching
    without code changes.
"""

from .factory import get_llm_client
from .base import BaseLLMProvider
from .models import LLMResponse, EmbeddingResponse
from .exceptions import (
    LLMException,
    UnsupportedProviderError,
    ProviderUnavailableError,
    GenerationError,
    EmbeddingError,
    RateLimitError,
)
from .ollama_provider import OllamaProvider
from .openai_provider import OpenAIProvider
from .gemini_provider import GeminiProvider

__all__ = [
    "get_llm_client",
    "BaseLLMProvider",
    "OllamaProvider",
    "OpenAIProvider",
    "GeminiProvider",
    "LLMResponse",
    "EmbeddingResponse",
    "LLMException",
    "UnsupportedProviderError",
    "ProviderUnavailableError",
    "GenerationError",
    "EmbeddingError",
    "RateLimitError",
]
