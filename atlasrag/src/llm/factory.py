"""LLM Provider Factory

Singleton factory function that instantiates the correct provider
based on configuration. Uses @lru_cache() for singleton pattern.
"""

from functools import lru_cache

from atlasrag.src.config import get_settings
from .base import BaseLLMProvider
from .exceptions import UnsupportedProviderError


@lru_cache()
def get_llm_client() -> BaseLLMProvider:
    """
    Get LLM provider instance based on settings.

    Returns the appropriate provider (Ollama, OpenAI, or Gemini)
    based on settings.llm_provider. Uses @lru_cache() singleton pattern.

    Returns:
        BaseLLMProvider: Instantiated provider matching current settings

    Raises:
        UnsupportedProviderError: If provider is not supported

    Example:
        from atlasrag.src.llm import get_llm_client

        llm = get_llm_client()
        response = llm.generate("Hello world")
        embeddings = llm.embed("test")
    """
    settings = get_settings()
    llm_config = settings.get_llm_config()

    provider_name = llm_config.provider

    if provider_name == "ollama":
        from .ollama_provider import OllamaProvider

        return OllamaProvider(llm_config.ollama)

    elif provider_name == "openai":
        from .openai_provider import OpenAIProvider

        return OpenAIProvider(llm_config.openai)

    elif provider_name == "gemini":
        from .gemini_provider import GeminiProvider

        return GeminiProvider(llm_config.gemini)

    else:
        raise UnsupportedProviderError(
            f"LLM provider '{provider_name}' is not supported. "
            f"Choose from: ollama, openai, gemini"
        )
