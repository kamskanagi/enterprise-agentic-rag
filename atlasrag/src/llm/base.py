"""Abstract Base Class for LLM Providers

All concrete LLM provider implementations must inherit from BaseLLMProvider
and implement the abstract methods.
"""

from abc import ABC, abstractmethod
from .models import LLMResponse, EmbeddingResponse


class BaseLLMProvider(ABC):
    """
    Abstract interface for LLM providers.

    All providers (Ollama, OpenAI, Gemini) must implement these methods
    to ensure a consistent interface across different LLM backends.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """
        Generate text from a prompt.

        Args:
            prompt: The input prompt for generation
            max_tokens: Maximum tokens in the response (default 2000)
            temperature: Sampling temperature for creativity (0.0-1.0, default 0.7)

        Returns:
            LLMResponse: Response object containing generated text and metadata

        Raises:
            GenerationError: If text generation fails
            RateLimitError: If provider rate limit is exceeded
            ProviderUnavailableError: If provider API is not accessible
        """
        pass

    @abstractmethod
    def embed(self, text: str) -> EmbeddingResponse:
        """
        Generate embedding vector for text.

        Args:
            text: The input text to embed

        Returns:
            EmbeddingResponse: Response object containing embedding vector

        Raises:
            EmbeddingError: If embedding generation fails
            RateLimitError: If provider rate limit is exceeded
            ProviderUnavailableError: If provider API is not accessible
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the provider API is accessible.

        Returns:
            bool: True if provider is reachable, False otherwise
        """
        pass
