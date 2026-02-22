"""OpenAI Provider Implementation

Cloud-based LLM provider using OpenAI API.
Provides GPT models for text generation and embeddings.
"""

from openai import OpenAI, RateLimitError as OpenAIRateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential

from atlasrag.src.config.llm_config import OpenAIConfig
from .base import BaseLLMProvider
from .models import LLMResponse, EmbeddingResponse
from .exceptions import GenerationError, EmbeddingError, RateLimitError


class OpenAIProvider(BaseLLMProvider):
    """Provider for OpenAI GPT models.

    Uses OpenAI's official Python SDK to communicate with their API.
    Requires OPENAI_API_KEY environment variable or config.
    """

    def __init__(self, config: OpenAIConfig):
        """Initialize OpenAI provider.

        Args:
            config: OpenAIConfig with api_key, model, embedding_model
        """
        self.client = OpenAI(api_key=config.api_key.get_secret_value())
        self.model = config.model
        self.embedding_model = config.embedding_model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def generate(
        self,
        prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Generate text using OpenAI API.

        Args:
            prompt: Input prompt for generation
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature

        Returns:
            LLMResponse: Response with generated text

        Raises:
            GenerationError: If generation fails
            RateLimitError: If API rate limit exceeded
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )

            return LLMResponse(
                content=response.choices[0].message.content or "",
                tokens_used=response.usage.total_tokens,
                model=self.model,
                provider="openai",
            )
        except OpenAIRateLimitError as e:
            raise RateLimitError(f"OpenAI rate limit exceeded: {str(e)}")
        except Exception as e:
            raise GenerationError(f"OpenAI generation failed: {str(e)}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def embed(self, text: str) -> EmbeddingResponse:
        """Generate embedding using OpenAI API.

        Args:
            text: Input text to embed

        Returns:
            EmbeddingResponse: Response with embedding vector

        Raises:
            EmbeddingError: If embedding generation fails
            RateLimitError: If API rate limit exceeded
        """
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.embedding_model,
            )
            embedding = response.data[0].embedding

            return EmbeddingResponse(
                embedding=embedding,
                dimensions=len(embedding),
                model=self.embedding_model,
                provider="openai",
            )
        except OpenAIRateLimitError as e:
            raise RateLimitError(f"OpenAI rate limit exceeded: {str(e)}")
        except Exception as e:
            raise EmbeddingError(f"OpenAI embedding failed: {str(e)}")

    def is_available(self) -> bool:
        """Check if OpenAI API is accessible.

        Performs a minimal embedding call to verify connectivity.

        Returns:
            bool: True if API is accessible, False otherwise
        """
        try:
            self.client.embeddings.create(
                input="test",
                model=self.embedding_model,
            )
            return True
        except Exception:
            return False
