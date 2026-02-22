"""Google Gemini Provider Implementation

Cloud-based LLM provider using Google Gemini API.
Provides Gemini models for text generation and embeddings.
"""

import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

from atlasrag.src.config.llm_config import GeminiConfig
from .base import BaseLLMProvider
from .models import LLMResponse, EmbeddingResponse
from .exceptions import GenerationError, EmbeddingError, RateLimitError


class GeminiProvider(BaseLLMProvider):
    """Provider for Google Gemini models.

    Uses Google's generativeai SDK to communicate with Gemini API.
    Requires GEMINI_API_KEY environment variable or config.
    """

    def __init__(self, config: GeminiConfig):
        """Initialize Gemini provider.

        Args:
            config: GeminiConfig with api_key, model, embedding_model
        """
        genai.configure(api_key=config.api_key.get_secret_value())
        self.model = config.model
        self.embedding_model = config.embedding_model
        self.generation_model = genai.GenerativeModel(self.model)

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
        """Generate text using Gemini API.

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
            response = self.generation_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                ),
            )
            # Gemini doesn't directly return token count, estimate from response
            estimated_tokens = len(response.text.split()) * 1.3

            return LLMResponse(
                content=response.text,
                tokens_used=int(estimated_tokens),
                model=self.model,
                provider="gemini",
            )
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "quota" in error_str.lower():
                raise RateLimitError(f"Gemini rate limit exceeded: {error_str}")
            raise GenerationError(f"Gemini generation failed: {error_str}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def embed(self, text: str) -> EmbeddingResponse:
        """Generate embedding using Gemini API.

        Args:
            text: Input text to embed

        Returns:
            EmbeddingResponse: Response with embedding vector

        Raises:
            EmbeddingError: If embedding generation fails
            RateLimitError: If API rate limit exceeded
        """
        try:
            response = genai.embed_content(
                model=self.embedding_model,
                content=text,
            )
            embedding = response["embedding"]

            return EmbeddingResponse(
                embedding=embedding,
                dimensions=len(embedding),
                model=self.embedding_model,
                provider="gemini",
            )
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "quota" in error_str.lower():
                raise RateLimitError(f"Gemini rate limit exceeded: {error_str}")
            raise EmbeddingError(f"Gemini embedding failed: {error_str}")

    def is_available(self) -> bool:
        """Check if Gemini API is accessible.

        Performs a minimal embedding call to verify connectivity.

        Returns:
            bool: True if API is accessible, False otherwise
        """
        try:
            genai.embed_content(model=self.embedding_model, content="test")
            return True
        except Exception:
            return False
