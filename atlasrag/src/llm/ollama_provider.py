"""Ollama Provider Implementation

Local LLM provider using Ollama HTTP API.
Provides both text generation and embedding endpoints.
"""

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from atlasrag.src.config.llm_config import OllamaConfig
from .base import BaseLLMProvider
from .models import LLMResponse, EmbeddingResponse
from .exceptions import ProviderUnavailableError, GenerationError, EmbeddingError


class OllamaProvider(BaseLLMProvider):
    """Provider for local Ollama LLM runtime.

    Communicates with Ollama via HTTP API on localhost:11434.
    Requires Ollama to be running locally.
    """

    def __init__(self, config: OllamaConfig):
        """Initialize Ollama provider.

        Args:
            config: OllamaConfig with base_url, model, embedding_model
        """
        self.base_url = config.base_url
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
        """Generate text using Ollama API.

        Args:
            prompt: Input prompt for generation
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature

        Returns:
            LLMResponse: Response with generated text

        Raises:
            GenerationError: If generation fails
            RateLimitError: Not applicable for Ollama
        """
        try:
            response = httpx.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature,
                    },
                },
                timeout=120.0,
            )
            response.raise_for_status()
            data = response.json()

            return LLMResponse(
                content=data["response"],
                tokens_used=data.get("eval_count", 0),
                model=self.model,
                provider="ollama",
            )
        except httpx.HTTPError as e:
            raise GenerationError(f"Ollama generation failed: {str(e)}")
        except Exception as e:
            raise GenerationError(f"Ollama generation error: {str(e)}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def embed(self, text: str) -> EmbeddingResponse:
        """Generate embedding using Ollama API.

        Args:
            text: Input text to embed

        Returns:
            EmbeddingResponse: Response with embedding vector

        Raises:
            EmbeddingError: If embedding generation fails
        """
        try:
            response = httpx.post(
                f"{self.base_url}/api/embed",
                json={"model": self.embedding_model, "input": text},
                timeout=120.0,
            )
            response.raise_for_status()
            data = response.json()
            embedding = data["embeddings"][0]

            return EmbeddingResponse(
                embedding=embedding,
                dimensions=len(embedding),
                model=self.embedding_model,
                provider="ollama",
            )
        except httpx.HTTPError as e:
            raise EmbeddingError(f"Ollama embedding failed: {str(e)}")
        except Exception as e:
            raise EmbeddingError(f"Ollama embedding error: {str(e)}")

    def is_available(self) -> bool:
        """Check if Ollama is running and accessible.

        Returns:
            bool: True if Ollama API responds, False otherwise
        """
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False
