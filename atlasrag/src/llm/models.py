"""Pydantic Models for LLM Responses

Type-safe response models for LLM operations with immutable (frozen) BaseModels.
"""

from typing import List
from pydantic import BaseModel


class LLMResponse(BaseModel, frozen=True):
    """
    Response from LLM text generation.

    Attributes:
        content: The generated text response
        tokens_used: Number of tokens consumed (estimated for some providers)
        model: Name of the model used
        provider: Name of the provider (ollama, openai, gemini)
    """

    content: str
    tokens_used: int
    model: str
    provider: str


class EmbeddingResponse(BaseModel, frozen=True):
    """
    Response from embedding generation.

    Attributes:
        embedding: Vector representation of the input text
        dimensions: Length of the embedding vector
        model: Name of the embedding model used
        provider: Name of the provider (ollama, openai, gemini)
    """

    embedding: List[float]
    dimensions: int
    model: str
    provider: str
