"""LLM Provider Configuration Models

Frozen BaseModel classes for LLM provider configurations.
"""

from typing import Literal
from pydantic import BaseModel, SecretStr


class OllamaConfig(BaseModel, frozen=True):
    """Configuration for Ollama local LLM runtime."""

    base_url: str
    model: str
    embedding_model: str


class OpenAIConfig(BaseModel, frozen=True):
    """Configuration for OpenAI cloud LLM."""

    api_key: SecretStr
    model: str
    embedding_model: str


class GeminiConfig(BaseModel, frozen=True):
    """Configuration for Google Gemini cloud LLM."""

    api_key: SecretStr
    model: str
    embedding_model: str


class LLMConfig(BaseModel, frozen=True):
    """Unified LLM configuration with provider-specific sub-configs."""

    provider: Literal["ollama", "openai", "gemini"]
    ollama: OllamaConfig
    openai: OpenAIConfig
    gemini: GeminiConfig

    @property
    def active_model(self) -> str:
        """Get the model name for the active provider."""
        if self.provider == "ollama":
            return self.ollama.model
        elif self.provider == "openai":
            return self.openai.model
        elif self.provider == "gemini":
            return self.gemini.model
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    @property
    def active_embedding_model(self) -> str:
        """Get the embedding model name for the active provider."""
        if self.provider == "ollama":
            return self.ollama.embedding_model
        elif self.provider == "openai":
            return self.openai.embedding_model
        elif self.provider == "gemini":
            return self.gemini.embedding_model
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")
