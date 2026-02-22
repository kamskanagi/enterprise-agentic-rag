"""Custom Exceptions for LLM Layer

Exception hierarchy for the LLM provider abstraction layer.
All exceptions inherit from LLMException for easy catching.
"""


class LLMException(Exception):
    """Base exception for all LLM operations."""

    pass


class UnsupportedProviderError(LLMException):
    """Raised when the specified LLM provider is not supported."""

    pass


class ProviderUnavailableError(LLMException):
    """Raised when the LLM provider API is not accessible."""

    pass


class GenerationError(LLMException):
    """Raised when text generation fails."""

    pass


class EmbeddingError(LLMException):
    """Raised when embedding generation fails."""

    pass


class RateLimitError(LLMException):
    """Raised when the provider rate limit is exceeded."""

    pass
