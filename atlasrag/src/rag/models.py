"""Data models for the RAG pipeline."""

from pydantic import BaseModel


class SourceReference(BaseModel, frozen=True):
    """A reference to a source document chunk used in generating an answer."""

    content: str
    source: str
    page: int | None = None
    chunk_index: int | None = None
    similarity_score: float


class RAGResponse(BaseModel, frozen=True):
    """Response from the RAG pipeline."""

    answer: str
    sources: list[SourceReference]
    query: str
    model: str
    provider: str
    retrieval_count: int


class RAGConfig(BaseModel):
    """Configuration for the RAG pipeline."""

    top_k: int = 5
    similarity_threshold: float = 0.5
    max_tokens: int = 2000
    temperature: float = 0.3
    system_prompt: str | None = None
