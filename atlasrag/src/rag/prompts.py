"""Prompt templates for the RAG pipeline."""

from typing import List

from atlasrag.src.rag.exceptions import ContextBuildError
from atlasrag.src.rag.models import SourceReference

DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context documents.

Rules:
1. Answer ONLY based on the provided context. Do not use outside knowledge.
2. Cite your sources by referencing the source number in brackets, e.g. [1], [2].
3. If the context does not contain enough information to answer the question, say "I don't have enough information in the provided documents to answer this question."
4. Be concise and factual. Do not speculate or make assumptions beyond what the sources state.
5. If multiple sources support a claim, cite all relevant ones."""


def build_rag_prompt(query: str, context_chunks: List[SourceReference]) -> str:
    """Build a RAG prompt with numbered source references.

    Args:
        query: The user's question.
        context_chunks: Source references to include as context.

    Returns:
        Formatted prompt string with context and question.

    Raises:
        ContextBuildError: If prompt construction fails.
    """
    try:
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            source_label = chunk.source
            if chunk.page is not None:
                source_label += f", page {chunk.page}"
            context_parts.append(
                f"[{i}] (Source: {source_label})\n{chunk.content}"
            )

        context_text = "\n\n".join(context_parts)

        return (
            f"Context:\n{context_text}\n\n"
            f"Question: {query}\n\n"
            f"Answer based on the context above, citing sources with [1], [2], etc.:"
        )
    except Exception as e:
        raise ContextBuildError(f"Failed to build RAG prompt: {e}") from e
