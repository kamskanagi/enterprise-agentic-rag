"""Answerer Agent Node

Builds a RAG prompt from retrieved context chunks, generates an answer
with citations, and extracts citation references from the text.
"""

import logging
import re
from typing import Dict, List

from atlasrag.src.agents.state import AgentState
from atlasrag.src.llm.factory import get_llm_client
from atlasrag.src.rag.models import SourceReference
from atlasrag.src.rag.prompts import DEFAULT_SYSTEM_PROMPT, build_rag_prompt

logger = logging.getLogger(__name__)


def _context_to_source_refs(context: List[Dict]) -> List[SourceReference]:
    """Convert context dicts to SourceReference objects for prompt building."""
    return [
        SourceReference(
            content=chunk["content"],
            source=chunk.get("source", "unknown"),
            page=chunk.get("page"),
            chunk_index=chunk.get("chunk_index"),
            similarity_score=chunk.get("similarity_score", 0.0),
        )
        for chunk in context
    ]


def _extract_citations(text: str) -> List[str]:
    """Extract citation references like [1], [2] from answer text."""
    return sorted(set(re.findall(r"\[\d+\]", text)))


def answerer_node(state: AgentState, llm=None) -> dict:
    """Generate an answer with citations from retrieved context.

    Args:
        state: Current agent state with ``query`` and ``context``.
        llm: Optional LLM provider override.

    Returns:
        Partial state update with ``answer``, ``citations``,
        ``model``, and ``provider``.
    """
    if llm is None:
        llm = get_llm_client()

    query = state["query"]
    context = state.get("context", [])

    source_refs = _context_to_source_refs(context)
    user_prompt = build_rag_prompt(query, source_refs)
    prompt = f"{DEFAULT_SYSTEM_PROMPT}\n\n{user_prompt}"

    response = llm.generate(prompt)
    citations = _extract_citations(response.content)

    logger.info(
        "Answerer generated response with %d citations using %s/%s",
        len(citations),
        response.provider,
        response.model,
    )

    return {
        "answer": response.content,
        "citations": citations,
        "model": response.model,
        "provider": response.provider,
    }
