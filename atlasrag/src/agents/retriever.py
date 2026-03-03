"""Retriever Agent Node

Embeds each sub-query, searches the vector store, deduplicates results,
and returns context chunks for the Answerer.
"""

import logging
from typing import Dict, List

from atlasrag.src.agents.state import AgentState
from atlasrag.src.config import get_settings
from atlasrag.src.llm.factory import get_llm_client
from atlasrag.src.retrieval.factory import get_vector_store

logger = logging.getLogger(__name__)


def retriever_node(state: AgentState, llm=None, vector_store=None) -> dict:
    """Retrieve relevant document chunks for the sub-queries.

    Args:
        state: Current agent state with ``sub_queries``.
        llm: Optional LLM provider override.
        vector_store: Optional vector store override.

    Returns:
        Partial state update with ``context`` and ``repair_iterations``.
    """
    if llm is None:
        llm = get_llm_client()
    if vector_store is None:
        vector_store = get_vector_store()

    settings = get_settings()
    sub_queries = state.get("sub_queries", [state["query"]])

    # Track repair iterations — if repair_iterations > 0, this is a repair pass
    repair_iterations = state.get("repair_iterations", 0)
    is_repair = repair_iterations > 0 or bool(state.get("answer"))
    if is_repair:
        repair_iterations += 1

    # On repair, increase top_k by 1.5x to fetch more diverse sources (capped at 20)
    top_k = settings.retrieval_top_k
    if is_repair:
        top_k = min(int(top_k * 1.5), 20)
        logger.info("Repair pass: increased top_k from %d to %d", settings.retrieval_top_k, top_k)

    seen_contents: set = set()
    all_chunks: List[Dict] = []

    for sub_query in sub_queries:
        embedding_response = llm.embed(sub_query)
        search_results = vector_store.similarity_search(
            query_embedding=embedding_response.embedding,
            top_k=top_k,
            similarity_threshold=settings.similarity_threshold,
        )

        for result in search_results.results:
            if result.content in seen_contents:
                continue
            seen_contents.add(result.content)
            all_chunks.append({
                "content": result.content,
                "source": result.metadata.get("source", "unknown"),
                "page": result.metadata.get("page"),
                "chunk_index": result.metadata.get("chunk_index"),
                "similarity_score": result.similarity_score,
            })

    logger.info(
        "Retriever found %d unique chunks from %d sub-queries",
        len(all_chunks),
        len(sub_queries),
    )

    return {
        "context": all_chunks,
        "repair_iterations": repair_iterations,
    }
