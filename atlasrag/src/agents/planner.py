"""Planner Agent Node

Decomposes a user query into sub-queries for retrieval.
When query planning is disabled, passes the original query through unchanged.
"""

import logging

from atlasrag.src.agents.state import AgentState
from atlasrag.src.config import get_settings
from atlasrag.src.llm.factory import get_llm_client

logger = logging.getLogger(__name__)

PLANNER_PROMPT = """You are a query planner. Given a user question, break it down into simpler sub-queries that can be used to retrieve relevant documents.

Rules:
1. Return one sub-query per line.
2. Each sub-query should target a specific piece of information.
3. If the question is already simple enough, return it unchanged as a single line.
4. Do NOT include numbering, bullets, or prefixes — just the plain text of each sub-query.

User question: {query}

Sub-queries:"""


def planner_node(state: AgentState, llm=None) -> dict:
    """Decompose a user query into retrieval sub-queries.

    Args:
        state: Current agent state with ``query`` field.
        llm: Optional LLM provider override (for testing).

    Returns:
        Partial state update with ``sub_queries`` list.
    """
    query = state["query"]
    settings = get_settings()

    if not settings.enable_query_planning:
        logger.info("Query planning disabled, passing query through")
        return {"sub_queries": [query]}

    if llm is None:
        llm = get_llm_client()

    prompt = PLANNER_PROMPT.format(query=query)
    response = llm.generate(prompt)

    raw_lines = response.content.strip().split("\n")
    sub_queries = [line.strip() for line in raw_lines if line.strip()]

    if not sub_queries:
        sub_queries = [query]

    logger.info("Planner decomposed query into %d sub-queries", len(sub_queries))
    return {"sub_queries": sub_queries}
