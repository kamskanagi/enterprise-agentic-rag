"""Agent State Definition

TypedDict shared across all LangGraph agent nodes. Each node reads
from and writes partial updates to this state.
"""

from typing import Dict, List, TypedDict


class AgentState(TypedDict, total=False):
    """Shared state flowing through the agent graph.

    Fields:
        query: Original user question.
        sub_queries: Decomposed sub-queries from the Planner.
        context: Retrieved document chunks (list of dicts with
            content, source, page, chunk_index, similarity_score).
        answer: Generated answer text.
        citations: List of citation references extracted from the answer.
        confidence: Verification confidence score (0.0-1.0).
        verification_passed: Whether the answer passed verification.
        repair_iterations: Number of repair loop iterations completed.
        model: LLM model name used for generation.
        provider: LLM provider name used for generation.
    """

    query: str
    sub_queries: List[str]
    context: List[Dict]
    answer: str
    citations: List[str]
    confidence: float
    verification_passed: bool
    repair_iterations: int
    model: str
    provider: str
