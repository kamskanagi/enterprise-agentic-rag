"""Agent Graph Factory

Builds a LangGraph StateGraph that orchestrates the four agent nodes
(Planner → Retriever → Answerer → Verifier) with a conditional repair loop.
"""

import logging
from functools import partial

from langgraph.graph import END, StateGraph

from atlasrag.src.agents.answerer import answerer_node
from atlasrag.src.agents.planner import planner_node
from atlasrag.src.agents.retriever import retriever_node
from atlasrag.src.agents.state import AgentState
from atlasrag.src.agents.verifier import should_repair, verifier_node

logger = logging.getLogger(__name__)


def create_agent_graph(llm=None, vector_store=None):
    """Create and compile the agent workflow graph.

    Args:
        llm: Optional LLM provider override (injected into nodes).
        vector_store: Optional vector store override (injected into nodes).

    Returns:
        Compiled LangGraph graph ready for invocation.
    """
    # Bind optional dependencies via partial application
    _planner = partial(planner_node, llm=llm) if llm else planner_node
    _answerer = partial(answerer_node, llm=llm) if llm else answerer_node

    if llm or vector_store:
        kwargs = {}
        if llm:
            kwargs["llm"] = llm
        if vector_store:
            kwargs["vector_store"] = vector_store
        _retriever = partial(retriever_node, **kwargs)
    else:
        _retriever = retriever_node

    graph = StateGraph(AgentState)

    graph.add_node("planner", _planner)
    graph.add_node("retriever", _retriever)
    graph.add_node("answerer", _answerer)
    graph.add_node("verifier", verifier_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "answerer")
    graph.add_edge("answerer", "verifier")

    graph.add_conditional_edges(
        "verifier",
        should_repair,
        {
            "repair": "retriever",
            "complete": END,
        },
    )

    compiled = graph.compile()
    logger.info("Agent graph compiled successfully")
    return compiled
