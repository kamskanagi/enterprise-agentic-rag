# LangGraph Agents (Phase 7)

**Status:** Planning phase - to be implemented in Phase 7

**Purpose:** Multi-agent system for complex RAG workflows.

## Agent Roster

### Planner Agent
**Role:** Query decomposition and strategy

Breaks complex questions into retrieval sub-queries.

Example:
- Input: "What's our vacation policy and how does it compare to remote work allowances?"
- Output: ["vacation days allowed per year", "remote work policy", "comparison"]

### Retriever Agent
**Role:** Document discovery

Searches vector database for relevant document chunks.

Example:
- Input: ["vacation days allowed per year"]
- Output: [Chunk 1, Chunk 2, Chunk 3] from HR policy

### Answerer Agent
**Role:** Response generation with citations

Generates answer using retrieved documents, automatically citing sources.

Example:
- Input: Query + retrieved chunks
- Output: "Our vacation policy provides 20 days annually (HR Policy, p. 3). ..."

### Verifier Agent
**Role:** Quality control

Checks that every claim has a citation to a source document.

Example:
- Input: Answer with claimed sources
- Output: Pass ✅ OR Fail → Trigger repair loop

## Agent Graph Flow

```
User Query ("What's our vacation policy?")
    ↓
Planner
  ├─ Breaks into sub-queries
  └─ Output: ["vacation days", "policy details"]
    ↓
Retriever
  ├─ Searches for each sub-query
  └─ Output: [Chunk1, Chunk2, Chunk3] (relevant docs)
    ↓
Answerer
  ├─ Generates response with citations
  └─ Output: "20 days annually (Policy p.3). ..."
    ↓
Verifier
  ├─ Checks citation coverage
  ├─ Validates sources exist
  └─ Output: Pass ✅ OR Fail → Repair Loop
    ↓ (If repair needed, max 2 iterations)
Repair Loop
  ├─ Retrieve more context
  ├─ Rewrite answer
  └─ Re-verify
    ↓
Final Answer ✅
```

## Key Files (to be implemented in Phase 7)

- `planner.py` - Query planning agent
- `retriever.py` - Document retrieval agent
- `answerer.py` - Answer generation agent
- `verifier.py` - Citation verification agent
- `graph.py` - LangGraph workflow orchestration
- `state.py` - Agent state and message passing
- `tools.py` - Tool definitions for agents

## Design Pattern

All agents are implemented using LangGraph nodes:

```python
from langgraph.graph import StateGraph
from src.agents.state import AgentState

# Create graph
graph = StateGraph(AgentState)

# Add agent nodes
graph.add_node("planner", planner_node)
graph.add_node("retriever", retriever_node)
graph.add_node("answerer", answerer_node)
graph.add_node("verifier", verifier_node)

# Add edges (control flow)
graph.add_edge("planner", "retriever")
graph.add_edge("retriever", "answerer")
graph.add_edge("answerer", "verifier")

# Conditional edge for repair loop
graph.add_conditional_edges(
    "verifier",
    should_repair,
    {
        "repair": "retriever",
        "complete": END
    }
)
```

## Usage Example (Phase 7+)

```python
from src.agents.graph import create_agent_graph

# Create the agent graph
graph = create_agent_graph()

# Execute with a query
result = graph.invoke({
    "query": "What's our vacation policy?",
    "context": []
})

print(f"Answer: {result['answer']}")
print(f"Citations: {result['citations']}")
print(f"Confidence: {result['confidence']:.0%}")
```

## Configuration Dependencies

Relies on Phase 2 configuration:
- `settings.max_repair_iterations` - Max verification attempts (default: 2)
- `settings.min_citation_coverage` - Min % with citations (default: 95%)
- `settings.verification_mode` - strict/moderate/lenient
- `settings.agent_timeout_seconds` - Max time per agent (default: 60)
- `settings.retrieval_top_k` - Docs to retrieve per query (default: 5)

## Integration with Other Modules

- **Phase 3 (LLM):** Uses `llm.generate()` for agent responses
- **Phase 4 (Retrieval):** Uses `vector_store.search()` for document lookup
- **Phase 9 (API):** Exposed via `/query` endpoint

## Testing (Phase 7+)

```python
import pytest
from src.agents.graph import create_agent_graph

@pytest.fixture
def agent_graph():
    return create_agent_graph()

def test_planner_agent(agent_graph):
    """Test that planner breaks down complex queries"""
    result = agent_graph.invoke({
        "query": "What's our policy on X and Y?",
        "context": []
    })

    # Should have multiple sub-queries
    assert len(result["sub_queries"]) >= 2

def test_verification_passes(agent_graph):
    """Test when answer is well-cited"""
    result = agent_graph.invoke({
        "query": "What's our vacation policy?"
    })

    # Should pass verification without repair
    assert result["verification_passed"] == True
    assert result["repair_iterations"] == 0

def test_repair_loop(agent_graph):
    """Test repair loop when verification fails"""
    # Mock poor retrieval
    result = agent_graph.invoke({
        "query": "specific_policy_question"
    })

    # Should trigger repair
    assert result["repair_iterations"] > 0
```

## Performance Optimization

- **Parallel retrieval:** Get multiple sub-queries in parallel
- **Caching:** Cache embeddings to avoid re-computation
- **Early termination:** Stop if confidence is high

## Observability

Track:
- Each agent's execution time
- Quality of retrieved documents
- Citation quality and coverage
- Repair loop frequency

## Future Enhancements

- [ ] Multi-turn conversations (remember context)
- [ ] Follow-up question handling
- [ ] Clarification questions when ambiguous
- [ ] Multi-source synthesis (combine from multiple docs)
- [ ] Debate mode (multiple agents argue points)
- [ ] Confidence scoring
