"""Tests for Phase 7: LangGraph Agent Architecture

Tests cover all agent nodes, the verifier router, and full graph invocation.
"""

import pytest
from unittest.mock import MagicMock, patch

from atlasrag.src.agents.state import AgentState
from atlasrag.src.agents.planner import planner_node
from atlasrag.src.agents.retriever import retriever_node
from atlasrag.src.agents.answerer import answerer_node, _extract_citations
from atlasrag.src.agents.verifier import (
    verifier_node,
    should_repair,
    _count_sentences,
    _count_cited_sentences,
)
from atlasrag.src.agents.graph import create_agent_graph
from atlasrag.src.llm.models import LLMResponse, EmbeddingResponse
from atlasrag.src.retrieval.models import SearchResult, SearchResults


# ============================================================================
# Helpers
# ============================================================================

def _make_llm_mock(generate_content="Mock answer [1].", embed_vector=None):
    """Create a mock LLM provider."""
    mock = MagicMock()
    mock.generate.return_value = LLMResponse(
        content=generate_content,
        tokens_used=10,
        model="test-model",
        provider="test-provider",
    )
    mock.embed.return_value = EmbeddingResponse(
        embedding=embed_vector or [0.1, 0.2, 0.3],
        dimensions=3,
        model="test-embed",
        provider="test-provider",
    )
    return mock


def _make_vector_store_mock(results=None):
    """Create a mock vector store."""
    mock = MagicMock()
    if results is None:
        results = [
            SearchResult(
                document_id="doc1",
                content="Test document content about policies.",
                similarity_score=0.9,
                metadata={"source": "handbook.pdf", "page": 5, "chunk_index": 0},
            ),
            SearchResult(
                document_id="doc2",
                content="Another document about benefits.",
                similarity_score=0.85,
                metadata={"source": "benefits.pdf", "page": 2, "chunk_index": 1},
            ),
        ]
    mock.similarity_search.return_value = SearchResults(
        query=None,
        results=results,
        total_results=len(results),
        vector_store="mock",
    )
    return mock


# ============================================================================
# TestAgentState
# ============================================================================

class TestAgentState:
    """Tests for AgentState TypedDict."""

    def test_create_minimal_state(self):
        state: AgentState = {"query": "What is the vacation policy?"}
        assert state["query"] == "What is the vacation policy?"

    def test_create_full_state(self):
        state: AgentState = {
            "query": "What is the vacation policy?",
            "sub_queries": ["vacation policy", "time off"],
            "context": [{"content": "20 days", "source": "handbook.pdf"}],
            "answer": "You get 20 days [1].",
            "citations": ["[1]"],
            "confidence": 0.95,
            "verification_passed": True,
            "repair_iterations": 0,
            "model": "llama3.2",
            "provider": "ollama",
        }
        assert state["verification_passed"] is True
        assert state["confidence"] == 0.95


# ============================================================================
# TestPlannerAgent
# ============================================================================

class TestPlannerAgent:
    """Tests for the Planner node."""

    def test_decompose_query(self):
        llm = _make_llm_mock(generate_content="What is the vacation policy?\nHow many days off?")
        state: AgentState = {"query": "Tell me about vacation and time off policies"}

        with patch("atlasrag.src.agents.planner.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(enable_query_planning=True)
            result = planner_node(state, llm=llm)

        assert len(result["sub_queries"]) == 2
        assert "What is the vacation policy?" in result["sub_queries"]
        assert "How many days off?" in result["sub_queries"]

    def test_single_query_passthrough(self):
        llm = _make_llm_mock(generate_content="What is the vacation policy?")
        state: AgentState = {"query": "What is the vacation policy?"}

        with patch("atlasrag.src.agents.planner.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(enable_query_planning=True)
            result = planner_node(state, llm=llm)

        assert result["sub_queries"] == ["What is the vacation policy?"]

    def test_planning_disabled(self):
        state: AgentState = {"query": "What is the vacation policy?"}

        with patch("atlasrag.src.agents.planner.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(enable_query_planning=False)
            result = planner_node(state)

        assert result["sub_queries"] == ["What is the vacation policy?"]

    def test_empty_llm_response_fallback(self):
        llm = _make_llm_mock(generate_content="   ")
        state: AgentState = {"query": "My query"}

        with patch("atlasrag.src.agents.planner.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(enable_query_planning=True)
            result = planner_node(state, llm=llm)

        assert result["sub_queries"] == ["My query"]


# ============================================================================
# TestRetrieverAgent
# ============================================================================

class TestRetrieverAgent:
    """Tests for the Retriever node."""

    def test_basic_retrieval(self):
        llm = _make_llm_mock()
        vs = _make_vector_store_mock()
        state: AgentState = {
            "query": "vacation policy",
            "sub_queries": ["vacation policy"],
        }

        with patch("atlasrag.src.agents.retriever.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                retrieval_top_k=5, similarity_threshold=0.7
            )
            result = retriever_node(state, llm=llm, vector_store=vs)

        assert len(result["context"]) == 2
        assert result["context"][0]["source"] == "handbook.pdf"
        assert result["repair_iterations"] == 0

    def test_deduplication(self):
        """Duplicate content across sub-queries should be deduplicated."""
        duplicate_result = SearchResult(
            document_id="doc1",
            content="Same content",
            similarity_score=0.9,
            metadata={"source": "a.pdf"},
        )
        llm = _make_llm_mock()
        vs = _make_vector_store_mock(results=[duplicate_result])
        state: AgentState = {
            "query": "q",
            "sub_queries": ["sub1", "sub2"],
        }

        with patch("atlasrag.src.agents.retriever.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                retrieval_top_k=5, similarity_threshold=0.7
            )
            result = retriever_node(state, llm=llm, vector_store=vs)

        # Two sub-queries but same content → only 1 unique chunk
        assert len(result["context"]) == 1

    def test_repair_increments_iteration(self):
        llm = _make_llm_mock()
        vs = _make_vector_store_mock()
        state: AgentState = {
            "query": "q",
            "sub_queries": ["sub1"],
            "context": [{"content": "existing"}],
            "repair_iterations": 1,
        }

        with patch("atlasrag.src.agents.retriever.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                retrieval_top_k=5, similarity_threshold=0.7
            )
            result = retriever_node(state, llm=llm, vector_store=vs)

        assert result["repair_iterations"] == 2


# ============================================================================
# TestAnswererAgent
# ============================================================================

class TestAnswererAgent:
    """Tests for the Answerer node."""

    def test_generate_answer_with_citations(self):
        llm = _make_llm_mock(
            generate_content="The vacation policy allows 20 days [1]. Benefits include health insurance [2]."
        )
        state: AgentState = {
            "query": "What are the benefits?",
            "context": [
                {"content": "20 days vacation", "source": "handbook.pdf", "page": 5, "chunk_index": 0, "similarity_score": 0.9},
                {"content": "Health insurance", "source": "benefits.pdf", "page": 2, "chunk_index": 1, "similarity_score": 0.85},
            ],
        }

        result = answerer_node(state, llm=llm)

        assert "[1]" in result["answer"]
        assert "[2]" in result["answer"]
        assert result["citations"] == ["[1]", "[2]"]
        assert result["model"] == "test-model"
        assert result["provider"] == "test-provider"

    def test_no_citations_in_answer(self):
        llm = _make_llm_mock(generate_content="I don't have enough information.")
        state: AgentState = {
            "query": "Unknown question",
            "context": [],
        }

        result = answerer_node(state, llm=llm)

        assert result["citations"] == []

    def test_extract_citations_helper(self):
        assert _extract_citations("foo [1] bar [2] baz [1]") == ["[1]", "[2]"]
        assert _extract_citations("no citations here") == []
        assert _extract_citations("[10] and [3]") == ["[10]", "[3]"]


# ============================================================================
# TestVerifierAgent
# ============================================================================

class TestVerifierAgent:
    """Tests for the Verifier node and should_repair router."""

    def test_pass_strict_mode(self):
        state: AgentState = {
            "answer": "Vacation is 20 days [1]. Benefits include health [2].",
            "citations": ["[1]", "[2]"],
        }

        with patch("atlasrag.src.agents.verifier.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                verification_mode="strict", min_citation_coverage=0.8
            )
            result = verifier_node(state)

        assert result["verification_passed"] is True
        assert result["confidence"] == 1.0

    def test_fail_strict_mode(self):
        state: AgentState = {
            "answer": "Vacation is 20 days [1]. Benefits are good. Insurance is provided. Time off is flexible.",
            "citations": ["[1]"],
        }

        with patch("atlasrag.src.agents.verifier.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                verification_mode="strict", min_citation_coverage=0.8
            )
            result = verifier_node(state)

        assert result["verification_passed"] is False
        assert result["confidence"] < 0.8

    def test_lenient_mode_uses_half_threshold(self):
        # 1 out of 2 sentences cited = 0.5 coverage
        state: AgentState = {
            "answer": "Vacation is 20 days [1]. Benefits are good.",
            "citations": ["[1]"],
        }

        with patch("atlasrag.src.agents.verifier.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                verification_mode="lenient", min_citation_coverage=0.8
            )
            result = verifier_node(state)

        # threshold = 0.8 / 2 = 0.4, coverage = 0.5 → pass
        assert result["verification_passed"] is True

    def test_disabled_mode_always_passes(self):
        state: AgentState = {
            "answer": "No citations at all here.",
            "citations": [],
        }

        with patch("atlasrag.src.agents.verifier.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(verification_mode="disabled")
            result = verifier_node(state)

        assert result["verification_passed"] is True

    def test_empty_answer(self):
        state: AgentState = {"answer": "", "citations": []}

        with patch("atlasrag.src.agents.verifier.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                verification_mode="strict", min_citation_coverage=0.8
            )
            result = verifier_node(state)

        assert result["verification_passed"] is False
        assert result["confidence"] == 0.0

    def test_count_sentences(self):
        assert _count_sentences("One. Two. Three.") == 3
        assert _count_sentences("Single sentence") == 1
        assert _count_sentences("") == 0

    def test_count_cited_sentences(self):
        assert _count_cited_sentences("Cited [1]. Not cited.") == 1
        assert _count_cited_sentences("Both [1]. Also [2].") == 2

    # --- should_repair tests ---

    def test_should_repair_when_failed_and_iterations_remain(self):
        state: AgentState = {
            "verification_passed": False,
            "repair_iterations": 0,
        }

        with patch("atlasrag.src.agents.verifier.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(max_repair_iterations=3)
            assert should_repair(state) == "repair"

    def test_should_complete_when_passed(self):
        state: AgentState = {
            "verification_passed": True,
            "repair_iterations": 0,
        }

        assert should_repair(state) == "complete"

    def test_should_complete_when_max_iterations_reached(self):
        state: AgentState = {
            "verification_passed": False,
            "repair_iterations": 3,
        }

        with patch("atlasrag.src.agents.verifier.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(max_repair_iterations=3)
            assert should_repair(state) == "complete"


# ============================================================================
# TestAgentGraph
# ============================================================================

class TestAgentGraph:
    """Tests for the full agent graph."""

    def _build_mocks(self, answer_content="The answer is X [1]."):
        llm = _make_llm_mock(generate_content=answer_content)
        vs = _make_vector_store_mock()
        return llm, vs

    def test_graph_creation(self):
        llm, vs = self._build_mocks()
        graph = create_agent_graph(llm=llm, vector_store=vs)
        assert graph is not None

    def test_full_pipeline_passes_verification(self):
        llm, vs = self._build_mocks(
            answer_content="The policy allows 20 days [1]. Benefits include health [2]."
        )
        graph = create_agent_graph(llm=llm, vector_store=vs)

        with patch("atlasrag.src.agents.planner.get_settings") as p_settings, \
             patch("atlasrag.src.agents.retriever.get_settings") as r_settings, \
             patch("atlasrag.src.agents.verifier.get_settings") as v_settings:

            for m in [p_settings, r_settings, v_settings]:
                m.return_value = MagicMock(
                    enable_query_planning=False,
                    retrieval_top_k=5,
                    similarity_threshold=0.7,
                    verification_mode="strict",
                    min_citation_coverage=0.8,
                    max_repair_iterations=3,
                )

            result = graph.invoke({"query": "What is the vacation policy?"})

        assert "answer" in result
        assert result["verification_passed"] is True

    def test_repair_loop_triggered(self):
        """Test that a failing answer triggers the repair loop."""
        call_count = {"n": 0}
        original_generate_content = "No citations at all in this answer."
        repaired_generate_content = "The answer is X [1]. And Y [2]."

        llm = MagicMock()
        llm.embed.return_value = EmbeddingResponse(
            embedding=[0.1, 0.2, 0.3],
            dimensions=3,
            model="test-embed",
            provider="test-provider",
        )

        def side_effect_generate(prompt, **kwargs):
            call_count["n"] += 1
            if call_count["n"] <= 1:
                content = original_generate_content
            else:
                content = repaired_generate_content
            return LLMResponse(
                content=content,
                tokens_used=10,
                model="test-model",
                provider="test-provider",
            )

        llm.generate.side_effect = side_effect_generate

        vs = _make_vector_store_mock()
        graph = create_agent_graph(llm=llm, vector_store=vs)

        with patch("atlasrag.src.agents.planner.get_settings") as p_settings, \
             patch("atlasrag.src.agents.retriever.get_settings") as r_settings, \
             patch("atlasrag.src.agents.verifier.get_settings") as v_settings:

            for m in [p_settings, r_settings, v_settings]:
                m.return_value = MagicMock(
                    enable_query_planning=False,
                    retrieval_top_k=5,
                    similarity_threshold=0.7,
                    verification_mode="strict",
                    min_citation_coverage=0.8,
                    max_repair_iterations=3,
                )

            result = graph.invoke({"query": "What is X?"})

        assert result["verification_passed"] is True
        assert result["repair_iterations"] >= 1

    def test_graph_without_dependency_injection(self):
        """Graph can be created without explicit llm/vector_store args."""
        graph = create_agent_graph()
        assert graph is not None
