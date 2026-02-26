"""Query Endpoint

POST /query — Ask a question about company documents.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException

from atlasrag.src.api.dependencies import get_agent_graph
from atlasrag.src.api.models import CitationDetail, QueryRequest, QueryResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, graph=Depends(get_agent_graph)):
    """Run the agentic RAG pipeline for a user question.

    Invokes the full agent graph: Planner → Retriever → Answerer → Verifier
    (with optional repair loop).
    """
    logger.info("Query received: %s", request.query[:100])

    try:
        result = graph.invoke({"query": request.query})
    except Exception as e:
        logger.error("Agent graph failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Query processing failed: {e}")

    # Build citation details from context
    citation_details = []
    for chunk in result.get("context", []):
        citation_details.append(
            CitationDetail(
                source=chunk.get("source", "unknown"),
                page=chunk.get("page"),
                chunk_index=chunk.get("chunk_index"),
                similarity_score=chunk.get("similarity_score"),
            )
        )

    return QueryResponse(
        answer=result.get("answer", ""),
        citations=result.get("citations", []),
        citation_details=citation_details,
        confidence=result.get("confidence", 0.0),
        verification_passed=result.get("verification_passed", False),
        model=result.get("model"),
        provider=result.get("provider"),
    )
