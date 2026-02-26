"""API Request/Response Models

Pydantic models for all FastAPI endpoint inputs and outputs.
"""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ============================================================================
# Query endpoint
# ============================================================================


class QueryRequest(BaseModel):
    """Request body for POST /query."""

    query: str = Field(..., min_length=1, description="Question to ask")
    top_k: Optional[int] = Field(
        None, ge=1, le=50, description="Number of chunks to retrieve"
    )


class CitationDetail(BaseModel):
    """A single citation reference in the answer."""

    source: str
    page: Optional[int] = None
    chunk_index: Optional[int] = None
    similarity_score: Optional[float] = None


class QueryResponse(BaseModel):
    """Response body for POST /query."""

    answer: str
    citations: List[str]
    citation_details: List[CitationDetail] = []
    confidence: float
    verification_passed: bool
    model: Optional[str] = None
    provider: Optional[str] = None


# ============================================================================
# Ingest endpoint
# ============================================================================


class IngestResponse(BaseModel):
    """Response body for POST /ingest."""

    job_id: str
    status: str
    message: str


# ============================================================================
# Status endpoint
# ============================================================================


class JobStatusResponse(BaseModel):
    """Response body for GET /status/{job_id}."""

    job_id: str
    document_id: str
    status: str
    total_chunks: int = 0
    stored_chunks: int = 0
    progress: int = 0
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


# ============================================================================
# Health endpoint
# ============================================================================


class ComponentHealth(BaseModel):
    """Health status for a single component."""

    status: str  # "healthy" or "unhealthy"
    message: Optional[str] = None


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str  # "healthy", "degraded", "unhealthy"
    timestamp: datetime
    components: Dict[str, ComponentHealth] = {}


# ============================================================================
# Error responses
# ============================================================================


class ErrorResponse(BaseModel):
    """Standard error response body."""

    detail: str
    error_type: Optional[str] = None
