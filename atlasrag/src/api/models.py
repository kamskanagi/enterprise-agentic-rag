"""API Request/Response Models

Pydantic models for all FastAPI endpoint inputs and outputs.
"""

from datetime import datetime

from pydantic import BaseModel, Field

# ============================================================================
# Query endpoint
# ============================================================================


class QueryRequest(BaseModel):
    """Request body for POST /query."""

    query: str = Field(..., min_length=1, description="Question to ask")
    top_k: int | None = Field(None, ge=1, le=50, description="Number of chunks to retrieve")


class CitationDetail(BaseModel):
    """A single citation reference in the answer."""

    source: str
    page: int | None = None
    chunk_index: int | None = None
    similarity_score: float | None = None


class QueryResponse(BaseModel):
    """Response body for POST /query."""

    answer: str
    citations: list[str]
    citation_details: list[CitationDetail] = []
    confidence: float
    verification_passed: bool
    model: str | None = None
    provider: str | None = None


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
    error_message: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


# ============================================================================
# Health endpoint
# ============================================================================


class ComponentHealth(BaseModel):
    """Health status for a single component."""

    status: str  # "healthy" or "unhealthy"
    message: str | None = None


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str  # "healthy", "degraded", "unhealthy"
    timestamp: datetime
    components: dict[str, ComponentHealth] = {}


# ============================================================================
# Error responses
# ============================================================================


class ErrorResponse(BaseModel):
    """Standard error response body."""

    detail: str
    error_type: str | None = None
