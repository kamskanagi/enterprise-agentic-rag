"""Health Endpoint

GET /health — System health check with component status.
"""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter

from atlasrag.src.api.models import ComponentHealth, HealthResponse
from atlasrag.src.llm.factory import get_llm_client
from atlasrag.src.retrieval.factory import get_vector_store

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health():
    """Check system health including LLM and vector store availability."""
    components = {}
    overall = "healthy"

    # Check LLM provider
    try:
        llm = get_llm_client()
        if llm.is_available():
            components["llm"] = ComponentHealth(status="healthy")
        else:
            components["llm"] = ComponentHealth(
                status="unhealthy", message="LLM provider not reachable"
            )
            overall = "degraded"
    except Exception as e:
        components["llm"] = ComponentHealth(
            status="unhealthy", message=str(e)
        )
        overall = "degraded"

    # Check vector store
    try:
        vs = get_vector_store()
        if vs.is_available():
            components["vector_store"] = ComponentHealth(status="healthy")
        else:
            components["vector_store"] = ComponentHealth(
                status="unhealthy", message="Vector store not reachable"
            )
            overall = "degraded"
    except Exception as e:
        components["vector_store"] = ComponentHealth(
            status="unhealthy", message=str(e)
        )
        overall = "degraded"

    # If all components are unhealthy, mark overall as unhealthy
    if all(c.status == "unhealthy" for c in components.values()):
        overall = "unhealthy"

    return HealthResponse(
        status=overall,
        timestamp=datetime.now(timezone.utc),
        components=components,
    )
