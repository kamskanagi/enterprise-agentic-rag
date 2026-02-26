"""FastAPI Dependency Injection

Provides shared singletons (agent graph, ingestion pipeline) to route handlers.
"""

from functools import lru_cache

from atlasrag.src.agents.graph import create_agent_graph
from atlasrag.src.config import get_settings
from atlasrag.src.ingestion.pipeline import IngestionPipeline


@lru_cache()
def get_agent_graph():
    """Return a compiled agent graph singleton."""
    return create_agent_graph()


@lru_cache()
def get_ingestion_pipeline():
    """Return an ingestion pipeline singleton."""
    return IngestionPipeline()


def get_current_settings():
    """Return the current settings (delegates to config singleton)."""
    return get_settings()
