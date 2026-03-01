"""FastAPI Application Factory

Creates and configures the AtlasRAG HTTP API server.

Usage:
    python -m atlasrag.src.api.main
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from atlasrag.src.api.middleware import (
    ErrorHandlingMiddleware,
    MetricsMiddleware,
    RequestLoggingMiddleware,
)
from atlasrag.src.api.routes import health, ingest, query, status
from atlasrag.src.config import get_settings
from atlasrag.src.observability.logging import setup_logging
from atlasrag.src.observability.metrics import get_metrics_app
from atlasrag.src.observability.tracing import setup_tracing

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Build and return the FastAPI application."""
    settings = get_settings()
    api_config = settings.get_api_config()
    obs_config = settings.get_observability_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_logging(obs_config)
        logger.info(
            "AtlasRAG API starting on %s:%s", api_config.host, api_config.port
        )
        yield

    app = FastAPI(
        title="AtlasRAG",
        description="Enterprise Agentic RAG Platform",
        version="0.1.0",
        debug=api_config.debug,
        lifespan=lifespan,
    )

    # --- Tracing (if enabled) ---
    setup_tracing(app, obs_config)

    # --- Metrics endpoint ---
    if obs_config.enable_metrics:
        app.mount("/metrics", get_metrics_app())

    # --- Middleware (order matters: outermost first) ---
    app.add_middleware(RequestLoggingMiddleware)
    if obs_config.enable_metrics:
        app.add_middleware(MetricsMiddleware)
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=api_config.cors.origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Routes ---
    app.include_router(query.router, tags=["Query"])
    app.include_router(ingest.router, tags=["Ingest"])
    app.include_router(health.router, tags=["Health"])
    app.include_router(status.router, tags=["Status"])

    return app


app = create_app()


def run():
    """Entry point for ``atlasrag-api`` console script."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "atlasrag.src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        reload=settings.api_reload,
    )


if __name__ == "__main__":
    run()
