"""FastAPI Server

HTTP API for the AtlasRAG system.

Endpoints:
    POST /query        — Ask a question about company documents
    POST /ingest       — Upload a document to the knowledge base
    GET  /health       — System health check
    GET  /status/{id}  — Get ingestion job status
"""

from atlasrag.src.api.main import app, create_app

__all__ = ["app", "create_app"]
