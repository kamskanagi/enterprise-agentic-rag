"""Tests for Phase 9: FastAPI Server

Tests cover all endpoints (query, ingest, health, status), request validation,
error handling, and the app factory.
"""

import io
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from atlasrag.src.api.main import create_app
from atlasrag.src.api.models import (
    CitationDetail,
    ErrorResponse,
    HealthResponse,
    IngestResponse,
    JobStatusResponse,
    QueryRequest,
    QueryResponse,
)


# ============================================================================
# Helpers
# ============================================================================


def _make_test_client(graph_mock=None, pipeline_mock=None, settings_mock=None):
    """Create a TestClient with mocked dependencies."""
    app = create_app()

    if graph_mock is not None:
        app.dependency_overrides[
            _import("atlasrag.src.api.dependencies", "get_agent_graph")
        ] = lambda: graph_mock

    if pipeline_mock is not None:
        app.dependency_overrides[
            _import("atlasrag.src.api.dependencies", "get_ingestion_pipeline")
        ] = lambda: pipeline_mock

    if settings_mock is not None:
        app.dependency_overrides[
            _import("atlasrag.src.api.dependencies", "get_current_settings")
        ] = lambda: settings_mock

    return TestClient(app)


def _import(module_path: str, attr: str):
    """Import an attribute from a module path."""
    import importlib

    mod = importlib.import_module(module_path)
    return getattr(mod, attr)


def _make_graph_mock(result=None):
    """Create a mock agent graph that returns a fixed result."""
    mock = MagicMock()
    if result is None:
        result = {
            "answer": "The vacation policy allows 20 days [1].",
            "citations": ["[1]"],
            "context": [
                {
                    "content": "20 days vacation",
                    "source": "handbook.pdf",
                    "page": 5,
                    "chunk_index": 0,
                    "similarity_score": 0.92,
                }
            ],
            "confidence": 0.95,
            "verification_passed": True,
            "model": "test-model",
            "provider": "test-provider",
        }
    mock.invoke.return_value = result
    return mock


def _make_pipeline_mock(jobs=None):
    """Create a mock ingestion pipeline."""
    mock = MagicMock()
    mock.jobs = jobs or {}

    def get_status(job_id):
        return mock.jobs.get(job_id)

    mock.get_status = get_status
    return mock


def _make_settings_mock():
    """Create a mock settings object."""
    mock = MagicMock()
    mock.supported_file_types = [".pdf", ".docx", ".txt", ".md"]
    mock.get_api_config.return_value = MagicMock(
        host="0.0.0.0",
        port=8000,
        debug=False,
        cors=MagicMock(origins=["http://localhost:3000"]),
    )
    return mock


# ============================================================================
# TestAppFactory
# ============================================================================


class TestAppFactory:
    """Tests for the FastAPI app factory."""

    def test_create_app_returns_fastapi_instance(self):
        from fastapi import FastAPI

        app = create_app()
        assert isinstance(app, FastAPI)

    def test_app_has_correct_title(self):
        app = create_app()
        assert app.title == "AtlasRAG"

    def test_app_registers_routes(self):
        app = create_app()
        paths = [route.path for route in app.routes]
        assert "/query" in paths
        assert "/ingest" in paths
        assert "/health" in paths
        assert "/status/{job_id}" in paths


# ============================================================================
# TestQueryEndpoint
# ============================================================================


class TestQueryEndpoint:
    """Tests for POST /query."""

    def test_query_success(self):
        graph = _make_graph_mock()
        client = _make_test_client(graph_mock=graph)

        response = client.post("/query", json={"query": "What is the vacation policy?"})

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "The vacation policy allows 20 days [1]."
        assert data["citations"] == ["[1]"]
        assert data["confidence"] == 0.95
        assert data["verification_passed"] is True
        assert len(data["citation_details"]) == 1
        assert data["citation_details"][0]["source"] == "handbook.pdf"

    def test_query_invokes_graph_with_query(self):
        graph = _make_graph_mock()
        client = _make_test_client(graph_mock=graph)

        client.post("/query", json={"query": "My question"})

        graph.invoke.assert_called_once_with({"query": "My question"})

    def test_query_empty_query_rejected(self):
        graph = _make_graph_mock()
        client = _make_test_client(graph_mock=graph)

        response = client.post("/query", json={"query": ""})

        assert response.status_code == 422

    def test_query_missing_query_rejected(self):
        graph = _make_graph_mock()
        client = _make_test_client(graph_mock=graph)

        response = client.post("/query", json={})

        assert response.status_code == 422

    def test_query_graph_failure_returns_500(self):
        graph = MagicMock()
        graph.invoke.side_effect = RuntimeError("LLM unavailable")
        client = _make_test_client(graph_mock=graph)

        response = client.post("/query", json={"query": "test"})

        assert response.status_code == 500
        assert "Query processing failed" in response.json()["detail"]

    def test_query_with_optional_top_k(self):
        graph = _make_graph_mock()
        client = _make_test_client(graph_mock=graph)

        response = client.post("/query", json={"query": "test", "top_k": 10})

        assert response.status_code == 200

    def test_query_top_k_validation(self):
        graph = _make_graph_mock()
        client = _make_test_client(graph_mock=graph)

        response = client.post("/query", json={"query": "test", "top_k": 0})
        assert response.status_code == 422

        response = client.post("/query", json={"query": "test", "top_k": 100})
        assert response.status_code == 422


# ============================================================================
# TestIngestEndpoint
# ============================================================================


class TestIngestEndpoint:
    """Tests for POST /ingest."""

    def test_ingest_success(self):
        pipeline = _make_pipeline_mock()
        settings = _make_settings_mock()
        client = _make_test_client(pipeline_mock=pipeline, settings_mock=settings)

        response = client.post(
            "/ingest",
            files={"file": ("test.txt", io.BytesIO(b"Hello world"), "text/plain")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "pending"
        assert "job_id" in data
        assert "test.txt" in data["message"]

    def test_ingest_unsupported_file_type(self):
        pipeline = _make_pipeline_mock()
        settings = _make_settings_mock()
        client = _make_test_client(pipeline_mock=pipeline, settings_mock=settings)

        response = client.post(
            "/ingest",
            files={"file": ("test.exe", io.BytesIO(b"binary"), "application/octet-stream")},
        )

        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]

    def test_ingest_no_file_rejected(self):
        pipeline = _make_pipeline_mock()
        settings = _make_settings_mock()
        client = _make_test_client(pipeline_mock=pipeline, settings_mock=settings)

        response = client.post("/ingest")

        assert response.status_code == 422

    def test_ingest_pdf_accepted(self):
        pipeline = _make_pipeline_mock()
        settings = _make_settings_mock()
        client = _make_test_client(pipeline_mock=pipeline, settings_mock=settings)

        response = client.post(
            "/ingest",
            files={"file": ("doc.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
        )

        assert response.status_code == 200


# ============================================================================
# TestHealthEndpoint
# ============================================================================


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_all_healthy(self):
        client = _make_test_client()

        with patch("atlasrag.src.api.routes.health.get_llm_client") as mock_llm, \
             patch("atlasrag.src.api.routes.health.get_vector_store") as mock_vs:
            mock_llm.return_value = MagicMock(is_available=MagicMock(return_value=True))
            mock_vs.return_value = MagicMock(is_available=MagicMock(return_value=True))

            response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["components"]["llm"]["status"] == "healthy"
        assert data["components"]["vector_store"]["status"] == "healthy"
        assert "timestamp" in data

    def test_health_llm_unhealthy(self):
        client = _make_test_client()

        with patch("atlasrag.src.api.routes.health.get_llm_client") as mock_llm, \
             patch("atlasrag.src.api.routes.health.get_vector_store") as mock_vs:
            mock_llm.return_value = MagicMock(is_available=MagicMock(return_value=False))
            mock_vs.return_value = MagicMock(is_available=MagicMock(return_value=True))

            response = client.get("/health")

        data = response.json()
        assert data["status"] == "degraded"
        assert data["components"]["llm"]["status"] == "unhealthy"

    def test_health_all_unhealthy(self):
        client = _make_test_client()

        with patch("atlasrag.src.api.routes.health.get_llm_client") as mock_llm, \
             patch("atlasrag.src.api.routes.health.get_vector_store") as mock_vs:
            mock_llm.return_value = MagicMock(is_available=MagicMock(return_value=False))
            mock_vs.return_value = MagicMock(is_available=MagicMock(return_value=False))

            response = client.get("/health")

        data = response.json()
        assert data["status"] == "unhealthy"

    def test_health_llm_exception(self):
        client = _make_test_client()

        with patch("atlasrag.src.api.routes.health.get_llm_client") as mock_llm, \
             patch("atlasrag.src.api.routes.health.get_vector_store") as mock_vs:
            mock_llm.side_effect = RuntimeError("Connection refused")
            mock_vs.return_value = MagicMock(is_available=MagicMock(return_value=True))

            response = client.get("/health")

        data = response.json()
        assert data["status"] == "degraded"
        assert data["components"]["llm"]["status"] == "unhealthy"
        assert "Connection refused" in data["components"]["llm"]["message"]


# ============================================================================
# TestStatusEndpoint
# ============================================================================


class TestStatusEndpoint:
    """Tests for GET /status/{job_id}."""

    def test_status_found(self):
        from atlasrag.src.ingestion.models import IngestionJob

        job = IngestionJob(
            job_id="job-123",
            document_id="doc-456",
            status="completed",
            total_chunks=10,
            stored_chunks=10,
            progress=100,
            started_at=datetime(2025, 1, 1, 12, 0, 0),
            completed_at=datetime(2025, 1, 1, 12, 0, 5),
        )
        pipeline = _make_pipeline_mock(jobs={"job-123": job})
        client = _make_test_client(pipeline_mock=pipeline)

        response = client.get("/status/job-123")

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "job-123"
        assert data["status"] == "completed"
        assert data["progress"] == 100
        assert data["total_chunks"] == 10

    def test_status_not_found(self):
        pipeline = _make_pipeline_mock()
        client = _make_test_client(pipeline_mock=pipeline)

        response = client.get("/status/nonexistent")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_status_processing_job(self):
        from atlasrag.src.ingestion.models import IngestionJob

        job = IngestionJob(
            job_id="job-789",
            document_id="doc-abc",
            status="processing",
            total_chunks=20,
            stored_chunks=0,
            progress=40,
            started_at=datetime(2025, 1, 1, 12, 0, 0),
        )
        pipeline = _make_pipeline_mock(jobs={"job-789": job})
        client = _make_test_client(pipeline_mock=pipeline)

        response = client.get("/status/job-789")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "processing"
        assert data["progress"] == 40

    def test_status_failed_job(self):
        from atlasrag.src.ingestion.models import IngestionJob

        job = IngestionJob(
            job_id="job-fail",
            document_id="doc-fail",
            status="failed",
            error_message="File too large",
            started_at=datetime(2025, 1, 1, 12, 0, 0),
        )
        pipeline = _make_pipeline_mock(jobs={"job-fail": job})
        client = _make_test_client(pipeline_mock=pipeline)

        response = client.get("/status/job-fail")

        data = response.json()
        assert data["status"] == "failed"
        assert data["error_message"] == "File too large"


# ============================================================================
# TestRequestResponseModels
# ============================================================================


class TestRequestResponseModels:
    """Tests for Pydantic request/response models."""

    def test_query_request_validation(self):
        req = QueryRequest(query="What is X?")
        assert req.query == "What is X?"
        assert req.top_k is None

    def test_query_request_with_top_k(self):
        req = QueryRequest(query="test", top_k=10)
        assert req.top_k == 10

    def test_query_response_creation(self):
        resp = QueryResponse(
            answer="Answer [1].",
            citations=["[1]"],
            confidence=0.9,
            verification_passed=True,
        )
        assert resp.answer == "Answer [1]."

    def test_health_response_creation(self):
        resp = HealthResponse(
            status="healthy",
            timestamp=datetime.now(),
        )
        assert resp.status == "healthy"

    def test_error_response_creation(self):
        resp = ErrorResponse(detail="Something went wrong", error_type="ValueError")
        assert resp.detail == "Something went wrong"
