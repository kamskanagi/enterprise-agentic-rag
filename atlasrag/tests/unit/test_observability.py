"""Tests for Phase 10: Observability

Tests structured logging, Prometheus metrics, tracing setup, and /metrics endpoint.
"""

import json
import logging
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from atlasrag.src.config.observability_config import ObservabilityConfig
from atlasrag.src.observability.logging import JSONFormatter, setup_logging
from atlasrag.src.observability.metrics import (
    ACTIVE_QUERIES,
    INGESTION_JOBS,
    REQUEST_COUNT,
    REQUEST_LATENCY,
    VERIFICATION_RESULTS,
    get_metrics_app,
)
from atlasrag.src.observability.tracing import setup_tracing


# ============================================================================
# JSON Formatter Tests
# ============================================================================


class TestJSONFormatter:
    def test_json_formatter_output(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=None,
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test.logger"
        assert parsed["message"] == "Test message"
        assert "timestamp" in parsed

    def test_json_formatter_with_exception(self):
        formatter = JSONFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Error occurred",
            args=None,
            exc_info=exc_info,
        )
        output = formatter.format(record)
        parsed = json.loads(output)

        assert "exception" in parsed
        assert "ValueError" in parsed["exception"]


# ============================================================================
# Setup Logging Tests
# ============================================================================


class TestSetupLogging:
    def test_setup_logging_json_mode(self):
        config = ObservabilityConfig(
            enable_tracing=False,
            enable_metrics=True,
            log_level="DEBUG",
            log_format="json",
        )
        setup_logging(config)

        root = logging.getLogger()
        assert root.level == logging.DEBUG
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, JSONFormatter)

    def test_setup_logging_text_mode(self):
        config = ObservabilityConfig(
            enable_tracing=False,
            enable_metrics=True,
            log_level="WARNING",
            log_format="text",
        )
        setup_logging(config)

        root = logging.getLogger()
        assert root.level == logging.WARNING
        assert len(root.handlers) == 1
        assert not isinstance(root.handlers[0].formatter, JSONFormatter)


# ============================================================================
# Prometheus Metrics Tests
# ============================================================================


class TestPrometheusMetrics:
    def test_prometheus_metrics_registered(self):
        assert REQUEST_COUNT is not None
        assert REQUEST_LATENCY is not None
        assert ACTIVE_QUERIES is not None
        assert INGESTION_JOBS is not None
        assert VERIFICATION_RESULTS is not None

    def test_metrics_increment(self):
        # Counter should be incrementable without error
        REQUEST_COUNT.labels(method="GET", path="/health", status=200).inc()
        INGESTION_JOBS.labels(status="success").inc()
        VERIFICATION_RESULTS.labels(result="pass").inc()

    def test_latency_observe(self):
        REQUEST_LATENCY.labels(path="/query").observe(0.5)

    def test_active_queries_gauge(self):
        ACTIVE_QUERIES.inc()
        ACTIVE_QUERIES.dec()

    def test_get_metrics_app(self):
        metrics_app = get_metrics_app()
        assert metrics_app is not None


# ============================================================================
# Tracing Tests
# ============================================================================


class TestTracing:
    def test_tracing_setup_disabled(self):
        config = ObservabilityConfig(
            enable_tracing=False,
            enable_metrics=True,
            log_level="INFO",
            log_format="text",
        )
        mock_app = MagicMock()
        # Should not raise, just no-op
        setup_tracing(mock_app, config)

    @patch("atlasrag.src.observability.tracing.logger")
    def test_tracing_setup_disabled_logs(self, mock_logger):
        config = ObservabilityConfig(
            enable_tracing=False,
            enable_metrics=True,
            log_level="INFO",
            log_format="text",
        )
        mock_app = MagicMock()
        setup_tracing(mock_app, config)
        mock_logger.info.assert_called_once()
        assert "disabled" in mock_logger.info.call_args[0][0].lower()


# ============================================================================
# Metrics Endpoint Integration Test
# ============================================================================


class TestMetricsEndpoint:
    @patch("atlasrag.src.config.settings.get_settings")
    def test_metrics_endpoint_returns_200(self, mock_settings):
        from atlasrag.src.config.settings import Settings

        mock_settings.return_value = Settings(
            _env_file=None,
            enable_metrics=True,
            enable_tracing=False,
        )

        from atlasrag.src.api.main import create_app

        test_app = create_app()
        client = TestClient(test_app)
        response = client.get("/metrics")
        assert response.status_code == 200
        # Prometheus text format should contain metric names
        assert "atlasrag_http_requests_total" in response.text
