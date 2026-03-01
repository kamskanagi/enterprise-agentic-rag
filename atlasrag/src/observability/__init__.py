"""Observability Package

Structured logging, Prometheus metrics, and OpenTelemetry tracing.
"""

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

__all__ = [
    "JSONFormatter",
    "setup_logging",
    "REQUEST_COUNT",
    "REQUEST_LATENCY",
    "ACTIVE_QUERIES",
    "INGESTION_JOBS",
    "VERIFICATION_RESULTS",
    "get_metrics_app",
    "setup_tracing",
]
