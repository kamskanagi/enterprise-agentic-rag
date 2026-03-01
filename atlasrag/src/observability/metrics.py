"""Prometheus Metrics

Defines application metrics and provides an ASGI app for the /metrics endpoint.
"""

from prometheus_client import Counter, Gauge, Histogram, make_asgi_app

# Request metrics
REQUEST_COUNT = Counter(
    "atlasrag_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)

REQUEST_LATENCY = Histogram(
    "atlasrag_http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["path"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

ACTIVE_QUERIES = Gauge(
    "atlasrag_active_queries",
    "Number of queries currently being processed",
)

# Ingestion metrics
INGESTION_JOBS = Counter(
    "atlasrag_ingestion_jobs_total",
    "Total ingestion jobs",
    ["status"],
)

# Verification metrics
VERIFICATION_RESULTS = Counter(
    "atlasrag_verification_results_total",
    "Total verification results",
    ["result"],
)


def get_metrics_app():
    """Return a Prometheus ASGI app for mounting at /metrics."""
    return make_asgi_app()
