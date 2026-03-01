"""API Middleware

Error handling middleware, request logging, and Prometheus metrics collection.
"""

import logging
import time

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from atlasrag.src.observability.metrics import ACTIVE_QUERIES, REQUEST_COUNT, REQUEST_LATENCY

logger = logging.getLogger(__name__)


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Catch unhandled exceptions and return JSON error responses."""

    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            logger.exception("Unhandled error on %s %s", request.method, request.url.path)
            return JSONResponse(
                status_code=500,
                content={
                    "detail": "Internal server error",
                    "error_type": type(e).__name__,
                },
            )


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log request method, path, and response time."""

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "%s %s → %d (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response


class MetricsMiddleware(BaseHTTPMiddleware):
    """Collect Prometheus metrics for HTTP requests."""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        # Skip metrics endpoint to avoid self-referential counting
        if path == "/metrics":
            return await call_next(request)

        ACTIVE_QUERIES.inc()
        start = time.perf_counter()
        try:
            response = await call_next(request)
            REQUEST_COUNT.labels(
                method=request.method,
                path=path,
                status=response.status_code,
            ).inc()
            REQUEST_LATENCY.labels(path=path).observe(time.perf_counter() - start)
            return response
        except Exception:
            REQUEST_COUNT.labels(
                method=request.method,
                path=path,
                status=500,
            ).inc()
            raise
        finally:
            ACTIVE_QUERIES.dec()
