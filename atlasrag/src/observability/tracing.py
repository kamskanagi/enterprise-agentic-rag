"""OpenTelemetry Tracing Setup

Configures TracerProvider with OTLP exporter and FastAPI auto-instrumentation.
No-op when tracing is disabled.
"""

import logging

from fastapi import FastAPI

from atlasrag.src.config.observability_config import ObservabilityConfig

logger = logging.getLogger(__name__)


def setup_tracing(app: FastAPI, config: ObservabilityConfig) -> None:
    """Initialize OpenTelemetry tracing if enabled.

    Args:
        app: The FastAPI application to instrument.
        config: ObservabilityConfig with enable_tracing flag.
    """
    if not config.enable_tracing:
        logger.info("Tracing disabled, skipping OpenTelemetry setup")
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create({"service.name": "atlasrag"})
        provider = TracerProvider(resource=resource)
        processor = BatchSpanProcessor(OTLPSpanExporter())
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)

        FastAPIInstrumentor.instrument_app(app)
        logger.info("OpenTelemetry tracing initialized")
    except ImportError:
        logger.warning(
            "OpenTelemetry packages not installed, skipping tracing setup"
        )
