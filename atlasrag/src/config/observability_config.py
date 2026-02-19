"""Observability Configuration Model

Frozen BaseModel class for logging and tracing configuration.
"""

from typing import Literal
from pydantic import BaseModel


class ObservabilityConfig(BaseModel, frozen=True):
    """Observability settings for logging, tracing, and metrics."""

    enable_tracing: bool
    enable_metrics: bool
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    log_format: Literal["json", "text"]
