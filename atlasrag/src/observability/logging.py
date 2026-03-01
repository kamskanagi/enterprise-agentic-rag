"""Structured Logging Setup

JSON-formatted and text-formatted logging configuration for AtlasRAG.
"""

import json
import logging
import sys
from datetime import datetime, timezone

from atlasrag.src.config.observability_config import ObservabilityConfig


class JSONFormatter(logging.Formatter):
    """Logging formatter that outputs JSON lines."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "extra_data"):
            log_entry["extra"] = record.extra_data
        return json.dumps(log_entry)


def setup_logging(config: ObservabilityConfig) -> None:
    """Configure the root logger based on observability config.

    Args:
        config: ObservabilityConfig with log_level and log_format settings.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.log_level))

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)

    if config.log_format == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    root_logger.addHandler(handler)
