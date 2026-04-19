"""Structured logging via structlog. Import ``get_logger`` in every module."""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from scholarpeer.config import get_settings

_CONFIGURED = False


def configure_logging() -> None:
    """Configure structlog once per process. Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings = get_settings()
    level = getattr(logging, settings.log_level.upper())

    logging.basicConfig(
        level=level,
        format="%(message)s",
        stream=sys.stderr,
    )

    processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if settings.log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True


def get_logger(name: str | None = None, **bound: Any) -> structlog.stdlib.BoundLogger:
    """Return a configured bound logger. Call from the top of every module."""
    configure_logging()
    logger = structlog.get_logger(name or __name__)
    return logger.bind(**bound) if bound else logger
