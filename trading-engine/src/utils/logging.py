"""Structured logging configuration using structlog."""

import logging
import sys

import structlog

from src.config import settings


def setup_logging() -> None:
    """Configure structlog and route all logging through it."""

    # Shared processors for structlog-native loggers
    shared_processors: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if settings.logging.format == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    # Custom filter_by_level that handles None loggers (APScheduler compatibility)
    def safe_filter_by_level(logger, method_name, event_dict):
        """Filter by log level, handling None loggers gracefully."""
        if logger is None:
            # If logger is None, don't filter (allow through)
            return event_dict
        return structlog.stdlib.filter_by_level(logger, method_name, event_dict)

    # Processors for stdlib logging (uvicorn, sqlalchemy, etc.)
    stdlib_processors = [
        safe_filter_by_level,  # Use safe version instead of structlog.stdlib.filter_by_level
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ]

    # Configure structlog
    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure stdlib logging to use structlog formatter
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=stdlib_processors[:-1],
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, settings.logging.level.upper()))

    # Quiet noisy loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    
    # Configure APScheduler logger to work with structlog
    # APScheduler uses standard logging - let it propagate to root logger
    apscheduler_logger = logging.getLogger("apscheduler")
    apscheduler_logger.setLevel(logging.INFO)
    apscheduler_logger.propagate = True  # Propagate to root logger (don't add handler directly)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a configured logger instance."""
    return structlog.get_logger(name)
