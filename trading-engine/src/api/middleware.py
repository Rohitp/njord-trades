"""FastAPI middleware for request logging and correlation IDs."""

import time
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all HTTP requests with timing and correlation IDs."""

    async def dispatch(self, request: Request, call_next) -> Response:
        # Generate correlation ID
        correlation_id = str(uuid.uuid4())[:8]

        # Bind correlation ID to structlog context
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(correlation_id=correlation_id)

        log = structlog.get_logger()

        start_time = time.perf_counter()

        # Process request
        try:
            response = await call_next(request)
        except Exception as e:
            log.error(
                "request_failed",
                method=request.method,
                path=request.url.path,
                error=str(e),
            )
            raise

        # Calculate duration
        duration_ms = (time.perf_counter() - start_time) * 1000

        # Log request
        log.info(
            "request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=round(duration_ms, 2),
        )

        # Add correlation ID to response headers
        response.headers["X-Correlation-ID"] = correlation_id

        return response
