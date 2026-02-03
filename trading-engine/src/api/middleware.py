"""FastAPI middleware for request logging and trace IDs."""

import time
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all HTTP requests with timing and trace IDs."""

    async def dispatch(self, request: Request, call_next) -> Response:
        # Use X-Request-ID header if provided, otherwise generate one
        # This allows distributed tracing across services
        trace_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

        # Store trace_id in request state for access by route handlers
        request.state.trace_id = trace_id

        # Bind trace_id to structlog context - all logs in this request will include it
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(trace_id=trace_id)

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

        # Add trace ID to response headers for client correlation
        response.headers["X-Request-ID"] = trace_id

        return response
