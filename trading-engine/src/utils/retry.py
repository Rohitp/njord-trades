"""Retry utility with exponential backoff for external API calls."""

import asyncio
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import TypeVar

from src.config import settings
from src.utils.logging import get_logger

log = get_logger(__name__)

T = TypeVar("T")

# Exceptions that should trigger retry
# Note: Alpaca SDK may raise various exceptions - we catch base Exception
# and check for network-related errors. API errors (400, 401, etc.) are not retried.
RETRYABLE_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    OSError,
)


def retry_with_backoff(
    max_retries: int | None = None,
    backoff_base: float | None = None,
    backoff_max: float | None = None,
    retryable_exceptions: tuple[type[Exception], ...] | None = None,
):
    """
    Decorator for async functions with exponential backoff retry logic.
    
    Per requirements: 3 attempts, exponential backoff (2s, 4s, 8s).
    
    Args:
        max_retries: Maximum retry attempts (defaults to config)
        backoff_base: Base delay in seconds (defaults to config)
        backoff_max: Maximum delay in seconds (defaults to config)
        retryable_exceptions: Tuple of exception types to retry (defaults to network errors)
    """
    max_retries = max_retries or settings.llm.max_retries
    backoff_base = backoff_base or settings.llm.retry_backoff_base
    backoff_max = backoff_max or settings.llm.retry_backoff_max
    retryable_exceptions = retryable_exceptions or RETRYABLE_EXCEPTIONS

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        delay = min(backoff_base * (2 ** attempt), backoff_max)
                        log.warning(
                            "retry_attempt",
                            function=func.__name__,
                            attempt=attempt + 1,
                            max_retries=max_retries,
                            delay=delay,
                            error=str(e),
                        )
                        await asyncio.sleep(delay)
                    else:
                        log.error(
                            "retry_exhausted",
                            function=func.__name__,
                            attempts=max_retries,
                            error=str(e),
                        )
                except Exception as e:
                    # Non-retryable exception - raise immediately
                    log.error(
                        "non_retryable_error",
                        function=func.__name__,
                        error=str(e),
                        error_type=type(e).__name__,
                    )
                    raise
            
            # If we exhausted retries, raise the last exception
            if last_exception:
                raise last_exception
            
            # Should never reach here, but just in case
            raise RuntimeError(f"Unexpected retry state for {func.__name__}")
        
        return wrapper
    return decorator

