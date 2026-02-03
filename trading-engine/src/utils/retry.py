"""Retry utility with exponential backoff for external API calls."""

import asyncio
import random
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import TypeVar

from src.config import settings
from src.utils.logging import get_logger

log = get_logger(__name__)

T = TypeVar("T")

# Network-level exceptions that should trigger retry
NETWORK_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    OSError,
)

# For LLM calls, also retry on rate limits and server errors
# LangChain wraps HTTP errors in various exception types
LLM_RETRYABLE_PATTERNS = (
    "rate limit",
    "429",
    "500",
    "502",
    "503",
    "504",
    "overloaded",
    "capacity",
)


def is_llm_retryable(exception: Exception) -> bool:
    """
    Check if an LLM exception should be retried.

    Checks for:
    - Network errors (ConnectionError, TimeoutError)
    - Rate limit errors (429)
    - Server errors (500, 502, 503, 504)
    - Overloaded/capacity errors

    Args:
        exception: The exception to check

    Returns:
        True if the exception should trigger a retry
    """
    # Check for network exceptions
    if isinstance(exception, NETWORK_EXCEPTIONS):
        return True

    # Check exception message for retryable patterns
    error_str = str(exception).lower()
    for pattern in LLM_RETRYABLE_PATTERNS:
        if pattern in error_str:
            return True

    return False


def retry_with_backoff(
    max_retries: int | None = None,
    backoff_base: float | None = None,
    backoff_max: float | None = None,
    retryable_exceptions: tuple[type[Exception], ...] | None = None,
    jitter: bool = True,
):
    """
    Decorator for async functions with exponential backoff retry logic.

    Per requirements: 3 attempts, exponential backoff (2s, 4s, 8s).

    Args:
        max_retries: Maximum retry attempts (defaults to config)
        backoff_base: Base delay in seconds (defaults to config)
        backoff_max: Maximum delay in seconds (defaults to config)
        retryable_exceptions: Tuple of exception types to retry (defaults to network errors)
        jitter: If True, adds random jitter to prevent thundering herd
    """
    max_retries = max_retries or settings.llm.max_retries
    backoff_base = backoff_base or settings.llm.retry_backoff_base
    backoff_max = backoff_max or settings.llm.retry_backoff_max
    retryable_exceptions = retryable_exceptions or NETWORK_EXCEPTIONS

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
                        # Add jitter: random value between 0 and delay
                        if jitter:
                            delay = delay + random.uniform(0, delay * 0.5)
                        log.warning(
                            "retry_attempt",
                            function=func.__name__,
                            attempt=attempt + 1,
                            max_retries=max_retries,
                            delay=round(delay, 2),
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


async def retry_llm_call(
    func: Callable[..., Awaitable[T]],
    *args,
    max_retries: int | None = None,
    context: str = "LLM call",
    **kwargs,
) -> T:
    """
    Execute an LLM call with retry logic.

    Unlike retry_with_backoff decorator, this function checks exception
    messages to determine if retry is appropriate (for rate limits, etc.).

    Args:
        func: Async function to call
        *args: Positional arguments to pass to func
        max_retries: Maximum retry attempts (defaults to config)
        context: Description for logging (e.g., "DataAgent LLM call")
        **kwargs: Keyword arguments to pass to func

    Returns:
        Result from func

    Raises:
        Last exception if all retries exhausted
    """
    max_retries = max_retries or settings.llm.max_retries
    backoff_base = settings.llm.retry_backoff_base
    backoff_max = settings.llm.retry_backoff_max

    last_exception = None

    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if is_llm_retryable(e):
                last_exception = e
                if attempt < max_retries - 1:
                    delay = min(backoff_base * (2 ** attempt), backoff_max)
                    # Add jitter
                    delay = delay + random.uniform(0, delay * 0.5)
                    log.warning(
                        "llm_retry_attempt",
                        context=context,
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        delay=round(delay, 2),
                        error=str(e)[:200],  # Truncate long errors
                        error_type=type(e).__name__,
                    )
                    await asyncio.sleep(delay)
                else:
                    log.error(
                        "llm_retry_exhausted",
                        context=context,
                        attempts=max_retries,
                        error=str(e)[:200],
                        error_type=type(e).__name__,
                    )
            else:
                # Non-retryable - raise immediately
                log.error(
                    "llm_non_retryable_error",
                    context=context,
                    error=str(e)[:200],
                    error_type=type(e).__name__,
                )
                raise

    if last_exception:
        raise last_exception

    raise RuntimeError(f"Unexpected retry state for {context}")

