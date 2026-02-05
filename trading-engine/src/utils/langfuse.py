"""
Langfuse integration for LLM tracing.

Provides decorators and utilities to instrument LLM calls with Langfuse tracing.
"""

from contextlib import contextmanager
from typing import Any

from src.config import settings
from src.utils.logging import get_logger

log = get_logger(__name__)

# Lazy import - only import if configured
_langfuse_client = None


def get_langfuse_client():
    """Get or create Langfuse client (lazy initialization)."""
    global _langfuse_client
    
    if _langfuse_client is not None:
        return _langfuse_client
    
    if not settings.langfuse.tracing_enabled:
        return None
    
    try:
        from langfuse import Langfuse
        
        # For self-hosted Langfuse, public_key/secret_key are optional
        # For cloud Langfuse, they are required
        is_self_hosted = settings.langfuse.host.startswith("http://localhost") or settings.langfuse.host.startswith("http://127.0.0.1")
        if not is_self_hosted and not settings.langfuse.public_key:
            log.warning("langfuse_not_configured", reason="Missing public_key for cloud Langfuse")
            return None
        
        # For self-hosted, keys can be empty
        # For cloud, keys are required (checked above)
        _langfuse_client = Langfuse(
            public_key=settings.langfuse.public_key or "pk-lf-0000000000000000000000",  # Dummy key for self-hosted
            secret_key=settings.langfuse.secret_key or "sk-lf-0000000000000000000000",  # Dummy key for self-hosted
            host=settings.langfuse.host,
        )
        log.info("langfuse_initialized", host=settings.langfuse.host)
        return _langfuse_client
    except ImportError:
        log.warning("langfuse_not_installed", reason="langfuse package not installed")
        return None
    except Exception as e:
        log.error("langfuse_init_failed", error=str(e), exc_info=True)
        return None


@contextmanager
def langfuse_trace(
    name: str,
    metadata: dict[str, Any] | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
):
    """
    Context manager for Langfuse tracing.
    
    Usage:
        with langfuse_trace("agent_call", metadata={"agent": "DataAgent"}):
            result = await llm.ainvoke(messages)
    
    Args:
        name: Trace name
        metadata: Additional metadata
        user_id: User ID (optional)
        session_id: Session ID (optional, e.g., cycle_id)
    """
    client = get_langfuse_client()
    
    if client is None:
        # Langfuse not configured - yield without tracing
        yield None
        return
    
    try:
        trace = client.trace(
            name=name,
            metadata=metadata or {},
            user_id=user_id,
            session_id=session_id,
        )
        yield trace
    except Exception as e:
        log.error("langfuse_trace_error", error=str(e), exc_info=True)
        yield None


@contextmanager
def langfuse_span(
    trace,
    name: str,
    metadata: dict[str, Any] | None = None,
):
    """
    Context manager for Langfuse span within a trace.
    
    Usage:
        with langfuse_trace("agent_call") as trace:
            with langfuse_span(trace, "llm_call", metadata={"model": "gpt-4"}):
                result = await llm.ainvoke(messages)
    
    Args:
        trace: Langfuse trace object
        name: Span name
        metadata: Additional metadata
    """
    if trace is None:
        yield None
        return
    
    try:
        span = trace.span(
            name=name,
            metadata=metadata or {},
        )
        yield span
    except Exception as e:
        log.error("langfuse_span_error", error=str(e), exc_info=True)
        yield None


def langfuse_generation(
    trace,
    name: str,
    model: str,
    input_messages: list[dict[str, Any]],
    output: str,
    metadata: dict[str, Any] | None = None,
):
    """
    Log an LLM generation to Langfuse.
    
    Args:
        trace: Langfuse trace object
        name: Generation name
        model: Model name
        input_messages: Input messages (list of dicts with "role" and "content")
        output: LLM output text
        metadata: Additional metadata
    """
    if trace is None:
        return
    
    try:
        trace.generation(
            name=name,
            model=model,
            input=input_messages,
            output=output,
            metadata=metadata or {},
        )
    except Exception as e:
        log.error("langfuse_generation_error", error=str(e), exc_info=True)

