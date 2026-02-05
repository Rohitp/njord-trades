"""
Unit tests for Langfuse integration utilities.
"""

import builtins
import pytest
from unittest.mock import MagicMock, patch, Mock

from src.utils.langfuse import (
    get_langfuse_client,
    langfuse_trace,
    langfuse_span,
    langfuse_generation,
)


class TestLangfuseClient:
    """Test Langfuse client initialization."""

    def test_client_not_created_when_disabled(self):
        """Test that client is None when tracing is disabled."""
        with patch("src.utils.langfuse.settings") as mock_settings:
            mock_settings.langfuse.tracing_enabled = False
            
            client = get_langfuse_client()
            
            assert client is None

    def test_client_created_when_enabled(self):
        """Test that client is created when tracing is enabled."""
        with patch("src.utils.langfuse.settings") as mock_settings:
            mock_settings.langfuse.tracing_enabled = True
            mock_settings.langfuse.public_key = "test-key"
            mock_settings.langfuse.secret_key = "test-secret"
            mock_settings.langfuse.host = "https://cloud.langfuse.com"
            
            with patch("langfuse.Langfuse") as mock_langfuse_class:
                mock_client = MagicMock()
                mock_langfuse_class.return_value = mock_client
                
                # Reset the global client cache
                import src.utils.langfuse
                src.utils.langfuse._langfuse_client = None
                
                client = get_langfuse_client()
                
                assert client is not None
                mock_langfuse_class.assert_called_once()

    def test_client_handles_import_error(self):
        """Test that ImportError is handled gracefully."""
        with patch("src.utils.langfuse.settings") as mock_settings:
            mock_settings.langfuse.tracing_enabled = True
            mock_settings.langfuse.public_key = "test-key"
            mock_settings.langfuse.secret_key = "test-secret"
            mock_settings.langfuse.host = "https://cloud.langfuse.com"
            
            # Reset the global client cache
            import src.utils.langfuse
            src.utils.langfuse._langfuse_client = None
            
            # Test ImportError handling by temporarily making the import fail
            import sys
            
            # Save original langfuse module if it exists
            original_langfuse = sys.modules.get("langfuse")
            langfuse_installed = "langfuse" in sys.modules
            
            # Remove langfuse from sys.modules to force re-import
            if langfuse_installed:
                del sys.modules["langfuse"]
                # Also remove any submodules
                keys_to_remove = [k for k in list(sys.modules.keys()) if k.startswith("langfuse.")]
                for key in keys_to_remove:
                    del sys.modules[key]
            
            # Create a mock __import__ that raises ImportError for langfuse
            original_import = __import__
            def mock_import(name, *args, **kwargs):
                if name == "langfuse":
                    raise ImportError(f"No module named '{name}'")
                return original_import(name, *args, **kwargs)
            
            try:
                with patch("builtins.__import__", side_effect=mock_import):
                    client = get_langfuse_client()
                    assert client is None
            finally:
                # Restore original langfuse module
                if original_langfuse:
                    sys.modules["langfuse"] = original_langfuse

    def test_client_handles_init_error(self):
        """Test that initialization errors are handled gracefully."""
        with patch("src.utils.langfuse.settings") as mock_settings:
            mock_settings.langfuse.tracing_enabled = True
            mock_settings.langfuse.public_key = "test-key"
            mock_settings.langfuse.secret_key = "test-secret"
            mock_settings.langfuse.host = "https://cloud.langfuse.com"
            
            # Reset the global client cache
            import src.utils.langfuse
            src.utils.langfuse._langfuse_client = None
            
            with patch("langfuse.Langfuse", side_effect=Exception("Init error")):
                client = get_langfuse_client()
                
                assert client is None


class TestLangfuseTrace:
    """Test langfuse_trace context manager."""

    def test_trace_works_when_enabled(self):
        """Test that trace context manager works when Langfuse is enabled."""
        mock_client = MagicMock()
        mock_trace = MagicMock()
        mock_client.trace.return_value = mock_trace
        
        with patch("src.utils.langfuse.get_langfuse_client", return_value=mock_client):
            with langfuse_trace("test_trace", metadata={"key": "value"}) as trace:
                assert trace is not None
                mock_client.trace.assert_called_once_with(
                    name="test_trace",
                    metadata={"key": "value"},
                    user_id=None,
                    session_id=None,
                )

    def test_trace_none_when_disabled(self):
        """Test that trace is None when Langfuse is disabled."""
        with patch("src.utils.langfuse.get_langfuse_client", return_value=None):
            with langfuse_trace("test_trace") as trace:
                assert trace is None

    def test_trace_handles_errors(self):
        """Test that trace errors are handled gracefully."""
        mock_client = MagicMock()
        mock_client.trace.side_effect = Exception("Trace error")
        
        with patch("src.utils.langfuse.get_langfuse_client", return_value=mock_client):
            with langfuse_trace("test_trace") as trace:
                assert trace is None  # Should return None on error


class TestLangfuseSpan:
    """Test langfuse_span context manager."""

    def test_span_works_with_trace(self):
        """Test that span context manager works with a trace."""
        mock_trace = MagicMock()
        mock_span = MagicMock()
        mock_trace.span.return_value = mock_span
        
        with langfuse_span(mock_trace, "test_span", metadata={"key": "value"}) as span:
            assert span is not None
            mock_trace.span.assert_called_once_with(
                name="test_span",
                metadata={"key": "value"},
            )

    def test_span_none_when_trace_none(self):
        """Test that span is None when trace is None."""
        with langfuse_span(None, "test_span") as span:
            assert span is None

    def test_span_handles_errors(self):
        """Test that span errors are handled gracefully."""
        mock_trace = MagicMock()
        mock_trace.span.side_effect = Exception("Span error")
        
        with langfuse_span(mock_trace, "test_span") as span:
            assert span is None  # Should return None on error


class TestLangfuseGeneration:
    """Test langfuse_generation function."""

    def test_generation_logs_to_trace(self):
        """Test that generation is logged to trace."""
        mock_trace = MagicMock()
        
        langfuse_generation(
            trace=mock_trace,
            name="test_generation",
            model="gpt-4",
            input_messages=[{"role": "user", "content": "test"}],
            output="response",
            metadata={"key": "value"},
        )
        
        mock_trace.generation.assert_called_once_with(
            name="test_generation",
            model="gpt-4",
            input=[{"role": "user", "content": "test"}],
            output="response",
            metadata={"key": "value"},
        )

    def test_generation_handles_none_trace(self):
        """Test that generation handles None trace gracefully."""
        # Should not raise error
        langfuse_generation(
            trace=None,
            name="test_generation",
            model="gpt-4",
            input_messages=[],
            output="response",
        )

    def test_generation_handles_errors(self):
        """Test that generation errors are handled gracefully."""
        mock_trace = MagicMock()
        mock_trace.generation.side_effect = Exception("Generation error")
        
        # Should not raise error
        langfuse_generation(
            trace=mock_trace,
            name="test_generation",
            model="gpt-4",
            input_messages=[],
            output="response",
        )

