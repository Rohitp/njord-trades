"""
Tests for LLMPicker multi-provider LLM support.

Tests provider selection, fallback logic, and provider-specific LLM creation for LLMPicker.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.services.discovery.pickers.llm import LLMPicker
from src.config import settings


class TestLLMPickerProviderInference:
    """Test provider inference for LLMPicker."""

    def test_infer_openai_from_gpt_model(self):
        """Test that GPT models infer OpenAI provider."""
        picker = LLMPicker(model_name="gpt-4o-mini")
        provider = picker._infer_provider("gpt-4o-mini")
        assert provider == "openai"

    def test_infer_anthropic_from_claude_model(self):
        """Test that Claude models infer Anthropic provider."""
        picker = LLMPicker(model_name="claude-3-5-sonnet-20241022")
        provider = picker._infer_provider("claude-3-5-sonnet-20241022")
        assert provider == "anthropic"

    def test_infer_google_from_gemini_model(self):
        """Test that Gemini models infer Google provider."""
        picker = LLMPicker()
        provider = picker._infer_provider("gemini-pro")
        assert provider == "google"

    def test_infer_deepseek_from_deepseek_model(self):
        """Test that DeepSeek models infer DeepSeek provider."""
        picker = LLMPicker()
        provider = picker._infer_provider("deepseek-chat")
        assert provider == "deepseek"

    def test_infer_returns_none_for_ambiguous_model(self):
        """Test that ambiguous model names return None."""
        picker = LLMPicker()
        provider = picker._infer_provider("unknown-model")
        assert provider is None


class TestLLMPickerProviderSelection:
    """Test provider selection priority for LLMPicker."""

    def test_explicit_provider_override(self):
        """Test that explicit provider override takes highest priority."""
        picker = LLMPicker(provider="anthropic")
        provider = picker._get_provider()
        assert provider == "anthropic"

    def test_llm_picker_provider_from_config(self):
        """Test that llm_picker_provider config is used."""
        with patch.object(settings.llm, "llm_picker_provider", "google"):
            picker = LLMPicker()
            provider = picker._get_provider()
            assert provider == "google"

    def test_model_name_inference(self):
        """Test that model name inference is used when no explicit provider."""
        with patch.object(settings.llm, "llm_picker_provider", "auto"):
            picker = LLMPicker(model_name="gpt-4o-mini")
            provider = picker._get_provider()
            assert provider == "openai"

    def test_default_provider_fallback(self):
        """Test that default provider is used when all else fails."""
        with patch.object(settings.llm, "llm_picker_provider", "auto"):
            with patch.object(settings.llm, "default_provider", "anthropic"):
                picker = LLMPicker(model_name="unknown-model")
                provider = picker._get_provider()
                assert provider == "anthropic"

    def test_auto_provider_uses_inference(self):
        """Test that 'auto' provider uses inference."""
        picker = LLMPicker(provider="auto", model_name="claude-3-5-sonnet-20241022")
        provider = picker._get_provider()
        assert provider == "anthropic"


class TestLLMPickerProviderLLMCreation:
    """Test LLM creation for different providers in LLMPicker."""

    def test_create_openai_llm(self):
        """Test OpenAI LLM creation."""
        with patch.object(settings.llm, "openai_api_key", "test-key"):
            with patch.object(settings.llm, "max_retries", 3):
                picker = LLMPicker(provider="openai", model_name="gpt-4o-mini")
                assert picker.llm is not None
                from langchain_openai import ChatOpenAI
                assert isinstance(picker.llm, ChatOpenAI)

    def test_create_anthropic_llm(self):
        """Test Anthropic LLM creation."""
        with patch.object(settings.llm, "anthropic_api_key", "test-key"):
            with patch.object(settings.llm, "max_retries", 3):
                picker = LLMPicker(provider="anthropic", model_name="claude-3-5-sonnet-20241022")
                assert picker.llm is not None
                from langchain_anthropic import ChatAnthropic
                assert isinstance(picker.llm, ChatAnthropic)

    def test_create_google_llm(self):
        """Test Google/Gemini LLM creation."""
        with patch("src.services.discovery.pickers.llm.ChatGoogleGenerativeAI") as mock_google:
            with patch.object(settings.llm, "google_api_key", "test-key"):
                with patch.object(settings.llm, "max_retries", 3):
                    picker = LLMPicker(provider="google", model_name="gemini-pro")
                    assert picker.llm is not None
                    mock_google.assert_called_once()

    def test_create_deepseek_llm(self):
        """Test DeepSeek LLM creation (uses OpenAI-compatible API)."""
        with patch.object(settings.llm, "deepseek_api_key", "test-key"):
            with patch.object(settings.llm, "max_retries", 3):
                picker = LLMPicker(provider="deepseek", model_name="deepseek-chat")
                assert picker.llm is not None
                from langchain_openai import ChatOpenAI
                assert isinstance(picker.llm, ChatOpenAI)
                # Verify openai_api_base is set for DeepSeek
                assert picker.llm.openai_api_base == "https://api.deepseek.com/v1"

    def test_missing_api_key_falls_back(self):
        """Test that missing API key triggers fallback (not error)."""
        with patch.object(settings.llm, "openai_api_key", ""):
            with patch.object(settings.llm, "anthropic_api_key", "test-key"):
                with patch.object(settings.llm, "default_provider", "openai"):
                    with patch.object(settings.llm, "fallback_provider", "anthropic"):
                        with patch.object(settings.llm, "max_retries", 3):
                            # Should fallback to Anthropic, not raise error
                            picker = LLMPicker(provider="openai")
                            assert picker.llm is not None
                            from langchain_anthropic import ChatAnthropic
                            assert isinstance(picker.llm, ChatAnthropic)

    def test_unsupported_provider_raises_error(self):
        """Test that unsupported provider raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            picker = LLMPicker()
            picker._create_llm_for_provider("unknown-provider")


class TestLLMPickerFallbackLogic:
    """Test fallback behavior for LLMPicker."""

    def test_fallback_on_missing_api_key(self):
        """Test that fallback is used when primary provider API key is missing."""
        with patch.object(settings.llm, "openai_api_key", ""):
            with patch.object(settings.llm, "anthropic_api_key", "test-key"):
                with patch.object(settings.llm, "default_provider", "openai"):
                    with patch.object(settings.llm, "fallback_provider", "anthropic"):
                        with patch.object(settings.llm, "max_retries", 3):
                            picker = LLMPicker(provider="openai")
                            # Should fallback to Anthropic
                            assert picker.llm is not None
                            from langchain_anthropic import ChatAnthropic
                            assert isinstance(picker.llm, ChatAnthropic)

    def test_fallback_logs_warning(self):
        """Test that fallback logs a warning."""
        with patch.object(settings.llm, "openai_api_key", ""):
            with patch.object(settings.llm, "anthropic_api_key", "test-key"):
                with patch.object(settings.llm, "default_provider", "openai"):
                    with patch.object(settings.llm, "fallback_provider", "anthropic"):
                        with patch.object(settings.llm, "max_retries", 3):
                            with patch("src.services.discovery.pickers.llm.log") as mock_log:
                                picker = LLMPicker(provider="openai")
                                assert picker.llm is not None
                                # Verify warning was logged
                                mock_log.warning.assert_called_once()
                                call_args = mock_log.warning.call_args
                                assert "llm_picker_provider_fallback" in str(call_args)

