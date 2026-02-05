"""
Tests for BaseAgent multi-provider LLM support.

Tests provider selection, fallback logic, and provider-specific LLM creation.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.agents.base import BaseAgent
from src.agents.data_agent import DataAgent
from src.config import settings
from src.workflows.state import TradingState


class TestProviderInference:
    """Test provider inference from model names."""

    def test_infer_openai_from_gpt_model(self):
        """Test that GPT models infer OpenAI provider."""
        agent = DataAgent()
        provider = agent._infer_provider("gpt-4o")
        assert provider == "openai"

    def test_infer_openai_from_o1_model(self):
        """Test that O1 models infer OpenAI provider."""
        agent = DataAgent()
        provider = agent._infer_provider("o1-preview")
        assert provider == "openai"

    def test_infer_anthropic_from_claude_model(self):
        """Test that Claude models infer Anthropic provider."""
        agent = DataAgent()
        provider = agent._infer_provider("claude-3-5-sonnet-20241022")
        assert provider == "anthropic"

    def test_infer_google_from_gemini_model(self):
        """Test that Gemini models infer Google provider."""
        agent = DataAgent()
        provider = agent._infer_provider("gemini-pro")
        assert provider == "google"

    def test_infer_deepseek_from_deepseek_model(self):
        """Test that DeepSeek models infer DeepSeek provider."""
        agent = DataAgent()
        provider = agent._infer_provider("deepseek-chat")
        assert provider == "deepseek"

    def test_infer_returns_none_for_ambiguous_model(self):
        """Test that ambiguous model names return None."""
        agent = DataAgent()
        provider = agent._infer_provider("unknown-model")
        assert provider is None


class TestProviderSelection:
    """Test provider selection priority."""

    def test_explicit_provider_override(self):
        """Test that explicit provider override takes highest priority."""
        agent = DataAgent(provider="anthropic")
        provider = agent._get_provider()
        assert provider == "anthropic"

    def test_agent_specific_provider_from_config(self):
        """Test that agent-specific config provider is used."""
        with patch.object(settings.llm, "data_agent_provider", "google"):
            agent = DataAgent()
            provider = agent._get_provider()
            assert provider == "google"

    def test_model_name_inference(self):
        """Test that model name inference is used when no explicit provider."""
        with patch.object(settings.llm, "data_agent_provider", "auto"):
            agent = DataAgent(model_name="gpt-4o")
            provider = agent._get_provider()
            assert provider == "openai"

    def test_default_provider_fallback(self):
        """Test that default provider is used when all else fails."""
        with patch.object(settings.llm, "data_agent_provider", "auto"):
            with patch.object(settings.llm, "default_provider", "anthropic"):
                agent = DataAgent(model_name="unknown-model")
                provider = agent._get_provider()
                assert provider == "anthropic"

    def test_auto_provider_uses_inference(self):
        """Test that 'auto' provider uses inference."""
        agent = DataAgent(provider="auto", model_name="claude-3-5-sonnet-20241022")
        provider = agent._get_provider()
        assert provider == "anthropic"


class TestProviderLLMCreation:
    """Test LLM creation for different providers."""

    def test_create_openai_llm(self):
        """Test OpenAI LLM creation."""
        with patch.object(settings.llm, "openai_api_key", "test-key"):
            with patch.object(settings.llm, "max_retries", 3):
                agent = DataAgent(provider="openai", model_name="gpt-4o")
                assert agent.llm is not None
                # Verify it's a ChatOpenAI instance
                from langchain_openai import ChatOpenAI
                assert isinstance(agent.llm, ChatOpenAI)

    def test_create_anthropic_llm(self):
        """Test Anthropic LLM creation."""
        with patch.object(settings.llm, "anthropic_api_key", "test-key"):
            with patch.object(settings.llm, "max_retries", 3):
                agent = DataAgent(provider="anthropic", model_name="claude-3-5-sonnet-20241022")
                assert agent.llm is not None
                from langchain_anthropic import ChatAnthropic
                assert isinstance(agent.llm, ChatAnthropic)

    def test_create_google_llm(self):
        """Test Google/Gemini LLM creation."""
        with patch("src.agents.base.ChatGoogleGenerativeAI") as mock_google:
            with patch.object(settings.llm, "google_api_key", "test-key"):
                with patch.object(settings.llm, "max_retries", 3):
                    agent = DataAgent(provider="google", model_name="gemini-pro")
                    assert agent.llm is not None
                    mock_google.assert_called_once()

    def test_create_deepseek_llm(self):
        """Test DeepSeek LLM creation (uses OpenAI-compatible API)."""
        with patch.object(settings.llm, "deepseek_api_key", "test-key"):
            with patch.object(settings.llm, "max_retries", 3):
                agent = DataAgent(provider="deepseek", model_name="deepseek-chat")
                assert agent.llm is not None
                from langchain_openai import ChatOpenAI
                assert isinstance(agent.llm, ChatOpenAI)
                # Verify openai_api_base is set for DeepSeek (ChatOpenAI uses openai_api_base, not base_url)
                assert agent.llm.openai_api_base == "https://api.deepseek.com/v1"

    def test_missing_api_key_raises_error_when_no_fallback(self):
        """Test that missing API key raises ValueError when fallback is same as primary."""
        with patch.object(settings.llm, "openai_api_key", ""):
            with patch.object(settings.llm, "fallback_provider", "openai"):  # Same as primary
                with pytest.raises(ValueError, match="OpenAI API key not configured"):
                    DataAgent(provider="openai")

    def test_unsupported_provider_raises_error(self):
        """Test that unsupported provider raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            agent = DataAgent()
            agent._create_llm_for_provider("unknown-provider")


class TestFallbackLogic:
    """Test fallback behavior when primary provider fails."""

    def test_fallback_on_missing_api_key(self):
        """Test that fallback is used when primary provider API key is missing."""
        with patch.object(settings.llm, "openai_api_key", ""):
            with patch.object(settings.llm, "anthropic_api_key", "test-key"):
                with patch.object(settings.llm, "default_provider", "openai"):
                    with patch.object(settings.llm, "fallback_provider", "anthropic"):
                        with patch.object(settings.llm, "max_retries", 3):
                            agent = DataAgent(provider="openai")
                            # Should fallback to Anthropic
                            assert agent.llm is not None
                            from langchain_anthropic import ChatAnthropic
                            assert isinstance(agent.llm, ChatAnthropic)

    def test_fallback_on_missing_package(self):
        """Test that fallback is used when Google package is not installed."""
        with patch("src.agents.base.ChatGoogleGenerativeAI", None):
            with patch.object(settings.llm, "google_api_key", "test-key"):
                with patch.object(settings.llm, "anthropic_api_key", "test-key"):
                    with patch.object(settings.llm, "fallback_provider", "anthropic"):
                        with patch.object(settings.llm, "max_retries", 3):
                            agent = DataAgent(provider="google")
                            # Should fallback to Anthropic
                            assert agent.llm is not None
                            from langchain_anthropic import ChatAnthropic
                            assert isinstance(agent.llm, ChatAnthropic)

    def test_no_fallback_if_fallback_provider_same_as_primary(self):
        """Test that error is raised if fallback is same as primary."""
        with patch.object(settings.llm, "openai_api_key", ""):
            with patch.object(settings.llm, "fallback_provider", "openai"):
                with pytest.raises(ValueError, match="OpenAI API key not configured"):
                    DataAgent(provider="openai")

    def test_fallback_logs_warning(self):
        """Test that fallback logs a warning."""
        with patch.object(settings.llm, "openai_api_key", ""):
            with patch.object(settings.llm, "anthropic_api_key", "test-key"):
                with patch.object(settings.llm, "default_provider", "openai"):
                    with patch.object(settings.llm, "fallback_provider", "anthropic"):
                        with patch.object(settings.llm, "max_retries", 3):
                            with patch("src.agents.base.log") as mock_log:
                                agent = DataAgent(provider="openai")
                                assert agent.llm is not None
                                # Verify warning was logged
                                mock_log.warning.assert_called_once()
                                call_args = mock_log.warning.call_args
                                assert "llm_provider_fallback" in str(call_args)


class TestPerAgentProviderConfig:
    """Test per-agent provider configuration."""

    def test_data_agent_provider_config(self):
        """Test DataAgent uses data_agent_provider config."""
        with patch.object(settings.llm, "data_agent_provider", "google"):
            with patch.object(settings.llm, "google_api_key", "test-key"):
                with patch.object(settings.llm, "max_retries", 3):
                    with patch("src.agents.base.ChatGoogleGenerativeAI") as mock_google:
                        agent = DataAgent()
                        assert agent.llm is not None
                        mock_google.assert_called_once()

    def test_risk_manager_provider_config(self):
        """Test RiskManager uses risk_agent_provider config."""
        from src.agents.risk_manager import RiskManager
        
        with patch.object(settings.llm, "risk_agent_provider", "anthropic"):
            with patch.object(settings.llm, "anthropic_api_key", "test-key"):
                with patch.object(settings.llm, "max_retries", 3):
                    agent = RiskManager()
                    assert agent.llm is not None
                    from langchain_anthropic import ChatAnthropic
                    assert isinstance(agent.llm, ChatAnthropic)

    def test_validator_provider_config(self):
        """Test Validator uses validator_provider config."""
        from src.agents.validator import Validator
        
        with patch.object(settings.llm, "validator_provider", "deepseek"):
            with patch.object(settings.llm, "deepseek_api_key", "test-key"):
                with patch.object(settings.llm, "max_retries", 3):
                    agent = Validator()
                    assert agent.llm is not None
                    from langchain_openai import ChatOpenAI
                    assert isinstance(agent.llm, ChatOpenAI)
                    assert agent.llm.openai_api_base == "https://api.deepseek.com/v1"

    def test_meta_agent_provider_config(self):
        """Test MetaAgent uses meta_agent_provider config."""
        from src.agents.meta_agent import MetaAgent
        
        with patch.object(settings.llm, "meta_agent_provider", "openai"):
            with patch.object(settings.llm, "openai_api_key", "test-key"):
                with patch.object(settings.llm, "max_retries", 3):
                    agent = MetaAgent()
                    assert agent.llm is not None
                    from langchain_openai import ChatOpenAI
                    assert isinstance(agent.llm, ChatOpenAI)

