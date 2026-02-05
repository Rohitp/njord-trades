# Multi-Provider LLM Setup

## Overview

The system now supports **4 LLM providers** with automatic fallback and per-component provider selection:

- **OpenAI** (primary, default)
- **Anthropic** (fallback)
- **Google Gemini** (experimental)
- **DeepSeek** (experimental)

## Configuration

### Environment Variables

Add to your `.env` file:

```bash
# Primary provider (OpenAI)
LLM_OPENAI_API_KEY=sk-...

# Fallback provider (Anthropic)
LLM_ANTHROPIC_API_KEY=sk-ant-...

# Optional providers for experimentation
LLM_GOOGLE_API_KEY=...  # For Gemini
LLM_DEEPSEEK_API_KEY=...  # For DeepSeek

# Provider selection (optional - defaults to "auto")
LLM_DEFAULT_PROVIDER=openai
LLM_FALLBACK_PROVIDER=anthropic

# Per-component provider selection (for experimentation)
LLM_DATA_AGENT_PROVIDER=openai  # or "anthropic", "google", "deepseek", "auto"
LLM_RISK_AGENT_PROVIDER=openai
LLM_VALIDATOR_PROVIDER=openai
LLM_META_AGENT_PROVIDER=openai
LLM_LLM_PICKER_PROVIDER=openai  # For LLMPicker in discovery
```

### Default Models

**OpenAI (default)**:
- Data/Risk agents: `gpt-4o-mini` (fast, cost-effective)
- Validator/Meta-Agent: `gpt-4o` (higher quality)
- LLM Picker: `gpt-4o-mini`

**Anthropic (fallback)**:
- Data/Risk/Validator: `claude-3-5-sonnet-20241022`
- Meta-Agent: `claude-3-opus-20240229`

**Google Gemini**:
- Use model names like `gemini-pro`, `gemini-ultra`

**DeepSeek**:
- Use model names like `deepseek-chat`, `deepseek-coder`

## How It Works

### Provider Selection Priority

1. **Explicit override** (if provided in code)
2. **Per-component config** (e.g., `LLM_DATA_AGENT_PROVIDER`)
3. **Model name inference** (e.g., `gpt-4o` → OpenAI, `claude-3-5-sonnet` → Anthropic)
4. **Default provider** (`LLM_DEFAULT_PROVIDER`)

### Automatic Fallback

If the primary provider fails (API error, rate limit, etc.), the system automatically falls back to `LLM_FALLBACK_PROVIDER` (default: Anthropic).

Example:
- Primary: OpenAI → fails → Fallback: Anthropic

## Experimentation Setup

### Compare Providers for Metrics/Fuzzy/LLM Pickers

To compare different providers for discovery pickers:

```bash
# Test 1: OpenAI
LLM_LLM_PICKER_PROVIDER=openai
LLM_LLM_PICKER_MODEL=gpt-4o-mini

# Test 2: Anthropic
LLM_LLM_PICKER_PROVIDER=anthropic
LLM_LLM_PICKER_MODEL=claude-3-5-sonnet-20241022

# Test 3: Gemini
LLM_LLM_PICKER_PROVIDER=google
LLM_LLM_PICKER_MODEL=gemini-pro

# Test 4: DeepSeek
LLM_LLM_PICKER_PROVIDER=deepseek
LLM_LLM_PICKER_MODEL=deepseek-chat
```

Run discovery cycles and compare:
- Symbol selection quality
- Response time
- Cost per call
- Reasoning quality

### Compare Providers for Trading Agents

Test different providers for each agent:

```bash
# Test OpenAI for all agents
LLM_DATA_AGENT_PROVIDER=openai
LLM_RISK_AGENT_PROVIDER=openai
LLM_VALIDATOR_PROVIDER=openai
LLM_META_AGENT_PROVIDER=openai

# Or mix and match
LLM_DATA_AGENT_PROVIDER=openai  # Fast, cheap
LLM_VALIDATOR_PROVIDER=anthropic  # Better reasoning
LLM_META_AGENT_PROVIDER=openai  # Fast decision-making
```

## Installation

### Required Packages

```bash
cd trading-engine
uv sync
```

This installs:
- `langchain-openai` (OpenAI)
- `langchain-anthropic` (Anthropic)
- `langchain-google-genai` (Gemini)

**Note**: DeepSeek uses OpenAI-compatible API, so it works with `langchain-openai` (just change the `base_url`).

## Model Name Patterns

The system automatically infers provider from model name:

- **OpenAI**: `gpt-*`, `o1-*`, `o3-*`
- **Anthropic**: `claude-*`
- **Google**: `gemini-*`
- **DeepSeek**: `deepseek-*`

If ambiguous, uses `LLM_DEFAULT_PROVIDER`.

## Logging

Provider usage is logged in debug logs:

```json
{
  "event": "llm_call_complete",
  "agent": "DataAgent",
  "model": "gpt-4o-mini",
  "provider": "openai",
  "response_length": 1234
}
```

## Troubleshooting

### "Unsupported LLM provider"
- Check provider name is lowercase: `"openai"`, not `"OpenAI"`
- Valid options: `"openai"`, `"anthropic"`, `"google"`, `"deepseek"`, `"auto"`

### "API key not configured"
- Make sure the API key environment variable is set
- Check `.env` file is loaded (restart app after changes)

### "langchain-google-genai not installed"
- Run: `uv sync` (package is in dependencies)

### Fallback not working
- Check `LLM_FALLBACK_PROVIDER` is set to a provider with valid API key
- Check logs for fallback warnings

## Example: A/B Testing Providers

1. **Set up two test runs**:
   ```bash
   # Run 1: OpenAI
   LLM_LLM_PICKER_PROVIDER=openai
   # Run discovery cycle, save results
   
   # Run 2: Anthropic
   LLM_LLM_PICKER_PROVIDER=anthropic
   # Run discovery cycle, save results
   ```

2. **Compare results**:
   - Check `/api/discovery/performance` for picker metrics
   - Compare symbol selection quality
   - Check logs for response times
   - Calculate cost per call

3. **Choose winner** based on:
   - Win rate
   - Average return
   - Cost efficiency
   - Response time

## Next Steps

- Add metrics tracking per provider (Phase 9: Observability)
- Store provider used in database for analysis
- Create comparison dashboard in Grafana

