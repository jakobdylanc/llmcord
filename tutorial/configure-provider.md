# Configure LLM Providers

This guide explains how to configure different LLM providers.

## Provider Types

| Provider | Type | API Key Required |
|----------|------|------------------|
| Ollama | Local | No |
| OpenAI | Remote | Yes |
| OpenRouter | Remote | Yes |
| Google | Remote | Yes |
| xAI | Remote | Yes |
| Groq | Remote | Yes |

## Basic Configuration

```yaml
providers:
  ollama:
    base_url: "http://localhost:11434"
  
  openrouter:
    base_url: "https://openrouter.ai"
    api_key: "your-openrouter-key"

models:
  ollama/qwen3:14b:
    # Local model
  
  openrouter/openrouter/free:
    # Free model via OpenRouter
```

## Ollama Setup

1. Install [Ollama](https://ollama.ai)
2. Pull a model: `ollama pull qwen3:14b`
3. Configure:
   ```yaml
   providers:
     ollama:
       base_url: "http://localhost:11434"
   
   models:
     ollama/qwen3:14b:
       tools: ["web_search"]
   ```

## OpenAI Setup

```yaml
providers:
  openai:
    base_url: "https://api.openai.com/v1"
    api_key: "sk-your-key"

models:
  openai/gpt-4o:
    tools: ["web_search", "visuals_core"]
```

## OpenRouter Setup

```yaml
providers:
  openrouter:
    base_url: "https://openrouter.ai"
    api_key: "your-key"

models:
  openrouter/anthropic/claude-3.5-sonnet:
    tools: ["web_search"]
```

## Fallback Models

Configure fallback models if the primary fails:

```yaml
models:
  openrouter/openrouter/free:
    fallback_models:
      - "ollama/qwen3:14b"
      - "openai/gpt-4o-mini"

# Or global fallback
fallback_models:
  - "ollama/qwen3:14b"
```

## Technical Details

For AI agents, see [Provider Spec](../openspec/specs/providers/spec.md) for full API details.