# Provider Spec

## Supported Providers

| Provider | Key Format | API Type | Config |
|----------|------------|----------|--------|
| Ollama | `ollama/model:version` | OpenAI-compatible | base_url (e.g. http://localhost:11434) |
| OpenAI | `openai/model` | OpenAI API | api_key required |
| OpenRouter | `openrouter/provider/model` | OpenAI-compatible | api_key required |
| Google | `google/model` | Google AI | api_key required |
| xAI | `xai/model` | OpenAI-compatible | api_key required |
| Groq | `groq/model` | OpenAI-compatible | api_key required |

## Provider Interface

```python
# Each provider config:
{
    "base_url": str,  # API endpoint
    "api_key": str,   # Optional
}
```

## Fallback Chain

1. Model-level `fallback_models` (first checked)
2. Global `fallback_models` (second checked)
3. Return error if all fail

## Timeout Handling

- Per-model `RESPONSE_TIMEOUT_SECONDS` in config
- On timeout: try next fallback model
- On Ollama 500 with tools: retry without tools (for qwen3 think mode)

## Tool Calling

- Ollama: Uses `/chat/completions` with `tools` param
- OpenAI-compatible: Uses `AsyncOpenAI` with tool_call loop
- Tools executed bot-side (not sent to provider)