## ADDED Requirements

### Requirement: Provider interface defined
Each provider SHALL have:
- `base_url`: API endpoint URL
- `api_key`: Optional API key for authentication

#### Scenario: Provider config
- **WHEN** AI reads provider config
- **THEN** it can construct client using: base_url + api_key

### Requirement: Ollama provider uses OpenAI-compatible API
The system SHALL connect to Ollama using OpenAI-compatible `/chat/completions` endpoint.

#### Scenario: Ollama call
- **WHEN** model is `ollama/qwen3:14b`
- **THEN** use provider's base_url + /chat/completions

### Requirement: OpenAI-compatible providers use AsyncOpenAI
The system SHALL use `AsyncOpenAI` client for OpenAI, OpenRouter, Google, xAI, Groq.

#### Scenario: OpenAI call
- **WHEN** model uses OpenAI-compatible API
- **THEN** use AsyncOpenAI(client).chat.completions.create()

### Requirement: Fallback chain defined
The system SHALL try fallback models in order: model-level fallbacks → global fallbacks.

#### Scenario: Fallback
- **WHEN** primary model fails
- **THEN** try next model in fallback_models list