---
name: web_search
description: Search the web and fetch URLs via Ollama's native web_search and web_fetch tools.
metadata: {"clawhub":{"emoji":"🌐","requires":{"provider":"ollama"},"tools":["web_search","web_fetch"]}}
---

# web_search / web_fetch

Use these tools to retrieve current information from the internet.
Both are Ollama-native — the Ollama server handles the actual HTTP request.

## When to use web_search

- User asks for recent or real-time information (news, prices, weather, events)
- The answer may have changed since your training cutoff
- You need to verify a fact you're not confident about

## When to use web_fetch

- User provides a specific URL and wants its content
- A web_search result returned a URL you need to read in full

## Tool signatures

```
web_search(query: str) -> WebSearchResponse
web_fetch(url: str) -> WebFetchResponse
```

## Usage examples

Search for recent news:
```
web_search(query="台北今日天氣預報")
web_search(query="Bitcoin price today")
```

Fetch a specific page:
```
web_fetch(url="https://example.com/article")
```

## Result handling

- **web_search**: returns a list of results, each with `title`, `url`, `content`
- **web_fetch**: returns `title`, `content`, and `links`
- Extract only the facts the user asked for — do NOT summarize what the website is
- Do NOT describe the source or tool used in your reply

## Notes

- Only works with Ollama provider — not available for OpenAI/OpenRouter models
- Search results may be in English even for Chinese queries; translate as needed
- `web_fetch` on large pages will be truncated to `max_tool_chars` (default 8000)
- Confirm before acting on fetched data (e.g. sending email based on calendar page)
