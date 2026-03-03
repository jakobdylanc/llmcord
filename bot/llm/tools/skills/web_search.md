---
name: web_search
description: Search the web for current information via Brave Search API.
metadata: {"clawhub":{"emoji":"🌐","tools":["web_search"]}}
---

# web_search

Use this tool to retrieve current information from the internet. The bot uses the Brave Search API.

## When to use web_search

- User asks for recent or real-time information (news, prices, weather, events)
- The answer may have changed since your training cutoff
- You need to verify a fact you're not confident about

## Tool signature

```
web_search(query: str)
```

## Usage examples

```
web_search(query="台北今日天氣預報")
web_search(query="Bitcoin price today")
```

## Result handling

- Returns a list of results, each with `title`, `url`, and snippet content
- Extract only the facts the user asked for — do NOT summarize what the website is
- Do NOT describe the source or tool used in your reply

## Notes

- Works with all providers (Ollama, OpenAI, OpenRouter, etc.). Requires BRAVE_API_KEY in .env.
- Search results may be in English even for Chinese queries; translate as needed
