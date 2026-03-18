## Architecture

```
get_weather(city: str) → str
         │
         ▼
┌─────────────────┐
│ _detect_language│ ──► Detect from Unicode (zh/ja/ko/ru/etc.)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ _get_city_coords│ ──► Open-Meteo Geocoding API
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  _fetch_weather │ ──► Open-Meteo Weather API
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Format +     │
│    Return       │
└─────────────────┘
```

## Related Words System (Hybrid Approach)

To solve the problem of models not following tool instructions (hallucinating instead of calling tools), we use a **hybrid approach** with `related_words` (formerly called "keywords"):

```
┌─────────────────────────────────────────────────────────────────┐
│                    RELATED WORDS FLOW                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Tool Definition (weather.py)                             │  │
│  │                                                          │  │
│  │  TOOL_ENTRY = ToolEntry(                                 │  │
│  │      schema=WEATHER_SCHEMA,                              │  │
│  │      fn=get_weather,                                     │  │
│  │      keywords=[                                           │  │
│  │          "天氣", "weather", "temperature", "forecast",   │  │
│  │          "氣溫", "降雨", "晴天", "雨天", ...             │  │
│  │      ]                                                    │  │
│  │  )                                                        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                               │                                 │
│              ┌────────────────┼────────────────┐               │
│              ▼                ▼                ▼               │
│     ┌─────────────┐   ┌──────────────┐  ┌──────────────┐       │
│     │ Forced      │   │ Skill Doc    │  │ Future:      │       │
│     │ Execution   │   │ Injection    │  │ Auto-detect  │       │
│     │ (fallback)  │   │ (guidance)   │  │ new tools    │       │
│     │             │   │              │  │              │       │
│     │ If keyword  │   │ Append to    │  │              │       │
│     │ found →     │   │ skill docs:  │  │              │       │
│     │ execute     │   │ "Use         │  │              │       │
│     │ tool +      │   │  get_weather │  │              │       │
│     │ inject      │   │  when user   │  │              │       │
│     │ result      │   │  asks about  │  │              │       │
│     └─────────────┘   │  天氣,天氣"   │  └──────────────┘       │
│                       └──────────────┘                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Why This Approach?

| Aspect | Description |
|--------|-------------|
| **Dynamic** | No hardcoded keywords in llmcord.py - each tool defines its own |
| **Fallback** | When model doesn't call tools, we detect and pre-execute |
| **Guidance** | Also injects into skill docs for models that DO follow instructions |
| **Extensible** | New tools automatically get related words support |
| **Flexible** | Fuzzy/partial matching handles variations |

### Implementation

1. **ToolEntry dataclass** gets `keywords: list[str]` field
2. **Each tool** defines its own keywords
3. **llmcord.py** reads keywords from registry dynamically
4. **Skill docs** can optionally use keywords for better prompting

## Components

### 1. Language Detection (`_detect_language`)

```python
def _detect_language(text: str) -> str:
    """Detect language from text using Unicode character ranges."""
    if re.search(r'[\u4e00-\u9fff]', text):  # Chinese
        return "zh"
    if re.search(r'[\u3040-\u30ff\u31f0-\u31ff]', text):  # Japanese
        return "ja"
    if re.search(r'[\uac00-\ud7af]', text):  # Korean
        return "ko"
    if re.search(r'[\u0400-\u04ff]', text):  # Cyrillic
        return "ru"
    return "en"  # Default
```

### 2. Geocoding (`_get_city_coords`)

- Endpoint: `https://geocoding-api.open-meteo.com/v1/search`
- Parameters: `name`, `count=1`, `language={detected}`, `format=json`
- Returns: `latitude`, `longitude`, `timezone`

### 3. Weather Fetch (`_fetch_weather`)

- Endpoint: `https://api.open-meteo.com/v1/forecast`
- Current data: temperature, feels_like, humidity, wind, weather_code
- Daily data: weather_code, temp_max, temp_min, precip_prob, uv_index
- Units: Fahrenheit, mph, inches

### 4. WMO Weather Codes

Map numeric codes to human-readable descriptions (Clear, Rain, Snow, etc.)

## API Contract

### Tool Schema

```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Get current weather and 7-day forecast...",
    "parameters": {
      "type": "object",
      "properties": {
        "city": {"type": "string", "description": "City name"}
      },
      "required": ["city"]
    }
  }
}
```

### ToolEntry

```python
TOOL_NAME = "get_weather"
TOOL_ENTRY = ToolEntry(
    schema=WEATHER_SCHEMA,
    fn=get_weather,
)
```

## Language Support

| Input Example | Detected | Notes |
|---------------|----------|-------|
| 台北 | zh | Chinese (Traditional) |
| 北京 | zh | Chinese (Simplified) |
| 東京 | ja | Japanese |
| 서울 | ko | Korean |
| Москва | ru | Russian |
| New York | en | English (default) |

## Test Plan

| Phase | Unit | Test Case | Expected |
|-------|------|-----------|----------|
| 1 | _detect_language | "台北" | "zh" |
| 1 | _detect_language | "Tokyo" | "en" |
| 2 | _get_city_coords | "台北" | lat/lng/tz |
| 2 | _get_city_coords | "板橋" | lat/lng/tz |
| 3 | _fetch_weather | valid coords | dict |
| 4 | get_weather | "台北" | formatted string |