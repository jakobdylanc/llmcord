## Why

Users want to check weather for their cities, particularly Chinese-speaking users who prefer using city names in Chinese (e.g., 台北, 板橋, 桃園). The bot currently lacks weather functionality, and users would benefit from a simple, free weather tool that supports multiple languages.

## What Changes

- Add new `get_weather` tool to `bot/llm/tools/`
- Uses Open-Meteo API (free, no API key required)
- Dynamic language detection based on city name characters
- Returns current weather + 7-day forecast
- Auto-discovery via existing tool registry

## Capabilities

### New Capabilities

- **weather-tool**: Get current weather and 7-day forecast for any city worldwide

### Modified Capabilities

- (None - new tool only)

## Impact

- New file: `bot/llm/tools/weather.py`
- New file: `bot/llm/tools/skills/weather.md` (optional)
- Config update: add `"get_weather"` to model's tools list