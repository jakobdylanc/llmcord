# Implementation Tasks

## Phase 1: Language Detection

- [x] 1.1 Create `bot/llm/tools/weather.py` - add `_detect_language` function
- [x] 1.2 Write unit test for `_detect_language` with multiple languages
- [x] 1.3 Run test - verify all language codes detected correctly

## Phase 2: Geocoding

- [x] 2.1 Add `_get_city_coords` function using Nominatim (OpenStreetMap) API
- [x] 2.2 Add `_estimate_timezone` for timezone estimation
- [x] 2.3 Test with Chinese cities: 台北, 板橋, 桃園 (all pass)
- [x] 2.4 Test with English cities: New York, London (all pass)

Note: Open-Meteo Geocoding doesn't support Chinese well. Switched to Nominatim.

## Phase 3: Weather API

- [x] 3.1 Add `_fetch_weather` function using Open-Meteo Weather API
- [x] 3.2 Add WMO weather codes mapping
- [x] 3.3 Test fetching weather data with coordinates

## Phase 4: Main Function + Formatting

- [x] 4.1 Create `get_weather(city)` main function
- [x] 4.2 Format output: current weather + 7-day forecast
- [x] 4.3 Test with "台北" - verify Chinese output ✅
- [x] 4.4 Test with "London" - verify English output ✅

## Phase 5: ToolEntry Registration

- [x] 5.1 Define `WEATHER_SCHEMA` (OpenAI format)
- [x] 5.2 Create `TOOL_NAME` and `TOOL_ENTRY` for auto-discovery
- [x] 5.3 Verify tool is auto-discovered by registry ✅ (get_weather found)

## Phase 6: Skill Doc (Optional)

- [x] 6.1 Create `bot/llm/tools/skills/weather.md`
- [x] 6.2 Document tool usage and examples

## Phase 7: Config Update

- [x] 7.1 Add `"get_weather"` to model's tools list in `config.yaml`
- [x] 7.2 Test end-to-end with bot

---

## Phase 8: Dynamic Keywords for Tool Detection

Problem: qwen3:14b doesn't follow tool instructions properly - it hallucinates weather data instead of calling the tool. Hardcoded keywords in llmcord.py violate the dynamic tool principle.

Solution: Add `keywords` to ToolEntry and update llmcord.py to read them dynamically (not hardcoded). Other tools can be updated in separate changes.

### 8.1 Add `keywords` field to ToolEntry dataclass

- [x] 8.1.1 Update `bot/llm/tools/_types.py` - add `keywords` field (list[str]) to ToolEntry ✅
- [x] 8.1.2 Verify no breaking changes in existing tools ✅

### 8.2 Add keywords to weather tool

- [x] 8.2.1 Add `keywords` to `TOOL_ENTRY` in `weather.py` ✅

### 8.3 Update llmcord.py to use dynamic keywords

- [x] 8.3.1 Remove hardcoded `_TOOL_KEYWORDS` dict ✅
- [x] 8.3.2 Add `_get_tool_keywords(registry)` function ✅
- [x] 8.3.3 Update `_extract_keyword_tool` to use dynamic keywords ✅
- [x] 8.3.4 Use partial/fuzzy matching for better flexibility ✅

### 8.4 Test

- [x] 8.4.1 Test weather query "明天新北市板橋的天氣如何?" → ('get_weather', '新北市板橋') ✅
- [x] 8.4.2 Test "天氣怎麼樣" → ('get_weather', '') ✅
- [x] 8.4.3 Verify no false triggers on unrelated queries ✅

> **Note:** Adding keywords to other tools (web_search, yahoo_finance, etc.) should be done in a separate change to keep changes focused.

---

## Dependencies

- External: Nominatim (OpenStreetMap) - Geocoding (free, no key, 1 req/sec)
- External: Open-Meteo Weather API (free, no key)
- Internal: `bot.llm.tools._types.ToolEntry`
- Internal: `requests` library (already in requirements.txt)