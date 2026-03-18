---
name: get_weather
description: Get current weather and 7-day forecast for any city worldwide.
metadata: {"clawhub":{"emoji":"🌤️","requires":{},"tools":["get_weather"]}}
---

# get_weather

Fetches current weather conditions and 7-day forecast for any city worldwide. Uses Open-Meteo API (free, no API key required) and Nominatim for geocoding.

## When to use

- User asks about current weather in a specific city
- User wants to know temperature, humidity, or wind conditions
- User asks for forecast or "what's the weather like this week"
- User asks about rain probability or upcoming weather

## Features

- **Multi-language**: Supports Chinese (台北, 北京), Japanese (東京), Korean, English, etc.
- **Current conditions**: Temperature, feels-like, humidity, wind, conditions
- **7-day forecast**: Daily high/low temps, precipitation probability, conditions
- **No API key**: Uses free Open-Meteo and OpenStreetMap APIs

## Tool signature

```
get_weather(city: str) -> str
```

## Parameters

| Name | Type | Required | Description |
|------|------|----------|-------------|
| city | string | Yes | City name in any language (e.g., "台北", "Tokyo", "London") |

## Example calls

```
get_weather(city="台北")
get_weather(city="台北市松山區")
get_weather(city="新北市")
get_weather(city="新北市板橋區")
get_weather(city="London")
get_weather(city="New York")
get_weather(city="東京")
```

## Returns

Formatted string with:
- Current weather conditions (emoji, description)
- Temperature in Celsius
- "Feels like" temperature
- Humidity percentage
- Wind speed in km/h
- 7-day forecast with dates, conditions, high/low temps, rain probability

## Example output

```
🌤️ Current weather in 台北:
   Overcast
   🌡️ 23.2°C (feels like 23.3°C)
   💧 Humidity: 59%
   💨 Wind: 9.7 km/h

📅 7-Day Forecast:
   • Today (2026-03-18): Overcast, High 27°C / Low 16°C, Rain 0%
   • 2026-03-19: Partly cloudy, High 28°C / Low 15°C, Rain 10%
   • ...
```

## Limitations

- Rate limited to 1 request/second (Nominatim geocoding)
- Timezone is estimated from coordinates (not exact)
- Some smaller cities may not be found