"""
Weather tool using Open-Meteo API (free, no API key required).
Provides current weather and 7-day forecast for any city worldwide.
"""

from __future__ import annotations

import logging
import re
import urllib.parse
from typing import Any

import requests

# ── Language Detection ───────────────────────────────────────────────────────

def _detect_language(text: str) -> str:
    """
    Detect language from text using Unicode character ranges.
    
    Args:
        text: Input text (city name)
        
    Returns:
        Language code (zh, ja, ko, ru, ar, th, he, en)
    """
    # Chinese (including Traditional & Simplified)
    if re.search(r'[\u4e00-\u9fff]', text):
        return "zh"
    
    # Japanese (Hiragana, Katakana, Kanji)
    if re.search(r'[\u3040-\u30ff\u31f0-\u31ff]', text):
        return "ja"
    
    # Korean (Hangul)
    if re.search(r'[\uac00-\ud7af]', text):
        return "ko"
    
    # Cyrillic (Russian, Ukrainian, etc.)
    if re.search(r'[\u0400-\u04ff]', text):
        return "ru"
    
    # Arabic
    if re.search(r'[\u0600-\u06ff]', text):
        return "ar"
    
    # Thai
    if re.search(r'[\u0e00-\u0e7f]', text):
        return "th"
    
    # Hebrew
    if re.search(r'[\u0590-\u05ff]', text):
        return "he"
    
    # Default to English
    return "en"


# ── WMO Weather Code Mapping ─────────────────────────────────────────────────

WMO_CODES = {
    "0": "Clear sky",
    "1": "Mainly clear",
    "2": "Partly cloudy",
    "3": "Overcast",
    "45": "Fog",
    "48": "Depositing rime fog",
    "51": "Light drizzle",
    "53": "Moderate drizzle",
    "55": "Dense drizzle",
    "56": "Light freezing drizzle",
    "57": "Dense freezing drizzle",
    "61": "Slight rain",
    "63": "Moderate rain",
    "65": "Heavy rain",
    "66": "Light freezing rain",
    "67": "Heavy freezing rain",
    "71": "Slight snow",
    "73": "Moderate snow",
    "75": "Heavy snow",
    "77": "Snow grains",
    "80": "Slight rain showers",
    "81": "Moderate rain showers",
    "82": "Violent rain showers",
    "85": "Slight snow showers",
    "86": "Heavy snow showers",
    "95": "Thunderstorm",
    "96": "Thunderstorm with slight hail",
    "99": "Thunderstorm with heavy hail",
}

# ── Geocoding ─────────────────────────────────────────────────────────────────

# Nominatim (OpenStreetMap) rate limit
_last_request_time = 0.0

def _get_city_coords(city: str) -> tuple[float, float, str] | None:
    """
    Geocode a city name to lat/lng/timezone via Nominatim (OpenStreetMap).
    
    Args:
        city: Name of the city (supports any language)
        
    Returns:
        Tuple of (latitude, longitude, timezone) or None if not found
        
    Note:
        Nominatim has 1 request/second rate limit. This function enforces it.
    """
    global _last_request_time
    
    import time
    
    # Enforce rate limit (1 req/sec)
    elapsed = time.time() - _last_request_time
    if elapsed < 1.0:
        time.sleep(1.0 - elapsed)
    
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        'q': city,
        'format': 'json',
        'limit': 1,
        'addressdetails': 1,
    }
    
    try:
        resp = requests.get(url, params=params, timeout=10, headers={
            'User-Agent': 'gpt-discord-bot/1.0'
        })
        resp.raise_for_status()
        data = resp.json()
        
        if not data:
            logging.warning("weather: no geocode results for '%s'", city)
            return None
            
        result = data[0]
        lat = float(result["lat"])
        lon = float(result["lon"])
        
        # Determine timezone from coordinates (rough approximation)
        # This is a simplification - ideally we'd use a timezone API
        tz = _estimate_timezone(lat, lon)
        
        _last_request_time = time.time()
        return (lat, lon, tz)
        
    except Exception as e:
        logging.warning("weather: geocode failed for '%s': %s", city, e)
        return None


def _estimate_timezone(lat: float, lon: float) -> str:
    """
    Estimate timezone from coordinates (simplified).
    
    Args:
        lat: Latitude
        lon: Longitude
        
    Returns:
        IANA timezone string (approximate)
    """
    # Korea (check BEFORE Japan since they're close)
    if 33 <= lat <= 43 and 124 <= lon <= 132:
        return "Asia/Seoul"
    # Taiwan
    if 21.5 <= lat <= 26.5 and 118 <= lon <= 123:
        return "Asia/Taipei"
    # Japan
    if 24 <= lat <= 46 and 122 <= lon <= 154:
        return "Asia/Tokyo"
    # China (mainland)
    if 18 <= lat <= 54 and 73 <= lon <= 135:
        return "Asia/Shanghai"
    # Hong Kong / Macau
    if 22.1 <= lat <= 22.6 and 113.8 <= lon <= 114.5:
        return "Asia/Hong_Kong"
    # Singapore
    if 1.1 <= lat <= 1.5 and 103.6 <= lon <= 104.0:
        return "Asia/Singapore"
    # US West (check before East since -100 is the dividing line)
    if 24 <= lat <= 50 and -130 <= lon <= -100:
        return "America/Los_Angeles"
    # US East
    if 24 <= lat <= 50 and -100 < lon <= -65:
        return "America/New_York"
    # UK
    if 49 <= lat <= 61 and -8 <= lon <= 2:
        return "Europe/London"
    # Europe (central)
    if 45 <= lat <= 55 and -5 <= lon <= 20:
        return "Europe/Paris"
    # Australia
    if -44 <= lat <= -10 and 112 <= lon <= 154:
        return "Australia/Sydney"
    # Default UTC
    return "UTC"


def _fetch_weather(lat: float, lng: float, tz: str) -> dict | str:
    """
    Fetch weather data from Open-Meteo API.
    
    Args:
        lat: Latitude
        lng: Longitude
        tz: Timezone string
        
    Returns:
        Dictionary with current + daily weather data, or error string
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lng,
        "current": [
            "temperature_2m",
            "relative_humidity_2m",
            "apparent_temperature",
            "wind_speed_10m",
            "weather_code",
        ],
        "daily": [
            "weather_code",
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_probability_max",
        ],
        "timezone": "auto",
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
        "forecast_days": 7,
    }
    
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        if "error" in data:
            return f"Error: {data.get('reason', 'Unknown error')}"
        
        return data
        
    except requests.RequestException as e:
        return f"Error fetching weather: {e}"


def get_weather(city: str) -> str:
    """
    Get current weather and 7-day forecast for a given city.
    
    Args:
        city: Name of the city (e.g., "New York", "台北", "Tokyo")
        
    Returns:
        Formatted current weather and forecast.
    """
    # Step 1: Geocode the city
    coords = _get_city_coords(city)
    if not coords:
        return f"Error: Could not find city '{city}'"
    
    lat, lng, tz = coords
    
    # Step 2: Fetch weather data
    data = _fetch_weather(lat, lng, tz)
    if isinstance(data, str):
        return data  # Error message
    
    # Step 3: Format current weather
    curr = data.get("current", {})
    curr_units = data.get("current_units", {})
    weather_desc = WMO_CODES.get(str(curr.get("weather_code", 0)), "Unknown")
    
    lines = [
        f"🌤️ Current weather in {city}:",
        f"   {weather_desc}",
        f"   🌡️ {curr.get('temperature_2m', 'N/A')}°C (feels like {curr.get('apparent_temperature', 'N/A')}°C)",
        f"   💧 Humidity: {curr.get('relative_humidity_2m', 'N/A')}%",
        f"   💨 Wind: {curr.get('wind_speed_10m', 'N/A')} km/h",
    ]
    
    # Step 4: Add 7-day forecast
    daily = data.get("daily", {})
    daily_units = data.get("daily_units", {})
    
    if daily.get("time"):
        lines.append("")
        lines.append("📅 7-Day Forecast:")
        
        for i, date_str in enumerate(daily["time"]):
            # Mark today
            if i == 0:
                date_str = f"Today ({date_str})"
            
            weather = WMO_CODES.get(str(daily["weather_code"][i]), "Unknown")
            temp_max = daily["temperature_2m_max"][i]
            temp_min = daily["temperature_2m_min"][i]
            precip = daily["precipitation_probability_max"][i]
            
            lines.append(
                f"   • {date_str}: {weather}, "
                f"High {temp_max}°C / Low {temp_min}°C, "
                f"Rain {precip}%"
            )
    
    return "\n".join(lines)


# ── Schema ────────────────────────────────────────────────────────────────────

WEATHER_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather and 7-day forecast for any city worldwide. Returns temperature, feels-like, humidity, wind, conditions, and daily forecasts.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The name of the city (e.g., 'New York', '台北', 'Tokyo')",
                },
            },
            "required": ["city"],
        },
    },
}


# ── Dynamic Tool Registration ────────────────────────────────────────────────
# Keywords are now loaded from bot/llm/tools/skills/weather.md frontmatter

from bot.llm.tools._types import ToolEntry

TOOL_NAME = "get_weather"
TOOL_ENTRY = ToolEntry(
    schema=WEATHER_SCHEMA,
    fn=get_weather,
)