# weather-tool Specification

## ADDED Requirements

### Requirement: get_weather tool provides current weather
The system SHALL provide a `get_weather(city: str)` tool that returns current weather conditions including temperature, feels-like, humidity, wind speed, and weather description.

#### Scenario: Current weather
- **WHEN** user asks for weather in "台北"
- **THEN** tool returns current temperature, conditions, humidity, wind

### Requirement: get_weather tool provides 7-day forecast
The system SHALL return a 7-day forecast with daily high/low temperatures, precipitation probability, and weather conditions.

#### Scenario: Forecast
- **WHEN** user asks for weather
- **THEN** tool returns 7-day forecast with dates and conditions

### Requirement: Dynamic language detection
The system SHALL auto-detect the language from the city name using Unicode character ranges and use it for the geocoding API.

#### Scenario: Chinese city
- **WHEN** city is "台北"
- **THEN** language detected as "zh" for geocoding

#### Scenario: English city
- **WHEN** city is "New York"
- **THEN** language defaults to "en"

### Requirement: Open-Meteo API integration
The system SHALL use the free Open-Meteo API (no API key required) for both geocoding and weather data.

#### Scenario: API call
- **WHEN** get_weather is called
- **THEN** uses Open-Meteo Geocoding + Weather APIs

### Requirement: ToolEntry registration
The system SHALL register `get_weather` as a ToolEntry with schema for auto-discovery by the tool registry.

#### Scenario: Auto-discovery
- **WHEN** registry discovers tools
- **THEN** get_weather is found and available