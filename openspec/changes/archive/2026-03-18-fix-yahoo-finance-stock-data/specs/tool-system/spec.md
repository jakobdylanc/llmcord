# tool-system Specification

## Purpose
This specification defines requirements for the tool system in the llmcord Discord bot, specifically addressing reliability improvements for the `get_market_prices` tool.

## MODIFIED Requirements

### Requirement: get_market_prices handles incomplete data gracefully
The `get_market_prices` tool SHALL handle cases where Yahoo Finance returns NaN or missing data without crashing.

#### Scenario: NaN close price
- **WHEN** Yahoo Finance returns NaN for the close price
- **THEN** the tool SHALL output "[unavailable]" instead of crashing
- **AND** continue processing other tickers

#### Scenario: Missing previous day data
- **WHEN** there is only 1 day of trading data available
- **THEN** the tool SHALL display the close price without % change
- **AND** indicate that previous day data is unavailable

#### Scenario: Empty data with retry
- **WHEN** Yahoo Finance returns empty data on first attempt
- **THEN** the tool SHALL retry up to 3 times with exponential backoff
- **AND** if all retries fail, return clear error message with ticker symbol

### Requirement: get_market_prices provides clear error messages
The tool SHALL return human-readable error messages that the LLM can understand.

#### Scenario: Invalid ticker
- **WHEN** an invalid ticker symbol is provided
- **THEN** the tool SHALL return "{ticker}: no data returned (check ticker symbol)"

#### Scenario: Network error
- **WHEN** yfinance encounters a network error
- **THEN** the tool SHALL return "{ticker}: error — {error message}"

### Requirement: get_market_prices schema has correct defaults
The tool SHALL have a schema that prevents insufficient data retrieval.

#### Scenario: Days parameter default
- **WHEN** the LLM calls the tool without specifying days
- **THEN** the tool SHALL default to 10 days of historical data

#### Scenario: Days parameter minimum
- **WHEN** the LLM specifies days less than 5
- **THEN** the tool SHALL reject the value and use 5 instead

### Requirement: get_market_prices returns accurate price data
The tool SHALL return reliable and fresh price data.

#### Scenario: Missing historical data
- **WHEN** yfinance historical data is missing recent trading days
- **THEN** the tool SHALL use the info endpoint to get previous close
- **AND** calculate percentage from the reliable previous close value

#### Scenario: Intraday price during market hours
- **WHEN** Taiwan market is open (9:00-13:30 TW)
- **THEN** the tool SHALL fetch intraday 15-minute data
- **AND** return current price instead of previous close
- **AND** append "*" marker to indicate intraday price

#### Scenario: Market closed
- **WHEN** Taiwan market is closed
- **THEN** the tool SHALL use daily close price from historical data
- **AND** not append any marker

### Requirement: get_market_prices LLM behavior
The tool output SHALL be used exactly as returned without modification.

#### Scenario: Percentage values
- **WHEN** the tool returns percentage values
- **THEN** the LLM SHALL use the exact values without modification
- **AND** not add decimals, recalculate, or alter the values