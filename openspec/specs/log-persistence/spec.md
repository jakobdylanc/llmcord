## ADDED Requirements

### Requirement: Event logs persisted to database
The system SHALL store event logs in SQLite database for query and retention.

#### Scenario: Log entry stored
- **WHEN** bot processes a message or event
- **THEN** log entry is written to SQLite with timestamp, level, event_type, message

### Requirement: Logs queryable via API
The system SHALL provide an endpoint to query stored logs with filtering.

#### Scenario: Query all logs
- **WHEN** user requests GET /api/logs without filters
- **THEN** returns recent log entries (default limit 100)

#### Scenario: Filter logs by level
- **WHEN** user requests GET /api/logs?level=ERROR
- **THEN** returns only ERROR level logs

#### Scenario: Filter logs by time range
- **WHEN** user requests GET /api/logs?since=2026-03-18T00:00:00
- **THEN** returns logs from that time forward

### Requirement: Real-time log streaming via WebSocket
The system SHALL provide a WebSocket endpoint that streams new log entries in real-time.

#### Scenario: WebSocket connection established
- **WHEN** client connects to /ws/logs
- **THEN** connection remains open and receives new log entries as JSON

#### Scenario: Client receives log event
- **WHEN** bot generates a new log event
- **THEN** WebSocket clients receive JSON with timestamp, level, message

### Requirement: Configurable log retention
The system SHALL automatically delete logs older than configured retention period.

#### Scenario: Default retention is 7 days
- **WHEN** no retention config is set
- **THEN** logs older than 7 days are deleted

#### Scenario: Custom retention period
- **WHEN** config specifies logs.retention_days: 30
- **THEN** logs older than 30 days are deleted

### Requirement: Log level filtering
The system SHALL support filtering which log levels are captured and displayed.

#### Scenario: Filter to INFO, WARNING, ERROR
- **WHEN** config specifies logs.levels: [INFO, WARNING, ERROR]
- **THEN** DEBUG level logs are not stored or streamed