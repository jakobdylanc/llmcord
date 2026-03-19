## ADDED Requirements

### Requirement: Web portal serves on configurable port
The system SHALL provide a web portal accessible at a configurable port (default 8080) that allows users to monitor and configure the Discord bot.

#### Scenario: Portal accessible on default port
- **WHEN** portal is enabled and PORT env var is not set
- **THEN** portal is accessible at http://localhost:8080

#### Scenario: Portal accessible on custom port
- **WHEN** PORT environment variable is set to 9000
- **THEN** portal is accessible at http://localhost:9000

### Requirement: First-time setup wizard
The system SHALL provide a setup wizard on first login to create an admin password when no users exist in the database.

#### Scenario: First login shows setup wizard
- **WHEN** no users exist in DB and user accesses /login
- **THEN** setup wizard UI is shown to create initial admin user

#### Scenario: Setup wizard creates user
- **WHEN** POST /api/auth/setup is called with username and password
- **THEN** user is created with bcrypt-hashed password, JWT token returned

#### Scenario: Setup wizard blocked after users exist
- **WHEN** POST /api/auth/setup is called when users already exist
- **THEN** HTTP 403 is returned

### Requirement: Authentication required for portal access
The system SHALL require authentication before allowing access to any portal API endpoint.

#### Scenario: Unauthenticated request rejected
- **WHEN** a request is made to /api/* without valid JWT
- **THEN** HTTP 401 is returned

#### Scenario: Authenticated request allowed
- **WHEN** a request is made with valid JWT token
- **THEN** HTTP 200 is returned with requested data

### Requirement: Portal displays bot status
The system SHALL provide an endpoint that returns current bot status including online/offline state, server count, and uptime.

#### Scenario: Bot online returns status
- **WHEN** user requests /api/status while bot is connected
- **THEN** response includes: online: true, server_count: N, uptime_seconds: N

#### Scenario: Bot offline returns status
- **WHEN** user requests /api/status while bot is disconnected
- **THEN** response includes: online: false

### Requirement: Portal lists servers and channels
The system SHALL provide an endpoint that returns a list of connected Discord servers and their text channels.

#### Scenario: Returns server list
- **WHEN** user requests /api/servers
- **THEN** response includes array of servers with id, name, and text_channels

### Requirement: Simple config fields editable
The system SHALL allow editing of simple config fields: status_message, max_text, max_images, max_messages, allow_dms, use_plain_responses, show_embed_color.

#### Scenario: Update status_message
- **WHEN** user PUTs new status_message to /api/config
- **THEN** config.yaml is updated and bot reloads config

#### Scenario: Update max_text
- **WHEN** user PUTs new max_text value to /api/config
- **THEN** config.yaml is updated with new value

### Requirement: Persona list viewable
The system SHALL provide read-only access to list all available personas.

#### Scenario: List personas
- **WHEN** user requests GET /api/personas
- **THEN** response includes array of persona names

#### Scenario: Get persona content
- **WHEN** user requests GET /api/personas/{name}
- **THEN** response includes full persona markdown content

### Requirement: Task list viewable
The system SHALL provide read-only access to list all scheduled tasks.

#### Scenario: List tasks
- **WHEN** user requests GET /api/tasks
- **THEN** response includes array of task configurations

### Requirement: Refresh command available via API
The system SHALL provide an endpoint to trigger config reload equivalent to /refresh slash command.

#### Scenario: Trigger refresh via API
- **WHEN** user POSTs to /api/refresh
- **THEN** config is reloaded and response confirms success