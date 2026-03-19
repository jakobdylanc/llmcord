## Why

The Discord bot currently operates as a "black box" with no external visibility. Users cannot see bot status, event logs, or easily modify configuration without directly editing YAML files. A web portal would provide real-time monitoring and simplified configuration management.

## What Changes

- Add FastAPI web server running alongside Discord bot (port 8080)
- Add SQLite database for event log persistence and stats
- Add WebSocket endpoint for real-time log streaming
- Add REST API for bot status, logs query, and simple config editing
- Add React-based web portal frontend
- Add authentication via password + optional Discord admin check

### New Capabilities

- **web-portal**: Web-based admin dashboard for monitoring and configuration
- **log-persistence**: SQLite-backed event logging with configurable retention
- **api-endpoints**: REST API for status, logs, and config operations
- **real-time-logs**: WebSocket stream of bot events

### Modified Capabilities

- None - this is a net-new capability that doesn't change existing spec requirements

## Impact

- **New dependencies**: fastapi, uvicorn, sqlalchemy, aiosqlite (Python); React + Vite (frontend)
- **New files**: bot/web/ (FastAPI server), bot/db/ (SQLite), web/ (React)
- **Modified files**: llmcord.py (minimal - 2 lines to start web server), requirements.txt, config.yaml
- **No breaking changes**: Existing functionality unchanged

## Additional Features Implemented (Post-Original Scope)

During implementation, the following enhancements were added:

1. **Bot State Integration**: Added BotState singleton that tracks Discord bot reference, providing:
   - Real-time server count, user count, channel count
   - Bot uptime tracking
   - Online/offline status indicator

2. **Live Config Reload**: Enhanced /api/refresh endpoint to reload BOTH:
   - Portal configuration (web server settings)
   - Main bot configuration (models, providers, tasks, personas)

3. **Improved UX**:
   - Changed portal branding to "GPT Discord Bot"
   - Fixed SPA routing issues
   - Added login redirect after successful authentication
   - Enhanced status page with detailed bot information

---

## Phase 2: Enhanced Features (Post-Original Scope)

The following features will be implemented in separate phases after the initial web portal:

### Phase 1: Bug Fixes & UI Improvements
- Fix config page "[object Object]" display issues (models, permissions, providers, tools)
- Rename status fields for clarity (Servers→Guilds, remove Users)
- Add bot avatar to status page
- Fix Chinese character encoding in YAML (UTF-8/Unicode)
- Add three-button config workflow (Save/Apply/Save&Apply)

### Phase 2: Servers Detail View
- Server detail subpage with member/channel lists
- Server icon display
- Permission management UI (grant/revoke)

### Phase 3: Persona Management
- Full CRUD for persona files
- Markdown editor with live preview
- File storage: bot/config/personas/*.md

### Phase 4: Task Management
- Full CRUD for task files
- YAML editor with validation
- Enable/disable toggles
- Manual task execution ("Run Now" button)
- File storage: bot/config/tasks/*.yaml

### Phase 5: Skills (Tools) Display
- Read-only list of available skills
- Display name and description
- Modification deferred to future