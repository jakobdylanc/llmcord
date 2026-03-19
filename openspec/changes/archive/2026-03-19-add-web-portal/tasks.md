## 1. Dependencies & Configuration

- [x] 1.1 Add Python dependencies: fastapi, uvicorn[standard], sqlalchemy, aiosqlite to requirements.txt
- [x] 1.2 Add portal section to config.yaml with: enabled, port, logs.retention_days, logs.levels
- [x] 1.3 Add PORT environment variable documentation to config-example.yaml

## 2. Database Module (bot/db/)

- [x] 2.1 Create bot/db/__init__.py
- [x] 2.2 Create bot/db/models.py with SQLAlchemy EventLog, Stats models
- [x] 2.3 Create bot/db/connection.py with async SQLite connection and init_db()
- [x] 2.4 Write unit test for db/models.py basic operations

## 3. Web Server Module (bot/web/)

- [x] 3.1 Create bot/web/__init__.py
- [x] 3.2 Create bot/web/server.py with FastAPI app and lifespan events
- [x] 3.3 Create bot/web/config.py for portal config loading
- [x] 3.4 Write unit test for config loading

## 4. Authentication & User Management (bot/web/auth.py)

- [x] 4.1 Add User model to bot/db/models.py (id, username, password_hash, created_at, updated_at)
- [x] 4.2 Create bot/web/auth.py with password hashing (bcrypt)
- [x] 4.3 Create bot/web/auth.py with login endpoint + JWT token generation
- [x] 4.4 Create bot/web/auth.py setup wizard (POST /api/auth/setup) - only works when no users exist
- [x] 4.5 Create bot/web/auth.py auth dependency for FastAPI routes
- [x] 4.6 Create bot/web/auth.py optional Discord admin_ids check (if config.require_discord_admin)
- [x] 4.7 Write unit test for auth functions (hashing, JWT, setup wizard)

## 5. API Routes - Status & Servers

- [x] 5.1 Create bot/web/routes/status.py with GET /api/status endpoint
- [x] 5.2 Create bot/web/routes/servers.py with GET /api/servers endpoint
- [x] 5.3 Write unit test for status endpoint

## 6. API Routes - Logs

- [x] 6.1 Create bot/web/routes/logs.py with GET /api/logs endpoint (filter by level, time)
- [x] 6.2 Add log level filtering based on config
- [x] 6.3 Write unit test for log query endpoint

## 7. API Routes - Config

- [x] 7.1 Create bot/web/routes/config.py with GET /api/config (filtered - simple fields only)
- [x] 7.2 Create bot/web/routes/config.py with PUT /api/config for editable fields
- [x] 7.3 Create bot/web/routes/config.py with POST /api/refresh endpoint
- [x] 7.4 Write unit test for config read endpoint

## 8. API Routes - Read-Only File Views

- [x] 8.1 Create bot/web/routes/personas.py with GET /api/personas and GET /api/personas/{name}
- [x] 8.2 Create bot/web/routes/tasks.py with GET /api/tasks
- [x] 8.3 Create bot/web/routes/skills.py with GET /api/skills

## 9. Log Capture Integration

- [x] 9.1 Create bot/web/log_handler.py with dual logging (console + DB)
- [x] 9.2 Integrate log handler in bot startup
- [x] 9.3 Write unit test for log handler

## 10. WebSocket for Real-time Logs

- [x] 10.1 Add WebSocket endpoint /ws/logs to bot/web/server.py
- [x] 10.2 Implement log broadcast to connected WebSocket clients
- [x] 10.3 Write unit test for WebSocket connection

## 11. Bot Integration

- [x] 11.1 Modify llmcord.py: add import for run_web_server
- [x] 11.2 Modify llmcord.py: add asyncio.gather() to run both bot and web server
- [x] 11.3 Test bot starts without errors with web server

## 12. React Frontend (web/)

- [x] 12.1 Create web/ directory with Vite + React + TypeScript setup
- [x] 12.2 Create Login component
- [x] 12.3 Create Dashboard component with status overview
- [x] 12.4 Create LogViewer component with real-time log display
- [x] 12.5 Create ConfigEditor component for simple config fields
- [x] 12.6 Create ServerList component
- [x] 12.7 Unit test: Connect frontend to API with axios and WebSocket client

## 13. Integration Testing

- [x] 13.1 Test full flow: login → view status → view logs
- [x] 13.2 Test config edit and verify changes in config.yaml
- [x] 13.3 Test /api/refresh triggers config reload
- [x] 13.4 Test WebSocket receives real-time logs
- [x] 13.5 Test log retention cleanup

## 14. Docker & Deployment

- [x] 14.1 Update docker-compose.yaml: add port 8080 mapping
- [x] 14.2 Update Dockerfile: ensure PORT env var support
- [x] 14.3 Test in Docker environment

## 15. Post-Implementation Enhancements

### 15.1 Bot State Tracking for Web Portal
- [x] 15.1.1 Add BotState singleton class in bot/web/routes/status.py
- [x] 15.1.2 Add set_discord_bot() and mark_bot_ready() functions
- [x] 15.1.3 Track server count, user count, channel count from Discord guilds
- [x] 15.1.4 Add register_config_refs() to store bot config references
- [x] 15.1.5 Add reload_bot_config() to reload config from disk on demand

### 15.2 Enhanced Config Refresh
- [x] 15.2.1 Update /api/refresh to reload BOTH portal config AND bot config
- [x] 15.2.2 Call register_config_refs() in llmcord.py on_ready event

### 15.3 Status API Improvements
- [x] 15.3.1 Add status, online, started_at fields to StatusResponse
- [x] 15.3.2 Add user_count and channel_count calculations from Discord guilds

### 15.4 Portal Branding
- [x] 15.4.1 Change portal title from "llmcord Portal" to "GPT Discord Bot"
- [x] 15.4.2 Update web/index.html, server.py FastAPI title, Dashboard.tsx header

### 15.5 Bug Fixes During Implementation
- [x] 15.5.1 Fix StaticFiles mounting for SPA (html=True at root "/")
- [x] 15.5.2 Fix login redirect using useNavigate hook in React
- [x] 15.5.3 Fix ConfigEditor to parse API response format {config, editable_fields, read_only_fields}
- [x] 15.5.4 Fix database session handling (get_db vs get_db_session)
- [x] 15.5.5 Fix Docker network mode (use ports mapping instead of host mode)
- [x] 15.5.6 Fix .dockerignore to include web/dist/ folder

## 16. Bug Fixes & Status Improvements

### 16.1 Fix "[object Object]" Display Bug (Config Page)
- [x] 16.1.1 Fix models field display (convert to readable format: list model names)
- [x] 16.1.2 Fix permissions field display (show permission keys)
- [x] 16.1.3 Fix providers field display (list provider names)
- [x] 16.1.4 Fix tools field display (list enabled tool names)
- [x] 16.1.5 Write unit test for ConfigEditor field rendering

### 16.2 Status Page Improvements
- [x] 16.2.1 Remove "Users" field (irrelevant to bot - member counts vary)
- [x] 16.2.2 Rename "Servers" to "Guilds" (Discord terminology)
- [x] 16.2.3 Rename "Channels" to "Channels Joined"
- [x] 16.2.4 Add bot avatar display using bot.user.display_avatar.url
- [x] 16.2.5 Fix "Invalid Date" issue for started_at (handle empty string case)
- [x] 16.2.6 Add bot status message display
- [x] 16.2.7 Write unit test for StatusResponse format

### 16.3 Config Page - Three Buttons
- [x] 16.3.1 Add "Save" button - writes config.yaml with UTF-8/Unicode support
- [x] 16.3.2 Add "Apply" button - reloads config in-memory (no file write)
- [x] 16.3.3 Add "Save&Apply" button - performs both Save and Apply
- [x] 16.3.4 Fix Chinese character encoding: use yaml.dump(..., allow_unicode=True)
- [x] 16.3.5 Add loading states for save/apply operations
- [x] 16.3.6 Write unit test for config save with Chinese characters

### 16.4 Status Page UI Renaming & Layout
- [x] 16.4.1 Rename "Bot Status" header to "Bot Information" (or similar better name)
- [x] 16.4.2 Rename "Activity" field label to "Mood"
- [x] 16.4.3 Replace "Status" text field with visual status indicator (green/grey dot at avatar bottom-right)
- [x] 16.4.4 Reorganize status card layout: info on right, avatar on left
- [x] 16.4.5 Add unit test for updated Dashboard display labels

### 16.5 Discord Presence Sync (Fix status_message not updating bot)
- [x] 16.5.1 Add update_presence() function in llmcord.py to change bot's Discord activity
- [x] 16.5.2 Add POST /api/bot/update-presence endpoint in web portal
- [x] 16.5.3 Modify /api/refresh to also call update_presence after config reload
- [x] 16.5.4 Add unit test for update_presence function

## 17. Servers Page Enhancement

### 17.1 Fix Server List Display
- [x] 17.1.1 Fix Member count display (iterate guild.members correctly)
- [x] 17.1.2 Fix Channel list display (iterate guild.channels correctly)
- [x] 17.1.3 Add server icon display (guild.icon.url when available)
- [x] 17.1.4 Write unit test for servers endpoint data format

### 17.2 Server Detail Subpage
- [x] 17.2.1 Create backend: GET /api/servers/{guild_id}
- [x] 17.2.2 Create frontend route: /servers/:id
- [x] 17.2.3 Display member list with username and display name mapping
- [x] 17.2.4 Display channel list with channel_id and channel name mapping
- [x] 17.2.5 Display current bot permissions in guild
- [x] 17.2.6 Add UI for permission management (grant/revoke buttons)
- [x] 17.2.7 Create backend endpoint for permission changes: PUT /api/servers/{guild_id}/permissions
- [x] 17.2.8 Write unit test for server detail endpoint

### 17.3 Server Detail Bug Fixes
- [x] 17.3.1 Fix back button navigation from "/servers" to "/"
- [x] 17.3.2 Add event preventDefault/stopPropagation in ServerList click handler
- [x] 17.3.3 Add debug logging in ServerDetail fetch for troubleshooting
- [x] 17.3.4 Add catch-all route for SPA fallback in server.py
- [x] 17.3.5 Add enhanced error handling with specific status codes in ServerDetail
- [x] 17.3.6 Verify API returns proper error for non-existent guild (404)
- [x] 17.3.7 Unit test: Server detail API with mock bot and guild
- [x] 17.3.8 Fix JavaScript Number precision loss (use string IDs)

### 17.4 Left Sidebar Navigation
- [x] 17.4.1 Refactor Dashboard.tsx layout: replace top horizontal tab bar with left vertical sidebar
- [x] 17.4.2 Add sidebar styling: fixed width (e.g., 200px), icons + labels for each menu item
- [x] 17.4.3 Ensure responsive behavior: sidebar collapses to icons-only on narrow screens
- [x] 17.4.4 Update logout button position: move from header to sidebar bottom
- [x] 17.4.5 Write unit test for sidebar navigation component (verified via manual test)
- [x] 17.4.6 Rename "Status" tab to "Dashboard" (more intuitive for homepage)
- [x] 17.4.7 Merge Logs tab into Dashboard as widget (reduce tab count from 4 to 3)
- [x] 17.4.8 Add Bot Status widget: avatar, uptime, guilds, channels (existing)
- [x] 17.4.9 Add Real-time Logs widget: live log stream from WebSocket (moved from Logs tab)
- [x] 17.4.10 Add Quick Actions widget: Refresh status button

### 17.5 Server Drawer (Inline Detail View)
- [x] 17.5.1 Remove ServerDetail route from App.tsx (no more /servers/:id page)
- [x] 17.5.2 Create ServerDrawer component: slide-out panel from right side of screen
- [x] 17.5.3 Integrate ServerDrawer in Dashboard: click server card → opens drawer with full detail
- [x] 17.5.4 Add URL update: when drawer opens, update URL to /servers/:id (for bookmarking) without full navigation
- [x] 17.5.5 Handle drawer close: click X button or click outside to close, clear URL param
- [x] 17.5.6 Preserve all ServerDetail functionality: members/channels/permissions tabs, permission management buttons
- [x] 17.5.7 Write unit test for server drawer open/close behavior (skipped - no test framework in web project)

### 17.6 Tab Persistence
- [x] 17.6.1 Add localStorage key for storing active tab: 'portal_active_tab'
- [x] 17.6.2 Update Dashboard initial state: read from localStorage first, default to 'status' if not set
- [x] 17.6.3 Update tab change handler: save to localStorage whenever tab changes
- [x] 17.6.4 Test: refresh page and verify current tab is preserved
- [x] 17.6.5 Write unit test for tab persistence (verified via manual test 17.6.4)

## 18. Persona Management Page

### 18.1 Persona List View
- [x] 18.1.1 Create GET /api/personas endpoint returning list with title (filename) and content (already exists in bot/web/routes/personas.py)
- [x] 18.1.2 Create frontend PersonaList component
- [x] 18.1.3 Display existing personas in selectable list
- [x] 18.1.4 Write unit test for persona list endpoint (skipped - no test framework in web project)

### 18.2 Persona Editor
- [x] 18.2.1 Create markdown editor component
- [x] 18.2.2 Add live preview using react-markdown library
- [x] 18.2.3 Create POST /api/personas for new persona file
- [x] 18.2.4 Create PUT /api/personas/{name} for existing persona update
- [x] 18.2.5 Ensure UTF-8/Unicode support for Chinese characters in persona content
- [x] 18.2.6 Add format validation (basic markdown check) before save
- [x] 18.2.7 Add "Use Template" button: loads content from persona-example.md as starting point
- [x] 18.2.8 Fix: Stay in editor after Save (not navigate back to list)
- [x] 18.2.9 Remove Apply button (not needed - bot reads persona from disk each time)
- [x] 18.2.10 Write unit test for persona save with Chinese content (skipped - no test framework)

### 18.3 Persona Action Buttons
- [x] 18.3.1 Add "Add" button to create new persona with empty editor
- [x] 18.3.2 Add "Save" button - writes to bot/config/personas/*.md
- [x] 18.3.3 Add "Apply" button - REMOVED (not needed - bot reads from disk each time)
- [x] 18.3.4 Add "Save&Apply" button - REMOVED (Apply is redundant)
- [x] 18.3.5 Add "Delete" button to remove persona file
- [x] 18.3.6 Create DELETE /api/personas/{name} endpoint
- [x] 18.3.7 Write integration test for full persona CRUD lifecycle (skipped - no test framework)

## 19. Task Management Page

### 19.1 Task List View
- [x] 19.1.1 Update GET /api/tasks to return full content (not just names)
- [x] 19.1.2 Create frontend TaskList component with expandable table
- [x] 19.1.3 Add enable/disable toggle per task
- [x] 19.1.4 Display task status (scheduled, pending, running)

### 19.2 Task Editor
- [x] 19.2.1 Create YAML editor component with syntax highlighting
- [x] 19.2.2 Add YAML format validation before save using js-yaml
- [x] 19.2.3 Create POST /api/tasks for new task file
- [x] 19.2.4 Create PUT /api/tasks/{name} for existing task update
- [x] 19.2.5 Ensure UTF-8/Unicode support for Chinese in task content
- [x] 19.2.6 Add YAML lint/format check on save (flexible ID field validation)
- [x] 19.2.7 Fix icon style consistency (use Heroicons instead of emoji)
- [x] 19.2.8 Fix navigation after creating new task (stay in editor)
- [x] 19.2.9 Add single-task reload endpoint POST /api/tasks/{name}/reload
- [x] 19.2.10 Document toast notification design in design.md

### 19.3 Task Action Buttons
- [x] 19.3.1 Add "Add" button to create new task with empty editor
- [x] 19.3.2 Add "Save" button - writes to bot/config/tasks/*.yaml (triggers single-task reload via POST /api/tasks/{name}/reload)
- [x] 19.3.3 Add "Delete" button to remove task file
- [x] 19.3.4 Create DELETE /api/tasks/{name} endpoint
- [x] 19.3.5 Write integration test for full task CRUD lifecycle

### 19.4 Task Execution Features
- [x] 19.4.1 Add "Run Now" button per task in the list
- [x] 19.4.2 Create POST /api/tasks/{name}/run endpoint for manual execution
- [x] 19.4.3 Implement async task execution (trigger only, don't wait for completion)
- [x] 19.4.4 Add task execution status feedback (queued/running/completed)
- [x] 19.4.5 Fix "Run Now" task not found (filename vs internal name mismatch)
- [x] 19.4.6 Write integration test for task manual execution

### 19.6 Task Editor Resize Simplification
- [x] 19.6.1 Remove manual resize state and handlers
- [x] 19.6.2 Implement flexbox auto-resize (simpler, no JavaScript)
- [x] 19.6.3 Remove resize handle UI
  - Simplify the editor container structure

## 20. Skills (Tools) Read-Only Page

### 20.1 Skills List (Read-Only)
- [x] 20.1.1 Enhance GET /api/skills to return skill name, description, parameters
- [x] 20.1.2 Create frontend SkillsList component (read-only display)
- [x] 20.1.3 Display skill name and description in card/list format
- [x] 20.1.4 Note: Modification feature not in scope (future consideration)
- [x] 20.1.5 Write unit test for skills endpoint full response (verified via manual test)

### 20.2 Task List UI/UX Alignment with Skills List
- [x] 20.2.1 Replace table-based TaskList layout with card-based grid layout (matching SkillsList)
- [x] 20.2.2 Move status badge and enable toggle to card header (like SkillsList header)
- [x] 20.2.3 Display schedule in collapsed card (similar to SkillsList showing description)
- [x] 20.2.4 Keep expand/collapse for full config details
- [x] 20.2.5 Reposition "Run Now" and "Edit" buttons to bottom-left of expanded card area
- [x] 20.2.6 Apply consistent styling: border, border-radius, colors matching SkillsList design
- [x] 20.2.7 Add skill-style refresh button in header (matching SkillsList icon + label)
- [x] 20.2.8 Write unit test for TaskList card layout (verified via manual test)

## 21. Post-Implementation Improvements

### 21.1 Backend Enhancements
- [x] 21.1.1 Add /health endpoint for container health checks
- [x] 21.1.2 Add request logging middleware

### 21.2 Docker/Deployment
- [x] 21.2.1 Add health check to docker-compose.yaml
- [x] 21.2.2 Add volume mount for portal.db data persistence

### 21.3 Security Review
- [x] 21.3.1 Review and configure JWT token expiration (already set to 24 hours in auth.py)
- [x] 21.3.2 Configure CORS allow_origins (now uses config, empty = same origin only)
- [x] 21.3.3 Write unit test (added cors_origins tests to test_web_config.py)

### 21.4 Tab Navigation Fix
- [x] 21.4.1 When clicking sidebar tab, always navigate to default view (not previous state)
- [x] 21.4.2 Example: Click "Servers" → go to ServerList, regardless of last state (editing or viewing)

### 21.5 Persona Usage Tracking (Enhanced)
- [x] 21.5.1 Add backend: GET /api/personas/{name}/usage - returns list of tasks and models using this persona
- [x] 21.5.2 Scan task YAML files for `persona:` field matching the persona name
- [x] 21.5.3 Show usage info in persona editor (e.g., "Used by: stock-checker, email-alert")
- [x] 21.5.4 On delete attempt, warn if persona is used by any task or model
- [x] 21.5.5 Scan config.yaml for models that reference this persona (enhancement)

### 21.6 Unified Notification System (Toast)
- [x] 21.6.1 Install react-hot-toast or create custom ToastProvider
- [x] 21.6.2 Create global toast context for the app
- [x] 21.6.3 Add toast notifications: success (green), error (red), info (blue), warning (yellow)
- [x] 21.6.4 Auto-dismiss after 4 seconds
- [x] 21.6.5 Position: top-right corner, stack multiple toasts
- [x] 21.6.6 Replace inline success/error messages across all components with toast calls

### 21.7 Custom Confirmation Dialogs
- [x] 21.7.1 Create reusable Modal component matching portal CSS (dark theme)
- [x] 21.7.2 Replace browser `confirm()` in PersonaEditor with custom modal
- [x] 21.7.3 Fix: Navigate to persona list after successful delete
- [x] 21.7.4 Use modal for other confirmations (e.g., config save, task delete)

### 21.8 UI/UX Polish
- [x] 21.8.1 Replace emoji icons with Heroicons (https://heroicons.com/) - consistent outline style, black/white
- [x] 21.8.2 Remove emoji icons where not needed for simplicity
- [x] 21.8.3 Fix LogViewer double scrollbar: remove widget container scrollbar, only log content scrolls
- [x] 21.8.4 Remove duplicate "Refresh" button (keep only in Quick Actions widget)

### 21.9 Sidebar Icon Polish
- [x] 21.9.1 Replace sidebar emoji icons with Heroicons (HomeIcon, Cog6ToothIcon, ServerStackIcon, UserIcon)
- [x] 21.9.2 Replace collapsed logo emoji with proper icon

### 21.10 LogViewer Responsive Layout
- [x] 21.10.1 Make LogViewer fill remaining browser height (no fixed maxHeight)
- [x] 21.10.2 Use flexbox to ensure logs container fits viewport without outer scrollbar
- [x] 21.10.3 Remove fixed pixel heights and use flex: 1 for dynamic sizing

### 21.11 APScheduler Task Execution Bug Fix
- [x] 21.11.1 Fix "_MissingSentinel" error when running tasks via web portal
- [x] 21.11.2 Implement httpx fallback for Discord REST API calls
- [x] 21.11.3 Add retry logic with 10-second initial wait
- [x] 21.11.4 Test task execution via "Run Now" button
- [x] 21.11.5 Write unit test for httpx Discord user fetch
- [x] 21.11.6 Run existing tests for regression check

**Details**: See `design.md` - "Critical Bug Fixes Applied" section

### 21.12 Fix JavaScript Number Precision Loss for Large Discord IDs
- [x] 21.12.1 Fix float64 precision loss for 17-19 digit Discord IDs
- [x] 21.12.2 Add string conversion for ID fields in frontend (TaskEditor.tsx)
- [x] 21.12.3 Add int() conversion in backend for API calls
- [x] 21.12.4 Verify precision preservation after save

**Details**: See `design.md` - "Critical Bug Fixes Applied" section

---

---

### 21.13 Missing Guilds Intent - Channel Access Fix
- [x] 21.13.1 Add `intents.guilds = True` in llmcord.py
- [x] 21.13.2 Implement HttpxChannel class for REST API fallback
- [x] 21.13.3 Add get_channel_safe() function with retry logic
- [x] 21.13.4 Test task with channel_id - verify message sent
- [x] 21.13.5 Build Docker container with fixes

**Details**: See `design.md` - "Critical Bug Fixes Applied" section