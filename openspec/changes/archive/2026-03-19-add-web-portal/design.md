## Context

The llmcord Discord bot currently runs as a standalone process with no external visibility or administration interface. Configuration requires direct YAML file editing, and operational monitoring relies on console output only.

This design addresses adding a web-based administration portal while:
- Minimizing changes to existing llmcord.py
- Supporting both development and production deployment
- Providing real-time monitoring without performance impact
- Maintaining security with proper authentication

## Goals / Non-Goals

**Goals:**
- Add web portal for bot monitoring (status, logs, servers/channels)
- Add simple config editing (status_message, max_text, allow_dms, etc.)
- Implement real-time log streaming via WebSocket
- Store event logs in SQLite for query and retention
- Use minimal modifications to existing code

**Non-Goals:**
- File write for personas, tasks, or skills (read-only portal)
- Full configuration editor (only simple fields editable)
- Discord OAuth2 authentication (password-based only initially)
- Multi-user collaboration features

## Decisions

### 1. FastAPI Integration
**Decision**: Run FastAPI in same process as Discord bot via asyncio.gather()

**Rationale**: 
- Both are async-native, share event loop cleanly
- Minimal code change in llmcord.py (2-line import + start call)
- Simplifies deployment (single container/service)

**Alternative considered**: Separate web service - rejected for added complexity

### 2. Database
**Decision**: Single SQLite file (data/portal.db)

**Rationale**:
- Simple setup, no external dependency
- Sufficient for single-instance bot
- Configurable path via config.yaml

### 3. Port
**Decision**: Port 8080 default, configurable via PORT env var

**Rationale**: Common alternate port, avoids conflicts with 8000

### 4. Task Editor Resize Approach
**Decision**: Flexbox auto-resize instead of manual drag handle

**Rationale**:
- Manual resize handles with JavaScript state management cause jittery behavior
- Browser natively handles flex container resize smoothly
- Simpler code: `flex: 1` on textarea vs complex `useState` + event listeners
- Works perfectly with window resize without additional event handling

**Implementation**:
```jsx
<div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
  <textarea style={{ flex: 1, minHeight: '200px' }} />
</div>
```

**Iterative improvements**:
1. First attempt: Fixed height with manual resize handle (jittery)
2. Second attempt: Dynamic height with useEffect (complex, still had issues)
3. Final decision: Remove manual resize, use flexbox (simplest, works best)

**Alternative considered**: Manual drag handle - rejected due to complexity and jitter

### 5. Authentication
**Decision**: First-time setup wizard + DB-stored users + optional Discord admin_ids check

**Rationale**:
- No hardcoded passwords anywhere
- First login triggers setup wizard to create admin password
- After setup, users stored in DB (hashed)
- Optional: can also check Discord admin_ids from config for login
- Cleaner security model than env var

**Login Flow**:
```
┌─────────────────────────────────────────┐
│ Request /login                          │
│       │                                 │
│       ▼                                 │
│ ┌──────────────────┐                    │
│ │ DB has users?    │                    │
│ └──────────────────┘                    │
│       │                                 │
│  ┌────┴────┐                             │
│  │         │                             │
│ YES       NO                            │
│  │         │                             │
│  ▼         ▼                             │
│ Login   Setup Wizard                    │
│ Form    (create password)               │
│  │         │                             │
│  └────┬────┘                             │
│       ▼                                  │
│ Validate & JWT                          │
└─────────────────────────────────────────┘
```

**Environment Variable**: 
- `PORTAL_PASSWORD` removed (not needed)
- Optional: `PORTAL_DB` for custom database path (dev/test only)

### 5. Log Capture
**Decision**: Dual handler - console + SQLite

**Rationale**:
- Maintains existing console output
- Adds DB persistence for portal
- Filterable by level (INFO, WARNING, ERROR)

## Risks / Trade-offs

| Risk | Impact | Mitigation |
|------|--------|------------|
| Web server crash affects bot | High | Run in separate task, catch exceptions |
| Log volume overwhelms DB | Medium | Configurable retention, level filtering |
| Concurrent config edits | Low | Single user assumption initially |
| Docker file permissions | Medium | Configurable file paths, volume mounts |

## Migration Plan

1. Add dependencies to requirements.txt
2. Create bot/web/ module (FastAPI + routes)
3. Create bot/db/ module (SQLite models)
4. Add portal section to config.yaml
5. Add 2-line integration to llmcord.py
6. Build React frontend in web/ directory
7. Test in development environment
8. Deploy with updated docker-compose

## Post-Implementation Enhancements (Applied During Development)

### Bot State Integration
The original design had a simple /api/status endpoint, but we enhanced it:
- Added `BotState` singleton class to track Discord bot reference
- Bot registers itself via `set_discord_bot()` and `mark_bot_ready()` in `on_ready` event
- Status API now returns: online status, server count, user count, channel count, uptime, started_at

### Config Reload Feature
Originally, config refresh only reloaded portal config:
- Added `register_config_refs()` to store references to bot's global config
- Added `reload_bot_config()` to reload bot config from disk
- The `/api/refresh` endpoint now reloads BOTH portal and bot config

### UI/UX Guidelines
**Icon Standards:**
- Always use @heroicons/react (24/outline variant) for all UI icons
- Never use emoji characters (➕, 🔄, etc.) as icons - they break visual consistency
- Import from '@heroicons/react/24/outline' and use the IconNameIcon format (e.g., PlusIcon, ArrowPathIcon)
- This applies to all buttons, tabs, and interactive elements throughout the portal
- Always use black or white color for icon, depends on the theme

**Toast Notification Standards:**
- Use react-hot-toast for all user feedback notifications
- Position: top-right corner, stacking multiple toasts vertically
- Duration: 4000ms (4 seconds) auto-dismiss
- Theme: Dark mode matching portal (#333 background, #fff text)
- Color coding:
  - Success: Green background (#10b981) - for successful operations (save, create, delete)
  - Error: Red background (#ef4444) - for failures and validation errors
  - Info: Default dark - for informational messages
  - Warning: Default dark with custom icon - for warnings
- All components must use toast.success(), toast.error(), toast() instead of inline alerts or console.log
- Example: `toast.success('Task saved!')` or `toast.error('Failed to save: ' + error.message)`

### Frontend Updates
- Changed portal name from "llmcord Portal" to "GPT Discord Bot"
- Fixed SPA routing with StaticFiles mounted at "/" with html=True
- Fixed login redirect with React Router's useNavigate
- Fixed config editor to properly parse API response format

### Bug Fixes Applied
- Database session handling (get_db vs get_db_session async context managers)
- Docker network mode (ports mapping instead of host mode for Windows compatibility)
- .dockerignore to include web/dist/ folder

## Open Questions

1. Should log levels be configurable per-component?
2. Session timeout duration? (Recommended: 24 hours)
3. Whether to add HTTPS in production?
4. Whether to support password reset flow?

---

## Phase-Based Implementation Plan

### Phase 1: Bug Fixes & Status Improvements
Focus on fixing existing bugs and improving status page clarity.

**Key Changes:**
- Rename "Servers" → "Guilds", remove "Users" (irrelevant)
- Add bot avatar display
- Fix "[object Object]" in config page
- Add three-button config workflow (Save/Apply/Save&Apply)
- Fix Chinese character encoding with UTF-8/Unicode support

### Phase 2: Servers Page Enhancement
Add server detail subpage for guild management.

**Key Changes:**
- Fix member/channel display
- Add server icon display
- Create /servers/:id subpage with member/channel lists
- Add permission management UI (grant/revoke)

### Phase 3: Persona Management Page
Full CRUD for persona files with markdown editor.

**Key Changes:**
- List existing personas with title and content preview
- Markdown editor with live preview
- Four buttons: Add, Save, Apply, Save&Apply, Delete
- Save to bot/config/personas/*.md

### Phase 4: Task Management Page
Full CRUD for task files with YAML editor and execution control.

**Key Changes:**
- List existing tasks in expandable table
- YAML editor with syntax validation
- Toggle enable/disable per task
- "Run Now" button for manual execution
- Save to bot/config/tasks/*.yaml
- After save, call POST /api/tasks/{name}/reload to reload single task in scheduler (not full reload)

### Phase 5: Skills (Tools) Read-Only Page
Display available skills/tools (read-only for now).

**Key Changes:**
- List skill name and description
- No modification (future consideration)

## Cleanup Tasks (Post-Implementation)

- [x] Remove PORTAL_PASSWORD from .env and .env.example
- [x] Remove password field from config.yaml portal section
- [x] Update config-example.yaml to remove password reference
- [x] All cleanup tasks completed

---

## Critical Bug Fixes Applied

### Bug Fix 1: APScheduler + Discord HTTP Session Isolation

**Problem**: Tasks triggered via "Run Now" button failed with `_MissingSentinel` error, but worked via slash commands or cron jobs.

**Root Cause**: APScheduler runs tasks in a separate async context from Discord's gateway connection. The `discord_bot.http` session isn't accessible from this separate context.

**Solution**: 
- Added 10-second initial wait at task start
- Implemented retry logic (3 attempts)
- Primary: Use `discord_bot.get_user()` from cache
- Fallback: Use global `httpx_client` to call Discord REST API directly with bot token
- Construct `discord.User` object from API response using `discord_bot._get_state()`

**Files Modified**: `llmcord.py`, `bot/web/routes/tasks.py`

---

### Bug Fix 2: JavaScript Number Precision Loss for Large Discord IDs

**Problem**: Discord IDs (17-19 digits) were corrupted when saving tasks due to JavaScript float64 precision limits.

**Root Cause**: JavaScript's `Number` type uses float64 (52-bit mantissa) which can only safely represent ~15.95 decimal digits.

**Solution**:
- **Frontend** (TaskEditor.tsx): After YAML parse, explicitly convert ID fields to strings
- **Backend** (llmcord.py): Convert string IDs back to integers before Discord API calls

```typescript
// Frontend - convert after YAML parse
const idFields = ['user_id', 'channel_id', 'user_id|channel_id']
for (const field of idFields) {
  if (parsed[field] !== undefined) {
    parsed[field] = String(parsed[field])
  }
}
```

```python
# Backend - convert before API calls
channel_id = int(channel_id) if isinstance(channel_id, str) else channel_id
user_id = int(user_id) if isinstance(user_id, str) else user_id
```

**Files Modified**: `web/src/components/TaskEditor.tsx`, `llmcord.py`

---

### Bug Fix 3: Missing Guilds Intent - Channel Access Failure

**Problem**: Tasks with `channel_id` failed with "Unknown Channel" error (HTTP 404).

**Root Cause**:
1. Bot was initialized without `intents.guilds = True`
2. Without guilds intent, bot cannot access guild channels via REST API or cache

**Solution**:
- Added `intents.guilds = True` to Discord client initialization
- Added `HttpxChannel` class for REST API fallback when Discord.py can't access channel
- Added `get_channel_safe()` function with retry logic

```python
intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True
intents.guilds = True  # Required for channel access
discord_bot = discord.Client(intents=intents)
```

**Files Modified**: `llmcord.py`

---

### Bug Fix 4: Task Filename vs Internal Name Mismatch

**Problem**: "Run Now" returned "Task not found" for tasks like `stock-market-checker.yaml`.

**Root Cause**: Filename uses kebab-case (`stock-market-checker.yaml`) but internal YAML uses underscore (`name: stock_market_check`).

**Solution**: Modified `run_task()` and `reload_single_task()` to try BOTH filename AND name field lookup.

---

### Bug Fix 5: FastAPI Request Body Parameter Extraction

**Problem**: Task create/update endpoints failed silently.

**Root Cause**: FastAPI couldn't extract parameters from JSON body without Pydantic model.

**Solution**: Added `TaskCreate` and `TaskUpdate` Pydantic models for proper request body parsing.

---

### Bug Fix 6: Tab Navigation - Clicking Same Tab Doesn't Reset Views

**Problem**: Clicking the currently active tab (e.g., "Personas" while already on Personas) didn't collapse expanded items or return to list view. Users expected clicking a tab to always show the default/list view.

**Root Cause**: 
1. `handleTabChange` in Dashboard.tsx had `if (tab !== activeTab)` condition - only reset views when clicking DIFFERENT tabs
2. TaskList component had local `expandedTask` state that wasn't being reset when switching tabs

**Solution**:
- Removed `if (tab !== activeTab)` condition so clicking same tab always resets views
- Added `tasksKey` state in Dashboard.tsx to force TaskList re-render when clicking Tasks tab
- When clicking "Tasks" tab, increment `tasksKey` to remount TaskList and reset expanded state

```typescript
// Dashboard.tsx - handleTabChange
const handleTabChange = (tab: TabType) => {
  // Always reset views when clicking a tab (even if same tab)
  setEditingPersona(null)
  setEditingTask(undefined)
  if (tab === 'tasks') {
    setTasksKey(k => k + 1)  // Force TaskList re-render
  }
  if (tab !== 'servers' && onCloseServer) {
    onCloseServer()
  }
  setActiveTab(tab)
}
```

**Files Modified**: `web/src/components/Dashboard.tsx`

---

## Security Enhancements

### JWT Token Expiration
- Already configured: `ACCESS_TOKEN_EXPIRE_HOURS = 24` in `bot/web/auth.py`
- Tokens expire after 24 hours, requiring re-authentication

### CORS Restriction
- Previously: `allow_origins=["*"]` - any website could access the API
- Now: Reads from config `cors_origins` setting
- Default: `cors_origins: []` (empty) = same origin only - only the portal itself can access the API
- For cross-origin access (e.g., separate frontend): add URLs to config, e.g., `cors_origins: ["http://localhost:3000"]`

**Files Modified**: 
- `bot/web/config.py` - added `cors_origins` property
- `bot/web/server.py` - reads CORS origins from config
- `config.yaml` - added `cors_origins: []` setting
- `bot/test/test_web_config.py` - added unit tests for cors_origins