---
name: google_tools
description: Access Gmail emails and Google Calendar events (read-only).
metadata: {"clawhub":{"emoji":"📧","tools":["google_tools"]}}
---

# google_tools

Use this tool to access the user's Gmail inbox and Google Calendar. All operations are **read-only** for security.

## When to use google_tools

- User wants to check their emails
- User wants to read a specific email
- User wants to see upcoming calendar events
- User asks "do I have any new emails?" or "what's on my calendar today?"

## Security

- Uses read-only OAuth scopes (`gmail.readonly`, `calendar.readonly`)
- Cannot send, delete, or modify emails
- Cannot create or modify calendar events

## Tool signature

```
google_tools(action: str, count: int = 10, label_id: str = "INBOX", message_id: str = "", calendar_id: str = "primary", query: str = "")
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| action | string | Required. One of: `get_emails`, `get_email_content`, `get_events`, `get_remaining_emails` |
| count | integer | Number of items to fetch (default: 10, max: 50) |
| label_id | string | Gmail label for `get_emails`: `INBOX`, `UNREAD`, `STARRED`, `SENT`, `IMPORTANT` |
| query | string | Gmail search query (e.g., `in:inbox -category:promotions -in:spam`). Use instead of label_id for advanced filtering |
| message_id | string | Required for `get_email_content` action |
| calendar_id | string | Calendar ID for `get_events` (default: `primary`) |

## Usage examples

```python
# Get recent emails from inbox
google_tools(action="get_emails", count=10, label_id="INBOX")

# Get emails excluding promotions and spam
google_tools(action="get_emails", query="in:inbox -category:promotions -in:spam")

# Get remaining emails (if truncated)
google_tools(action="get_remaining_emails")

# Get unread emails
google_tools(action="get_emails", count=5, label_id="UNREAD")

# Get full content of a specific email (need message ID from get_emails)
google_tools(action="get_email_content", message_id="abc123def456")

# Get upcoming calendar events
google_tools(action="get_events", count=10, calendar_id="primary")
```

## Result handling

- Results are formatted as readable text with email/event details
- For emails: shows date, sender, subject, unread status, and body preview
- For calendar: shows title, start/end time, location, and calendar name
- Error messages are sanitized for security (no internal paths exposed)

## Setup Prerequisites

Before using this tool, the bot owner must set up OAuth credentials:

1. **Get OAuth credentials from Google Cloud Console:**
   - Go to https://console.cloud.google.com/apis/credentials
   - Create a project (or use existing)
   - Enable Gmail API and Google Calendar API
   - Create OAuth 2.0 Client ID (Desktop app)
   - Download the JSON file as `credentials.json`

2. **Place credentials.json** in the bot root directory (or path specified in config)

3. **First run:** The first time the tool is used, it will open a browser window for OAuth authentication. A `token.json` will be created for subsequent runs.

## Configuration

In `config.yaml`:

```yaml
tools:
  google_tools:
    default_calendar_entries: 10    # Number of calendar events to fetch
    default_email_entries: 10       # Number of emails to fetch
    path_to_credentials: "credentials.json"  # OAuth client secrets file
```

Environment variables (optional overrides):
- `GOOGLE_TOOLS_CALENDAR_ENTRIES`
- `GOOGLE_TOOLS_EMAIL_ENTRIES`
- `GOOGLE_TOOLS_CREDENTIALS_PATH`