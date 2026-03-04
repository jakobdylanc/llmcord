"""
Google Tools for Gmail and Calendar access.

Use google_tools when:
- The user wants to check their emails
- The user wants to see their calendar events
- The user wants to read a specific email by ID

Security: Uses read-only OAuth scopes (gmail.readonly, calendar.readonly)
"""

from __future__ import annotations

import os
import base64
import logging
import html
import json

from datetime import datetime
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Read-only scopes only - no write permissions
SCOPES = [
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/gmail.readonly",
]

# Default configuration
DEFAULT_CALENDAR_ENTRIES = 10
DEFAULT_EMAIL_ENTRIES = 10
DEFAULT_CREDENTIALS_PATH = "credentials.json"

# ── Logging ─────────────────────────────────────────────────────────────────

def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("google_tools")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
    return logger

logger = _setup_logger()

# Store remaining emails for follow-up retrieval
_remaining_emails: list = []


# ── Configuration ──────────────────────────────────────────────────────────

def get_config() -> dict:
    """Get configuration from config.yaml or environment."""
    config = {
        "default_calendar_entries": DEFAULT_CALENDAR_ENTRIES,
        "default_email_entries": DEFAULT_EMAIL_ENTRIES,
        "path_to_credentials": DEFAULT_CREDENTIALS_PATH,
    }
    
    # Try to load from config.yaml
    try:
        from bot.config.loader import get_config as load_config
        yaml_config = load_config()
        tools_config = yaml_config.get("tools", {}).get("google_tools", {})
        if tools_config:
            config.update(tools_config)
    except Exception:
        pass
    
    # Environment variables override
    if os.getenv("GOOGLE_TOOLS_CALENDAR_ENTRIES"):
        config["default_calendar_entries"] = int(os.getenv("GOOGLE_TOOLS_CALENDAR_ENTRIES"))
    if os.getenv("GOOGLE_TOOLS_EMAIL_ENTRIES"):
        config["default_email_entries"] = int(os.getenv("GOOGLE_TOOLS_EMAIL_ENTRIES"))
    if os.getenv("GOOGLE_TOOLS_CREDENTIALS_PATH"):
        config["path_to_credentials"] = os.getenv("GOOGLE_TOOLS_CREDENTIALS_PATH")
    
    return config


# ── Credentials ────────────────────────────────────────────────────────────

def get_google_creds(config: dict) -> Credentials | None:
    """Get Google OAuth credentials, refreshing if needed."""
    creds = None
    
    # Resolve paths from bot directory
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    token_path = os.path.join(base_dir, "token.json")
    
    try:
        if os.path.exists(token_path):
            creds = Credentials.from_authorized_user_file(token_path, SCOPES)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                credentials_path = config.get("path_to_credentials", DEFAULT_CREDENTIALS_PATH)
                # Resolve relative paths from bot directory
                if not os.path.isabs(credentials_path):
                    credentials_path = os.path.join(base_dir, credentials_path)
                
                if os.path.exists(credentials_path):
                    flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
                    creds = flow.run_local_server(port=0)
                    with open(token_path, "w") as f:
                        f.write(creds.to_json())
                else:
                    logger.error(f"Credentials file not found: {credentials_path}")
                    return None
    except Exception as e:
        logger.error(f"Error getting credentials: {e}")
        # Try to recover by removing stale token
        if os.path.exists(token_path):
            try:
                os.remove(token_path)
            except Exception:
                pass
        return None
    
    return creds


# ── Helper Functions ──────────────────────────────────────────────────────

def _get_header_value(headers: list, name: str) -> str:
    """Extract header value from list of headers."""
    for header in headers:
        if header.get("name", "").lower() == name.lower():
            return header.get("value", "")
    return ""


def _decode_email_body(payload: dict) -> str:
    """Decode email body from payload."""
    try:
        mime_type = payload.get("mimeType", "")
        
        if mime_type in ("multipart/alternative", "multipart/mixed"):
            for part in payload.get("parts", []):
                part_mime = part.get("mimeType", "")
                if part_mime in ("text/html", "text/plain"):
                    data = part.get("body", {}).get("data")
                    if data:
                        return base64.b64decode(data).decode("utf-8", errors="replace")
        
        # Direct body
        data = payload.get("body", {}).get("data")
        if data:
            return base64.b64decode(data).decode("utf-8", errors="replace")
        
        # Parts without body data
        for part in payload.get("parts", []):
            result = _decode_email_body(part)
            if result:
                return result
                
    except Exception as e:
        return f"Error decoding body: {e}"
    
    return ""


def _get_current_time() -> str:
    """Get current UTC time in ISO format."""
    return datetime.utcnow().isoformat() + "Z"


def _strip_html(text: str) -> str:
    """Remove HTML tags from text for readable plain text output."""
    import re
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Decode common HTML entities
    text = text.replace("&nbsp;", " ")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&amp;", "&")
    text = text.replace("&quot;", '"')
    text = text.replace("&#39;", "'")
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _format_email(mail: dict, service: Any, max_body_length: int = 1500) -> str:
    """Format email data into a readable string."""
    headers = mail.get("payload", {}).get("headers", [])
    
    msg_id = mail.get("id", "")
    date = _get_header_value(headers, "Date")
    sender = _get_header_value(headers, "From")
    subject = _get_header_value(headers, "Subject")
    snippet = mail.get("snippet", "")
    label_ids = mail.get("labelIds", [])
    is_unread = "UNREAD" in label_ids
    
    # Get full body if needed
    body = ""
    try:
        body = _decode_email_body(mail.get("payload", {}))
    except Exception as e:
        body = f"[Could not decode body: {e}]"
    
    # Strip HTML tags for plain text output
    body = _strip_html(body)
    
    # Truncate body to max_body_length
    if len(body) > max_body_length:
        body = body[:max_body_length] + "..."
    
    lines = [
        f"ID: {msg_id}",
        f"Date: {date}",
        f"From: {sender}",
        f"Subject: {subject}",
        f"Unread: {'Yes' if is_unread else 'No'}",
        f"",
        f"Body:",
        body,
    ]
    
    return "\n".join(lines)


def _format_event(event: dict) -> str:
    """Format calendar event into a readable string."""
    start = event.get("start", {})
    end = event.get("end", {})
    
    start_time = start.get("dateTime", start.get("date", ""))
    end_time = end.get("dateTime", end.get("date", ""))
    summary = event.get("summary", "(No title)")
    calendar = event.get("calendar", "Unknown")
    location = event.get("location", "")
    
    lines = [
        f"Title: {summary}",
        f"Start: {start_time}",
        f"End: {end_time}",
        f"Calendar: {calendar}",
    ]
    
    if location:
        lines.append(f"Location: {location}")
    
    return "\n".join(lines)


# ── Main Functions ────────────────────────────────────────────────────────

def get_user_emails(count: int = -1, label_id: str = "INBOX", query: str = "", max_body_length: int = 1500) -> str:
    """
    Retrieve emails from user's Gmail inbox.
    
    Args:
        count: Number of emails to fetch (default from config, or 10)
        label_id: Gmail label (INBOX, UNREAD, STARRED, SENT, IMPORTANT)
        query: Gmail search query (e.g., "in:inbox -category:promotions -in:spam")
        max_body_length: Maximum length of email body to include (default 1500)
    
    Returns:
        Formatted string with email details. If truncated, includes marker for remaining emails.
    """
    config = get_config()
    
    if count <= 0:
        count = config.get("default_email_entries", DEFAULT_EMAIL_ENTRIES)
    
    # Validate count
    count = max(1, min(count, 50))  # Limit between 1-50
    
    # Validate label_id (basic sanitization)
    label_id = label_id.upper()
    allowed_labels = {"INBOX", "UNREAD", "STARRED", "IMPORTANT", "SENT", "SPAM", "TRASH"}
    if label_id not in allowed_labels:
        label_id = "INBOX"
    
    # Build API request
    request_params = {
        "userId": "me",
        "maxResults": count,
    }
    
    # Use query if provided, otherwise use labelIds
    if query:
        request_params["q"] = query
    else:
        request_params["labelIds"] = [label_id]
    
    creds = get_google_creds(config)
    if not creds:
        return "Error: Could not authenticate with Google. Please check credentials.json and token.json"
    
    try:
        service = build("gmail", "v1", credentials=creds)
        
        results = service.users().messages().list(**request_params).execute()
        
        messages = results.get("messages", [])
        
        if not messages:
            return f"No emails found"
        
        output = [f"Found {len(messages)} email(s):", ""]
        
        total_length = 0
        max_total = 1800  # Leave room for header/footer
        
        global _remaining_emails
        _remaining_emails = []  # Reset stored emails
        
        for msg in messages:
            mail = service.users().messages().get(
                userId="me", 
                id=msg["id"],
                format="full"
            ).execute()
            
            email_text = _format_email(mail, service, max_body_length)
            email_text += "\n" + "-" * 40
            
            # Check if adding this email would exceed limit
            if total_length + len(email_text) > max_total:
                # Store remaining emails for follow-up
                _remaining_emails.append({"id": msg["id"], "mail": mail, "formatted": email_text})
                output.append(f"... (showing {len(output) - 2} of {len(messages)} emails)")
                output.append(f"📬 Use `google_tools(action='get_remaining_emails')` to get remaining emails")
                break
                
            output.append(email_text)
            total_length += len(email_text)
        
        return "\n".join(output)
    
    except HttpError as e:
        return f"Google API error: {e}"
    except Exception as e:
        logger.error(f"Error fetching emails: {e}")
        return f"Error fetching emails: {e}"


def get_email_content(message_id: str) -> str:
    """
    Get the full content of a specific email by message ID.
    
    Args:
        message_id: The Gmail message ID
    
    Returns:
        Formatted string with email content
    """
    # Validate message_id format (basic sanitization)
    if not message_id or len(message_id) < 5:
        return "Error: Invalid message ID"
    
    # Only allow alphanumeric and limited characters
    if not all(c.isalnum() or c in "-_" for c in message_id):
        return "Error: Invalid message ID format"
    
    config = get_config()
    creds = get_google_creds(config)
    
    if not creds:
        return "Error: Could not authenticate with Google. Please check credentials.json and token.json"
    
    try:
        service = build("gmail", "v1", credentials=creds)
        
        mail = service.users().messages().get(
            userId="me",
            id=message_id,
            format="full"
        ).execute()
        
        return _format_email(mail, service)
    
    except HttpError as e:
        if e.resp.status == 404:
            return f"Error: Message not found or access denied"
        return f"Google API error: {e}"
    except Exception as e:
        logger.error(f"Error fetching email: {e}")
        return f"Error fetching email: {e}"


def get_user_events(count: int = -1, calendar_id: str = "primary") -> str:
    """
    Get upcoming calendar events.
    
    Args:
        count: Number of events to fetch (default from config, or 10)
        calendar_id: Calendar ID (default: 'primary')
    
    Returns:
        Formatted string with event details
    """
    config = get_config()
    
    if count <= 0:
        count = config.get("default_calendar_entries", DEFAULT_CALENDAR_ENTRIES)
    
    # Validate count
    count = max(1, min(count, 50))  # Limit between 1-50
    
    # Validate calendar_id
    if not calendar_id or calendar_id == "primary":
        calendar_id = "primary"
    
    creds = get_google_creds(config)
    if not creds:
        return "Error: Could not authenticate with Google. Please check credentials.json and token.json"
    
    try:
        service = build("calendar", "v3", credentials=creds)
        
        now = _get_current_time()
        
        events_result = service.events().list(
            calendarId=calendar_id,
            timeMin=now,
            maxResults=count,
            singleEvents=True,
            orderBy="startTime",
        ).execute()
        
        events = events_result.get("items", [])
        
        if not events:
            return f"No upcoming events found"
        
        output = [f"Found {len(events)} upcoming event(s):", ""]
        
        for event in events:
            output.append(_format_event(event))
            output.append("-" * 40)
        
        return "\n".join(output)
    
    except HttpError as e:
        return f"Google API error: {e}"
    except Exception as e:
        logger.error(f"Error fetching events: {e}")
        return f"Error fetching events: {e}"


# ── Schemas ────────────────────────────────────────────────────────────────

GOOGLE_TOOLS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "google_tools",
        "description": "Tools for accessing Gmail emails and Google Calendar events. Use these when the user wants to check their emails or see upcoming calendar events.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "The action to perform: 'get_emails', 'get_email_content', 'get_events', or 'get_remaining_emails'",
                    "enum": ["get_emails", "get_email_content", "get_events", "get_remaining_emails"]
                },
                "count": {
                    "type": "integer", 
                    "description": "Number of items to fetch (default: 10, max: 50)"
                },
                "label_id": {
                    "type": "string",
                    "description": "Gmail label to fetch from (INBOX, UNREAD, STARRED, SENT, IMPORTANT)",
                    "default": "INBOX"
                },
                "query": {
                    "type": "string",
                    "description": "Gmail search query (e.g., 'in:inbox -category:promotions -in:spam'). Use this to filter emails instead of label_id."
                },
                "message_id": {
                    "type": "string",
                    "description": "Gmail message ID (required for get_email_content action)"
                },
                "calendar_id": {
                    "type": "string",
                    "description": "Calendar ID (default: 'primary')",
                    "default": "primary"
                }
            },
            "required": ["action"],
        },
    },
}


def get_remaining_emails(max_body_length: int = 1500) -> str:
    """
    Get remaining emails that were truncated from a previous get_emails call.
    
    Args:
        max_body_length: Maximum length of email body to include (default 1500)
    
    Returns:
        Formatted string with remaining email details
    """
    global _remaining_emails
    
    if not _remaining_emails:
        return "No remaining emails. Run get_emails first to store truncated emails."
    
    output = [f"Remaining {len(_remaining_emails)} email(s):", ""]
    
    for email_data in _remaining_emails:
        output.append(email_data["formatted"])
    
    _remaining_emails = []  # Clear after retrieval
    return "\n".join(output)


def google_tools_wrapper(
    action: str,
    count: int = -1,
    label_id: str = "INBOX",
    message_id: str = "",
    calendar_id: str = "primary",
    query: str = ""
) -> str:
    """
    Unified wrapper for Google tools.
    
    Args:
        action: One of 'get_emails', 'get_email_content', 'get_events', 'get_remaining_emails'
        count: Number of items to fetch
        label_id: Gmail label (for get_emails)
        message_id: Message ID (for get_email_content)
        calendar_id: Calendar ID (for get_events)
        query: Gmail search query (for get_emails)
    
    Returns:
        Formatted result string
    """
    if action == "get_emails":
        return get_user_emails(count=count, label_id=label_id, query=query)
    elif action == "get_remaining_emails":
        return get_remaining_emails()
    elif action == "get_email_content":
        if not message_id:
            return "Error: message_id is required for get_email_content"
        return get_email_content(message_id=message_id)
    elif action == "get_events":
        return get_user_events(count=count, calendar_id=calendar_id)
    else:
        return f"Error: Unknown action '{action}'. Use 'get_emails', 'get_email_content', or 'get_events'"
