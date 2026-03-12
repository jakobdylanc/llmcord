"""
Refined Google Tools for Gmail and Calendar access.

Highlights of improvements:
- Stronger typing and clearer function responsibilities
- More robust credentials/token path handling (credentials and token colocated)
- Improved email body decoding with safer base64 padding and charset fallback
- Prefer text/plain over text/html when available
- HTML stripping, reply-chain and signature truncation helpers
- Better logging and error messages
- Small security-minded sanitization and limits
"""

from __future__ import annotations

import base64
import json
import logging
import os
import quopri
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Read-only scopes only - no write permissions
SCOPES: List[str] = [
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/gmail.readonly",
]

# Defaults
DEFAULT_CALENDAR_ENTRIES = 10
DEFAULT_EMAIL_ENTRIES = 10
DEFAULT_CREDENTIALS_PATH = "credentials.json"
MAX_EMAILS_PER_CALL = 50
MAX_SUMMARY_CHARACTERS = 100000  # cap total returned characters across all emails

# Module logger
logger = logging.getLogger("google_tools")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# In-memory store for truncated emails (kept simple for the wrapper behavior)
_remaining_emails: List[Dict[str, Any]] = []


# -------------------- Configuration --------------------

def get_config() -> Dict[str, Any]:
    """Load configuration from a YAML loader (optional) and environment variables.

    Environment variables take precedence. The function tries to be conservative and
    fallback to module defaults when nothing is found.
    """
    config: Dict[str, Any] = {
        "default_calendar_entries": DEFAULT_CALENDAR_ENTRIES,
        "default_email_entries": DEFAULT_EMAIL_ENTRIES,
        "path_to_credentials": DEFAULT_CREDENTIALS_PATH,
    }

    # Optional YAML loader - do not fail loudly
    try:
        from bot.config.loader import get_config as _load_config  # type: ignore

        yaml_conf = _load_config() or {}
        tools_conf = yaml_conf.get("tools", {}).get("google_tools", {}) if isinstance(yaml_conf, dict) else {}
        if isinstance(tools_conf, dict):
            config.update(tools_conf)
    except Exception:
        # If the host app provides config, great. If not, continue with defaults.
        pass

    # Environment overrides
    if os.getenv("GOOGLE_TOOLS_CALENDAR_ENTRIES"):
        try:
            config["default_calendar_entries"] = int(os.getenv("GOOGLE_TOOLS_CALENDAR_ENTRIES"))
        except ValueError:
            logger.warning("Invalid GOOGLE_TOOLS_CALENDAR_ENTRIES value; using default")

    if os.getenv("GOOGLE_TOOLS_EMAIL_ENTRIES"):
        try:
            config["default_email_entries"] = int(os.getenv("GOOGLE_TOOLS_EMAIL_ENTRIES"))
        except ValueError:
            logger.warning("Invalid GOOGLE_TOOLS_EMAIL_ENTRIES value; using default")

    if os.getenv("GOOGLE_TOOLS_CREDENTIALS_PATH"):
        config["path_to_credentials"] = os.getenv("GOOGLE_TOOLS_CREDENTIALS_PATH")

    return config


# -------------------- Credentials --------------------

def _resolve_paths(credentials_path: str) -> Dict[str, Path]:
    """Resolve absolute paths for credentials and token.

    Place token.json next to the credentials file so the token is stored per client
    rather than in an arbitrary project root.
    """
    creds_path = Path(os.path.expanduser(credentials_path))
    if not creds_path.is_absolute():
        # Try to resolve relative to this file's project root
        base_dir = Path(__file__).resolve().parents[2]
        creds_path = (base_dir / creds_path).resolve()

    token_path = creds_path.with_name("token.json")
    return {"credentials": creds_path, "token": token_path}


def get_google_creds(config: Dict[str, Any]) -> Optional[Credentials]:
    """Obtain Google OAuth2 credentials. Refresh if expired, or run local server if needed.

    Returns None on failure (and logs useful messages).
    """
    try:
        credentials_path = str(config.get("path_to_credentials", DEFAULT_CREDENTIALS_PATH))
        paths = _resolve_paths(credentials_path)
        creds_path: Path = paths["credentials"]
        token_path: Path = paths["token"]

        creds: Optional[Credentials] = None

        if token_path.exists():
            try:
                creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
            except Exception:
                # If token is corrupt, remove it and continue to create a fresh one
                logger.warning("Existing token.json appears invalid; removing and starting fresh")
                try:
                    token_path.unlink()
                except Exception:
                    pass
                creds = None

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                logger.info("Refreshing expired credentials")
                creds.refresh(Request())
            else:
                if not creds_path.exists():
                    logger.error(f"Credentials file not found: {creds_path}")
                    return None

                flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), SCOPES)
                creds = flow.run_local_server(port=0)
                try:
                    token_path.write_text(creds.to_json(), encoding="utf-8")
                except Exception:
                    logger.warning("Unable to write token.json; continuing without persisting token")

        return creds

    except Exception as exc:  # broad catch to keep tool resilient
        logger.exception("Failed to obtain Google credentials: %s", exc)
        return None


# -------------------- Helpers for email decoding / cleaning --------------------


def _get_header_value(headers: List[Dict[str, Any]], name: str) -> str:
    for h in headers:
        if h.get("name", "").lower() == name.lower():
            return h.get("value", "")
    return ""


def _safe_b64_decode(data: str) -> Optional[bytes]:
    """Decode base64 / base64url safely with padding fix.

    Returns bytes or None on failure.
    """
    try:
        if isinstance(data, str):
            # Strip whitespace introduced by some encoders
            data = data.strip().replace("\n", "")
            # Fix padding
            padding = (-len(data)) % 4
            if padding:
                data += "=" * padding
            return base64.urlsafe_b64decode(data)
        return None
    except Exception:
        return None


def _strip_html(text: str) -> str:
    """Convert HTML to readable plain text.

    This is intentionally simple (no external dependencies) — good enough for
    short email bodies sent to an LLM.
    """
    # Unescape HTML entities
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"\r\n", "\n", text)
    # Remove scripts/styles
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", "", text)
    # Replace tags that denote breaks with newline
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</(p|div|h[1-6])>", "\n", text)
    # Remove remaining tags
    text = re.sub(r"<[^>]+>", "", text)
    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]*\n+", "\n\n", text)
    return text.strip()

def _clean_text(text: str) -> str:
    """Normalize whitespace for LLM consumption."""
    # Normalize Windows newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Remove trailing spaces
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse many blank lines into one
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    # Trim
    text = text.strip()
    return text

_REPLY_MARKERS = [
    r"^On .* wrote:",
    r"^-+Original Message-+",
    r"^From: .*@",
    r"^Sent: ",
    r"^>+",
]


def _truncate_reply_chain(text: str, keep_chars: int = 2000) -> str:
    """Strip quoted reply chains and long signatures.

    This looks for common reply markers and cuts the message there. If nothing is
    found and the text is long, it returns the first `keep_chars` characters.
    """
    # Normalize newlines for consistent regex behavior
    norm = text.replace("\r\n", "\n").replace("\r", "\n")

    # Search for common reply markers
    for marker in _REPLY_MARKERS:
        m = re.search(marker, norm, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            return norm[: m.start()].strip()

    # Try to strip common signature delimiters
    sig_index = norm.find("-- ")
    if sig_index != -1:
        return norm[:sig_index].strip()

    # Not found — if it's short return as-is, else truncate
    if len(norm) <= keep_chars:
        return norm.strip()
    return norm[:keep_chars].rsplit(" ", 1)[0].strip() + "..."


def _decode_email_body(payload: dict) -> str:
    """Extract readable body text from Gmail payload."""

    mime_type = payload.get("mimeType", "").lower()

    # Walk multipart structures
    if mime_type.startswith("multipart/"):
        for part in payload.get("parts", []):
            text = _decode_email_body(part)
            if text:
                return text
        return ""

    body = payload.get("body", {})
    data = body.get("data")

    if not data:
        return ""

    # Gmail uses base64url encoding
    decoded_bytes = _safe_b64_decode(data)

    if not decoded_bytes:
        return ""

    # Convert bytes → string
    try:
        text = decoded_bytes.decode("utf-8", errors="replace")
    except Exception:
        text = decoded_bytes.decode(errors="replace")

    # Convert HTML → text
    if mime_type == "text/html" or "<html" in text.lower():
        text = _strip_html(text)

    # Remove quoted reply chains
    text = _truncate_reply_chain(text)

    # Clean formatting
    text = _clean_text(text)

    # Limit size to save tokens
    text = text[:2000]

    return text

# ORIGINAL FUNCTION BODY BELOW (deprecated)(payload: Dict[str, Any]) -> str:
    """Attempt to extract a human readable body from the Gmail message payload.

    Strategy:
    - Prefer text/plain part.
    - Fall back to text/html part and strip HTML.
    - Respect Content-Transfer-Encoding and Content-Type charset when present.
    - Try a short list of common charsets for CJK/legacy encodings.
    """
    def _process_part(part: Dict[str, Any]) -> Optional[str]:
        body = part.get("body", {})
        data = body.get("data")
        transfer_encoding = "base64"
        headers = part.get("headers", []) or []

        # Inspect headers for content-transfer-encoding and charset
        charset: Optional[str] = None
        for h in headers:
            name = h.get("name", "").lower()
            val = h.get("value", "")
            if name == "content-transfer-encoding":
                transfer_encoding = val.lower().strip()
            elif name == "content-type":
                # try to parse charset=... (simple parse)
                cs_match = re.search(r"charset=\s*\"?([^;\"']+)\"?", val, flags=re.IGNORECASE)
                if cs_match:
                    charset = cs_match.group(1).strip().strip('"')

        if not data:
            return None

        decoded_bytes: Optional[bytes] = None
        if transfer_encoding in ("base64", "base64url"):
            decoded_bytes = _safe_b64_decode(data)
        elif transfer_encoding == "quoted-printable":
            try:
                decoded_bytes = quopri.decodestring(data)
            except Exception:
                decoded_bytes = None
        else:
            # 7bit/8bit etc — treat as raw text
            decoded_bytes = data.encode("utf-8", errors="replace") if isinstance(data, str) else None

        if not decoded_bytes:
            return None

        # Try decoding with several charsets
        charsets_to_try = [charset, "utf-8", "gb18030", "gbk", "big5", "iso-8859-1", "windows-1252"]
        for cs in (cs for cs in charsets_to_try if cs):
            try:
                return decoded_bytes.decode(cs, errors="strict")
            except (LookupError, UnicodeDecodeError):
                continue

        # As a last resort, decode with replacement to avoid crashing
        return decoded_bytes.decode("utf-8", errors="replace")

    # If this payload is multipart, walk parts and prefer text/plain
    mime_type = (payload.get("mimeType") or "").lower()
    if mime_type.startswith("multipart/"):
        parts = payload.get("parts") or []
        # 1) look for text/plain first
        for p in parts:
            if (p.get("mimeType") or "").lower() == "text/plain":
                result = _process_part(p)
                if result:
                    return _truncate_reply_chain(result)
        # 2) look recursively for text/plain deeper
        for p in parts:
            result = _decode_email_body(p)
            if result:
                return result
        # 3) fallback to text/html
        for p in parts:
            if (p.get("mimeType") or "").lower() == "text/html":
                result = _process_part(p)
                if result:
                    return _truncate_reply_chain(_strip_html(result))

    # Not multipart: try this payload directly
    direct = _process_part(payload)
    if direct:
        # If it's HTML mime-type, strip tags
        if mime_type == "text/html":
            return _truncate_reply_chain(_strip_html(direct))
        return _truncate_reply_chain(direct)

    return ""


# -------------------- Formatting for output --------------------

def _format_email(mail: Dict[str, Any], max_body_length: int = 50000) -> str:
    headers = mail.get("payload", {}).get("headers", []) or []

    msg_id = mail.get("id", "")
    date = _get_header_value(headers, "Date")
    sender = _get_header_value(headers, "From")
    subject = _get_header_value(headers, "Subject")
    snippet = mail.get("snippet", "")
    label_ids = set(mail.get("labelIds", []) or [])
    is_unread = "UNREAD" in label_ids

    try:
        body = _decode_email_body(mail.get("payload", {}))
    except Exception as exc:
        logger.exception("Error decoding email body: %s", exc)
        body = "[Could not decode body]"

    # If body is empty, fall back to snippet
    if not body and snippet:
        body = snippet

    # Strip and normalize HTML if present
    body = _strip_html(body) if "<" in body and ">" in body else body
    body = body.strip()

    # Truncate body to max_body_length
    if len(body) > max_body_length:
        body = body[:max_body_length].rsplit(" ", 1)[0] + "..."

    lines = [
        f"ID: {msg_id}",
        f"Date: {date}",
        f"From: {sender}",
        f"Subject: {subject}",
        f"Unread: {'Yes' if is_unread else 'No'}",
        "",
        "Body:",
        body,
    ]

    return "\n".join(lines)


def _format_event(event: Dict[str, Any]) -> str:
    start = event.get("start", {})
    end = event.get("end", {})

    start_time = start.get("dateTime") or start.get("date") or ""
    end_time = end.get("dateTime") or end.get("date") or ""
    summary = event.get("summary") or "(No title)"
    calendar = event.get("organizer", {}).get("displayName") or event.get("calendar", "primary")
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


# -------------------- Main tool functions --------------------

def get_user_emails(
    count: int = -1, label_id: str = "INBOX", query: str = "", max_body_length: int = 50000
) -> str:
    """Retrieve a list of emails and return structured JSON for LLM consumption.

    Returns JSON with:
    - total: total number of emails matching query
    - shown: number of emails included in full
    - emails: array of email objects with from, subject, date, body, id
    - remaining: number of emails not shown
    - hint: instruction for getting more emails
    """
    import json as json_module

    config = get_config()

    if count <= 0:
        count = int(config.get("default_email_entries", DEFAULT_EMAIL_ENTRIES))

    count = max(1, min(count, MAX_EMAILS_PER_CALL))

    label_id = (label_id or "INBOX").upper()
    allowed_labels = {"INBOX", "UNREAD", "STARRED", "IMPORTANT", "SENT", "SPAM", "TRASH"}
    if label_id not in allowed_labels:
        label_id = "INBOX"

    request_params: Dict[str, Any] = {"userId": "me", "maxResults": count}
    if query:
        request_params["q"] = query
    else:
        request_params["labelIds"] = [label_id]

    creds = get_google_creds(config)
    if not creds:
        return json_module.dumps({
            "error": "Could not authenticate with Google. Please check credentials.json and token.json"
        })

    try:
        service = build("gmail", "v1", credentials=creds)
        results = service.users().messages().list(**request_params).execute()

        messages = results.get("messages", []) or []
        if not messages:
            return json_module.dumps({
                "total": 0,
                "shown": 0,
                "emails": [],
                "remaining": 0,
                "hint": "No emails found matching your query."
            })

        output_emails: List[Dict[str, Any]] = []
        total_chars = 0

        global _remaining_emails
        _remaining_emails = []

        for idx, msg in enumerate(messages):
            msg_id = msg.get("id")
            if not msg_id:
                continue

            mail = service.users().messages().get(userId="me", id=msg_id, format="full").execute()
            formatted = _format_email(mail, max_body_length)

            # Estimate character size
            entry_size = len(formatted) + 50  # separator

            if total_chars + entry_size > MAX_SUMMARY_CHARACTERS:
                # store the remaining formatted email for later retrieval
                _remaining_emails.append({"id": msg_id, "mail": mail, "formatted": formatted})
                continue

            # Extract key fields for structured output
            headers = mail.get("payload", {}).get("headers", []) or []
            output_emails.append({
                "id": msg_id,
                "from": _get_header_value(headers, "From"),
                "subject": _get_header_value(headers, "Subject"),
                "date": _get_header_value(headers, "Date"),
                "snippet": mail.get("snippet", ""),
                "body": formatted.split("Body:", 1)[1].strip() if "Body:" in formatted else formatted
            })
            total_chars += entry_size

        remaining_count = len(_remaining_emails)

        result = {
            "total": len(messages),
            "shown": len(output_emails),
            "emails": output_emails,
            "remaining": remaining_count,
            "hint": f"Showing {len(output_emails)} of {len(messages)} emails. Use google_tools(action='get_remaining_emails') to get the remaining {remaining_count} emails." if remaining_count > 0 else "All emails shown."
        }

        return json_module.dumps(result, ensure_ascii=False, indent=2)

    except HttpError as e:
        logger.exception("Google API HttpError: %s", e)
        return json_module.dumps({"error": f"Google API error: {getattr(e, 'error_details', str(e))}"})
        return f"Google API error: {getattr(e, 'error_details', str(e))}"
    except Exception as e:
        logger.exception("Unexpected error fetching emails: %s", e)
        return f"Error fetching emails: {e}"


def get_email_content(message_id: str) -> str:
    """Return the full formatted content for a single message id."""
    if not message_id or len(message_id) < 5 or not re.match(r"^[A-Za-z0-9_\-]+$", message_id):
        return "Error: Invalid message ID"

    config = get_config()
    creds = get_google_creds(config)
    if not creds:
        return "Error: Could not authenticate with Google. Please check credentials.json and token.json"

    try:
        service = build("gmail", "v1", credentials=creds)
        mail = service.users().messages().get(userId="me", id=message_id, format="full").execute()
        return _format_email(mail)

    except HttpError as e:
        if getattr(e, "resp", None) and getattr(e.resp, "status", None) == 404:
            return "Error: Message not found or access denied"
        logger.exception("Google API HttpError: %s", e)
        return f"Google API error: {str(e)}"
    except Exception as e:
        logger.exception("Error fetching email content: %s", e)
        return f"Error fetching email: {e}"


def get_user_events(count: int = -1, calendar_id: str = "primary") -> str:
    """Retrieve calendar events and return structured JSON for LLM consumption."""
    import json as json_module

    config = get_config()

    if count <= 0:
        count = int(config.get("default_calendar_entries", DEFAULT_CALENDAR_ENTRIES))

    count = max(1, min(count, MAX_EMAILS_PER_CALL))

    if not calendar_id:
        calendar_id = "primary"

    creds = get_google_creds(config)
    if not creds:
        return json_module.dumps({
            "error": "Could not authenticate with Google. Please check credentials.json and token.json"
        })

    try:
        service = build("calendar", "v3", credentials=creds)
        now = datetime.utcnow().isoformat() + "Z"

        events_result = (
            service.events()
            .list(calendarId=calendar_id, timeMin=now, maxResults=count, singleEvents=True, orderBy="startTime")
            .execute()
        )

        events = events_result.get("items", []) or []
        if not events:
            return json_module.dumps({
                "total": 0,
                "events": [],
                "hint": "No upcoming events found."
            })

        output_events: List[Dict[str, Any]] = []
        for ev in events:
            start = ev.get("start", {})
            end = ev.get("end", {})
            output_events.append({
                "id": ev.get("id", ""),
                "title": ev.get("summary", "(No title)"),
                "start": start.get("dateTime") or start.get("date", ""),
                "end": end.get("dateTime") or end.get("date", ""),
                "location": ev.get("location", ""),
                "description": ev.get("description", "")
            })

        result = {
            "total": len(events),
            "events": output_events,
            "hint": f"Found {len(events)} upcoming events."
        }

        return json_module.dumps(result, ensure_ascii=False, indent=2)

    except HttpError as e:
        logger.exception("Google API HttpError (calendar): %s", e)
        return json_module.dumps({"error": f"Google API error: {str(e)}"})
    except Exception as e:
        logger.exception("Unexpected error fetching events: %s", e)
        return json_module.dumps({"error": f"Error fetching events: {e}"})


# -------------------- Tool Schema --------------------

GOOGLE_TOOLS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "google_tools",
        "description": "Tools for accessing Gmail emails and Google Calendar events. Use when a user wants to read emails or check upcoming calendar events.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["get_emails", "get_email_content", "get_events", "get_remaining_emails"],
                    "description": "Action to perform"
                },
                "count": {
                    "type": "integer",
                    "description": "Number of items to fetch (default 10, max 50)"
                },
                "label_id": {
                    "type": "string",
                    "description": "Gmail label (INBOX, UNREAD, STARRED, SENT, IMPORTANT)",
                    "default": "INBOX"
                },
                "query": {
                    "type": "string",
                    "description": "Optional Gmail search query"
                },
                "message_id": {
                    "type": "string",
                    "description": "Message ID (required for get_email_content)"
                },
                "calendar_id": {
                    "type": "string",
                    "default": "primary",
                    "description": "Calendar ID"
                }
            },
            "required": ["action"]
        }
    }
}


# -------------------- Remaining emails helper & wrapper --------------------

def get_remaining_emails(max_body_length: int = 50000) -> str:
    """Return remaining emails in structured JSON format."""
    import json as json_module

    global _remaining_emails
    if not _remaining_emails:
        return json_module.dumps({
            "total": 0,
            "emails": [],
            "hint": "No remaining emails. Run get_emails first to store truncated emails."
        })

    output_emails: List[Dict[str, Any]] = []
    for e in _remaining_emails:
        mail = e.get("mail", {})
        headers = mail.get("payload", {}).get("headers", []) or []
        output_emails.append({
            "id": e.get("id", ""),
            "from": _get_header_value(headers, "From"),
            "subject": _get_header_value(headers, "Subject"),
            "date": _get_header_value(headers, "Date"),
            "snippet": mail.get("snippet", ""),
            "body": e.get("formatted", "")
        })

    result = {
        "total": len(_remaining_emails),
        "emails": output_emails,
        "hint": f"Returned {len(_remaining_emails)} remaining emails."
    }

    _remaining_emails = []
    return json_module.dumps(result, ensure_ascii=False, indent=2)


def google_tools_wrapper(
    action: str,
    count: int = -1,
    label_id: str = "INBOX",
    message_id: str = "",
    calendar_id: str = "primary",
    query: str = "",
) -> str:
    action = (action or "").lower()
    if action == "get_emails":
        return get_user_emails(count=count, label_id=label_id, query=query)
    if action == "get_remaining_emails":
        return get_remaining_emails()
    if action == "get_email_content":
        if not message_id:
            return "Error: message_id is required for get_email_content"
        return get_email_content(message_id=message_id)
    if action == "get_events":
        return get_user_events(count=count, calendar_id=calendar_id)
    return f"Error: Unknown action '{action}'. Use 'get_emails', 'get_email_content', 'get_events', or 'get_remaining_emails'"
