import sys
sys.path.append('.')

import base64
import json
from bot.llm.tools.registry import build_tool_registry
from bot.llm.tools.google_tools import get_google_creds, get_config
from googleapiclient.discovery import build

def test_raw_email():
    """Debug: Check raw email data structure"""
    print("\n=== Debug: Raw Email Structure ===")
    config = get_config()
    creds = get_google_creds(config)
    if not creds:
        print("No credentials")
        return
    
    service = build("gmail", "v1", credentials=creds)
    
    # Get first message
    results = service.users().messages().list(
        userId="me",
        q="in:inbox -category:promotions -in:spam newer_than:1d",
        maxResults=1
    ).execute()
    
    messages = results.get("messages", [])
    if not messages:
        print("No messages found")
        return
    
    msg_id = messages[0]["id"]
    print(f"Message ID: {msg_id}")
    
    # Get full message
    mail = service.users().messages().get(userId="me", id=msg_id, format="full").execute()
    
    # Check payload structure
    payload = mail.get("payload", {})
    print(f"\nMimeType: {payload.get('mimeType')}")
    
    # Check headers
    headers = payload.get("headers", [])
    for h in headers:
        name = h.get("name", "").lower()
        if name in ["content-type", "content-transfer-encoding"]:
            print(f"{h.get('name')}: {h.get('value')}")
    
    # Check body
    body = payload.get("body", {})
    print(f"\nBody keys: {body.keys()}")
    if body.get("data"):
        data = body["data"]
        print(f"Body data (first 100 chars): {data[:100]}")
        # Try decoding
        try:
            decoded = base64.urlsafe_b64decode(data + "==")
            print(f"Decoded (first 100): {decoded[:100]}")
        except Exception as e:
            print(f"Decode error: {e}")
    
    # Check parts
    parts = payload.get("parts", [])
    print(f"\nNumber of parts: {len(parts)}")
    for i, part in enumerate(parts[:3]):
        print(f"Part {i}: mimeType={part.get('mimeType')}, partId={part.get('partId')}")
        if part.get("body", {}).get("data"):
            print(f"  Has body data: {len(part['body']['data'])} chars")
        # Check headers in part
        part_headers = part.get("headers", [])
        for h in part_headers:
            name = h.get("name", "").lower()
            if name in ["content-type", "content-transfer-encoding"]:
                print(f"  {h.get('name')}: {h.get('value')}")

def test_get_emails():
    """Test Gmail API integration for email retrieval"""
    print("\n=== Testing Gmail Email Retrieval ===")
    print("Query: in:inbox -category:promotions -in:spam newer_than:1d")
    
    try:
        config = get_config()
        creds = get_google_creds(config)
        if not creds:
            print("\n⚠️ No credentials")
            return
        
        service = build("gmail", "v1", credentials=creds)
        
        # Get messages with query
        results = service.users().messages().list(
            userId="me",
            q="in:inbox -category:promotions -in:spam newer_than:1d",
            maxResults=50
        ).execute()
        
        messages = results.get("messages", [])
        if not messages:
            print("\n⚠️ No emails found")
            return
        
        print(f"\n✅ Found {len(messages)} emails:\n")
        
        for i, msg in enumerate(messages, 1):
            # Get full message for headers
            mail = service.users().messages().get(userId="me", id=msg["id"], format="full").execute()
            payload = mail.get("payload", {})
            headers = payload.get("headers", [])
            
            # Extract headers
            subject = ""
            sender = ""
            date = ""
            for h in headers:
                name = h.get("name", "").lower()
                if name == "subject":
                    subject = h.get("value", "")
                elif name == "from":
                    sender = h.get("value", "")
                elif name == "date":
                    date = h.get("value", "")
            
            # Get snippet (short preview)
            snippet = mail.get("snippet", "")
            
            # Try to get full body using google_tools decode
            from bot.llm.tools.google_tools import _decode_email_body
            import re
            import html
            payload = mail.get("payload", {})
            body = _decode_email_body(payload)
            
            # Strip HTML tags and decode HTML entities to shorten output
            if body:
                # Decode HTML entities
                body = html.unescape(body)
                # Remove HTML tags
                body = re.sub(r'<[^>]+>', '', body)
                # Clean up whitespace
                body = re.sub(r'\s+', ' ', body).strip()
            
            print(f"{i}. {subject}")
            print(f"   From: {sender}")
            print(f"   Date: {date}")
            print(f"   Preview: {snippet}")
            print(f"   Body:")
            print("   " + "-"*40)
            print("   " + (body if body else 'N/A'))
            print("   " + "-"*40)
            print()

    except Exception as e:
        print(f"\n❌ Exception occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_get_emails()