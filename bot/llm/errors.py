from __future__ import annotations

from typing import Tuple


class LLMError(Exception):
    """Base error for LLM-related failures."""


class LLMRateLimitError(LLMError):
    pass


class LLMAuthError(LLMError):
    pass


class LLMNotFoundError(LLMError):
    pass


class LLMForbiddenError(LLMError):
    pass


class LLMConnectionError(LLMError):
    pass


class LLMToolError(LLMError):
    pass


def parse_error_message(error: Exception) -> str:
    """
    Map raw exceptions into short, human-readable messages.
    Used for admin notifications and logs.
    """
    s, t = str(error), type(error).__name__
    if "429" in s or t == "RateLimitError":
        return "⚠️ Rate Limited: API provider is temporarily rate-limited. Please retry shortly."
    if "401" in s or "Unauthorized" in s:
        return "❌ Authentication Error: Invalid API key or credentials."
    if "404" in s or t == "NotFound":
        return "❌ Not Found: The requested resource was not found."
    if "403" in s or t == "Forbidden":
        return "❌ Forbidden: You don't have permission to access this resource."
    if "Connection" in t or "ECONNREFUSED" in s or "ETIMEDOUT" in s:
        return "❌ Connection Error: Unable to connect to the API provider."
    return f"❌ {t}: {s.split(chr(10))[0][:100]}"


def format_user_friendly_error(error: Exception) -> str:
    """
    Short, safe error message suitable for end users.
    """
    s, t = str(error), type(error).__name__
    if "429" in s or t == "RateLimitError":
        return "目前模型流量較高，請稍後再試。"
    if "401" in s or "Unauthorized" in s:
        return "目前無法連線到模型服務，請聯絡管理員檢查金鑰設定。"
    if "404" in s or t == "NotFound":
        return "找不到指定的模型或資源。"
    if "403" in s or t == "Forbidden":
        return "目前沒有權限使用這個模型或資源。"
    if "Connection" in t or "ECONNREFUSED" in s or "ETIMEDOUT" in s:
        return "連線模型服務失敗，可能是網路或伺服器問題。"
    return "呼叫模型時發生未預期錯誤，已通知管理員。"


def error_messages(error: Exception) -> Tuple[str, str]:
    """
    Convenience helper returning (admin_message, user_message).
    """
    return parse_error_message(error), format_user_friendly_error(error)

