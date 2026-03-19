"""Status API endpoints."""

import logging
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from bot.web.auth import CurrentUser, get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["status"])


# Bot reference for updating presence
_discord_bot_ref = None


def set_discord_bot_ref(bot):
    """Store reference to Discord bot for presence updates."""
    global _discord_bot_ref
    _discord_bot_ref = bot


# Bot state singleton (updated by main bot)
class BotState:
    """Singleton to hold bot state accessible to web portal."""
    
    _instance: Optional["BotState"] = None
    _discord_bot = None
    _start_time: Optional[datetime] = None
    _is_ready: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def set_bot(self, bot):
        """Set the Discord bot reference."""
        self._discord_bot = bot
    
    def set_ready(self, is_ready: bool):
        """Mark bot as ready."""
        self._is_ready = is_ready
        if is_ready and self._start_time is None:
            self._start_time = datetime.now()
    
    @property
    def is_online(self) -> bool:
        """Check if bot is online."""
        return self._discord_bot is not None and self._is_ready
    
    @property
    def server_count(self) -> int:
        """Get number of connected servers."""
        if self._discord_bot and hasattr(self._discord_bot, 'guilds'):
            return len(self._discord_bot.guilds)
        return 0
    
    @property
    def uptime_seconds(self) -> int:
        """Get uptime in seconds."""
        if self._start_time:
            return int((datetime.now() - self._start_time).total_seconds())
        return 0
    
    @property
    def user_name(self) -> Optional[str]:
        """Get bot username."""
        if self._discord_bot and self._discord_bot.user:
            return self._discord_bot.user.name
        return None
    
    @property
    def user_id(self) -> Optional[int]:
        """Get bot user ID."""
        if self._discord_bot and self._discord_bot.user:
            return self._discord_bot.user.id
        return None
    
    @property
    def avatar_url(self) -> Optional[str]:
        """Get bot avatar URL."""
        try:
            if self._discord_bot and self._discord_bot.user:
                user = self._discord_bot.user
                # Try display_avatar first (discord.py 2.0+)
                if hasattr(user, 'display_avatar') and user.display_avatar:
                    return str(user.display_avatar.url)
                # Fallback to avatar (discord.py 1.x)
                if hasattr(user, 'avatar') and user.avatar:
                    return str(user.avatar.url)
                # Last resort: use default avatar based on user ID
                # Discord default avatars are numbered 0-4 based on (user_id >> 22) % 5
                if user.id:
                    avatar_index = (user.id >> 22) % 6
                    return f"https://cdn.discordapp.com/embed/avatars/{avatar_index}.png"
            return None
        except Exception:
            return None

    @property
    def status_message(self) -> Optional[str]:
        """Get bot status message from config."""
        try:
            if _bot_config_ref is not None:
                return _bot_config_ref.get("status_message")
            return None
        except Exception:
            return None


# Global bot state
bot_state = BotState()


# Response models
class StatusResponse(BaseModel):
    """Bot status response."""
    status: str
    online: bool
    server_count: int
    uptime_seconds: int
    started_at: str
    channel_count: int
    user_name: Optional[str] = None
    user_id: Optional[int] = None
    avatar_url: Optional[str] = None
    status_message: Optional[str] = None


# API endpoints
@router.get("/status", response_model=StatusResponse)
async def get_status(current_user: CurrentUser = Depends(get_current_user)) -> StatusResponse:
    """Get current bot status."""
    # Get additional stats from Discord bot
    channel_count = 0
    
    if bot_state._discord_bot and hasattr(bot_state._discord_bot, 'guilds'):
        for guild in bot_state._discord_bot.guilds:
            channel_count += len(guild.channels) if hasattr(guild, 'channels') else 0
    
    started_at = bot_state._start_time.isoformat() if bot_state._start_time else ""
    
    return StatusResponse(
        status="online" if bot_state.is_online else "offline",
        online=bot_state.is_online,
        server_count=bot_state.server_count,
        uptime_seconds=bot_state.uptime_seconds,
        started_at=started_at,
        channel_count=channel_count,
        user_name=bot_state.user_name,
        user_id=bot_state.user_id,
        avatar_url=bot_state.avatar_url,
        status_message=bot_state.status_message,
    )


def set_discord_bot(bot):
    """Called by main bot to register itself."""
    bot_state.set_bot(bot)


def mark_bot_ready(is_ready: bool = True):
    """Called by main bot when ready."""
    bot_state.set_ready(is_ready)


# Store references to bot's config for external refresh
_bot_config_ref = None
_bot_curr_model_ref = None
_bot_curr_persona_ref = None


def register_config_refs(config, curr_model, curr_persona):
    """Register references to bot's global config for external refresh."""
    global _bot_config_ref, _bot_curr_model_ref, _bot_curr_persona_ref
    _bot_config_ref = config
    _bot_curr_model_ref = curr_model
    _bot_curr_persona_ref = curr_persona


async def reload_bot_config():
    """
    Reload the bot's config from disk.
    Returns True if successful, False otherwise.
    """
    global _bot_config_ref
    
    if _bot_config_ref is None:
        logger.warning("No config reference registered")
        return False
    
    try:
        # Import here to avoid circular imports
        from bot.config.loader import get_config
        
        # Reload config
        new_config = get_config()
        
        # Update the global config reference
        _bot_config_ref.clear()
        _bot_config_ref.update(new_config)
        
        logger.info("Bot config reloaded via web portal")
        return True
    except Exception as e:
        logger.error(f"Failed to reload bot config: {e}")
        return False


async def update_bot_presence(status_message: Optional[str] = None) -> dict[str, Any]:
    """
    Update the Discord bot's presence/activity.
    
    Args:
        status_message: The new status message. If None, uses config or default.
    
    Returns:
        dict with success status and message
    """
    global _discord_bot_ref
    
    if _discord_bot_ref is None:
        logger.warning("No Discord bot reference registered")
        return {"success": False, "message": "Discord bot not available"}
    
    try:
        import discord
        
        # Get status message from config if not provided
        if status_message is None:
            if _bot_config_ref is not None:
                status_message = _bot_config_ref.get("status_message")
            if not status_message:
                status_message = "github.com/jakobdylanc/llmcord"
        
        # Create the new activity
        activity = discord.CustomActivity(name=status_message[:128])
        
        # Update the bot's presence
        await _discord_bot_ref.change_presence(activity=activity)
        
        logger.info(f"Bot presence updated: {status_message[:50]}...")
        return {"success": True, "message": f"Presence updated to: {status_message[:50]}..."}
    except Exception as e:
        logger.error(f"Failed to update bot presence: {e}")
        return {"success": False, "message": str(e)}


class UpdatePresenceResponse(BaseModel):
    """Response for presence update."""
    success: bool
    message: str


@router.post("/bot/update-presence", response_model=UpdatePresenceResponse)
async def update_presence(
    current_user: CurrentUser = Depends(get_current_user),
    status_message: Optional[str] = None,
) -> UpdatePresenceResponse:
    """
    Update the Discord bot's presence/activity.
    
    Optionally provide a status_message in the request body.
    If not provided, uses the current config value.
    """
    result = await update_bot_presence(status_message)
    return UpdatePresenceResponse(**result)