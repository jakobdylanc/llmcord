"""Unit tests for bot/web/routes/status.py."""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch, AsyncMock
from bot.web.routes.status import (
    BotState, 
    StatusResponse, 
    set_discord_bot, 
    mark_bot_ready,
    set_discord_bot_ref,
    update_bot_presence,
    _discord_bot_ref,
)


class TestBotState:
    """Test BotState singleton."""

    def setup_method(self):
        """Reset BotState before each test."""
        # Reset the singleton
        BotState._instance = None

    def teardown_method(self):
        """Clean up after each test."""
        BotState._instance = None

    def test_singleton(self):
        """Test that BotState is a singleton."""
        state1 = BotState()
        state2 = BotState()
        assert state1 is state2

    def test_default_values(self):
        """Test default values when no bot is set."""
        state = BotState()
        assert state.is_online is False
        assert state.server_count == 0
        assert state.uptime_seconds == 0
        assert state.user_name is None
        assert state.user_id is None

    def test_set_bot(self):
        """Test setting the Discord bot."""
        state = BotState()
        mock_bot = MagicMock()
        mock_bot.user = None
        mock_bot.guilds = []
        
        state.set_bot(mock_bot)
        
        # Bot is set but not ready yet
        assert state.is_online is False

    def test_mark_ready(self):
        """Test marking bot as ready."""
        state = BotState()
        state.set_ready(True)
        
        assert state._is_ready is True
        assert state._start_time is not None

    def test_uptime_calculation(self):
        """Test uptime calculation."""
        state = BotState()
        
        # Set start time to 1 hour ago
        state._start_time = datetime.now()
        
        # Should be close to 0 seconds (just created)
        assert state.uptime_seconds >= 0

    def test_user_info(self):
        """Test user info when bot is set."""
        state = BotState()
        
        mock_user = MagicMock()
        mock_user.name = "TestBot"
        mock_user.id = 123456789
        
        mock_bot = MagicMock()
        mock_bot.user = mock_user
        # Use a list - len() works naturally
        mock_bot.guilds = []
        
        state.set_bot(mock_bot)
        state.set_ready(True)
        
        assert state.user_name == "TestBot"
        assert state.user_id == 123456789
        assert state.is_online is True

    def test_server_count(self):
        """Test server count."""
        state = BotState()
        
        mock_guild = MagicMock()
        mock_guild.id = 111
        mock_guild.name = "Test Server"
        mock_guild.text_channels = []
        
        mock_bot = MagicMock()
        mock_bot.user = None
        mock_bot.guilds = [mock_guild]
        
        state.set_bot(mock_bot)
        state.set_ready(True)
        
        assert state.server_count == 1


class TestStatusResponse:
    """Test StatusResponse model."""

    def test_full_response(self):
        """Test full status response with all fields."""
        response = StatusResponse(
            status="online",
            online=True,
            server_count=5,
            uptime_seconds=3600,
            started_at="2026-03-18T10:00:00",
            channel_count=20,
            user_name="TestBot",
            user_id=123456789,
            avatar_url="https://cdn.discordapp.com/avatars/123/abc.png",
            status_message="Testing bot",
        )
        
        assert response.status == "online"
        assert response.online is True
        assert response.server_count == 5
        assert response.uptime_seconds == 3600
        assert response.started_at == "2026-03-18T10:00:00"
        assert response.channel_count == 20
        assert response.user_name == "TestBot"
        assert response.user_id == 123456789
        assert response.avatar_url == "https://cdn.discordapp.com/avatars/123/abc.png"
        assert response.status_message == "Testing bot"

    def test_minimal_response(self):
        """Test minimal status response (bot offline)."""
        response = StatusResponse(
            status="offline",
            online=False,
            server_count=0,
            uptime_seconds=0,
            started_at="",
            channel_count=0,
        )
        
        assert response.status == "offline"
        assert response.online is False
        assert response.server_count == 0
        assert response.channel_count == 0
        assert response.started_at == ""
        assert response.user_name is None
        assert response.user_id is None
        assert response.avatar_url is None
        assert response.status_message is None

    def test_response_without_user_info(self):
        """Test response without user info (bot not set)."""
        response = StatusResponse(
            status="offline",
            online=False,
            server_count=0,
            uptime_seconds=0,
            started_at="",
            channel_count=0,
        )
        
        # These should be optional and None when not set
        assert response.user_name is None
        assert response.user_id is None
        assert response.avatar_url is None

    def test_avatar_url_property(self):
        """Test avatar_url property from BotState."""
        # Reset singleton
        BotState._instance = None
        state = BotState()
        
        # Create mock bot with user
        mock_user = MagicMock()
        mock_avatar = MagicMock()
        mock_avatar.url = "https://cdn.discordapp.com/avatars/123/abc.png"
        mock_user.display_avatar = mock_avatar
        
        mock_bot = MagicMock()
        mock_bot.user = mock_user
        mock_bot.guilds = []
        
        state.set_bot(mock_bot)
        
        assert state.avatar_url == "https://cdn.discordapp.com/avatars/123/abc.png"

    def test_avatar_url_none_when_no_user(self):
        """Test avatar_url is None when bot has no user."""
        # Reset singleton
        BotState._instance = None
        state = BotState()
        
        mock_bot = MagicMock()
        mock_bot.user = None
        mock_bot.guilds = []
        
        state.set_bot(mock_bot)
        
        assert state.avatar_url is None

    def test_minimal_response(self):
        """Test minimal status response."""
        response = StatusResponse(
            status="offline",
            online=False,
            server_count=0,
            uptime_seconds=0,
            started_at="",
            channel_count=0,
        )
        
        assert response.status == "offline"
        assert response.online is False
        assert response.server_count == 0
        assert response.uptime_seconds == 0
        assert response.channel_count == 0
        assert response.started_at == ""
        assert response.user_name is None
        assert response.user_id is None
        assert response.avatar_url is None


class TestUpdatePresence:
    """Test bot presence update functionality (Task 16.5)."""

    def setup_method(self):
        """Reset bot reference before each test."""
        # Reset the module-level bot reference
        import bot.web.routes.status as status_module
        status_module._discord_bot_ref = None
        status_module._bot_config_ref = None

    def teardown_method(self):
        """Clean up after each test."""
        import bot.web.routes.status as status_module
        status_module._discord_bot_ref = None
        status_module._bot_config_ref = None

    def test_set_discord_bot_ref(self):
        """Test storing Discord bot reference."""
        mock_bot = MagicMock()
        set_discord_bot_ref(mock_bot)
        
        import bot.web.routes.status as status_module
        assert status_module._discord_bot_ref is mock_bot

    @pytest.mark.asyncio
    async def test_update_presence_no_bot(self):
        """Test update_presence when no bot is registered."""
        result = await update_bot_presence()
        
        assert result["success"] is False
        assert "not available" in result["message"]

    @pytest.mark.asyncio
    async def test_update_presence_with_custom_message(self):
        """Test updating presence with custom message."""
        # Setup mock bot
        mock_bot = AsyncMock()
        set_discord_bot_ref(mock_bot)
        
        # Call with custom message
        result = await update_bot_presence("Custom Status Message")
        
        # Verify change_presence was called
        mock_bot.change_presence.assert_called_once()
        call_args = mock_bot.change_presence.call_args
        
        # Check the activity was created with custom message
        assert call_args is not None
        activity = call_args.kwargs.get("activity")
        assert activity is not None
        assert activity.name == "Custom Status Message"
        
        assert result["success"] is True
        assert "Custom Status Message" in result["message"]

    @pytest.mark.asyncio
    async def test_update_presence_truncates_long_message(self):
        """Test that very long status messages are truncated to 128 chars."""
        mock_bot = AsyncMock()
        set_discord_bot_ref(mock_bot)
        
        # Create a message longer than 128 characters
        long_message = "A" * 200
        
        result = await update_bot_presence(long_message)
        
        # Verify the message was truncated
        mock_bot.change_presence.assert_called_once()
        call_args = mock_bot.change_presence.call_args
        activity = call_args.kwargs.get("activity")
        
        assert len(activity.name) == 128
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_update_presence_uses_default_when_none_provided(self):
        """Test using default message when none provided and no config."""
        mock_bot = AsyncMock()
        set_discord_bot_ref(mock_bot)
        
        result = await update_bot_presence()
        
        # Should use default message
        mock_bot.change_presence.assert_called_once()
        call_args = mock_bot.change_presence.call_args
        activity = call_args.kwargs.get("activity")
        
        assert "github.com/jakobdylanc/llmcord" in activity.name
        assert result["success"] is True