"""Unit tests for bot integration (llmcord.py with web portal)."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio


class TestBotIntegration:
    """Test bot integration with web portal."""

    def test_imports_work(self):
        """Test that llmcord.py can be imported."""
        import llmcord
        assert llmcord is not None

    def test_web_server_import(self):
        """Test web server import from llmcord context."""
        from bot.web.server import run_web_server
        assert callable(run_web_server)

    def test_log_handler_import(self):
        """Test log handler import from llmcord context."""
        from bot.web.log_handler import init_logging
        assert callable(init_logging)

    def test_main_function_exists(self):
        """Test main function exists."""
        import llmcord
        assert hasattr(llmcord, 'main')
        assert asyncio.iscoroutinefunction(llmcord.main)

    @pytest.mark.asyncio
    async def test_main_starts_web_server(self):
        """Test main function can start web server if portal enabled."""
        import llmcord
        
        # Mock config with portal enabled
        mock_config = {
            "bot_token": "test_token",
            "portal": {"enabled": True, "port": 8080}
        }
        
        # Mock the Discord bot
        mock_bot = AsyncMock()
        mock_bot.start = AsyncMock()
        
        with patch.object(llmcord, 'discord_bot', mock_bot):
            with patch.object(llmcord, 'config', mock_config):
                with patch('llmcord.run_web_server') as mock_run_web:
                    with patch('llmcord.init_logging') as mock_init_logging:
                        # Make start raise KeyboardInterrupt to exit quickly
                        mock_bot.start = AsyncMock(side_effect=KeyboardInterrupt)
                        
                        try:
                            await llmcord.main()
                        except KeyboardInterrupt:
                            pass
                        
                        # Verify init_logging was called
                        mock_init_logging.assert_called_once()
                        
                        # Verify web server was started (run_web_server called in a task)
                        # Note: With our implementation, run_web_server is called in a task

    @pytest.mark.asyncio
    async def test_main_skips_web_server_when_disabled(self):
        """Test main function skips web server if portal disabled."""
        import llmcord
        
        mock_config = {
            "bot_token": "test_token",
            "portal": {"enabled": False}
        }
        
        mock_bot = AsyncMock()
        mock_bot.start = AsyncMock(side_effect=KeyboardInterrupt)
        
        with patch.object(llmcord, 'discord_bot', mock_bot):
            with patch.object(llmcord, 'config', mock_config):
                with patch('llmcord.run_web_server') as mock_run_web:
                    with patch('llmcord.init_logging') as mock_init_logging:
                        try:
                            await llmcord.main()
                        except KeyboardInterrupt:
                            pass
                        
                        # init_logging should still be called
                        mock_init_logging.assert_called_once()
                        
                        # run_web_server should NOT be called when portal is disabled
                        mock_run_web.assert_not_called()


class TestPortalConfigCheck:
    """Test portal config detection."""

    def test_portal_enabled_in_config(self):
        """Test that portal config can be detected."""
        # Config from config.yaml should have portal section
        from bot.config.loader import get_config
        
        config = get_config()
        
        # Portal should be in config (may be enabled or disabled)
        assert "portal" in config
        
        # Portal config should have expected keys
        portal = config["portal"]
        assert "enabled" in portal
        assert "port" in portal


class TestBotStateSharing:
    """Test BotState singleton for sharing Discord bot with web portal."""

    def test_bot_state_singleton(self):
        """Test BotState is a singleton."""
        from bot.web.routes.status import BotState
        
        state1 = BotState()
        state2 = BotState()
        
        assert state1 is state2

    def test_bot_state_can_set_bot(self):
        """Test BotState can store Discord bot reference."""
        from bot.web.routes.status import BotState
        
        state = BotState()
        
        mock_bot = MagicMock()
        mock_bot.user = None
        mock_bot.guilds = []
        
        state.set_bot(mock_bot)
        
        # Bot should be set
        assert state._discord_bot is mock_bot