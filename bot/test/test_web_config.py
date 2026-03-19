"""Unit tests for bot/web/config.py."""

import pytest
from unittest.mock import patch, MagicMock


class TestPortalConfig:
    """Test PortalConfig class."""

    def setup_method(self):
        """Reset global config before each test."""
        import bot.web.config
        bot.web.config._config = None

    def teardown_method(self):
        """Reset global config after each test."""
        import bot.web.config
        bot.web.config._config = None

    def test_enabled_default(self):
        """Test default enabled is False."""
        from bot.web.config import PortalConfig

        config = PortalConfig({})
        assert config.enabled is False

    def test_enabled_true(self):
        """Test enabled when set to True."""
        from bot.web.config import PortalConfig

        config = PortalConfig({"enabled": True})
        assert config.enabled is True

    def test_port_default(self):
        """Test default port is 8080."""
        from bot.web.config import PortalConfig

        config = PortalConfig({})
        assert config.port == 8080

    def test_port_custom(self):
        """Test custom port."""
        from bot.web.config import PortalConfig

        config = PortalConfig({"port": 9000})
        assert config.port == 9000

    def test_port_from_string(self):
        """Test port from string value."""
        from bot.web.config import PortalConfig

        config = PortalConfig({"port": "9000"})
        assert config.port == 9000

    def test_require_discord_admin_default(self):
        """Test default require_discord_admin is False."""
        from bot.web.config import PortalConfig

        config = PortalConfig({})
        assert config.require_discord_admin is False

    def test_require_discord_admin_true(self):
        """Test require_discord_admin when set."""
        from bot.web.config import PortalConfig

        config = PortalConfig({"require_discord_admin": True})
        assert config.require_discord_admin is True

    def test_logs_retention_days_default(self):
        """Test default log retention is 7 days."""
        from bot.web.config import PortalConfig

        config = PortalConfig({})
        assert config.logs_retention_days == 7

    def test_logs_retention_days_custom(self):
        """Test custom log retention."""
        from bot.web.config import PortalConfig

        config = PortalConfig({"logs": {"retention_days": 30}})
        assert config.logs_retention_days == 30

    def test_logs_levels_default(self):
        """Test default log levels."""
        from bot.web.config import PortalConfig

        config = PortalConfig({})
        assert config.logs_levels == ["INFO", "WARNING", "ERROR"]

    def test_logs_levels_custom(self):
        """Test custom log levels."""
        from bot.web.config import PortalConfig

        config = PortalConfig({"logs": {"levels": ["DEBUG", "INFO"]}})
        assert config.logs_levels == ["DEBUG", "INFO"]

    def test_admin_ids_default(self):
        """Test default admin_ids is empty list."""
        from bot.web.config import PortalConfig

        config = PortalConfig({})
        assert config.admin_ids == []

    def test_admin_ids_from_config(self):
        """Test admin_ids from config."""
        from bot.web.config import PortalConfig

        config = PortalConfig({"admin_ids": [123, 456]})
        assert config.admin_ids == [123, 456]

    def test_cors_origins_default(self):
        """Test default cors_origins is empty list (same origin only - secure)."""
        from bot.web.config import PortalConfig

        config = PortalConfig({})
        assert config.cors_origins == []

    def test_cors_origins_from_config(self):
        """Test cors_origins from config."""
        from bot.web.config import PortalConfig

        config = PortalConfig({"cors_origins": ["http://localhost:3000"]})
        assert config.cors_origins == ["http://localhost:3000"]

    def test_cors_origins_multiple(self):
        """Test cors_origins with multiple origins."""
        from bot.web.config import PortalConfig

        config = PortalConfig({"cors_origins": ["http://localhost:3000", "https://example.com"]})
        assert config.cors_origins == ["http://localhost:3000", "https://example.com"]


class TestGetPortalConfig:
    """Test get_portal_config function."""

    def setup_method(self):
        """Reset global config before each test."""
        import bot.web.config
        bot.web.config._config = None

    def teardown_method(self):
        """Reset global config after each test."""
        import bot.web.config
        bot.web.config._config = None

    def test_singleton(self):
        """Test that get_portal_config returns singleton."""
        from bot.web.config import get_portal_config, PortalConfig

        with patch("bot.web.config.get_config") as mock_get_config:
            mock_get_config.return_value = {"enabled": True}

            config1 = get_portal_config()
            config2 = get_portal_config()

            assert config1 is config2
            assert isinstance(config1, PortalConfig)

    def test_reload_portal_config(self):
        """Test reload_portal_config creates new instance."""
        from bot.web.config import reload_portal_config

        with patch("bot.web.config.get_config") as mock_get_config:
            mock_get_config.return_value = {"enabled": True}

            config1 = reload_portal_config()
            mock_get_config.return_value = {"enabled": False}
            config2 = reload_portal_config()

            assert config1.enabled is True
            assert config2.enabled is False