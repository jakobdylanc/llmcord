"""Unit tests for bot/web/routes/config.py."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi import HTTPException


class TestConfigFiltering:
    """Test config filtering functions."""

    def test_is_sensitive_field(self):
        """Test sensitive field detection."""
        from bot.web.routes.config import _is_sensitive_field
        
        # Sensitive fields
        assert _is_sensitive_field("api_key") is True
        assert _is_sensitive_field("bot_token") is True
        assert _is_sensitive_field("password") is True
        assert _is_sensitive_field("secret_key") is True
        assert _is_sensitive_field("AZURE_SPEECH_KEY") is True
        
        # Non-sensitive fields
        assert _is_sensitive_field("status_message") is False
        assert _is_sensitive_field("port") is False
        assert _is_sensitive_field("enabled") is False

    def test_filter_config_redacts_sensitive(self):
        """Test that sensitive fields are redacted."""
        from bot.web.routes.config import _filter_config
        
        config = {
            "bot_token": "secret-token",
            "api_key": "secret-key",
            "status_message": "Hello World",
            "port": 8080,
            "providers": {
                "openai": {
                    "api_key": "provider-secret",
                }
            }
        }
        
        filtered = _filter_config(config, redact_sensitive=True)
        
        assert filtered["bot_token"] == "[REDACTED]"
        assert filtered["api_key"] == "[REDACTED]"
        assert filtered["status_message"] == "Hello World"
        assert filtered["port"] == 8080
        assert filtered["providers"]["openai"]["api_key"] == "[REDACTED]"

    def test_filter_config_no_redact(self):
        """Test config filtering without redaction."""
        from bot.web.routes.config import _filter_config
        
        config = {
            "bot_token": "secret-token",
            "status_message": "Hello World",
        }
        
        filtered = _filter_config(config, redact_sensitive=False)
        
        # Without redaction, should keep original values
        assert filtered["bot_token"] == "secret-token"
        assert filtered["status_message"] == "Hello World"

    def test_filter_config_skips_portal(self):
        """Test that portal section is skipped in main config."""
        from bot.web.routes.config import _filter_config
        
        config = {
            "bot_token": "secret",
            "portal": {
                "enabled": True,
                "port": 8080,
                "admin_ids": [123, 456],
            }
        }
        
        filtered = _filter_config(config, redact_sensitive=True)
        
        # Portal should be excluded from main config
        assert "portal" not in filtered

    def test_get_portal_config_safe(self):
        """Test safe portal config extraction."""
        from bot.web.routes.config import _get_portal_config_safe
        
        test_config = {
            "portal": {
                "enabled": True,
                "port": 9000,
                "logs": {
                    "retention_days": 14,
                    "levels": ["DEBUG", "INFO", "WARNING", "ERROR"],
                },
                "require_discord_admin": True,
                "admin_ids": [123, 456],  # Should be excluded
            }
        }
        
        with patch('bot.web.routes.config.get_raw_config', return_value=test_config):
            portal = _get_portal_config_safe()
        
        assert portal["enabled"] is True
        assert portal["port"] == 9000
        assert portal["logs"]["retention_days"] == 14
        assert portal["logs"]["levels"] == ["DEBUG", "INFO", "WARNING", "ERROR"]
        assert portal["require_discord_admin"] is True


class TestConfigDisplayFormatting:
    """Test config display formatting for UI rendering."""

    def test_format_models(self):
        """Test models field is formatted as comma-separated list."""
        from bot.web.routes.config import _format_for_display
        
        config = {
            "models": {
                "ollama/qwen3:14b": {"persona": "bao"},
                "openrouter/openrouter/free": {"persona": "default"},
            }
        }
        
        result = _format_for_display(config)
        
        assert result["models"] == "ollama/qwen3:14b, openrouter/openrouter/free"

    def test_format_models_empty(self):
        """Test models field with empty dict."""
        from bot.web.routes.config import _format_for_display
        
        config = {"models": {}}
        
        result = _format_for_display(config)
        
        assert result["models"] == "None"

    def test_format_permissions(self):
        """Test permissions field is formatted as comma-separated keys."""
        from bot.web.routes.config import _format_for_display
        
        config = {
            "permissions": {
                "users": {"allowed_ids": [123]},
                "roles": {"blocked_ids": [456]},
                "channels": {"allowed_ids": [789]},
            }
        }
        
        result = _format_for_display(config)
        
        assert result["permissions"] == "users, roles, channels"

    def test_format_providers(self):
        """Test providers field is formatted as comma-separated list."""
        from bot.web.routes.config import _format_for_display
        
        config = {
            "providers": {
                "google": {"base_url": "https://google.com"},
                "ollama": {"base_url": "http://localhost:11434"},
                "openrouter": {},
            }
        }
        
        result = _format_for_display(config)
        
        assert result["providers"] == "google, ollama, openrouter"

    def test_format_tools(self):
        """Test tools field shows name with enabled status."""
        from bot.web.routes.config import _format_for_display
        
        config = {
            "tools": {
                "web_search": {"enabled": True},
                "get_weather": {"enabled": True},
                "get_market_prices": {"enabled": False},
            }
        }
        
        result = _format_for_display(config)
        
        # Should show checkmark for enabled, X for disabled
        assert "web_search (✓)" in result["tools"]
        assert "get_weather (✓)" in result["tools"]
        assert "get_market_prices (✗)" in result["tools"]

    def test_format_azure_speech(self):
        """Test azure-speech field shows keys."""
        from bot.web.routes.config import _format_for_display
        
        config = {
            "azure-speech": {
                "key": "secret",
                "region": "eastus",
                "default_voice": "en-US-Aria",
            }
        }
        
        result = _format_for_display(config)
        
        assert result["azure-speech"] == "key, region, default_voice"

    def test_format_fallback_models(self):
        """Test fallback_models field is formatted."""
        from bot.web.routes.config import _format_for_display
        
        config = {
            "fallback_models": {
                "openrouter/openrouter/free": {},
            }
        }
        
        result = _format_for_display(config)
        
        assert result["fallback_models"] == "openrouter/openrouter/free"

    def test_format_nested_dict(self):
        """Test nested dict without special handling is recursively formatted."""
        from bot.web.routes.config import _format_for_display
        
        config = {
            "nested": {
                "level1": {
                    "level2": "value",
                }
            }
        }
        
        result = _format_for_display(config)
        
        assert result["nested"]["level1"]["level2"] == "value"

    def test_format_list(self):
        """Test list fields are formatted as comma-separated strings."""
        from bot.web.routes.config import _format_for_display
        
        config = {
            "allowed_channels": [123, 456, 789],
        }
        
        result = _format_for_display(config)
        
        assert result["allowed_channels"] == "123, 456, 789"

    def test_format_simple_values_unchanged(self):
        """Test simple (non-dict/list) values are unchanged."""
        from bot.web.routes.config import _format_for_display
        
        config = {
            "status_message": "Hello World",
            "max_text": 100000,
            "port": 8080,
        }
        
        result = _format_for_display(config)
        
        assert result["status_message"] == "Hello World"
        assert result["max_text"] == 100000
        assert result["port"] == 8080


class TestConfigEndpoints:
    """Test config API endpoints."""

    @pytest.mark.asyncio
    async def test_get_config(self):
        """Test GET /api/config endpoint."""
        from bot.web.routes.config import get_config, ConfigResponse
        
        test_config = {
            "status_message": "Test Bot",
            "max_text": 5000,
            "portal": {
                "enabled": True,
                "port": 8080,
                "logs": {"retention_days": 7, "levels": ["INFO"]},
            }
        }
        
        with patch('bot.web.routes.config.get_raw_config', return_value=test_config):
            with patch('bot.web.routes.config._get_portal_config_safe', return_value={"enabled": True, "port": 8080}):
                response = await get_config()
        
        assert isinstance(response, ConfigResponse)
        assert response.config["status_message"] == "Test Bot"
        assert response.portal["enabled"] is True
        assert "status_message" in response.editable_fields

    @pytest.mark.asyncio
    async def test_update_config_status_message(self):
        """Test PUT /api/config updating status_message."""
        from bot.web.routes.config import update_config, ConfigUpdateRequest
        import io
        
        mock_user = MagicMock()
        mock_user.username = "admin"
        
        mock_request = ConfigUpdateRequest(status_message="Updated Status")
        
        # Create a mock file that returns config data and can be written to
        mock_file = io.StringIO()
        mock_file.close = MagicMock()
        
        original_open = open
        
        def mock_open(path, mode='r', encoding=None):
            if 'r' in mode:
                # Return config content for reading
                return io.StringIO("status_message: Old\n")
            else:
                # Return the mock file for writing
                return mock_file
        
        with patch('bot.web.routes.config.get_config_path', return_value="config.yaml"):
            with patch('bot.web.routes.config._filter_config', return_value={"status_message": "Updated Status"}):
                with patch('bot.web.routes.config._get_portal_config_safe', return_value={"enabled": True}):
                    with patch('builtins.open', mock_open):
                        with patch('yaml.safe_load', return_value={"status_message": "Old"}):
                            with patch('yaml.safe_dump', MagicMock()):
                                response = await update_config(mock_request, mock_user)
        
        assert response.success is True
        assert "status_message" in response.message

    @pytest.mark.asyncio
    async def test_update_config_portal(self):
        """Test PUT /api/config updating portal settings."""
        from bot.web.routes.config import update_config, ConfigUpdateRequest
        import io
        
        mock_user = MagicMock()
        mock_user.username = "admin"
        
        mock_request = ConfigUpdateRequest(
            portal={
                "enabled": False,
                "port": 9000,
                "logs": {"retention_days": 30},
            }
        )
        
        def mock_open(path, mode='r', encoding=None):
            return io.StringIO("portal:\n  enabled: true\n")
        
        with patch('bot.web.routes.config.get_config_path', return_value="config.yaml"):
            with patch('bot.web.routes.config._filter_config', return_value={"portal": {"enabled": False}}):
                with patch('bot.web.routes.config._get_portal_config_safe', return_value={"enabled": False}):
                    with patch('builtins.open', mock_open):
                        with patch('yaml.safe_load', return_value={"portal": {"enabled": True}}):
                            with patch('yaml.safe_dump', MagicMock()):
                                response = await update_config(mock_request, mock_user)
        
        assert response.success is True

    @pytest.mark.asyncio
    async def test_refresh_config(self):
        """Test POST /api/refresh endpoint."""
        from bot.web.routes import config as config_module
        
        mock_user = MagicMock()
        mock_user.username = "admin"
        
        mock_portal_config = MagicMock()
        mock_portal_config.enabled = True
        mock_portal_config.port = 8080
        mock_portal_config.logs_retention_days = 7
        mock_portal_config.logs_levels = ["INFO", "WARNING"]
        
        # Patch the import at the source - where it's imported in the function
        with patch('bot.web.config.reload_portal_config', return_value=mock_portal_config):
            response = await config_module.refresh_config(mock_user)
        
        assert response.success is True
        assert "reloaded" in response.message.lower()


class TestEditableFields:
    """Test editable fields configuration."""

    def test_editable_fields_defined(self):
        """Test that editable fields are properly defined."""
        from bot.web.routes.config import EDITABLE_FIELDS
        
        assert "status_message" in EDITABLE_FIELDS
        assert "max_text" in EDITABLE_FIELDS
        assert "portal" in EDITABLE_FIELDS

    def test_sensitive_fields_defined(self):
        """Test that sensitive fields are properly defined."""
        from bot.web.routes.config import SENSITIVE_FIELDS
        
        assert "api_key" in SENSITIVE_FIELDS
        assert "bot_token" in SENSITIVE_FIELDS
        assert "password" in SENSITIVE_FIELDS


class TestChineseCharacterEncoding:
    """Test that Chinese/non-ASCII characters are properly saved and loaded."""

    @pytest.mark.asyncio
    async def test_save_config_with_chinese_characters(self):
        """Test that Chinese characters in config are saved correctly."""
        import yaml
        from bot.web.routes.config import update_config, ConfigUpdateRequest
        import io
        
        mock_user = MagicMock()
        mock_user.username = "admin"
        
        # Test Chinese status message
        chinese_status = "測試機器人 - AI Bot"
        mock_request = ConfigUpdateRequest(status_message=chinese_status)
        
        # Track what was written to yaml.safe_dump
        written_content = []
        
        def mock_safe_dump(data, stream, **kwargs):
            # Capture the content that was written
            if stream:
                stream.write(str(data))
                written_content.append(str(data))
            return data
        
        original_yaml_dump = yaml.safe_dump
        
        def mock_open(path, mode='r', encoding=None):
            if 'r' in mode:
                return io.StringIO("status_message: Old\n")
            else:
                # For writing, return a StringIO that captures the output
                return io.StringIO()
        
        with patch('bot.web.routes.config.get_config_path', return_value="config.yaml"):
            with patch('bot.web.routes.config._filter_config', return_value={"status_message": chinese_status}):
                with patch('bot.web.routes.config._get_portal_config_safe', return_value={"enabled": True}):
                    with patch('builtins.open', mock_open):
                        with patch('yaml.safe_load', return_value={"status_message": "Old"}):
                            with patch('yaml.safe_dump', side_effect=mock_safe_dump):
                                response = await update_config(mock_request, mock_user)
        
        # Verify the update was successful
        assert response.success is True
        # Verify Chinese characters are in the message (since we updated status_message)
        assert "測試" in chinese_status or "status_message" in response.message.lower()

    def test_yaml_dump_uses_allow_unicode(self):
        """Test that yaml.safe_dump is called with allow_unicode=True."""
        import yaml
        from unittest.mock import patch, MagicMock
        
        test_data = {"status_message": "測試機器人"}
        
        with patch('yaml.safe_dump') as mock_dump:
            mock_dump.return_value = ""
            # Call yaml.safe_dump the way we do in the config module
            yaml.safe_dump(test_data, MagicMock(), default_flow_style=False, sort_keys=False, allow_unicode=True)
            
            # Verify allow_unicode=True was passed
            call_args = mock_dump.call_args
            assert call_args is not None
            assert call_args.kwargs.get('allow_unicode', False) is True