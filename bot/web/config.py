"""Portal configuration loading for web server."""

from typing import Any

from bot.config.loader import get_config


class PortalConfig:
    """Portal configuration loaded from config.yaml."""

    def __init__(self, config: dict[str, Any]):
        # Handle both full config dict and portal-only config
        # Full config has {"portal": {...}}, portal-only has keys directly
        if "portal" in config:
            self._config = config.get("portal", {})
        else:
            # Already the portal section
            self._config = config

    @property
    def enabled(self) -> bool:
        """Whether the portal is enabled."""
        return self._config.get("enabled", False)

    @property
    def port(self) -> int:
        """Port to run the web server on."""
        return int(self._config.get("port", 8080))

    @property
    def require_discord_admin(self) -> bool:
        """Whether to require Discord admin_ids check in addition to password."""
        return self._config.get("require_discord_admin", False)

    @property
    def logs_retention_days(self) -> int:
        """Number of days to retain logs."""
        return int(self._config.get("logs", {}).get("retention_days", 7))

    @property
    def logs_levels(self) -> list[str]:
        """Log levels to capture."""
        return self._config.get("logs", {}).get("levels", ["INFO", "WARNING", "ERROR"])

    @property
    def admin_ids(self) -> list[int]:
        """Discord admin user IDs for additional auth check."""
        return self._config.get("admin_ids", [])

    @property
    def cors_origins(self) -> list[str]:
        """Allowed CORS origins. Empty = same origin only (secure)."""
        return self._config.get("cors_origins", [])


# Global config instance
_config: PortalConfig | None = None


def get_portal_config() -> PortalConfig:
    """Get the portal configuration (singleton)."""
    global _config
    if _config is None:
        config = get_config()
        _config = PortalConfig(config)
    return _config


def reload_portal_config() -> PortalConfig:
    """Reload the portal configuration."""
    global _config
    config = get_config()
    _config = PortalConfig(config)
    return _config