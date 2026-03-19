"""Web routes module."""

# Import only existing routes
from bot.web.routes import status, servers, logs, config, personas, tasks, skills

__all__ = ["status", "servers", "logs", "config", "personas", "tasks", "skills"]