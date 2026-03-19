"""Log handler for dual logging (console + database)."""

import logging
import asyncio
from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from bot.db.connection import get_db
from bot.db.models import EventLog
from bot.web.config import get_portal_config

logger = logging.getLogger(__name__)

# Global list to track connected WebSocket clients for real-time logs
_log_clients: list = []


class DatabaseLogHandler(logging.Handler):
    """
    Custom logging handler that writes logs to both console and database.
    
    This handler:
    1. Emits logs to the standard console handler (preserving normal logging)
    2. Writes logs to the database for the web portal's log viewer
    3. Broadcasts logs to connected WebSocket clients
    """
    
    def __init__(self):
        super().__init__()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._pending_logs: list = []
        self._batch_size = 10
        self._batch_timeout = 2.0  # seconds
    
    def set_loop(self, loop: asyncio.AbstractEventLoop):
        """Set the asyncio event loop for async operations."""
        self._loop = loop
    
    def emit(self, record: logging.LogRecord):
        """Emit a log record to console and schedule database write."""
        try:
            # Get log level from record
            level = record.levelname
            
            # Check if we should log this level (based on config)
            try:
                config = get_portal_config()
                allowed_levels = config.logs_levels
            except Exception:
                # If config fails, allow all
                allowed_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            
            if level not in allowed_levels:
                return
            
            # Create log entry data
            log_data = {
                "timestamp": datetime.fromtimestamp(record.created),
                "level": level,
                "event_type": record.name or "unknown",
                "message": record.getMessage(),
                "extra_data": None,
            }
            
            # Add source info if available
            if record.filename:
                log_data["extra_data"] = {
                    "filename": record.filename,
                    "line": record.lineno,
                    "function": record.funcName,
                }
            
            # Schedule async write
            if self._loop and self._loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self._write_log_async(log_data), 
                    self._loop
                )
            else:
                # Store for later if loop not available
                self._pending_logs.append(log_data)
                
        except Exception as e:
            # Don't let logging errors crash the app
            print(f"Error in log handler: {e}")
    
    async def _write_log_async(self, log_data: dict):
        """Write log to database asynchronously."""
        try:
            async with get_db() as session:
                event_log = EventLog(
                    timestamp=log_data["timestamp"],
                    level=log_data["level"],
                    event_type=log_data["event_type"],
                    message=log_data["message"],
                    extra_data=log_data.get("extra_data"),
                )
                session.add(event_log)
                await session.commit()
                
            # Broadcast to WebSocket clients
            await self._broadcast_log(log_data)
            
        except Exception as e:
            logger.error(f"Failed to write log to database: {e}")
    
    async def _broadcast_log(self, log_data: dict):
        """Broadcast log to all connected WebSocket clients."""
        if not _log_clients:
            return
        
        # Format log for JSON broadcast
        broadcast_data = {
            "timestamp": log_data["timestamp"].isoformat(),
            "level": log_data["level"],
            "event_type": log_data["event_type"],
            "message": log_data["message"],
        }
        
        # Remove clients that have closed
        dead_clients = []
        for client in _log_clients:
            try:
                await client.send_json(broadcast_data)
            except Exception:
                dead_clients.append(client)
        
        # Clean up dead clients
        for client in dead_clients:
            if client in _log_clients:
                _log_clients.remove(client)


def add_log_client(client):
    """Add a WebSocket client to receive real-time logs."""
    if client not in _log_clients:
        _log_clients.append(client)
        logger.info(f"WebSocket client added. Total clients: {len(_log_clients)}")


def remove_log_client(client):
    """Remove a WebSocket client from receiving logs."""
    if client in _log_clients:
        _log_clients.remove(client)
        logger.info(f"WebSocket client removed. Total clients: {len(_log_clients)}")


def get_log_client_count() -> int:
    """Get the number of connected log clients."""
    return len(_log_clients)


async def cleanup_old_logs(retention_days: int = 7):
    """
    Clean up logs older than retention_days.
    
    This should be called periodically (e.g., daily) to prevent database bloat.
    """
    try:
        async with get_db() as session:
            from datetime import timedelta
            
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            # Delete old logs
            result = await session.execute(
                select(EventLog.id).where(EventLog.timestamp < cutoff_date)
            )
            old_log_ids = result.scalars().all()
            
            if old_log_ids:
                await session.execute(
                    EventLog.__table__.delete().where(EventLog.id.in_(old_log_ids))
                )
                await session.commit()
                logger.info(f"Cleaned up {len(old_log_ids)} old log entries")
                
    except Exception as e:
        logger.error(f"Failed to cleanup old logs: {e}")


def setup_log_handler() -> DatabaseLogHandler:
    """
    Set up and return the database log handler.
    
    Call this during bot startup to enable dual logging.
    """
    handler = DatabaseLogHandler()
    
    # Set the asyncio loop
    try:
        loop = asyncio.get_event_loop()
        handler.set_loop(loop)
    except RuntimeError:
        # No event loop yet, will set later
        pass
    
    # Add handler to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    
    logger.info("Database log handler initialized")
    
    return handler


# Module-level function to initialize logging during bot startup
def init_logging():
    """Initialize the logging system with database handler."""
    return setup_log_handler()