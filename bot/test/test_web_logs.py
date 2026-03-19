"""Unit tests for bot/web/routes/logs.py."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch
from sqlalchemy.ext.asyncio import AsyncSession

from bot.web.routes.logs import (
    LogEntry,
    LogsResponse,
    get_logs,
    get_log_levels,
    get_log_types,
)


class TestLogEntry:
    """Test LogEntry model."""

    def test_from_attributes(self):
        """Test creating LogEntry from model attributes."""
        entry = LogEntry(
            id=1,
            timestamp="2026-03-18T12:00:00",
            level="INFO",
            event_type="test_event",
            message="Test message",
            metadata={"key": "value"},
        )
        
        assert entry.id == 1
        assert entry.level == "INFO"
        assert entry.message == "Test message"
        assert entry.metadata == {"key": "value"}


class TestLogsEndpoint:
    """Test get_logs endpoint."""

    @pytest.mark.asyncio
    async def test_get_logs_basic(self):
        """Test basic log retrieval."""
        # Create mock logs
        mock_log1 = MagicMock()
        mock_log1.id = 1
        mock_log1.timestamp = datetime.now()
        mock_log1.level = "INFO"
        mock_log1.event_type = "test"
        mock_log1.message = "Test message 1"
        mock_log1.extra_data = {"key": "value1"}
        
        mock_log2 = MagicMock()
        mock_log2.id = 2
        mock_log2.timestamp = datetime.now() - timedelta(hours=1)
        mock_log2.level = "WARNING"
        mock_log2.event_type = "test"
        mock_log2.message = "Test message 2"
        mock_log2.extra_data = None
        
        # Create mock db session
        mock_db = AsyncMock(spec=AsyncSession)
        
        # Mock the execute result - scalars().all() returns logs
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [mock_log1, mock_log2]
        
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_db.execute = AsyncMock(return_value=mock_result)
        
        # Mock config
        with patch('bot.web.routes.logs.get_portal_config') as mock_config:
            mock_config.return_value = MagicMock(
                logs_levels=["INFO", "WARNING", "ERROR"]
            )
            
            # Call the endpoint
            response = await get_logs(
                db=mock_db,
                level=None,
                event_type=None,
                since=None,
                until=None,
                page=1,
                page_size=50,
            )
        
        assert isinstance(response, LogsResponse)
        assert len(response.logs) == 2
        assert response.page == 1
        assert response.page_size == 50

    @pytest.mark.asyncio
    async def test_get_logs_with_level_filter(self):
        """Test filtering logs by level."""
        mock_db = AsyncMock(spec=AsyncSession)
        
        # Mock empty result
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_db.execute = AsyncMock(return_value=mock_result)
        
        with patch('bot.web.routes.logs.get_portal_config') as mock_config:
            mock_config.return_value = MagicMock(
                logs_levels=["INFO", "WARNING", "ERROR"]
            )
            
            response = await get_logs(
                db=mock_db,
                level="ERROR",
                event_type=None,
                since=None,
                until=None,
                page=1,
                page_size=50,
            )
        
        assert isinstance(response, LogsResponse)
        assert response.logs == []

    @pytest.mark.asyncio
    async def test_get_logs_pagination(self):
        """Test pagination parameters."""
        mock_db = AsyncMock(spec=AsyncSession)
        
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_db.execute = AsyncMock(return_value=mock_result)
        
        with patch('bot.web.routes.logs.get_portal_config') as mock_config:
            mock_config.return_value = MagicMock(
                logs_levels=["INFO", "WARNING", "ERROR"]
            )
            
            response = await get_logs(
                db=mock_db,
                level=None,
                event_type=None,
                since=None,
                until=None,
                page=3,
                page_size=10,
            )
        
        assert response.page == 3
        assert response.page_size == 10

    @pytest.mark.asyncio
    async def test_get_logs_invalid_level(self):
        """Test that invalid log levels are filtered out by config."""
        mock_db = AsyncMock(spec=AsyncSession)
        
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_db.execute = AsyncMock(return_value=mock_result)
        
        # Config only allows INFO, WARNING, ERROR - DEBUG should be filtered
        with patch('bot.web.routes.logs.get_portal_config') as mock_config:
            mock_config.return_value = MagicMock(
                logs_levels=["INFO", "WARNING", "ERROR"]
            )
            
            response = await get_logs(
                db=mock_db,
                level="DEBUG",  # Not in allowed levels
                event_type=None,
                since=None,
                until=None,
                page=1,
                page_size=50,
            )
        
        # Should return empty because DEBUG is not in allowed levels
        assert response.logs == []


class TestLogLevels:
    """Test get_log_levels endpoint."""

    @pytest.mark.asyncio
    async def test_get_log_levels(self):
        """Test getting available log levels."""
        with patch('bot.web.routes.logs.get_portal_config') as mock_config:
            mock_config.return_value = MagicMock(
                logs_levels=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            )
            
            levels = await get_log_levels()
        
        assert levels == ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class TestLogTypes:
    """Test get_log_types endpoint."""

    @pytest.mark.asyncio
    async def test_get_log_types(self):
        """Test getting unique event types."""
        mock_db = AsyncMock(spec=AsyncSession)
        
        # Mock the distinct query result
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [
            "bot_start",
            "message_received",
            "command_executed",
            "error",
        ]
        mock_db.execute = AsyncMock(return_value=mock_result)
        
        types = await get_log_types(db=mock_db)
        
        assert len(types) == 4
        assert "bot_start" in types
        assert "message_received" in types
        assert "command_executed" in types
        assert "error" in types