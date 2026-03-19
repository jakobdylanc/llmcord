"""Unit tests for bot/web/log_handler.py."""

import pytest
import logging
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch


class TestDatabaseLogHandler:
    """Test DatabaseLogHandler class."""

    def test_handler_initialization(self):
        """Test handler can be initialized."""
        from bot.web.log_handler import DatabaseLogHandler
        
        handler = DatabaseLogHandler()
        assert handler is not None
        assert handler._batch_size == 10

    def test_handler_has_emit_method(self):
        """Test handler has emit method."""
        from bot.web.log_handler import DatabaseLogHandler
        
        handler = DatabaseLogHandler()
        assert hasattr(handler, 'emit')
        assert callable(handler.emit)

    def test_handler_set_loop(self):
        """Test handler can set event loop."""
        from bot.web.log_handler import DatabaseLogHandler
        
        handler = DatabaseLogHandler()
        mock_loop = MagicMock()
        mock_loop.is_running.return_value = False
        
        handler.set_loop(mock_loop)
        assert handler._loop == mock_loop


class TestLogClientManagement:
    """Test WebSocket client management functions."""

    def test_add_log_client(self):
        """Test adding a WebSocket client."""
        from bot.web.log_handler import _log_clients, add_log_client, get_log_client_count
        
        # Clear any existing clients
        _log_clients.clear()
        
        mock_client = MagicMock()
        add_log_client(mock_client)
        
        assert get_log_client_count() == 1
        assert mock_client in _log_clients
        
        # Clean up
        _log_clients.clear()

    def test_add_duplicate_client(self):
        """Test that duplicate clients are not added."""
        from bot.web.log_handler import _log_clients, add_log_client, get_log_client_count
        
        _log_clients.clear()
        
        mock_client = MagicMock()
        add_log_client(mock_client)
        add_log_client(mock_client)
        
        assert get_log_client_count() == 1
        
        _log_clients.clear()

    def test_remove_log_client(self):
        """Test removing a WebSocket client."""
        from bot.web.log_handler import _log_clients, add_log_client, remove_log_client, get_log_client_count
        
        _log_clients.clear()
        
        mock_client = MagicMock()
        add_log_client(mock_client)
        assert get_log_client_count() == 1
        
        remove_log_client(mock_client)
        assert get_log_client_count() == 0
        
        _log_clients.clear()

    def test_remove_nonexistent_client(self):
        """Test removing a client that doesn't exist."""
        from bot.web.log_handler import _log_clients, remove_log_client, get_log_client_count
        
        _log_clients.clear()
        
        mock_client = MagicMock()
        remove_log_client(mock_client)  # Should not raise
        
        assert get_log_client_count() == 0


class TestSetupLogHandler:
    """Test log handler setup function."""

    def test_setup_log_handler_returns_handler(self):
        """Test setup_log_handler returns a handler."""
        from bot.web.log_handler import setup_log_handler, DatabaseLogHandler
        
        handler = setup_log_handler()
        assert isinstance(handler, DatabaseLogHandler)

    def test_setup_log_handler_adds_to_root_logger(self):
        """Test setup_log_handler adds handler to root logger."""
        from bot.web.log_handler import setup_log_handler
        
        root_logger = logging.getLogger()
        initial_handlers = len(root_logger.handlers)
        
        setup_log_handler()
        
        # Handler should be added
        assert len(root_logger.handlers) > initial_handlers


class TestCleanupOldLogs:
    """Test log cleanup function."""

    @pytest.mark.asyncio
    async def test_cleanup_old_logs(self):
        """Test cleanup function runs without error."""
        from bot.web.log_handler import cleanup_old_logs
        
        # Create mock session that works as async context manager
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [1, 2, 3]
        mock_session.execute.return_value = mock_result
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        async def mock_get_db():
            yield mock_session
        
        with patch('bot.web.log_handler.get_db_session', return_value=mock_get_db()):
            # Should not raise
            await cleanup_old_logs(retention_days=7)

    @pytest.mark.asyncio
    async def test_cleanup_with_zero_logs(self):
        """Test cleanup with no old logs."""
        from bot.web.log_handler import cleanup_old_logs
        
        # Create mock session that works as async context manager
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []  # No old logs
        mock_session.execute.return_value = mock_result
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        async def mock_get_db():
            yield mock_session
        
        with patch('bot.web.log_handler.get_db_session', return_value=mock_get_db()):
            # Should run without error even with no old logs
            await cleanup_old_logs(retention_days=7)


class TestInitLogging:
    """Test init_logging function."""

    def test_init_logging_returns_handler(self):
        """Test init_logging returns DatabaseLogHandler."""
        from bot.web.log_handler import init_logging, DatabaseLogHandler
        
        handler = init_logging()
        assert isinstance(handler, DatabaseLogHandler)