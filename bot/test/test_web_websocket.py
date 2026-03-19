"""Unit tests for WebSocket endpoints."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi import WebSocketDisconnect


class TestWebSocketEndpoint:
    """Test WebSocket /ws/logs endpoint."""

    def test_websocket_endpoint_exists(self):
        """Test that WebSocket endpoint is registered."""
        from bot.web.server import app
        
        # Check that the route exists
        routes = [route.path for route in app.routes]
        assert "/ws/logs" in routes

    def test_websocket_status_endpoint_exists(self):
        """Test that WebSocket status endpoint is registered."""
        from bot.web.server import app
        
        routes = [route.path for route in app.routes]
        assert "/ws/status" in routes


class TestWebSocketFunctions:
    """Test WebSocket-related functions in log_handler."""

    def test_add_log_client_function_exists(self):
        """Test add_log_client function exists."""
        from bot.web.log_handler import add_log_client
        assert callable(add_log_client)

    def test_remove_log_client_function_exists(self):
        """Test remove_log_client function exists."""
        from bot.web.log_handler import remove_log_client
        assert callable(remove_log_client)

    def test_get_log_client_count_function_exists(self):
        """Test get_log_client_count function exists."""
        from bot.web.log_handler import get_log_client_count
        assert callable(get_log_client_count)


class TestWebSocketBroadcast:
    """Test WebSocket log broadcast functionality."""

    def test_broadcast_removes_dead_clients(self):
        """Test that broadcast removes disconnected clients."""
        from bot.web.log_handler import _log_clients, DatabaseLogHandler
        
        # Clear clients
        _log_clients.clear()
        
        handler = DatabaseLogHandler()
        
        # Create mock client that raises exception on send
        mock_client = MagicMock()
        mock_client.send_json = AsyncMock(side_effect=Exception("Connection closed"))
        
        # Add mock client
        _log_clients.append(mock_client)
        
        # Create test log data
        log_data = {
            "timestamp": MagicMock(isoformat=MagicMock(return_value="2026-03-18T12:00:00")),
            "level": "INFO",
            "event_type": "test",
            "message": "Test message",
        }
        
        # Run broadcast - should remove dead client
        import asyncio
        loop = asyncio.new_event_loop()
        loop.run_until_complete(handler._broadcast_log(log_data))
        loop.close()
        
        # Client should be removed
        assert mock_client not in _log_clients
        
        # Clean up
        _log_clients.clear()

    def test_broadcast_sends_to_active_clients(self):
        """Test that broadcast sends to active clients."""
        from bot.web.log_handler import _log_clients
        
        # Clear clients
        _log_clients.clear()
        
        # Create mock client that works
        mock_client = MagicMock()
        mock_client.send_json = AsyncMock()
        
        # Add mock client
        _log_clients.append(mock_client)
        
        # Import handler
        from bot.web.log_handler import DatabaseLogHandler
        handler = DatabaseLogHandler()
        
        log_data = {
            "timestamp": MagicMock(isoformat=MagicMock(return_value="2026-03-18T12:00:00")),
            "level": "INFO",
            "event_type": "test",
            "message": "Test message",
        }
        
        # Run broadcast
        import asyncio
        loop = asyncio.new_event_loop()
        loop.run_until_complete(handler._broadcast_log(log_data))
        loop.close()
        
        # Client should have been called
        mock_client.send_json.assert_called_once()
        
        # Clean up
        _log_clients.clear()


class TestWebSocketStatus:
    """Test WebSocket status endpoint."""

    @pytest.mark.asyncio
    async def test_websocket_status_no_clients(self):
        """Test status when no clients connected."""
        from bot.web.server import websocket_status
        from bot.web.log_handler import _log_clients
        
        _log_clients.clear()
        
        result = await websocket_status()
        
        assert result["log_clients"] == 0
        assert result["status"] == "no_clients"

    @pytest.mark.asyncio
    async def test_websocket_status_with_clients(self):
        """Test status when clients connected."""
        from bot.web.server import websocket_status
        from bot.web.log_handler import _log_clients
        
        _log_clients.clear()
        
        # Add mock clients
        mock_client = MagicMock()
        _log_clients.append(mock_client)
        _log_clients.append(MagicMock())
        
        result = await websocket_status()
        
        assert result["log_clients"] == 2
        assert result["status"] == "connected"
        
        # Clean up
        _log_clients.clear()


class TestWebSocketConnection:
    """Test WebSocket connection handling."""

    @pytest.mark.asyncio
    async def test_websocket_accepts_connection(self):
        """Test that WebSocket accepts connection."""
        from bot.web.server import websocket_logs
        
        mock_websocket = MagicMock()
        mock_websocket.accept = AsyncMock()
        mock_websocket.receive_text = AsyncMock(side_effect=WebSocketDisconnect())
        
        # Should not raise
        try:
            await websocket_logs(mock_websocket)
        except Exception:
            pass  # Expected on disconnect
        
        mock_websocket.accept.assert_called_once()

    @pytest.mark.asyncio
    async def test_websocket_handles_ping(self):
        """Test WebSocket ping/pong."""
        from bot.web.server import websocket_logs
        
        mock_websocket = MagicMock()
        mock_websocket.accept = AsyncMock()
        
        # First return "ping", then disconnect
        mock_websocket.receive_text = AsyncMock(side_effect=["ping", WebSocketDisconnect()])
        mock_websocket.send_text = AsyncMock()
        
        # Should handle ping
        try:
            await websocket_logs(mock_websocket)
        except Exception:
            pass
        
        # Should have sent pong
        mock_websocket.send_text.assert_called_with("pong")