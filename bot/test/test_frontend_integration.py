"""Test frontend API integration with axios and WebSocket client."""

import pytest
import os
from unittest.mock import AsyncMock, patch, MagicMock


class TestFrontendAPI:
    """Tests for frontend API integration."""

    def test_imports(self):
        """Verify required frontend dependencies are available."""
        # These would be tested in the frontend itself with Vitest/Jest
        # Here we just verify the Python API structure exists
        from bot.web.server import app
        from bot.web.auth import login, setup_portal
        from bot.web.routes.status import router as status_router
        from bot.web.routes.logs import router as logs_router
        from bot.web.routes.config import router as config_router
        from bot.web.routes.servers import router as servers_router
        
        assert app is not None
        assert login is not None
        assert setup_portal is not None

    @pytest.mark.asyncio
    async def test_auth_endpoints_exist(self):
        """Test that authentication endpoints are registered."""
        from bot.web.server import app
        
        routes = [r.path for r in app.routes]
        
        assert "/api/auth/setup" in routes
        assert "/api/auth/has-users" in routes
        assert "/api/auth/login" in routes

    @pytest.mark.asyncio
    async def test_status_endpoint_exists(self):
        """Test that status endpoint is registered."""
        from bot.web.server import app
        
        routes = [r.path for r in app.routes]
        
        assert "/api/status" in routes

    @pytest.mark.asyncio
    async def test_logs_endpoint_exists(self):
        """Test that logs endpoint is registered."""
        from bot.web.server import app
        
        routes = [r.path for r in app.routes]
        
        assert "/api/logs" in routes

    @pytest.mark.asyncio
    async def test_config_endpoints_exist(self):
        """Test that config endpoints are registered."""
        from bot.web.server import app
        
        routes = [r.path for r in app.routes]
        
        assert "/api/config" in routes
        assert "/api/refresh" in routes

    @pytest.mark.asyncio
    async def test_servers_endpoint_exists(self):
        """Test that servers endpoint is registered."""
        from bot.web.server import app
        
        routes = [r.path for r in app.routes]
        
        assert "/api/servers" in routes

    @pytest.mark.asyncio
    async def test_websocket_endpoint_exists(self):
        """Test that WebSocket endpoint is registered."""
        from bot.web.server import app
        
        # WebSocket routes have a different format
        websocket_routes = [r.path for r in app.routes if hasattr(r, 'ws') or 'websocket' in str(r.path).lower()]
        
        assert "/ws/logs" in [r.path for r in app.routes]


class TestWebSocketConnection:
    """Tests for WebSocket log streaming."""

    def test_websocket_protocol(self):
        """Test WebSocket uses correct protocol for frontend."""
        # Frontend uses: `${protocol}//${window.location.host}/ws/logs`
        # Where protocol is 'ws:' or 'wss:' based on page protocol
        # This is validated by the frontend LogViewer component
        
        # Verify WebSocket endpoint exists and accepts connections
        from bot.web.server import app
        
        assert any("/ws/logs" in str(r.path) for r in app.routes)

    def test_websocket_ping_pong(self):
        """Test WebSocket ping/pong mechanism."""
        # Frontend sends "ping", expects "pong" back
        # This is handled in the websocket_logs function
        
        from bot.web.server import app
        import inspect
        
        # Get the websocket_logs function source
        for route in app.routes:
            if hasattr(route, 'endpoint') and route.path == '/ws/logs':
                source = inspect.getsource(route.endpoint)
                assert 'ping' in source
                assert 'pong' in source
                break


class TestDashboardDisplayLabels:
    """Tests for Dashboard display labels (Task 16.4)."""

    def test_dashboard_file_exists(self):
        """Verify Dashboard.tsx exists in frontend."""
        dashboard_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'web', 'src', 'components', 'Dashboard.tsx'
        )
        assert os.path.exists(dashboard_path), f"Dashboard.tsx not found at {dashboard_path}"

    def test_dashboard_has_bot_information_header(self):
        """Verify Dashboard uses 'Bot Information' header."""
        dashboard_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'web', 'src', 'components', 'Dashboard.tsx'
        )
        with open(dashboard_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Should have "Bot Information" header
        assert 'Bot Information' in content, "Dashboard should have 'Bot Information' header"
        # Should NOT have old "Bot Status" header
        assert 'Bot Status' not in content, "Dashboard should not have 'Bot Status' header"

    def test_dashboard_uses_mood_label(self):
        """Verify Dashboard uses 'Mood' label instead of 'Activity'."""
        dashboard_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'web', 'src', 'components', 'Dashboard.tsx'
        )
        with open(dashboard_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Should use "Mood" for status message
        assert '<strong>Mood:</strong>' in content, "Dashboard should use 'Mood' label"
        # Should NOT have old "Activity" label
        assert '<strong>Activity:</strong>' not in content, "Dashboard should not have 'Activity' label"

    def test_dashboard_has_status_indicator(self):
        """Verify Dashboard has visual status indicator (status dot)."""
        dashboard_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'web', 'src', 'components', 'Dashboard.tsx'
        )
        with open(dashboard_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Should have online status indicator with green color
        assert '#22c55e' in content, "Dashboard should have green status indicator (#22c55e)"
        # Should have offline status indicator with grey color
        assert '#6b7280' in content, "Dashboard should have grey offline indicator (#6b7280)"
        # Should use status.online for the conditional
        assert 'status.online' in content, "Dashboard should use status.online for status indicator"

    def test_dashboard_has_flex_layout(self):
        """Verify Dashboard uses flex layout with avatar on left."""
        dashboard_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'web', 'src', 'components', 'Dashboard.tsx'
        )
        with open(dashboard_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Should use flex layout
        assert "display: 'flex'" in content, "Dashboard should use flex layout"
        # Should have gap between avatar and info
        assert "gap: '2rem'" in content, "Dashboard should have gap between avatar and info"


class TestServersEndpoint:
    """Tests for servers endpoint data format (Task 17.1)."""

    def test_servers_endpoint_response_format(self):
        """Verify servers endpoint returns expected fields."""
        from bot.web.routes.servers import Server, ServersResponse, TextChannel
        
        # Test with mock data
        server = Server(
            id=123456789,
            name="Test Server",
            icon="https://cdn.discordapp.com/icons/123/abc.png",
            member_count=100,
            channel_count=25,
            owner_id=987654321,
            text_channels=[
                TextChannel(id=111, name="general"),
                TextChannel(id=222, name="bot-commands"),
            ]
        )
        
        assert server.id == 123456789
        assert server.name == "Test Server"
        assert server.icon == "https://cdn.discordapp.com/icons/123/abc.png"
        assert server.member_count == 100
        assert server.channel_count == 25
        assert server.owner_id == 987654321
        assert len(server.text_channels) == 2

    def test_servers_endpoint_route_exists(self):
        """Verify /api/servers endpoint is registered."""
        from bot.web.server import app
        
        routes = [r.path for r in app.routes]
        assert "/api/servers" in routes

    def test_server_list_frontend_interface(self):
        """Verify ServerList.tsx has correct interface."""
        import os
        
        server_list_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'web', 'src', 'components', 'ServerList.tsx'
        )
        with open(server_list_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Should have member_count
        assert 'member_count: number' in content, "ServerList should have member_count field"
        # Should have channel_count
        assert 'channel_count: number' in content, "ServerList should have channel_count field"
        # Should display member count
        assert 'Members:' in content, "ServerList should display member count"
        # Should display channel count
        assert 'Channels:' in content, "ServerList should display channel count"


class TestServerDetailEndpoint:
    """Tests for server detail endpoint (Task 17.2)."""

    def test_server_detail_response_models(self):
        """Verify server detail response models."""
        from bot.web.routes.servers import (
            ServerDetailResponse,
            GuildMember,
            GuildChannel,
            GuildPermissions,
            PermissionUpdateRequest,
            PermissionUpdateResponse,
        )
        
        # Test GuildMember
        member = GuildMember(id=123, username="testuser", display_name="Test", is_owner=False)
        assert member.id == 123
        assert member.username == "testuser"
        
        # Test GuildChannel
        channel = GuildChannel(id=456, name="general", type="text")
        assert channel.id == 456
        assert channel.name == "general"
        
        # Test GuildPermissions
        perms = GuildPermissions(can_send_messages=True, can_manage_guild=False)
        assert perms.can_send_messages is True
        assert perms.can_manage_guild is False
        
        # Test PermissionUpdateRequest
        req = PermissionUpdateRequest(action="grant", permission="send_messages")
        assert req.action == "grant"
        
        # Test PermissionUpdateResponse
        resp = PermissionUpdateResponse(success=True, message="Granted")
        assert resp.success is True

    def test_server_detail_endpoint_exists(self):
        """Verify server detail endpoint is registered."""
        from bot.web.server import app
        
        routes = [r.path for r in app.routes]
        # Check for the pattern, FastAPI converts {guild_id} to {guild_id}
        assert any('/servers/{guild_id}' in r for r in routes), "Server detail endpoint should exist"

    def test_permission_update_endpoint_exists(self):
        """Verify permission update endpoint is registered."""
        from bot.web.server import app
        
        routes = [r.path for r in app.routes]
        assert any('/servers/{guild_id}/permissions' in r for r in routes), "Permission update endpoint should exist"

    def test_server_detail_frontend_exists(self):
        """Verify ServerDetail.tsx component exists."""
        import os
        
        server_detail_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'web', 'src', 'components', 'ServerDetail.tsx'
        )
        assert os.path.exists(server_detail_path), "ServerDetail.tsx should exist"

    def test_server_detail_route_in_app(self):
        """Verify App.tsx has route for server detail."""
        import os
        
        app_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'web', 'src', 'App.tsx'
        )
        with open(app_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert '/servers/:id' in content or 'servers/:id' in content, "App should have route for server detail"
        assert 'ServerDetail' in content, "App should import ServerDetail component"

    @pytest.mark.asyncio
    async def test_server_detail_with_mock_bot(self):
        """Test server detail endpoint with mocked Discord bot."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from fastapi.testclient import TestClient
        
        # Create mock guild
        mock_member = MagicMock()
        mock_member.id = 123
        mock_member.name = "testuser"
        mock_member.nick = "Test Nick"
        mock_member.display_name = "Test Nick"
        
        mock_channel = MagicMock()
        mock_channel.id = 456
        mock_channel.name = "general"
        mock_channel.type = MagicMock()
        mock_channel.type.name = "text"
        
        mock_guild = MagicMock()
        mock_guild.id = 111
        mock_guild.name = "Test Server"
        mock_guild.icon = None
        mock_guild.owner_id = 123
        mock_guild.members = [mock_member]
        mock_guild.channels = [mock_channel]
        mock_guild.me = MagicMock()
        mock_guild.me.guild_permissions = MagicMock()
        mock_guild.me.guild_permissions.send_messages = True
        mock_guild.me.guild_permissions.embed_links = True
        mock_guild.me.guild_permissions.attach_files = False
        mock_guild.me.guild_permissions.external_emojis = True
        mock_guild.me.guild_permissions.manage_messages = False
        mock_guild.me.guild_permissions.manage_channels = False
        mock_guild.me.guild_permissions.kick_members = False
        mock_guild.me.guild_permissions.ban_members = False
        mock_guild.me.guild_permissions.manage_guild = False
        
        # Create mock bot
        mock_bot = MagicMock()
        mock_bot.guilds = [mock_guild]
        
        with patch('bot.web.routes.servers.bot_state') as mock_state:
            mock_state._discord_bot = mock_bot
            
            # Import and test directly
            from bot.web.routes.servers import get_server_detail, ServerDetailResponse
            from bot.web.auth import CurrentUser
            
            # Create mock user
            mock_user = CurrentUser(id=1, username="test", user_id=1)
            
            # Call the endpoint
            result = await get_server_detail(guild_id=111, current_user=mock_user)
            
            # Verify response
            assert isinstance(result, ServerDetailResponse)
            assert result.id == 111
            assert result.name == "Test Server"
            assert len(result.members) == 1
            assert result.members[0].username == "testuser"
            assert len(result.channels) == 1
            assert result.permissions.can_send_messages is True

    def test_spa_catchall_route_exists(self):
        """Verify SPA catch-all route exists for frontend routing."""
        from bot.web.server import app
        
        routes = [r.path for r in app.routes]
        # Check for catch-all route pattern
        assert any('{full_path:path}' in r for r in routes), "SPA catch-all route should exist"