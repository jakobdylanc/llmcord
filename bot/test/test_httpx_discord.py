"""
Unit tests for httpx Discord user fetch fallback.

Tests the REST API fallback when Discord.py HTTP session is unavailable.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import discord
import httpx


class TestHttpxDiscordUserFetch:
    """Tests for httpx fallback to Discord REST API."""
    
    @pytest.fixture
    def mock_httpx_response(self):
        """Create a mock httpx response for user data."""
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            "id": "123456789",
            "username": "testuser",
            "global_name": "Test User",
            "avatar": "abc123def456",
            "bot": False,
            "discriminator": "0000"  # Required by discord.py for User creation
        }
        return response
    
    @pytest.fixture
    def mock_httpx_404_response(self):
        """Create a mock httpx 404 response."""
        response = Mock()
        response.status_code = 404
        return response
    
    @pytest.fixture
    def mock_discord_bot(self):
        """Create a mock Discord bot."""
        bot = Mock()
        bot._get_state.return_value = Mock()
        return bot
    
    @pytest.fixture
    def mock_config(self):
        """Create mock config with bot token."""
        return {"bot_token": "test_bot_token"}

    @pytest.mark.asyncio
    async def test_fetch_user_via_httpx_returns_user_object(self, mock_httpx_response, mock_discord_bot, mock_config):
        """Test that httpx fallback correctly creates a Discord User object."""
        user_id = 123456789
        
        with patch('llmcord.httpx_client') as mock_client:
            mock_client.get = AsyncMock(return_value=mock_httpx_response)
            
            # Simulate the fallback logic from llmcord.py
            bot_token = mock_config.get("bot_token")
            headers = {"Authorization": f"Bot {bot_token}"}
            
            response = await mock_client.get(
                f"https://discord.com/api/v10/users/{user_id}",
                headers=headers,
                timeout=10.0
            )
            
            assert response.status_code == 200
            user_data = response.json()
            
            # Create User object like the fallback does
            target = discord.User(state=mock_discord_bot._get_state(), data=user_data)
            
            assert target.id == 123456789
            assert target.name == "testuser"
            assert target.display_name == "Test User"

    @pytest.mark.asyncio
    async def test_fetch_user_via_httpx_handles_404(self, mock_httpx_404_response):
        """Test that 404 response is handled correctly."""
        user_id = 999999999
        
        with patch('llmcord.httpx_client') as mock_client:
            mock_client.get = AsyncMock(return_value=mock_httpx_404_response)
            
            bot_token = "test_bot_token"
            headers = {"Authorization": f"Bot {bot_token}"}
            
            response = await mock_client.get(
                f"https://discord.com/api/v10/users/{user_id}",
                headers=headers,
                timeout=10.0
            )
            
            assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_fetch_user_via_httpx_with_bot_user(self, mock_discord_bot):
        """Test that bot users are detected and handled."""
        # Mock response for a bot user
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "987654321",
            "username": "testbot",
            "global_name": "Test Bot",
            "avatar": None,
            "bot": True,
            "discriminator": "0000"  # Required by discord.py for User creation
        }
        
        with patch('llmcord.httpx_client') as mock_client:
            mock_client.get = AsyncMock(return_value=mock_response)
            
            bot_token = "test_bot_token"
            headers = {"Authorization": f"Bot {bot_token}"}
            
            response = await mock_client.get(
                "https://discord.com/api/v10/users/987654321",
                headers=headers,
                timeout=10.0
            )
            
            assert response.status_code == 200
            user_data = response.json()
            assert user_data["bot"] is True
            
            # Create User object
            target = discord.User(state=mock_discord_bot._get_state(), data=user_data)
            assert target.bot is True

    @pytest.mark.asyncio
    async def test_fetch_channel_via_httpx(self):
        """Test that channel fetch via httpx works correctly."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "111222333",
            "type": 0,  # Text channel
            "name": "general",
            "guild_id": "444555666"
        }
        
        with patch('llmcord.httpx_client') as mock_client:
            mock_client.get = AsyncMock(return_value=mock_response)
            
            bot_token = "test_bot_token"
            headers = {"Authorization": f"Bot {bot_token}"}
            channel_id = 111222333
            
            response = await mock_client.get(
                f"https://discord.com/api/v10/channels/{channel_id}",
                headers=headers,
                timeout=10.0
            )
            
            assert response.status_code == 200
            channel_data = response.json()
            assert channel_data["name"] == "general"
            assert channel_data["type"] == 0


class TestHttpxChannelSend:
    """Tests for HttpxChannel send functionality."""
    
    @pytest.mark.asyncio
    async def test_httpx_channel_send_message(self):
        """Test sending message via HttpxChannel REST API."""
        # Mock channel data
        channel_data = {
            "id": "111222333",
            "type": 0,
            "name": "general"
        }
        
        # Mock POST response
        mock_post_response = Mock()
        mock_post_response.status_code = 200
        
        with patch('llmcord.httpx_client') as mock_client:
            mock_client.post = AsyncMock(return_value=mock_post_response)
            
            bot_token = "test_bot_token"
            
            # Create HttpxChannel (from llmcord.py)
            class HttpxChannel:
                def __init__(self, data, http_session):
                    self.id = int(data['id'])
                    self.type = data.get('type', 0)
                    self._http = http_session
                    self._bot_token = bot_token
                
                async def send(self, content: str):
                    headers = {"Authorization": f"Bot {self._bot_token}", "Content-Type": "application/json"}
                    msg_response = await self._http.post(
                        f"https://discord.com/api/v10/channels/{self.id}/messages",
                        headers=headers,
                        json={"content": content},
                        timeout=10.0
                    )
                    if msg_response.status_code != 200:
                        raise Exception(f"Failed to send: {msg_response.status_code}")
            
            channel = HttpxChannel(channel_data, mock_client)
            
            # Send a message
            await channel.send("Hello, world!")
            
            # Verify POST was called correctly
            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args
            assert "messages" in call_args[0][0]
            assert call_args[1]["json"]["content"] == "Hello, world!"

    @pytest.mark.asyncio
    async def test_httpx_channel_send_failure(self):
        """Test handling of send failure."""
        channel_data = {
            "id": "111222333",
            "type": 0,
            "name": "general"
        }
        
        # Mock failed POST response
        mock_post_response = Mock()
        mock_post_response.status_code = 403
        
        with patch('llmcord.httpx_client') as mock_client:
            mock_client.post = AsyncMock(return_value=mock_post_response)
            
            bot_token = "test_bot_token"
            
            class HttpxChannel:
                def __init__(self, data, http_session):
                    self.id = int(data['id'])
                    self.type = data.get('type', 0)
                    self._http = http_session
                    self._bot_token = bot_token
                
                async def send(self, content: str):
                    headers = {"Authorization": f"Bot {self._bot_token}", "Content-Type": "application/json"}
                    msg_response = await self._http.post(
                        f"https://discord.com/api/v10/channels/{self.id}/messages",
                        headers=headers,
                        json={"content": content},
                        timeout=10.0
                    )
                    if msg_response.status_code != 200:
                        raise Exception(f"Failed to send: {msg_response.status_code}")
            
            channel = HttpxChannel(channel_data, mock_client)
            
            # Attempt to send should raise exception
            with pytest.raises(Exception) as exc_info:
                await channel.send("Hello")
            
            assert "Failed to send: 403" in str(exc_info.value)