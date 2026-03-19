"""Unit tests for bot/web/auth.py."""

import pytest
from datetime import timedelta
from unittest.mock import AsyncMock, patch, MagicMock

from bot.web.auth import (
    hash_password,
    verify_password,
    create_access_token,
    decode_token,
    SetupRequest,
    LoginRequest,
    check_has_users,
)


class TestPasswordHashing:
    """Test password hashing functions."""

    def test_hash_password(self):
        """Test password hashing produces valid hash."""
        password = "testpassword123"
        hashed = hash_password(password)
        
        assert hashed != password
        assert len(hashed) > 0

    def test_verify_password_correct(self):
        """Test verifying correct password."""
        password = "testpassword123"
        hashed = hash_password(password)
        
        assert verify_password(password, hashed) is True

    def test_verify_password_incorrect(self):
        """Test verifying incorrect password."""
        password = "testpassword123"
        wrong_password = "wrongpassword"
        hashed = hash_password(password)
        
        assert verify_password(wrong_password, hashed) is False

    def test_verify_password_empty(self):
        """Test verifying with empty password."""
        password = "testpassword123"
        hashed = hash_password(password)
        
        assert verify_password("", hashed) is False


class TestJWT:
    """Test JWT token functions."""

    def test_create_access_token(self):
        """Test creating access token."""
        data = {"sub": "testuser", "user_id": 1}
        token = create_access_token(data)
        
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0

    def test_create_access_token_with_expiry(self):
        """Test creating access token with custom expiry."""
        data = {"sub": "testuser", "user_id": 1}
        expires = timedelta(hours=2)
        token = create_access_token(data, expires_delta=expires)
        
        assert token is not None

    def test_decode_token_valid(self):
        """Test decoding valid token."""
        data = {"sub": "testuser", "user_id": 1}
        token = create_access_token(data)
        payload = decode_token(token)
        
        assert payload is not None
        assert payload["sub"] == "testuser"
        assert payload["user_id"] == 1

    def test_decode_token_invalid(self):
        """Test decoding invalid token."""
        payload = decode_token("invalid.token.here")
        
        assert payload is None

    def test_decode_token_tampered(self):
        """Test decoding tampered token."""
        data = {"sub": "testuser", "user_id": 1}
        token = create_access_token(data)
        # Tamper with the token
        tampered = token[:-5] + "xxxxx"
        payload = decode_token(tampered)
        
        assert payload is None


class TestCheckHasUsers:
    """Test check_has_users function."""

    @pytest.mark.asyncio
    async def test_has_users_true(self):
        """Test returns True when users exist."""
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = MagicMock(id=1)  # Sync return
        mock_db.execute = AsyncMock(return_value=mock_result)
        
        result = await check_has_users(mock_db)
        
        assert result is True

    @pytest.mark.asyncio
    async def test_has_users_false(self):
        """Test returns False when no users exist."""
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None  # Sync return
        mock_db.execute = AsyncMock(return_value=mock_result)
        
        result = await check_has_users(mock_db)
        
        assert result is False


class TestSetupRequest:
    """Test SetupRequest Pydantic model."""

    def test_valid_request(self):
        """Test valid setup request."""
        request = SetupRequest(username="admin", password="password123")
        
        assert request.username == "admin"
        assert request.password == "password123"

    def test_min_password_length(self):
        """Test password validation."""
        request = SetupRequest(username="admin", password="12345")
        
        # Password is stored as-is, validation happens in endpoint
        assert request.password == "12345"


class TestLoginRequest:
    """Test LoginRequest Pydantic model."""

    def test_valid_request(self):
        """Test valid login request."""
        request = LoginRequest(username="admin", password="password123")
        
        assert request.username == "admin"
        assert request.password == "password123"