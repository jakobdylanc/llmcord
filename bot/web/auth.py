"""Authentication and authorization for web portal."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from bot.db.connection import get_db_dependency
from bot.db.models import User
from bot.web.config import get_portal_config

logger = logging.getLogger(__name__)

# Security
SECRET_KEY = "llmcord-portal-secret-key"  # TODO: Move to config/env
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24

# Bearer token scheme
security = HTTPBearer()


# Pydantic models
class Token(BaseModel):
    """JWT token response."""
    access_token: str
    token_type: str
    user: dict


class LoginRequest(BaseModel):
    """Login request body."""
    username: str
    password: str


class SetupRequest(BaseModel):
    """First-time setup request body."""
    username: str
    password: str


class CurrentUser(BaseModel):
    """Current authenticated user."""
    id: int
    username: str


# Password functions
def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode(), salt).decode()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash."""
    return bcrypt.checkpw(plain_password.encode(), hashed_password.encode())


# JWT functions
def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_token(token: str) -> dict | None:
    """Decode and validate a JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None


# Auth dependency
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> CurrentUser:
    """Get current authenticated user from JWT token."""
    token = credentials.credentials
    
    payload = decode_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    username: str = payload.get("sub")
    user_id: int = payload.get("user_id")
    
    if username is None or user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
        )
    
    return CurrentUser(id=user_id, username=username)


# Optional auth (for endpoints that work with or without auth)
async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
) -> Optional[CurrentUser]:
    """Get current user if authenticated, None otherwise."""
    if credentials is None:
        return None
    
    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None


# Discord admin check
async def check_discord_admin(user: CurrentUser, discord_user_id: int) -> bool:
    """Check if a Discord user ID is in the admin list."""
    config = get_portal_config()
    if not config.require_discord_admin:
        return True
    
    return discord_user_id in config.admin_ids


# API Routes
async def setup_portal(
    request: SetupRequest,
    db: AsyncSession,
) -> Token:
    """First-time setup: create initial admin user.
    
    Only works when no users exist in the database.
    """
    # Check if users exist
    result = await db.execute(select(User))
    existing_users = result.scalars().all()
    
    if existing_users:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Setup already complete. Users exist.",
        )
    
    # Validate password length
    if len(request.password) < 6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 6 characters",
        )
    
    # Create user
    password_hash = hash_password(request.password)
    user = User(
        username=request.username,
        password_hash=password_hash,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    
    logger.info(f"Created initial user: {user.username}")
    
    # Create token
    access_token = create_access_token(
        data={"sub": user.username, "user_id": user.id}
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        user=user.to_dict(),
    )


async def login(
    request: LoginRequest,
    db: AsyncSession,
) -> Token:
    """Login with username and password."""
    # Find user
    result = await db.execute(
        select(User).where(User.username == request.username)
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )
    
    # Verify password
    if not verify_password(request.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )
    
    # Create token
    access_token = create_access_token(
        data={"sub": user.username, "user_id": user.id}
    )
    
    logger.info(f"User logged in: {user.username}")
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        user=user.to_dict(),
    )


async def get_users(
    db: AsyncSession,
    current_user: CurrentUser = Depends(get_current_user),
) -> list[dict]:
    """Get list of users (admin only)."""
    result = await db.execute(select(User))
    users = result.scalars().all()
    return [user.to_dict() for user in users]


async def check_has_users(db: AsyncSession) -> bool:
    """Check if any users exist in the database."""
    result = await db.execute(select(User).limit(1))
    return result.scalar_one_or_none() is not None