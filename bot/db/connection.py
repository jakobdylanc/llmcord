"""Database connection and initialization."""

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool

from bot.db.models import Base

# Database URL - configurable via environment
DATABASE_URL = os.environ.get(
    "PORTAL_DB",
    "sqlite+aiosqlite:///data/portal.db"
)

# Async engine and session maker
_engine = None
_session_maker = None


def get_database_url() -> str:
    """Get the database URL, creating directory if needed."""
    url = DATABASE_URL
    # For SQLite, ensure data directory exists
    if url.startswith("sqlite"):
        # Extract path from URL
        db_path = url.replace("sqlite+aiosqlite:///", "")
        if db_path and db_path != ":memory:":
            db_dir = os.path.dirname(db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
    return url


def init_engine():
    """Initialize the database engine."""
    global _engine, _session_maker
    
    if _engine is None:
        url = get_database_url()
        _engine = create_async_engine(
            url,
            echo=False,
            poolclass=NullPool,  # Use NullPool for SQLite
        )
        _session_maker = async_sessionmaker(
            _engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
    
    return _engine, _session_maker


async def init_db():
    """Initialize database tables."""
    engine, _ = init_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db():
    """Close database connections."""
    global _engine, _session_maker
    
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_maker = None


@asynccontextmanager
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Get database session context manager."""
    _, session_maker = init_engine()
    
    async with session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_db_session() -> AsyncSession:
    """Get a database session (caller must close)."""
    _, session_maker = init_engine()
    return session_maker()


# FastAPI dependency - yields a session and handles commit/rollback
async def get_db_dependency() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for database sessions.
    
    This properly handles the async context manager protocol for FastAPI's Depends().
    Usage: async def endpoint(db: AsyncSession = Depends(get_db_dependency)):
    """
    _, session_maker = init_engine()
    
    async with session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()