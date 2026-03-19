"""Unit tests for bot/db/models.py and bot/db/connection.py"""

import pytest
import pytest_asyncio
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest_asyncio.fixture
async def db_session():
    """Create a test database session."""
    # Use in-memory SQLite for testing
    import sqlalchemy.ext.asyncio
    from bot.db.models import Base, EventLog, Stats
    
    # Create engine with in-memory database
    engine = sqlalchemy.ext.asyncio.create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
    )
    
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Create session
    async_session = sqlalchemy.ext.asyncio.async_sessionmaker(
        engine, class_=sqlalchemy.ext.asyncio.AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session
    
    await engine.dispose()


@pytest.mark.asyncio
async def test_create_event_log(db_session):
    """Test creating an event log entry."""
    from bot.db.models import EventLog
    
    log = EventLog(
        timestamp=datetime.now(),
        level="INFO",
        event_type="TEST",
        message="Test message",
        extra_data={"user_id": 12345}
    )
    
    db_session.add(log)
    await db_session.commit()
    
    # Refresh to get the ID
    await db_session.refresh(log)
    
    assert log.id is not None
    assert log.level == "INFO"
    assert log.event_type == "TEST"
    assert log.message == "Test message"


@pytest.mark.asyncio
async def test_event_log_to_dict(db_session):
    """Test EventLog.to_dict() method."""
    from bot.db.models import EventLog
    
    log = EventLog(
        timestamp=datetime(2026, 3, 18, 12, 0, 0),
        level="WARNING",
        event_type="MESSAGE",
        message="Test message",
        extra_data={"channel_id": 123}
    )
    
    db_session.add(log)
    await db_session.commit()
    
    result = log.to_dict()
    
    assert result["level"] == "WARNING"
    assert result["event_type"] == "MESSAGE"
    assert "timestamp" in result
    assert "metadata" in result


@pytest.mark.asyncio
async def test_create_stats(db_session):
    """Test creating a stats entry."""
    from bot.db.models import Stats
    
    stat = Stats(
        key="messages_processed",
        value=100
    )
    
    db_session.add(stat)
    await db_session.commit()
    
    await db_session.refresh(stat)
    
    assert stat.key == "messages_processed"
    assert stat.value == 100


@pytest.mark.asyncio
async def test_update_stats(db_session):
    """Test updating stats value."""
    from bot.db.models import Stats
    
    stat = Stats(key="test_counter", value=0)
    db_session.add(stat)
    await db_session.commit()
    
    # Update the value
    stat.value = 42
    await db_session.commit()
    
    # Query again to verify
    from sqlalchemy import select
    result = await db_session.execute(select(Stats).where(Stats.key == "test_counter"))
    updated_stat = result.scalar_one()
    
    assert updated_stat.value == 42


@pytest.mark.asyncio
async def test_query_logs_by_level(db_session):
    """Test querying logs by level."""
    from bot.db.models import EventLog
    
    # Create logs with different levels
    logs = [
        EventLog(level="INFO", event_type="TEST", message="Info message"),
        EventLog(level="WARNING", event_type="TEST", message="Warning message"),
        EventLog(level="ERROR", event_type="TEST", message="Error message"),
        EventLog(level="INFO", event_type="TEST", message="Another info"),
    ]
    
    for log in logs:
        db_session.add(log)
    await db_session.commit()
    
    # Query only INFO logs
    from sqlalchemy import select
    result = await db_session.execute(
        select(EventLog).where(EventLog.level == "INFO")
    )
    info_logs = result.scalars().all()
    
    assert len(info_logs) == 2


@pytest.mark.asyncio
async def test_stats_to_dict(db_session):
    """Test Stats.to_dict() method."""
    from bot.db.models import Stats
    
    stat = Stats(
        key="test_key",
        value=99
    )
    
    db_session.add(stat)
    await db_session.commit()
    
    result = stat.to_dict()
    
    assert result["key"] == "test_key"
    assert result["value"] == 99
    assert "updated_at" in result