"""SQLAlchemy models for web portal database."""

from datetime import datetime
from typing import Optional
from sqlalchemy import String, Integer, DateTime, Text, JSON, Index
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


class EventLog(Base):
    """Event log stored in database."""

    __tablename__ = "event_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False)
    level: Mapped[str] = mapped_column(String(20), nullable=False, default="INFO")
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    extra_data: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)

    __table_args__ = (
        Index("ix_event_logs_timestamp", "timestamp"),
        Index("ix_event_logs_level", "level"),
        Index("ix_event_logs_event_type", "event_type"),
    )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "level": self.level,
            "event_type": self.event_type,
            "message": self.message,
            "metadata": self.extra_data,
        }


class Stats(Base):
    """Key-value stats storage."""

    __tablename__ = "stats"

    key: Mapped[str] = mapped_column(String(100), primary_key=True)
    value: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "value": self.value,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class User(Base):
    """Portal user for authentication."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary (excludes password_hash)."""
        return {
            "id": self.id,
            "username": self.username,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }