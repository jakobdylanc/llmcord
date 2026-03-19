"""Logs API endpoints."""

import logging
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from bot.db.models import EventLog
from bot.db.connection import get_db_dependency
from bot.web.auth import get_current_user
from bot.web.config import get_portal_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["logs"])


# Response models
class LogEntry(BaseModel):
    """Single log entry."""
    id: int
    timestamp: str
    level: str
    event_type: str
    message: str
    metadata: Optional[dict] = None

    model_config = {"from_attributes": True}


class LogsResponse(BaseModel):
    """Response for logs query."""
    logs: list[LogEntry]
    total: int
    page: int
    page_size: int


# Query parameters model
class LogQueryParams:
    """Query parameters for log filtering."""
    
    def __init__(
        self,
        level: Optional[str] = Query(None, description="Filter by log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"),
        event_type: Optional[str] = Query(None, description="Filter by event type"),
        since: Optional[str] = Query(None, description="ISO format timestamp to filter logs since"),
        until: Optional[str] = Query(None, description="ISO format timestamp to filter logs until"),
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(50, ge=1, le=500, description="Items per page"),
    ):
        self.level = level
        self.event_type = event_type
        self.since = since
        self.until = until
        self.page = page
        self.page_size = page_size


@router.get("/logs", response_model=LogsResponse)
async def get_logs(
    db: AsyncSession = Depends(get_db_dependency),
    level: Optional[str] = Query(None, description="Filter by log level"),
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    since: Optional[str] = Query(None, description="ISO timestamp to filter logs since"),
    until: Optional[str] = Query(None, description="ISO timestamp to filter logs until"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=500, description="Items per page"),
) -> LogsResponse:
    """
    Get event logs with optional filtering.
    
    - **level**: Filter by log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - **event_type**: Filter by event type
    - **since**: ISO format timestamp - only return logs after this time
    - **until**: ISO format timestamp - only return logs before this time
    - **page**: Page number (1-indexed)
    - **page_size**: Number of items per page (max 500)
    """
    # Get config for allowed log levels
    config = get_portal_config()
    allowed_levels = config.logs_levels
    
    # Build filters
    filters = []
    
    # Level filter - only allow levels configured in portal
    if level:
        level_upper = level.upper()
        if level_upper in allowed_levels:
            filters.append(EventLog.level == level_upper)
    else:
        # If no level specified, show only allowed levels
        filters.append(EventLog.level.in_(allowed_levels))
    
    # Event type filter
    if event_type:
        filters.append(EventLog.event_type == event_type)
    
    # Time range filters
    if since:
        try:
            since_dt = datetime.fromisoformat(since.replace('Z', '+00:00'))
            filters.append(EventLog.timestamp >= since_dt)
        except ValueError:
            logger.warning(f"Invalid 'since' timestamp format: {since}")
    
    if until:
        try:
            until_dt = datetime.fromisoformat(until.replace('Z', '+00:00'))
            filters.append(EventLog.timestamp <= until_dt)
        except ValueError:
            logger.warning(f"Invalid 'until' timestamp format: {until}")
    
    # Build query with filters
    query = select(EventLog)
    if filters:
        query = query.where(and_(*filters))
    
    # Order by timestamp descending (newest first)
    query = query.order_by(EventLog.timestamp.desc())
    
    # Get total count
    from sqlalchemy import func
    if filters:
        count_result = await db.execute(
            select(func.count()).where(and_(*filters))
        )
    else:
        count_result = await db.execute(
            select(func.count()).select_from(EventLog)
        )
    total = count_result.scalar() or 0
    
    # Apply pagination
    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size)
    
    # Execute query
    result = await db.execute(query)
    logs = result.scalars().all()
    
    # Convert to response format
    log_entries = [
        LogEntry(
            id=log.id,
            timestamp=log.timestamp.isoformat() if log.timestamp else "",
            level=log.level,
            event_type=log.event_type,
            message=log.message,
            metadata=log.extra_data,
        )
        for log in logs
    ]
    
    return LogsResponse(
        logs=log_entries,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/logs/levels", response_model=list[str])
async def get_log_levels() -> list[str]:
    """Get available log levels based on portal configuration."""
    config = get_portal_config()
    return config.logs_levels


@router.get("/logs/types", response_model=list[str])
async def get_log_types(db: AsyncSession = Depends(get_db_dependency)) -> list[str]:
    """Get list of unique event types in the logs."""
    result = await db.execute(
        select(EventLog.event_type).distinct().order_by(EventLog.event_type)
    )
    types = result.scalars().all()
    return list(types)