"""FastAPI web server for llmcord portal."""

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from bot.db.connection import close_db, init_db, get_db_dependency
from bot.web.config import get_portal_config

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan events for the web server."""
    # Startup
    logger.info("Starting web portal server...")
    
    # Initialize database
    await init_db()
    logger.info("Database initialized")
    
    # Check if portal is enabled
    config = get_portal_config()
    if not config.enabled:
        logger.warning("Portal is disabled in config")
    else:
        logger.info(f"Portal enabled on port {config.port}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down web portal server...")
    await close_db()


# Create FastAPI app
app = FastAPI(
    title="GPT Discord Bot",
    description="Web administration portal for llmcord Discord bot",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
config = get_portal_config()
cors_origins = config.cors_origins if config.cors_origins else []
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,  # Empty = same origin only (secure)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def run_web_server() -> None:
    """Run the web server (blocking)."""
    import uvicorn

    config = get_portal_config()
    if not config.enabled:
        logger.info("Portal disabled, skipping web server start")
        return

    logger.info(f"Starting web server on port {config.port}")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=config.port,
        log_level="info",
    )


# Import routes after app creation to avoid circular imports
from bot.web.auth import (
    setup_portal,
    login,
    get_users,
    check_has_users,
    SetupRequest,
    LoginRequest,
    Token,
    CurrentUser,
    get_current_user,
)


# Health check endpoint for container orchestration
@app.get("/health")
async def health_check():
    """Health check endpoint for Docker/container health checks."""
    return {"status": "healthy", "service": "gpt-discord-bot-portal"}


# Request logging middleware (21.1.2)
from fastapi import Request
import time


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests with timing information."""
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Calculate duration
    duration = time.time() - start_time
    
    # Log the request
    logger.info(
        f"{request.method} {request.url.path} "
        f"status={response.status_code} duration={duration:.3f}s"
    )
    
    return response


# Auth routes
@app.post("/api/auth/setup", response_model=Token)
async def api_setup(request: SetupRequest, db: AsyncSession = Depends(get_db_dependency)):
    """First-time setup - create initial admin user."""
    return await setup_portal(request, db)


@app.get("/api/auth/has-users")
async def api_has_users(db: AsyncSession = Depends(get_db_dependency)):
    """Check if any users exist (for showing setup wizard vs login)."""
    has_users = await check_has_users(db)
    return {"has_users": has_users}


@app.post("/api/auth/login", response_model=Token)
async def api_login(request: LoginRequest, db: AsyncSession = Depends(get_db_dependency)):
    """Login with username and password."""
    return await login(request, db)


@app.get("/api/auth/users", response_model=list[dict])
async def api_get_users(
    db: AsyncSession = Depends(get_db_dependency),
    current_user: CurrentUser = Depends(get_current_user),
):
    """Get list of users (authenticated only)."""
    return await get_users(db, current_user)


# Status & Servers routes
from bot.web.routes.status import router as status_router
from bot.web.routes.servers import router as servers_router
from bot.web.routes.logs import router as logs_router
from bot.web.routes.config import router as config_router
from bot.web.routes.personas import router as personas_router
from bot.web.routes.tasks import router as tasks_router
from bot.web.routes.skills import router as skills_router

app.include_router(status_router)
app.include_router(servers_router)
app.include_router(logs_router)
app.include_router(config_router)
app.include_router(personas_router)
app.include_router(tasks_router)
app.include_router(skills_router)


# WebSocket endpoint for real-time logs
from fastapi import WebSocket, WebSocketDisconnect
from bot.web.log_handler import add_log_client, remove_log_client, get_log_client_count


@app.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    """
    WebSocket endpoint for real-time log streaming.
    
    Clients connect to receive log events as they occur.
    The logs are broadcast from the DatabaseLogHandler.
    """
    await websocket.accept()
    add_log_client(websocket)
    
    logger.info(f"WebSocket client connected. Total clients: {get_log_client_count()}")
    
    try:
        # Keep connection alive and handle incoming messages
        while True:
            # Wait for any message from client (ping/pong mechanism)
            data = await websocket.receive_text()
            
            # Echo back for connection health check
            if data == "ping":
                await websocket.send_text("pong")
                
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        remove_log_client(websocket)
        logger.info(f"WebSocket client removed. Total clients: {get_log_client_count()}")


@app.get("/ws/status")
async def websocket_status():
    """Get WebSocket connection status."""
    return {
        "log_clients": get_log_client_count(),
        "status": "connected" if get_log_client_count() > 0 else "no_clients",
    }


# Serve React frontend static files
# Determine the path to the web/dist folder
import pathlib

def get_frontend_dist_path() -> pathlib.Path:
    """Get the path to the frontend dist folder."""
    # Go up from bot/web/ to project root, then into web/dist
    project_root = pathlib.Path(__file__).parent.parent.parent
    return project_root / "web" / "dist"


frontend_dist = get_frontend_dist_path()

if frontend_dist.exists():
    # Mount all static files from dist root
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="static")
    
    # Catch-all route for SPA - only for non-API routes
    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Serve SPA for any non-API route."""
        # Skip if it's an API route
        if full_path.startswith("api/"):
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Not Found")
        
        # Serve index.html for SPA routing
        from fastapi.responses import FileResponse
        index_path = frontend_dist / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        return {"detail": "Not found"}
    
    logger.info(f"Serving React frontend from {frontend_dist}")
else:
    logger.warning(f"Frontend dist folder not found at {frontend_dist}")