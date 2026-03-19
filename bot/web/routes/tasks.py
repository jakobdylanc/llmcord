"""Tasks API endpoints for reading task YAML files."""

import asyncio
import logging
import os
import sys
from datetime import datetime
from typing import Optional, Any

import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Ensure the project root (/app) is in sys.path for imports
# File: /app/bot/web/routes/tasks.py -> need /app (4 levels up)
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["tasks"])

# Path to tasks directory
# Use _project_root (already correctly calculated 4 levels up to project root)
# TASKS_DIR should point to bot/config/tasks (same as bot/config/tasks.py)
TASKS_DIR = os.path.join(_project_root, "bot", "config", "tasks")

# Global scheduler reference for reloading tasks
_scheduler_ref: Optional[Any] = None

# Job status tracking for manual task executions
# job_id -> {status: str, task_name: str, started_at: str, completed_at: Optional[str], result: Optional[str], error: Optional[str]}
_job_status: dict = {}


def set_scheduler_ref(scheduler):
    """Set reference to the APScheduler for task reload functionality."""
    global _scheduler_ref
    _scheduler_ref = scheduler
    _setup_job_listeners()


def _setup_job_listeners():
    """Setup APScheduler listeners for job execution tracking."""
    global _job_status
    
    try:
        from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR
        
        def on_job_executed(event):
            job_id = event.job_id
            if job_id in _job_status:
                _job_status[job_id]['status'] = 'completed'
                _job_status[job_id]['completed_at'] = datetime.now().isoformat()
                if event.exception:
                    _job_status[job_id]['error'] = str(event.exception)
                logger.info(f"Job completed: {job_id}")
        
        def on_job_error(event):
            job_id = event.job_id
            if job_id in _job_status:
                _job_status[job_id]['status'] = 'failed'
                _job_status[job_id]['completed_at'] = datetime.now().isoformat()
                _job_status[job_id]['error'] = str(event.exception) if event.exception else 'Unknown error'
                logger.error(f"Job failed: {job_id} - {event.exception}")
        
        # Add listeners if scheduler is available
        if _scheduler_ref:
            _scheduler_ref.add_listener(on_job_executed, EVENT_JOB_EXECUTED)
            _scheduler_ref.add_listener(on_job_error, EVENT_JOB_ERROR)
            logger.info("Job execution listeners registered")
    except ImportError:
        logger.warning("APScheduler events not available for job tracking")


class TaskInfo(BaseModel):
    """Task file info."""
    name: str
    filename: str
    enabled: Optional[bool] = None
    schedule: Optional[str] = None
    description: Optional[str] = None
    status: str = "unknown"  # scheduled, pending, running, disabled


class TaskDetail(BaseModel):
    """Full task config."""
    name: str
    filename: str
    config: dict
    status: str = "unknown"


class TaskCreate(BaseModel):
    """Request to create a new task."""
    name: str
    config: dict


class TaskUpdate(BaseModel):
    """Request to update an existing task."""
    config: dict


def _determine_task_status(task_name: str, task_config: dict) -> str:
    """Determine task status based on config and scheduler."""
    # If not enabled, it's disabled
    if not task_config.get('enabled', True):
        return "disabled"
    
    # Check if there's a schedule
    schedule = task_config.get('schedule') or task_config.get('cron')
    if schedule:
        return "scheduled"
    
    return "pending"


@router.get("/tasks", response_model=list[TaskInfo])
async def get_tasks() -> list[TaskInfo]:
    """
    Get list of all available scheduled tasks.
    
    Returns task names and basic info including full content if requested.
    """
    tasks = []
    
    if not os.path.exists(TASKS_DIR):
        logger.warning(f"Tasks directory not found: {TASKS_DIR}")
        return []
    
    for filename in os.listdir(TASKS_DIR):
        if filename.endswith('.yaml') or filename.endswith('.yml'):
            filepath = os.path.join(TASKS_DIR, filename)
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    task_config = yaml.safe_load(f) or {}
                
                # Extract info
                name = filename.rsplit('.', 1)[0]  # Remove extension
                
                # Get schedule and description from config
                schedule = task_config.get('schedule') or task_config.get('cron')
                description = task_config.get('description') or task_config.get('name')
                enabled = task_config.get('enabled', True)
                
                # Determine status
                status = _determine_task_status(name, task_config)
                
                tasks.append(TaskInfo(
                    name=name,
                    filename=filename,
                    enabled=enabled,
                    schedule=schedule,
                    description=str(description) if description else None,
                    status=status,
                ))
            except Exception as e:
                logger.warning(f"Error reading task {filename}: {e}")
                # Still add the task with basic info
                name = filename.rsplit('.', 1)[0]
                tasks.append(TaskInfo(
                    name=name,
                    filename=filename,
                    enabled=None,
                    schedule=None,
                    description=None,
                    status="error",
                ))
    
    # Sort by name
    tasks.sort(key=lambda t: t.name)
    
    return tasks


@router.get("/tasks/{name}", response_model=TaskDetail)
async def get_task(name: str) -> TaskDetail:
    """
    Get full task configuration by name.
    
    - **name**: Task name (without .yaml/.yml extension)
    """
    # Try both .yaml and .yml extensions
    for ext in ['.yaml', '.yml']:
        filename = f"{name}{ext}"
        filepath = os.path.join(TASKS_DIR, filename)
        if os.path.exists(filepath):
            break
    else:
        raise HTTPException(status_code=404, detail=f"Task '{name}' not found")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            task_config = yaml.safe_load(f) or {}
        
        return TaskDetail(
            name=name,
            filename=os.path.basename(filepath),
            config=task_config,
            status=_determine_task_status(name, task_config),
        )
    except Exception as e:
        logger.error(f"Error reading task {name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading task: {str(e)}")


@router.put("/tasks/{name}")
async def update_task(name: str, data: TaskUpdate) -> TaskDetail:
    """
    Update task configuration.
    
    - **name**: Task name (without .yaml/.yml extension)
    - **data**: Task update data (config)
    """
    config = data.config
    
    # Try both .yaml and .yml extensions
    for ext in ['.yaml', '.yml']:
        filename = f"{name}{ext}"
        filepath = os.path.join(TASKS_DIR, filename)
        if os.path.exists(filepath):
            break
    else:
        raise HTTPException(status_code=404, detail=f"Task '{name}' not found")
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
        
        return TaskDetail(
            name=name,
            filename=os.path.basename(filepath),
            config=config,
            status=_determine_task_status(name, config),
        )
    except Exception as e:
        logger.error(f"Error writing task {name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error writing task: {str(e)}")


@router.delete("/tasks/{name}")
async def delete_task(name: str) -> dict:
    """
    Delete a task file.
    
    - **name**: Task name (without .yaml/.yml extension)
    """
    # Try both .yaml and .yml extensions
    for ext in ['.yaml', '.yml']:
        filename = f"{name}{ext}"
        filepath = os.path.join(TASKS_DIR, filename)
        if os.path.exists(filepath):
            break
    else:
        raise HTTPException(status_code=404, detail=f"Task '{name}' not found")
    
    try:
        os.remove(filepath)
        return {"success": True, "message": f"Task '{name}' deleted"}
    except Exception as e:
        logger.error(f"Error deleting task {name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting task: {str(e)}")


@router.post("/tasks", status_code=201)
async def create_task(data: TaskCreate) -> TaskDetail:
    """
    Create a new task.
    
    - **data**: Task creation data (name and config)
    """
    name = data.name
    config = data.config
    
    filename = f"{name}.yaml"
    filepath = os.path.join(TASKS_DIR, filename)
    
    if os.path.exists(filepath):
        raise HTTPException(status_code=409, detail=f"Task '{name}' already exists")
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
        
        return TaskDetail(
            name=name,
            filename=filename,
            config=config,
            status=_determine_task_status(name, config),
        )
    except Exception as e:
        logger.error(f"Error creating task {name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating task: {str(e)}")


@router.post("/tasks/reload")
async def reload_tasks() -> dict:
    """
    Reload scheduled tasks from disk and update the scheduler.
    This endpoint triggers a reload of all task configurations.
    """
    global _scheduler_ref
    
    if _scheduler_ref is None:
        raise HTTPException(status_code=500, detail="Scheduler not available")
    
    try:
        # Import here to avoid circular imports
        from bot.config.loader import get_config
        from bot.config.tasks import load_scheduled_tasks
        
        # Reload config
        config = get_config()
        
        # Reload tasks from disk
        tasks = load_scheduled_tasks(config)
        
        # Get current jobs from scheduler
        current_jobs = {job.id.replace("scheduled_task_", ""): job 
                       for job in _scheduler_ref.get_jobs() 
                       if job.id.startswith("scheduled_task_")}
        
        # Remove jobs that no longer exist
        for task_name in list(current_jobs.keys()):
            if task_name not in tasks:
                _scheduler_ref.remove_job(f"scheduled_task_{task_name}")
                logger.info(f"Removed scheduled task: {task_name}")
        
        # Add or update jobs
        for task_name, task_config in tasks.items():
            job_id = f"scheduled_task_{task_name}"
            
            # Remove existing job if present
            if job_id in current_jobs:
                _scheduler_ref.remove_job(job_id)
            
            # Add new job if enabled
            if task_config.get('enabled', True):
                cron = task_config.get('cron') or task_config.get('schedule')
                if cron:
                    from llmcord import run_scheduled_task
                    _scheduler_ref.add_job(
                        run_scheduled_task,
                        'cron',
                        cron=cron,
                        id=job_id,
                        replace_existing=True,
                        args=[task_name, task_config]
                    )
                    logger.info(f"Reloaded scheduled task: {task_name}")
        
        return {"success": True, "message": f"Reloaded {len(tasks)} tasks"}
    except Exception as e:
        logger.error(f"Error reloading tasks: {e}")
        raise HTTPException(status_code=500, detail=f"Error reloading tasks: {str(e)}")


@router.post("/tasks/{name}/reload")
async def reload_single_task(name: str) -> dict:
    """
    Reload a single task from disk and update the scheduler.
    This only reloads the specified task, not all tasks - avoiding disruption.
    
    - **name**: Task name (without .yaml/.yml extension). Can be either:
      - Filename (e.g., "stock-market-checker")
      - Internal name field (e.g., "stock_market_check")
    """
    global _scheduler_ref
    
    if _scheduler_ref is None:
        raise HTTPException(status_code=500, detail="Scheduler not available")
    
    try:
        # Import here to avoid circular imports
        from bot.config.loader import get_config
        from bot.config.tasks import load_scheduled_tasks
        
        # Reload config
        config = get_config()
        
        # Reload tasks from disk
        tasks = load_scheduled_tasks(config)
        
        # Find task by name - try both filename and internal name field
        # This handles the case where filename differs from the internal name field
        # 
        # IMPORTANT: load_scheduled_tasks stores tasks by INTERNAL name (data.get('name')), 
        # not by filename (path.stem). So dict keys are like "stock_market_check", not "stock-market-checker".
        task_name = None
        task_config = None
        
        # First, try direct lookup by name (dict key - this is the internal name)
        if name in tasks:
            task_name = name
            task_config = tasks[name]
        else:
            # Search through all tasks - try matching both:
            # 1. The dict key (task_key) - handles filename lookup
            # 2. The internal name field (task_cfg.get('name')) - handles internal name lookup
            for task_key, task_cfg in tasks.items():
                if task_key == name or task_cfg.get('name') == name:
                    task_name = task_key
                    task_config = task_cfg
                    break
        
        # Check if task exists
        if task_config is None:
            raise HTTPException(status_code=404, detail=f"Task '{name}' not found")
        
        job_id = f"scheduled_task_{task_name}"
        
        # Remove existing job if present
        try:
            _scheduler_ref.remove_job(job_id)
        except Exception:
            pass  # Job might not exist
        
        # Add new job if enabled
        if task_config.get('enabled', True):
            cron = task_config.get('cron') or task_config.get('schedule')
            if cron:
                from llmcord import run_scheduled_task
                _scheduler_ref.add_job(
                    run_scheduled_task,
                    'cron',
                    cron=cron,
                    id=job_id,
                    replace_existing=True,
                    args=[task_name, task_config]
                )
                logger.info(f"Reloaded single task: {task_name}")
                return {"success": True, "message": f"Task '{task_name}' reloaded"}
            else:
                # Task exists but has no cron - it's in "pending" state
                return {"success": True, "message": f"Task '{task_name}' updated (no schedule, pending)"}
        else:
            return {"success": True, "message": f"Task '{task_name}' disabled"}
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reloading task {name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error reloading task: {str(e)}")


@router.post("/tasks/{name}/run")
async def run_task(name: str) -> dict:
    """
    Manually trigger a task to run immediately (async - doesn't wait for completion).
    
    - **name**: Task name (without .yaml/.yml extension). Can be either:
      - Filename (e.g., "stock-market-checker")
      - Internal name field (e.g., "stock_market_check")
    
    Returns immediately after triggering - task runs in background.
    """
    global _scheduler_ref
    
    if _scheduler_ref is None:
        raise HTTPException(status_code=500, detail="Scheduler not available")
    
    try:
        # Import here to avoid circular imports
        from bot.config.loader import get_config
        from bot.config.tasks import load_scheduled_tasks
        
        # Reload config
        config = get_config()
        
        # Load tasks from disk
        tasks = load_scheduled_tasks(config)
        
        # Find task by name - try both filename and internal name field
        # This handles the case where filename differs from the internal name field
        # e.g., "stock-market-checker.yaml" has name: "stock_market_check"
        # 
        # IMPORTANT: load_scheduled_tasks stores tasks by INTERNAL name (data.get('name')), 
        # not by filename (path.stem). So dict keys are like "stock_market_check", not "stock-market-checker".
        task_name = None
        task_config = None
        
        # Normalize the name parameter (replace dashes with underscores for comparison)
        name_normalized = name.replace('-', '_')
        
        # First, try direct lookup by name (dict key - this is the internal name)
        if name in tasks:
            task_name = name
            task_config = tasks[name]
        elif name_normalized in tasks:
            task_name = name_normalized
            task_config = tasks[name_normalized]
        else:
            # Search through all tasks - try matching both:
            # 1. The dict key (task_key) - handles internal name lookup
            # 2. The internal name field (task_cfg.get('name')) - handles alternate name
            # 3. Also compare with normalized versions (dash <-> underscore)
            for task_key, task_cfg in tasks.items():
                task_key_normalized = task_key.replace('-', '_')
                internal_name = task_cfg.get('name', '')
                internal_name_normalized = internal_name.replace('-', '_') if internal_name else ''
                
                if task_key == name or task_key_normalized == name_normalized:
                    task_name = task_key
                    task_config = task_cfg
                    break
                elif internal_name == name or internal_name_normalized == name_normalized:
                    task_name = task_key
                    task_config = task_cfg
                    break
            
            # If still not found, search by FILENAME on disk (path.stem)
            # This handles the case where filename "stock-market-checker" != internal name "stock_market_check"
            if task_config is None:
                from pathlib import Path
                # Go from /app/bot/web/routes/tasks.py -> /app/bot/config/tasks
                TASKS_DIR = Path(__file__).parent.parent.parent / "config" / "tasks"
                for filepath in TASKS_DIR.glob("*.yaml"):
                    if filepath.stem == name or filepath.stem.replace('-', '_') == name_normalized:
                        # Found matching filename - now find corresponding task in loaded tasks
                        file_data = yaml.safe_load(filepath.read_text(encoding='utf-8')) or {}
                        internal_file_name = str(file_data.get('name') or filepath.stem)
                        
                        if internal_file_name in tasks:
                            task_name = internal_file_name
                            task_config = tasks[internal_file_name]
                            break
        
        # Check if task exists
        if task_config is None:
            raise HTTPException(status_code=404, detail=f"Task '{name}' not found")
        
        # Check if task is enabled
        if not task_config.get('enabled', True):
            return {"success": False, "message": f"Task '{task_name}' is disabled"}
        
        # Trigger the task asynchronously using add_job with 'date' trigger
        from llmcord import run_scheduled_task
        job_id = f"manual_task_{task_name}_{int(asyncio.get_event_loop().time())}"
        
        # Track job status
        _job_status[job_id] = {
            'status': 'queued',
            'task_name': task_name,
            'started_at': datetime.now().isoformat(),
            'completed_at': None,
            'result': None,
            'error': None
        }
        
        _scheduler_ref.add_job(
            run_scheduled_task,
            'date',
            run_date=datetime.now(),
            id=job_id,
            args=[task_name, task_config],
            replace_existing=False
        )
        
        logger.info(f"Manually triggered task: {task_name} with job_id: {job_id}")
        return {"success": True, "message": f"Task '{task_name}' queued for execution", "job_id": job_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running task {name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error running task: {str(e)}")


@router.get("/tasks/{name}/status")
async def get_task_execution_status(name: str) -> dict:
    """
    Get the execution status of a manually triggered task.
    
    - **name**: Task name (without .yaml/.yml extension)
    
    Returns the latest job execution status for this task.
    """
    global _job_status
    
    # Find the most recent job for this task
    # Try both original name and normalized name
    name_normalized = name.replace('-', '_')
    
    matching_jobs = []
    for job_id, status in _job_status.items():
        task_name = status.get('task_name', '')
        if task_name == name or task_name == name_normalized:
            matching_jobs.append((job_id, status))
    
    if not matching_jobs:
        return {
            "status": "not_found",
            "message": f"No execution found for task '{name}'"
        }
    
    # Sort by started_at descending and return the most recent
    matching_jobs.sort(key=lambda x: x[1].get('started_at', ''), reverse=True)
    job_id, status = matching_jobs[0]
    
    return {
        "job_id": job_id,
        "status": status.get('status', 'unknown'),
        "task_name": status.get('task_name'),
        "started_at": status.get('started_at'),
        "completed_at": status.get('completed_at'),
        "result": status.get('result'),
        "error": status.get('error')
    }