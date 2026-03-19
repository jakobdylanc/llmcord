"""Personas API endpoints for reading persona files."""

import logging
import os
import yaml
from typing import Optional, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["personas"])

# Path to personas directory (bot/config/personas/)
PERSONAS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "bot", "config", "personas")

# Path to tasks directory (bot/config/tasks/)
TASKS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "bot", "config", "tasks")

# Path to config.yaml
CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "config.yaml")


class PersonaUsageInfo(BaseModel):
    """Information about a task using this persona."""
    name: str
    filename: str
    type: str  # "task" or "model"
    schedule: Optional[str] = None


async def get_persona_usage_from_models(persona_name: str) -> list[dict[str, Any]]:
    """
    Scan config.yaml for models that use this persona.
    
    Returns list of models that reference this persona in their 'persona' field.
    """
    usage = []
    
    if not os.path.exists(CONFIG_FILE):
        return usage
    
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if not config or 'models' not in config:
            return usage
        
        for model_name, model_config in config.get('models', {}).items():
            if isinstance(model_config, dict):
                model_persona = model_config.get('persona', '')
                if model_persona == persona_name:
                    usage.append({
                        'name': model_name,
                        'filename': 'config.yaml',
                        'type': 'model',
                    })
                    logger.info(f"Persona '{persona_name}' is used by model '{model_name}'")
    except Exception as e:
        logger.warning(f"Error scanning config.yaml for persona usage: {e}")
    
    return usage


async def get_persona_usage(persona_name: str) -> list[dict[str, Any]]:
    """
    Scan all task YAML files for usage of a specific persona.
    
    Returns list of tasks that reference this persona in their 'persona' field.
    """
    usage = []
    
    if not os.path.exists(TASKS_DIR):
        return usage
    
    for filename in os.listdir(TASKS_DIR):
        if filename.endswith('.yaml') or filename.endswith('.yml'):
            filepath = os.path.join(TASKS_DIR, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    task_config = yaml.safe_load(f)
                
                if task_config and isinstance(task_config, dict):
                    # Check if task uses this persona
                    task_persona = task_config.get('persona', '')
                    if task_persona == persona_name:
                        task_name = task_config.get('name', filename[:-5])
                        schedule = task_config.get('schedule')
                        usage.append({
                            'name': task_name,
                            'filename': filename,
                            'type': 'task',
                            'schedule': schedule,
                        })
                        logger.info(f"Persona '{persona_name}' is used by task '{task_name}' ({filename})")
            except Exception as e:
                logger.warning(f"Error scanning task {filename} for persona usage: {e}")
    
    return usage


@router.get("/personas/{name}/usage", response_model=list[PersonaUsageInfo])
async def get_persona_usage_endpoint(name: str) -> list[PersonaUsageInfo]:
    """
    Get list of tasks and models that use this persona.
    
    - **name**: Persona name (without .md extension)
    
    Use this before deleting a persona to check if it's in use.
    Includes:
    - Tasks that reference this persona in their 'persona' field
    - Models in config.yaml that use this persona
    """
    # Validate persona exists
    filename = f"{name}.md"
    filepath = os.path.join(PERSONAS_DIR, filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"Persona '{name}' not found")
    
    # Get usage from tasks
    task_usage = await get_persona_usage(name)
    
    # Get usage from models (config.yaml)
    model_usage = await get_persona_usage_from_models(name)
    
    # Combine both
    usage = task_usage + model_usage
    
    return [
        PersonaUsageInfo(
            name=u['name'],
            filename=u['filename'],
            type=u.get('type', 'task'),
            schedule=u.get('schedule'),
        )
        for u in usage
    ]


class PersonaInfo(BaseModel):
    """Persona file info."""
    name: str
    filename: str
    description: Optional[str] = None


class PersonaDetail(BaseModel):
    """Full persona content."""
    name: str
    filename: str
    content: str
    word_count: int


def _get_persona_description(content: str) -> Optional[str]:
    """Extract description from persona content (first line or first paragraph)."""
    if not content:
        return None
    
    lines = content.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            # Return first non-heading, non-empty line as description
            return line[:100] if len(line) > 100 else line
    return None


@router.get("/personas", response_model=list[PersonaInfo])
async def get_personas() -> list[PersonaInfo]:
    """
    Get list of all available personas.
    
    Returns persona names and filenames (not content).
    """
    personas = []
    
    if not os.path.exists(PERSONAS_DIR):
        logger.warning(f"Personas directory not found: {PERSONAS_DIR}")
        return []
    
    for filename in os.listdir(PERSONAS_DIR):
        if filename.endswith('.md'):
            filepath = os.path.join(PERSONAS_DIR, filename)
            
            # Read first part of file for description
            description = None
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read(500)  # Read first 500 chars
                    description = _get_persona_description(content)
            except Exception as e:
                logger.warning(f"Error reading persona {filename}: {e}")
            
            # Name from filename (without .md extension)
            name = filename[:-3]
            
            personas.append(PersonaInfo(
                name=name,
                filename=filename,
                description=description,
            ))
    
    # Sort by name
    personas.sort(key=lambda p: p.name)
    
    return personas


@router.get("/personas/{name}", response_model=PersonaDetail)
async def get_persona(name: str) -> PersonaDetail:
    """
    Get full persona content by name.
    
    - **name**: Persona name (without .md extension)
    """
    filename = f"{name}.md"
    filepath = os.path.join(PERSONAS_DIR, filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"Persona '{name}' not found")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return PersonaDetail(
            name=name,
            filename=filename,
            content=content,
            word_count=len(content.split()),
        )
    except Exception as e:
        logger.error(f"Error reading persona {name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading persona: {str(e)}")


class PersonaCreate(BaseModel):
    """Request to create a new persona."""
    name: str
    content: str


class PersonaUpdate(BaseModel):
    """Request to update an existing persona."""
    content: str


@router.post("/personas", status_code=201)
async def create_persona(data: PersonaCreate) -> PersonaDetail:
    """
    Create a new persona file.
    
    - **name**: Persona name (without .md extension)
    - **content**: Markdown content for the persona
    """
    # Validate name
    if not data.name or not data.name.strip():
        raise HTTPException(status_code=400, detail="Persona name is required")
    
    # Sanitize name - only allow alphanumeric, dash, underscore
    import re
    if not re.match(r'^[a-zA-Z0-9_-]+$', data.name):
        raise HTTPException(status_code=400, detail="Persona name can only contain letters, numbers, dash, and underscore")
    
    filename = f"{data.name}.md"
    filepath = os.path.join(PERSONAS_DIR, filename)
    
    if os.path.exists(filepath):
        raise HTTPException(status_code=409, detail=f"Persona '{data.name}' already exists")
    
    # Ensure directory exists
    os.makedirs(PERSONAS_DIR, exist_ok=True)
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(data.content)
        
        logger.info(f"Created persona: {filename}")
        
        return PersonaDetail(
            name=data.name,
            filename=filename,
            content=data.content,
            word_count=len(data.content.split()),
        )
    except Exception as e:
        logger.error(f"Error creating persona {data.name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating persona: {str(e)}")


@router.put("/personas/{name}", response_model=PersonaDetail)
async def update_persona(name: str, data: PersonaUpdate) -> PersonaDetail:
    """
    Update an existing persona file.
    
    - **name**: Persona name (without .md extension)
    - **content**: New markdown content
    """
    filename = f"{name}.md"
    filepath = os.path.join(PERSONAS_DIR, filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"Persona '{name}' not found")
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(data.content)
        
        logger.info(f"Updated persona: {filename}")
        
        return PersonaDetail(
            name=name,
            filename=filename,
            content=data.content,
            word_count=len(data.content.split()),
        )
    except Exception as e:
        logger.error(f"Error updating persona {name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating persona: {str(e)}")


@router.delete("/personas/{name}", status_code=204)
async def delete_persona(name: str):
    """
    Delete a persona file.
    
    - **name**: Persona name (without .md extension)
    
    NOTE: Use GET /api/personas/{name}/usage first to check if persona is used by any tasks.
    """
    filename = f"{name}.md"
    filepath = os.path.join(PERSONAS_DIR, filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"Persona '{name}' not found")
    
    # Prevent deletion of example files
    if name.endswith('-example'):
        raise HTTPException(status_code=403, detail="Cannot delete example personas")
    
    # Check if persona is used by any tasks OR models (21.5.4)
    task_usage = await get_persona_usage(name)
    model_usage = await get_persona_usage_from_models(name)
    all_usage = task_usage + model_usage
    
    if all_usage:
        task_names = [u['name'] for u in task_usage]
        model_names = [u['name'] for u in model_usage]
        
        msg_parts = []
        if task_names:
            msg_parts.append(f"{len(task_names)} task(s): {', '.join(task_names)}")
        if model_names:
            msg_parts.append(f"{len(model_names)} model(s): {', '.join(model_names)}")
        
        raise HTTPException(
            status_code=409, 
            detail=f"Cannot delete persona '{name}': it is used by {', '.join(msg_parts)}"
        )
    
    try:
        os.remove(filepath)
        logger.info(f"Deleted persona: {filename}")
    except Exception as e:
        logger.error(f"Error deleting persona {name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting persona: {str(e)}")