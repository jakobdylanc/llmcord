"""Unit tests for bot/web/routes/personas.py, tasks.py, skills.py."""

import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock


class TestPersonasEndpoint:
    """Test personas API endpoints."""

    @pytest.mark.asyncio
    async def test_get_personas_returns_list(self):
        """Test that get_personas returns a list."""
        from bot.web.routes.personas import get_personas, PersonaInfo
        
        with patch('bot.web.routes.personas.PERSONAS_DIR', tempfile.gettempdir()):
            with patch('os.listdir', return_value=[]):
                result = await get_personas()
        
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_get_personas_with_files(self):
        """Test get_personas with persona files."""
        from bot.web.routes.personas import get_personas
        
        mock_persona_content = """# Bao Persona

This is a test persona for the bot.
"""
        
        with patch('bot.web.routes.personas.PERSONAS_DIR', tempfile.gettempdir()):
            with patch('os.listdir', return_value=['bao.md']):
                with patch('os.path.exists', return_value=True):
                    with patch('builtins.open', MagicMock()):
                        with patch('os.path.join', return_value='temp/bao.md'):
                            result = await get_personas()
        
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_get_persona_not_found(self):
        """Test get_persona raises 404 for unknown persona."""
        from bot.web.routes.personas import get_persona
        from fastapi import HTTPException
        
        with patch('bot.web.routes.personas.PERSONAS_DIR', tempfile.gettempdir()):
            with patch('os.path.exists', return_value=False):
                with pytest.raises(HTTPException) as exc_info:
                    await get_persona("nonexistent")
        
        assert exc_info.value.status_code == 404

    def test_persona_description_extraction(self):
        """Test description extraction from persona content."""
        from bot.web.routes.personas import _get_persona_description
        
        # Test with content
        content = """# My Persona

This is the persona description.
More content here.
"""
        desc = _get_persona_description(content)
        assert desc == "This is the persona description."
        
        # Test with empty content
        assert _get_persona_description("") is None
        assert _get_persona_description(None) is None


class TestTasksEndpoint:
    """Test tasks API endpoints."""

    @pytest.mark.asyncio
    async def test_get_tasks_returns_list(self):
        """Test that get_tasks returns a list."""
        from bot.web.routes.tasks import get_tasks
        
        with patch('bot.web.routes.tasks.TASKS_DIR', tempfile.gettempdir()):
            with patch('os.listdir', return_value=[]):
                result = await get_tasks()
        
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_get_tasks_with_files(self):
        """Test get_tasks with task files."""
        from bot.web.routes.tasks import get_tasks
        
        mock_yaml = """
name: Test Task
schedule: "0 * * * *"
enabled: true
description: A test task
"""
        
        with patch('bot.web.routes.tasks.TASKS_DIR', tempfile.gettempdir()):
            with patch('os.listdir', return_value=['test-task.yaml']):
                with patch('builtins.open', MagicMock()):
                    with patch('yaml.safe_load', return_value={"name": "Test Task", "schedule": "0 * * * *", "enabled": True}):
                        result = await get_tasks()
        
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_get_task_not_found(self):
        """Test get_task raises 404 for unknown task."""
        from bot.web.routes.tasks import get_task
        from fastapi import HTTPException
        
        with patch('bot.web.routes.tasks.TASKS_DIR', tempfile.gettempdir()):
            with patch('os.path.exists', return_value=False):
                with pytest.raises(HTTPException) as exc_info:
                    await get_task("nonexistent")
        
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_reload_tasks_without_scheduler(self):
        """Test reload_tasks raises 500 when scheduler not available."""
        from bot.web.routes.tasks import reload_tasks
        from fastapi import HTTPException
        
        with patch('bot.web.routes.tasks._scheduler_ref', None):
            with pytest.raises(HTTPException) as exc_info:
                await reload_tasks()
        
        assert exc_info.value.status_code == 500
        assert "Scheduler not available" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_reload_single_task_not_found(self):
        """Test reload_single_task raises 404 for unknown task."""
        from bot.web.routes.tasks import reload_single_task
        from fastapi import HTTPException
        from unittest.mock import MagicMock
        
        mock_scheduler = MagicMock()
        mock_scheduler.get_jobs.return_value = []
        
        # Mock load_scheduled_tasks to return empty dict - import from correct location
        with patch('bot.web.routes.tasks._scheduler_ref', mock_scheduler):
            with patch('bot.config.tasks.load_scheduled_tasks', return_value={}):
                with pytest.raises(HTTPException) as exc_info:
                    await reload_single_task("nonexistent")
        
        assert exc_info.value.status_code == 404
        assert "not found" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_reload_single_task_success(self):
        """Test reload_single_task successfully reloads a task."""
        from bot.web.routes.tasks import reload_single_task
        from unittest.mock import MagicMock, patch, sys
        
        mock_scheduler = MagicMock()
        mock_scheduler.get_jobs.return_value = []
        
        # Create a mock run_scheduled_task function with proper name attribute
        mock_run_scheduled_task = MagicMock()
        mock_run_scheduled_task.__name__ = "run_scheduled_task"
        
        # Add mock to sys.modules to intercept the import
        mock_llmcord_module = MagicMock()
        mock_llmcord_module.run_scheduled_task = mock_run_scheduled_task
        
        original_modules = sys.modules.copy()
        sys.modules['bot.llmcord'] = mock_llmcord_module
        
        task_config = {
            "name": "test-task",
            "schedule": "0 * * * *",
            "enabled": True,
            "channel_id": 12345
        }
        
        try:
            with patch('bot.web.routes.tasks._scheduler_ref', mock_scheduler):
                with patch('bot.config.tasks.load_scheduled_tasks', return_value={"test-task": task_config}):
                    result = await reload_single_task("test-task")
            
            assert result["success"] is True
            assert "test-task" in result["message"]
            # Verify add_job was called with correct function
            mock_scheduler.add_job.assert_called_once()
            call_args = mock_scheduler.add_job.call_args
            # First arg should be run_scheduled_task, not call_llm_with_tools
            assert call_args[0][0].__name__ == "run_scheduled_task"
        finally:
            # Restore sys.modules
            sys.modules.clear()
            sys.modules.update(original_modules)

    @pytest.mark.asyncio
    async def test_reload_single_task_disabled(self):
        """Test reload_single_task handles disabled task."""
        from bot.web.routes.tasks import reload_single_task
        from unittest.mock import MagicMock
        
        mock_scheduler = MagicMock()
        mock_scheduler.get_jobs.return_value = []
        
        task_config = {
            "name": "test-task",
            "schedule": "0 * * * *",
            "enabled": False  # Disabled
        }
        
        with patch('bot.web.routes.tasks._scheduler_ref', mock_scheduler):
            with patch('bot.config.tasks.load_scheduled_tasks', return_value={"test-task": task_config}):
                result = await reload_single_task("test-task")
        
        assert result["success"] is True
        assert "disabled" in result["message"]
        # add_job should NOT be called for disabled task
        mock_scheduler.add_job.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_task(self):
        """Test creating a new task via POST /api/tasks."""
        from bot.web.routes.tasks import create_task
        from unittest.mock import MagicMock, patch
        import sys
        
        task_config = {
            "name": "new-task",
            "enabled": True,
            "cron": "0 * * * *",
            "channel_id": 12345,
            "model": "test-model",
            "prompt": "Test prompt"
        }
        
        # Mock llmcord module for run_scheduled_task
        mock_run_task = MagicMock()
        mock_run_task.__name__ = "run_scheduled_task"
        mock_llmcord = MagicMock()
        mock_llmcord.run_scheduled_task = mock_run_task
        
        from bot.web.routes.tasks import TaskCreate
        
        with patch('bot.web.routes.tasks.TASKS_DIR', tempfile.gettempdir()):
            with patch('bot.web.routes.tasks._scheduler_ref', MagicMock()):
                with patch('builtins.open', MagicMock()):
                    with patch('yaml.dump', return_value="name: new-task"):
                        result = await create_task(TaskCreate(name="new-task", config=task_config))
        
        assert result.name == "new-task"
        assert result.config["name"] == "new-task"

    @pytest.mark.asyncio
    async def test_update_task(self):
        """Test updating an existing task via PUT /api/tasks/{name}."""
        from bot.web.routes.tasks import update_task, TaskUpdate
        from unittest.mock import MagicMock, patch
        
        task_config = {
            "name": "test-task",
            "enabled": True,
            "cron": "0 * * * *",
            "channel_id": 12345,
            "model": "updated-model",
            "prompt": "Updated prompt"
        }
        
        with patch('bot.web.routes.tasks.TASKS_DIR', tempfile.gettempdir()):
            with patch('os.path.exists', return_value=True):
                with patch('builtins.open', MagicMock()):
                    with patch('yaml.dump', return_value="name: test-task"):
                        result = await update_task("test-task", TaskUpdate(config=task_config))
        
        assert result.name == "test-task"
        assert result.config["model"] == "updated-model"

    @pytest.mark.asyncio
    async def test_delete_task(self):
        """Test deleting a task via DELETE /api/tasks/{name}."""
        from bot.web.routes.tasks import delete_task
        from unittest.mock import patch, MagicMock
        
        with patch('bot.web.routes.tasks.TASKS_DIR', tempfile.gettempdir()):
            with patch('os.path.exists', return_value=True):
                with patch('os.remove', MagicMock()) as mock_remove:
                    result = await delete_task("test-task")
        
        assert result["success"] is True
        assert "test-task" in result["message"]

    @pytest.mark.asyncio
    async def test_run_task_not_found(self):
        """Test run_task raises 404 for unknown task."""
        from bot.web.routes.tasks import run_task
        from fastapi import HTTPException
        from unittest.mock import MagicMock
        
        mock_scheduler = MagicMock()
        mock_scheduler.get_jobs.return_value = []
        
        with patch('bot.web.routes.tasks._scheduler_ref', mock_scheduler):
            with patch('bot.config.tasks.load_scheduled_tasks', return_value={}):
                with pytest.raises(HTTPException) as exc_info:
                    await run_task("nonexistent")
        
        assert exc_info.value.status_code == 404
        assert "not found" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_run_task_disabled(self):
        """Test run_task returns failure for disabled task."""
        from bot.web.routes.tasks import run_task
        from unittest.mock import MagicMock
        
        mock_scheduler = MagicMock()
        
        task_config = {
            "name": "test-task",
            "enabled": False,  # Disabled
            "cron": "0 * * * *"
        }
        
        with patch('bot.web.routes.tasks._scheduler_ref', mock_scheduler):
            with patch('bot.config.tasks.load_scheduled_tasks', return_value={"test-task": task_config}):
                result = await run_task("test-task")
        
        assert result["success"] is False
        assert "disabled" in result["message"]

    @pytest.mark.asyncio
    async def test_run_task_success(self):
        """Test run_task successfully queues task for execution."""
        from bot.web.routes.tasks import run_task
        from unittest.mock import MagicMock, patch
        import sys
        
        mock_scheduler = MagicMock()
        
        # Mock llmcord module for run_scheduled_task
        mock_run_task = MagicMock()
        mock_run_task.__name__ = "run_scheduled_task"
        
        task_config = {
            "name": "test-task",
            "enabled": True,
            "cron": "0 * * * *",
            "channel_id": 12345
        }
        
        # Add mock to sys.modules to intercept the import
        mock_llmcord_module = MagicMock()
        mock_llmcord_module.run_scheduled_task = mock_run_task
        
        original_modules = sys.modules.copy()
        sys.modules['bot.llmcord'] = mock_llmcord_module
        
        try:
            with patch('bot.web.routes.tasks._scheduler_ref', mock_scheduler):
                with patch('bot.config.tasks.load_scheduled_tasks', return_value={"test-task": task_config}):
                    result = await run_task("test-task")
            
            assert result["success"] is True
            assert "test-task" in result["message"]
            # Verify add_job was called
            mock_scheduler.add_job.assert_called_once()
        finally:
            # Restore sys.modules
            sys.modules.clear()
            sys.modules.update(original_modules)

    @pytest.mark.asyncio
    async def test_run_task_by_internal_name(self):
        """Test run_task finds task by internal name field when filename differs.
        
        This tests the fix for: filename 'stock-market-checker.yaml' has internal 
        name 'stock_market_check'. The run endpoint should find it either way.
        """
        from bot.web.routes.tasks import run_task
        from unittest.mock import MagicMock, patch
        import sys
        
        mock_scheduler = MagicMock()
        
        # Mock llmcord module for run_scheduled_task
        mock_run_task = MagicMock()
        mock_run_task.__name__ = "run_scheduled_task"
        
        # Task loaded from file - key is filename but internal name differs
        task_config = {
            "name": "stock_market_check",  # Internal name (underscore)
            "enabled": True,
            "cron": "0 * * * *",
            "channel_id": 12345
        }
        
        # Simulate: tasks dict keyed by filename, but name field has underscore
        tasks_from_file = {"stock-market-checker": task_config}
        
        mock_llmcord_module = MagicMock()
        mock_llmcord_module.run_scheduled_task = mock_run_task
        
        original_modules = sys.modules.copy()
        sys.modules['bot.llmcord'] = mock_llmcord_module
        
        try:
            # Test 1: Lookup by filename (stock-market-checker)
            with patch('bot.web.routes.tasks._scheduler_ref', mock_scheduler):
                with patch('bot.config.tasks.load_scheduled_tasks', return_value=tasks_from_file):
                    result = await run_task("stock-market-checker")
            
            assert result["success"] is True
            assert "stock-market-checker" in result["message"]
            
            # Reset mock
            mock_scheduler.reset_mock()
            
            # Test 2: Lookup by internal name field (stock_market_check)
            with patch('bot.web.routes.tasks._scheduler_ref', mock_scheduler):
                with patch('bot.config.tasks.load_scheduled_tasks', return_value=tasks_from_file):
                    result = await run_task("stock_market_check")
            
            assert result["success"] is True
            assert "stock_market_check" in result["message"] or "queued" in result["message"]
            
        finally:
            sys.modules.clear()
            sys.modules.update(original_modules)


class TestSkillsEndpoint:
    """Test skills API endpoints."""

    @pytest.mark.asyncio
    async def test_get_skills_returns_list(self):
        """Test that get_skills returns a list."""
        from bot.web.routes.skills import get_skills
        
        with patch('bot.web.routes.skills.SKILLS_DIR', tempfile.gettempdir()):
            with patch('os.listdir', return_value=[]):
                result = await get_skills()
        
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_get_skills_with_files(self):
        """Test get_skills with skill files."""
        from bot.web.routes.skills import get_skills
        
        mock_content = """# Web Search Skill

This skill allows searching the web.
"""
        
        with patch('bot.web.routes.skills.SKILLS_DIR', tempfile.gettempdir()):
            with patch('os.listdir', return_value=['web_search.md']):
                with patch('builtins.open', MagicMock()):
                    with patch('os.path.join', return_value='temp/web_search.md'):
                        result = await get_skills()
        
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_get_skill_not_found(self):
        """Test get_skill raises 404 for unknown skill."""
        from bot.web.routes.skills import get_skill
        from fastapi import HTTPException
        
        with patch('bot.web.routes.skills.SKILLS_DIR', tempfile.gettempdir()):
            with patch('os.path.exists', return_value=False):
                with pytest.raises(HTTPException) as exc_info:
                    await get_skill("nonexistent")
        
        assert exc_info.value.status_code == 404

    def test_skill_description_extraction(self):
        """Test description extraction from skill content."""
        from bot.web.routes.skills import _extract_skill_description
        
        # Test with content
        content = """# Web Search

This skill enables web searching capability.
More content here.
"""
        desc = _extract_skill_description(content)
        assert "web searching" in desc.lower() or "skill" in desc.lower()
        
        # Test with empty content
        assert _extract_skill_description("") is None
        assert _extract_skill_description(None) is None


class TestFilePathResolution:
    """Test that file paths resolve correctly."""

    def test_personas_path_uses_config_dir(self):
        """Test personas path includes config/personas."""
        from bot.web.routes.personas import PERSONAS_DIR
        assert "personas" in PERSONAS_DIR.lower()

    def test_tasks_path_uses_config_dir(self):
        """Test tasks path includes config/tasks."""
        from bot.web.routes.tasks import TASKS_DIR
        assert "tasks" in TASKS_DIR.lower()

    def test_skills_path_uses_tools_skills(self):
        """Test skills path includes llm/tools/skills."""
        from bot.web.routes.skills import SKILLS_DIR
        assert "skills" in SKILLS_DIR.lower()