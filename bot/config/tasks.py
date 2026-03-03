from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


TASKS_DIR = Path(__file__).parent / "tasks"


def load_scheduled_tasks(config: dict[str, Any]) -> Dict[str, dict[str, Any]]:
    """
    Load scheduled task definitions from separate YAML files under bot/config/tasks,
    optionally merging any inline tasks from config['scheduled_tasks'] (non-legacy format).
    File-based tasks override inline ones on name conflicts.
    """
    tasks: Dict[str, dict[str, Any]] = {}

    inline = config.get("scheduled_tasks", {})
    # Skip legacy single-task format here; it is handled separately in llmcord.setup_scheduled_tasks.
    if isinstance(inline, dict) and not ("enabled" in inline and "cron" in inline):
        for name, tc in inline.items():
            if isinstance(tc, dict):
                tasks[name] = dict(tc)

    if TASKS_DIR.is_dir():
        for path in TASKS_DIR.glob("*.yaml"):
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if not isinstance(data, dict):
                continue
            name = str(data.get("name") or path.stem)
            tasks[name] = data

    return tasks

