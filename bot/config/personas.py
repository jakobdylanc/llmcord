from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml


PERSONAS_DIR = Path(__file__).parent / "personas"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _read_yaml_prompt(path: Path) -> str:
    data: Any = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if isinstance(data, dict):
        # Allow either `prompt` or `system_prompt` as the key.
        prompt = data.get("prompt") or data.get("system_prompt")
        if isinstance(prompt, str):
            return prompt.strip()
    raise ValueError(f"Persona YAML at {path} must contain a 'prompt' or 'system_prompt' string")


def load_persona(name: str) -> str:
    """
    Load a persona (人格設定檔) by name.

    Looks for, in order:
    - <name>.md
    - <name>.txt
    - <name>.yaml / <name>.yml
    """
    base = PERSONAS_DIR
    candidates = [
        base / f"{name}.md",
        base / f"{name}.txt",
        base / f"{name}.yaml",
        base / f"{name}.yml",
    ]

    for path in candidates:
        if path.is_file():
            if path.suffix in {".md", ".txt"}:
                return _read_text(path)
            return _read_yaml_prompt(path)

    raise FileNotFoundError(f"Persona '{name}' not found in {PERSONAS_DIR}")


def try_load_persona(name: str | None) -> str | None:
    """
    Best-effort persona loader that logs a warning instead of raising.
    """
    if not name:
        return None
    try:
        return load_persona(name)
    except Exception as e:
        logging.warning("Failed to load persona '%s': %s", name, e)
        return None

