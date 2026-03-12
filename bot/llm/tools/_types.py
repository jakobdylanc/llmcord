"""
Shared types for bot tools.
"""

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ToolEntry:
    """Tool entry with schema, function, and optional formatter."""
    schema: dict                       # OpenAI-format schema sent to the model
    fn: Callable | None = None         # called locally when model invokes this tool
    formatter: Callable | None = None  # optional: formatter(result, args) -> str