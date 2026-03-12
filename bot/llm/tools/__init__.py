# Dynamic tool imports - fully lazy loading
# No manual updates needed when adding/removing tools!

from ._types import ToolEntry

from .registry import (
    get_openai_tools,
    get_tools,
    build_tool_registry,
    build_brave_registry,
    format_tool_result,
    execute_tool_call,
    reload_tools,
)

# Lazy load any attribute from tool modules dynamically
def __getattr__(name):
    """Lazy load any attribute from tool modules - auto-discovers all modules."""
    import os
    import importlib
    
    tools_dir = os.path.dirname(__file__)
    
    # Scan all .py files in tools directory (except this file and private)
    for filename in os.listdir(tools_dir):
        if not filename.endswith('.py') or filename.startswith('_'):
            continue
        if filename == '__init__.py':
            continue
            
        module_name = filename[:-3]  # remove .py
        
        try:
            module = importlib.import_module(f'.{module_name}', package=__name__)
            if hasattr(module, name):
                return getattr(module, name)
        except (ImportError, AttributeError):
            continue
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
