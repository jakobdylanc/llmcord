from __future__ import annotations

import logging
import os
import sys
from typing import Any

import yaml

from .validator import validate_config, ConfigValidationError


DEFAULT_CONFIG_FILE = "config.yaml"
CONFIG_ENV_VAR = "CONFIG_PATH"


def get_config_path() -> str:
    """
    Resolve the config path, preferring an explicit environment override.
    """
    env_path = os.environ.get(CONFIG_ENV_VAR)
    if env_path:
        return env_path
    return DEFAULT_CONFIG_FILE


def _load_raw_config(path: str | None = None) -> dict[str, Any]:
    cfg_path = path or get_config_path()
    try:
        with open(cfg_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        logging.error("Config file not found: %s", cfg_path)
        sys.exit(1)
    except yaml.YAMLError as e:
        logging.error("YAML parsing error in %s: %s", cfg_path, e)
        sys.exit(1)
    
    if not isinstance(data, dict):
        logging.error("Config root must be a mapping, got %s", type(data).__name__)
        sys.exit(1)
    
    return data


def get_config(path: str | None = None) -> dict[str, Any]:
    """
    Public helper for loading configuration.

    - Respects CONFIG_PATH if set.
    - Performs comprehensive YAML validation.
    - Exits with error code 1 if validation fails.
    - Returns the raw dict; higher-level helpers live in bot.config.models.
    """
    cfg_path = path or get_config_path()
    cfg = _load_raw_config(cfg_path)
    
    try:
        validate_config(cfg, cfg_path)
    except ConfigValidationError:
        sys.exit(1)
    
    return cfg

