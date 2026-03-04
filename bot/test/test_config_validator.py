#!/usr/bin/env python3
"""
Test script to validate config.yaml format.

Usage:
    python test_config_validator.py [config_file]

Examples:
    python test_config_validator.py                    # Uses default config.yaml
    python test_config_validator.py config.yaml        # Explicit path
"""

import sys
import logging

from bot.config.loader import get_config

# Setup logging to see validation messages
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)

if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    
    print(f"Validating config: {config_file}")
    print("-" * 70)
    
    try:
        config = get_config(config_file)
        print("-" * 70)
        print("✅ Config validation PASSED")
        print(f"   Providers: {list(config.get('providers', {}).keys())}")
        print(f"   Models: {list(config.get('models', {}).keys())}")
        print(f"   Tasks: {list(config.get('scheduled_tasks', {}).keys())}")
        sys.exit(0)
    except SystemExit as e:
        if e.code != 0:
            print("-" * 70)
            print("❌ Config validation FAILED")
            sys.exit(1)
        raise
