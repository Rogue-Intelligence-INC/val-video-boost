#!/usr/bin/env python3
"""DEPRECATED: Use settings.py instead.

This module is kept for backward compatibility only. All configuration
has moved to settings.py which provides the same interface plus additional
configuration options.

New code should import from settings.py instead:
    from settings import MODELS_ROOT, TAU, API_KEY, HOST, PORT, ...
"""
from __future__ import annotations

import sys
from pathlib import Path

# Import everything from settings for backward compatibility
sys.path.insert(0, str(Path(__file__).resolve().parent))
from settings import (
    MODELS_ROOT,
    TAU,
    API_KEY,
    HOST,
    PORT,
    PIPELINE_OUT,
    JUGGERNAUT_DIR,
    I2V_DIR,
    COGVIDEO_DIR,
    describe,
)

# Re-export for backward compatibility
__all__ = [
    "MODELS_ROOT",
    "TAU",
    "API_KEY",
    "HOST",
    "PORT",
    "PIPELINE_OUT",
    "JUGGERNAUT_DIR",
    "I2V_DIR",
    "COGVIDEO_DIR",
    "describe",
]
