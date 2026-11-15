#!/usr/bin/env python3
"""Runtime configuration, externalised for packaging.

Every machine-specific value the stack needs lives here and can be
overridden by environment variables -- no code edits per deployment:

  VAL_MODELS_ROOT   model root scanned by the framework (default C:\\Ai\\models)
  FATE_TAU          verify/select abstention threshold (default 0.5)
  FATE_API_KEY      if set, the service requires header X-API-Key
  FATE_HOST/FATE_PORT  service bind (default 127.0.0.1:8790)
  VAL_PIPELINE_OUT  default output dir for pipeline runs
"""
from __future__ import annotations

import os
from pathlib import Path

MODELS_ROOT = Path(os.environ.get("VAL_MODELS_ROOT", r"C:\Ai\models"))
TAU = float(os.environ.get("FATE_TAU", "0.5"))
API_KEY = os.environ.get("FATE_API_KEY", "")
HOST = os.environ.get("FATE_HOST", "127.0.0.1")
PORT = int(os.environ.get("FATE_PORT", "8790"))
PIPELINE_OUT = Path(os.environ.get("VAL_PIPELINE_OUT", "reports/pipeline_run"))
JUGGERNAUT_DIR = MODELS_ROOT / "juggernaut-xl-v9"
I2V_DIR = MODELS_ROOT / "CogVideoX-5b-I2V"
COGVIDEO_DIR = MODELS_ROOT / "CogVideoX-2b"


def describe() -> dict:
    return {"models_root": str(MODELS_ROOT), "tau": TAU,
            "api_key_required": bool(API_KEY), "host": HOST, "port": PORT,
            "pipeline_out": str(PIPELINE_OUT),
            "models_present": {"juggernaut-xl-v9": JUGGERNAUT_DIR.exists(),
                               "CogVideoX-5b-I2V": I2V_DIR.exists(),
                               "CogVideoX-2b": COGVIDEO_DIR.exists()}}
