#!/usr/bin/env python3
"""Centralized settings for val-video-boost.

All machine-specific paths and configuration are loaded from environment
variables with sensible defaults. This module is the single source of truth
for all path configuration.

Environment Variables:
    VAL_MODELS_ROOT: Root directory for model weights (default: ./models)
    VAL_VALHALLA_ROOT: Valhalla-main repository root (required for Valhalla engine)
    VAL_VALHALLA_EXE: Path to valhalla_base.exe (default: VAL_VALHALLA_ROOT/target/release/valhalla_base.exe)
    VAL_VALHALLA_EXPORT_EXE: Path to valhalla_model_export.exe (default: VAL_VALHALLA_ROOT/target/release/valhalla_model_export.exe)
    VAL_VALHALLA_ENCODE: Path to encode.py (default: VAL_VALHALLA_ROOT/tools/valhalla_dl_front/encode.py)
    VAL_VALHALLA_QWEN_MODEL: Path to Qwen model directory (default: VAL_MODELS_ROOT/Qwen2.5-3B-Instruct)
    VAL_YUNET_MODEL: Path to YuNet face detection model (default: VAL_MODELS_ROOT/yunet/face_detection_yunet_2023mar.onnx)
    VAL_IP_ADAPTER_ROOT: Path to IP-Adapter models (default: VAL_MODELS_ROOT/ip-adapter)
    FATE_TAU: Verify/select abstention threshold (default: 0.5)
    FATE_API_KEY: API key for service authentication (optional)
    FATE_HOST: Service bind host (default: 127.0.0.1)
    FATE_PORT: Service bind port (default: 8790)
    VAL_PIPELINE_OUT: Default output directory for pipeline runs (default: reports/pipeline_run)
    VAL_CHARACTERS_DIR: Directory for character data files (default: ./characters)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# Project root is parent of code/ directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Models root - defaults to ./models relative to project root
MODELS_ROOT = Path(os.environ.get("VAL_MODELS_ROOT", PROJECT_ROOT / "models"))

# Valhalla engine paths - these are EXTERNAL dependencies not in this repo
# They must be configured via environment variables or will raise clear errors when accessed
_valhalla_root_env = os.environ.get("VAL_VALHALLA_ROOT")
VALHALLA_ROOT = Path(_valhalla_root_env) if _valhalla_root_env else None

# Valhalla executables
_valhalla_exe_env = os.environ.get("VAL_VALHALLA_EXE")
if _valhalla_exe_env:
    VALHALLA_EXE = Path(_valhalla_exe_env)
elif VALHALLA_ROOT:
    VALHALLA_EXE = VALHALLA_ROOT / "target" / "release" / "valhalla_base.exe"
else:
    VALHALLA_EXE = None

_valhalla_export_exe_env = os.environ.get("VAL_VALHALLA_EXPORT_EXE")
if _valhalla_export_exe_env:
    VALHALLA_EXPORT_EXE = Path(_valhalla_export_exe_env)
elif VALHALLA_ROOT:
    VALHALLA_EXPORT_EXE = VALHALLA_ROOT / "target" / "release" / "valhalla_model_export.exe"
else:
    VALHALLA_EXPORT_EXE = None

_valhalla_encode_env = os.environ.get("VAL_VALHALLA_ENCODE")
if _valhalla_encode_env:
    VALHALLA_ENCODE = Path(_valhalla_encode_env)
elif VALHALLA_ROOT:
    VALHALLA_ENCODE = VALHALLA_ROOT / "tools" / "valhalla_dl_front" / "encode.py"
else:
    # Fallback: check if encode.py was copied to our tools directory
    _local_encode = PROJECT_ROOT / "tools" / "valhalla_dl_front" / "encode.py"
    VALHALLA_ENCODE = _local_encode if _local_encode.is_file() else None

# Qwen model directory
_qwen_model_env = os.environ.get("VAL_VALHALLA_QWEN_MODEL")
if _qwen_model_env:
    VALHALLA_QWEN_MODEL = Path(_qwen_model_env)
else:
    VALHALLA_QWEN_MODEL = MODELS_ROOT / "Qwen2.5-3B-Instruct"

# YuNet face detection model
_yunet_env = os.environ.get("VAL_YUNET_MODEL")
if _yunet_env:
    YUNET_MODEL = Path(_yunet_env)
else:
    YUNET_MODEL = MODELS_ROOT / "yunet" / "face_detection_yunet_2023mar.onnx"

# IP-Adapter models root
_ip_adapter_env = os.environ.get("VAL_IP_ADAPTER_ROOT")
if _ip_adapter_env:
    IP_ADAPTER_ROOT = Path(_ip_adapter_env)
else:
    IP_ADAPTER_ROOT = MODELS_ROOT / "ip-adapter"

# Specific model directories
JUGGERNAUT_DIR = MODELS_ROOT / "juggernaut-xl-v9"
COGVIDEO_DIR = MODELS_ROOT / "CogVideoX-2b"
I2V_DIR = MODELS_ROOT / "CogVideoX-5b-I2V"
CONSISID_DIR = MODELS_ROOT / "ConsisID-preview"

# Service configuration
TAU = float(os.environ.get("FATE_TAU", "0.5"))
API_KEY = os.environ.get("FATE_API_KEY", "")
HOST = os.environ.get("FATE_HOST", "127.0.0.1")
PORT = int(os.environ.get("FATE_PORT", "8790"))

# Output directories
PIPELINE_OUT = Path(os.environ.get("VAL_PIPELINE_OUT", PROJECT_ROOT / "reports" / "pipeline_run"))

# Characters directory
CHARACTERS_DIR = Path(os.environ.get("VAL_CHARACTERS_DIR", PROJECT_ROOT / "characters"))


def require_valhalla_exe() -> Path:
    """Get path to valhalla_base.exe, raising clear error if not configured."""
    if VALHALLA_EXE is None:
        raise RuntimeError(
            "Valhalla engine path not configured. Set environment variable:\n"
            "  VAL_VALHALLA_EXE=/path/to/valhalla_base.exe\n"
            "or\n"
            "  VAL_VALHALLA_ROOT=/path/to/Valhalla-main"
        )
    if not VALHALLA_EXE.is_file():
        raise FileNotFoundError(
            f"Valhalla engine not found at {VALHALLA_EXE}\n"
            f"Set VAL_VALHALLA_EXE or VAL_VALHALLA_ROOT to the correct path."
        )
    return VALHALLA_EXE


def require_valhalla_export_exe() -> Path:
    """Get path to valhalla_model_export.exe, raising clear error if not configured."""
    if VALHALLA_EXPORT_EXE is None:
        raise RuntimeError(
            "Valhalla model export tool not configured. Set environment variable:\n"
            "  VAL_VALHALLA_EXPORT_EXE=/path/to/valhalla_model_export.exe\n"
            "or\n"
            "  VAL_VALHALLA_ROOT=/path/to/Valhalla-main"
        )
    if not VALHALLA_EXPORT_EXE.is_file():
        raise FileNotFoundError(
            f"Valhalla model export tool not found at {VALHALLA_EXPORT_EXE}\n"
            f"Set VAL_VALHALLA_EXPORT_EXE or VAL_VALHALLA_ROOT to the correct path."
        )
    return VALHALLA_EXPORT_EXE


def require_valhalla_encode() -> Path:
    """Get path to encode.py, raising clear error if not configured."""
    if VALHALLA_ENCODE is None:
        raise RuntimeError(
            "Valhalla encode.py not configured. Set environment variable:\n"
            "  VAL_VALHALLA_ENCODE=/path/to/encode.py\n"
            "or\n"
            "  VAL_VALHALLA_ROOT=/path/to/Valhalla-main"
        )
    if not VALHALLA_ENCODE.is_file():
        raise FileNotFoundError(
            f"Valhalla encode.py not found at {VALHALLA_ENCODE}\n"
            f"Set VAL_VALHALLA_ENCODE or VAL_VALHALLA_ROOT to the correct path."
        )
    return VALHALLA_ENCODE


def describe() -> dict[str, Any]:
    """Return current configuration as a dictionary for diagnostics."""
    return {
        "project_root": str(PROJECT_ROOT),
        "models_root": str(MODELS_ROOT),
        "models_present": {
            "juggernaut-xl-v9": JUGGERNAUT_DIR.exists(),
            "CogVideoX-2b": COGVIDEO_DIR.exists(),
            "CogVideoX-5b-I2V": I2V_DIR.exists(),
            "ConsisID-preview": CONSISID_DIR.exists(),
        },
        "valhalla": {
            "root": str(VALHALLA_ROOT) if VALHALLA_ROOT else None,
            "exe": str(VALHALLA_EXE) if VALHALLA_EXE else None,
            "exe_exists": VALHALLA_EXE.is_file() if VALHALLA_EXE else False,
            "export_exe": str(VALHALLA_EXPORT_EXE) if VALHALLA_EXPORT_EXE else None,
            "export_exe_exists": VALHALLA_EXPORT_EXE.is_file() if VALHALLA_EXPORT_EXE else False,
            "encode": str(VALHALLA_ENCODE) if VALHALLA_ENCODE else None,
            "encode_exists": VALHALLA_ENCODE.is_file() if VALHALLA_ENCODE else False,
            "qwen_model": str(VALHALLA_QWEN_MODEL),
            "qwen_model_exists": VALHALLA_QWEN_MODEL.exists(),
        },
        "yunet_model": str(YUNET_MODEL),
        "yunet_model_exists": YUNET_MODEL.exists(),
        "ip_adapter_root": str(IP_ADAPTER_ROOT),
        "ip_adapter_root_exists": IP_ADAPTER_ROOT.exists(),
        "service": {
            "tau": TAU,
            "api_key_required": bool(API_KEY),
            "host": HOST,
            "port": PORT,
        },
        "pipeline_out": str(PIPELINE_OUT),
        "characters_dir": str(CHARACTERS_DIR),
        "characters_dir_exists": CHARACTERS_DIR.exists(),
    }


def check_required_paths() -> list[str]:
    """Check which required paths are missing. Returns list of error messages."""
    errors = []
    
    if not MODELS_ROOT.exists():
        errors.append(f"Models root not found: {MODELS_ROOT} (set VAL_MODELS_ROOT)")
    
    if not YUNET_MODEL.exists():
        errors.append(f"YuNet model not found: {YUNET_MODEL} (set VAL_YUNET_MODEL)")
    
    # Valhalla paths are optional for basic operation, but required for certain features
    # They'll raise errors when actually needed via require_* functions
    
    return errors


# Legacy compatibility: make old fate_config importable
# This allows existing code to import fate_config during migration
if __name__ != "__main__" and "fate_config" not in sys.modules:
    sys.modules["fate_config"] = sys.modules[__name__]
    # Add aliases for backward compatibility
    MODELS_ROOT.__class__.describe = staticmethod(describe)
