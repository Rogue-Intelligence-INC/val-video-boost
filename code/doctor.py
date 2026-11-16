#!/usr/bin/env python3
"""Environment doctor -- run before anything else on a new machine.

Checks (exit 0 only if all hard checks pass):
  python >= 3.10 · torch + CUDA GPU with >= 12 GB VRAM · core deps
  importable · model dirs present under VAL_MODELS_ROOT · ArcFace
  (insightface buffalo_l) initialises · ffmpeg available via imageio
  · free disk >= 5 GB · (optional) service deps fastapi/uvicorn.
Soft checks only warn (a missing I2V model disables one route,
it does not block the image stack).
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import settings

hard_fail = []


def check(name: str, ok: bool, detail: str, hard: bool = True) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name:<28} {detail}")
    if not ok and hard:
        hard_fail.append(name)


def main() -> int:
    print("=" * 70)
    print("val-video-boost environment doctor")
    print("=" * 70)
    print()
    
    print(f"python {sys.version.split()[0]}")
    check("python>=3.10", sys.version_info >= (3, 10), sys.version.split()[0])
    
    # Core dependencies
    for mod in ("torch", "diffusers", "transformers", "cv2", "imageio", "numpy", "PIL"):
        check(f"import {mod}", importlib.util.find_spec(mod) is not None, "installed" if importlib.util.find_spec(mod) else "missing")
    for mod in ("fastapi", "uvicorn"):
        check(f"import {mod} (service)", importlib.util.find_spec(mod) is not None,
              "installed" if importlib.util.find_spec(mod) else "missing - service only", hard=False)
    
    # GPU
    try:
        import torch
        cuda = torch.cuda.is_available()
        name = torch.cuda.get_device_name(0) if cuda else "-"
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9 if cuda else 0
        check("CUDA GPU", cuda, name)
        check("VRAM>=12GB", vram >= 12, f"{vram:.1f} GB")
        print(f"      torch {torch.__version__}  TF32/cuDNN switches live in adaptive_framework.enable_gpu_accel()")
    except Exception as e:
        check("CUDA GPU", False, repr(e))
    
    # Models
    cfg_desc = settings.describe()
    present = cfg_desc["models_present"]
    check("model juggernaut-xl-v9", present["juggernaut-xl-v9"], str(settings.JUGGERNAUT_DIR))
    check("model CogVideoX-2b", present["CogVideoX-2b"], str(settings.COGVIDEO_DIR), hard=False)
    check("model CogVideoX-5b-I2V", present["CogVideoX-5b-I2V"], str(settings.I2V_DIR), hard=False)
    
    # YuNet face detection
    check("YuNet model", settings.YUNET_MODEL.exists(), str(settings.YUNET_MODEL))
    
    # ArcFace
    try:
        from face_id import FaceID
        fid = FaceID()
        hero = Path(__file__).resolve().parents[1] / "reports" / "pipeline_run" / "hero_studio.png"
        ok = fid.embedding(hero) is not None if hero.exists() else True
        check("ArcFace buffalo_l", True, "initialised" + ("; hero face read ok" if hero.exists() and ok else ""))
    except Exception as e:
        check("ArcFace buffalo_l", False, repr(e)[:120])
    
    # FFmpeg
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        check("ffmpeg (imageio)", Path(exe).exists(), exe)
    except Exception as e:
        check("ffmpeg (imageio)", False, repr(e)[:120], hard=False)
    
    # Disk space
    free = shutil.disk_usage(str(Path(__file__).resolve().parents[1])).free / 1e9
    check("disk free>=5GB", free >= 5, f"{free:.0f} GB free")
    
    # Characters directory
    chars_ok = settings.CHARACTERS_DIR.exists()
    check("characters/ directory", chars_ok, str(settings.CHARACTERS_DIR), hard=False)
    if chars_ok:
        try:
            from character_loader import list_characters
            chars = list_characters()
            print(f"      available characters: {', '.join(chars) if chars else 'none'}")
        except Exception as e:
            print(f"      warning: could not list characters: {e}")
    
    # Valhalla engine (optional, but report status)
    print()
    print("Valhalla engine (optional, required for certain features):")
    valhalla = cfg_desc["valhalla"]
    valhalla_ok = valhalla["exe_exists"] and valhalla["export_exe_exists"] and valhalla["encode_exists"]
    check("  valhalla_base.exe", valhalla["exe_exists"], valhalla["exe"] or "not configured", hard=False)
    check("  valhalla_model_export.exe", valhalla["export_exe_exists"], valhalla["export_exe"] or "not configured", hard=False)
    check("  encode.py", valhalla["encode_exists"], valhalla["encode"] or "not configured", hard=False)
    check("  Qwen model", valhalla["qwen_model_exists"], valhalla["qwen_model"], hard=False)
    
    # Configuration summary
    print()
    print("Configuration:")
    print(f"  VAL_MODELS_ROOT:      {cfg_desc['models_root']}")
    print(f"  VAL_VALHALLA_ROOT:    {valhalla['root'] or 'not set'}")
    print(f"  VAL_CHARACTERS_DIR:   {cfg_desc['characters_dir']}")
    print(f"  FATE_TAU:             {cfg_desc['service']['tau']}")
    print(f"  FATE_API_KEY:         {'set' if cfg_desc['service']['api_key_required'] else 'not set'}")
    print(f"  FATE_HOST:            {cfg_desc['service']['host']}")
    print(f"  FATE_PORT:            {cfg_desc['service']['port']}")
    
    print()
    print("DOCTOR:", "OK - ready" if not hard_fail else f"NOT READY: {hard_fail}")
    
    # Write full config to file for debugging
    config_file = Path(__file__).resolve().parents[1] / "doctor-config.json"
    config_file.write_text(json.dumps(cfg_desc, indent=2), encoding="utf-8")
    print(f"Full configuration written to: {config_file}")
    
    return 0 if not hard_fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
