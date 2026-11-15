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
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fate_config as cfg

hard_fail = []


def check(name: str, ok: bool, detail: str, hard: bool = True) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name:<28} {detail}")
    if not ok and hard:
        hard_fail.append(name)


def main() -> int:
    print(f"python {sys.version.split()[0]}")
    check("python>=3.10", sys.version_info >= (3, 10), sys.version.split()[0])
    for mod in ("torch", "diffusers", "transformers", "cv2", "imageio", "numpy", "PIL"):
        check(f"import {mod}", importlib.util.find_spec(mod) is not None, "installed" if importlib.util.find_spec(mod) else "missing")
    for mod in ("fastapi", "uvicorn"):
        check(f"import {mod} (service)", importlib.util.find_spec(mod) is not None,
              "installed" if importlib.util.find_spec(mod) else "missing - service only", hard=False)
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
    present = cfg.describe()["models_present"]
    check("model juggernaut-xl-v9", present["juggernaut-xl-v9"], str(cfg.JUGGERNAUT_DIR))
    check("model CogVideoX-2b", present["CogVideoX-2b"], str(cfg.COGVIDEO_DIR), hard=False)
    check("model CogVideoX-5b-I2V", present["CogVideoX-5b-I2V"], str(cfg.I2V_DIR), hard=False)
    try:
        from face_id import FaceID
        fid = FaceID()
        hero = Path(__file__).resolve().parents[1] / "reports" / "pipeline_run" / "hero_studio.png"
        ok = fid.embedding(hero) is not None if hero.exists() else True
        check("ArcFace buffalo_l", True, "initialised" + ("; hero face read ok" if hero.exists() and ok else ""))
    except Exception as e:
        check("ArcFace buffalo_l", False, repr(e)[:120])
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        check("ffmpeg (imageio)", Path(exe).exists(), exe)
    except Exception as e:
        check("ffmpeg (imageio)", False, repr(e)[:120], hard=False)
    free = shutil.disk_usage(str(Path(__file__).resolve().parents[1])).free / 1e9
    check("disk free>=5GB", free >= 5, f"{free:.0f} GB free")
    print("\nDOCTOR:", "OK - ready" if not hard_fail else f"NOT READY: {hard_fail}")
    return 0 if not hard_fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
