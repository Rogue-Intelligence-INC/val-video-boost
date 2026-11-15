#!/usr/bin/env python3
"""Video-only rerun at 25 steps (2-6 steps leave CogVideoX latents unformed).

Updates reports/media_pipeline/full_quality/report.json in place: replaces the
video section and the consistency_video / performance verdicts with measured
25-step values. A short untimed warmup runs first so offload paging stays out
of the timings.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import module_patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "media_pipeline" / "full_quality"
PATCH_DIR = ROOT / "reports" / "media_pipeline" / "transformer_aligned" / "patches"
PROMPT = "a square ceramic sign in a quiet room, slow steady camera"
STEPS = 25


def clip_metrics(frames: list[Path]) -> dict:
    import numpy as np
    from PIL import Image

    means, stds, adjacent = [], [], []
    previous = None
    for frame in frames:
        arr = np.asarray(Image.open(frame).convert("L")).astype(np.float64)
        means.append(float(arr.mean()))
        stds.append(float(arr.std()))
        if previous is not None:
            adjacent.append(float(np.abs(arr - previous).mean()))
        previous = arr
    first = np.asarray(Image.open(frames[0]).convert("L")).astype(np.float64)
    last = np.asarray(Image.open(frames[-1]).convert("L")).astype(np.float64)
    return {
        "frames": len(frames),
        "frame_std_min": round(min(stds), 2),
        "blank_frames": sum(1 for s in stds if s < 8),
        "brightness_flicker_std": round(float(np.std(means)), 2),
        "adjacent_mean_abs": round(sum(adjacent) / max(len(adjacent), 1), 3),
        "first_last_mean_abs": round(float(np.abs(first - last).mean()), 3),
    }


def main() -> None:
    import torch

    triad = module_patch.load_export_patch(PATCH_DIR / "triad.json")
    pipe = module_patch._load_video()
    # untimed warmup: offload paging + T5 encode happen here
    module_patch.generate_video(pipe, PROMPT, OUT / "video_warmup", 7, steps=2)
    torch.cuda.reset_peak_memory_stats()

    results, timings = {}, []
    for name, patch in (("baseline", None), ("triad_total", triad)):
        folder = OUT / f"video_{name}"
        started = time.time()
        if patch is None:
            frames = module_patch.generate_video(pipe, PROMPT, folder, 7, steps=STEPS)
        else:
            handle = module_patch.ModulePatch(patch, mode="total")
            handle.attach(pipe.transformer, "transformer")
            try:
                frames = module_patch.generate_video(pipe, PROMPT, folder, 7, steps=STEPS)
            finally:
                handle.remove()
        seconds = round(time.time() - started, 2)
        timings.append({"name": name, "seconds": seconds})
        results[name] = clip_metrics(frames)
        print(f"{name}: {seconds}s {results[name]}")
    peak_gb = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    del pipe
    torch.cuda.empty_cache()

    report_path = OUT / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["video"] = {
        "timings": timings,
        "peak_gpu_gb": peak_gb,
        "clips": results,
        "prompt": PROMPT,
        "steps": STEPS,
    }
    report["performance"]["video_peak_gpu_gb"] = peak_gb
    report["performance"]["video_timings"] = timings
    base = results["baseline"]
    triad_clip = results["triad_total"]
    report["verdicts"]["consistency_video"] = (
        base["frames"] == triad_clip["frames"] == 9
        and base["blank_frames"] == triad_clip["blank_frames"] == 0
    )
    report["video"]["temporal_stability"] = {
        "baseline_adjacent_mean_abs": base["adjacent_mean_abs"],
        "triad_adjacent_mean_abs": triad_clip["adjacent_mean_abs"],
        "triad_first_last_mean_abs": triad_clip["first_last_mean_abs"],
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"updated {report_path}")


if __name__ == "__main__":
    main()
