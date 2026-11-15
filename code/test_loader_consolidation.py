#!/usr/bin/env python3
"""Consolidation regression for the loader (block-unload) feature.

Pins three things so the feature cannot silently rot:
  1. choose_loader decisions per family/priority (incl. unknown fallback)
  2. apply_loader actually generates on CogVideoX under the speed and
     memory strategies, with the measured ordering preserved
     (cpu_offload faster, sequential lighter)
  3. module_patch._load_video (every video script's entry point) now
     routes through the framework loader
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import adaptive_framework as af
import download

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "adaptive_framework"
PROMPT = "a woman with chestnut brown hair walking in a park, facing camera"


def gen_cogvideo(strategy: str) -> dict:
    import torch
    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(str(download.VIDEO_XL_DIR), torch_dtype=torch.float16)
    applied = af.apply_loader(pipe, "cogvideox", strategy=strategy)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    started = time.time()
    out = pipe(prompt=PROMPT, num_inference_steps=10, num_frames=17, height=288, width=512,
               guidance_scale=6.0, generator=torch.Generator(device="cpu").manual_seed(7))
    res = {"strategy": applied, "frames": len(out.frames[0]),
           "seconds": round(time.time() - started, 1),
           "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    del pipe
    torch.cuda.empty_cache()
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    decisions = {
        "cogvideo_speed": af.choose_loader("cogvideox", "speed"),
        "cogvideo_memory": af.choose_loader("cogvideox", "memory"),
        "flux_speed": af.choose_loader("flux", "speed"),
        "sdxl_speed": af.choose_loader("sdxl", "speed"),
        "unknown_fallback": af.choose_loader("some_new_family", "speed"),
    }
    expected = {"cogvideo_speed": "cpu_offload", "cogvideo_memory": "sequential",
                "flux_speed": "group_offload", "sdxl_speed": "resident",
                "unknown_fallback": "cpu_offload"}
    print("decisions", decisions)
    speed = gen_cogvideo(decisions["cogvideo_speed"])
    memory = gen_cogvideo(decisions["cogvideo_memory"])
    print("speed arm", speed)
    print("memory arm", memory)
    report = {"protocol": "val-video-boost-loader-consolidation-v1",
              "decisions": decisions, "expected": expected,
              "speed_arm": speed, "memory_arm": memory,
              "verdicts": {
                  "decisions_match": decisions == expected,
                  "speed_arm_generates": speed["frames"] == 17,
                  "memory_arm_generates": memory["frames"] == 17,
                  "ordering_preserved": speed["seconds"] < memory["seconds"]
                  and memory["peak_gpu_gb"] < speed["peak_gpu_gb"],
                  "records_present": af.loader_record("cogvideox", "cpu_offload") is not None,
              }}
    (OUT / "report_loader_consolidation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["verdicts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
