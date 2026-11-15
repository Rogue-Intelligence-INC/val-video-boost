#!/usr/bin/env python3
"""FLUX.1-schnell with the adaptive Valhalla module patch.

FLUX is a different architecture family (MM-DiT: 19 dual-stream
FluxTransformerBlock + 38 single-stream blocks, T5-XXL text encoder), so
this exercises profile_denoiser / adaptive_plan / ModulePatch outside the
SDXL UNet they were built on. Arms: baseline, fixed total 0.015, and the
probe-calibrated adaptive strength. Prompts reuse the sign / hand /
portrait / cat set from the full-quality run for cross-model comparison.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import download  # noqa: E402
import module_patch  # noqa: E402
from test_full_quality import SIGN, pairwise, stats_for  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "media_pipeline" / "flux"
PATCH_DIR = ROOT / "reports" / "media_pipeline" / "transformer_aligned" / "patches"
SEEDS = (7, 8, 9)
STEPS = 4
TARGET_L2 = 3.0

PROMPTS = {
    "sign": SIGN,
    "hand": "a human hand with five fingers, palm facing the camera",
    "portrait": "portrait of a woman smiling, studio light, facing camera",
    "cat": "an orange cat sitting on green grass",
}


def load_flux():
    import torch
    from diffusers import FluxPipeline

    pipe = FluxPipeline.from_pretrained(
        str(download.FLUX_DIR), torch_dtype=torch.bfloat16
    )
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def gen(pipe, prompt: str, out_path: Path, seed: int, patch=None, strength=None) -> float:
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    started = time.time()
    handle = None
    if patch is not None:
        handle = module_patch.ModulePatch(patch, mode="total", strength=strength)
        handle.attach(pipe.transformer, "transformer")
    try:
        image = pipe(
            prompt=prompt,
            num_inference_steps=STEPS,
            guidance_scale=0.0,
            max_sequence_length=256,
            generator=generator,
            height=1024,
            width=1024,
        ).images[0]
    finally:
        if handle is not None:
            handle.remove()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    return round(time.time() - started, 2)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    triad = module_patch.load_export_patch(PATCH_DIR / "triad.json")
    pipe = load_flux()

    profile = module_patch.profile_denoiser(pipe.transformer, "transformer")
    plan = module_patch.adaptive_plan(profile, triad["body_source"], "total", steps=STEPS)
    print(f"profile {profile}")
    print(f"plan {plan}")

    baseline = OUT / "probe_baseline.png"
    print(f"baseline {gen(pipe, SIGN, baseline, 7)}s")
    probes = []
    chosen = None
    for strength in plan["probe_ladder"]:
        path = OUT / f"probe_s{strength}.png"
        seconds = gen(pipe, SIGN, path, 7, patch=triad, strength=strength)
        distance = round(module_patch.pixel_distance(baseline, path), 3)
        stats = stats_for(path)
        ok = distance >= TARGET_L2 and stats["clipped_frac"] < 0.35 and stats["sharpness"] > 10 and not stats["blank"]
        probes.append({"strength": strength, "l2_vs_baseline": distance, "seconds": seconds, "in_band": ok, **stats})
        print(f"  probe {strength}: l2={distance} sharp={stats['sharpness']} {seconds}s")
        if ok and chosen is None:
            chosen = strength
    if chosen is None:
        chosen = plan["probe_ladder"][-1]
    print(f"chosen adaptive strength {chosen} (fixed reference {module_patch.MODE_STRENGTH['total']})")

    arms = {"baseline": [], "fixed_total": [], "adaptive": []}
    timings = {}
    for seed in SEEDS:
        for arm, patch, strength in (
            ("baseline", None, None),
            ("fixed_total", triad, None),
            ("adaptive", triad, chosen),
        ):
            path = OUT / f"cons_{arm}_s{seed}.png"
            timings[f"{arm}_s{seed}"] = gen(pipe, SIGN, path, seed, patch=patch, strength=strength)
            arms[arm].append(path)
    consistency = {
        arm: pairwise(paths) for arm, paths in arms.items()
    }
    consistency["ratio_fixed_over_baseline"] = round(consistency["fixed_total"] / max(consistency["baseline"], 1e-9), 3)
    consistency["ratio_adaptive_over_baseline"] = round(consistency["adaptive"] / max(consistency["baseline"], 1e-9), 3)
    print(f"consistency {consistency}")

    quality = {}
    for key, prompt in PROMPTS.items():
        entry = {}
        for arm, patch, strength in (("baseline", None, None), ("adaptive", triad, chosen)):
            path = OUT / f"acc_{key}_{arm}.png"
            seconds = gen(pipe, prompt, path, 7, patch=patch, strength=strength)
            entry[arm] = {**stats_for(path), "seconds": seconds}
        quality[key] = {"prompt": prompt, **entry}
        print(f"quality {key}: base {entry['baseline']['seconds']}s adaptive {entry['adaptive']['seconds']}s")

    import torch

    peak = round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None
    del pipe
    torch.cuda.empty_cache()

    report = {
        "protocol": "valhalla-flux-schnell-v1",
        "model": download.FLUX_REPO,
        "steps": STEPS,
        "profile": profile,
        "plan": plan,
        "probes": probes,
        "chosen_strength": chosen,
        "consistency": consistency,
        "quality": quality,
        "timings": timings,
        "peak_gpu_gb": peak,
        "verdicts": {
            "profile_found_blocks": profile["block_count"] > 0,
            "probe_found_strength": any(p["in_band"] for p in probes),
            "adaptive_consistency_gain": consistency["ratio_adaptive_over_baseline"] < 1.0,
            "distortion_guard": all(
                (not entry[arm]["blank"]) and entry[arm]["clipped_frac"] < 0.35 and entry[arm]["sharpness"] > 10
                for entry in quality.values() for arm in ("baseline", "adaptive")
            ),
        },
    }
    path = OUT / "report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["verdicts"], ensure_ascii=False))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
