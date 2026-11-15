#!/usr/bin/env python3
"""Bigger SDXL comparison: full SDXL Base and Juggernaut XL vs SDXL-Turbo.

Same ModulePatch (triad total), same prompts and seeds as the full-quality
run on the turbo model. Full models sample 25 steps at their native guidance
instead of the turbo 4-step schedule, so accuracy (sign text, hands) and
cross-seed consistency can be compared across model scale.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import download  # noqa: E402
import module_patch  # noqa: E402
from test_full_quality import ACCURACY_PROMPTS, SIGN, colour_probe, pairwise, stats_for  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "media_pipeline" / "big_models"
PATCH_DIR = ROOT / "reports" / "media_pipeline" / "transformer_aligned" / "patches"
TURBO_REPORT = ROOT / "reports" / "media_pipeline" / "full_quality" / "report.json"
SEEDS = (7, 8, 9)
STEPS = 25

MODELS = {
    "sdxl_base": download.IMAGE_BASE_DIR,
    "juggernaut_xl": download.IMAGE_JUG_DIR,
}


def load_pipe(model_dir: Path):
    import torch
    from diffusers import StableDiffusionXLPipeline

    pipe = StableDiffusionXLPipeline.from_pretrained(
        str(model_dir), torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def gen(pipe, prompt: str, out_path: Path, seed: int, size: int, patch=None, strength=None) -> float:
    import torch

    generator = torch.Generator(device="cuda").manual_seed(seed)
    started = time.time()
    handle = None
    if patch is not None:
        handle = module_patch.ModulePatch(patch, mode="total", strength=strength)
        handle.attach(pipe.unet, "unet")
    try:
        image = pipe(
            prompt=prompt,
            num_inference_steps=STEPS,
            guidance_scale=7.0,
            generator=generator,
            height=size,
            width=size,
        ).images[0]
    finally:
        if handle is not None:
            handle.remove()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    return round(time.time() - started, 2)


def run_model(name: str, model_dir: Path, triad: dict) -> dict:
    import torch

    pipe = load_pipe(model_dir)
    torch.cuda.reset_peak_memory_stats()
    folder = OUT / name
    timings = {}
    consistency_paths = {"baseline": [], "triad_total": []}
    for seed in SEEDS:
        base = folder / f"cons_base_s{seed}.png"
        timings[f"cons_base_s{seed}"] = gen(pipe, SIGN, base, seed, 768)
        consistency_paths["baseline"].append(base)
        patched = folder / f"cons_triad_s{seed}.png"
        timings[f"cons_triad_s{seed}"] = gen(pipe, SIGN, patched, seed, 768, patch=triad)
        consistency_paths["triad_total"].append(patched)
    accuracy = {}
    for key, prompt in ACCURACY_PROMPTS.items():
        base = folder / f"acc_{key}_base.png"
        tb = gen(pipe, prompt, base, 7, 1024)
        patched = folder / f"acc_{key}_triad.png"
        tt = gen(pipe, prompt, patched, 7, 1024, patch=triad)
        entry = {
            "prompt": prompt,
            "baseline": {**stats_for(base), "seconds": tb},
            "triad_total": {**stats_for(patched), "seconds": tt},
        }
        if key in ("cube", "cat"):
            entry["baseline"]["colours"] = colour_probe(base)
            entry["triad_total"]["colours"] = colour_probe(patched)
        accuracy[key] = entry
        print(f"  {name} {key}: base {tb}s triad {tt}s")
    peak = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    del pipe
    torch.cuda.empty_cache()
    base_pair = pairwise(consistency_paths["baseline"])
    triad_pair = pairwise(consistency_paths["triad_total"])
    return {
        "model_dir": str(model_dir),
        "steps": STEPS,
        "peak_gpu_gb": peak,
        "timings": timings,
        "consistency": {
            "baseline_pairwise_l2": base_pair,
            "triad_total_pairwise_l2": triad_pair,
            "ratio_triad_over_baseline": round(triad_pair / max(base_pair, 1e-9), 3),
        },
        "accuracy": accuracy,
    }


def turbo_reference() -> dict:
    if not TURBO_REPORT.is_file():
        return {}
    report = json.loads(TURBO_REPORT.read_text(encoding="utf-8"))
    image = report.get("image", {})
    return {
        "model": "sdxl-turbo (4 steps, guidance 0)",
        "consistency": image.get("consistency", {}),
        "accuracy_stats": {
            key: {
                side: {k: entry[side][k] for k in ("sharpness", "std", "clipped_frac") if k in entry[side]}
                for side in ("baseline", "triad_total")
            }
            for key, entry in image.get("accuracy", {}).items()
        },
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    triad = module_patch.load_export_patch(PATCH_DIR / "triad.json")
    results = {"turbo_reference": turbo_reference()}
    for name, model_dir in MODELS.items():
        if not (model_dir / "model_index.json").is_file():
            print(f"skip {name}: {model_dir} not downloaded")
            continue
        print(f"model {name} ({STEPS} steps)")
        results[name] = run_model(name, model_dir, triad)
        print(f"  consistency {results[name]['consistency']}")
    path = OUT / "report.json"
    path.write_text(json.dumps({"protocol": "valhalla-media-big-models-v1", "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
