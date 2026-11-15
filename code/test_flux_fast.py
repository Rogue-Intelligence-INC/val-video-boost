#!/usr/bin/env python3
"""FLUX.1-schnell, fast path + adaptive Valhalla patch (reduced scope).

enable_model_cpu_offload swaps the whole 24GB transformer per sampling
step and measured 577s for one 1024px image. This script instead encodes
prompts with the text encoders on GPU once, then streams the transformer
block-by-block (group offload), which keeps per-image time in the tens of
seconds. It reuses reports/media_pipeline/flux/probe_baseline.png from the
slow run and adds the adaptive-strength arms.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import download  # noqa: E402
import module_patch  # noqa: E402
from test_full_quality import SIGN, stats_for  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "media_pipeline" / "flux"
PATCH_DIR = ROOT / "reports" / "media_pipeline" / "transformer_aligned" / "patches"
STEPS = 4


def load_fast():
    import torch
    from diffusers import FluxPipeline
    from diffusers.hooks import apply_group_offloading

    try:  # framework GPU-accel switches (TF32 etc); loader stays as measured
        import adaptive_framework as _af

        _af.enable_gpu_accel()
    except Exception:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    pipe = FluxPipeline.from_pretrained(str(download.FLUX_DIR), torch_dtype=torch.bfloat16)
    apply_group_offloading(
        pipe.transformer,
        onload_device="cuda",
        offload_device="cpu",
        offload_type="block_level",
        num_blocks_per_group=1,
        use_stream=True,
    )
    pipe.vae.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def encode(pipe, prompt: str):
    import torch

    pipe.text_encoder.to("cuda")
    pipe.text_encoder_2.to("cuda")
    try:
        with torch.no_grad():
            embeds, pooled, _ids = pipe.encode_prompt(
                prompt=prompt, prompt_2=None, device="cuda", max_sequence_length=256
            )
    finally:
        pipe.text_encoder.to("cpu")
        pipe.text_encoder_2.to("cpu")
        torch.cuda.empty_cache()
    return embeds, pooled


def gen(pipe, prompt: str, out_path: Path, seed: int, patch=None, strength=None) -> float:
    import torch

    embeds, pooled = encode(pipe, prompt)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    started = time.time()
    handle = None
    if patch is not None:
        handle = module_patch.ModulePatch(patch, mode="total", strength=strength)
        handle.attach(pipe.transformer, "transformer")
    try:
        image = pipe(
            prompt_embeds=embeds,
            pooled_prompt_embeds=pooled,
            num_inference_steps=STEPS,
            guidance_scale=0.0,
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
    pipe = load_fast()
    profile = module_patch.profile_denoiser(pipe.transformer, "transformer")
    plan = module_patch.adaptive_plan(profile, triad["body_source"], "total", steps=STEPS)
    chosen = plan["recommended_strength"]
    print(f"profile {profile}")
    print(f"plan {plan}")

    baseline_sign = OUT / "probe_baseline.png"
    adaptive_sign = OUT / "adaptive_sign.png"
    t_sign = gen(pipe, SIGN, adaptive_sign, 7, patch=triad, strength=chosen)
    cross = round(module_patch.pixel_distance(baseline_sign, adaptive_sign), 3)
    print(f"adaptive sign {t_sign}s cross_patch_l2={cross}")

    hand = "a human hand with five fingers, palm facing the camera"
    base_hand = OUT / "acc_hand_baseline.png"
    adapt_hand = OUT / "acc_hand_adaptive.png"
    t_base = gen(pipe, hand, base_hand, 7)
    t_adapt = gen(pipe, hand, adapt_hand, 7, patch=triad, strength=chosen)
    print(f"hand baseline {t_base}s adaptive {t_adapt}s")

    portrait = "portrait of a woman smiling, studio light, facing camera"
    adapt_portrait = OUT / "acc_portrait_adaptive.png"
    t_portrait = gen(pipe, portrait, adapt_portrait, 7, patch=triad, strength=chosen)
    print(f"portrait adaptive {t_portrait}s")

    import torch

    del pipe
    torch.cuda.empty_cache()

    report = {
        "protocol": "valhalla-flux-schnell-v1",
        "model": download.FLUX_REPO,
        "load_strategy": "group_offload block_level + encoders on GPU only for encode_prompt",
        "slow_path_reference": {"strategy": "enable_model_cpu_offload", "baseline_sign_seconds": 577.43},
        "steps": STEPS,
        "profile": profile,
        "plan": plan,
        "chosen_strength": chosen,
        "fixed_strength_reference": module_patch.MODE_STRENGTH["total"],
        "sign": {
            "baseline_seconds_slow_path": 577.43,
            "adaptive_seconds": t_sign,
            "cross_patch_l2_vs_baseline": cross,
            "baseline_stats": stats_for(baseline_sign),
            "adaptive_stats": stats_for(adaptive_sign),
        },
        "hand": {
            "baseline_seconds": t_base,
            "adaptive_seconds": t_adapt,
            "baseline_stats": stats_for(base_hand),
            "adaptive_stats": stats_for(adapt_hand),
        },
        "portrait": {"adaptive_seconds": t_portrait, "adaptive_stats": stats_for(adapt_portrait)},
        "verdicts": {
            "profile_found_blocks": profile["block_count"] == 57,
            "patch_reaches_flux_blocks": cross > 0.5,
            "distortion_guard": all(
                (not s["blank"]) and s["clipped_frac"] < 0.35 and s["sharpness"] > 10
                for s in (stats_for(adaptive_sign), stats_for(adapt_hand), stats_for(adapt_portrait))
            ),
        },
        "notes": "Cross-seed consistency suite deferred: even the fast path is far slower than SDXL; the slow path measured 577s per image.",
    }
    path = OUT / "report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["verdicts"], ensure_ascii=False))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
