#!/usr/bin/env python3
"""Block-unload strategies on the models themselves (simple test).

Same task per model, three residency strategies:
  resident        everything on GPU
  cpu_offload     diffusers model-level offload (current framework choice)
  group/stream    block-level group offloading (FLUX) / sequential (video)
Measured: wall time and peak GPU memory. Known from before: unloading is
not free -- CogVideoX 49f@720p cpu_offload thrashed to 50-200s/step, and
FLUX full-resident does not fit at all (that arm is recorded as OOM by
design rather than attempted blindly twice).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import download

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "adaptive_framework"
PROMPT_V = "a woman with chestnut brown hair walking in a park, facing camera, steady camera"


def peak_gb(torch):
    return round(torch.cuda.max_memory_allocated() / 1e9, 2)


def cogvideo_arm(strategy: str) -> dict:
    import torch
    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(str(download.VIDEO_XL_DIR), torch_dtype=torch.float16)
    pipe.set_progress_bar_config(disable=True)
    if strategy == "resident":
        pipe = pipe.to("cuda")
    elif strategy == "cpu_offload":
        pipe.enable_model_cpu_offload()
    elif strategy == "sequential":
        pipe.enable_sequential_cpu_offload()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    try:
        out = pipe(prompt=PROMPT_V, num_inference_steps=10, num_frames=17,
                   height=288, width=512, guidance_scale=6.0,
                   generator=torch.Generator(device="cpu").manual_seed(7))
        ok, err = True, None
        n = len(out.frames[0])
    except Exception as e:  # OOM is a measured outcome, not a crash
        ok, err, n = False, str(e)[:200], 0
    res = {"strategy": strategy, "ok": ok, "error": err, "frames": n,
           "seconds": round(time.time() - started, 1), "peak_gpu_gb": peak_gb(torch)}
    del pipe
    torch.cuda.empty_cache()
    return res


def flux_arm(group_blocks: int) -> dict:
    import torch
    from diffusers import FluxPipeline
    from diffusers.hooks import apply_group_offloading

    pipe = FluxPipeline.from_pretrained(str(download.FLUX_DIR), torch_dtype=torch.bfloat16)
    apply_group_offloading(pipe.transformer, onload_device="cuda", offload_device="cpu",
                           offload_type="block_level", num_blocks_per_group=group_blocks,
                           use_stream=True)
    pipe.vae.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    pipe.text_encoder.to("cuda"); pipe.text_encoder_2.to("cuda")
    with torch.no_grad():
        embeds, pooled, _ = pipe.encode_prompt(prompt="portrait of a woman, studio light",
                                               prompt_2=None, device="cuda", max_sequence_length=256)
    pipe.text_encoder.to("cpu"); pipe.text_encoder_2.to("cpu")
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    started = time.time()
    try:
        img = pipe(prompt_embeds=embeds, pooled_prompt_embeds=pooled, num_inference_steps=4,
                   guidance_scale=0.0, generator=torch.Generator(device="cpu").manual_seed(7),
                   height=1024, width=1024).images[0]
        img.save(OUT / f"flux_unload_g{group_blocks}.png")
        ok, err = True, None
    except Exception as e:
        ok, err = False, str(e)[:200]
    res = {"strategy": f"group_offload_x{group_blocks}", "ok": ok, "error": err,
           "seconds": round(time.time() - started, 1), "peak_gpu_gb": peak_gb(torch)}
    del pipe
    torch.cuda.empty_cache()
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"protocol": "val-video-boost-unload-strategies-v1",
              "task_video": "CogVideoX-2b 17f 512x288 10 steps",
              "task_flux": "FLUX 1024x1024 4 steps", "video": [], "flux": []}
    for s in ("resident", "cpu_offload", "sequential"):
        r = cogvideo_arm(s)
        report["video"].append(r)
        print("cogvideo", r)
    for g in (1, 4):
        r = flux_arm(g)
        report["flux"].append(r)
        print("flux", r)
    ok_v = [r for r in report["video"] if r["ok"]]
    if ok_v:
        fastest = min(ok_v, key=lambda r: r["seconds"])
        lightest = min(ok_v, key=lambda r: r["peak_gpu_gb"])
        report["verdicts"] = {"video_all_strategies_completed": len(ok_v) == 3,
                              "video_fastest": fastest["strategy"],
                              "video_lightest": lightest["strategy"]}
    (OUT / "report_unload.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report.get("verdicts"), ensure_ascii=False))


if __name__ == "__main__":
    main()
