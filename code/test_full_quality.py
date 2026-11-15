#!/usr/bin/env python3
"""Full quality test: health, consistency, accuracy, distortion, video, speed.

Categories (each judged on its own, valhalla-transformer style):
  health      valhalla_base native + lm_patch fact, export patches, GPU stack
  consistency same triad patch across seeds vs baseline across seeds (SDXL),
              and frame-to-frame stability inside CogVideoX clips
  accuracy    colour-dominance probes for colour prompts; sign/cat/person/hand
              prompts are saved for visual inspection and scored by stats here
  distortion  blank / clipping / sharpness guards on every generated image
  video       frame count, blank frames, temporal flicker, motion range
  performance load time, per-sample time, hook overhead, peak GPU memory
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bridge  # noqa: E402
import module_patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "media_pipeline" / "full_quality"
PATCH_DIR = ROOT / "reports" / "media_pipeline" / "transformer_aligned" / "patches"
SIGN = "a square ceramic sign standing in a quiet room"
SEEDS = (7, 8, 9)

ACCURACY_PROMPTS = {
    "cube": "a red cube on a blue table, studio photo",
    "cat": "an orange cat sitting on green grass",
    "sign": SIGN,
    "portrait": "portrait of a woman smiling, studio light, facing camera",
    "hand": "a human hand with five fingers, palm facing the camera",
}


def stats_for(path: Path) -> dict:
    import numpy as np
    from PIL import Image

    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float64)
    gray = arr.mean(axis=2)
    lap = (
        -4 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    )
    hsv = np.asarray(img.convert("HSV")).astype(np.float64)
    return {
        "size": list(img.size),
        "mean": round(float(arr.mean()), 2),
        "std": round(float(arr.std()), 2),
        "sharpness": round(float(lap.var()), 1),
        "clipped_frac": round(float(((arr <= 2) | (arr >= 253)).mean()), 4),
        "saturation": round(float(hsv[..., 1].mean()), 1),
        "blank": bool(arr.std() < 8),
    }


def colour_probe(path: Path) -> dict:
    """Dominant-colour shares (HSV bands) for the colour accuracy prompts."""
    import numpy as np
    from PIL import Image

    hsv = np.asarray(Image.open(path).convert("RGB").convert("HSV")).astype(np.float64)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    vivid = (sat > 70) & (val > 50)

    def share(mask) -> float:
        return round(float((mask & vivid).mean()), 4)

    return {
        "red": share((hue < 12) | (hue > 168)),
        "orange": share((hue >= 12) & (hue < 28)),
        "green": share((hue >= 35) & (hue < 95)),
        "blue": share((hue >= 95) & (hue < 135)),
    }


def pairwise(paths: list[Path]) -> float:
    values = []
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            values.append(module_patch.pixel_distance(paths[i], paths[j]))
    return round(sum(values) / max(len(values), 1), 3)


def health() -> dict:
    started = time.time()
    body = bridge.valhalla(
        bridge.build_payload("What is the passphrase?", {}, None, [{"text": bridge.FACT}], None, bridge.CFG)
    )
    native_seconds = time.time() - started
    row = (body.get("rows") or [{}])[0]
    session = body.get("session") or {}

    pipe = bridge.ValhallaPipeline()
    try:
        prompt = (
            "Retrieved evidence:\n" + bridge.FACT + "\nBase the answer on this evidence. "
            "Do not replace it with a different fact.\n\nUser: What is the passphrase?"
        )
        started = time.time()
        lm = pipe.ask(prompt, {}, session, None, None, timeout=300)
        lm_seconds = time.time() - started
    finally:
        pipe.close()
    lm_row = (lm.get("rows") or [{}])[0]
    patches = {
        body_name: module_patch.load_export_patch(PATCH_DIR / f"{body_name}.json")
        for body_name in ("hub", "tile", "stemcell", "triad")
    }
    return {
        "native": {
            "protocol": body.get("protocol"),
            "decode_path": row.get("decode_path", ""),
            "fingerprint": session.get("fingerprint", ""),
            "seconds": round(native_seconds, 2),
            "ok": body.get("protocol") == "valhalla-base-v1" and "pipeline_v4" in row.get("decode_path", ""),
        },
        "lm_patch": {
            "answer": lm_row.get("answer", ""),
            "decode_path": lm_row.get("decode_path", ""),
            "seconds": round(lm_seconds, 1),
            "fact_held": "Qingniao" in lm_row.get("answer", ""),
            "ok": "lm_patch" in lm_row.get("decode_path", "") and "Qingniao" in lm_row.get("answer", ""),
        },
        "patches": {name: module_patch.dump_patch(p) for name, p in patches.items()},
        "patches_ok": all(p["dim"] == 256 for p in patches.values()),
    }


def image_suite(patches: dict) -> dict:
    import torch

    triad = patches["triad"]
    pipe = module_patch._load_sdxl()
    torch.cuda.reset_peak_memory_stats()
    timings = []
    paths = {"baseline": {}, "triad_total": {}}

    def gen(name, prompt, seed, patch):
        target = OUT / f"{name}.png"
        started = time.time()
        if patch is None:
            module_patch.generate_image(pipe, prompt, target, seed)
        else:
            handle = module_patch.ModulePatch(patch, mode="total")
            handle.attach(pipe.unet, "unet")
            try:
                module_patch.generate_image(pipe, prompt, target, seed)
            finally:
                handle.remove()
        timings.append({"name": name, "seconds": round(time.time() - started, 2)})
        return target

    # consistency across seeds
    for seed in SEEDS:
        paths["baseline"][seed] = gen(f"cons_base_s{seed}", SIGN, seed, None)
        paths["triad_total"][seed] = gen(f"cons_triad_s{seed}", SIGN, seed, triad)
    # accuracy / distortion prompts, seed 7
    accuracy = {}
    for key, prompt in ACCURACY_PROMPTS.items():
        base = gen(f"acc_{key}_base", prompt, 7, None)
        patched = gen(f"acc_{key}_triad", prompt, 7, triad)
        accuracy[key] = {
            "prompt": prompt,
            "baseline": {"path": str(base), **stats_for(base)},
            "triad_total": {"path": str(patched), **stats_for(patched)},
        }
        if key in ("cube", "cat"):
            accuracy[key]["baseline"]["colours"] = colour_probe(base)
            accuracy[key]["triad_total"]["colours"] = colour_probe(patched)

    peak_gb = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    del pipe
    torch.cuda.empty_cache()

    base_pair = pairwise([paths["baseline"][s] for s in SEEDS])
    triad_pair = pairwise([paths["triad_total"][s] for s in SEEDS])
    all_stats = [entry[side] for entry in accuracy.values() for side in ("baseline", "triad_total")]
    return {
        "timings": timings,
        "peak_gpu_gb": peak_gb,
        "consistency": {
            "baseline_pairwise_l2": base_pair,
            "triad_total_pairwise_l2": triad_pair,
            "ratio_triad_over_baseline": round(triad_pair / max(base_pair, 1e-9), 3),
            "patch_more_consistent": triad_pair < base_pair,
        },
        "accuracy": accuracy,
        "distortion": {
            "blank_any": any(s["blank"] for s in all_stats),
            "max_clipped_frac": max(s["clipped_frac"] for s in all_stats),
            "min_sharpness": min(s["sharpness"] for s in all_stats),
        },
    }


def video_suite(patches: dict) -> dict:
    import numpy as np
    import torch
    from PIL import Image

    triad = patches["triad"]
    pipe = module_patch._load_video()
    torch.cuda.reset_peak_memory_stats()
    prompt = "a square ceramic sign in a quiet room, slow steady camera"
    results = {}
    timings = []
    for name, patch in (("baseline", None), ("triad_total", triad)):
        folder = OUT / f"video_{name}"
        started = time.time()
        if patch is None:
            frames = module_patch.generate_video(pipe, prompt, folder, 7, steps=6)
        else:
            handle = module_patch.ModulePatch(patch, mode="total")
            handle.attach(pipe.transformer, "transformer")
            try:
                frames = module_patch.generate_video(pipe, prompt, folder, 7, steps=6)
            finally:
                handle.remove()
        timings.append({"name": name, "seconds": round(time.time() - started, 2)})
        means, stds, adjacent = [], [], []
        previous = None
        for frame in frames:
            arr = np.asarray(Image.open(frame).convert("L")).astype(np.float64)
            means.append(float(arr.mean()))
            stds.append(float(arr.std()))
            if previous is not None:
                adjacent.append(float(np.abs(arr - previous).mean()))
            previous = arr
        results[name] = {
            "frames": len(frames),
            "frame_std_min": round(min(stds), 2),
            "blank_frames": sum(1 for s in stds if s < 8),
            "brightness_flicker_std": round(float(np.std(means)), 2),
            "adjacent_mean_abs": round(sum(adjacent) / max(len(adjacent), 1), 2),
            "first_last_mean_abs": round(float(np.abs(
                np.asarray(Image.open(frames[0]).convert("L")).astype(np.float64)
                - np.asarray(Image.open(frames[-1]).convert("L")).astype(np.float64)
            ).mean()), 2),
        }
    peak_gb = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    del pipe
    torch.cuda.empty_cache()
    return {"timings": timings, "peak_gpu_gb": peak_gb, "clips": results, "prompt": prompt}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("health: valhalla native + lm_patch + patches")
    health_result = health()
    print(f"  native ok={health_result['native']['ok']} lm_patch ok={health_result['lm_patch']['ok']} "
          f"answer={health_result['lm_patch']['answer'][:80]!r}")
    patches = {
        name: module_patch.load_export_patch(PATCH_DIR / f"{name}.json")
        for name in ("hub", "tile", "stemcell", "triad")
    }
    print("image suite: consistency seeds + accuracy prompts")
    image = image_suite(patches)
    print(f"  pairwise baseline={image['consistency']['baseline_pairwise_l2']} "
          f"triad={image['consistency']['triad_total_pairwise_l2']} "
          f"ratio={image['consistency']['ratio_triad_over_baseline']}")
    print("video suite: baseline + triad_total")
    video = video_suite(patches)
    for name, clip in video["clips"].items():
        print(f"  {name}: {clip}")

    patched_times = [t["seconds"] for t in image["timings"] if "triad" in t["name"]]
    base_times = [t["seconds"] for t in image["timings"] if "triad" not in t["name"]]
    overhead = round(
        (sum(patched_times) / max(len(patched_times), 1)) / max(sum(base_times) / max(len(base_times), 1), 1e-9) - 1,
        3,
    )
    cube = image["accuracy"]["cube"]
    cat = image["accuracy"]["cat"]
    colour_ok = (
        cube["baseline"]["colours"]["red"] > 0.02 and cube["triad_total"]["colours"]["red"] > 0.02
        and cat["baseline"]["colours"]["orange"] > 0.02 and cat["triad_total"]["colours"]["orange"] > 0.02
    )
    verdicts = {
        "health": health_result["native"]["ok"] and health_result["lm_patch"]["ok"] and health_result["patches_ok"],
        "consistency_image": image["consistency"]["patch_more_consistent"],
        "consistency_video": all(
            c["frames"] == 9 and c["blank_frames"] == 0 for c in video["clips"].values()
        ),
        "accuracy_colour": colour_ok,
        "distortion_guard": (
            not image["distortion"]["blank_any"]
            and image["distortion"]["max_clipped_frac"] < 0.35
            and image["distortion"]["min_sharpness"] > 5
        ),
        "performance": overhead < 0.5,
    }
    report = {
        "protocol": "valhalla-media-full-quality-v1",
        "health": health_result,
        "image": image,
        "video": video,
        "performance": {
            "image_hook_overhead_ratio": overhead,
            "image_peak_gpu_gb": image["peak_gpu_gb"],
            "video_peak_gpu_gb": video["peak_gpu_gb"],
            "image_timings": image["timings"],
            "video_timings": video["timings"],
        },
        "verdicts": verdicts,
    }
    path = OUT / "report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(verdicts, ensure_ascii=False))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
