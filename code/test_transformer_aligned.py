#!/usr/bin/env python3
"""valhalla-transformer aligned internal patch test for image and video.

Arms follow the Tier A protocol of the valhalla-transformer repo:
  * bodies reported separately: hub / tile / stemcell / triad
  * geometry tiers: legacy 0.08, full 0.02, total 0.015
  * controls: baseline (no patch), random same-geometry patch (seed 20260616),
    prompt-only (fact in the prompt, weights untouched)
  * fresh application per generation; hooks removed after each arm
  * per-arm change versus baseline plus injected delta norm

Patches come from the real export path (valhalla_model_export ExportDoc),
the same ValhallaModelPatch the transformer repo patches into Qwen.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bridge  # noqa: E402
import module_patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "media_pipeline" / "transformer_aligned"
PATCH_DIR = OUT / "patches"
PROMPT = "a square ceramic sign standing in a quiet room"
SEED = 7


def load_patches() -> dict[str, dict]:
    patches = {}
    for body in ("hub", "tile", "stemcell", "triad"):
        path = PATCH_DIR / f"{body}.json"
        if not path.is_file():
            raise RuntimeError(f"missing export patch {path}; run valhalla_model_export first")
        patches[body] = module_patch.load_export_patch(path)
    return patches


def image_arm(pipe, root, patch, mode, prompt, name):
    path = OUT / f"sdxl_{name}.png"
    if patch is None:
        module_patch.generate_image(pipe, prompt, path, SEED)
        return path, {"modules": 0, "fires": 0, "delta_norm": 0.0, "strength": 0.0}
    handle = module_patch.ModulePatch(patch, mode=mode)
    names = handle.attach(root, "unet")
    try:
        module_patch.generate_image(pipe, prompt, path, SEED)
    finally:
        stats = {
            "modules": len(names),
            "fires": handle.fires,
            "delta_norm": round(handle.delta_norm ** 0.5, 4),
            "strength": handle.strength,
        }
        handle.remove()
    return path, stats


def run_image(patches: dict) -> dict:
    pipe = module_patch._load_sdxl()
    root = pipe.unet
    arms = {}
    paths = {}

    def arm(name, patch, mode, prompt=PROMPT):
        path, stats = image_arm(pipe, root, patch, mode, prompt, name)
        paths[name] = path
        arms[name] = stats
        print(f"  sdxl {name}: modules={stats['modules']} fires={stats['fires']} delta={stats['delta_norm']}")

    arm("baseline", None, "legacy")
    arm("hub_legacy", patches["hub"], "legacy")
    arm("tile_legacy", patches["tile"], "legacy")
    arm("stemcell_legacy", patches["stemcell"], "legacy")
    arm("triad_legacy", patches["triad"], "legacy")
    arm("triad_legacy_repeat", patches["triad"], "legacy")
    arm("triad_full", patches["triad"], "full")
    arm("triad_total", patches["triad"], "total")
    arm("triad_random", module_patch.random_patch(patches["triad"]), "legacy")
    arm("prompt_only", None, "legacy", prompt=f"{PROMPT}, the passphrase is Qingniao")

    del pipe
    import torch

    torch.cuda.empty_cache()
    base = paths["baseline"]
    change = {name: round(module_patch.pixel_distance(base, path), 3) for name, path in paths.items()}
    repeat = module_patch.pixel_distance(paths["triad_legacy"], paths["triad_legacy_repeat"])
    return {
        "arms": arms,
        "change_vs_baseline": change,
        "triad_repeat_l2": round(repeat, 3),
        "fresh_reapply_deterministic": repeat < 0.5,
        "triad_changes_more_than_random": change["triad_legacy"] > change["triad_random"],
        "geometry_surface_grows": arms["triad_total"]["modules"] > arms["triad_full"]["modules"] > arms["triad_legacy"]["modules"],
        "geometry_change_grows": change["triad_total"] > change["triad_full"] > change["triad_legacy"],
    }


def video_arm(pipe, root, patch, mode, name):
    folder = OUT / f"video_{name}"
    if patch is None:
        module_patch.generate_video(pipe, PROMPT, folder, SEED)
        return folder, {"modules": 0, "fires": 0, "delta_norm": 0.0, "strength": 0.0}
    handle = module_patch.ModulePatch(patch, mode=mode)
    names = handle.attach(root, "transformer")
    try:
        module_patch.generate_video(pipe, PROMPT, folder, SEED)
    finally:
        stats = {
            "modules": len(names),
            "fires": handle.fires,
            "delta_norm": round(handle.delta_norm ** 0.5, 4),
            "strength": handle.strength,
        }
        handle.remove()
    return folder, stats


def run_video(patches: dict) -> dict:
    pipe = module_patch._load_video()
    root = pipe.transformer
    arms = {}
    folders = {}

    def arm(name, patch, mode):
        folder, stats = video_arm(pipe, root, patch, mode, name)
        folders[name] = folder
        arms[name] = stats
        print(f"  cogvideox {name}: modules={stats['modules']} fires={stats['fires']} delta={stats['delta_norm']}")

    arm("baseline", None, "legacy")
    arm("triad_legacy", patches["triad"], "legacy")
    arm("triad_total", patches["triad"], "total")
    arm("triad_random", module_patch.random_patch(patches["triad"]), "legacy")

    del pipe
    import torch

    torch.cuda.empty_cache()

    def frame_mean(folder_a: Path, folder_b: Path) -> float:
        values = []
        for index in range(9):
            a = folder_a / f"frame_{index:02d}.png"
            b = folder_b / f"frame_{index:02d}.png"
            if a.is_file() and b.is_file():
                values.append(module_patch.pixel_distance(a, b))
        return round(sum(values) / max(len(values), 1), 3)

    base = folders["baseline"]
    change = {name: frame_mean(base, folder) for name, folder in folders.items()}
    return {
        "arms": arms,
        "change_vs_baseline_mean_frame": change,
        "triad_changes_more_than_random": change["triad_legacy"] > change["triad_random"],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    patches = load_patches()
    print("bodies: " + ", ".join(
        f"{name}(tiles={p['tile_count']}, stems={p['stem_cell_count']})" for name, p in patches.items()
    ))
    print(f"sdxl arms, prompt={PROMPT!r}")
    image = run_image(patches)
    print(f"  change_vs_baseline={image['change_vs_baseline']}")
    print("cogvideox arms, 2 steps")
    video = run_video(patches)
    print(f"  change_vs_baseline={video['change_vs_baseline_mean_frame']}")

    passed = (
        all(p["dim"] == 256 for p in patches.values())
        and patches["tile"]["tile_count"] > 0
        and patches["stemcell"]["stem_cell_count"] > 0
        and all(s["modules"] > 0 and s["fires"] > 0 for s in image["arms"].values() if s["strength"] > 0)
        and all(s["modules"] > 0 and s["fires"] > 0 for s in video["arms"].values() if s["strength"] > 0)
        and image["fresh_reapply_deterministic"]
        and image["geometry_surface_grows"]
        and image["geometry_change_grows"]
        and image["change_vs_baseline"]["triad_legacy"] > 0.5
        and video["change_vs_baseline_mean_frame"]["triad_legacy"] > 0.5
    )
    report = {
        "protocol": "valhalla-transformer-tierA-media-v1",
        "reference_repo": "C:/Ai/rogue-products/valhalla-transformer",
        "mode_strength": module_patch.MODE_STRENGTH,
        "random_seed": module_patch.RANDOM_SEED,
        "prompt": PROMPT,
        "seed": SEED,
        "patches": {name: module_patch.dump_patch(p) for name, p in patches.items()},
        "image": image,
        "video": video,
        "verdict": "PASS" if passed else "FAIL",
    }
    path = OUT / "report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"verdict {report['verdict']}")
    print(f"wrote {path}")
    if not passed:
        sys.exit(2)


if __name__ == "__main__":
    main()
