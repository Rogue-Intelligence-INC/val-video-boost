#!/usr/bin/env python3
"""Character (person) identity consistency.

Same identity lock across seeds and scenes, baseline vs Valhalla triad
total patch. A second identity is the negative control: if the metric
cannot tell two described people apart, the numbers are not identity.

Face-crop L2 (upper-center 50% of the frame) is the identity score;
full-frame L2 is reported as a scene-sensitive companion.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import download  # noqa: E402
import module_patch  # noqa: E402
from test_big_models import gen, load_pipe  # noqa: E402
from test_full_quality import pairwise, stats_for  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "media_pipeline" / "character"
PATCH_DIR = ROOT / "reports" / "media_pipeline" / "transformer_aligned" / "patches"
SEEDS = (7, 8, 9)

ID_A = (
    "a 28-year-old woman with shoulder-length chestnut brown hair, "
    "a small mole on her left cheek, hazel eyes, fair skin, "
    "looking at the camera"
)
ID_B = (
    "a 45-year-old man with a short grey beard, receding hairline, "
    "dark brown eyes, olive skin, looking at the camera"
)
SCENES = {
    "studio": "close-up portrait, studio light, plain grey background, facing camera",
    "outdoor": "standing outdoors in a sunlit park, natural daylight, waist-up",
    "profile": "three-quarter view looking slightly to her right, studio light",
}


def prompt_for(identity: str, scene: str) -> str:
    return f"photorealistic photograph of {identity}, {SCENES[scene]}"


def face_crop(path: Path, size: int = 128):
    from PIL import Image

    img = Image.open(path).convert("RGB")
    w, h = img.size
    left = int(w * 0.25)
    right = int(w * 0.75)
    top = int(h * 0.08)
    bottom = int(h * 0.58)
    return img.crop((left, top, right, bottom)).resize((size, size))


def face_distance(path_a: Path, path_b: Path) -> float:
    left = face_crop(path_a)
    right = face_crop(path_b)
    pairs = zip(left.getdata(), right.getdata())
    total = 0.0
    count = 0
    for a, b in pairs:
        for channel in range(3):
            delta = a[channel] - b[channel]
            total += delta * delta
            count += 1
    return (total / max(count, 1)) ** 0.5


def pairwise_face(paths: list[Path]) -> float:
    values = []
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            values.append(face_distance(paths[i], paths[j]))
    return round(sum(values) / max(len(values), 1), 3)


def save_crop(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    face_crop(src, size=256).save(dest)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    triad = module_patch.load_export_patch(PATCH_DIR / "triad.json")
    pipe = load_pipe(download.IMAGE_JUG_DIR)

    paths: dict[str, dict[str, list[Path]]] = {}
    timings = {}

    def shoot(arm: str, identity_key: str, identity: str, scene: str, seed: int, patch):
        name = f"{arm}_{identity_key}_{scene}_s{seed}"
        target = OUT / f"{name}.png"
        seconds = gen(pipe, prompt_for(identity, scene), target, seed, 768, patch=patch)
        timings[name] = seconds
        save_crop(target, OUT / "crops" / f"{name}.png")
        print(f"  {name} {seconds}s")
        return target

    print("Juggernaut XL character suite")
    for arm, patch in (("baseline", None), ("triad_total", triad)):
        paths[arm] = {scene: [] for scene in SCENES}
        paths[arm]["id_b_studio"] = []
        for scene in SCENES:
            for seed in SEEDS:
                paths[arm][scene].append(shoot(arm, "idA", ID_A, scene, seed, patch))
        for seed in SEEDS:
            paths[arm]["id_b_studio"].append(shoot(arm, "idB", ID_B, "studio", seed, patch))

    import torch

    del pipe
    torch.cuda.empty_cache()

    def pack(arm: str) -> dict:
        studio = paths[arm]["studio"]
        outdoor = paths[arm]["outdoor"]
        profile = paths[arm]["profile"]
        other = paths[arm]["id_b_studio"]
        same_scene = pairwise_face(studio)
        cross_scene = round(
            (
                pairwise_face(studio + outdoor)
                + pairwise_face(studio + profile)
                + pairwise_face(outdoor + profile)
            )
            / 3,
            3,
        )
        vs_other = round(
            sum(face_distance(a, b) for a in studio for b in other) / (len(studio) * len(other)),
            3,
        )
        return {
            "same_id_same_scene_face_l2": same_scene,
            "same_id_same_scene_full_l2": pairwise(studio),
            "same_id_cross_scene_face_l2": cross_scene,
            "different_id_same_scene_face_l2": vs_other,
            "identity_margin": round(vs_other - same_scene, 3),
            "cross_scene_holds": cross_scene < vs_other,
        }

    baseline = pack("baseline")
    patched = pack("triad_total")
    report = {
        "protocol": "valhalla-character-consistency-v1",
        "model": "RunDiffusion/Juggernaut-XL-v9",
        "identity_a": ID_A,
        "identity_b": ID_B,
        "scenes": list(SCENES),
        "seeds": list(SEEDS),
        "timings": timings,
        "baseline": baseline,
        "triad_total": patched,
        "verdicts": {
            "baseline_separates_identities": baseline["identity_margin"] > 2.0,
            "patch_separates_identities": patched["identity_margin"] > 2.0,
            "patch_same_scene_tighter": patched["same_id_same_scene_face_l2"]
            < baseline["same_id_same_scene_face_l2"],
            "patch_cross_scene_tighter": patched["same_id_cross_scene_face_l2"]
            < baseline["same_id_cross_scene_face_l2"],
            "patch_still_holds_identity": patched["cross_scene_holds"],
        },
    }
    path = OUT / "report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"baseline": baseline, "triad_total": patched, "verdicts": report["verdicts"]}, ensure_ascii=False))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
