#!/usr/bin/env python3
"""L2 crossover (no plugin): prompt says A, patch says B -- who wins?

Arms (Juggernaut XL, park scene, seeds 7/8, total mode @0.015):
  A_none / B_none        prompt-only heroes' scene baseline
  A_rawA / A_rawB        historical exported patches, signatures intact
  A_vecA / A_vecB        patch_vector only (tile/stem signatures stripped)
  A_centerA / A_centerB  centred patch_vector (common mode removed)
  B_centerA              reverse crossover for symmetry
  A_random               same-geometry random vector (perturbation control)

Readouts per image: ArcFace cos to the seed-matched A hero and B hero
(identity index = cosA - cosB), pixel distance to the same-seed
prompt-only image (does the patch move the picture at all), face found.
If centred patches shift the identity index toward their donor while
raw patches do not, the improvement point is export-side centring.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import character_bible as cb
import download
import module_patch
from face_id import FaceID
from test_big_models import gen, load_pipe

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "context_diagnostic"
PATCH_DIR = OUT / "patches"
SIZE = 768
SEEDS = (7, 8)
PARK = "standing outdoors in a sunlit park, natural daylight, waist-up, facing camera"
STUDIO = "close-up portrait, studio light, plain grey background, facing camera"
FRAG_B = ("the same 45-year-old man Marco with short grey hair and a grey beard, "
          "dark brown eyes, olive skin")


def prompt(frag: str, scene: str) -> str:
    return f"photorealistic photograph of {frag}, {scene}"


def norm(v):
    n = sum(x * x for x in v) ** 0.5 or 1e-9
    return [x / n for x in v]


def stripped(patch: dict, vector=None) -> dict:
    p = dict(patch)
    if vector is not None:
        p["patch_vector"] = vector
    p["tile_signatures"] = []
    p["stem_signatures"] = []
    p["tile_activations"] = []
    return p


def main() -> None:
    raw = {n: module_patch.load_export_patch(PATCH_DIR / f"{n}.json")
           for n in ("A_aria", "B_marco", "C_sofia", "D_yuki")}
    dim = len(raw["A_aria"]["patch_vector"])
    mean = [sum(raw[n]["patch_vector"][i] for n in raw) / len(raw) for i in range(dim)]
    center = {n: norm([v - m for v, m in zip(raw[n]["patch_vector"], mean)]) for n in raw}
    rnd = module_patch.random_patch(raw["A_aria"])

    frags = {"A": cb.IDENTITY_A["prompt_fragment"], "B": FRAG_B}
    arms: dict[str, tuple[str, dict | None]] = {
        "A_none": ("A", None),
        "B_none": ("B", None),
        "A_rawA": ("A", raw["A_aria"]),
        "A_rawB": ("A", raw["B_marco"]),
        "A_vecA": ("A", stripped(raw["A_aria"])),
        "A_vecB": ("A", stripped(raw["B_marco"])),
        "A_centerA": ("A", stripped(raw["A_aria"], center["A_aria"])),
        "A_centerB": ("A", stripped(raw["B_marco"], center["B_marco"])),
        "B_centerA": ("B", stripped(raw["A_aria"], center["A_aria"])),
        "A_random": ("A", stripped(rnd, rnd["patch_vector"])),
    }

    pipe = load_pipe(download.IMAGE_JUG_DIR)
    fid = FaceID()
    report: dict = {"protocol": "context-diagnostic-L2-crossover-v1", "arms": {}}
    try:
        heroes: dict[tuple[str, int], Path] = {}
        for who in ("A", "B"):
            for seed in SEEDS:
                p = OUT / "heroes" / f"{who}_s{seed}_studio.png"
                if who == "A" and not p.exists():
                    old = ROOT / "reports" / "fate_iterative_character" / "baseline" / f"s{seed}_studio.png"
                    if old.exists():
                        p.parent.mkdir(parents=True, exist_ok=True)
                        import shutil
                        shutil.copy(old, p)
                if not p.exists():
                    gen(pipe, prompt(frags[who], STUDIO), p, seed, SIZE)
                heroes[(who, seed)] = p
        hero_emb = {k: fid.embedding(v) for k, v in heroes.items()}

        for arm, (who, patch) in arms.items():
            rows = []
            for seed in SEEDS:
                target = OUT / "crossover" / f"{arm}_s{seed}_park.png"
                secs = 0.0 if target.exists() else gen(pipe, prompt(frags[who], PARK), target, seed, SIZE, patch=patch)
                emb = fid.embedding(target)
                ref_none = OUT / "crossover" / f"{who}_none_s{seed}_park.png"
                cos_a = fid.cosine(hero_emb[("A", seed)], emb) if emb is not None else None
                cos_b = fid.cosine(hero_emb[("B", seed)], emb) if emb is not None else None
                rows.append({
                    "seed": seed, "seconds": secs, "face": emb is not None,
                    "cos_to_A": None if cos_a is None else round(cos_a, 4),
                    "cos_to_B": None if cos_b is None else round(cos_b, 4),
                    "identity_index": None if cos_a is None else round(cos_a - cos_b, 4),
                    "pixel_dist_to_prompt_only": (round(module_patch.pixel_distance(target, ref_none), 3)
                                                  if ref_none.exists() and ref_none != target else None),
                })
                print(arm, seed, rows[-1])
            idx = [r["identity_index"] for r in rows if r["identity_index"] is not None]
            report["arms"][arm] = {"rows": rows,
                                   "identity_index_mean": round(sum(idx) / len(idx), 4) if idx else None}
    finally:
        import torch
        del pipe
        torch.cuda.empty_cache()

    (OUT / "report_L2.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({a: v["identity_index_mean"] for a, v in report["arms"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
