#!/usr/bin/env python3
"""L5 stacking (no plugin): on top of the detailed-prompt arm --
  detailed_neg+centerA / +centerB : does the centred patch add a donor
      gap once the text channel is already fed properly?
  detailed_paraphrased : same identity facts, surface form rewritten
      per scene (hero keeps the canonical wording). If verbatim pinning
      wins, the Valhalla supply rule is byte-identical identity blocks.
Reference arms from L4: detailed 0.5889, detailed_neg 0.6102.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import download
import module_patch
from face_id import FaceID
from test_big_models import load_pipe
from test_context_crossover import norm, stripped
from test_context_prompt import DETAILED, NEGATIVE, SIZE, STEPS
from test_fate_iterative_character import SCENES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "context_diagnostic"
PATCH_DIR = OUT / "patches"
SEEDS = (7, 8, 9)

PARAPHRASED = ("Aria, a woman of twenty-eight, her chestnut-brown hair falling to "
               "her shoulders and parted in the middle, hazel eyes over fair skin, "
               "a tiny mole upon her left cheek, an oval slender face with prominent "
               "cheekbones, a straight nose, dark brows and naturally pink lips")


def gen(pipe, frag, scene, out_path, seed, negative=None, patch=None):
    import torch
    if out_path.exists():
        return 0.0
    started = time.time()
    handle = None
    if patch is not None:
        handle = module_patch.ModulePatch(patch, mode="total")
        handle.attach(pipe.unet, "unet")
    kw = dict(prompt=f"photorealistic photograph of {frag}, {SCENES[scene]}",
              num_inference_steps=STEPS, guidance_scale=7.0,
              generator=torch.Generator(device="cuda").manual_seed(seed),
              height=SIZE, width=SIZE)
    if negative:
        kw["negative_prompt"] = negative
    try:
        img = pipe(**kw).images[0]
    finally:
        if handle is not None:
            handle.remove()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return round(time.time() - started, 2)


def main() -> None:
    raw = {n: module_patch.load_export_patch(PATCH_DIR / f"{n}.json")
           for n in ("A_aria", "B_marco", "C_sofia", "D_yuki")}
    dim = len(raw["A_aria"]["patch_vector"])
    mean = [sum(raw[n]["patch_vector"][i] for n in raw) / len(raw) for i in range(dim)]
    center = {n: norm([v - m for v, m in zip(raw[n]["patch_vector"], mean)]) for n in raw}

    pipe = load_pipe(download.IMAGE_JUG_DIR)
    fid = FaceID()
    report: dict = {"protocol": "context-diagnostic-L5-stacking-v1", "arms": {}}
    arms = {
        "detailedneg_centerA": ("detailed_neg", stripped(raw["A_aria"], center["A_aria"])),
        "detailedneg_centerB": ("detailed_neg", stripped(raw["B_marco"], center["B_marco"])),
        "detailed_paraphrased": ("detailed", None),
    }
    try:
        for arm, (hero_arm, patch) in arms.items():
            rows = []
            for seed in SEEDS:
                hero = OUT / "prompt_channel" / hero_arm / f"s{seed}_studio.png"
                hero_emb = fid.embedding(hero)
                for scene in ("park", "harbour"):
                    target = OUT / "stacking" / arm / f"s{seed}_{scene}.png"
                    if arm == "detailed_paraphrased":
                        secs = gen(pipe, PARAPHRASED, scene, target, seed)
                    else:
                        secs = gen(pipe, DETAILED, scene, target, seed, negative=NEGATIVE, patch=patch)
                    emb = fid.embedding(target)
                    cos = fid.cosine(hero_emb, emb) if (emb is not None and hero_emb is not None) else None
                    rows.append({"seed": seed, "scene": scene, "seconds": secs,
                                 "arcface_cos": None if cos is None else round(cos, 4)})
                    print(arm, seed, scene, rows[-1]["arcface_cos"])
            ok = [r["arcface_cos"] for r in rows if r["arcface_cos"] is not None]
            report["arms"][arm] = {"rows": rows, "mean_cos": round(sum(ok) / len(ok), 4) if ok else None,
                                   "min_cos": min(ok) if ok else None}
    finally:
        import torch
        del pipe
        torch.cuda.empty_cache()
    a = report["arms"]["detailedneg_centerA"]["mean_cos"]
    b = report["arms"]["detailedneg_centerB"]["mean_cos"]
    report["donor_gap_on_detailed"] = None if (a is None or b is None) else round(a - b, 4)
    report["reference"] = {"detailed": 0.5889, "detailed_neg": 0.6102, "donor_gap_on_short_L2": 0.0953}
    (OUT / "report_L5.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({**{k: v["mean_cos"] for k, v in report["arms"].items()},
                      "donor_gap_on_detailed": report["donor_gap_on_detailed"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
