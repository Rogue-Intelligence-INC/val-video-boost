#!/usr/bin/env python3
"""Gap 1: dual-character separation (alternating scenes, two anchors).

Aria's chain exists (detailed_neg). Sofia gets the parallel treatment:
her own detailed block, her own negative facts (which must NOT negate
her own blonde hair), her own enrolled hero. Question: in one story
with two similar-age women alternating scenes, do the chains stay
apart? Readouts per character: same-person cos, cross-person cos to
the OTHER character's hero, and the margin between them.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import download
from face_id import FaceID
from fate_interface import FateInterface
from test_big_models import load_pipe
from test_context_prompt import SCENES, SIZE, STEPS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "dual_character"
DIAG = ROOT / "reports" / "context_diagnostic"
SEEDS = (7, 8, 9)

SOFIA_BLOCK = ("the same single woman Sofia, 28 years old, long platinum blonde "
               "hair with a side parting, blue eyes, pale skin, round face, "
               "low cheekbones, small upturned nose, thin eyebrows, full lips, "
               "round chin")
SOFIA_NEG = ("chestnut brown hair, black hair, grey hair, different person, "
             "another woman, changing face, beard, masculine features")


def gen(pipe, block, negative, scene, out_path, seed):
    import torch
    if out_path.exists():
        return 0.0
    started = time.time()
    img = pipe(prompt=f"photorealistic photograph of {block}, {SCENES[scene]}",
               negative_prompt=negative, num_inference_steps=STEPS, guidance_scale=7.0,
               generator=torch.Generator(device="cuda").manual_seed(seed),
               height=SIZE, width=SIZE).images[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return round(time.time() - started, 2)


def main() -> None:
    pipe = load_pipe(download.IMAGE_JUG_DIR)
    fid = FaceID()
    aria = FateInterface(fid=fid)
    rows = {"aria": [], "sofia": []}
    try:
        heroes = {}
        for seed in SEEDS:
            sofia_hero = OUT / "sofia" / f"s{seed}_studio.png"
            gen(pipe, SOFIA_BLOCK, SOFIA_NEG, "studio", sofia_hero, seed)
            heroes[seed] = {
                "aria": fid.embedding(DIAG / "prompt_channel" / "detailed_neg" / f"s{seed}_studio.png"),
                "sofia": fid.embedding(sofia_hero),
            }
            for scene in ("park", "harbour"):
                s_img = OUT / "sofia" / f"s{seed}_{scene}.png"
                gen(pipe, SOFIA_BLOCK, SOFIA_NEG, scene, s_img, seed)
                e = fid.embedding(s_img)
                rows["sofia"].append({
                    "seed": seed, "scene": scene,
                    "cos_same_sofia": None if e is None else round(fid.cosine(heroes[seed]["sofia"], e), 4),
                    "cos_cross_aria": None if e is None else round(fid.cosine(heroes[seed]["aria"], e), 4)})
                a_img = DIAG / "prompt_channel" / "detailed_neg" / f"s{seed}_{scene}.png"
                ea = fid.embedding(a_img)
                rows["aria"].append({
                    "seed": seed, "scene": scene,
                    "cos_same_aria": None if ea is None else round(fid.cosine(heroes[seed]["aria"], ea), 4),
                    "cos_cross_sofia": None if ea is None else round(fid.cosine(heroes[seed]["sofia"], ea), 4)})
                print(seed, scene, rows["sofia"][-1], rows["aria"][-1])
    finally:
        import torch
        del pipe
        torch.cuda.empty_cache()

    def agg(rows_, same, cross):
        s = [r[same] for r in rows_ if r[same] is not None]
        c = [r[cross] for r in rows_ if r[cross] is not None]
        return {"same_mean": round(sum(s) / len(s), 4), "cross_mean": round(sum(c) / len(c), 4),
                "margin": round(sum(s) / len(s) - sum(c) / len(c), 4),
                "cross_max": max(c) if c else None}

    summary = {"aria": agg(rows["aria"], "cos_same_aria", "cos_cross_sofia"),
               "sofia": agg(rows["sofia"], "cos_same_sofia", "cos_cross_aria")}
    report = {"protocol": "dual-character-separation-v1", "rows": rows, "summary": summary,
              "verdicts": {"both_margins_ge_0.10": all(v["margin"] >= 0.10 for v in summary.values()),
                           "cross_below_threshold": all(v["cross_mean"] < 0.5 for v in summary.values())}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    print(json.dumps(report["verdicts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
