#!/usr/bin/env python3
"""Gap 2: FLUX with the detailed supply.

FLUX baseline with the short fragment measured 0.480 (test_flux_character),
statistically tied with Juggernaut's 0.471 despite several times the
parameters. But FLUX's T5 text channel accepts far longer conditioning
than SDXL's 77 tokens, so the bible can supply MORE identity facts to
FLUX than to Juggernaut. This runs the detailed block on FLUX
(studio hero per seed, then park/harbour) and compares against that
0.480 short baseline. FLUX.1-schnell at guidance 0 has no effective
negative prompt, so this is detailed-only by design.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import character_bible as cb
from face_id import FaceID
from test_context_prompt import SCENES
from test_flux_fast import gen, load_fast

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "flux_detailed"
SEEDS = (7, 8)
BLOCK = cb.IDENTITY_A["detailed_block"]


def main() -> None:
    pipe = load_fast()
    fid = FaceID()
    rows = []
    try:
        for seed in SEEDS:
            hero = OUT / f"s{seed}_studio.png"
            if not hero.exists():
                gen(pipe, f"photorealistic photograph of {BLOCK}, {SCENES['studio']}", hero, seed)
            hero_emb = fid.embedding(hero)
            for scene in ("park", "harbour"):
                target = OUT / f"s{seed}_{scene}.png"
                if not target.exists():
                    gen(pipe, f"photorealistic photograph of {BLOCK}, {SCENES[scene]}", target, seed)
                emb = fid.embedding(target)
                cos = None if (emb is None or hero_emb is None) else round(fid.cosine(hero_emb, emb), 4)
                rows.append({"seed": seed, "scene": scene, "arcface_cos": cos})
                print(seed, scene, cos)
    finally:
        import torch
        del pipe
        torch.cuda.empty_cache()
    ok = [r["arcface_cos"] for r in rows if r["arcface_cos"] is not None]
    report = {"protocol": "flux-detailed-supply-v1", "rows": rows,
              "mean_cos": round(sum(ok) / len(ok), 4) if ok else None,
              "min_cos": min(ok) if ok else None,
              "reference": {"flux_short_baseline": 0.480, "juggernaut_detailed_neg": 0.6102,
                            "juggernaut_short": 0.471},
              "verdicts": {"flux_detailed_beats_flux_short": bool(ok) and (sum(ok) / len(ok)) > 0.480}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"mean": report["mean_cos"], "min": report["min_cos"],
                      "verdicts": report["verdicts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
