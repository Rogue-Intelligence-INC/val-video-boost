#!/usr/bin/env python3
"""L4b control: is the detailed-prompt gain real specificity, or just
prototype convergence (all detailed faces becoming similar)?

Generates Sofia with a parallel detailed description (different hair,
eyes, face shape) and measures ArcFace between the detailed-A heroes
and detailed-Sofia heroes, seed-matched. If specificity is real this
stays at different-person level (~0.15-0.3), far below the 0.589
same-person detailed score.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import download
from face_id import FaceID
from test_big_models import load_pipe
from test_context_prompt import SIZE, STEPS, gen
from test_fate_iterative_character import SCENES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "context_diagnostic"

SOFIA_DETAILED = ("the same single woman Sofia, 28 years old, long platinum blonde "
                  "hair with a side parting, blue eyes, pale skin, round face, "
                  "low cheekbones, small upturned nose, thin eyebrows, full lips, "
                  "round chin")


def main() -> None:
    pipe = load_pipe(download.IMAGE_JUG_DIR)
    fid = FaceID()
    cross, sofia_self_ref = [], None
    rows = []
    try:
        for seed in (7, 8, 9):
            sofia = OUT / "prompt_channel" / "sofia_detailed" / f"s{seed}_studio.png"
            gen(pipe, SOFIA_DETAILED, "studio", None, sofia, seed)
            a_hero = OUT / "prompt_channel" / "detailed" / f"s{seed}_studio.png"
            ea, es = fid.embedding(a_hero), fid.embedding(sofia)
            c = fid.cosine(ea, es) if (ea is not None and es is not None) else None
            rows.append({"seed": seed, "cos_A_detailed_vs_Sofia_detailed": None if c is None else round(c, 4)})
            if c is not None:
                cross.append(c)
            print(rows[-1])
    finally:
        import torch
        del pipe
        torch.cuda.empty_cache()
    report = {"protocol": "context-diagnostic-L4b-cross-person-control-v1", "rows": rows,
              "cross_person_detailed_mean": round(sum(cross) / len(cross), 4) if cross else None,
              "same_person_detailed_mean_from_L4": 0.5889,
              "verdicts": {"specificity_real": bool(cross) and (sum(cross) / len(cross)) < 0.40}}
    (OUT / "report_L4b.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
