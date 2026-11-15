#!/usr/bin/env python3
"""L3 dose response (no plugin): centred donor-A vs donor-B patches on
a fixed prompt-A, strength 0.03 / 0.06 (0.015 already measured in L2).

If the donor gap (index_centerA - index_centerB) grows with strength
before faces break, the improvement point is export centring + dose
calibration. If pixel churn grows but the gap stays ~0.1, the
injection projection itself (repeat-tiled 256-d vector) is the wall
and no dose fixes it. Face-found and cos_to_A are the distortion guard.
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
from test_context_crossover import PARK, SEEDS, norm, prompt, stripped

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "context_diagnostic"
PATCH_DIR = OUT / "patches"
STRENGTHS = (0.03, 0.06)


def main() -> None:
    raw = {n: module_patch.load_export_patch(PATCH_DIR / f"{n}.json")
           for n in ("A_aria", "B_marco", "C_sofia", "D_yuki")}
    dim = len(raw["A_aria"]["patch_vector"])
    mean = [sum(raw[n]["patch_vector"][i] for n in raw) / len(raw) for i in range(dim)]
    center = {n: norm([v - m for v, m in zip(raw[n]["patch_vector"], mean)]) for n in raw}

    pipe = load_pipe(download.IMAGE_JUG_DIR)
    fid = FaceID()
    hero_emb = {}
    for who in ("A", "B"):
        for seed in SEEDS:
            hero_emb[(who, seed)] = fid.embedding(OUT / "heroes" / f"{who}_s{seed}_studio.png")
    report: dict = {"protocol": "context-diagnostic-L3-dose-v1", "arms": {}}
    try:
        for strength in STRENGTHS:
            for donor in ("A_aria", "B_marco"):
                arm = f"center_{donor[0]}_s{strength}"
                patch = stripped(raw[donor], center[donor])
                rows = []
                for seed in SEEDS:
                    target = OUT / "dose" / f"{arm}_s{seed}_park.png"
                    secs = 0.0 if target.exists() else gen(
                        pipe, prompt(cb.IDENTITY_A["prompt_fragment"], PARK),
                        target, seed, 768, patch=patch, strength=strength)
                    emb = fid.embedding(target)
                    cos_a = fid.cosine(hero_emb[("A", seed)], emb) if emb is not None else None
                    cos_b = fid.cosine(hero_emb[("B", seed)], emb) if emb is not None else None
                    ref = OUT / "crossover" / f"A_none_s{seed}_park.png"
                    rows.append({"seed": seed, "seconds": secs, "face": emb is not None,
                                 "cos_to_A": None if cos_a is None else round(cos_a, 4),
                                 "cos_to_B": None if cos_b is None else round(cos_b, 4),
                                 "identity_index": None if cos_a is None else round(cos_a - cos_b, 4),
                                 "pixel_dist_to_prompt_only": round(module_patch.pixel_distance(target, ref), 3) if ref.exists() else None})
                    print(arm, seed, rows[-1])
                idx = [r["identity_index"] for r in rows if r["identity_index"] is not None]
                report["arms"][arm] = {"strength": strength, "donor": donor, "rows": rows,
                                       "identity_index_mean": round(sum(idx) / len(idx), 4) if idx else None}
    finally:
        import torch
        del pipe
        torch.cuda.empty_cache()
    report["donor_gap_by_strength"] = {
        str(s): round(report["arms"][f"center_A_s{s}"]["identity_index_mean"]
                      - report["arms"][f"center_B_s{s}"]["identity_index_mean"], 4)
        for s in STRENGTHS}
    report["donor_gap_by_strength"]["0.015"] = 0.0953  # from report_L2.json
    (OUT / "report_L3.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["donor_gap_by_strength"], ensure_ascii=False))


if __name__ == "__main__":
    main()
