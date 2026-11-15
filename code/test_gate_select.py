#!/usr/bin/env python3
"""No-plugin Valhalla gate: candidate grasp + auditable abstention.

Generation is the untouched local Juggernaut (no patch, no IP-Adapter).
Valhalla contributes what Fate actually is -- a gate over candidates:
4 seeds per segment are generated, the face gate (ArcFace to the pinned
hero, standing in for fate margin) grasps the best candidate, and if all
candidates fall below tau the segment is ABSTAINED, not shipped. This is
selection, not injection: no identity information enters the denoiser,
so any gain is an upper bound on what pure gating can buy.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import download
from face_id import FaceID
from test_big_models import load_pipe
from test_fate_iterative_character import SEEDS, prompt
from test_valhalla_native import gen_plain

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "valhalla_native"
TAU = 0.50
CANDIDATES = 4


def main():
    fid = FaceID()
    pipe = load_pipe(download.IMAGE_JUG_DIR)
    rows = []
    try:
        for seed in SEEDS:
            hero = ROOT / "reports" / "fate_iterative_character" / "static_triad" / f"s{seed}_studio.png"
            hero_emb = fid.embedding(hero)
            for scene in ("park", "harbour"):
                cands = []
                for k in range(CANDIDATES):
                    cand_seed = seed * 100 + k
                    target = OUT / "gate_select" / f"s{seed}_{scene}_c{k}.png"
                    gen_plain(pipe, prompt(scene), target, cand_seed)
                    emb = fid.embedding(target)
                    cos = None if (emb is None or hero_emb is None) else fid.cosine(hero_emb, emb)
                    cands.append({"k": k, "seed": cand_seed, "cos": cos})
                best = max(cands, key=lambda c: c["cos"] if c["cos"] is not None else -1)
                rows.append({"seed": seed, "scene": scene, "candidates": cands,
                             "selected_cos": best["cos"],
                             "status": "accepted" if (best["cos"] or 0) >= TAU else "abstained",
                             "first_candidate_cos": cands[0]["cos"]})
                print(seed, scene, [c["cos"] for c in cands], "->", best["cos"], rows[-1]["status"])
    finally:
        import torch
        del pipe
        torch.cuda.empty_cache()
    sel = [r["selected_cos"] for r in rows]
    first = [r["first_candidate_cos"] for r in rows]
    report = {"protocol": "val-video-boost-gate-select-v1", "tau": TAU, "candidates": CANDIDATES,
              "rows": rows,
              "summary": {"selected_mean_cos": round(sum(sel) / len(sel), 4),
                          "single_candidate_mean_cos": round(sum(first) / len(first), 4),
                          "accepted": sum(1 for r in rows if r["status"] == "accepted"),
                          "segments": len(rows)}}
    (OUT / "report_gate_select.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
