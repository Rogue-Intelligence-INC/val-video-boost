#!/usr/bin/env python3
"""Simple test of the Fate interface, Fate-only comparison (no plugin).

Per (seed, scene): one pool of 4 candidates from Fate.supply().
Readouts from the SAME pool, so the only variable is Fate's choice:
  first     candidate 0 taken blindly (no Fate face functions)
  pool_mean what an average draw scores against the designated face
  select    Fate.select() against the enrolled Aria hero (the face
            designated for this chain)
  select_X  Fate.select() re-aimed at a DIFFERENT designated face
            (Sofia's hero) on the same pool -- can a designated face
            steer the outcome without touching generation?
Anchors: Aria heroes are the detailed_neg studio frames (seeds 7-9);
the foreign anchor is Sofia's detailed studio frame, same seeds.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import download
from fate_interface import FateInterface
from test_big_models import load_pipe

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "fate_interface"
DIAG = ROOT / "reports" / "context_diagnostic"
SEEDS = (7, 8, 9)
SCENES = ("park", "harbour")
K = 4
SIZE, STEPS = 768, 25


def gen(pipe, spec: dict, out_path: Path, seed: int) -> float:
    import torch
    if out_path.exists():
        return 0.0
    started = time.time()
    img = pipe(prompt=spec["prompt"], negative_prompt=spec["negative_prompt"],
               num_inference_steps=STEPS, guidance_scale=7.0,
               generator=torch.Generator(device="cuda").manual_seed(seed),
               height=SIZE, width=SIZE).images[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return round(time.time() - started, 2)


def main() -> None:
    pipe = load_pipe(download.IMAGE_JUG_DIR)
    fate = FateInterface()
    report: dict = {"protocol": "fate-interface-simple-v1", "k": K, "rows": []}
    sofia = FateInterface(fid=fate.fid)
    try:
        for seed in SEEDS:
            hero = DIAG / "prompt_channel" / "detailed_neg" / f"s{seed}_studio.png"
            assert fate.enroll_face(hero), f"no face in Aria hero {hero}"
            sofia_hero = DIAG / "prompt_channel" / "sofia_detailed" / f"s{seed}_studio.png"
            assert sofia.enroll_face(sofia_hero), f"no face in Sofia hero {sofia_hero}"
            for scene in SCENES:
                spec = fate.supply(scene)
                pool = []
                for k in range(K):
                    p = OUT / "pool" / f"s{seed}_{scene}_c{k}.png"
                    gen(pipe, spec, p, 5000 + seed * 100 + k)
                    pool.append(p)
                picked = fate.select(pool)
                picked_x = sofia.select(pool)
                scores = [s["cos"] for s in picked["scores"]]
                row = {"seed": seed, "scene": scene,
                       "first": scores[0],
                       "pool_mean": round(sum(scores) / len(scores), 4),
                       "pool_scores": scores,
                       "select": picked["chosen_cos"], "select_abstained": picked["abstained"],
                       "select_file": picked["chosen"],
                       "select_X_sofia_cos": picked_x["chosen_cos"],
                       "select_X_chosen_vs_aria": fate.score(picked_x["chosen"])}
                report["rows"].append(row)
                print(row)
    finally:
        import torch
        del pipe
        torch.cuda.empty_cache()

    def mean(key):
        vals = [r[key] for r in report["rows"] if r[key] is not None]
        return round(sum(vals) / len(vals), 4)

    report["summary"] = {
        "first_mean": mean("first"), "pool_mean": mean("pool_mean"),
        "select_mean": mean("select"), "select_min": min(r["select"] for r in report["rows"]),
        "select_pass_rate": round(sum(1 for r in report["rows"] if not r["select_abstained"]) / len(report["rows"]), 3),
        "select_gain_over_first": round(mean("select") - mean("first"), 4),
        "select_gain_over_pool_mean": round(mean("select") - mean("pool_mean"), 4),
        "designated_X_mean_cos_to_X": mean("select_X_sofia_cos"),
        "designated_X_picks_cos_to_aria": mean("select_X_chosen_vs_aria"),
    }
    report["reference"] = {"supply_only_detailed_neg_L4": 0.6102,
                           "ip_plugin_uniform": 0.5811, "different_person": 0.1522}
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
