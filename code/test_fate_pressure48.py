#!/usr/bin/env python3
"""Free gap B: 48-segment pressure + scripted anchor version switch.

Part 1 (pressure): segments 24-47 repeat the long-chain scenes with
fresh seeds against the SAME never-updated anchor. Combined with the
recorded segments 0-23, consistency is reported per quarter -- the
mechanistic claim (DESIGN ss.15) is that quarter means do not decay.

Part 2 (valid-time anchor switch): the story ages Aria. A v2 identity
block (45 years old, silver streaks, fine lines) generates a NEW hero,
which is enrolled as anchor version 2; 4 scenes x pool-of-4 must then
hold against v2. This tests the bible's valid-time rule: identity may
change only at scripted points, and each version is internally a
star topology around its own frozen anchor.
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
from test_fate_long_chain import SCENES, gen

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "fate_pressure48"
DIAG = ROOT / "reports" / "context_diagnostic"
K = 4

AGED_BLOCK = ("the same single woman Aria, 45 years old, shoulder-length chestnut brown "
              "hair with silver-grey streaks and a center parting, small mole on her left "
              "cheek, hazel eyes, fair skin with fine lines around the eyes, slim oval face, "
              "high cheekbones, straight nose, dark brown eyebrows, natural pink lips, small chin")
AGED_SCENES = [
    ("park", "standing outdoors in a sunlit park, natural daylight, waist-up, facing camera"),
    ("harbour", "at a quiet harbour at dusk, sea breeze, waist-up, facing camera"),
    ("neon_night", "on a neon-lit city street at night, wet pavement reflections, waist-up, facing camera"),
    ("snow_village", "in a snowy mountain village, wearing a red winter parka, rosy cold cheeks, waist-up, facing camera"),
]


def supply_aged(fate: FateInterface, desc: str) -> dict:
    spec = fate.supply("park")
    spec["prompt"] = f"photorealistic photograph of {AGED_BLOCK}, {desc}"
    return spec


def main() -> None:
    fate = FateInterface()
    hero = DIAG / "prompt_channel" / "detailed_neg" / "s7_studio.png"
    assert fate.enroll_face(hero)
    pipe = load_pipe(download.IMAGE_JUG_DIR)
    rows = []
    aged_rows = []
    try:
        seg = 24
        for scene, desc in SCENES:
            spec = fate.supply_desc(desc)
            for seed in (31, 32):
                pool = []
                for k in range(K):
                    p = OUT / "pool" / f"seg{seg:02d}_{scene}_s{seed}_c{k}.png"
                    gen(pipe, spec, p, 40000 + seg * 100 + k)
                    pool.append(p)
                picked = fate.select(pool)
                rows.append({"segment": seg, "scene": scene, "seed": seed,
                             "chosen_cos": picked["chosen_cos"], "abstained": picked["abstained"],
                             "pool_scores": [s["cos"] for s in picked["scores"]]})
                print(seg, scene, picked["chosen_cos"])
                seg += 1
        # Part 2: anchor version 2 (aged)
        aged_hero = OUT / "aged_hero_studio.png"
        if not aged_hero.exists():
            gen(pipe, supply_aged(fate, "close-up portrait, studio light, plain grey background, facing camera"),
                aged_hero, 7)
        fate2 = FateInterface(fid=fate.fid)
        assert fate2.enroll_face(aged_hero), "no face in aged hero"
        for scene, desc in AGED_SCENES:
            spec = supply_aged(fate2, desc)
            pool = []
            for k in range(K):
                p = OUT / "aged_pool" / f"{scene}_c{k}.png"
                gen(pipe, spec, p, 50000 + k)
                pool.append(p)
            picked = fate2.select(pool)
            aged_rows.append({"scene": scene, "chosen_cos": picked["chosen_cos"],
                              "abstained": picked["abstained"],
                              "pool_scores": [s["cos"] for s in picked["scores"]]})
            print("aged", scene, picked["chosen_cos"])
    finally:
        import torch
        del pipe
        torch.cuda.empty_cache()

    prev = json.loads((ROOT / "reports" / "fate_long_chain" / "report.json").read_text(encoding="utf-8"))
    all_cos = [r["chosen_cos"] for r in prev["rows"]] + [r["chosen_cos"] for r in rows]
    quarters = [round(sum(all_cos[i * 12:(i + 1) * 12]) / 12, 4) for i in range(4)]
    aged_ok = [r["chosen_cos"] for r in aged_rows if r["chosen_cos"] is not None]
    report = {"protocol": "fate-pressure48-anchor-switch-v1", "new_rows": rows,
              "combined_48": {"quarter_means": quarters, "mean": round(sum(all_cos) / 48, 4),
                              "min": min(all_cos),
                              "pct_ge_0.5": round(100 * sum(1 for v in all_cos if v >= 0.5) / 48, 1)},
              "aged_v2": {"rows": aged_rows,
                          "mean": round(sum(aged_ok) / len(aged_ok), 4) if aged_ok else None,
                          "min": min(aged_ok) if aged_ok else None},
              "verdicts": {
                  "no_decay_q4_vs_q1": quarters[3] >= quarters[0] - 0.05,
                  "all_48_ge_0.5": min(all_cos) >= 0.5,
                  "aged_anchor_holds": bool(aged_ok) and min(aged_ok) >= 0.5}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"quarters": quarters, "combined": report["combined_48"],
                      "aged": report["aged_v2"], "verdicts": report["verdicts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
