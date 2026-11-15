#!/usr/bin/env python3
"""No-plugin long chain: the extreme test's protocol, Fate-native arm.

Same hostile chain and SAME pre-registered pass bar as the IP-plugin
extreme test (reports/extreme/report.json), so the two are directly
comparable:
  * ONE hero, enrolled once via FateInterface.enroll_face, never updated
    (here: the detailed-supply studio hero, seed 7 -- a same-source face,
    which is the only kind the interface claims to anchor)
  * 12 scenes x 2 seeds = 24 sequential segments (noon/dusk/neon/rain/
    beach glare/snow/fog/golden hour, outfits and expressions vary)
  * per segment: pool of 4 candidates from FateInterface.supply_desc(),
    Fate.select() keeps the closest; tau=0.5 abstentions are logged,
    never silently shipped (the pool IS the regeneration budget)
  * pass bar: mean >= 0.5, >= 90% of segments >= 0.5, worst >= 0.40,
    second-half mean within 0.05 of first-half (anchor does not fade)
No plugin, no patch, no reference conditioning anywhere.
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
OUT = ROOT / "reports" / "fate_long_chain"
DIAG = ROOT / "reports" / "context_diagnostic"
SIZE, STEPS, K = 768, 25, 4
SEEDS = (21, 22)
SCENES = [
    ("park_noon", "standing in a sunlit park at noon, wearing a green summer dress, smiling, waist-up, facing camera"),
    ("harbour_dusk", "at a quiet harbour at dusk, wearing a navy coat, sea breeze in her hair, waist-up, facing camera"),
    ("neon_night", "on a neon-lit city street at night, wearing a black leather jacket, wet pavement reflections, waist-up, facing camera"),
    ("cafe_rain", "inside a cafe by a rainy window, wearing a burgundy turtleneck, warm indoor light, gentle smile, waist-up, facing camera"),
    ("beach_noon", "on a bright beach at noon, wearing a white sundress and sun hat in hand, squinting slightly in sunlight, waist-up, facing camera"),
    ("snow_village", "in a snowy mountain village, wearing a red winter parka with fur hood down, rosy cold cheeks, waist-up, facing camera"),
    ("library", "in an old library between tall shelves, wearing a brown cardigan, soft window light, waist-up, facing camera"),
    ("office", "in a modern office, wearing a grey blazer, neutral expression, fluorescent light, waist-up, facing camera"),
    ("desert_sunset", "on a desert road at sunset, wearing a denim jacket, warm backlight, hair blowing, waist-up, facing camera"),
    ("forest_fog", "on a forest trail in morning fog, wearing an olive rain jacket, diffused grey light, waist-up, facing camera"),
    ("crosswalk_evening", "at a busy city crosswalk in the evening, wearing a yellow raincoat, city lights behind, waist-up, facing camera"),
    ("rooftop_golden", "close-up on a rooftop at golden hour, wearing a cream blouse, warm side light, laughing, facing camera"),
]


def gen(pipe, spec, out_path: Path, seed: int) -> float:
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
    fate = FateInterface()
    hero = DIAG / "prompt_channel" / "detailed_neg" / "s7_studio.png"
    assert fate.enroll_face(hero), f"no face in hero {hero}"
    pipe = load_pipe(download.IMAGE_JUG_DIR)
    rows = []
    try:
        seg = 0
        for scene, desc in SCENES:
            spec = fate.supply_desc(desc)
            for seed in SEEDS:
                pool = []
                secs = 0.0
                for k in range(K):
                    p = OUT / "pool" / f"seg{seg:02d}_{scene}_s{seed}_c{k}.png"
                    secs += gen(pipe, spec, p, 20000 + seg * 100 + k)
                    pool.append(p)
                picked = fate.select(pool)
                rows.append({"segment": seg, "scene": scene, "seed": seed,
                             "pool_scores": [s["cos"] for s in picked["scores"]],
                             "chosen": picked["chosen"], "chosen_cos": picked["chosen_cos"],
                             "abstained": picked["abstained"], "seconds": round(secs, 1),
                             "fingerprint": spec["identity_fingerprint"]})
                print(seg, scene, seed, "pool", rows[-1]["pool_scores"], "->", picked["chosen_cos"],
                      "ABSTAIN" if picked["abstained"] else "")
                seg += 1
    finally:
        import torch
        del pipe
        torch.cuda.empty_cache()

    ok = [r["chosen_cos"] for r in rows if r["chosen_cos"] is not None]
    half = len(ok) // 2
    summary = {"mean_cos": round(sum(ok) / len(ok), 4), "min_cos": min(ok),
               "pct_ge_0.5": round(100.0 * sum(1 for v in ok if v >= 0.5) / len(ok), 1),
               "first_half_mean": round(sum(ok[:half]) / half, 4),
               "second_half_mean": round(sum(ok[half:]) / (len(ok) - half), 4),
               "abstentions": sum(1 for r in rows if r["abstained"]),
               "segments": len(rows), "pool_size": K}
    verdicts = {"mean_ge_0.5": summary["mean_cos"] >= 0.5,
                "pct_ge_0.5_at_least_90": summary["pct_ge_0.5"] >= 90.0,
                "worst_ge_0.40": summary["min_cos"] >= 0.40,
                "anchor_does_not_fade": abs(summary["second_half_mean"] - summary["first_half_mean"]) <= 0.05}
    verdicts["LONG_CHAIN_PASS"] = all(verdicts.values())
    report = {"protocol": "fate-interface-long-chain-v1", "anchor": str(hero),
              "rows": rows, "summary": summary, "verdicts": verdicts,
              "plugin_reference_extreme": {"mean_cos": 0.5516, "min_cos": 0.508,
                                           "note": "IP-Adapter arm, reports/extreme/report.json"}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    print(json.dumps(verdicts, ensure_ascii=False))


if __name__ == "__main__":
    main()
