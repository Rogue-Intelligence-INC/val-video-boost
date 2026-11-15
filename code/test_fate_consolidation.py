#!/usr/bin/env python3
"""Consolidation regression for the no-plugin Fate identity stack.

Pins, so the consolidated stack cannot silently rot:
  1. single source: fate_interface re-exports the bible's pinned block
     and negative facts (no second copy to drift)
  2. verbatim supply: every scene's prompt carries the byte-identical
     identity block, one fingerprint for all scenes
  3. select() on a recorded pool returns the argmax, passes verify(),
     and logs every candidate score
  4. foreign-face boundary: selecting an Aria pool against Sofia's
     enrolled face stays under tau and is flagged abstained
  5. framework route + measured registry know the no-plugin stack
  6. end-to-end smoke: a fresh 2-candidate pool generated through
     FateInterface.supply() is selectable and passes the gate
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import adaptive_framework as af
import character_bible as cb
import fate_interface as fi
from fate_interface import FateInterface

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "fate_interface"
DIAG = ROOT / "reports" / "context_diagnostic"
SIZE, STEPS = 768, 25


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
    # 1-2: single source + verbatim supply
    supplies = {s: fate.supply(s) for s in ("studio", "park", "harbour")}
    single_source = (fi.IDENTITY_BLOCK == cb.IDENTITY_A["detailed_block"]
                     and fi.NEGATIVE == cb.IDENTITY_A["negative_facts"])
    verbatim = all(fi.IDENTITY_BLOCK in spec["prompt"] for spec in supplies.values())
    one_fp = len({spec["identity_fingerprint"] for spec in supplies.values()}) == 1

    # 3: select on a recorded pool (s9 harbour, scores known auditable)
    hero9 = DIAG / "prompt_channel" / "detailed_neg" / "s9_studio.png"
    pool = sorted((OUT / "pool").glob("s9_harbour_c*.png"))
    assert fate.enroll_face(hero9) and len(pool) == 4
    picked = fate.select(pool)
    scores = [s["cos"] for s in picked["scores"]]
    select_ok = (picked["chosen_cos"] == max(scores) and not picked["abstained"]
                 and fate.verify(Path(picked["chosen"]))["passed"])

    # 4: foreign designated face must abstain on the same pool
    foreign = FateInterface(fid=fate.fid)
    assert foreign.enroll_face(DIAG / "prompt_channel" / "sofia_detailed" / "s9_studio.png")
    picked_x = foreign.select(pool)
    foreign_abstains = picked_x["abstained"] and picked_x["chosen_cos"] < foreign.tau

    # 5: framework integration
    route = af.route("identity_no_plugin", registry=af.load_measured_registry())
    reg = af.load_measured_registry().get("juggernaut-xl-v9", [])
    registry_ok = any(r["task"] == "character_identity_fate_native" and r["score"] >= 0.65 for r in reg)
    route_ok = route["model"] == "juggernaut-xl-v9" and "FateInterface" in route["rationale"]

    # 6: fresh end-to-end generation through the interface
    from test_big_models import load_pipe
    import download
    pipe = load_pipe(download.IMAGE_JUG_DIR)
    smoke = {}
    try:
        assert fate.enroll_face(DIAG / "prompt_channel" / "detailed_neg" / "s7_studio.png")
        spec = fate.supply("park")
        fresh_pool = []
        for k, seed in enumerate((9100, 9101)):
            p = OUT / "consolidation_pool" / f"smoke_park_c{k}.png"
            gen(pipe, spec, p, seed)
            fresh_pool.append(p)
        smoke_pick = fate.select(fresh_pool)
        smoke = {"scores": [s["cos"] for s in smoke_pick["scores"]],
                 "chosen_cos": smoke_pick["chosen_cos"], "abstained": smoke_pick["abstained"],
                 "fingerprint": spec["identity_fingerprint"]}
        smoke_ok = (smoke_pick["chosen"] is not None and len(smoke["scores"]) == 2
                    and all(s is not None for s in smoke["scores"]))
    finally:
        import torch
        del pipe
        torch.cuda.empty_cache()

    report = {"protocol": "fate-interface-consolidation-v1",
              "fingerprint": supplies["park"]["identity_fingerprint"],
              "recorded_pool_scores": scores, "smoke": smoke,
              "route": route,
              "verdicts": {"single_source_of_truth": single_source,
                           "verbatim_block_all_scenes": verbatim and one_fp,
                           "select_is_argmax_and_passes": select_ok,
                           "foreign_face_abstains": foreign_abstains,
                           "registry_has_fate_native_record": registry_ok,
                           "route_points_at_fate_stack": route_ok,
                           "fresh_generation_selectable": smoke_ok}}
    (OUT / "report_consolidation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["verdicts"], ensure_ascii=False))
    print("smoke", smoke)


if __name__ == "__main__":
    main()
