#!/usr/bin/env python3
"""Simple framework test: discover -> registry -> calibrate -> route -> run.

Live parts kept small: strength calibration on Juggernaut (the adaptive
probe from val-videopic-fusion: seeds 7/8 pair, distortion guards), one
FLUX portrait at the framework's prior strength, and the route table.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import adaptive_framework as af
import module_patch
from test_big_models import gen, load_pipe
from test_full_quality import SIGN, pairwise, stats_for

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "adaptive_framework"
PATCH_JSON = ROOT / "data" / "identity_patch.json"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    catalog = af.discover()
    registry = af.load_measured_registry()
    print(f"discovered {len(catalog)} models")
    for m in catalog:
        print(" ", m["name"], m["pipeline_class"], "->", m["family"])
    report = {"protocol": "val-video-boost-adaptive-framework-v1",
              "catalog": catalog, "registry": registry}

    triad = module_patch.load_export_patch(PATCH_JSON)

    # --- live calibration: Juggernaut, adaptive probe ladder ------------
    import download
    pipe = load_pipe(download.IMAGE_JUG_DIR)
    profile = module_patch.profile_denoiser(pipe.unet, "unet")
    plan = module_patch.adaptive_plan(profile, triad["body_source"], "total", steps=25, patch=triad)
    print("jug profile", profile["block_count"], "plan", plan["probe_ladder"])
    probes, chosen, best = [], None, None
    for strength in plan["probe_ladder"][:3]:  # simple test: first 3 rungs
        p7, p8 = OUT / f"probe_{strength}_s7.png", OUT / f"probe_{strength}_s8.png"
        gen(pipe, SIGN, p7, 7, 512, patch=triad, strength=strength)
        gen(pipe, SIGN, p8, 8, 512, patch=triad, strength=strength)
        pair = round(module_patch.pixel_distance(p7, p8), 3)
        st = stats_for(p7)
        guard = (not st["blank"]) and st["clipped_frac"] < 0.35 and st["sharpness"] > 10
        probes.append({"strength": strength, "pair_l2": pair, "guard_ok": guard})
        print("  rung", strength, "pair", pair, "guard", guard)
        if guard and (best is None or pair < best):
            best, chosen = pair, strength
    report["calibration"] = {"model": "juggernaut-xl-v9", "profile_blocks": profile["block_count"],
                             "ladder": plan["probe_ladder"][:3], "probes": probes,
                             "chosen_strength": chosen}
    import torch
    del pipe
    torch.cuda.empty_cache()

    # --- FLUX through the framework: prior strength, one portrait --------
    from test_flux_fast import gen as flux_gen
    from test_flux_fast import load_fast
    fpipe = load_fast()
    fprofile = module_patch.profile_denoiser(fpipe.transformer, "transformer")
    target = OUT / "flux_portrait_adaptive.png"
    secs = flux_gen(fpipe, "portrait of Aria, a 28-year-old woman with chestnut brown hair, studio light, facing camera",
                    target, 7, patch=triad, strength=af.FAMILY_KB["flux"]["prior_strength"])
    report["flux"] = {"profile": fprofile, "seconds": secs, "portrait": str(target),
                      "stats": stats_for(target)}
    print("flux blocks", fprofile["block_count"], "portrait", secs, "s")
    del fpipe
    torch.cuda.empty_cache()

    report["routes"] = {task: af.route(task, registry) for task in af.ROUTES}
    report["verdicts"] = {
        "catalog_found_video_and_image": any(m["family"] == "cogvideox" for m in catalog) and any(m["family"] == "flux" for m in catalog),
        "calibration_chose_strength": chosen is not None,
        "flux_patch_applied": report["flux"]["stats"]["blank"] is False,
        "routes_cover_tasks": len(report["routes"]) == len(af.ROUTES),
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["verdicts"], ensure_ascii=False))
    for task, r in report["routes"].items():
        print(task, "->", r["model"])


if __name__ == "__main__":
    main()
