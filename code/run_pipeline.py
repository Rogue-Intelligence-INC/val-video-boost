#!/usr/bin/env python3
"""Consolidated runtime pipeline -- the production flow in one entry.

This is the canonical runner the scattered test scripts were proving
out. One command executes the whole no-plugin identity stack:

  route     adaptive_framework.route("identity_no_plugin") picks the
            model + settings from measured records (logged, not assumed)
  hero      first segment (studio) is generated from Fate.supply();
            its face is enrolled ONCE as the designated anchor and
            never updated afterwards
  segments  per scene: K candidates from Fate.supply_desc(), then
            FateInterface.select() keeps the closest to the anchor;
            FateInterface.verify() gates at tau -- an abstained segment
            is reported as ABSTAINED, never silently shipped
  ledger    every candidate score, the identity fingerprint, and the
            chosen file per segment go to manifest.json (auditable);
            a contact sheet of anchor + chosen frames is rendered

Usage:
  python run_pipeline.py                          # smoke: 2 scenes x 1 seed
  python run_pipeline.py --scenes park,harbour,library --seeds 7,8 --k 4
  python run_pipeline.py --hero path/to/hero.png  # designate an existing
                                                  # same-source hero face
Exit code is 0 only if no segment abstained.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import adaptive_framework as af
import download
from fate_interface import FateInterface
from test_big_models import load_pipe

ROOT = Path(__file__).resolve().parents[1]
SIZE, STEPS = 768, 25
SCENE_TEXT = {
    "park": "standing outdoors in a sunlit park, natural daylight, waist-up, facing camera",
    "harbour": "at a quiet harbour at dusk, sea breeze, waist-up, facing camera",
    "library": "in an old library between tall shelves, soft window light, waist-up, facing camera",
    "office": "in a modern office, neutral expression, fluorescent light, waist-up, facing camera",
    "neon_night": "on a neon-lit city street at night, wet pavement reflections, waist-up, facing camera",
    "snow_village": "in a snowy mountain village, wearing a red winter parka, rosy cold cheeks, waist-up, facing camera",
}


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


def main() -> int:
    ap = argparse.ArgumentParser(description="Fate no-plugin identity pipeline")
    ap.add_argument("--scenes", default="park,harbour")
    ap.add_argument("--seeds", default="7")
    ap.add_argument("--k", type=int, default=4, help="candidates per segment")
    ap.add_argument("--hero", default=None, help="existing same-source hero image to enroll")
    ap.add_argument("--out", default=str(ROOT / "reports" / "pipeline_run"))
    args = ap.parse_args()

    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    route = af.route("identity_no_plugin", registry=af.load_measured_registry())
    fate = FateInterface()
    pipe = load_pipe(download.IMAGE_JUG_DIR)
    manifest: dict = {"protocol": "fate-pipeline-run-v1", "route": route,
                      "k": args.k, "tau": fate.tau, "segments": []}
    chosen_paths: list[Path] = []
    try:
        # hero: generate (or designate) and enroll once
        hero_path = Path(args.hero) if args.hero else out / "hero_studio.png"
        if not args.hero:
            gen(pipe, fate.supply("studio"), hero_path, seeds[0])
        if not fate.enroll_face(hero_path):
            print(f"FATAL: no face detectable in hero {hero_path}; abstaining whole run")
            return 2
        manifest["anchor"] = str(hero_path)
        chosen_paths.append(hero_path)

        seg = 0
        for seed in seeds:
            for scene in scenes:
                spec = fate.supply_desc(SCENE_TEXT.get(scene, scene))
                pool = []
                for k in range(args.k):
                    p = out / "pool" / f"seg{seg:02d}_s{seed}_{scene}_c{k}.png"
                    gen(pipe, spec, p, 70000 + seg * 100 + k)
                    pool.append(p)
                picked = fate.select(pool)
                gate = fate.verify(Path(picked["chosen"])) if picked["chosen"] else {"passed": False}
                entry = {"segment": seg, "seed": seed, "scene": scene,
                         "fingerprint": spec["identity_fingerprint"],
                         "pool_scores": [s["cos"] for s in picked["scores"]],
                         "chosen": picked["chosen"], "chosen_cos": picked["chosen_cos"],
                         "status": "ABSTAINED" if picked["abstained"] else "ACCEPTED",
                         "verify_passed": bool(gate["passed"])}
                manifest["segments"].append(entry)
                if picked["chosen"]:
                    chosen_paths.append(Path(picked["chosen"]))
                print(seg, scene, seed, entry["pool_scores"], "->", picked["chosen_cos"], entry["status"])
                seg += 1
    finally:
        import torch
        del pipe
        torch.cuda.empty_cache()

    ok = [s["chosen_cos"] for s in manifest["segments"] if s["chosen_cos"] is not None]
    manifest["summary"] = {
        "segments": len(manifest["segments"]),
        "accepted": sum(1 for s in manifest["segments"] if s["status"] == "ACCEPTED"),
        "abstained": sum(1 for s in manifest["segments"] if s["status"] == "ABSTAINED"),
        "mean_cos": round(sum(ok) / len(ok), 4) if ok else None,
        "min_cos": min(ok) if ok else None,
        "identity_fingerprints": sorted({s["fingerprint"] for s in manifest["segments"]}),
    }
    # contact sheet: anchor + chosen frames in order
    from PIL import Image, ImageDraw
    S, LH = 224, 16
    sheet = Image.new("RGB", (S * len(chosen_paths), S + LH), (18, 18, 18))
    d = ImageDraw.Draw(sheet)
    for i, p in enumerate(chosen_paths):
        d.text((i * S + 4, 3), "anchor" if i == 0 else f"seg{i - 1}", fill=(255, 255, 0))
        sheet.paste(Image.open(p).convert("RGB").resize((S, S)), (i * S, LH))
    sheet.save(out / "contact_sheet.png")
    manifest["contact_sheet"] = str(out / "contact_sheet.png")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest["summary"], ensure_ascii=False))
    return 0 if manifest["summary"]["abstained"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
