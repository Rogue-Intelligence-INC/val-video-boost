#!/usr/bin/env python3
"""Ultra-long-context character consistency on CogVideoX-2b.

Question: bible ~20k lines (~150k words, and by extension the 14.95M-token
Fate scale) -- can the person stay the same person across segments?

Arms (same seed, same prompts, planner identical):
  baseline      no patch
  identity_lock pinned Triad identity patch (total) + verifier ledger
Segment 0 (studio) face = hero reference for both arms. Verdicts require:
identity margin vs the negative-identity prompt arm is not needed here --
cross-segment face distance + face-found rate + flicker are the measures,
and `cross_segment_holds` is only claimed if patched cross-segment face
L2 beats baseline AND stays under the image-suite's different-person gap.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import character_bible as cb
import module_patch
import video_boost as vb

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "long_video_character"
PATCH_JSON = ROOT / "data" / "identity_patch.json"
SEED, STEPS, FRAMES = 7, 25, 17
SCENES = [(0, "Aria studio looking at camera"),
          (1, "Aria walking in sunlit park"),
          (2, "Aria harbour dusk sea breeze")]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bible = cb.Bible.build_demo(n_filler=20000)
    conflicts = bible.conflicts()
    print(f"bible lines={len(bible.lines)} conflicts={conflicts}")
    if not PATCH_JSON.exists():
        cb.export_identity_patch(PATCH_JSON)
    identity = module_patch.load_export_patch(PATCH_JSON)
    print(f"identity patch dim={identity['dim']} tiles={len(identity.get('tile_signatures') or [])} "
          f"stems={len(identity.get('stem_signatures') or [])}")
    (OUT / "prompts.json").write_text(json.dumps(
        {str(s): bible.prompt_for_segment(s, q)["prompt"] for s, q in SCENES},
        indent=1, ensure_ascii=False), encoding="utf-8")

    pipe = module_patch._load_video()
    verifier = vb.FaceVerifier()
    report = {"protocol": "val-video-boost-longctx-character-v1",
              "model": "THUDM/CogVideoX-2b", "seed": SEED, "steps": STEPS,
              "frames_per_segment": FRAMES,
              "bible": {"lines": len(bible.lines),
                        "approx_words": sum(len(l.split()) for l in bible.lines),
                        "conflicts": conflicts,
                        "note": "context held outside diffusion model; CogVideoX T5 cap ~226 tokens"},
              "identity_patch": {"dim": identity["dim"], "tiles": identity["tile_count"],
                                 "stems": identity["stem_cell_count"]},
              "arms": {}}
    try:
        for arm, patch in (("baseline", None), ("identity_lock", identity)):
            hero_crop, rows = None, []
            for seg, query in SCENES:
                planned = bible.prompt_for_segment(seg, query)
                entry = vb.verified_segment(
                    pipe, planned["prompt"], OUT / arm / f"seg{seg}", SEED, patch,
                    verifier, hero_crop, ladder=(1.0,) if patch is None else (1.0, 0.5),
                    steps=STEPS, num_frames=FRAMES)
                crop = entry.pop("accepted_crop", None)
                if seg == 0:
                    # hero = segment-0 face if found, else its mid-frame crop attempt
                    mid = Path(entry["mid_frame"])
                    hero_crop = verifier.face_crop(mid) or crop
                    if hero_crop is not None:
                        hero_crop.save(OUT / arm / "hero_face.png")
                    entry["is_hero"] = True
                elif crop is not None and hero_crop is not None:
                    entry["face_l2_to_hero"] = round(verifier.distance(hero_crop, crop), 3)
                entry["prompt"] = planned["prompt"]
                entry["grasp_lines"] = planned["grasp"]["line_nos"]
                rows.append(entry)
                print(arm, seg, entry["status"], entry["face_l2_to_hero"], entry["seconds"], "s")
            report["arms"][arm] = rows
    finally:
        import torch
        del pipe
        torch.cuda.empty_cache()

    def cross(arm):
        vals = [r["face_l2_to_hero"] for r in report["arms"][arm]
                if r.get("face_l2_to_hero") is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    b, p = cross("baseline"), cross("identity_lock")
    report["summary"] = {
        "baseline_cross_segment_face_l2": b,
        "patched_cross_segment_face_l2": p,
        "faces_found": {a: sum(1 for r in report["arms"][a] if r["face_found"]) for a in report["arms"]},
        "accepted": {a: sum(1 for r in report["arms"][a] if r["status"] == "accepted") for a in report["arms"]},
        "flicker": {a: [r["metrics"]["flicker_std"] for r in report["arms"][a]] for a in report["arms"]},
    }
    report["verdicts"] = {
        "patch_cross_segment_tighter": (b is not None and p is not None and p < b),
        "all_faces_found_patched": report["summary"]["faces_found"]["identity_lock"] == len(SCENES),
        # honest ceiling: do not claim person-level lock from this alone
        "person_level_lock_claimed": False,
        "person_level_lock_note": "see docs/DESIGN.md ceiling: needs I2V/IP-Adapter-Face reference conditioner",
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False))
    print(f"wrote {OUT / 'report.json'}")


if __name__ == "__main__":
    main()
