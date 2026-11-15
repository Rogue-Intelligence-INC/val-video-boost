#!/usr/bin/env python3
"""Fate iteration v3 -- reinforcement writeback (identity-only derived facts).

v2 diagnosis: derived facts contained scene words ("appeared in park..."),
so each writeback tilted the session mean AWAY from identity (cos 0.989)
and cross-segment L2 (44.93) stayed behind the frozen static patch (42.93).
v3 writeback is a pure identity restatement with no scene vocabulary, so
the cumulative corpus's identity fraction GROWS every segment -- the
always-activated part gets re-activated, the scene line stays a minority
of one. Same tilt, same seeds/scenes/model as v1/v2 for direct comparison.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import character_bible as cb
import download
import fate_iterative as fi
import module_patch
from test_big_models import gen, load_pipe
from test_fate_iterative_character import SCENES, SEEDS, prompt
from test_fate_reingest_character import session_vector
from video_boost import FaceVerifier

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "fate_iterative_character"
PATCH_JSON = ROOT / "data" / "identity_patch.json"
SIZE = 768
RESTATE = ("Aria is the same single 28-year-old woman with shoulder-length chestnut brown hair, "
           "a small mole on her left cheek, hazel eyes and fair skin.")


def main():
    triad = module_patch.load_export_patch(PATCH_JSON)
    verifier = FaceVerifier()
    pipe = load_pipe(download.IMAGE_JUG_DIR)
    rows, drift = [], []
    try:
        for seed in SEEDS:
            derived, anchor_vec, hero = [], None, None
            for seg_idx, scene in enumerate(SCENES):
                lines = list(cb.IDENTITY_A["lines"]) + derived + [f"Aria in {scene}: {SCENES[scene]}."]
                vec, sess = session_vector(lines)
                if seg_idx == 0:
                    anchor_vec = vec
                patch = dict(triad)
                patch["patch_vector"] = fi._norm([0.65 * a + 0.35 * b for a, b in zip(triad["patch_vector"], vec)])
                patch["body_source"] = "fate_reinforce"
                drift.append({"seed": seed, "seg": seg_idx, "cos_to_anchor": fi.cosine(vec, anchor_vec),
                              "corpus_lines": len(lines)})
                target = OUT / "fate_reinforce" / f"s{seed}_{scene}.png"
                secs = gen(pipe, prompt(scene), target, seed, SIZE, patch=patch)
                crop = verifier.face_crop(target)
                dist = None
                if crop is not None and hero is not None:
                    from PIL import Image
                    dist = round(verifier.distance(Image.open(hero), crop), 3)
                if seg_idx == 0 and crop is not None:
                    hero = OUT / "fate_reinforce" / f"s{seed}_hero_crop.png"
                    crop.save(hero)
                if crop is not None:
                    derived.append(RESTATE)
                rows.append({"seed": seed, "scene": scene, "seconds": secs,
                             "face_found": crop is not None, "face_l2_to_seg0": dist})
                print("fate_reinforce", seed, scene, "L2", dist, "cos", drift[-1]["cos_to_anchor"])
    finally:
        import torch
        del pipe
        torch.cuda.empty_cache()
    vals = [r["face_l2_to_seg0"] for r in rows if r["face_l2_to_seg0"] is not None]
    v2 = json.loads((OUT / "report_v2.json").read_text(encoding="utf-8"))
    combined = dict(v2["combined_cross_segment_face_l2"])
    combined["fate_reinforce_v3"] = round(sum(vals) / len(vals), 3)
    report = {"protocol": "val-video-boost-fate-reinforce-v3",
              "cross_segment_face_l2_all": vals, "patch_drift": drift,
              "combined_cross_segment_face_l2": combined,
              "verdicts": {
                  "reinforce_beats_static_triad": combined["fate_reinforce_v3"] < combined["static_triad"],
                  "reinforce_beats_reingest_v2": combined["fate_reinforce_v3"] < combined["fate_reingest"],
                  "reinforce_beats_baseline": combined["fate_reinforce_v3"] < combined["baseline"]}}
    (OUT / "report_v3.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(combined, ensure_ascii=False))
    print(json.dumps(report["verdicts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
