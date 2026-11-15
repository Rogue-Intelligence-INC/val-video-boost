#!/usr/bin/env python3
"""Fate iteration v2 -- re-ingestion over a cumulative corpus.

v1 finding (report fate_iterative_character): a persistent Fate session
ingests appended lines (corpus 5->7->9) but closes ZERO new fragments, so
the generation patch never moves (cos=1.0, iterative == reset pixel-for-
pixel). The always-activated identity fragments are frozen, not iterating.

v2 therefore iterates the way the engine actually re-forms structure:
each segment re-ingests the WHOLE cumulative corpus -- identity lines
(always present, always re-activated) + verified derived facts from all
previous segments + this segment's scene line -- and Fate closes fresh
fragments over it (module_patch.fragment_patch, two turns). The patch
tilt (0.65 triad + 0.35 session mean) is identical to v1; only the
memory pathway changes. Ablation `scene_only` re-ingests identity+scene
without derived facts, isolating the writeback effect.
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
from video_boost import FaceVerifier

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "fate_iterative_character"
V1 = OUT / "report.json"
PATCH_JSON = ROOT / "data" / "identity_patch.json"
SIZE = 768


def session_vector(lines: list[str]) -> list[float]:
    patch = module_patch.fragment_patch(lines)
    return fi._norm(patch["patch_vector"]), patch


def main():
    triad = module_patch.load_export_patch(PATCH_JSON)
    verifier = FaceVerifier()
    pipe = load_pipe(download.IMAGE_JUG_DIR)
    report = {"protocol": "val-video-boost-fate-reingest-v2", "arms": {}}
    try:
        for arm in ("fate_reingest", "scene_only"):
            rows, drift = [], []
            for seed in SEEDS:
                derived: list[str] = []
                anchor_vec, hero = None, None
                for seg_idx, scene in enumerate(SCENES):
                    scene_line = f"Aria in {scene}: {SCENES[scene]}."
                    lines = list(cb.IDENTITY_A["lines"])
                    if arm == "fate_reingest":
                        lines += derived
                    lines.append(scene_line)
                    vec, sess_patch = session_vector(lines)
                    if seg_idx == 0:
                        anchor_vec = vec
                    patch = dict(triad)
                    patch["patch_vector"] = fi._norm(
                        [0.65 * a + 0.35 * b for a, b in zip(triad["patch_vector"], vec)])
                    patch["body_source"] = arm
                    drift.append({"seed": seed, "seg": seg_idx,
                                  "cos_to_anchor": fi.cosine(vec, anchor_vec),
                                  "session_fragments": sess_patch.get("fragments")})
                    target = OUT / arm / f"s{seed}_{scene}.png"
                    secs = gen(pipe, prompt(scene), target, seed, SIZE, patch=patch)
                    crop = verifier.face_crop(target)
                    dist = None
                    if crop is not None and hero is not None:
                        from PIL import Image
                        dist = round(verifier.distance(Image.open(hero), crop), 3)
                    if seg_idx == 0 and crop is not None:
                        hero = OUT / arm / f"s{seed}_hero_crop.png"
                        crop.save(hero)
                    if crop is not None:
                        derived.append(f"Segment {seg_idx} accepted: the same Aria with chestnut "
                                       f"brown shoulder-length hair appeared in {scene}, face verified.")
                    rows.append({"seed": seed, "scene": scene, "seconds": secs,
                                 "face_found": crop is not None, "face_l2_to_seg0": dist})
                    print(arm, seed, scene, "L2", dist, "cos", drift[-1]["cos_to_anchor"])
            report["arms"][arm] = {"rows": rows, "patch_drift": drift}
    finally:
        import torch
        del pipe
        torch.cuda.empty_cache()

    def mean_cross(arm):
        vals = [r["face_l2_to_seg0"] for r in report["arms"][arm]["rows"] if r["face_l2_to_seg0"] is not None]
        return round(sum(vals) / len(vals), 3), vals

    v1 = json.loads(V1.read_text(encoding="utf-8")) if V1.exists() else {}
    combined = {"baseline": v1.get("summary", {}).get("baseline", {}).get("cross_segment_face_l2_mean"),
                "static_triad": v1.get("summary", {}).get("static_triad", {}).get("cross_segment_face_l2_mean"),
                "fate_session_frozen_v1": v1.get("summary", {}).get("fate_iterative", {}).get("cross_segment_face_l2_mean")}
    for arm in ("fate_reingest", "scene_only"):
        m, vals = mean_cross(arm)
        report["arms"][arm]["cross_segment_face_l2_mean"] = m
        report["arms"][arm]["cross_segment_face_l2_all"] = vals
        combined[arm] = m
    report["combined_cross_segment_face_l2"] = combined
    best_patch = min((v for k, v in combined.items() if k != "baseline" and v), default=None)
    report["verdicts"] = {
        "reingest_beats_static_triad": combined["fate_reingest"] is not None and combined["static_triad"] is not None and combined["fate_reingest"] < combined["static_triad"],
        "reingest_beats_scene_only": combined["fate_reingest"] < combined["scene_only"],
        "reingest_patch_actually_moves": any(d["cos_to_anchor"] < 0.999 for a in report["arms"].values() for d in a["patch_drift"]),
        "all_patch_arms_beat_baseline": all(v < combined["baseline"] for k, v in combined.items() if k != "baseline" and v),
    }
    (OUT / "report_v2.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(combined, ensure_ascii=False))
    print(json.dumps(report["verdicts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
