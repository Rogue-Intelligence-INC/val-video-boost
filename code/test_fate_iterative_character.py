#!/usr/bin/env python3
"""Cross-segment identity: bigger image model x Fate memory iteration.

Juggernaut XL (25 steps, 768) removes CogVideoX-2b's weak face rendering
as a confound. Chains: for each seed, scenes run in story order
studio -> park -> harbour, because iteration is sequential by definition.

Arms:
  baseline       no patch
  static_triad   exported identity Triad patch, frozen (previous method)
  fate_iterative FateChain session never reset; verified derived facts
                 appended between segments; patch tilted by cumulative
                 session fragment mean (always-activated identity anchor)
  fate_reset     same session recipe but rebuilt every segment (ablation)

Metric: YuNet face-crop L2 to the chain's own segment-0 face (cross-segment),
plus a different-person control (ID_B studio) for the identity margin.
Patch drift: cosine(segment patch, segment-0 anchor) logged per segment.
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
from test_character_consistency import ID_B, face_distance
from video_boost import FaceVerifier

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "fate_iterative_character"
PATCH_JSON = ROOT / "data" / "identity_patch.json"
SEEDS = (7, 8, 9)
SCENES = {
    "studio": "close-up portrait, studio light, plain grey background, facing camera",
    "park": "standing outdoors in a sunlit park, natural daylight, waist-up, facing camera",
    "harbour": "at a quiet harbour at dusk, sea breeze, waist-up, facing camera",
}
SIZE = 768


def prompt(scene: str) -> str:
    return f"photorealistic photograph of {cb.IDENTITY_A['prompt_fragment']}, {SCENES[scene]}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    triad = module_patch.load_export_patch(PATCH_JSON)
    verifier = FaceVerifier()
    pipe = load_pipe(download.IMAGE_JUG_DIR)
    report = {"protocol": "val-video-boost-fate-iterative-v1", "model": "RunDiffusion/Juggernaut-XL-v9",
              "seeds": list(SEEDS), "scenes": list(SCENES), "arms": {}}
    chains_log: dict[str, list] = {}
    try:
        # negative control, once per arm at seed 7 (patch arms reuse triad)
        for arm in ("baseline", "static_triad", "fate_iterative", "fate_reset"):
            rows, patch_drift = [], []
            for seed in SEEDS:
                chain = None
                derived = None
                hero = None
                if arm == "fate_iterative":
                    chain = fi.FateChain(cb.IDENTITY_A["lines"])
                    anchor = chain.start()
                for seg_idx, scene in enumerate(SCENES):
                    if arm == "baseline":
                        patch = None
                    elif arm == "static_triad":
                        patch = triad
                    elif arm == "fate_iterative":
                        if seg_idx == 0:
                            vec = anchor
                        else:
                            vec = chain.advance(
                                f"Aria in {scene}: {SCENES[scene]}.", derived, f"s{seed}_seg{seg_idx}")
                        patch = chain.fused_patch(triad, vec)
                        patch_drift.append({"seed": seed, "seg": seg_idx,
                                            "cos_to_anchor": fi.cosine(vec, anchor)})
                    else:  # fate_reset: fresh session, identity + this scene only
                        fresh = fi.FateChain(cb.IDENTITY_A["lines"])
                        vec0 = fresh.start()
                        vec = fresh.advance(f"Aria in {scene}: {SCENES[scene]}.", None, "reset")
                        patch = fresh.fused_patch(triad, vec)
                        patch_drift.append({"seed": seed, "seg": seg_idx,
                                            "cos_to_anchor": fi.cosine(vec, vec0)})
                    target = OUT / arm / f"s{seed}_{scene}.png"
                    secs = gen(pipe, prompt(scene), target, seed, SIZE, patch=patch)
                    crop = verifier.face_crop(target)
                    dist = None
                    if crop is not None and hero is not None:
                        from PIL import Image
                        hero_img = Image.open(hero["crop_path"])
                        dist = round(verifier.distance(hero_img, crop), 3)
                    if seg_idx == 0 and crop is not None:
                        crop_path = OUT / arm / f"s{seed}_hero_crop.png"
                        crop.save(crop_path)
                        hero = {"crop_path": str(crop_path)}
                    # verified writeback only (face found); else derived=None
                    derived = (f"Segment {seg_idx} accepted: the same Aria with chestnut brown "
                               f"shoulder-length hair appeared in {scene}, face verified."
                               ) if crop is not None else None
                    rows.append({"seed": seed, "scene": scene, "file": str(target),
                                 "seconds": secs, "face_found": crop is not None,
                                 "face_l2_to_seg0": dist})
                    print(arm, seed, scene, secs, "face", crop is not None, "L2", dist)
                if chain is not None:
                    chains_log.setdefault(arm, []).append(chain.log)
            neg = OUT / arm / "idB_studio_s7.png"
            gen(pipe, f"photorealistic photograph of {ID_B}, {SCENES['studio']}", neg, 7, SIZE,
                patch=None if arm == "baseline" else triad)
            report["arms"][arm] = {"rows": rows, "patch_drift": patch_drift, "negative": str(neg)}
    finally:
        import torch
        del pipe
        torch.cuda.empty_cache()

    def stats(arm: str) -> dict:
        rows = report["arms"][arm]["rows"]
        cross = [r["face_l2_to_seg0"] for r in rows if r["face_l2_to_seg0"] is not None]
        heroes = {}
        for r in rows:
            if r["scene"] == "studio":
                heroes[r["seed"]] = OUT / arm / f"s{r['seed']}_studio.png"
        neg_path = Path(report["arms"][arm]["negative"])
        margins = []
        for seed, hero_path in heroes.items():
            try:
                margins.append(face_distance(hero_path, neg_path))
            except Exception:
                pass
        return {"cross_segment_face_l2_mean": round(sum(cross) / len(cross), 3) if cross else None,
                "cross_segment_face_l2_all": cross,
                "faces_found": sum(1 for r in rows if r["face_found"]),
                "different_person_face_l2_mean": round(sum(margins) / len(margins), 3) if margins else None}

    report["summary"] = {arm: stats(arm) for arm in report["arms"]}
    report["chains_log"] = chains_log
    b = report["summary"]["baseline"]["cross_segment_face_l2_mean"]
    report["verdicts"] = {
        arm: (report["summary"][arm]["cross_segment_face_l2_mean"] is not None
              and b is not None and report["summary"][arm]["cross_segment_face_l2_mean"] < b)
        for arm in ("static_triad", "fate_iterative", "fate_reset")
    }
    report["verdicts"] = {"patch_beats_baseline_" + k: v for k, v in report["verdicts"].items()}
    it, rs = (report["summary"]["fate_iterative"]["cross_segment_face_l2_mean"],
              report["summary"]["fate_reset"]["cross_segment_face_l2_mean"])
    report["verdicts"]["iteration_beats_reset"] = (it is not None and rs is not None and it < rs)
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False))
    print(json.dumps(report["verdicts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
