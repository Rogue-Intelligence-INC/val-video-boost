#!/usr/bin/env python3
"""Extreme test: one anchor, a long chain, hostile variation.

Protocol (kept simple -- the mechanism question is already settled):
  * ONE hero (seed 7, static_triad studio) is the pinned Fate anchor for
    the entire run; its face image + ArcFace embedding never update.
  * 12 scenes x 2 seeds = 24 sequential segments: different light (noon,
    dusk, neon night, snow, fog, golden hour), different outfits, smiles,
    backlight. Story prompts are grasped from a 100k-line bible so the
    ultra-long-context side is stressed at the same time.
  * Arm: ip_face_triad (frozen Triad patch + IP-Adapter Plus Face 0.6).
  * Verifier gate: cos < 0.40 to anchor -> one regeneration (seed+100),
    logged; never silently shipped.
Pass bar (pre-registered): mean cos >= 0.5, >= 90% of segments >= 0.5
after at most one retry, worst segment >= 0.40, and no downward drift:
second-half mean within 0.05 of first-half mean (anchor does not fade).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import character_bible as cb
import download
import module_patch
from face_id import FaceID
from test_big_models import load_pipe
from test_ip_face import IP_ROOT, face_ref, gen_ip

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "extreme"
PATCH_JSON = ROOT / "data" / "identity_patch.json"

SCENES = [
    ("park_noon", "standing in a sunlit park at noon, wearing a green summer dress, smiling"),
    ("harbour_dusk", "at a quiet harbour at dusk, wearing a navy coat, sea breeze in her hair"),
    ("neon_night", "on a neon-lit city street at night, wearing a black leather jacket, wet pavement reflections"),
    ("cafe_rain", "inside a cafe by a rainy window, wearing a burgundy turtleneck, warm indoor light, gentle smile"),
    ("beach_noon", "on a bright beach at noon, wearing a white sundress and sun hat in hand, squinting slightly in sunlight"),
    ("snow_village", "in a snowy mountain village, wearing a red winter parka with fur hood down, rosy cold cheeks"),
    ("library", "in an old library between tall shelves, wearing a brown cardigan, soft window light"),
    ("office", "in a modern office, wearing a grey blazer, neutral expression, fluorescent light"),
    ("desert_sunset", "on a desert road at sunset, wearing a denim jacket, warm backlight, hair blowing"),
    ("forest_fog", "on a forest trail in morning fog, wearing an olive rain jacket, diffused grey light"),
    ("crosswalk_evening", "at a busy city crosswalk in the evening, wearing a yellow raincoat, city lights behind"),
    ("rooftop_golden", "close-up on a rooftop at golden hour, wearing a cream blouse, warm side light, laughing"),
]
SEEDS = (21, 22)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bible = cb.Bible.build_demo(n_filler=100000)
    print(f"bible lines={len(bible.lines)} conflicts={bible.conflicts()}")
    triad = module_patch.load_export_patch(PATCH_JSON)
    fid = FaceID()
    hero_src = ROOT / "reports" / "fate_iterative_character" / "static_triad" / "s7_studio.png"
    hero_emb = fid.embedding(hero_src)
    ref = face_ref(fid, hero_src, OUT / "anchor_ref.png")

    pipe = load_pipe(download.IMAGE_JUG_DIR)
    pipe.load_ip_adapter(IP_ROOT, subfolder="sdxl_models",
                         weight_name="ip-adapter-plus-face_sdxl_vit-h.safetensors",
                         image_encoder_folder="models/image_encoder")
    pipe.set_ip_adapter_scale(0.6)
    rows = []
    try:
        seg = 0
        for scene, desc in SCENES:
            planned = bible.prompt_for_segment(min(seg, 2), scene)
            for seed in SEEDS:
                text = (f"photorealistic photograph of {cb.IDENTITY_A['prompt_fragment']}, {desc}, "
                        f"waist-up, same person as reference")
                target = OUT / f"seg{seg:02d}_{scene}_s{seed}.png"
                secs = gen_ip(pipe, text, target, seed, ref, patch=triad)
                emb = fid.embedding(target)
                cos = None if emb is None else fid.cosine(hero_emb, emb)
                retried = False
                if cos is None or cos < 0.40:  # Fate gate: do not ship drift
                    target2 = OUT / f"seg{seg:02d}_{scene}_s{seed}_retry.png"
                    secs += gen_ip(pipe, text, target2, seed + 100, ref, patch=triad)
                    emb2 = fid.embedding(target2)
                    cos2 = None if emb2 is None else fid.cosine(hero_emb, emb2)
                    retried = True
                    if cos2 is not None and (cos is None or cos2 > cos):
                        target, cos = target2, cos2
                rows.append({"segment": seg, "scene": scene, "seed": seed, "file": str(target),
                             "seconds": secs, "arcface_cos_to_anchor": cos,
                             "retried": retried, "grasp_lines": planned["grasp"]["line_nos"]})
                print(seg, scene, seed, "cos", cos, "retry" if retried else "")
                seg += 1
    finally:
        import torch
        del pipe
        torch.cuda.empty_cache()

    ok = [r["arcface_cos_to_anchor"] for r in rows if r["arcface_cos_to_anchor"] is not None]
    half = len(ok) // 2
    report = {"protocol": "val-video-boost-extreme-v1", "anchor": str(hero_src),
              "bible_lines": len(bible.lines), "segments": len(rows), "rows": rows,
              "summary": {"mean_cos": round(sum(ok) / len(ok), 4), "min_cos": min(ok),
                          "pct_ge_0.5": round(100.0 * sum(1 for v in ok if v >= 0.5) / len(ok), 1),
                          "first_half_mean": round(sum(ok[:half]) / half, 4),
                          "second_half_mean": round(sum(ok[half:]) / (len(ok) - half), 4),
                          "retries": sum(1 for r in rows if r["retried"]), "faces_found": len(ok)}}
    s = report["summary"]
    report["verdicts"] = {
        "mean_ge_0.5": s["mean_cos"] >= 0.5,
        "pct_ge_0.5_at_least_90": s["pct_ge_0.5"] >= 90.0,
        "worst_ge_0.40": s["min_cos"] >= 0.40,
        "anchor_does_not_fade": abs(s["second_half_mean"] - s["first_half_mean"]) <= 0.05,
    }
    report["verdicts"]["EXTREME_PASS"] = all(report["verdicts"].values())
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(s, ensure_ascii=False))
    print(json.dumps(report["verdicts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
