#!/usr/bin/env python3
"""Valhalla on the large local model: FLUX.1-schnell (12B, 57 blocks).

Copied over from val-videopic-fusion (load strategy: group offload
block-level + encoders on GPU only for encode_prompt; 577s slow path vs
~5s fast path). Same character protocol as the Juggernaut suite so the
numbers compare directly: seeds 7/8/9 x studio/park/harbour, ArcFace
cosine to each seed's studio hero. Arms: baseline vs frozen Triad patch
(total, adaptive strength 0.0166 taken from the prior flux plan).
Reference numbers (Juggernaut): baseline 0.471, triad 0.433, IP-face 0.581.
FLUX has no IP-Adapter on this machine -- this tests the patch channel
on the biggest denoiser available locally.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import character_bible as cb
import module_patch
from face_id import FaceID
from test_fate_iterative_character import SCENES, prompt
from test_flux_fast import gen, load_fast

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "flux_character"
PATCH_JSON = ROOT / "data" / "identity_patch.json"
SEEDS = (7, 8, 9)
STRENGTH = 0.0166


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    triad = module_patch.load_export_patch(PATCH_JSON)
    fid = FaceID()
    pipe = load_fast()
    profile = module_patch.profile_denoiser(pipe.transformer, "transformer")
    print("profile", profile)
    report = {"protocol": "val-video-boost-flux-character-v1",
              "model": "YuCollection/FLUX.1-schnell-Diffusers", "profile": profile,
              "strength": STRENGTH, "arms": {}}
    try:
        for arm, patch in (("baseline", None), ("triad_total", triad)):
            rows = []
            for seed in SEEDS:
                hero_emb = None
                for scene in SCENES:
                    target = OUT / arm / f"s{seed}_{scene}.png"
                    secs = gen(pipe, prompt(scene), target, seed,
                               patch=patch, strength=STRENGTH if patch else None)
                    emb = fid.embedding(target)
                    if scene == "studio":
                        hero_emb = emb
                    cos = None if (emb is None or hero_emb is None or scene == "studio") else fid.cosine(hero_emb, emb)
                    rows.append({"seed": seed, "scene": scene, "seconds": secs,
                                 "face": emb is not None, "arcface_cos_to_hero": cos})
                    print(arm, seed, scene, secs, "cos", cos)
            ok = [r["arcface_cos_to_hero"] for r in rows if r["arcface_cos_to_hero"] is not None]
            # negative control once, seed 7 studio, different person
            from test_character_consistency import ID_B
            neg = OUT / arm / "idB_studio_s7.png"
            gen(pipe, f"photorealistic photograph of {ID_B}, {SCENES['studio']}", neg, 7,
                patch=patch, strength=STRENGTH if patch else None)
            report["arms"][arm] = {"rows": rows, "mean_cos": round(sum(ok) / len(ok), 4) if ok else None,
                                   "min_cos": min(ok) if ok else None, "faces": sum(1 for r in rows if r["face"])}
    finally:
        import torch
        del pipe
        torch.cuda.empty_cache()
    b = report["arms"]["baseline"]["mean_cos"]
    t = report["arms"]["triad_total"]["mean_cos"]
    report["comparison"] = {"juggernaut_baseline": 0.471, "juggernaut_triad": 0.4333,
                            "juggernaut_ip_face": 0.5811, "flux_baseline": b, "flux_triad": t}
    report["verdicts"] = {"flux_beats_juggernaut_baseline": (b or 0) > 0.471,
                          "flux_patch_beats_flux_baseline": (t or 0) > (b or 0),
                          "flux_patch_reaches_ip_level": (t or 0) >= 0.5811}
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["comparison"], ensure_ascii=False))
    print(json.dumps(report["verdicts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
