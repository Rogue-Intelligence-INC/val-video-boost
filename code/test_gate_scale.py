#!/usr/bin/env python3
"""Disentangle gate SHAPE from gate SCALE (follow-up to fate_profile).

fate_profile scored 0.6201, but with quad_layer_scales ~= 1.000 its
formula collapses to a near-uniform ~0.30 -- i.e. half the usual IP
scale. Arms here separate the two explanations:
  uniform_0.30   plain half scale
  uniform_0.45   intermediate
  block_matched  mean-matched 0.297 with measured region shape
                 (down weak / mid strong, from the down_only ablation)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import download
from face_id import FaceID
from test_big_models import load_pipe
from test_fate_iterative_character import SEEDS, prompt
from test_ip_face import IP_ROOT, face_ref
from test_valhalla_native import OUT as NATIVE_OUT
from test_valhalla_native import gen_plain, region_of

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "valhalla_native"


def set_scales(pipe, fn):
    for name, proc in pipe.unet.attn_processors.items():
        if hasattr(proc, "scale"):
            proc.scale = [fn(name)]


def main():
    fid = FaceID()
    pipe = load_pipe(download.IMAGE_JUG_DIR)
    pipe.load_ip_adapter(IP_ROOT, subfolder="sdxl_models",
                         weight_name="ip-adapter-plus-face_sdxl_vit-h.safetensors",
                         image_encoder_folder="models/image_encoder")
    arms = {
        "ip_gate_uniform_0.30": lambda name: 0.30,
        "ip_gate_uniform_0.45": lambda name: 0.45,
        "ip_gate_block_matched": lambda name: {"down": 0.15, "mid": 0.42, "up": 0.36}[region_of(name)],
    }
    report = {"protocol": "val-video-boost-gate-scale-v1", "arms": {}}
    refs = {}
    try:
        for arm, fn in arms.items():
            set_scales(pipe, fn)
            rows = []
            for seed in SEEDS:
                hero = ROOT / "reports" / "fate_iterative_character" / "static_triad" / f"s{seed}_studio.png"
                if seed not in refs:
                    refs[seed] = face_ref(fid, hero, OUT / "refs" / f"s{seed}_ref.png")
                hero_emb = fid.embedding(hero)
                for scene in ("park", "harbour"):
                    target = OUT / arm / f"s{seed}_{scene}.png"
                    gen_plain(pipe, prompt(scene), target, seed, ref=refs[seed])
                    emb = fid.embedding(target)
                    cos = None if (emb is None or hero_emb is None) else fid.cosine(hero_emb, emb)
                    rows.append({"seed": seed, "scene": scene, "arcface_cos": cos})
                    print(arm, seed, scene, cos)
            ok = [r["arcface_cos"] for r in rows if r["arcface_cos"] is not None]
            report["arms"][arm] = {"rows": rows, "mean_cos": round(sum(ok) / len(ok), 4), "min_cos": min(ok)}
    finally:
        import torch
        del pipe
        torch.cuda.empty_cache()
    base = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    combined = {a: v["mean_cos"] for a, v in base["arms"].items()}
    combined.update({a: v["mean_cos"] for a, v in report["arms"].items()})
    report["combined"] = combined
    (OUT / "report_gate_scale.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(combined, ensure_ascii=False))


if __name__ == "__main__":
    main()
