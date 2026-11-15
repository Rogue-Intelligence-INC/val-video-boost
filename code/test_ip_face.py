#!/usr/bin/env python3
"""Mechanism completion: reference-face conditioning (IP-Adapter Plus Face).

Previous arms all route identity through a 256-d text-side patch. ArcFace
rescoring (reports/fate_iterative_character/arcface_rescore.json) shows
that channel locks attributes, not identity: baseline cos 0.471, every
patch arm 0.417-0.433, different-person control 0.15-0.19.

This arm feeds the ACCEPTED hero face itself into the denoiser via the
local IP-Adapter Plus Face SDXL weights (ViT-H face encoder), optionally
stacked with the frozen Triad patch. Fate's role becomes the always-on
memory around it: hero is generated once, ArcFace-verified, and its
embedding + image are the pinned anchor every later segment reads.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import download
import module_patch
from face_id import FaceID
from test_big_models import load_pipe
from test_fate_iterative_character import SCENES, SEEDS, prompt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "fate_iterative_character"
PATCH_JSON = ROOT / "data" / "identity_patch.json"
IP_ROOT = r"C:\Ai\models\ip-adapter"
SIZE, STEPS = 768, 25
IP_SCALE = 0.6


def face_ref(fid: FaceID, hero_path: Path, out_path: Path):
    import cv2
    from PIL import Image

    img = cv2.imread(str(hero_path))
    faces = fid.app.get(img)
    if not faces:
        return None
    f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    x0, y0, x1, y1 = (int(v) for v in f.bbox)
    m = int(0.35 * max(x1 - x0, y1 - y0))
    x0, y0 = max(x0 - m, 0), max(y0 - m, 0)
    x1, y1 = min(x1 + m, img.shape[1]), min(y1 + m, img.shape[0])
    crop = Image.fromarray(cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)).resize((512, 512))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out_path)
    return out_path


def gen_ip(pipe, text: str, out_path: Path, seed: int, ref, patch=None) -> float:
    import torch
    from PIL import Image

    generator = torch.Generator(device="cuda").manual_seed(seed)
    started = time.time()
    handle = None
    if patch is not None:
        handle = module_patch.ModulePatch(patch, mode="total")
        handle.attach(pipe.unet, "unet")
    try:
        image = pipe(prompt=text, num_inference_steps=STEPS, guidance_scale=7.0,
                     generator=generator, height=SIZE, width=SIZE,
                     ip_adapter_image=Image.open(ref).convert("RGB")).images[0]
    finally:
        if handle is not None:
            handle.remove()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    return round(time.time() - started, 2)


def main():
    triad = module_patch.load_export_patch(PATCH_JSON)
    fid = FaceID()
    pipe = load_pipe(download.IMAGE_JUG_DIR)
    pipe.load_ip_adapter(IP_ROOT, subfolder="sdxl_models",
                         weight_name="ip-adapter-plus-face_sdxl_vit-h.safetensors",
                         image_encoder_folder="models/image_encoder")
    pipe.set_ip_adapter_scale(IP_SCALE)
    report = {"protocol": "val-video-boost-ipface-v1", "ip_scale": IP_SCALE, "arms": {}}
    try:
        for arm, patch in (("ip_face", None), ("ip_face_triad", triad)):
            rows = []
            for seed in SEEDS:
                hero_src = OUT / "static_triad" / f"s{seed}_studio.png"
                ref = face_ref(fid, hero_src, OUT / arm / f"s{seed}_ref.png")
                hero_emb = fid.embedding(hero_src)
                for scene in ("park", "harbour"):
                    target = OUT / arm / f"s{seed}_{scene}.png"
                    secs = gen_ip(pipe, prompt(scene), target, seed, ref, patch=patch)
                    emb = fid.embedding(target)
                    cos = None if (emb is None or hero_emb is None) else fid.cosine(hero_emb, emb)
                    rows.append({"seed": seed, "scene": scene, "seconds": secs,
                                 "arcface_cos_to_hero": cos})
                    print(arm, seed, scene, secs, "cos", cos)
            ok = [r["arcface_cos_to_hero"] for r in rows if r["arcface_cos_to_hero"] is not None]
            report["arms"][arm] = {"rows": rows, "mean_cos": round(sum(ok) / len(ok), 4) if ok else None,
                                   "min_cos": min(ok) if ok else None, "found": len(ok)}
    finally:
        import torch
        del pipe
        torch.cuda.empty_cache()
    rescore = json.loads((OUT / "arcface_rescore.json").read_text(encoding="utf-8"))
    report["comparison_arcface_mean"] = {
        "baseline": rescore["baseline"]["mean"], "static_triad": rescore["static_triad"]["mean"],
        **{a: report["arms"][a]["mean_cos"] for a in report["arms"]},
        "different_person_control": rescore["baseline"]["neg_cos"]}
    report["verdicts"] = {
        a: (report["arms"][a]["mean_cos"] is not None and report["arms"][a]["mean_cos"] > 0.5
            and report["arms"][a]["mean_cos"] > rescore["baseline"]["mean"])
        for a in report["arms"]}
    (OUT / "report_ipface.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["comparison_arcface_mean"], ensure_ascii=False))
    print(json.dumps(report["verdicts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
