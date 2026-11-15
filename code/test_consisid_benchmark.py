#!/usr/bin/env python3
"""Competitor head-to-head (free, same judge): ConsisID vs our stack.

ConsisID (CVPR 2025, diffusers ConsisIDPipeline, weights
BestWishYsh/ConsisID-preview) is the open designated-face VIDEO
competitor: one face image in, identity-preserved video out.
Protocol, aligned with our own video tests:
  designated face : the same Aria hero (detailed-supply studio frame)
  task            : harbour + park video, 49 frames, detailed block
  judge           : ArcFace of sampled frames vs the hero embedding
                    (identical FaceID, identical threshold as ours)
Our comparable arms on record: Fate T2V selection pool (0.258 vs a
video anchor -- failed), Fate + CogVideoX-5b-I2V from anchor frame
(0.633 vs that frame). ConsisID is scored against the hero directly,
which is the harsher anchor; numbers are reported with anchors named.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import character_bible as cb
from face_id import FaceID

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "competitor_consisid"
MODEL = Path(r"C:\Ai\models\ConsisID-preview")
HERO = ROOT / "reports" / "context_diagnostic" / "prompt_channel" / "detailed_neg" / "s7_studio.png"
BLOCK = cb.IDENTITY_A["detailed_block"]
SCENES = {
    "harbour": f"photorealistic video of {BLOCK}, wearing a cream sweater, walking slowly toward the camera along a quiet harbour at dusk, same person throughout, steady camera",
    "park": f"photorealistic video of {BLOCK}, wearing a cream sweater, walking slowly toward the camera through a quiet park at golden hour, same person throughout, steady camera",
}
STEPS, FRAMES = 25, 49


def score(fid, frames, anchor):
    vals = []
    for i in range(0, len(frames), 6):
        e = fid.embedding(frames[i])
        vals.append(None if (e is None or anchor is None) else round(fid.cosine(anchor, e), 4))
    ok = [v for v in vals if v is not None]
    return {"samples": vals, "mean": round(sum(ok) / len(ok), 4) if ok else None, "faces": len(ok)}


def main() -> None:
    import torch
    from diffusers import ConsisIDPipeline
    from PIL import Image

    fid = FaceID()
    anchor = fid.embedding(HERO)
    pipe = ConsisIDPipeline.from_pretrained(str(MODEL), torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    pipe.set_progress_bar_config(disable=True)
    from diffusers.pipelines.consisid.consisid_utils import (
        prepare_face_models, process_face_embeddings_infer)
    helpers = prepare_face_models(str(MODEL), device="cuda", dtype=torch.bfloat16)
    (face_helper_1, face_helper_2, face_clip_model, face_main_model,
     eva_mean, eva_std) = helpers
    id_cond, id_vit_hidden, face, face_kps = process_face_embeddings_infer(
        face_helper_1, face_clip_model, face_helper_2, eva_mean, eva_std,
        face_main_model, "cuda", torch.bfloat16, str(HERO), is_align_face=True)
    report: dict = {"protocol": "competitor-consisid-headtohead-v1",
                    "competitor": "ConsisID-preview (CogVideoX-5B, diffusers ConsisIDPipeline)",
                    "designated_face": str(HERO), "scenes": {}}
    try:
        for scene, prompt in SCENES.items():
            folder = OUT / scene
            existing = sorted(folder.glob("frame_*.png"))
            started = time.time()
            if len(existing) >= FRAMES:
                frames, secs = existing, 0.0
            else:
                out = pipe(image=face, prompt=prompt, num_inference_steps=STEPS,
                           num_frames=FRAMES, guidance_scale=6.0, height=480, width=720,
                           id_vit_hidden=id_vit_hidden, id_cond=id_cond, kps_cond=face_kps,
                           generator=torch.Generator(device="cpu").manual_seed(21))
                folder.mkdir(parents=True, exist_ok=True)
                frames = []
                for j, fr in enumerate(out.frames[0]):
                    p = folder / f"frame_{j:02d}.png"
                    fr.save(p) if hasattr(fr, "save") else Image.fromarray(fr).save(p)
                    frames.append(p)
                secs = round(time.time() - started, 1)
            s = score(fid, frames, anchor)
            report["scenes"][scene] = {"seconds": secs, **s}
            print(scene, s)
    finally:
        del pipe
        torch.cuda.empty_cache()
    means = [v["mean"] for v in report["scenes"].values() if v["mean"] is not None]
    report["summary"] = {"mean_of_scenes": round(sum(means) / len(means), 4) if means else None}
    report["ours_reference"] = {
        "fate_image_select_harbour_park": 0.657,
        "fate_plus_i2v_vs_video_anchor": 0.633,
        "fate_t2v_selection_vs_video_anchor": 0.2577,
        "note": "anchors differ by design (hero still vs video frame); judge and threshold identical"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
