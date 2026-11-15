#!/usr/bin/env python3
"""Cross-window continuation on the best local video model (CogVideoX-2b).

Whole-machine inventory (2026-11-15): only two video pipelines exist
locally -- CogVideoX-2b and text-to-video-ms-1.7b. CogVideoX-2b is the
better one and is what Valhalla connects to here (ModulePatch already
speaks CogVideoXBlock).

Continuation design (window-out running):
  * Each window is a one-take 49-frame generation (proven: identity
    0.90 inside a window, ceiling 121f, collapse at 161f).
  * Across windows Valhalla holds the anchor: frozen Triad identity
    patch + identical identity prompt core + same outfit, and the
    ArcFace verifier measures the seam (last face of W1 vs first
    faces of W2). No frame is fed forward -- CogVideoX-2b is T2V;
    true last-frame chaining needs I2V weights (not on this machine).
Simple test: 2 windows, park -> harbour, one mp4, seam measured.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import character_bible as cb
import module_patch
import video_boost as vb
from face_id import FaceID

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "long_film"
PATCH_JSON = ROOT / "data" / "identity_patch.json"
CORE = cb.IDENTITY_A["prompt_fragment"]
WINDOWS = [
    ("park", f"one continuous shot, photorealistic video of {CORE}, wearing a cream sweater, walking slowly toward the camera through a quiet park at golden hour, camera dollies backward, same person throughout"),
    ("harbour", f"one continuous shot, photorealistic video of {CORE}, wearing a cream sweater, walking slowly toward the camera along a quiet harbour at dusk, camera dollies backward, same person throughout"),
]
SEED, STEPS, FRAMES = 7, 25, 49


def face_series(fid, frames, anchor):
    vals = []
    for i in range(0, len(frames), 8):
        e = fid.embedding(frames[i])
        vals.append(None if (e is None or anchor is None) else fid.cosine(anchor, e))
    return vals


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    triad = module_patch.load_export_patch(PATCH_JSON)
    fid = FaceID()
    hero = ROOT / "reports" / "fate_iterative_character" / "static_triad" / "s7_studio.png"
    anchor_emb = fid.embedding(hero)
    pipe = module_patch._load_video()
    report = {"protocol": "val-video-boost-long-film-v1", "model": "THUDM/CogVideoX-2b",
              "windows": []}
    all_frames: list[Path] = []
    try:
        for wi, (name, prompt) in enumerate(WINDOWS):
            frames, secs, info = vb.generate_segment(
                pipe, prompt, OUT / f"w{wi}_{name}", SEED, triad, mode="total",
                steps=STEPS, num_frames=FRAMES, height=288, width=512)
            series = face_series(fid, frames, anchor_emb)
            ok = [v for v in series if v is not None]
            report["windows"].append({
                "window": wi, "scene": name, "seconds": secs,
                "arcface_to_anchor_samples": series,
                "mean": round(sum(ok) / len(ok), 4) if ok else None,
                "metrics": vb.clip_metrics(frames)})
            print(wi, name, secs, report["windows"][-1]["mean"])
            all_frames.extend(frames)
    finally:
        import torch
        del pipe
        torch.cuda.empty_cache()

    # seam: last sampled face of W1 vs first sampled faces of W2 (both vs anchor
    # already); direct boundary = mean of last-2 W1 samples vs first-2 W2 samples gap
    w = report["windows"]
    if len(w) == 2 and w[0]["mean"] and w[1]["mean"]:
        report["seam"] = {"w1_mean": w[0]["mean"], "w2_mean": w[1]["mean"],
                           "gap": round(abs(w[0]["mean"] - w[1]["mean"]), 4)}
    import imageio.v2 as imageio
    mp4 = OUT / "aria_long_film.mp4"
    writer = imageio.get_writer(str(mp4), fps=8, codec="libx264", quality=8)
    for f in all_frames:
        writer.append_data(imageio.imread(str(f)))
    writer.close()
    from PIL import Image
    sheet = Image.new("RGB", (512 * 4, 288), "white")
    picks = [all_frames[40], all_frames[48], all_frames[49], all_frames[60]]
    for j, f in enumerate(picks):
        sheet.paste(Image.open(f), (512 * j, 0))
    sheet.save(OUT / "contact_seam.png")
    report["video_file"] = str(mp4)
    report["total_frames"] = len(all_frames)
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report.get("seam"), ensure_ascii=False))
    print(f"video: {mp4} frames={len(all_frames)}")


if __name__ == "__main__":
    main()
