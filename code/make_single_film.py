#!/usr/bin/env python3
"""One-shot coherent film: a single diffusion pass, no segments, no stitch.

Valhalla does not render pixels (it is the memory/gate/patch layer, as
the LLM is not its retrieval). The honest one-shot test is therefore:
the longest single continuous take the local video model can hold in
one latent window (CogVideoX-2b: 49 frames), with the frozen Triad
identity patch applied to every block of that single pass, one seed,
one prompt describing one continuous shot.

Consistency is then measured INSIDE the take: ArcFace cosine of sampled
frames against the take's own first detected face, first frame to last.
If a single window holds the person end to end, any longer film is a
sequence of such windows -- which is exactly the long-context problem
Valhalla already solves for text (external anchor, not bigger windows).
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
OUT = ROOT / "reports" / "single_film"
PATCH_JSON = ROOT / "data" / "identity_patch.json"
PROMPT = (f"one continuous shot, photorealistic video of {cb.IDENTITY_A['prompt_fragment']}, "
          "wearing a cream sweater, walking slowly toward the camera through a quiet park "
          "at golden hour, camera dollies backward smoothly, she keeps looking at the camera, "
          "same person throughout, steady continuous take")
SEED, STEPS, FRAMES = 7, 25, 49


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    triad = module_patch.load_export_patch(PATCH_JSON)
    pipe = module_patch._load_video()
    try:
        # 512x288 measured: 720x480 at 49 frames thrashes shared GPU memory
        # (~50-200 s/step); 512x288 runs ~5 s/step and keeps the whole take.
        frames, secs, info = vb.generate_segment(
            pipe, PROMPT, OUT / "take", SEED, triad, mode="total",
            steps=STEPS, num_frames=FRAMES, height=288, width=512)
    finally:
        import torch
        del pipe
        torch.cuda.empty_cache()
    print(f"generated {len(frames)} frames in {secs}s patch={info}")

    fid = FaceID()
    samples = list(range(0, len(frames), 4)) + [len(frames) - 1]
    anchor_emb, rows = None, []
    for i in samples:
        emb = fid.embedding(frames[i])
        if anchor_emb is None and emb is not None:
            anchor_emb = emb
        cos = None if (emb is None or anchor_emb is None) else fid.cosine(anchor_emb, emb)
        rows.append({"frame": i, "face": emb is not None, "cos_to_first_face": cos})
        print(i, rows[-1])
    ok = [r["cos_to_first_face"] for r in rows if r["cos_to_first_face"] is not None]
    metrics = vb.clip_metrics(frames)

    import imageio.v2 as imageio
    mp4 = OUT / "aria_one_take.mp4"
    writer = imageio.get_writer(str(mp4), fps=8, codec="libx264", quality=8)
    for f in frames:
        writer.append_data(imageio.imread(str(f)))
    writer.close()
    from PIL import Image
    sheet = Image.new("RGB", (360 * len(samples), 203), "white")
    for j, i in enumerate(samples):
        sheet.paste(Image.open(frames[i]).resize((360, 203)), (360 * j, 0))
    sheet.save(OUT / "contact_take.png")

    report = {"protocol": "val-video-boost-single-take-v1", "model": "THUDM/CogVideoX-2b",
              "prompt": PROMPT, "seed": SEED, "steps": STEPS, "frames": len(frames),
              "seconds": secs, "patch_sites": info, "clip_metrics": metrics,
              "frame_samples": rows, "video_file": str(mp4),
              "summary": {"faces_found": sum(1 for r in rows if r["face"]),
                          "mean_cos_to_first": round(sum(ok) / len(ok), 4) if ok else None,
                          "min_cos_to_first": min(ok) if ok else None,
                          "first_cos": ok[0] if ok else None, "last_cos": ok[-1] if ok else None}}
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False))
    print(f"video: {mp4}")


if __name__ == "__main__":
    main()
