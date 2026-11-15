#!/usr/bin/env python3
"""Free gap C: cross-window identity via I2V pixel anchoring.

CogVideoX-5b-I2V (downloaded free, 21.6 GB) takes over where T2V
selection failed (pool centred 0.25 from the anchor): W2 starts FROM
a W1 frame, so identity enters as pixels, and Fate only measures.
  start image : W1 park frame_00 (the enrolled anchor frame itself)
  prompt      : harbour scene, same detailed identity block
  takes       : seeds 21, 22 (49f, 720x480 native, 25 steps, offload)
  readout     : sampled-frame ArcFace vs the W1 anchor, per take;
                compare with the T2V takes (0.258 / 0.254 / 0.081).
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
OUT = ROOT / "reports" / "i2v_window"
MODEL = Path(r"C:\Ai\models\CogVideoX-5b-I2V")
W1 = ROOT / "reports" / "video_window_select" / "w1_park" / "frame_00.png"
BLOCK = cb.IDENTITY_A["detailed_block"]
PROMPT = (f"photorealistic video of {BLOCK}, wearing a cream sweater, walking slowly "
          f"toward the camera along a quiet harbour at dusk, same person throughout, steady camera")
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
    from diffusers import CogVideoXImageToVideoPipeline
    from PIL import Image

    fid = FaceID()
    anchor = fid.embedding(W1)
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(str(MODEL), torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    pipe.set_progress_bar_config(disable=True)
    start = Image.open(W1).convert("RGB").resize((720, 480))
    report: dict = {"protocol": "i2v-cross-window-v1", "model": "CogVideoX-5b-I2V",
                    "start_image": str(W1), "takes": []}
    chosen = None
    try:
        for i, seed in enumerate((21, 22)):
            folder = OUT / f"w2_harbour_i2v_t{i}"
            existing = sorted(folder.glob("frame_*.png"))
            started = time.time()
            if len(existing) >= FRAMES:
                frames, secs = existing, 0.0
            else:
                out = pipe(image=start, prompt=PROMPT, num_inference_steps=STEPS,
                           num_frames=FRAMES, guidance_scale=6.0,
                           generator=torch.Generator(device="cpu").manual_seed(seed))
                folder.mkdir(parents=True, exist_ok=True)
                frames = []
                for j, fr in enumerate(out.frames[0]):
                    p = folder / f"frame_{j:02d}.png"
                    fr.save(p) if hasattr(fr, "save") else Image.fromarray(fr).save(p)
                    frames.append(p)
                secs = round(time.time() - started, 1)
            s = score(fid, frames, anchor)
            report["takes"].append({"take": i, "seed": seed, "seconds": secs, **s})
            print("i2v take", i, s)
            if s["mean"] is not None and (chosen is None or s["mean"] > chosen[0]):
                chosen = (s["mean"], i, frames)
    finally:
        del pipe
        torch.cuda.empty_cache()

    if chosen:
        report["selected_mean"] = chosen[0]
        report["selected_take"] = chosen[1]
        import imageio.v2 as imageio
        w1_frames = sorted((ROOT / "reports" / "video_window_select" / "w1_park").glob("frame_*.png"))
        mp4 = OUT / "aria_i2v_window.mp4"
        writer = imageio.get_writer(str(mp4), fps=8, codec="libx264", quality=8)
        import numpy as np
        from PIL import Image as _Im
        for f in w1_frames + chosen[2]:
            writer.append_data(np.asarray(_Im.open(f).convert("RGB").resize((720, 480))))
        writer.close()
        report["video_file"] = str(mp4)
    report["reference"] = {"t2v_takes_mean": [0.2577, 0.2541, 0.0809]}
    report["verdicts"] = {"i2v_beats_t2v_pool": (report.get("selected_mean") or 0) > 0.2577,
                          "i2v_ge_0.5": (report.get("selected_mean") or 0) >= 0.5}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"selected": report.get("selected_mean"), "verdicts": report["verdicts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
