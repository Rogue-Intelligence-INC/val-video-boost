#!/usr/bin/env python3
"""Gap 3: video cross-window loop, closed with take selection.

The old cross-window result (T2V restart, short prompt + patch) was a
0.221 seam -- another person. This closes the loop the no-plugin way:
  W1  one 49f take (park). Its first detected face is enrolled as the
      Fate anchor (same-source, never updated).
  W2  THREE 49f takes (harbour, seeds 21-23), each scored by mean
      ArcFace of sampled frames against the anchor.
  Fate's choice = the best take; blind baseline = take 1 as generated.
Prompts use the pinned detailed block (video wording); no patch, no
negative channel on CogVideoX's generate path, no plugin.
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
OUT = ROOT / "reports" / "video_window_select"
BLOCK = cb.IDENTITY_A["detailed_block"]
STEPS, FRAMES = 25, 49
PROMPTS = {
    "park": (f"photorealistic video of {BLOCK}, wearing a cream sweater, walking slowly "
             f"toward the camera through a quiet park at golden hour, same person throughout, steady camera"),
    "harbour": (f"photorealistic video of {BLOCK}, wearing a cream sweater, walking slowly "
                f"toward the camera along a quiet harbour at dusk, same person throughout, steady camera"),
}


def take_frames(pipe, name: str, scene: str, seed: int) -> tuple[list[Path], float]:
    folder = OUT / name
    existing = sorted(folder.glob("frame_*.png"))
    if len(existing) >= FRAMES:
        return existing, 0.0
    frames, secs, _ = vb.generate_segment(pipe, PROMPTS[scene], folder, seed,
                                          None, steps=STEPS, num_frames=FRAMES,
                                          height=288, width=512)
    return frames, secs


def take_score(fid, frames, anchor) -> dict:
    vals = []
    for i in range(0, len(frames), 6):
        e = fid.embedding(frames[i])
        vals.append(None if (e is None or anchor is None) else round(fid.cosine(anchor, e), 4))
    ok = [v for v in vals if v is not None]
    return {"samples": vals, "mean": round(sum(ok) / len(ok), 4) if ok else None,
            "faces": len(ok)}


def main() -> None:
    fid = FaceID()
    pipe = module_patch._load_video()
    report: dict = {"protocol": "video-window-select-v1", "model": "CogVideoX-2b"}
    chosen_frames: list[Path] = []
    try:
        w1, secs1 = take_frames(pipe, "w1_park", "park", 7)
        anchor, anchor_frame = None, None
        for f in w1:
            anchor = fid.embedding(f)
            if anchor is not None:
                anchor_frame = f
                break
        report["w1"] = {"seconds": secs1, "anchor_frame": str(anchor_frame),
                        "self_score": take_score(fid, w1, anchor)}
        takes = []
        for i, seed in enumerate((21, 22, 23)):
            frames, secs = take_frames(pipe, f"w2_harbour_t{i}", "harbour", seed)
            score = take_score(fid, frames, anchor)
            takes.append({"take": i, "seed": seed, "seconds": secs, **score,
                          "folder": str(OUT / f"w2_harbour_t{i}")})
            print("w2 take", i, score)
        ok = [t for t in takes if t["mean"] is not None]
        best = max(ok, key=lambda t: t["mean"]) if ok else None
        report["w2_takes"] = takes
        report["blind_first_take_mean"] = takes[0]["mean"] if takes else None
        report["selected_take"] = best["take"] if best else None
        report["selected_mean"] = best["mean"] if best else None
        chosen_frames = w1 + sorted((OUT / f"w2_harbour_t{best['take']}").glob("frame_*.png")) if best else w1
    finally:
        import torch
        del pipe
        torch.cuda.empty_cache()

    if chosen_frames:
        import imageio.v2 as imageio
        mp4 = OUT / "aria_window_select.mp4"
        writer = imageio.get_writer(str(mp4), fps=8, codec="libx264", quality=8)
        for f in chosen_frames:
            writer.append_data(imageio.imread(str(f)))
        writer.close()
        report["video_file"] = str(mp4)
    report["verdicts"] = {
        "selected_beats_blind": (report.get("selected_mean") is not None
                                 and report.get("blind_first_take_mean") is not None
                                 and report["selected_mean"] > report["blind_first_take_mean"]),
        "selected_ge_0.5": (report.get("selected_mean") or 0) >= 0.5,
    }
    report["reference"] = {"old_cross_window_seam_short_patch": 0.2211,
                           "within_take_single_film_49f": 0.9004}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"blind": report["blind_first_take_mean"], "selected": report["selected_mean"],
                      "take": report["selected_take"], "verdicts": report["verdicts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
