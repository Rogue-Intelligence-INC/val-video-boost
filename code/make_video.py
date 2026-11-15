#!/usr/bin/env python3
"""End-to-end: Fate learns the identity, then a video is generated.

Pipeline (every stage measured, nothing patched over):
  1. LEARN   Fate iterative learning (error-driven phrasing coverage,
             v5): identity facts taught as QA variants until the train
             probes pass; learning curve recorded.
  2. PLAN    segment prompts assembled from the identity core + scene
             graph (studio -> park -> harbour -> rooftop).
  3. GENERATE CogVideoX-2b per segment, frozen Triad identity patch
             (total 0.015 -- the measured ceiling, never doubled).
  4. VERIFY  ArcFace gate per segment against the pinned hero anchor
             (Juggernaut keyframe face). Below tau=0.35: one retry with
             a new seed; the better take is kept and logged either way.
  5. ASSEMBLE accepted frames -> mp4 (imageio-ffmpeg) + contact sheet.
Known ceiling carried honestly: CogVideoX-2b is text-only, so cross-
segment pixel identity is bounded (measured before); the video is the
test artefact, the report says exactly how far it held.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bridge
import character_bible as cb
import module_patch
import video_boost as vb
from face_id import FaceID
from test_fate_iterative_learning import ACCEPT
from test_fate_iterative_learning4 import HOLDOUT_Q, TRAIN_Q, VARIANTS
from test_fate_learning import DECOYS, IDENTITY, filler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "generated_video"
PATCH_JSON = ROOT / "data" / "identity_patch.json"
TAU = 0.35
SEGMENTS = [
    ("studio", f"photorealistic video of {cb.IDENTITY_A['prompt_fragment']}, wearing a cream sweater, in a grey studio, looking at the camera, waist-up, steady camera"),
    ("park", f"photorealistic video of {cb.IDENTITY_A['prompt_fragment']}, wearing a green dress, walking slowly in a sunlit park, facing camera, steady camera"),
    ("harbour", f"photorealistic video of {cb.IDENTITY_A['prompt_fragment']}, wearing a navy coat, at a quiet harbour at dusk, facing camera, sea breeze, steady camera"),
    ("rooftop", f"photorealistic video of {cb.IDENTITY_A['prompt_fragment']}, wearing a cream blouse, on a rooftop at golden hour, smiling at camera, steady camera"),
]
SEED = 7
STEPS, FRAMES = 25, 17


def learn() -> dict:
    base = IDENTITY + DECOYS + filler(300)
    r = bridge.valhalla(bridge.build_payload(TRAIN_Q[0][1], {}, None,
                                             [{"text": t} for t in base], None, bridge.CFG))
    session, lines_now = r.get("session"), list(base)
    taught = {k: 0 for k, _ in TRAIN_Q}
    curve = []

    def probe(questions):
        nonlocal session
        out = {}
        for key, q in questions:
            rr = bridge.valhalla(bridge.build_payload(q, {}, session, None, None, bridge.CFG))
            if isinstance(rr.get("session"), dict):
                session = rr["session"]
            row = (rr.get("rows") or [{}])[0]
            g = row.get("fate_grasp") or {}
            idx = g.get("grasped_idx", row.get("retrieval_corpus_idx"))
            text = lines_now[idx] if isinstance(idx, int) and 0 <= idx < len(lines_now) else None
            out[key] = bool(text and ACCEPT[key](text))
        return out

    for rd in range(12):
        train = probe(TRAIN_Q)
        held = probe(HOLDOUT_Q)
        acc = sum(train.values()) / 3
        hacc = sum(held.values()) / 3
        curve.append({"round": rd, "train": acc, "holdout": hacc})
        print(f"learn round {rd}: train {acc:.3f} holdout {hacc:.3f}")
        if acc >= 2 / 3 and hacc >= 2 / 3 and rd >= 2:
            break
        for key, _q in TRAIN_Q:
            if not train[key] and taught[key] < len(VARIANTS[key]):
                line = VARIANTS[key][taught[key]]
                taught[key] += 1
                rr = bridge.valhalla(bridge.build_payload("Teach Aria facts", {}, session, None,
                                                          [{"text": line}], bridge.CFG))
                if isinstance(rr.get("session"), dict):
                    session = rr["session"]
                lines_now.append(line)
    return {"curve": curve, "taught_counts": taught, "corpus_lines_final": len(lines_now)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"protocol": "val-video-boost-make-video-v1", "model_video": "THUDM/CogVideoX-2b",
              "anchor_hero": "reports/fate_iterative_character/static_triad/s7_studio.png"}
    report["learning"] = learn()

    triad = module_patch.load_export_patch(PATCH_JSON)
    fid = FaceID()
    hero_emb = fid.embedding(ROOT / "reports" / "fate_iterative_character" / "static_triad" / "s7_studio.png")
    pipe = module_patch._load_video()
    segments = []
    all_frames: list[Path] = []
    try:
        for i, (name, prompt) in enumerate(SEGMENTS):
            best = None
            for attempt, seed in enumerate((SEED + i, SEED + 100 + i)):
                frames, secs, info = vb.generate_segment(
                    pipe, prompt, OUT / f"seg{i}_{name}" / f"try{attempt}", seed, triad,
                    mode="total", steps=STEPS, num_frames=FRAMES)
                mid = frames[len(frames) // 2]
                emb = fid.embedding(mid)
                cos = None if (emb is None or hero_emb is None) else fid.cosine(hero_emb, emb)
                entry = {"segment": i, "scene": name, "attempt": attempt, "seed": seed,
                         "seconds": secs, "arcface_cos_to_anchor": cos,
                         "metrics": vb.clip_metrics(frames), "frames_dir": str(frames[0].parent)}
                print(i, name, attempt, "cos", cos, secs, "s")
                if best is None or (cos or -1) > (best["arcface_cos_to_anchor"] or -1):
                    best = entry
                    best_frames = frames
                if cos is not None and cos >= TAU:
                    break
            best["accepted"] = (best["arcface_cos_to_anchor"] or -1) >= TAU
            segments.append(best)
            all_frames.extend(best_frames)
    finally:
        import torch
        del pipe
        torch.cuda.empty_cache()
    report["segments"] = segments

    # assemble mp4 (8 fps) + preview gif
    import imageio.v2 as imageio
    mp4 = OUT / "aria_video.mp4"
    writer = imageio.get_writer(str(mp4), fps=8, codec="libx264", quality=8)
    for f in all_frames:
        writer.append_data(imageio.imread(str(f)))
    writer.close()
    report["video_file"] = str(mp4)
    from PIL import Image
    sheet = Image.new("RGB", (720 * len(SEGMENTS), 480), "white")
    for i, seg in enumerate(segments):
        mid = Path(seg["frames_dir"]) / f"frame_{FRAMES // 2:02d}.png"
        sheet.paste(Image.open(mid), (720 * i, 0))
    sheet_path = OUT / "contact_video.png"
    sheet.save(sheet_path)
    report["contact_sheet"] = str(sheet_path)
    ok = [s["arcface_cos_to_anchor"] for s in segments if s["arcface_cos_to_anchor"] is not None]
    report["summary"] = {
        "segments_accepted_tau": sum(1 for s in segments if s["accepted"]),
        "segments_total": len(segments),
        "mean_arcface_cos": round(sum(ok) / len(ok), 4) if ok else None,
        "min_arcface_cos": min(ok) if ok else None,
        "learning_final": report["learning"]["curve"][-1],
        "total_frames": len(all_frames)}
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False))
    print(f"video: {mp4}")


if __name__ == "__main__":
    main()
