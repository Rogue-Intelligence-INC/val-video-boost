#!/usr/bin/env python3
"""Free gap A2: same-frame, token-budgeted compact identity blocks.

v1 failed mechanistically: two full detailed blocks exceed SDXL's
77-token text budget, the second character's facts are truncated away
and the model renders the first person twice (verified visually).
v2 tests the Fate supply fix: scene first, then COMPACT distinguishing
blocks (hair/eyes/face-shape only, ~15 words each) with left/right
positions. Same attribution criterion as v1.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import download
from face_id import FaceID
from test_big_models import load_pipe
from test_context_prompt import SIZE, STEPS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "dual_same_frame"
DIAG = ROOT / "reports" / "context_diagnostic"
DUAL = ROOT / "reports" / "dual_character"
SEEDS = (7, 8, 9)
ARIA_C = "Aria with chestnut-brown shoulder-length hair, hazel eyes, a mole on her left cheek, slim oval face"
SOFIA_C = "Sofia with long platinum-blonde hair, blue eyes, pale skin, round face and full lips"
PROMPT = (f"photorealistic photograph of two different women standing side by side in a sunlit park, "
          f"waist-up, both facing camera: on the left {ARIA_C}; on the right {SOFIA_C}")
NEG = "same person twice, twins, duplicate face, three people, one woman only"


def gen(pipe, out_path, seed):
    import torch
    if out_path.exists():
        return 0.0
    started = time.time()
    img = pipe(prompt=PROMPT, negative_prompt=NEG, num_inference_steps=STEPS, guidance_scale=7.0,
               generator=torch.Generator(device="cuda").manual_seed(seed),
               height=SIZE, width=SIZE).images[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return round(time.time() - started, 2)


def main() -> None:
    pipe = load_pipe(download.IMAGE_JUG_DIR)
    fid = FaceID()
    rows = []
    try:
        for seed in SEEDS:
            anchors = {
                "aria": fid.embedding(DIAG / "prompt_channel" / "detailed_neg" / f"s{seed}_studio.png"),
                "sofia": fid.embedding(DUAL / "sofia" / f"s{seed}_studio.png"),
            }
            target = OUT / f"compact_s{seed}_park.png"
            gen(pipe, target, seed)
            faces = fid.embeddings_all(target)
            row = {"seed": seed, "faces_found": len(faces)}
            if len(faces) >= 2:
                left, right = faces[0], faces[-1]
                row.update({
                    "left_to_aria": fid.cosine(anchors["aria"], left["embedding"]),
                    "left_to_sofia": fid.cosine(anchors["sofia"], left["embedding"]),
                    "right_to_sofia": fid.cosine(anchors["sofia"], right["embedding"]),
                    "right_to_aria": fid.cosine(anchors["aria"], right["embedding"])})
                row["attribution_correct"] = (row["left_to_aria"] > row["left_to_sofia"]
                                              and row["right_to_sofia"] > row["right_to_aria"])
            else:
                row["attribution_correct"] = False
            rows.append(row)
            print(row)
    finally:
        import torch
        del pipe
        torch.cuda.empty_cache()
    report = {"protocol": "dual-character-same-frame-v2-compact", "prompt": PROMPT,
              "rows": rows,
              "summary": {"frames": len(rows),
                          "attribution_correct": sum(1 for r in rows if r["attribution_correct"])},
              "verdicts": {"attribution_majority": sum(1 for r in rows if r["attribution_correct"]) >= 2}}
    (OUT / "report_v2.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
