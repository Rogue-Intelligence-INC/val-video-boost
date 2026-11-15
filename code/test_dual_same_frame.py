#!/usr/bin/env python3
"""Free gap A: two characters in the SAME frame.

Prompt places Aria (chestnut) left and Sofia (blonde) right. Every
detected face (embeddings_all, left-to-right) is scored against BOTH
enrolled heroes: the frame passes only if the left face is closer to
Aria than to Sofia AND the right face is closer to Sofia than to Aria
(correct attribution, not just two faces present). Single draws,
seeds 7-9, park + cafe. Anchors: the two detailed-supply studio heroes.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import download
from face_id import FaceID
from fate_interface import IDENTITY_BLOCK
from test_big_models import load_pipe
from test_dual_character import SOFIA_BLOCK, SOFIA_NEG
from fate_interface import NEGATIVE
from test_context_prompt import SIZE, STEPS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "dual_same_frame"
DIAG = ROOT / "reports" / "context_diagnostic"
DUAL = ROOT / "reports" / "dual_character"
SEEDS = (7, 8, 9)
SCENES = {
    "park": "standing side by side in a sunlit park, waist-up, both facing camera",
    "cafe": "sitting side by side at a cafe table by a window, waist-up, both facing camera",
}


def prompt(scene: str) -> str:
    return (f"photorealistic photograph of two women side by side: on the left {IDENTITY_BLOCK}; "
            f"on the right {SOFIA_BLOCK}; {SCENES[scene]}")


def gen(pipe, scene, out_path, seed):
    import torch
    if out_path.exists():
        return 0.0
    started = time.time()
    img = pipe(prompt=prompt(scene), negative_prompt=NEGATIVE,
               num_inference_steps=STEPS, guidance_scale=7.0,
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
            for scene in SCENES:
                target = OUT / f"s{seed}_{scene}.png"
                gen(pipe, scene, target, seed)
                faces = fid.embeddings_all(target)
                row = {"seed": seed, "scene": scene, "faces_found": len(faces)}
                if len(faces) >= 2:
                    left, right = faces[0], faces[-1]
                    row["left_to_aria"] = fid.cosine(anchors["aria"], left["embedding"])
                    row["left_to_sofia"] = fid.cosine(anchors["sofia"], left["embedding"])
                    row["right_to_sofia"] = fid.cosine(anchors["sofia"], right["embedding"])
                    row["right_to_aria"] = fid.cosine(anchors["aria"], right["embedding"])
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
    two = [r for r in rows if r["faces_found"] >= 2]
    report = {"protocol": "dual-character-same-frame-v1", "rows": rows,
              "summary": {"frames": len(rows), "two_faces": len(two),
                          "attribution_correct": sum(1 for r in rows if r["attribution_correct"])},
              "verdicts": {"two_faces_majority": len(two) >= len(rows) // 2 + 1,
                           "attribution_majority": sum(1 for r in rows if r["attribution_correct"]) >= len(rows) // 2 + 1}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False))
    print(json.dumps(report["verdicts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
