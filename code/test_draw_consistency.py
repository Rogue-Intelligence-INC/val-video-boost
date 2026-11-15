#!/usr/bin/env python3
"""Does the Valhalla pipeline make a later picture stay on the original fact?

Diffusion does not run inside pipeline.rs. There is no video port. The sampler
only becomes consistent when it is not allowed to see the raw chat tail.

After the passphrase is stored, the session is asked about the protocol.
"Draw that" would follow the latest topic. The pipeline instead reprobes the
original fact, the 3B writes one scene line from that evidence, and SDXL plus
CogVideoX receive only that line.

The control picture is drawn from the latest topic. Closer to the reference
still (the passphrase sentence) means the pipeline held the picture on the fact.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bridge  # noqa: E402
import download  # noqa: E402
import test_pipeline_fusion as fuse  # noqa: E402

OUT = fuse.OUT
DRAW = OUT / "draw_consistency"
REF = OUT / "pipeline_assets" / "still.png"


def first_line(answer: str) -> str:
    for line in answer.splitlines():
        text = line.strip().strip('"')
        if text:
            return text[:400]
    return answer.strip()[:400]


def main() -> None:
    image_dir = download.IMAGE_XL_DIR
    video_dir = download.VIDEO_XL_DIR
    if not (image_dir / "model_index.json").is_file() or not (video_dir / "model_index.json").is_file():
        raise SystemExit("sdxl-turbo or CogVideoX-2b is not downloaded")
    if not REF.is_file():
        raise SystemExit(f"missing reference still: {REF}")

    notes = [
        {"text": bridge.PROTOCOL},
        {"text": bridge.FACT},
        {"text": bridge.BRIDGE_NOTE},
    ]
    pipe = bridge.ValhallaPipeline(bridge.LM_DIR)
    try:
        print("store the passphrase")
        first = pipe.ask(fuse.PASSPHRASE_Q, {}, None, notes, None, timeout=120, config=bridge.CFG)
        session = first.get("session")
        if not isinstance(session, dict):
            raise RuntimeError("no session")
        print("talk about the protocol, which is the latest topic")
        protocol = pipe.ask(fuse.PROTOCOL_Q, {}, session, None, None, config=bridge.CFG)
        session = protocol.get("session") or session
        print("reprobe the original fact through the same session")
        again = pipe.ask(fuse.PASSPHRASE_Q, {}, session, None, None, config=bridge.CFG)
        session = again.get("session") or session
        evidence = fuse.evidence_line(again, notes)
        latest = fuse.evidence_line(protocol, notes)
        scene_ask = fuse.spoken(
            "Write one English picture prompt that depicts this fact. "
            "Include the passphrase string itself. One line only.",
            evidence,
        )
        print("3B writes the scene line on lm_patch")
        scene_turn = pipe.ask(scene_ask, {}, session, None, None, timeout=180)
    finally:
        pipe.close()

    scene = first_line((scene_turn.get("rows") or [{}])[0].get("answer", ""))
    drift = f"A picture of the latest topic: {latest}"
    decode_path = (scene_turn.get("rows") or [{}])[0].get("decode_path", "")
    print("evidence:", evidence)
    print("scene:", scene)
    print("drift:", drift)

    DRAW.mkdir(parents=True, exist_ok=True)
    held_png = DRAW / "pipeline.png"
    drift_png = DRAW / "latest_topic.png"
    print("sdxl pipeline scene")
    bridge.generate_image(image_dir, held_png, scene)
    print("sdxl latest topic")
    bridge.generate_image(image_dir, drift_png, drift)
    print("cogvideox pipeline scene")
    frames = bridge.generate_video(video_dir, DRAW / "frames", scene)

    ref_vec, _ = bridge.encode("image_path", path=str(REF))
    held_vec, _ = bridge.encode("image_path", path=str(held_png))
    drift_vec, _ = bridge.encode("image_path", path=str(drift_png))
    frame_vecs = [bridge.encode("image_path", path=str(path))[0] for path in frames]
    video_vec = bridge.pool_frames(frame_vecs)
    held_cos = fuse.cosine(held_vec, ref_vec)
    drift_cos = fuse.cosine(drift_vec, ref_vec)
    video_cos = fuse.cosine(video_vec, ref_vec)

    fact_held = "Qingniao" in evidence and "Qingniao" in scene
    scene_on_pipeline = decode_path.startswith("valhalla_base/lm_patch/") and "pipeline_v4" in decode_path
    drift_leaves = "Qingniao" not in drift
    closer = held_cos > drift_cos
    passed = fact_held and scene_on_pipeline and drift_leaves and closer
    report = {
        "protocol": "valhalla-draw-consistency-v1",
        "reference": str(REF),
        "evidence_after_interleave": evidence,
        "latest_topic": latest,
        "pipeline_scene": scene,
        "drift_prompt": drift,
        "scene_decode_path": decode_path,
        "cosine_to_reference": {
            "pipeline_image": held_cos,
            "latest_topic_image": drift_cos,
            "pipeline_video": video_cos,
        },
        "gates": {
            "reprobe_still_qingniao": "Qingniao" in evidence,
            "scene_line_on_lm_patch_pipeline": scene_on_pipeline,
            "scene_names_the_passphrase": "Qingniao" in scene,
            "latest_topic_does_not": drift_leaves,
            "pipeline_image_closer_than_latest_topic": closer,
        },
        "verdict": "PASS" if passed else "FAIL",
    }
    path = OUT / "draw_consistency.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"cosine pipeline={held_cos:.3f} latest={drift_cos:.3f} video={video_cos:.3f}"
    )
    print(f"verdict {report['verdict']}")
    print(f"wrote {path}")
    if not passed:
        missed = [name for name, ok in report["gates"].items() if not ok]
        print("missed:", ", ".join(missed))
        sys.exit(2)


if __name__ == "__main__":
    main()
