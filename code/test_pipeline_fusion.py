#!/usr/bin/env python3
"""Fuse a larger LLM, TTS, image, and video model through the Valhalla pipeline.

Media weights are the next size up that still fits this machine:
  image  stabilityai/sdxl-turbo
  video  THUDM/CogVideoX-2b
  speech facebook/mms-tts-eng
  LLM    Qwen2.5-3B-Instruct (already local; the pipeline default is 0.5B)

Each passphrase or protocol turn follows the Windows shell: native retrieval on
pipeline v4 first, then one lm_patch call. The patch call runs
run_question_pipeline on the fused Fate snapshot, patches Qwen2.5-3B, and
writes the sentence from the retrieved line. A turn whose decode_path is not
valhalla_base/lm_patch/...pipeline_v4... did not stay on the pipeline.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bridge  # noqa: E402
import download  # noqa: E402

OUT = REPO / "reports" / "media_pipeline"
ASSETS = OUT / "pipeline_assets"
PASSPHRASE_Q = "What is the passphrase?"
PROTOCOL_Q = "What is the primary protocol?"


def cosine(a: list[float], b: list[float]) -> float:
    width = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(width))
    na = sum(x * x for x in a[:width]) ** 0.5
    nb = sum(x * x for x in b[:width]) ** 0.5
    return dot / (na * nb + 1e-12)


def on_lm_pipeline(path: str) -> bool:
    return path.startswith("valhalla_base/lm_patch/") and "pipeline_v4" in path


def on_native_pipeline(path: str) -> bool:
    return path.startswith("valhalla_base/native/") and "pipeline_v4" in path


def evidence_line(body: dict, notes: list[dict]) -> str:
    answer = ((body.get("rows") or [{}])[0]).get("answer", "")
    found = re.search(r"\[corpus:(\d+)\]", answer)
    texts = [item["text"] for item in notes if item.get("text")]
    if found:
        index = int(found.group(1))
        if index < len(texts):
            return texts[index]
    return answer


def spoken(question: str, evidence: str) -> str:
    knowledge = "\n".join((bridge.PROTOCOL, bridge.FACT, bridge.BRIDGE_NOTE))
    block = ""
    if evidence:
        block = (
            "Retrieved evidence:\n"
            + evidence
            + "\nBase the answer on this evidence when it addresses the user. "
            "Do not replace it with a different fact.\n\n"
        )
    return (
        "You are Valhalla. Use the knowledge when it answers the user. "
        "Answer in one or two sentences.\n\n"
        + block
        + "Knowledge:\n"
        + knowledge
        + "\n\nUser: "
        + question
    )


def snapshot(label: str, body: dict) -> dict:
    row = (body.get("rows") or [{}])[0]
    session = body.get("session") or {}
    core = row.get("core_fusion") if isinstance(row.get("core_fusion"), dict) else {}
    path = row.get("decode_path", "")
    return {
        "label": label,
        "answer": row.get("answer", ""),
        "decode_path": path,
        "on_pipeline": on_lm_pipeline(path) or on_native_pipeline(path),
        "on_lm_pipeline": on_lm_pipeline(path),
        "question_history": len(session.get("question_history") or []),
        "fragments": len(session.get("fragments") or []),
        "fingerprint": session.get("fingerprint", ""),
        "core_fusion": {
            "universal_score": core.get("universal_score"),
            "universal_formed": core.get("universal_formed"),
            "shared_patch": core.get("shared_patch"),
        },
    }


def ensure_assets(reuse: bool) -> dict:
    image_dir = download.download_repo(download.IMAGE_XL_REPO, download.IMAGE_XL_DIR)
    video_dir = download.download_repo(download.VIDEO_XL_REPO, download.VIDEO_XL_DIR)
    voice_dir = download.download_repo(download.TTS_XL_REPO, download.TTS_XL_DIR)
    ASSETS.mkdir(parents=True, exist_ok=True)
    still = ASSETS / "still.png"
    wav = ASSETS / "speech.wav"
    frame_dir = ASSETS / "frames"
    backends = {
        "llm": str(bridge.LM_DIR),
        "image": download.IMAGE_XL_REPO,
        "video": download.VIDEO_XL_REPO,
        "tts": download.TTS_XL_REPO,
    }
    if reuse and still.is_file() and wav.is_file() and any(frame_dir.glob("frame_*.png")):
        print("reusing generated assets")
        return {"still": still, "wav": wav, "frames": sorted(frame_dir.glob("frame_*.png")), "backends": backends}
    print("image:", bridge.FACT)
    bridge.generate_image(image_dir, still, bridge.FACT)
    print("video:", bridge.FACT)
    frames = bridge.generate_video(video_dir, frame_dir, bridge.FACT)
    print("tts:", bridge.FACT)
    backends["tts_runtime"] = bridge.synthesize_mms(bridge.FACT, wav, voice_dir)
    return {"still": still, "wav": wav, "frames": frames, "backends": backends}


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM + TTS + image + video through the Valhalla pipeline")
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()

    assets = ensure_assets(args.reuse)
    audio_vec, audio_encoder = bridge.encode("audio_path", path=str(assets["wav"]))
    audio_again, _ = bridge.encode("audio_path", path=str(assets["wav"]))
    image_vec, image_encoder = bridge.encode("image_path", path=str(assets["still"]))
    frame_vecs = []
    for path in assets["frames"]:
        vec, _enc = bridge.encode("image_path", path=str(path))
        frame_vecs.append(vec)
    video_vec = bridge.pool_frames(frame_vecs)
    notes = [
        {"text": bridge.PROTOCOL},
        {"text": bridge.FACT},
        {"text": bridge.BRIDGE_NOTE},
        {"text": bridge.FACT, "audio_features": audio_vec},
        {"text": bridge.FACT, "image_features": image_vec},
    ]
    fused = {"audio_features": audio_vec, "image_features": image_vec}

    pipe = bridge.ValhallaPipeline(bridge.LM_DIR)
    try:
        print("native retrieval on pipeline v4")
        native = pipe.ask(PASSPHRASE_Q, fused, None, notes, None, timeout=120, config=bridge.CFG)
        session = native.get("session")
        if not isinstance(session, dict):
            raise RuntimeError("valhalla_base did not return a session")
        passphrase_evidence = evidence_line(native, notes)
        print("evidence:", passphrase_evidence)
        spoken_pass = spoken(PASSPHRASE_Q, passphrase_evidence)

        print("3B lm_patch after core_fusion")
        first = pipe.ask(spoken_pass, fused, session, None, None, timeout=300)
        session = first.get("session") or session
        print("session turn")
        held = pipe.ask(spoken_pass, fused, session, None, None)
        session = held.get("session") or session
        print("append video on the image port")
        video_turn = pipe.ask(
            spoken_pass,
            {"audio_features": audio_vec, "image_features": video_vec},
            session,
            None,
            [{"image_features": video_vec, "text": bridge.BRIDGE_NOTE}],
        )
        session = video_turn.get("session") or session
        print("native protocol retrieval")
        native_protocol = pipe.ask(PROTOCOL_Q, {}, session, None, None, config=bridge.CFG)
        session = native_protocol.get("session") or session
        protocol_evidence = evidence_line(native_protocol, notes)
        print("3B writes the protocol sentence")
        other = pipe.ask(spoken(PROTOCOL_Q, protocol_evidence), {}, session, None, None)
        session = other.get("session") or session
        print("reprobe")
        reprobe = pipe.ask(spoken_pass, fused, session, None, None)
    finally:
        pipe.close()

    turns = [
        snapshot("native_retrieve", native),
        snapshot("fused", first),
        snapshot("session_hold", held),
        snapshot("video_append", video_turn),
        snapshot("native_protocol", native_protocol),
        snapshot("interleave_protocol", other),
        snapshot("reprobe", reprobe),
    ]
    for turn in turns:
        print(f"  {turn['label']}: pipeline={turn['on_pipeline']} answer={turn['answer'][:160]!r}")
        print(f"    {turn['decode_path'][:180]}")

    lm_turns = [turns[1], turns[2], turns[3], turns[5], turns[6]]
    native_turns = [turns[0], turns[4]]
    pipeline_ok = all(turn["on_lm_pipeline"] for turn in lm_turns) and "pipeline_v4" in turns[0]["decode_path"]
    native_ok = all(turn["decode_path"].startswith("valhalla_base/native/") for turn in native_turns)
    fingerprint_held = len({turn["fingerprint"] for turn in turns if turn["fingerprint"]}) == 1
    history_grows = all(turns[i]["question_history"] >= i + 1 for i in range(len(turns)))
    shared = turns[1]["core_fusion"].get("shared_patch")
    if not isinstance(shared, (int, float)):
        shared = turns[0]["core_fusion"].get("shared_patch")
    core_present = isinstance(shared, (int, float))
    holds_fact = all("Qingniao" in turn["answer"] for turn in (turns[1], turns[2], turns[3], turns[6]))
    reprobe_holds = "Qingniao" in turns[6]["answer"]
    protocol_holds = "valhalla-base-v1" in turns[5]["answer"]
    passed = all([pipeline_ok, native_ok, fingerprint_held, history_grows, core_present, holds_fact, reprobe_holds, protocol_holds])
    report = {
        "protocol": "valhalla-pipeline-media-fusion-v1",
        "fact": bridge.FACT,
        "backends": assets["backends"],
        "encoders": {"audio": audio_encoder, "image": image_encoder, "audio_self_cosine": cosine(audio_vec, audio_again)},
        "turns": turns,
        "gates": {
            "lm_turns_on_pipeline_v4": pipeline_ok,
            "retrieval_stays_native": native_ok,
            "lm_turns_are_lm_patch": all(turn["on_lm_pipeline"] for turn in lm_turns),
            "session_fingerprint_stable": fingerprint_held,
            "question_history_grows": history_grows,
            "core_fusion_shared_patch": core_present,
            "passphrase_kept_by_3b": holds_fact,
            "reprobe_keeps_passphrase": reprobe_holds,
            "protocol_sentence_switches": protocol_holds,
        },
        "verdict": "PASS" if passed else "FAIL",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "pipeline_fusion.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"verdict {report['verdict']}")
    print(f"wrote {path}")
    if not passed:
        missed = [name for name, ok in report["gates"].items() if not ok]
        print("missed:", ", ".join(missed))
        sys.exit(2)


if __name__ == "__main__":
    main()
