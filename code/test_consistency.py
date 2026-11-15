#!/usr/bin/env python3
"""Hold one fact across image, speech, and video using a Valhalla Fate session.

The same sentence is drawn, spoken, and filmed. Features go into one
core_fusion turn (text + audio + still). Later turns send only the session,
the way the Windows shell does once a session exists. The video vector is
appended on the image port. A different question is asked in between, then
the original question is asked again.

The hold is the retrieved fact, not the pixels: the passphrase must still be
the answer after the other modalities and after the interleaved question.
The session must keep growing (question history, fusion rounds or fragments)
instead of being replaced.
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
ASSETS = OUT / "assets"

PASSPHRASE_Q = "What is the passphrase?"
PROTOCOL_Q = "What is the primary protocol?"


def cosine(a: list[float], b: list[float]) -> float:
    width = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(width))
    na = sum(a[i] * a[i] for i in range(width)) ** 0.5
    nb = sum(b[i] * b[i] for i in range(width)) ** 0.5
    return dot / (na * nb + 1e-12)


def modality_names(mask: list) -> list[str]:
    bits = 0
    for item in mask:
        try:
            bits |= int(item)
        except (TypeError, ValueError):
            continue
    names = []
    if bits & 1:
        names.append("text")
    if bits & 2:
        names.append("audio")
    if bits & 4:
        names.append("image")
    return names


def landing(answer: str) -> dict:
    corpus = re.search(r"\[corpus:(\d+)\]", answer)
    fate = re.search(r"\[fate:([^\]]+)\]", answer)
    return {
        "corpus": corpus.group(1) if corpus else "",
        "fate": fate.group(1) if fate else "",
    }


def snapshot(label: str, body: dict) -> dict:
    row = (body.get("rows") or [{}])[0]
    session = body.get("session") or {}
    meta = session.get("fate_algebra_meta") or {}
    core = row.get("core_fusion") if isinstance(row.get("core_fusion"), dict) else {}
    answer = row.get("answer", "")
    return {
        "label": label,
        "answer": answer,
        **landing(answer),
        "decode_path": row.get("decode_path", ""),
        "fusion_rounds": meta.get("fusion_rounds", 0),
        "question_history": len(session.get("question_history") or []),
        "fragments": len(session.get("fragments") or []),
        "lines_ingested": session.get("lines_ingested", 0),
        "fingerprint": session.get("fingerprint", ""),
        "modalities": modality_names(session.get("corpus_modality_mask") or []),
        "core_fusion": {
            "universal_score": core.get("universal_score"),
            "universal_formed": core.get("universal_formed"),
            "shared_patch": core.get("shared_patch"),
            "decode_snap": core.get("decode_snap"),
        },
        "features": body.get("features") or [],
    }


def ensure_assets(reuse: bool) -> dict:
    download.MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    image_dir = download.download_repo(download.IMAGE_REPO, download.IMAGE_DIR)
    voice_dir = download.download_tts(download.TTS_DIR)
    video_dir = download.download_repo(download.VIDEO_REPO, download.VIDEO_DIR)

    ASSETS.mkdir(parents=True, exist_ok=True)
    still = ASSETS / "still.png"
    wav = ASSETS / "speech.wav"
    frame_dir = ASSETS / "frames"

    backends = {}
    if reuse and still.is_file() and wav.is_file() and any(frame_dir.glob("frame_*.png")):
        print("reusing generated assets")
        backends["image"] = "cached"
        backends["video"] = "cached"
        backends["tts"] = "cached"
    else:
        print("image:", bridge.FACT)
        bridge.generate_image(image_dir, still, bridge.FACT)
        backends["image"] = "sd-turbo"
        print("video:", bridge.FACT)
        bridge.generate_video(video_dir, frame_dir, bridge.FACT, frames=8)
        backends["video"] = "text-to-video-ms-1.7b"
        print("tts:", bridge.FACT)
        backends["tts"] = bridge.synthesize_tts(bridge.FACT, wav, voice_dir)
    frames = sorted(frame_dir.glob("frame_*.png"))
    if not frames:
        raise RuntimeError("video frames were not written")
    return {"still": still, "wav": wav, "frames": frames, "backends": backends, "voice_dir": str(voice_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Valhalla multimodal consistency hold")
    parser.add_argument("--reuse", action="store_true", help="Reuse still.png, speech.wav, and frames")
    args = parser.parse_args()

    assets = ensure_assets(args.reuse)
    text_vec, text_encoder = bridge.encode("text", content=bridge.FACT)
    text_again, _ = bridge.encode("text", content=bridge.FACT)
    audio_vec, audio_encoder = bridge.encode("audio_path", path=str(assets["wav"]))
    audio_again, _ = bridge.encode("audio_path", path=str(assets["wav"]))
    image_vec, image_encoder = bridge.encode("image_path", path=str(assets["still"]))
    image_again, _ = bridge.encode("image_path", path=str(assets["still"]))
    frame_meta = []
    frame_vecs = []
    for path in assets["frames"]:
        vec, enc = bridge.encode("image_path", path=str(path))
        frame_vecs.append(vec)
        frame_meta.append({"file": path.name, "encoder": enc, "dim": len(vec)})
    video_vec = bridge.pool_frames(frame_vecs)

    notes = [
        {"text": bridge.PROTOCOL},
        {"text": bridge.FACT},
        {"text": bridge.BRIDGE_NOTE},
        {"text": bridge.FACT, "audio_features": audio_vec},
        {"text": bridge.FACT, "image_features": image_vec},
    ]
    fused_features = {"audio_features": audio_vec, "image_features": image_vec}

    print("fusion turn: text + speech + still")
    first = bridge.ask(PASSPHRASE_Q, fused_features, None, notes, None)
    session = first.get("session")
    if not isinstance(session, dict):
        raise RuntimeError("valhalla_base did not return a session")

    print("session turn: same question, no re-ingest")
    held = bridge.ask(PASSPHRASE_Q, fused_features, session, None, None)
    session = held.get("session") or session

    print("append video frames on the image port")
    video_turn = bridge.ask(
        PASSPHRASE_Q,
        {"audio_features": audio_vec, "image_features": video_vec},
        session,
        None,
        [{"image_features": video_vec, "text": bridge.BRIDGE_NOTE}],
    )
    session = video_turn.get("session") or session

    print("interleave a different fact")
    other = bridge.ask(PROTOCOL_Q, {}, session, None, None)
    session = other.get("session") or session

    print("reprobe the passphrase")
    reprobe = bridge.ask(PASSPHRASE_Q, fused_features, session, None, None)

    turns = [
        snapshot("fused", first),
        snapshot("session_hold", held),
        snapshot("video_append", video_turn),
        snapshot("interleave_protocol", other),
        snapshot("reprobe", reprobe),
    ]
    for turn in turns:
        print(
            f"  {turn['label']}: rounds={turn['fusion_rounds']} history={turn['question_history']} "
            f"fragments={turn['fragments']} mask={'+'.join(turn['modalities']) or '-'} "
            f"answer={turn['answer'][:120]!r}"
        )

    passphrase_turns = [turns[0], turns[1], turns[2], turns[4]]
    fate_ids = {turn["fate"] for turn in passphrase_turns}
    corpus_ids = {turn["corpus"] for turn in passphrase_turns}
    passphrase_corpus = next(iter(corpus_ids)) if len(corpus_ids) == 1 else ""
    holds_landing = len(fate_ids) == 1 and bool(passphrase_corpus) and "" not in fate_ids
    holds_fact = holds_landing and any("Qingniao" in turn["answer"] for turn in passphrase_turns)
    interleave_moves = (
        bool(passphrase_corpus)
        and turns[3]["corpus"] not in ("", passphrase_corpus)
        and "Qingniao" not in turns[3]["answer"]
    )
    fingerprint_held = len({turn["fingerprint"] for turn in turns}) == 1 and turns[0]["fingerprint"]
    history_grows = all(turns[i]["question_history"] >= i + 1 for i in range(len(turns)))
    rounds = [turn["fusion_rounds"] or 0 for turn in turns]
    rounds_hold = all(rounds[i] >= rounds[0] for i in range(len(rounds)))
    fragments_hold = all(turn["fragments"] >= turns[0]["fragments"] for turn in turns)
    masks = set(turns[0]["modalities"]) | set(turns[2]["modalities"])
    multimodal = {"text", "audio", "image"} <= masks
    encoders_stable = (
        cosine(text_vec, text_again) > 0.999
        and cosine(audio_vec, audio_again) > 0.999
        and cosine(image_vec, image_again) > 0.999
    )
    shared_values = [
        turn["core_fusion"].get("shared_patch")
        for turn in (turns[0], turns[4])
        if isinstance(turn["core_fusion"].get("shared_patch"), (int, float))
    ]
    core_present = len(shared_values) == 2 and min(shared_values) >= 0.5 and all(
        turn["core_fusion"].get("universal_formed") for turn in (turns[0], turns[4])
    )

    passed = all(
        [
            holds_fact,
            interleave_moves,
            fingerprint_held,
            history_grows,
            rounds_hold,
            fragments_hold,
            multimodal,
            encoders_stable,
            core_present,
        ]
    )
    report = {
        "protocol": "valhalla-media-consistency-v1",
        "fact": bridge.FACT,
        "backends": assets["backends"],
        "encoders": {
            "text": text_encoder,
            "audio": audio_encoder,
            "image": image_encoder,
            "video_frames": frame_meta,
            "self_cosine": {
                "text": cosine(text_vec, text_again),
                "audio": cosine(audio_vec, audio_again),
                "image": cosine(image_vec, image_again),
            },
        },
        "bridge": "video frames are mean-pooled into image_features and appended on the open Fate session",
        "turns": turns,
        "gates": {
            "passphrase_held": holds_fact,
            "fate_landing_stable": holds_landing,
            "session_fingerprint_stable": bool(fingerprint_held),
            "interleave_switches_fact": interleave_moves,
            "question_history_grows": history_grows,
            "fusion_rounds_do_not_drop": rounds_hold,
            "fragments_do_not_drop": fragments_hold,
            "modality_mask_has_text_audio_image": multimodal,
            "encoders_repeat": encoders_stable,
            "core_fusion_shared_patch": core_present,
        },
        "verdict": "PASS" if passed else "FAIL",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "consistency.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"verdict {report['verdict']}")
    print(f"wrote {path}")
    if not passed:
        missed = [name for name, ok in report["gates"].items() if not ok]
        print("missed:", ", ".join(missed))
        sys.exit(2)


if __name__ == "__main__":
    main()
