#!/usr/bin/env python3
"""Bridge image, video, and speech into one Valhalla core_fusion session.

Valhalla's modality front has text, image features, and audio features
(manifestsys/hub-f64/src/dl_front.rs). There is no video port. A clip enters
the same way a still does: each frame is encoded by tools/valhalla_dl_front/encode.py,
the frame vectors are mean-pooled, and that vector is sent as image_features.

The session call matches phone/server.py fuse and the Windows live check:
native retrieval, core_fusion, open Fate, quad-cycle ingress, and the Fate
session is sent back on the next turn so fusion rounds stay on the same notes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import settings

FACT = "The passphrase is Qingniao."
PROTOCOL = "ValhallaBase primary protocol is valhalla-base-v1."
BRIDGE_NOTE = "Video enters the Fate session as pooled frame features on the image port."

CFG = {
    "decode": "native",
    "core_fusion": True,
    "open_fate": True,
    "use_cpu": True,
    "proactive_enabled": False,
    "max_new": 64,
    "fate_ingress_routing": "quad_cycle",
    "fate_qa_feedback": True,
    "answer_mix": "mixed",
}

# lm_patch calls run_question_pipeline, then patches Qwen and writes the sentence.
# follow_up inherit keeps later turns on that same path instead of dropping to native.
PIPELINE_CFG = {
    "decode": "lm_patch",
    "lm_model": str(LM_DIR),
    "core_fusion": True,
    "open_fate": True,
    "use_cpu": False,
    "proactive_enabled": False,
    "max_new": 64,
    "follow_up_decode": "inherit",
    "fate_ingress_routing": "quad_cycle",
    "fate_qa_feedback": True,
    "answer_mix": "mixed",
}


def encode(modality: str, **fields) -> tuple[list[float], str]:
    encode_script = settings.require_valhalla_encode()
    payload = {"modality": modality, **fields}
    proc = subprocess.run(
        [sys.executable, str(encode_script)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=True,
    )
    body = json.loads(proc.stdout)
    return body["embedding"], body.get("encoder", "")


def normalize(values: list[float]) -> list[float]:
    norm = sum(v * v for v in values) ** 0.5 or 1e-9
    return [v / norm for v in values]


def pool_frames(vectors: list[list[float]]) -> list[float]:
    width = len(vectors[0])
    mean = [sum(row[i] for row in vectors) / len(vectors) for i in range(width)]
    return normalize(mean)


def has_fp16(model_dir: Path) -> bool:
    return any(model_dir.rglob("*.fp16.safetensors"))


def pipeline_class(model_dir: Path) -> str:
    index = model_dir / "model_index.json"
    if not index.is_file():
        return ""
    try:
        return str(json.loads(index.read_text(encoding="utf-8")).get("_class_name", ""))
    except json.JSONDecodeError:
        return ""


def generate_image(model_dir: Path, out_path: Path, prompt: str = FACT) -> None:
    import torch
    from diffusers import AutoPipelineForText2Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    kwargs = {"torch_dtype": dtype}
    if device == "cuda" and has_fp16(model_dir):
        kwargs["variant"] = "fp16"
    pipe = AutoPipelineForText2Image.from_pretrained(str(model_dir), **kwargs).to(device)
    pipe.set_progress_bar_config(disable=True)
    generator = torch.Generator(device=device).manual_seed(42)
    image = pipe(
        prompt=prompt,
        num_inference_steps=2 if device == "cuda" else 1,
        guidance_scale=0.0,
        generator=generator,
        height=512,
        width=512,
    ).images[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    del pipe
    if device == "cuda":
        torch.cuda.empty_cache()


def _as_image(frame):
    from PIL import Image

    if hasattr(frame, "save"):
        return frame
    import numpy as np

    array = np.asarray(frame)
    if array.dtype != np.uint8:
        scale = 255.0 if float(array.max()) <= 1.0 else 1.0
        array = np.clip(array * scale, 0, 255).astype(np.uint8)
    return Image.fromarray(array)


def generate_video(model_dir: Path, frame_dir: Path, prompt: str = FACT, frames: int = 8) -> list[Path]:
    import torch
    from diffusers import DiffusionPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    kwargs = {"torch_dtype": dtype}
    if has_fp16(model_dir):
        kwargs["variant"] = "fp16"
    pipe = DiffusionPipeline.from_pretrained(str(model_dir), **kwargs)
    if device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    generator = torch.Generator(device="cpu").manual_seed(42)
    kind = pipeline_class(model_dir)
    if "CogVideoX" in kind:
        call = {
            "prompt": prompt,
            "num_inference_steps": 10,
            "num_frames": 17,
            "height": 480,
            "width": 720,
            "guidance_scale": 6.0,
            "generator": generator,
        }
    else:
        call = {
            "prompt": prompt,
            "num_inference_steps": 8,
            "num_frames": frames,
            "height": 256,
            "width": 256,
            "guidance_scale": 7.5,
            "generator": generator,
        }
    result = pipe(**call)
    pictures = [_as_image(frame) for frame in result.frames[0]]
    frame_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for index, picture in enumerate(pictures):
        path = frame_dir / f"frame_{index:02d}.png"
        picture.save(path)
        saved.append(path)
    if len(saved) > 1:
        pictures[0].save(
            frame_dir / "clip.gif",
            save_all=True,
            append_images=pictures[1:],
            duration=125,
            loop=0,
        )
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return saved


def synthesize_mms(text: str, wav_path: Path, model_dir: Path) -> str:
    import numpy as np
    import torch
    import wave
    from transformers import AutoTokenizer, VitsModel

    wav_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = VitsModel.from_pretrained(str(model_dir))
    model.eval()
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        waveform = model(**inputs).waveform
    audio = waveform.squeeze().detach().cpu().numpy()
    rate = int(getattr(model.config, "sampling_rate", 16000))
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(wav_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())
    if wav_path.stat().st_size < 1000:
        raise RuntimeError("MMS TTS wrote an empty wav.")
    return "mms-tts-eng"


def synthesize_tts(text: str, wav_path: Path, voice_dir: Path) -> str:
    """Speak with the downloaded Piper voice when the runtime is present, else Windows SAPI."""
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    onnx = voice_dir / "en_US-lessac-medium.onnx"
    config = voice_dir / "en_US-lessac-medium.onnx.json"
    try:
        from piper import PiperVoice

        voice = PiperVoice.load(str(onnx), config_path=str(config))
        with wav_path.open("wb") as handle:
            voice.synthesize(text, handle)
        return "piper"
    except Exception:
        pass
    safe_text = text.replace("'", "''")
    safe_wav = str(wav_path).replace("'", "''")
    script = (
        "Add-Type -AssemblyName System.Speech; "
        "$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$synth.SetOutputToWaveFile('{safe_wav}'); "
        f"$synth.Speak('{safe_text}'); "
        "$synth.Dispose()"
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        check=True,
        capture_output=True,
        text=True,
    )
    if not wav_path.is_file() or wav_path.stat().st_size < 1000:
        raise RuntimeError("Windows SAPI did not write a wav.")
    return "sapi"


def valhalla(payload: dict, timeout: int = 120) -> dict:
    runtime = settings.require_valhalla_exe()
    proc = subprocess.run(
        [str(runtime), "--json-error"],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "valhalla_base failed").strip()
        raise RuntimeError(detail[:500])
    line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    return json.loads(line)


def build_payload(question: str, features: dict, session: dict | None, context_items: list[dict] | None, append_items: list[dict] | None, config: dict) -> dict:
    body = {"id": "q0", "text": question, "score_type": "open"}
    body.update(features)
    payload: dict = {"questions": [body], "config": config}
    if session is not None:
        payload["session"] = session
        payload["context"] = []
        payload["context_items"] = []
    else:
        payload["context_items"] = context_items or []
    if append_items:
        payload["append_items"] = append_items
    return payload


def ask(question: str, features: dict, session: dict | None, context_items: list[dict] | None, append_items: list[dict] | None) -> dict:
    return valhalla(build_payload(question, features, session, context_items, append_items, CFG))


class ValhallaPipeline:
    """One warm valhalla_base --repl. The 3B weights stay loaded across fusion turns."""

    def __init__(self, model_dir: Path | None = None) -> None:
        runtime = settings.require_valhalla_exe()
        if model_dir is None:
            model_dir = settings.VALHALLA_QWEN_MODEL
        if not (model_dir / "model.safetensors.index.json").is_file() and not (model_dir / "model.safetensors").is_file():
            raise FileNotFoundError(f"找不到 Qwen 权重：{model_dir}")
        env = os.environ.copy()
        env["VALHALLA_LM_MODEL"] = str(model_dir)
        env["VALHALLA_QWEN_MODEL_DIR"] = str(model_dir)
        env["VALHALLA_LM_F16"] = "1"
        self.model_dir = model_dir
        self.stderr: list[str] = []
        self.proc = subprocess.Popen(
            [str(runtime), "--repl"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            cwd=str(runtime.parent),
        )
        threading.Thread(target=self._drain, daemon=True).start()

    def _drain(self) -> None:
        assert self.proc.stderr is not None
        for chunk in iter(lambda: self.proc.stderr.read(4096), ""):
            self.stderr.append(chunk)

    def ask(
        self,
        question: str,
        features: dict,
        session: dict | None,
        context_items: list[dict] | None,
        append_items: list[dict] | None,
        timeout: int = 240,
        config: dict | None = None,
    ) -> dict:
        if self.proc.poll() is not None:
            tail = "".join(self.stderr)[-800:]
            raise RuntimeError(f"valhalla_base 已退出。{tail}")
        payload = build_payload(question, features, session, context_items, append_items, config or PIPELINE_CFG)
        box: dict = {}

        def read() -> None:
            assert self.proc.stdout is not None
            while True:
                line = self.proc.stdout.readline()
                if not line:
                    box["eof"] = True
                    return
                line = line.strip()
                if line.startswith("{"):
                    box["line"] = line
                    return

        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        thread = threading.Thread(target=read, daemon=True)
        thread.start()
        thread.join(timeout)
        if "line" not in box:
            tail = "".join(self.stderr)[-800:]
            raise RuntimeError(f"管线没有在 {timeout}s 内返回。{tail}")
        return json.loads(box["line"])

    def close(self) -> None:
        if self.proc.poll() is None and self.proc.stdin is not None:
            self.proc.stdin.close()
            self.proc.wait(timeout=20)
