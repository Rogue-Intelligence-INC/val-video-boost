#!/usr/bin/env python3
"""Download image, video, and TTS weights the way the Windows shell does.

Same contract as native/src/engine.cpp download_model:

- models live under VALHALLA_MODELS_ROOT, default C:\\Ai\\models, one folder per repo leaf
- the file list comes from https://huggingface.co/api/models/{repo}/tree/main
- a file is wanted when it ends in .safetensors, .json, or .jinja, or its name starts with tokenizer
- bytes land in name.partial, then replace the destination
- a repo that already has its weights is left alone

Two practical extensions, both still on that URL scheme:

- when a tree ships both foo.safetensors and foo.fp16.safetensors, keep the fp16 twin
  (the desktop image worker loads variant=fp16; the fp32 copy is the same weights twice)
- TTS is Piper ONNX, which the shell's suffix list does not cover, so those two files
  are fetched with the same /{repo}/resolve/main/{path} URL and the same partial rename
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import settings

UA = "Valhalla/1.0"
HF = "https://huggingface.co"
MIRROR = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
MODELS_ROOT = settings.MODELS_ROOT

IMAGE_REPO = "stabilityai/sd-turbo"
IMAGE_XL_REPO = "stabilityai/sdxl-turbo"
IMAGE_BASE_REPO = "stabilityai/stable-diffusion-xl-base-1.0"
IMAGE_JUG_REPO = "RunDiffusion/Juggernaut-XL-v9"
# BFL repo is gated (401 anonymous); this is the ungated diffusers-format mirror.
FLUX_REPO = "YuCollection/FLUX.1-schnell-Diffusers"
VIDEO_REPO = "damo-vilab/text-to-video-ms-1.7b"
VIDEO_XL_REPO = "THUDM/CogVideoX-2b"
TTS_REPO = "rhasspy/piper-voices"
TTS_XL_REPO = "facebook/mms-tts-eng"
TTS_FILES = (
    "en/en_US/lessac/medium/en_US-lessac-medium.onnx",
    "en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
)

IMAGE_DIR = MODELS_ROOT / "sd-turbo"
IMAGE_XL_DIR = MODELS_ROOT / "sdxl-turbo"
IMAGE_BASE_DIR = MODELS_ROOT / "sdxl-base-1.0"
IMAGE_JUG_DIR = MODELS_ROOT / "juggernaut-xl-v9"
FLUX_DIR = MODELS_ROOT / "flux1-schnell"
VIDEO_DIR = MODELS_ROOT / "text-to-video-ms-1.7b"
VIDEO_XL_DIR = MODELS_ROOT / "CogVideoX-2b"
TTS_DIR = MODELS_ROOT / "piper-en_US-lessac-medium"
TTS_XL_DIR = MODELS_ROOT / "mms-tts-eng"


def leaf(repo_id: str) -> str:
    return repo_id.split("/")[-1]


def hf_json(path: str) -> object:
    request = urllib.request.Request(HF + path, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def wanted_file(path: str) -> bool:
    base = path.rsplit("/", 1)[-1]
    if path.endswith((".safetensors", ".json", ".jinja")):
        return True
    if base.startswith("tokenizer"):
        return True
    # CLIP tokenizer companion. The shell filter misses it; the video text encoder cannot load without it.
    return base in {"merges.txt", "spiece.model"}


def prefer_fp16(paths: list[str]) -> list[str]:
    present = set(paths)
    kept: list[str] = []
    for path in paths:
        if path.endswith(".safetensors") and ".fp16.safetensors" not in path:
            twin = path[: -len(".safetensors")] + ".fp16.safetensors"
            if twin in present:
                continue
        kept.append(path)
    return kept


def drop_packed_duplicates(paths: list[str]) -> list[str]:
    """Skip a root all-in-one checkpoint when the component folders are in the tree."""
    split = any("/" in path and path.endswith(".safetensors") for path in paths)
    if not split:
        return paths
    return [path for path in paths if "/" in path or not path.endswith(".safetensors")]


def complete(path: str, size: int, expected: int | None) -> bool:
    if size <= 0:
        return False
    if expected is not None:
        return size == expected
    if path.endswith((".safetensors", ".onnx")):
        return size > 1000
    return True


def download_once(host: str, repo_id: str, remote_path: str, partial: Path) -> int:
    url_path = "/" + repo_id + "/resolve/main/" + urllib.request.quote(remote_path)
    request = urllib.request.Request(host + url_path, headers={"User-Agent": UA})
    got = 0
    with urllib.request.urlopen(request, timeout=180) as response, partial.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            got += len(chunk)
            if got // (32 * 1024 * 1024) != (got - len(chunk)) // (32 * 1024 * 1024):
                print(f"    {remote_path}  {got / (1024 * 1024):.0f} MB", flush=True)
    return got


def download_file(repo_id: str, remote_path: str, dest: Path, expected: int | None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and complete(remote_path, dest.stat().st_size, expected):
        print(f"  skip {remote_path}", flush=True)
        return
    partial = dest.with_suffix(dest.suffix + ".partial")
    hosts = [HF]
    if MIRROR and MIRROR != HF:
        hosts.append(MIRROR)
    last = "download failed"
    for host in hosts:
        for attempt in range(3):
            print(f"  download {remote_path}  ({host}, try {attempt + 1})", flush=True)
            try:
                got = download_once(host, repo_id, remote_path, partial)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                partial.unlink(missing_ok=True)
                last = str(exc)
                time.sleep(2 * (attempt + 1))
                continue
            if complete(remote_path, got, expected):
                partial.replace(dest)
                return
            partial.unlink(missing_ok=True)
            last = f"incomplete {got} bytes, expected {expected}"
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{remote_path}: {last}")


def weights_ready(directory: Path) -> bool:
    if (directory / "model.safetensors").is_file() or (directory / "model.safetensors.index.json").is_file():
        return True
    if (directory / "model_index.json").is_file() and any(directory.rglob("*.safetensors")):
        return True
    return False


def download_repo(repo_id: str, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / "model_id.txt"
    if marker.is_file() and weights_ready(dest):
        print(f"ready {repo_id} → {dest}")
        return dest
    print(f"listing {repo_id}")
    tree = hf_json(f"/api/models/{repo_id}/tree/main?recursive=true")
    if not isinstance(tree, list):
        raise RuntimeError("模型列表无法解析。")
    files = []
    sizes: dict[str, int] = {}
    for item in tree:
        if not isinstance(item, dict) or item.get("type") != "file":
            continue
        path = str(item.get("path", ""))
        if not wanted_file(path):
            continue
        files.append(path)
        if isinstance(item.get("size"), int):
            sizes[path] = item["size"]
    files = drop_packed_duplicates(prefer_fp16(files))
    if not files:
        raise RuntimeError(f"{repo_id} 没有可下载的权重。")
    for path in files:
        target = dest / path.replace("/", os.sep)
        download_file(repo_id, path, target, sizes.get(path))
    if not weights_ready(dest):
        raise RuntimeError(f"下载结束，但 {dest} 里没有权重。")
    (dest / "model_id.txt").write_text(repo_id + "\n", encoding="utf-8")
    print(f"ready {repo_id} → {dest}")
    return dest


def download_tts(dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    print(f"listing {TTS_REPO} (pinned voice)")
    for remote in TTS_FILES:
        name = remote.rsplit("/", 1)[-1]
        download_file(TTS_REPO, remote, dest / name, None)
    onnx = dest / "en_US-lessac-medium.onnx"
    config = dest / "en_US-lessac-medium.onnx.json"
    if not onnx.is_file() or not config.is_file():
        raise RuntimeError(f"TTS 权重不完整：{dest}")
    (dest / "model_id.txt").write_text(TTS_REPO + "\n", encoding="utf-8")
    print(f"ready tts → {dest}")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download sd-turbo, text-to-video, and Piper the Windows way")
    parser.add_argument(
        "--only",
        choices=("image", "video", "tts", "image-xl", "video-xl", "tts-xl", "pipeline", "all", "image-base", "image-jug", "image-flux"),
        default="all",
    )
    args = parser.parse_args()
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    if args.only in ("image", "all"):
        download_repo(IMAGE_REPO, IMAGE_DIR)
    if args.only in ("image-xl", "pipeline"):
        download_repo(IMAGE_XL_REPO, IMAGE_XL_DIR)
    if args.only in ("image-base",):
        download_repo(IMAGE_BASE_REPO, IMAGE_BASE_DIR)
    if args.only in ("image-jug",):
        download_repo(IMAGE_JUG_REPO, IMAGE_JUG_DIR)
    if args.only in ("image-flux",):
        download_repo(FLUX_REPO, FLUX_DIR)
    if args.only in ("video", "all"):
        download_repo(VIDEO_REPO, VIDEO_DIR)
    if args.only in ("video-xl", "pipeline"):
        download_repo(VIDEO_XL_REPO, VIDEO_XL_DIR)
    if args.only in ("tts", "all"):
        download_tts(TTS_DIR)
    if args.only in ("tts-xl", "pipeline"):
        download_repo(TTS_XL_REPO, TTS_XL_DIR)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        sys.exit(1)
