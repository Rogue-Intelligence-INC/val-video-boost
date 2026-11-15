#!/usr/bin/env python3
"""Same prompt, two fusion patches, injected inside the denoiser blocks.

The passphrase patch and the protocol patch both come from valhalla_base
fragments. SDXL's transformer blocks and CogVideoX's transformer blocks receive
the layernorm scale and the patch residual. The prompt and the seed stay fixed,
so a pixel change is the patch reaching the module.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bridge  # noqa: E402
import module_patch  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "reports" / "media_pipeline" / "module_patch"
PROMPT = "a square ceramic sign standing in a quiet room"


def run_image(passphrase: dict, protocol: dict) -> dict:
    pipe = module_patch._load_sdxl()
    root = pipe.unet
    same_a = OUT / "sdxl_passphrase_a.png"
    same_b = OUT / "sdxl_passphrase_b.png"
    other = OUT / "sdxl_protocol.png"
    hooked = []
    fires = []
    for patch, path, seed in (
        (passphrase, same_a, 7),
        (passphrase, same_b, 7),
        (protocol, other, 7),
    ):
        handle = module_patch.ModulePatch(patch)
        names = handle.attach(root, "unet")
        hooked = names
        try:
            module_patch.generate_image(pipe, PROMPT, path, seed)
        finally:
            fires.append(handle.fires)
            handle.remove()
    del pipe
    import torch

    torch.cuda.empty_cache()
    repeat = module_patch.pixel_distance(same_a, same_b)
    cross = module_patch.pixel_distance(same_a, other)
    return {
        "modules": len(hooked),
        "module_head": hooked[:6],
        "module_tail": hooked[-4:],
        "hook_fires": fires,
        "repeat_l2": repeat,
        "cross_patch_l2": cross,
        "patch_changes_pixels": cross > repeat + 0.5,
    }


def run_video(passphrase: dict, protocol: dict) -> dict:
    pipe = module_patch._load_video()
    root = pipe.transformer
    held_dir = OUT / "video_passphrase"
    other_dir = OUT / "video_protocol"
    hooked = []
    fires = []
    for patch, folder in ((passphrase, held_dir), (protocol, other_dir)):
        handle = module_patch.ModulePatch(patch)
        names = handle.attach(root, "transformer")
        hooked = names
        try:
            module_patch.generate_video(pipe, PROMPT, folder, 7)
        finally:
            fires.append(handle.fires)
            handle.remove()
    del pipe
    import torch

    torch.cuda.empty_cache()
    left = held_dir / "frame_00.png"
    right = other_dir / "frame_00.png"
    distance = module_patch.pixel_distance(left, right) if left.is_file() and right.is_file() else 0.0
    return {
        "modules": len(hooked),
        "module_head": hooked[:4],
        "hook_fires": fires,
        "frame0_cross_l2": distance,
        "patch_changes_frames": distance > 0.5,
    }


def main() -> None:
    print("fetch passphrase patch")
    passphrase = module_patch.fragment_patch([bridge.FACT])
    print("fetch protocol patch")
    protocol = module_patch.fragment_patch([bridge.PROTOCOL])
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"sdxl internal patch, prompt={PROMPT!r}")
    image = run_image(passphrase, protocol)
    print(
        f"  blocks={image['modules']} fires={image['hook_fires']} "
        f"repeat_l2={image['repeat_l2']:.3f} cross_l2={image['cross_patch_l2']:.3f}"
    )
    print("cogvideox internal patch, 2 steps")
    video = run_video(passphrase, protocol)
    print(
        f"  blocks={video['modules']} fires={video['hook_fires']} "
        f"frame0_l2={video['frame0_cross_l2']:.3f}"
    )
    passed = (
        passphrase["dim"] > 0
        and image["modules"] > 0
        and video["modules"] > 0
        and min(image["hook_fires"]) > 0
        and min(video["hook_fires"]) > 0
        and image["patch_changes_pixels"]
        and image["repeat_l2"] < 0.5
        and video["patch_changes_frames"]
    )
    report = {
        "protocol": "valhalla-module-patch-v1",
        "strength": module_patch.STRENGTH,
        "prompt": PROMPT,
        "passphrase_patch": module_patch.dump_patch(passphrase),
        "protocol_patch": module_patch.dump_patch(protocol),
        "image": image,
        "video": video,
        "verdict": "PASS" if passed else "FAIL",
    }
    path = OUT / "report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"verdict {report['verdict']}")
    print(f"wrote {path}")
    if not passed:
        sys.exit(2)


if __name__ == "__main__":
    main()
