#!/usr/bin/env python3
"""Inject a Valhalla fusion patch into the denoiser, the way patch_lm patches Qwen.

apply_legacy_patch_tensors does two things with the pipeline's ValhallaModelPatch:
  * each transformer group's input layernorm is scaled by
    1 + strength * (quad_layer_scales[q] - 1)
  * project_patch(patch_vector) is added, times strength, into the conditioned rows

Image and video models have no token embedding of the fact. The same two edits
land on the denoiser's own blocks: norm1 inside each transformer block, and a
residual of the projected patch on that block's output. Strength is the fusion
default, 0.08. The patch vector is the one the running valhalla_base session
already stored on its fragments.
"""

from __future__ import annotations

import json
from pathlib import Path

import bridge

STRENGTH = 0.08

# Geometry tiers from valhalla-transformer (08_TILE_STEMCELL_TOTAL_PATCH):
# legacy patches layernorms + one embedding-sized residual, full adds every
# layernorm with per-layer body signatures, total also walks attention and MLP.
MODE_STRENGTH = {"legacy": 0.08, "full": 0.02, "total": 0.015}
RANDOM_SEED = 20260616

# Decomposition follows the original Fate path, not hand-tuned factors:
# TriadSession routes signals through Hub (hub_routed quad counts), Tile
# (tiles complete with data_signature + activation), and StemCell (cells
# sorted by energy); build_model_patch then fuses them in stream order
# (tile signatures sequentially, stem signatures at 0.5 weight) and derives
# quad_layer_scales per body. This module consumes those exported Fate
# quantities as-is. The only constants kept are the transformer repo's
# empirical per-mode strengths (legacy 0.08 / full 0.02 / total 0.015);
# per-layer assignment and weighting come from Fate activations.
BLOCK_CLASS_NAMES = {
    "BasicTransformerBlock",
    "CogVideoXBlock",
    "FluxTransformerBlock",
    "FluxSingleTransformerBlock",
}


def profile_denoiser(root, root_name: str = "denoiser") -> dict:
    """Architecture profile of a denoiser, whatever family it belongs to."""
    blocks = []
    class_names: dict[str, int] = {}
    for name, module in root.named_modules():
        cls = module.__class__.__name__
        if cls in BLOCK_CLASS_NAMES or (hasattr(module, "norm1") and (hasattr(module, "attn1") or hasattr(module, "attn"))):
            blocks.append((name, module))
            class_names[cls] = class_names.get(cls, 0) + 1
    sites_per_block = {"legacy": 1, "full": 0, "total": 0}
    if blocks:
        norm_pool = ("norm1", "norm2", "norm3", "norm1_context", "norm")
        child_pool = ("attn1", "attn2", "ff", "attn")
        norms = max(sum(1 for n in norm_pool if getattr(m, n, None) is not None) for _, m in blocks)
        children = max(sum(1 for n in child_pool if getattr(m, n, None) is not None) for _, m in blocks)
        sites_per_block = {"legacy": 1, "full": norms + 1, "total": norms + 1 + children}
    hidden = set()
    for _, module in blocks:
        size = int(getattr(module, "dim", 0) or getattr(module, "inner_dim", 0) or 0)
        if not size:
            for child_name in ("attn1", "attn", "attn2"):
                child = getattr(module, child_name, None)
                if child is not None and getattr(child, "inner_dim", None):
                    size = int(child.inner_dim)
                    break
        if size:
            hidden.add(size)
    return {
        "root": root_name,
        "block_count": len(blocks),
        "block_classes": class_names,
        "sites_per_block": sites_per_block,
        "hidden_sizes": sorted(hidden),
    }


def adaptive_plan(profile: dict, body_source: str, mode: str, steps: int, patch: dict | None = None) -> dict:
    """Patch plan anchored on Fate quantities, with a calibration ladder.

    No model-size formulas: the transformer repo's per-mode strength is the
    anchor, per-layer weighting inside ModulePatch comes from Fate tile
    activations / stem signature energies, and the only adaptation loop is
    the probe (generate, measure, pick), mirroring Fate QA feedback. The
    ladder is a plain doubling search grid around the anchor.
    """
    base = MODE_STRENGTH[mode]
    ladder = sorted({round(base * factor, 4) for factor in (0.5, 1.0, 2.0, 4.0)})
    fate_stats = {}
    if patch:
        activations = patch.get("tile_activations") or []
        fate_stats = {
            "tile_avg_activation": patch.get("tile_avg_activation", 0.0),
            "tile_activations": len(activations),
            "signals_emitted": patch.get("signals_emitted", 0),
            "hub_spread": round(max(patch["hub_prefs"]) - min(patch["hub_prefs"]), 4) if patch.get("hub_prefs") else 0.0,
        }
    return {
        "mode": mode,
        "body_source": body_source,
        "steps": steps,
        "base_strength": base,
        "recommended_strength": base,
        "probe_ladder": ladder,
        "profile_blocks": profile.get("block_count", 0),
        "fate_stats": fate_stats,
    }


def load_export_patch(path) -> dict:
    """Load a valhalla_model_export ExportDoc as a module patch."""
    doc = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    mp = doc["report"]["model_patch"]
    return {
        "patch_vector": [float(v) for v in mp["patch_vector"]],
        "hub_prefs": [float(v) for v in mp.get("fate_affinities") or []],
        "quad_layer_scales": [float(v) for v in mp["quad_layer_scales"]],
        "tile_signatures": [[float(v) for v in s] for s in mp.get("tile_signatures") or []],
        "stem_signatures": [[float(v) for v in s] for s in mp.get("stem_signatures") or [] if s],
        "body_source": mp.get("body_source", "hub"),
        "tile_count": int(mp.get("tile_count", 0)),
        "stem_cell_count": int(mp.get("stem_cell_count", 0)),
        "tile_activations": [float(v) for v in mp.get("tile_activations") or []],
        "tile_avg_activation": float(mp.get("tile_avg_activation") or 0.0),
        "tile_connection_count": int(mp.get("tile_connection_count") or 0),
        "stem_cluster_energies": [float(v) for v in mp.get("stem_cluster_energies") or []],
        "signals_emitted": int(mp.get("signals_emitted") or 0),
        "dim": len(mp["patch_vector"]),
        "fragments": 0,
    }


def random_patch(like: dict, seed: int = RANDOM_SEED) -> dict:
    """Same-geometry random control (valhalla-transformer Tier A random arm)."""
    import random

    rng = random.Random(seed)
    dim = int(like.get("dim") or 256)
    vector = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    norm = sum(v * v for v in vector) ** 0.5 or 1e-9
    scales = list(like.get("quad_layer_scales") or [1.0, 1.0, 1.0, 1.0])
    deviation = max(abs(s - 1.0) for s in scales) or 0.001
    return {
        "patch_vector": [v / norm for v in vector],
        "hub_prefs": list(like.get("hub_prefs") or []),
        "quad_layer_scales": [1.0 + rng.uniform(-deviation, deviation) for _ in scales],
        "tile_signatures": [],
        "stem_signatures": [],
        "body_source": "random",
        "tile_count": 0,
        "stem_cell_count": 0,
        "dim": dim,
        "fragments": 0,
    }


def project_patch(patch_vec: list[float], hidden: int) -> list[float]:
    out = [0.0] * hidden
    if not patch_vec or hidden <= 0:
        return out
    width = len(patch_vec)
    for index in range(hidden):
        out[index] = float(patch_vec[index % width])
    norm = sum(value * value for value in out) ** 0.5 or 1e-9
    return [value / norm for value in out]


def quad_scales(hub_prefs: list[float]) -> list[float]:
    scales = [1.0, 1.0, 1.0, 1.0]
    prefs = [float(value) for value in list(hub_prefs)[:4]]
    if len(prefs) < 4:
        return scales
    mean = sum(prefs) / 4.0
    for index, value in enumerate(prefs):
        scales[index] = 1.0 + 0.05 * (value - mean)
    return scales


def fragment_patch(lines: list[str]) -> dict:
    """Two native turns, so the session closes a fragment that carries patch_vector."""
    notes = [{"text": line} for line in lines]
    question = "What is the fact?"
    first = bridge.valhalla(bridge.build_payload(question, {}, None, notes, None, bridge.CFG))
    session = first.get("session")
    if not isinstance(session, dict):
        raise RuntimeError("valhalla_base 没有返回会话。")
    second = bridge.valhalla(bridge.build_payload(question, {}, session, None, None, bridge.CFG))
    kept = second.get("session") if isinstance(second.get("session"), dict) else session
    fragments = [item for item in (kept.get("fragments") or []) if isinstance(item, dict)]
    vectors = [item.get("patch_vector") for item in fragments if item.get("patch_vector")]
    if not vectors:
        raise RuntimeError("会话片段里没有 patch_vector。")
    width = len(vectors[0])
    mean = [sum(row[index] for row in vectors) / len(vectors) for index in range(width)]
    prefs = fragments[-1].get("hub_prefs") or []
    return {
        "patch_vector": mean,
        "hub_prefs": prefs,
        "quad_layer_scales": quad_scales(prefs),
        "fragments": len(fragments),
        "dim": width,
    }


class ModulePatch:
    """Forward hooks. Removing them restores the untouched weights.

    mode follows valhalla-transformer geometry: legacy scales norm1 and adds
    one embedding-sized residual; full shifts every layernorm output with the
    per-layer body signature; total also shifts attention and MLP outputs.
    """

    def __init__(self, patch: dict, strength: float | None = None, mode: str = "legacy") -> None:
        if mode not in MODE_STRENGTH:
            raise ValueError(f"unknown patch mode {mode}")
        self.patch = patch
        self.mode = mode
        self.strength = MODE_STRENGTH[mode] if strength is None else strength
        self.handles: list = []
        self.modules: list[str] = []
        self.fires = 0
        self.delta_sites = 0
        self.delta_norm = 0.0

    def layer_vector(self, index: int, n_blocks: int = 0) -> tuple[list[float], float]:
        """Per-layer (direction, Fate weight) for one block.

        Assignment follows Fate's own order: tile/stem signatures keep the
        completion/energy order build_model_patch exported, spread across
        depth by position (block i reads signature floor(i * n_sig / n_blocks)).
        The weight is Fate's tile activation for that signature (enriched
        signatures carry it at index 2; tile_activations is the fallback),
        or the relative stem signature norm (build_model_patch sorts stem
        cells by energy). Legacy returns the fused vector at weight 1.
        """
        base = self.patch["patch_vector"]
        if self.mode == "legacy":
            return base, 1.0
        tiles = self.patch.get("tile_signatures") or []
        stems = self.patch.get("stem_signatures") or []
        if not tiles and not stems:
            return base, 1.0
        total = max(n_blocks, 1)

        def pick(sigs: list, i: int):
            pos = min(i * len(sigs) // total, len(sigs) - 1)
            return pos, sigs[pos]

        mixed = list(base)
        width = len(mixed)
        weight = 1.0
        if tiles:
            pos, sig = pick(tiles, index)
            for j, value in enumerate(sig):
                mixed[j % width] += float(value)
            activations = self.patch.get("tile_activations") or []
            if len(sig) > 2:
                weight = float(sig[2])
            elif pos < len(activations):
                weight = float(activations[pos])
        if stems:
            _pos, sig = pick(stems, index)
            for j, value in enumerate(sig):
                mixed[j % width] += 0.5 * float(value)
            if not tiles and sig:
                norms = [sum(float(v) * float(v) for v in s) ** 0.5 for s in stems if s]
                mean_norm = sum(norms) / max(len(norms), 1) or 1e-9
                weight = (sum(float(v) * float(v) for v in sig) ** 0.5) / mean_norm
        norm = sum(v * v for v in mixed) ** 0.5 or 1e-9
        return [v / norm for v in mixed], weight

    def attach(self, root, root_name: str) -> list[str]:
        import torch

        blocks = [
            (name, module)
            for name, module in root.named_modules()
            if module.__class__.__name__ in BLOCK_CLASS_NAMES
            or (hasattr(module, "norm1") and (hasattr(module, "attn1") or hasattr(module, "attn")))
        ]
        if not blocks:
            raise RuntimeError(f"{root_name} 里没有 TransformerBlock。")
        scales = self.patch["quad_layer_scales"]
        share = self.strength / max(len(blocks), 1)
        cache: dict[tuple, torch.Tensor] = {}

        def bias_for(vector: list[float], hidden: int, device):
            key = (id(vector), hidden, str(device))
            found = cache.get(key)
            if found is None:
                found = torch.tensor(project_patch(vector, hidden), device=device, dtype=torch.float32)
                cache[key] = found
            return found

        def shift(tensor, vector, portion: float, block_scale: float):
            hidden = tensor.shape[-1]
            proj = bias_for(vector, hidden, tensor.device)
            rms = tensor.detach().float().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-4)
            delta = proj * (hidden ** 0.5) * rms * (portion * block_scale)
            return tensor + delta.to(dtype=tensor.dtype)

        def shifted(output, vector, portion, block_scale, index: int = 0):
            tensor = output[index] if isinstance(output, tuple) and len(output) > index else output
            if not hasattr(tensor, "shape") or tensor.ndim < 2:
                return output
            self.delta_sites += 1
            self.delta_norm += float(portion) ** 2
            updated = shift(tensor, vector, portion, block_scale)
            if isinstance(output, tuple):
                items = list(output)
                items[index] = updated
                return tuple(items)
            return updated

        norm_names = ["norm1"] if self.mode == "legacy" else ["norm1", "norm2", "norm3", "norm1_context", "norm"]
        # Fate-weighted portions: legacy is one embedding-sized residual on
        # the first block (patch_lm adds the bias once); full/total weight
        # every block by its Fate activation, mean-normalised so the average
        # block still receives the mode strength.
        layered = [self.layer_vector(i, len(blocks)) for i in range(len(blocks))]
        mean_weight = sum(w for _v, w in layered) / max(len(layered), 1)
        for index, (name, module) in enumerate(blocks):
            scale = scales[min(index * 4 // max(len(blocks), 1), 3)]
            norm_factor = 1.0 + self.strength * (scale - 1.0)
            vector, fate_weight = layered[index]
            full_name = f"{root_name}.{name}"
            is_sdxl_block = module.__class__.__name__ == "BasicTransformerBlock"
            if self.mode == "legacy":
                site_portion = self.strength if index == 0 else 0.0
            else:
                site_portion = self.strength * (fate_weight / mean_weight if mean_weight > 0 else 1.0)

            for norm_name in norm_names:
                norm = getattr(module, norm_name, None)
                if norm is None:
                    continue
                # SDXL norms return the tensor attention reads. CogVideoX norms
                # return gate tuples, so their shift lands on the block output.
                # Flux single blocks expose a plain tensor norm as well.
                inject = (is_sdxl_block or module.__class__.__name__ == "FluxSingleTransformerBlock") and (
                    self.mode != "legacy" or norm_name in ("norm1", "norm")
                )

                def norm_hook(_m, _i, output, factor=norm_factor, portion=site_portion, bs=scale, vec=vector, inject=inject):
                    self.fires += 1
                    if not hasattr(output, "shape"):
                        return output
                    scaled = output * factor
                    if inject:
                        return shifted(scaled, vec, portion, bs)
                    return scaled

                self.handles.append(norm.register_forward_hook(norm_hook))
                self.modules.append(f"{full_name}.{norm_name}")

            # Block residual: CogVideoX always (its norms gate), plus every
            # block in full/total geometry. FluxTransformerBlock returns
            # (text, image) so the image stream is element 1.
            block_inject = (not is_sdxl_block) or self.mode != "legacy"
            block_index = 1 if module.__class__.__name__ == "FluxTransformerBlock" else 0

            def block_hook(_m, _i, output, portion=site_portion, bs=scale, vec=vector, inject=block_inject, idx=block_index):
                self.fires += 1
                if not inject:
                    return output
                return shifted(output, vec, portion, bs, index=idx)

            self.handles.append(module.register_forward_hook(block_hook))
            self.modules.append(full_name)

            if self.mode == "total":
                for child_name in ("attn1", "attn2", "ff", "attn"):
                    child = getattr(module, child_name, None)
                    if child is None:
                        continue

                    def child_hook(_m, _i, output, portion=self.strength, bs=scale, vec=vector):
                        self.fires += 1
                        return shifted(output, vec, portion, bs)

                    self.handles.append(child.register_forward_hook(child_hook))
                    self.modules.append(f"{full_name}.{child_name}")
        self.delta_norm = self.delta_norm ** 0.5
        return self.modules

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _load_sdxl():
    import torch
    from diffusers import AutoPipelineForText2Image

    model_dir = bridge_image_dir()
    dtype = torch.float16
    pipe = AutoPipelineForText2Image.from_pretrained(
        str(model_dir), torch_dtype=dtype, variant="fp16"
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _load_video(strategy: str | None = None, priority: str = "speed"):
    import torch
    from diffusers import DiffusionPipeline

    import adaptive_framework as af

    pipe = DiffusionPipeline.from_pretrained(str(bridge_video_dir()), torch_dtype=torch.float16)
    # Residency is a framework scheduling decision (measured: cpu_offload
    # 8.7s/15.3GB vs resident 193s/22.5GB vs sequential 101s/7.1GB).
    af.apply_loader(pipe, "cogvideox", strategy=strategy, priority=priority)
    return pipe


def bridge_image_dir() -> Path:
    from download import IMAGE_XL_DIR

    return IMAGE_XL_DIR


def bridge_video_dir() -> Path:
    from download import VIDEO_XL_DIR

    return VIDEO_XL_DIR


def generate_image(pipe, prompt: str, out_path: Path, seed: int) -> None:
    import torch

    generator = torch.Generator(device="cuda").manual_seed(seed)
    image = pipe(
        prompt=prompt,
        num_inference_steps=4,
        guidance_scale=0.0,
        generator=generator,
        height=512,
        width=512,
    ).images[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def generate_video(pipe, prompt: str, frame_dir: Path, seed: int, steps: int = 2) -> list[Path]:
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    result = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        num_frames=9,
        height=480,
        width=720,
        guidance_scale=6.0,
        generator=generator,
    )
    pictures = []
    frames = result.frames[0]
    frame_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for index, frame in enumerate(frames):
        if hasattr(frame, "save"):
            picture = frame
        else:
            import numpy as np
            from PIL import Image

            array = np.asarray(frame)
            if array.dtype != np.uint8:
                scale = 255.0 if float(array.max()) <= 1.0 else 1.0
                array = np.clip(array * scale, 0, 255).astype(np.uint8)
            picture = Image.fromarray(array)
        path = frame_dir / f"frame_{index:02d}.png"
        picture.save(path)
        pictures.append(picture)
        saved.append(path)
    return saved


def pixel_distance(path_a: Path, path_b: Path) -> float:
    from PIL import Image

    left = Image.open(path_a).convert("RGB").resize((64, 64))
    right = Image.open(path_b).convert("RGB").resize((64, 64))
    pairs = zip(left.getdata(), right.getdata())
    total = 0.0
    count = 0
    for a, b in pairs:
        for channel in range(3):
            delta = a[channel] - b[channel]
            total += delta * delta
            count += 1
    return (total / max(count, 1)) ** 0.5


def dump_patch(patch: dict) -> dict:
    vector = patch["patch_vector"]
    return {
        "dim": patch["dim"],
        "fragments": patch.get("fragments", 0),
        "body_source": patch.get("body_source", "session"),
        "tile_count": patch.get("tile_count", 0),
        "stem_cell_count": patch.get("stem_cell_count", 0),
        "tile_signatures": len(patch.get("tile_signatures") or []),
        "stem_signatures": len(patch.get("stem_signatures") or []),
        "quad_layer_scales": patch["quad_layer_scales"],
        "hub_prefs": patch["hub_prefs"],
        "norm": sum(value * value for value in vector) ** 0.5,
        "head": [round(value, 4) for value in vector[:8]],
    }
