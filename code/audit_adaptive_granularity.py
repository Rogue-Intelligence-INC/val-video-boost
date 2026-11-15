#!/usr/bin/env python3
"""Granularity audit: how finely does the adaptive patch decompose a model?

Builds each denoiser on the meta device (config only, no weights) and
measures, per model family and per patch mode:
  * structural decomposition: blocks, submodule inventory, parameter share
  * injection coverage: hooked modules vs total modules / parameters
  * direction granularity: how many *distinct* patch directions actually
    reach the blocks (layer_vector uniqueness, pairwise cosine)
  * channel granularity: project_patch tiles the 256-d vector, so channels
    repeat with period 256 -- measured directly
  * what stays coarse: quad grouping (4), one strength for all timesteps,
    text encoders / VAE untouched, FLUX text stream untouched
  * Valhalla-side usage: which ExportDoc fields exist vs which are consumed
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import download  # noqa: E402
import module_patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "media_pipeline" / "adaptive_patch"
PATCH_DIR = ROOT / "reports" / "media_pipeline" / "transformer_aligned" / "patches"


def build_meta(kind: str):
    import torch

    with torch.device("meta"):
        if kind == "sdxl_unet":
            from diffusers import UNet2DConditionModel

            return UNet2DConditionModel.from_config(
                UNet2DConditionModel.load_config(str(download.IMAGE_XL_DIR), subfolder="unet")
            )
        if kind == "cogvideox_transformer":
            from diffusers import CogVideoXTransformer3DModel

            return CogVideoXTransformer3DModel.from_config(
                CogVideoXTransformer3DModel.load_config(str(download.VIDEO_XL_DIR), subfolder="transformer")
            )
        if kind == "flux_transformer":
            from diffusers import FluxTransformer2DModel

            return FluxTransformer2DModel.from_config(
                FluxTransformer2DModel.load_config(str(download.FLUX_DIR), subfolder="transformer")
            )
    raise ValueError(kind)


def direction_stats(patch: dict, mode: str, n_blocks: int) -> dict:
    handle = module_patch.ModulePatch(patch, mode=mode)
    layered = [handle.layer_vector(i, n_blocks) for i in range(n_blocks)]
    vectors = [vec for vec, _w in layered]
    unique = {tuple(round(v, 4) for v in vec) for vec in vectors}
    cosines = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            a, b = vectors[i], vectors[j]
            cosines.append(sum(x * y for x, y in zip(a, b)))
    weights = [w for _v, w in layered]
    return {
        "blocks": n_blocks,
        "distinct_directions": len(unique),
        "mean_pairwise_cosine": round(sum(cosines) / max(len(cosines), 1), 4) if cosines else 1.0,
        "min_pairwise_cosine": round(min(cosines), 4) if cosines else 1.0,
        "fate_weight_min": round(min(weights), 4) if weights else 1.0,
        "fate_weight_max": round(max(weights), 4) if weights else 1.0,
    }


def audit_model(kind: str, patch: dict) -> dict:
    model = build_meta(kind)
    total_modules = sum(1 for _ in model.named_modules())
    total_params = sum(p.numel() for p in model.parameters())
    profile = module_patch.profile_denoiser(model, kind)
    modes = {}
    for mode in ("legacy", "full", "total"):
        handle = module_patch.ModulePatch(patch, mode=mode)
        names = handle.attach(model, kind)
        hooked_params = 0
        seen = set()
        for name, module in model.named_modules():
            for hooked in names:
                suffix = hooked[len(kind) + 1:]
                if name == suffix and id(module) not in seen:
                    seen.add(id(module))
                    hooked_params += sum(p.numel() for p in module.parameters(recurse=False))
        modes[mode] = {
            "hooked_modules": len(names),
            "coverage_modules_pct": round(100 * len(names) / max(total_modules, 1), 2),
            "direct_site_params_pct": round(100 * hooked_params / max(total_params, 1), 2),
            "directions": direction_stats(patch, mode, profile["block_count"]),
        }
        handle.remove()
    return {
        "total_modules": total_modules,
        "total_params_m": round(total_params / 1e6, 1),
        "profile": profile,
        "modes": modes,
    }


def main() -> None:
    triad = module_patch.load_export_patch(PATCH_DIR / "triad.json")
    results = {}
    for kind in ("sdxl_unet", "cogvideox_transformer", "flux_transformer"):
        print(f"audit {kind}")
        results[kind] = audit_model(kind, triad)
        p = results[kind]["profile"]
        print(f"  blocks={p['block_count']} classes={p['block_classes']} hidden={p['hidden_sizes']} sites={p['sites_per_block']}")

    proj = module_patch.project_patch(triad["patch_vector"], 3072)
    channel_period = all(abs(proj[i] - proj[i + 256]) < 1e-12 for i in range(3072 - 256))

    doc = json.loads((PATCH_DIR / "triad.json").read_text(encoding="utf-8-sig"))
    mp = doc["report"]["model_patch"]
    available = {
        "patch_vector": len(mp.get("patch_vector") or []),
        "quad_layer_scales": len(mp.get("quad_layer_scales") or []),
        "tile_signatures": len(mp.get("tile_signatures") or []),
        "stem_signatures": len([s for s in mp.get("stem_signatures") or [] if s]),
        "tile_activations": len(mp.get("tile_activations") or []),
        "stem_cluster_signatures": len(mp.get("stem_cluster_signatures") or []),
        "stem_cluster_energies": len(mp.get("stem_cluster_energies") or []),
        "tile_neurons": len(mp.get("tile_neurons") or []),
        "fate_affinities": len(mp.get("fate_affinities") or []),
    }
    consumed = {"patch_vector", "quad_layer_scales", "tile_signatures", "stem_signatures", "fate_affinities"}

    report = {
        "protocol": "valhalla-adaptive-granularity-audit-v1",
        "models": results,
        "channel_projection": {"hidden_example": 3072, "period_256_repeat": channel_period},
        "coarse_dimensions": {
            "quad_groups": 4,
            "strength_per_timestep": "single value for all sampling steps (no timestep input to ModulePatch)",
            "text_encoders_hooked": 0,
            "vae_hooked": 0,
            "flux_text_stream": "dual-stream blocks shift image stream (tuple element 1); text stream passes unshifted",
        },
        "valhalla_side": {
            "export_fields_available": available,
            "fields_consumed_by_module_patch": sorted(consumed),
            "fields_available_but_unused": sorted(set(available) - consumed),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "granularity_audit.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
