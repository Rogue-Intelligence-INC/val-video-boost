#!/usr/bin/env python3
"""Plugin decomposed into Valhalla gating / blocks, and a no-plugin arm.

Line 1 -- plugin, blockised: IP-Adapter normally sprays the reference
face into every cross-attention processor. Here it is decomposed per
UNet block region (down / mid / up) and gated: only the regions Fate's
depth profile (quad_layer_scales across depth) selects receive face
conditioning. Arms: uniform (control), mid+up only, down only, and
fate_profile (per-processor scale = quad profile interpolated across
depth, mean-matched to uniform so only the SHAPE differs).

Line 2 -- no plugin at all, Valhalla-native carriers:
  fate_face_patch : hero face -> Valhalla dl_front image encoder ->
                    image_features on the identity corpus line -> Fate
                    core_fusion session -> fragment patch_vector. The
                    face enters through Valhalla's own multimodal port.
  arcface_blocks  : ArcFace 512-d decomposed Valhalla-style: projected
                    to the 256-d patch_vector, injected per denoiser
                    block by ModulePatch (total). Plugin info, but the
                    carrier/gating is entirely Valhalla's block machinery.
Metric throughout: ArcFace cosine to each seed's studio hero (the
ip_face reference numbers: uniform plugin 0.581, no-plugin text patch
0.433, baseline 0.471, different person 0.152).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bridge
import character_bible as cb
import download
import fate_iterative as fi
import module_patch
from face_id import FaceID
from test_big_models import load_pipe
from test_fate_iterative_character import SCENES, SEEDS, prompt
from test_ip_face import IP_ROOT, IP_SCALE, face_ref

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "valhalla_native"
PATCH_JSON = ROOT / "data" / "identity_patch.json"
SIZE, STEPS = 768, 25


def gen_plain(pipe, text, out_path, seed, patch=None, ref=None):
    import torch
    from PIL import Image

    if out_path.exists():
        return 0.0
    gen = torch.Generator(device="cuda").manual_seed(seed)
    started = time.time()
    handle = None
    if patch is not None:
        handle = module_patch.ModulePatch(patch, mode="total")
        handle.attach(pipe.unet, "unet")
    kw = dict(prompt=text, num_inference_steps=STEPS, guidance_scale=7.0,
              generator=gen, height=SIZE, width=SIZE)
    if ref is not None:
        kw["ip_adapter_image"] = Image.open(ref).convert("RGB")
    try:
        img = pipe(**kw).images[0]
    finally:
        if handle is not None:
            handle.remove()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return round(time.time() - started, 2)


def fate_face_patch(hero: Path) -> tuple[dict, dict]:
    vec, encoder = bridge.encode("image_path", path=str(hero))
    lines = cb.IDENTITY_A["lines"]
    notes = [{"text": lines[0], "image_features": vec}] + [{"text": t} for t in lines[1:]]
    q = "What is the fact?"
    r1 = bridge.valhalla(bridge.build_payload(q, {}, None, notes, None, bridge.CFG))
    s = r1.get("session")
    r2 = bridge.valhalla(bridge.build_payload(q, {}, s, None, None, bridge.CFG))
    s2 = r2.get("session") or s
    frags = [f for f in (s2.get("fragments") or []) if f.get("patch_vector")]
    mean = fi._mean([[float(x) for x in f["patch_vector"]] for f in frags])
    text_only = module_patch.fragment_patch(lines)
    info = {"encoder": encoder, "fragments": len(frags),
            "cos_to_text_patch": fi.cosine(mean, text_only["patch_vector"])}
    patch = dict(text_only)
    patch["patch_vector"] = fi._norm(mean)
    patch["body_source"] = "fate_face_fusion"
    return patch, info


def arcface_block_patch(fid: FaceID, hero: Path, triad: dict) -> dict:
    emb = fid.embedding(hero)
    projected = [(emb[i] + emb[i + 1]) / 2.0 for i in range(0, 512, 2)]  # 512 -> 256
    patch = dict(triad)
    patch["patch_vector"] = fi._norm(projected)
    patch["body_source"] = "arcface_decomposed_blocks"
    return patch


def region_of(name: str) -> str:
    if name.startswith("down_blocks"):
        return "down"
    if name.startswith("mid_block"):
        return "mid"
    return "up"


def set_block_scales(pipe, mode: str, quad: list[float], base: float = IP_SCALE) -> dict:
    procs = pipe.unet.attn_processors
    names = list(procs.keys())
    counts = {}
    for i, name in enumerate(names):
        proc = procs[name]
        if not hasattr(proc, "scale"):
            continue
        region = region_of(name)
        frac = i / max(len(names) - 1, 1)
        if mode == "uniform":
            scale = base
        elif mode == "mid_up":
            scale = 0.0 if region == "down" else base
        elif mode == "down_only":
            scale = base if region == "down" else 0.0
        elif mode == "fate_profile":
            q = quad[min(int(frac * 4), 3)]
            scale = base * (0.5 + (q - 1.0) * 20.0)  # quad ~1.0+-0.0002 -> shape only
            scale = max(0.0, scale)
        else:
            raise ValueError(mode)
        proc.scale = [scale]  # diffusers 0.42: per-adapter list
        counts[region] = counts.get(region, 0) + 1
    return counts


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    triad = module_patch.load_export_patch(PATCH_JSON)
    fid = FaceID()
    hero7 = ROOT / "reports" / "fate_iterative_character" / "static_triad" / "s7_studio.png"
    ff_patch, ff_info = fate_face_patch(hero7)
    print("fate_face_patch", ff_info)
    ab_patch = arcface_block_patch(fid, hero7, triad)

    pipe = load_pipe(download.IMAGE_JUG_DIR)
    report = {"protocol": "val-video-boost-valhalla-native-v1", "fate_face_info": ff_info, "arms": {}}

    def run_arm(arm, patch=None, ref_for=None):
        rows = []
        for seed in SEEDS:
            hero = ROOT / "reports" / "fate_iterative_character" / "static_triad" / f"s{seed}_studio.png"
            hero_emb = fid.embedding(hero)
            ref = ref_for(seed, hero) if ref_for else None
            for scene in ("park", "harbour"):
                target = OUT / arm / f"s{seed}_{scene}.png"
                secs = gen_plain(pipe, prompt(scene), target, seed, patch=patch, ref=ref)
                emb = fid.embedding(target)
                cos = None if (emb is None or hero_emb is None) else fid.cosine(hero_emb, emb)
                rows.append({"seed": seed, "scene": scene, "seconds": secs, "arcface_cos": cos})
                print(arm, seed, scene, "cos", cos)
        ok = [r["arcface_cos"] for r in rows if r["arcface_cos"] is not None]
        report["arms"][arm] = {"rows": rows, "mean_cos": round(sum(ok) / len(ok), 4) if ok else None,
                               "min_cos": min(ok) if ok else None}

    try:
        # Line 2: no plugin
        run_arm("fate_face_patch", patch=ff_patch)
        run_arm("arcface_blocks", patch=ab_patch)
        # Line 1: plugin decomposed into block gates
        pipe.load_ip_adapter(IP_ROOT, subfolder="sdxl_models",
                             weight_name="ip-adapter-plus-face_sdxl_vit-h.safetensors",
                             image_encoder_folder="models/image_encoder")
        refs = {}
        def ref_for(seed, hero):
            if seed not in refs:
                refs[seed] = face_ref(fid, hero, OUT / "refs" / f"s{seed}_ref.png")
            return refs[seed]
        for mode in ("uniform", "mid_up", "down_only", "fate_profile"):
            counts = set_block_scales(pipe, mode, triad["quad_layer_scales"])
            print("gate", mode, counts)
            run_arm(f"ip_gate_{mode}", ref_for=ref_for)
    finally:
        import torch
        del pipe
        torch.cuda.empty_cache()
    report["reference"] = {"baseline_no_plugin": 0.471, "text_patch": 0.4333,
                           "ip_uniform_plugin_v1": 0.5811, "different_person": 0.1522}
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({a: v["mean_cos"] for a, v in report["arms"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
