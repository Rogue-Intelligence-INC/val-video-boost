#!/usr/bin/env python3
"""Adaptive model framework -- val-video-boost as a model-agnostic shell.

What was copied in from the other projects (mechanisms, not just weights):
  * valhalla-transformer / val-videopic-fusion adaptive patch:
    profile_denoiser -> adaptive_plan (probe ladder) -> per-rung probes
    scored on the objective itself under distortion guards (blank /
    clipped / sharpness), smallest winning strength kept. No hand-tuned
    per-model constants: anchors are per-FAMILY measured values and the
    probe decides (module_patch.adaptive_plan).
  * measured registries: every report this project produced becomes a
    capability record, so routing is by evidence, not by model fashion.

Framework responsibilities:
  discover   scan the local model root, classify pipelines by class name
  profile    architecture surface of a loaded denoiser (blocks, sites)
  plan       patch mode/strength ladder + hard operating limits learned
             on this machine (e.g. CogVideoX memory wall, FLUX load path)
  calibrate  run the probe ladder on a live model, pick strength
  route      pick (model, settings) per task from measured records

Hard limits are stored as data with their source report, so a future
model only needs a profile + a probe run to join the framework.
"""
from __future__ import annotations

import json
from pathlib import Path

MODELS_ROOT = Path(r"C:\Ai\models")
PROJECT = Path(__file__).resolve().parents[1]

# Family knowledge measured on THIS machine (source report in each record).
FAMILY_KB = {
    "sdxl": {
        "loader": "sdxl_full_gpu", "patch_modes": ["legacy", "full", "total"],
        "ip_adapter_face": True, "notes": "IP-Adapter Plus Face available (SDXL only)",
    },
    "flux": {
        "loader": "flux_group_offload", "patch_modes": ["legacy", "full", "total"],
        "ip_adapter_face": False,
        "notes": "enable_model_cpu_offload measured 577s/image; group offload ~5s. steps=4, guidance=0",
        "prior_strength": 0.0166, "source": "reports/prior_flux/report.json",
    },
    "cogvideox": {
        "loader": "video_cpu_offload", "patch_modes": ["legacy", "full", "total"],
        "ip_adapter_face": False,
        "resolution_wall": {"max_frames_tested_ok": 121, "collapse_at": 161,
                            "safe_resolution_over_33f": "512x288",
                            "native_resolution": "720x480 (<=33f only, memory thrash above)"},
        "notes": "T2V only: no last-frame input; cross-window identity needs I2V weights",
        "loader_measured": {"task": "17f 512x288 10 steps", "resident": "193.2s / 22.47GB (spills, slowest)",
                            "cpu_offload": "8.7s / 15.3GB (fastest)", "sequential": "101.5s / 7.05GB (lightest)",
                            "source": "reports/adaptive_framework/report_unload.json"},
        "source": "reports/single_film/report_limits.json",
    },
    "text2video_sd": {"loader": "video_cpu_offload", "patch_modes": [], "ip_adapter_face": False,
                      "notes": "legacy small model, superseded by cogvideox"},
    "llm": {"loader": "llm", "patch_modes": [], "ip_adapter_face": False, "notes": "not a denoiser"},
    "unknown": {"loader": "generic", "patch_modes": ["legacy", "full", "total"], "ip_adapter_face": False},
}

CLASS_TO_FAMILY = {
    "StableDiffusionXLPipeline": "sdxl", "AutoPipelineForText2Image": "sdxl",
    "FluxPipeline": "flux", "CogVideoXPipeline": "cogvideox",
    "TextToVideoSDPipeline": "text2video_sd",
}


def discover(root: Path = MODELS_ROOT) -> list[dict]:
    out = []
    for d in sorted(root.iterdir()):
        idx = d / "model_index.json"
        if not idx.is_file():
            continue
        try:
            cls = json.loads(idx.read_text(encoding="utf-8")).get("_class_name", "")
        except Exception:
            cls = ""
        fam = CLASS_TO_FAMILY.get(cls, "llm" if (d / "config.json").is_file() and not cls else "unknown")
        out.append({"name": d.name, "dir": str(d), "pipeline_class": cls,
                    "family": fam, "kb": FAMILY_KB.get(fam, FAMILY_KB["unknown"])})
    return out


def load_measured_registry() -> dict:
    """Capability records distilled from this project's own reports."""
    reg: dict[str, list] = {}

    def add(model, task, score, source, extra=None):
        reg.setdefault(model, []).append({"task": task, "score": score, "source": source, **(extra or {})})

    ip = PROJECT / "reports" / "fate_iterative_character" / "report_ipface.json"
    if ip.exists():
        d = json.loads(ip.read_text(encoding="utf-8"))
        c = d.get("comparison_arcface_mean", {})
        if c.get("ip_face"):
            add("juggernaut-xl-v9", "character_identity_arcface", c["ip_face_triad"], "report_ipface.json",
                {"arm": "ip_face_triad", "note": "best measured identity channel"})
        if c.get("baseline"):
            add("juggernaut-xl-v9", "character_identity_arcface_baseline", c["baseline"], "report_ipface.json")
    gate = PROJECT / "reports" / "valhalla_native" / "report_gate_scale.json"
    if gate.exists():
        d = json.loads(gate.read_text(encoding="utf-8"))
        best = max(((v["mean_cos"], k) for k, v in d["arms"].items()), default=None)
        if best:
            add("juggernaut-xl-v9", "character_identity_gate_tuned", best[0], "report_gate_scale.json",
                {"arm": best[1], "note": "mid+up gate, total strength 0.30"})
    sf = PROJECT / "reports" / "single_film" / "report_limits.json"
    if sf.exists():
        d = json.loads(sf.read_text(encoding="utf-8"))
        for t in d.get("takes", []):
            if isinstance(t.get("arcface_mean_to_first"), (int, float)):
                add("CogVideoX-2b", "single_take_identity", t["arcface_mean_to_first"], "report_limits.json",
                    {"frames": t["frames"]})
    ex = PROJECT / "reports" / "extreme" / "report.json"
    if ex.exists():
        d = json.loads(ex.read_text(encoding="utf-8"))
        add("juggernaut-xl-v9", "long_chain_identity_24seg", d["summary"]["mean_cos"], "extreme/report.json",
            {"min": d["summary"]["min_cos"], "anchor_fade": "none measured"})
    fi = PROJECT / "reports" / "fate_interface" / "report.json"
    if fi.exists():
        d = json.loads(fi.read_text(encoding="utf-8"))
        s = d.get("summary", {})
        if s.get("select_mean"):
            add("juggernaut-xl-v9", "character_identity_fate_native", s["select_mean"],
                "fate_interface/report.json",
                {"arm": "FateInterface supply+select best-of-4, no plugin/patch",
                 "min": s.get("select_min"), "pass_rate": s.get("select_pass_rate")})
    lc = PROJECT / "reports" / "fate_long_chain" / "report.json"
    if lc.exists():
        d = json.loads(lc.read_text(encoding="utf-8"))
        s = d.get("summary", {})
        if s.get("mean_cos"):
            add("juggernaut-xl-v9", "long_chain_identity_24seg_no_plugin", s["mean_cos"],
                "fate_long_chain/report.json",
                {"arm": "FateInterface supply_desc + best-of-4 select, no plugin/patch",
                 "min": s.get("min_cos"), "verdict": d.get("verdicts", {}).get("LONG_CHAIN_PASS")})
    dual = PROJECT / "reports" / "dual_character" / "report.json"
    if dual.exists():
        d = json.loads(dual.read_text(encoding="utf-8"))
        s = d.get("summary", {})
        if s.get("aria"):
            add("juggernaut-xl-v9", "dual_character_separation_margin", s["aria"]["margin"],
                "dual_character/report.json",
                {"arm": "two detailed-supply characters, alternating scenes",
                 "sofia_margin": s.get("sofia", {}).get("margin")})
    fd = PROJECT / "reports" / "flux_detailed" / "report.json"
    if fd.exists():
        d = json.loads(fd.read_text(encoding="utf-8"))
        if isinstance(d.get("mean_cos"), (int, float)):
            add("flux1-schnell", "character_identity_detailed_supply", d["mean_cos"],
                "flux_detailed/report.json",
                {"arm": "detailed block (negative n/a at guidance 0); does NOT beat flux short 0.480"})
    vw = PROJECT / "reports" / "video_window_select" / "report.json"
    if vw.exists():
        d = json.loads(vw.read_text(encoding="utf-8"))
        if isinstance(d.get("selected_mean"), (int, float)):
            add("CogVideoX-2b", "cross_window_selected_identity", d["selected_mean"],
                "video_window_select/report.json",
                {"arm": "3-take selection vs W1 anchor; KNOWN GAP: pool centred ~0.25, "
                        "selection cannot rescue -- needs I2V pixel anchoring"})
    i2v = PROJECT / "reports" / "i2v_window" / "report.json"
    if i2v.exists():
        d = json.loads(i2v.read_text(encoding="utf-8"))
        if isinstance(d.get("selected_mean"), (int, float)):
            add("CogVideoX-5b-I2V", "cross_window_identity_i2v", d["selected_mean"],
                "i2v_window/report.json",
                {"arm": "W2 started from W1 anchor frame (pixel anchor), Fate measures/selects",
                 "note": "closes the T2V cross-window gap (0.258 -> 0.633)"})
    cd = PROJECT / "reports" / "context_diagnostic" / "report_L4.json"
    if cd.exists():
        d = json.loads(cd.read_text(encoding="utf-8"))
        if d.get("arms", {}).get("detailed_neg"):
            add("juggernaut-xl-v9", "character_identity_detailed_supply",
                d["arms"]["detailed_neg"]["mean_cos"], "context_diagnostic/report_L4.json",
                {"arm": "detailed pinned block + bible negatives, single draw"})
    return reg


ROUTES = {
    "identity_keyframes": ("juggernaut-xl-v9", "ip_face_triad via framework gate (mid+up, 0.30); "
                           "measured 0.62 ArcFace, 24-segment chain no fade"),
    "identity_no_plugin": ("juggernaut-xl-v9", "FateInterface: pinned detailed supply + "
                           "negative facts, best-of-4 select against the enrolled hero, "
                           "tau=0.5 abstention; measured 0.657 (min 0.603, 6/6 pass), "
                           "no plugin, no patch -- fictional characters only"),
    "designated_external_face": ("juggernaut-xl-v9", "a designated EXTERNAL face still needs "
                                 "the IP reference channel (selection cannot steer a foreign "
                                 "face into a text pool: measured 0.372, abstained); "
                                 "FateInterface.enroll_face anchors same-source heroes only"),
    "coherent_take": ("CogVideoX-2b", "single pass <=121f @512x288, triad total 0.015; identity 0.90+"),
    "quality_still": ("flux1-schnell", "largest local denoiser; patch prior 0.0166, group-offload load; "
                      "no face-reference channel locally"),
    "long_film": ("CogVideoX-2b + juggernaut-xl-v9", "windowed takes anchored by keyframes; "
                  "seamless motion chaining pending I2V weights (not local)"),
    "fast_preview": ("sdxl-turbo", "4-step previews before committing to a route"),
}


def route(task: str, registry: dict | None = None) -> dict:
    model, why = ROUTES.get(task, ("juggernaut-xl-v9", "default route"))
    scores = (registry or {}).get(model.split(" ")[0], [])
    return {"task": task, "model": model, "rationale": why, "measured_records": scores}


def operating_limits(model_name: str, catalog: list[dict]) -> dict:
    for m in catalog:
        if m["name"] == model_name:
            return m["kb"]
    return FAMILY_KB["unknown"]


# ---------------------------------------------------------------------------
# Loader (block-unload) consolidation.
# Measured on this machine, reports/adaptive_framework/report_unload.json:
# unloading is a SCHEDULING decision, not a default. Resident can be the
# slowest arm when it spills past physical VRAM; sequential is lightest
# but ~12x slower; cpu_offload was both lighter and fastest for CogVideoX.
# ---------------------------------------------------------------------------
LOADER_MEASUREMENTS = {
    "cogvideox": {  # 17f 512x288, 10 steps
        "resident": {"seconds": 193.2, "peak_gpu_gb": 22.47},
        "cpu_offload": {"seconds": 8.7, "peak_gpu_gb": 15.3},
        "sequential": {"seconds": 101.5, "peak_gpu_gb": 7.05},
    },
    "flux": {  # 1024x1024, 4 steps; resident does not fit (>24GB)
        "group_offload": {"seconds": 7.6, "peak_gpu_gb": 2.74},
    },
    "sdxl": {  # full resident is normal for SDXL on 24GB (measured all day)
        "resident": {"seconds": None, "peak_gpu_gb": None},
    },
}
LOADER_BY_PRIORITY = {
    "cogvideox": {"speed": "cpu_offload", "memory": "sequential", "balanced": "cpu_offload"},
    "flux": {"speed": "group_offload", "memory": "group_offload", "balanced": "group_offload"},
    "sdxl": {"speed": "resident", "memory": "resident", "balanced": "resident"},
}


def choose_loader(family: str, priority: str = "speed") -> str:
    """Pick a residency strategy for a model family.

    priority: 'speed' | 'memory' | 'balanced'. Unknown families fall back
    to cpu_offload, the measured-safe middle for oversized denoisers."""
    table = LOADER_BY_PRIORITY.get(family)
    if table is None:
        return "cpu_offload"
    return table.get(priority, table["balanced"])


def enable_gpu_accel() -> dict:
    """GPU acceleration switches (consolidated).

    Already on by construction: denoiser compute runs on CUDA (resident
    or block-streamed onload), VAE on CUDA, and diffusers' default SDPA
    attention uses fused flash/mem-efficient kernels. What was missing
    and is enabled here: TF32 for matmul/cudnn (fp32 segments like VAE
    and text encoders). flash-attn/xformers are NOT installed on this
    machine (checked 2026-11-15); SDPA is the accelerated backend
    until they are. torch.compile is deliberately not used: one-off
    generations never repay its warmup."""
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    return {"tf32_matmul": True, "tf32_cudnn": True, "cudnn_benchmark": True,
            "attention": "sdpa (fused kernels)", "flash_attn_installed": False}


def apply_loader(pipe, family: str, strategy: str | None = None, priority: str = "speed"):
    """Apply a residency strategy to a loaded diffusers pipeline.

    Strategies: resident | cpu_offload | sequential | group_offload.
    group_offload streams transformer blocks (FLUX path); sequential is
    diffusers' submodule-level offload (lightest, slowest). Returns the
    strategy actually applied."""
    enable_gpu_accel()
    strategy = strategy or choose_loader(family, priority)
    if strategy == "resident":
        pipe = pipe.to("cuda")
    elif strategy == "cpu_offload":
        pipe.enable_model_cpu_offload()
    elif strategy == "sequential":
        pipe.enable_sequential_cpu_offload()
    elif strategy == "group_offload":
        from diffusers.hooks import apply_group_offloading

        apply_group_offloading(pipe.transformer, onload_device="cuda", offload_device="cpu",
                               offload_type="block_level", num_blocks_per_group=1,
                               use_stream=True)
        pipe.vae.to("cuda")
    else:
        raise ValueError(f"unknown loader strategy {strategy}")
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)
    return strategy


def loader_record(family: str, strategy: str) -> dict | None:
    return (LOADER_MEASUREMENTS.get(family) or {}).get(strategy)
