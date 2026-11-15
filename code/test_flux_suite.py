#!/usr/bin/env python3
"""FLUX cross-seed consistency suite on the fast group-offload path."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import module_patch  # noqa: E402
from test_flux_fast import gen, load_fast  # noqa: E402
from test_full_quality import SIGN, pairwise  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "media_pipeline" / "flux"
PATCH_DIR = ROOT / "reports" / "media_pipeline" / "transformer_aligned" / "patches"
SEEDS = (7, 8, 9)


def main() -> None:
    triad = module_patch.load_export_patch(PATCH_DIR / "triad.json")
    report_path = OUT / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    chosen = report["chosen_strength"]
    pipe = load_fast()
    arms = {"baseline": [], "fixed_total": [], "adaptive": []}
    timings = {}
    for seed in SEEDS:
        for arm, patch, strength in (
            ("baseline", None, None),
            ("fixed_total", triad, None),
            ("adaptive", triad, chosen),
        ):
            path = OUT / f"cons_{arm}_s{seed}.png"
            timings[f"{arm}_s{seed}"] = gen(pipe, SIGN, path, seed, patch=patch, strength=strength)
            arms[arm].append(path)
            print(f"{arm} s{seed}: {timings[f'{arm}_s{seed}']}s")
    import torch

    del pipe
    torch.cuda.empty_cache()
    consistency = {arm: pairwise(paths) for arm, paths in arms.items()}
    consistency["ratio_fixed_over_baseline"] = round(consistency["fixed_total"] / max(consistency["baseline"], 1e-9), 3)
    consistency["ratio_adaptive_over_baseline"] = round(consistency["adaptive"] / max(consistency["baseline"], 1e-9), 3)
    report["consistency_suite"] = {"timings": timings, **consistency}
    report["verdicts"]["adaptive_consistency_gain"] = consistency["ratio_adaptive_over_baseline"] < 1.0
    report["verdicts"]["fixed_consistency_gain"] = consistency["ratio_fixed_over_baseline"] < 1.0
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(consistency, ensure_ascii=False))
    print(f"updated {report_path}")


if __name__ == "__main__":
    main()
