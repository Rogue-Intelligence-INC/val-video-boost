#!/usr/bin/env python3
"""L1b: after removing the common mode shared by all identity patches,
do the residuals point in a person-specific, phrasing-stable direction?

Reads the patches exported by test_context_diagnostic.py. Criteria:
  * residual share: ||x - mean|| / ||x|| tells how much of each patch
    is identity at all (vs the common template).
  * same-person residual alignment: cos(r_A, r_A_detailed / r_A_qa)
    must be clearly positive if the residual is identity, not wording.
  * cross-person residual alignment should be near/below that.
If residuals are phrasing-stable, a centred patch (x - corpus mean,
renormalised) is a candidate carrier; if not, identity is simply not
in the export and the fix must go further upstream.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import module_patch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "context_diagnostic"
PATCH_DIR = OUT / "patches"
IDS = ["A_aria", "B_marco", "C_sofia", "D_yuki"]
ALL = IDS + ["A_detailed", "A_qa"]


def cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5 or 1e-9
    nb = sum(y * y for y in b) ** 0.5 or 1e-9
    return dot / (na * nb)


def main() -> None:
    vecs = {n: module_patch.load_export_patch(PATCH_DIR / f"{n}.json")["patch_vector"] for n in ALL}
    dim = len(vecs["A_aria"])
    mean = [sum(vecs[n][i] for n in IDS) / len(IDS) for i in range(dim)]
    resid = {n: [v - m for v, m in zip(vecs[n], mean)] for n in ALL}
    share = {n: round((sum(r * r for r in resid[n]) ** 0.5) / (sum(v * v for v in vecs[n]) ** 0.5), 4)
             for n in ALL}
    rmatrix = {a: {b: round(cos(resid[a], resid[b]), 4) for b in ALL} for a in ALL}
    same = [rmatrix["A_aria"]["A_detailed"], rmatrix["A_aria"]["A_qa"]]
    cross = [rmatrix[a][b] for i, a in enumerate(IDS) for b in IDS[i + 1:]]
    # per-dimension spread: in how many dims does identity beat wording?
    import statistics
    var_inter = []
    for i in range(dim):
        vals = [vecs[n][i] for n in IDS]
        var_inter.append(statistics.pvariance(vals))
    top = sorted(range(dim), key=lambda i: -var_inter[i])[:16]
    report = {
        "protocol": "context-diagnostic-L1b-residual-v1",
        "residual_share_of_patch": share,
        "residual_cosine_matrix": rmatrix,
        "same_person_residual_cos": {"A_vs_detailed": same[0], "A_vs_qa": same[1]},
        "cross_person_residual_cos_mean": round(sum(cross) / len(cross), 4),
        "top_discriminative_dims": top,
        "top_dim_values": {str(i): {n: round(vecs[n][i], 4) for n in ALL} for i in top[:6]},
        "verdicts": {
            "residual_is_small": share["A_aria"] < 0.35,
            "residual_phrasing_stable": min(same) > 0.3 and min(same) > sum(cross) / len(cross),
        },
    }
    (OUT / "report_L1b.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
