#!/usr/bin/env python3
"""L1c: the OTHER carrier -- Fate session fragment vectors (bridge),
geometry-checked the same way as the export patches in L1.

If session fragments separate identities better than the Triad export
(inter-cos well below 0.97), the improvement point is carrier choice.
Also compares each session vector against its export counterpart.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import module_patch
from test_context_diagnostic import IDENTITIES, VARIANTS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "context_diagnostic"
PATCH_DIR = OUT / "patches"


def cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5 or 1e-9
    nb = sum(y * y for y in b) ** 0.5 or 1e-9
    return dot / (na * nb)


def main() -> None:
    jobs = {**IDENTITIES, **VARIANTS}
    session_vecs: dict[str, list[float]] = {}
    for name, lines in jobs.items():
        p = module_patch.fragment_patch(lines)
        session_vecs[name] = [float(x) for x in p["patch_vector"]]
        print(name, "fragments", p["fragments"], "dim", p["dim"])
    names = list(jobs)
    matrix = {a: {b: round(cos(session_vecs[a], session_vecs[b]), 4) for b in names} for a in names}
    ids = list(IDENTITIES)
    inter = [matrix[a][b] for i, a in enumerate(ids) for b in ids[i + 1:]]
    export_vecs = {n: module_patch.load_export_patch(PATCH_DIR / f"{n}.json")["patch_vector"] for n in names}
    vs_export = {n: round(cos(session_vecs[n], export_vecs[n]), 4) for n in names}
    report = {"protocol": "context-diagnostic-L1c-session-geometry-v1",
              "cosine_matrix": matrix,
              "inter_identity_mean": round(sum(inter) / len(inter), 4),
              "intra_A": {"vs_detailed": matrix["A_aria"]["A_detailed"], "vs_qa": matrix["A_aria"]["A_qa"]},
              "session_vs_export_same_input": vs_export,
              "verdicts": {"session_more_discriminative_than_export": (sum(inter) / len(inter)) < 0.90}}
    (OUT / "report_L1c.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
