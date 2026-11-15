#!/usr/bin/env python3
"""L1 diagnostic (no plugin, no generation): is the Valhalla context
encoding itself discriminative?

Exports Triad patches for several distinct identities plus three
re-phrasings of the SAME identity, then measures patch-space geometry:
  * inter-identity cosine  -- if different people map to ~same vector,
    the bottleneck is upstream (Fate encoding / export), and no
    injection tuning downstream can recover identity.
  * intra-identity cosine across phrasings -- if the same person in
    different words drifts as far as different people, the patch is
    binding surface form, not identity.
Also logs Fate internals (tiles/stemcells/signals/quad spread) so a
collapse, if present, can be located inside the export.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import character_bible as cb
import module_patch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "context_diagnostic"
PATCH_DIR = OUT / "patches"

IDENTITIES: dict[str, list[str]] = {
    "A_aria": cb.IDENTITY_A["lines"],
    "B_marco": [
        "Marco is a 45-year-old man with short grey hair and a grey beard.",
        "Marco has dark brown eyes and olive skin and a square jaw.",
        "Marco has a broad rectangular face with a crooked nose and stubble.",
        "Marco's hair is always grey, never chestnut and never blonde.",
        "Marco is the same single person in every scene and every segment.",
    ],
    "C_sofia": [
        "Sofia is a 28-year-old woman with long platinum blonde hair.",
        "Sofia has blue eyes and pale skin and full lips.",
        "Sofia has a round face with a small upturned nose and heavy makeup.",
        "Sofia's hair is always platinum blonde, never chestnut and never black.",
        "Sofia is the same single person in every scene and every segment.",
    ],
    "D_yuki": [
        "Yuki is a 68-year-old East Asian woman with a short silver-white bob.",
        "Yuki has dark brown eyes and light skin and thin lips.",
        "Yuki has a heart-shaped face with a flat nose and no makeup.",
        "Yuki's hair is always silver-white, never chestnut and never blonde.",
        "Yuki is the same single person in every scene and every segment.",
    ],
}

# Same person as A, one fact per line (Fate-learning lesson: bundled
# facts get suppressed by the dominant fact in the bundle).
A_DETAILED = [
    "Aria is a 28-year-old woman.",
    "Aria has shoulder-length hair.",
    "Aria's hair is chestnut brown.",
    "Aria has a small mole on her left cheek.",
    "Aria's eyes are hazel.",
    "Aria's skin is fair.",
    "Aria has a slim oval face.",
    "Aria has a straight nose.",
    "Aria wears natural makeup.",
    "Aria's hair is never blonde.",
    "Aria's hair is never black.",
    "Aria is the same single person in every scene.",
]

# Same person again, as multi-phrasing QA rows (the v5 recipe that
# made Fate grasping generalise).
A_QA = [
    "Question: What colour is Aria's hair? Answer: Aria's hair is chestnut brown.",
    "Question: How long is the hair of Aria? Answer: Aria's hair is shoulder-length.",
    "Question: Does Aria have a mole? Answer: Aria has a small mole on her left cheek.",
    "Question: What is the colour of Aria's eyes? Answer: Aria's eyes are hazel.",
    "Question: How old is Aria? Answer: Aria is a 28-year-old woman.",
    "Question: What does Aria's face look like? Answer: Aria has a slim oval face and a straight nose.",
    "Question: Is Aria blonde? Answer: Aria is never blonde; her hair is chestnut brown.",
    "Question: Who stays the same in every scene? Answer: Aria is the same single person in every scene.",
]

VARIANTS = {"A_detailed": A_DETAILED, "A_qa": A_QA}


def cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5 or 1e-9
    nb = sum(y * y for y in b) ** 0.5 or 1e-9
    return dot / (na * nb)


def main() -> None:
    PATCH_DIR.mkdir(parents=True, exist_ok=True)
    patches: dict[str, dict] = {}
    internals: dict[str, dict] = {}
    jobs = {**IDENTITIES, **VARIANTS}
    for name, lines in jobs.items():
        path = PATCH_DIR / f"{name}.json"
        if not path.exists():
            cb.export_identity_patch(path, lines)
        p = module_patch.load_export_patch(path)
        patches[name] = p
        internals[name] = {
            "dim": p["dim"], "tile_count": p["tile_count"],
            "stem_cell_count": p["stem_cell_count"],
            "tile_signatures": len(p.get("tile_signatures") or []),
            "stem_signatures": len(p.get("stem_signatures") or []),
            "tile_avg_activation": p.get("tile_avg_activation"),
            "signals_emitted": p.get("signals_emitted"),
            "quad_layer_scales": [round(x, 5) for x in p["quad_layer_scales"]],
            "quad_spread": round(max(p["quad_layer_scales"]) - min(p["quad_layer_scales"]), 6),
            "hub_prefs_head": [round(x, 4) for x in (p.get("hub_prefs") or [])[:6]],
            "vec_norm": round(sum(v * v for v in p["patch_vector"]) ** 0.5, 4),
            "vec_head": [round(v, 4) for v in p["patch_vector"][:8]],
        }
        print(name, internals[name])

    names = list(patches)
    matrix = {a: {b: round(cos(patches[a]["patch_vector"], patches[b]["patch_vector"]), 4)
                  for b in names} for a in names}
    ids = list(IDENTITIES)
    inter = [matrix[a][b] for i, a in enumerate(ids) for b in ids[i + 1:]]
    intra = [matrix["A_aria"][v] for v in VARIANTS]
    report = {
        "protocol": "context-diagnostic-L1-patch-geometry-v1",
        "internals": internals,
        "cosine_matrix": matrix,
        "inter_identity_cos": {"values": [round(x, 4) for x in inter],
                               "mean": round(sum(inter) / len(inter), 4), "min": round(min(inter), 4)},
        "intra_identity_cos": {"A_vs_detailed": matrix["A_aria"]["A_detailed"],
                               "A_vs_qa": matrix["A_aria"]["A_qa"]},
        "verdicts": {
            "inter_distinct": min(inter) < 0.5,
            "intra_stable": min(intra) > max(inter),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report_L1.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"cosine_matrix": matrix,
                      "inter_mean": report["inter_identity_cos"]["mean"],
                      "intra": report["intra_identity_cos"],
                      "verdicts": report["verdicts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
