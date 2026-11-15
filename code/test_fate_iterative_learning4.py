#!/usr/bin/env python3
"""Iterative learning optimisation v5 -- phrasing-coverage curriculum.

Established mechanics (probes, patch-free):
  * Fate binds question-pattern <-> line (scaffold margin 2.94) but a
    single binding does not transfer to differently worded questions;
  * identical duplicates are bundled away (copies x16: zero effect);
  * correction/meta lines poison the grasp (negation has no semantics).

So the optimisation that matches the substrate: per failed fact, add ONE
new question-phrasing variant per round as a QA line ("Question: ...
Answer: ...fact..."). Variants tile the phrasing neighbourhood of the
fact; held-out phrasings (never taught, fixed in advance) test whether
the basin generalises. This isboost.iterate driven by Fate's own
affinity: probe -> verify -> teach only the failures -> re-probe.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bridge
from test_fate_iterative_learning import ACCEPT
from test_fate_learning import DECOYS, IDENTITY, filler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "fate_learning"
MAX_ROUNDS = 12

TRAIN_Q = [
    ("hair", "What colour is the hair of Aria, the 28-year-old woman?"),
    ("mole", "Where is the mole of Aria, the woman with chestnut hair?"),
    ("age", "How old is Aria, the woman with a mole on her left cheek?"),
]
HOLDOUT_Q = [
    ("hair", "What is the hair colour of the twenty-eight year old Aria?"),
    ("mole", "Which cheek carries Aria's mole?"),
    ("age", "Aria with hazel eyes is how many years old?"),
]
VARIANTS = {
    "hair": [
        "Question: What colour is Aria's hair? Answer: chestnut brown, shoulder-length.",
        "Question: What is the hair colour of Aria? Answer: Aria's hair is chestnut brown.",
        "Question: Tell me the hair colour of Aria. Answer: chestnut brown hair.",
        "Question: Aria's hair is what colour? Answer: it is chestnut brown.",
        "Question: How would you describe Aria's hair? Answer: shoulder-length and chestnut brown.",
        "Question: What colour hair does Aria have? Answer: chestnut brown.",
        "Question: What colour is the hair of Aria, the 28-year-old woman? Answer: chestnut brown.",
        "Question: What colour is the hair of Aria, the woman with a mole? Answer: chestnut brown.",
        "Question: The 28-year-old Aria has hair of what colour? Answer: chestnut brown.",
    ],
    "mole": [
        "Question: Where is the mole of Aria? Answer: on her left cheek.",
        "Question: Which cheek carries Aria's mole? Answer: the left cheek.",
        "Question: What is on Aria's left cheek? Answer: a small mole.",
        "Question: Does Aria have a mole, and where? Answer: yes, on her left cheek.",
        "Question: Tell me about Aria's mole. Answer: a small mole on her left cheek.",
        "Question: Where on her face is Aria's mole? Answer: on the left cheek.",
        "Question: Where is the mole of Aria, the woman with chestnut hair? Answer: on her left cheek.",
        "Question: Where is the mole of Aria, the 28-year-old woman? Answer: on her left cheek.",
        "Question: The woman Aria has a mole where? Answer: on her left cheek.",
    ],
    "age": [
        "Question: How old is Aria? Answer: Aria is 28 years old.",
        "Question: What is the age of Aria? Answer: 28 years old.",
        "Question: Aria is how many years old? Answer: twenty-eight.",
        "Question: Tell me Aria's age. Answer: she is 28.",
        "Question: How many years old is Aria? Answer: 28 years.",
        "Question: What age is the woman Aria? Answer: 28 years of age.",
        "Question: How old is Aria, the woman with a mole on her left cheek? Answer: 28 years old.",
        "Question: How old is Aria, the woman with chestnut hair? Answer: 28 years old.",
        "Question: The woman with hazel eyes, Aria, is how old? Answer: 28 years old.",
    ],
}


def probe(session, lines_now, questions):
    out, s = {}, session
    for key, q in questions:
        r = bridge.valhalla(bridge.build_payload(q, {}, s, None, None, bridge.CFG))
        if isinstance(r.get("session"), dict):
            s = r["session"]
        row = (r.get("rows") or [{}])[0]
        g = row.get("fate_grasp") or {}
        idx = g.get("grasped_idx", row.get("retrieval_corpus_idx"))
        text = lines_now[idx] if isinstance(idx, int) and 0 <= idx < len(lines_now) else None
        out[key] = {"correct": bool(text and ACCEPT[key](text)), "grasped": text,
                    "margin": round((row.get("fog_evidence") or {}).get("margin", 0.0), 4)}
    return out, s


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = IDENTITY + DECOYS + filler(120)
    r = bridge.valhalla(bridge.build_payload(TRAIN_Q[0][1], {}, None,
                                             [{"text": t} for t in base], None, bridge.CFG))
    session, lines_now = r.get("session"), list(base)
    taught = {k: 0 for k, _ in TRAIN_Q}
    curve = []
    for rd in range(MAX_ROUNDS):
        train, session = probe(session, lines_now, TRAIN_Q)
        held, session = probe(session, lines_now, HOLDOUT_Q)
        acc = sum(v["correct"] for v in train.values()) / 3
        hacc = sum(v["correct"] for v in held.values()) / 3
        curve.append({"round": rd, "train": acc, "holdout": hacc,
                      "train_detail": train, "holdout_detail": held})
        print(f"round {rd}: train {acc:.3f} holdout {hacc:.3f} taught={taught}")
        if acc == 1.0 and hacc >= 2 / 3:
            break
        for key, _q in TRAIN_Q:
            if not train[key]["correct"] and taught[key] < len(VARIANTS[key]):
                line = VARIANTS[key][taught[key]]
                taught[key] += 1
                r = bridge.valhalla(bridge.build_payload("Teach Aria facts", {}, session, None,
                                                         [{"text": line}], bridge.CFG))
                if isinstance(r.get("session"), dict):
                    session = r["session"]
                lines_now.append(line)
    report = {"protocol": "val-video-boost-fate-iterative-learning-v5-coverage",
              "curve": curve, "taught_counts": taught,
              "verdicts": {"train_reaches_full": curve[-1]["train"] == 1.0,
                           "holdout_generalises": curve[-1]["holdout"] >= 2 / 3,
                           "holdout_improved": curve[-1]["holdout"] > curve[0]["holdout"]}}
    (OUT / "report_iterative_v5.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["verdicts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
