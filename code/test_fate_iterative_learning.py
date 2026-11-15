#!/usr/bin/env python3
"""Iterative learning optimisation -- error-driven, not repetition.

Previous run plateaued at 0.667 because teaching was blind repetition
(no error signal, duplicates only). This loop is boost.iterate+verify
applied to Fate itself, patch-free:

  probe -> score per question -> for each FAILED question only, teach:
     1. a disambiguation line naming the wrongly grasped line as not-Aria
     2. a fresh paraphrase of the target fact, appended LAST (recency)
  rotate paraphrases so no exact line is ever taught twice.

Held-out paraphrase questions, never taught in that wording, are probed
at the end: memorisation would fail them, learning would not.
Arm B repeats arm A with incubation_cycles=8 (engine config lever,
default 2) to test whether more Fate cycles per turn restructure.
Correctness is semantic (predicate per question), since restatements
are new lines by design.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bridge
from test_fate_learning import DECOYS, IDENTITY, filler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "fate_learning"
MAX_ROUNDS = 8

ACCEPT = {
    "hair": lambda t: "chestnut" in t and "hair" in t and "Maria" not in t and "blonde" not in t,
    "mole": lambda t: "mole" in t and "left cheek" in t,
    "age": lambda t: ("28" in t or "twenty-eight" in t.lower()) and "Aria" in t and "60" not in t,
}
TRAIN_Q = [
    ("hair", "What colour is the hair of Aria, the 28-year-old woman?"),
    ("mole", "Where is the mole of Aria, the woman with chestnut hair?"),
    ("age", "How old is Aria, the woman with a mole on her left cheek?"),
]
HOLDOUT_Q = [
    ("hair", "The woman Aria who is twenty-eight: what is her hair colour?"),
    ("mole", "Which cheek carries Aria's mole?"),
    ("age", "Aria with hazel eyes is how many years old?"),
]
PARAPHRASES = {
    "hair": ["Aria, the 28-year-old woman, has chestnut brown hair that reaches her shoulders.",
             "The hair of Aria (age 28) is chestnut brown, never blonde and never grey.",
             "Chestnut brown is the hair colour of the 28-year-old Aria."],
    "mole": ["On Aria's left cheek there is a small mole.",
             "Aria's mole sits on her left cheek, beside her hazel eyes.",
             "The mole of Aria is located on the left side of her face, on the cheek."],
    "age": ["Aria is twenty-eight years old.",
            "The age of Aria, the woman with the left-cheek mole, is 28 years.",
            "Aria, who has hazel eyes and a mole, is 28 years of age."],
}


def probe(session, lines_now, questions):
    out, new_session = {}, session
    for key, q in questions:
        r = bridge.valhalla(bridge.build_payload(q, {}, new_session, None, None, bridge.CFG))
        if isinstance(r.get("session"), dict):
            new_session = r["session"]
        row = (r.get("rows") or [{}])[0]
        grasp = row.get("fate_grasp") or {}
        idx = grasp.get("grasped_idx", row.get("retrieval_corpus_idx"))
        text = lines_now[idx] if isinstance(idx, int) and 0 <= idx < len(lines_now) else None
        out[key] = {"correct": bool(text and ACCEPT[key](text)), "grasped": text,
                    "margin": round((row.get("fog_evidence") or {}).get("margin", 0.0), 4)}
    return out, new_session


def run_arm(arm: str, cfg_extra: dict) -> dict:
    cfg_backup = dict(bridge.CFG)
    bridge.CFG.update(cfg_extra)
    try:
        base_lines = IDENTITY + DECOYS + filler(120)
        r = bridge.valhalla(bridge.build_payload(TRAIN_Q[0][1], {}, None,
                                                 [{"text": t} for t in base_lines], None, bridge.CFG))
        session, lines_now = r.get("session"), list(base_lines)
        rounds, taught = [], {k: 0 for k, _ in TRAIN_Q}
        for rd in range(MAX_ROUNDS):
            res, session = probe(session, lines_now, TRAIN_Q)
            acc = sum(1 for v in res.values() if v["correct"]) / len(res)
            rounds.append({"round": rd, "accuracy": acc, "per_q": res})
            print(arm, rd, acc, {k: v["margin"] for k, v in res.items()})
            if acc == 1.0 and rd >= 1:
                break
            teach: list[str] = []
            for key, _q in TRAIN_Q:
                if not res[key]["correct"]:
                    wrong = res[key]["grasped"]
                    if wrong:
                        teach.append(f"Correction: the line '{wrong}' does not describe Aria the 28-year-old woman.")
                    pool = PARAPHRASES[key]
                    teach.append(pool[taught[key] % len(pool)])
                    taught[key] += 1
            if teach:
                r = bridge.valhalla(bridge.build_payload("What is the fact about Aria?", {}, session, None,
                                                         [{"text": t} for t in teach], bridge.CFG))
                if isinstance(r.get("session"), dict):
                    session = r["session"]
                lines_now.extend(teach)
        held, session = probe(session, lines_now, HOLDOUT_Q)
        held_acc = sum(1 for v in held.values() if v["correct"]) / len(held)
        print(arm, "holdout", held_acc, {k: v["correct"] for k, v in held.items()})
        return {"rounds": rounds, "holdout": held, "holdout_accuracy": held_acc,
                "final_accuracy": rounds[-1]["accuracy"], "taught_counts": taught}
    finally:
        bridge.CFG.clear()
        bridge.CFG.update(cfg_backup)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"protocol": "val-video-boost-fate-iterative-learning-v2",
              "reference_naive_repetition": {"final_accuracy": 0.667, "note": "test_fate_learning.py learn arm"},
              "arms": {}}
    report["arms"]["targeted"] = run_arm("targeted", {})
    report["arms"]["targeted_incubation8"] = run_arm("targeted_incubation8", {"incubation_cycles": 8})
    report["verdicts"] = {
        arm: {"reaches_full_accuracy": v["final_accuracy"] == 1.0,
              "generalises_to_holdout": v["holdout_accuracy"] >= 2 / 3}
        for arm, v in report["arms"].items()}
    (OUT / "report_iterative.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["verdicts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
