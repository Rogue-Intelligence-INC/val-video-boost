#!/usr/bin/env python3
"""Iterative learning v4 -- clean per-fact teaching (poison removed).

v2/v3 diagnosis from the grasp logs: the 'Correction:' lines quoted the
wrong line verbatim, so they carried MORE query keywords than the target
fact and Fate grasped the correction itself (recursively). Fate has no
negation semantics -- negative teaching in text form is poison, exactly
the BOOST discipline lesson. v4 therefore teaches positively and
narrowly:

  * one failed fact per teaching turn, appended ALONE (recency is the
    affinity lever Fate actually has), paraphrase aligned to the query;
  * no corrections, no bundles, no other facts in the same turn;
  * per-round candidate diagnostics: score/rank of each target fact in
    fate_grasp.candidate_scores, so sub-threshold movement is visible.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bridge
from test_fate_iterative_learning import ACCEPT, HOLDOUT_Q, TRAIN_Q
from test_fate_learning import DECOYS, IDENTITY, filler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "fate_learning"
MAX_ROUNDS = 10

TARGET_TEXT = {"hair": IDENTITY[0], "mole": IDENTITY[1], "age": IDENTITY[0]}
TEACH = {
    "hair": ["The hair of Aria is chestnut brown.",
             "Aria's hair colour is chestnut brown, shoulder-length."],
    "mole": ["The mole of Aria is on her left cheek.",
             "Aria's mole is on her left cheek."],
    "age": ["The age of Aria is 28 years.",
            "Aria is 28 years old, a young woman."],
}


def probe_diag(session, lines_now, questions):
    out, new_session = {}, session
    for key, q in questions:
        r = bridge.valhalla(bridge.build_payload(q, {}, new_session, None, None, bridge.CFG))
        if isinstance(r.get("session"), dict):
            new_session = r["session"]
        row = (r.get("rows") or [{}])[0]
        grasp = row.get("fate_grasp") or {}
        idx = grasp.get("grasped_idx", row.get("retrieval_corpus_idx"))
        text = lines_now[idx] if isinstance(idx, int) and 0 <= idx < len(lines_now) else None
        cands = grasp.get("candidate_indices") or []
        scores = grasp.get("candidate_scores") or []
        target_pos = [i for i, ci in enumerate(cands)
                      if isinstance(ci, int) and 0 <= ci < len(lines_now) and ACCEPT[key](lines_now[ci])]
        out[key] = {"correct": bool(text and ACCEPT[key](text)), "grasped": text,
                    "margin": round((row.get("fog_evidence") or {}).get("margin", 0.0), 4),
                    "first_acceptable_rank": target_pos[0] if target_pos else None,
                    "first_acceptable_score": scores[target_pos[0]] if target_pos else None,
                    "top_score": scores[0] if scores else None}
    return out, new_session


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base_lines = IDENTITY + DECOYS + filler(120)
    r = bridge.valhalla(bridge.build_payload(TRAIN_Q[0][1], {}, None,
                                             [{"text": t} for t in base_lines], None, bridge.CFG))
    session, lines_now = r.get("session"), list(base_lines)
    rounds, taught = [], {k: 0 for k, _ in TRAIN_Q}
    for rd in range(MAX_ROUNDS):
        res, session = probe_diag(session, lines_now, TRAIN_Q)
        acc = sum(1 for v in res.values() if v["correct"]) / len(res)
        rounds.append({"round": rd, "accuracy": acc, "per_q": res})
        print(rd, acc, {k: (v["correct"], v["first_acceptable_rank"]) for k, v in res.items()})
        if acc == 1.0 and rd >= 1:
            break
        for key, _q in TRAIN_Q:  # one fact per turn, alone, in fixed order
            if res[key]["correct"]:
                continue
            line = TEACH[key][taught[key] % len(TEACH[key])]
            taught[key] += 1
            r = bridge.valhalla(bridge.build_payload(f"Tell me about Aria: {key}", {}, session, None,
                                                     [{"text": line}], bridge.CFG))
            if isinstance(r.get("session"), dict):
                session = r["session"]
            lines_now.append(line)
            print("  taught", key, "->", line)
    held, session = probe_diag(session, lines_now, HOLDOUT_Q)
    held_acc = sum(1 for v in held.values() if v["correct"]) / len(held)
    print("holdout", held_acc, {k: v["correct"] for k, v in held.items()})
    report = {"protocol": "val-video-boost-fate-iterative-learning-v4",
              "rounds": rounds, "holdout": held, "holdout_accuracy": held_acc,
              "final_accuracy": rounds[-1]["accuracy"], "taught_counts": taught,
              "verdicts": {"reaches_full_accuracy": rounds[-1]["accuracy"] == 1.0,
                           "generalises_to_holdout": held_acc >= 2 / 3}}
    (OUT / "report_iterative_v4.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["verdicts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
