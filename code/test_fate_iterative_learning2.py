#!/usr/bin/env python3
"""Iterative learning v3: pathways that actually re-form Fate structure.

v2 findings: error-driven teaching moves 0 -> 0.667 in two rounds then
freezes digit-for-digit; incubation_cycles=8 changes nothing on the
native path. Two remaining structural pathways:

  orchestrated  persistent session + fate_orchestrate/fate_warmup flags
                (the valhalla-server hybrid config keys our CFG lacked)
  reingest      NO persistent session: every round rebuilds the session
                from the cumulative corpus (identity + everything taught
                so far) in one ingestion, so Fate re-closes fragments
                over the whole accumulated memory each round -- the
                same pathway that moved the patch in the image tests.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bridge
from test_fate_iterative_learning import HOLDOUT_Q, PARAPHRASES, TRAIN_Q, probe
from test_fate_learning import DECOYS, IDENTITY, filler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "fate_learning"
MAX_ROUNDS = 8


def teach_for(res, taught):
    lines = []
    for key, _q in TRAIN_Q:
        if not res[key]["correct"]:
            wrong = res[key]["grasped"]
            if wrong:
                lines.append(f"Correction: the line '{wrong}' does not describe Aria the 28-year-old woman.")
            pool = PARAPHRASES[key]
            lines.append(pool[taught[key] % len(pool)])
            taught[key] += 1
    return lines


def run(arm: str, cfg_extra: dict, reingest: bool) -> dict:
    backup = dict(bridge.CFG)
    bridge.CFG.update(cfg_extra)
    try:
        cumulative = IDENTITY + DECOYS + filler(120)
        session, lines_now = None, list(cumulative)
        rounds, taught = [], {k: 0 for k, _ in TRAIN_Q}

        def fresh_session():
            r = bridge.valhalla(bridge.build_payload(TRAIN_Q[0][1], {}, None,
                                                     [{"text": t} for t in lines_now], None, bridge.CFG))
            return r.get("session")

        if not reingest:
            session = fresh_session()
        for rd in range(MAX_ROUNDS):
            if reingest:
                session = fresh_session()
            res, session = probe(session, lines_now, TRAIN_Q)
            acc = sum(1 for v in res.values() if v["correct"]) / len(res)
            tel_lines = len(lines_now)
            rounds.append({"round": rd, "accuracy": acc, "per_q": res, "corpus_lines": tel_lines})
            print(arm, rd, acc, {k: v["margin"] for k, v in res.items()}, "lines", tel_lines)
            if acc == 1.0 and rd >= 1:
                break
            teach = teach_for(res, taught)
            if not teach:
                break
            if reingest:
                lines_now.extend(teach)
            else:
                r = bridge.valhalla(bridge.build_payload("What is the fact about Aria?", {}, session, None,
                                                         [{"text": t} for t in teach], bridge.CFG))
                if isinstance(r.get("session"), dict):
                    session = r["session"]
                lines_now.extend(teach)
        if reingest:
            session = fresh_session()
        held, session = probe(session, lines_now, HOLDOUT_Q)
        held_acc = sum(1 for v in held.values() if v["correct"]) / len(held)
        print(arm, "holdout", held_acc)
        return {"rounds": rounds, "holdout": held, "holdout_accuracy": held_acc,
                "final_accuracy": rounds[-1]["accuracy"], "taught_counts": taught}
    finally:
        bridge.CFG.clear()
        bridge.CFG.update(backup)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"protocol": "val-video-boost-fate-iterative-learning-v3", "arms": {}}
    report["arms"]["orchestrated"] = run("orchestrated",
                                         {"fate_orchestrate": True, "fate_warmup_enabled": True,
                                          "incubation_cycles": 8}, reingest=False)
    report["arms"]["reingest"] = run("reingest", {}, reingest=True)
    report["verdicts"] = {arm: {"reaches_full_accuracy": v["final_accuracy"] == 1.0,
                                 "generalises_to_holdout": v["holdout_accuracy"] >= 2 / 3}
                          for arm, v in report["arms"].items()}
    (OUT / "report_iterative_v3.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["verdicts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
