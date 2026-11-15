#!/usr/bin/env python3
"""Fate learning ability, patch removed, plugin removed.

Nothing touches a denoiser here. The question is purely: does the Fate
session itself get better at grasping the true identity facts (against
same-name decoys) when it is repeatedly taught, compared with a session
that receives the same number of neutral lines?

Arms:
  learn    persistent session; after every probe round, the verified
           identity facts are appended again + one derived restatement
           (boost.iterate writeback, fate_qa_feedback on)
  control  persistent session; same number of neutral filler appends
  bulk     one-shot session built at the end with ALL of learn's taught
           lines present from the start (incremental vs bulk exposure)

Per probe we read the engine's own learning-relevant telemetry:
retrieval/grasped index correctness, fog margin (top-second), rank of
the first correct line in fate_grasp.candidate_indices, and
corpus_line_affinity_max (docs: 'affinity is feedback memory').
Target lines are tracked by TEXT, since appends duplicate lines.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bridge

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "fate_learning"
ROUNDS = 8

IDENTITY = [
    "Aria is a 28-year-old woman with shoulder-length chestnut brown hair.",
    "Aria has a small mole on her left cheek and hazel eyes and fair skin.",
    "Aria wears a cream sweater in the studio portrait.",
]
DECOYS = [
    "Another Aria is a blonde child who runs at the harbour.",
    "Aria Smith is a 60-year-old woman with short grey hair.",
    "Maria is a 28-year-old woman with chestnut brown hair and blue eyes.",
]
QUESTIONS = [
    ("hair", "What colour is the hair of Aria, the 28-year-old woman?", IDENTITY[0]),
    ("mole", "Where is the mole of Aria, the woman with chestnut hair?", IDENTITY[1]),
    ("age", "How old is Aria, the woman with a mole on her left cheek?", IDENTITY[0]),
]
DERIVED = "Verified fact: Aria the 28-year-old woman has chestnut brown hair and a mole on her left cheek."


def filler(n: int, start: int = 0) -> list[str]:
    people = ["Marco the ferryman", "Old Lena", "Priest Tomas", "Captain Ilya", "Sister Mara", "Doctor Venn"]
    places = ["harbour", "salt market", "north gate", "cinder yard", "old library", "ferry dock"]
    return [f"Filler note {i}: {people[i % 6]} waits at the {places[i % 6]}." for i in range(start, start + n)]


def probe(session, lines_now: list[str]) -> tuple[dict, dict | None]:
    """lines_now tracks session corpus texts in order (appends extend it)."""
    results = {}
    new_session = session
    for key, q, target_text in QUESTIONS:
        payload = bridge.build_payload(q, {}, new_session, None, None, bridge.CFG)
        r = bridge.valhalla(payload)
        if isinstance(r.get("session"), dict):
            new_session = r["session"]
        row = (r.get("rows") or [{}])[0]
        grasp = row.get("fate_grasp") or {}
        cands = grasp.get("candidate_indices") or []
        grasped = grasp.get("grasped_idx", row.get("retrieval_corpus_idx"))
        grasped_text = lines_now[grasped] if isinstance(grasped, int) and 0 <= grasped < len(lines_now) else None
        rank = next((i for i, idx in enumerate(cands)
                     if isinstance(idx, int) and 0 <= idx < len(lines_now) and lines_now[idx] == target_text), None)
        tel = r.get("fate_telemetry") or {}
        results[key] = {
            "correct": grasped_text == target_text,
            "grasped_idx": grasped, "grasped_text": grasped_text,
            "margin": round((row.get("fog_evidence") or {}).get("margin", 0.0), 4),
            "target_rank": rank,
            "affinity_max": tel.get("corpus_line_affinity_max"),
            "decision": (row.get("fog_decision") or {}).get("decision"),
        }
    return results, new_session


def summarise(rounds: list[dict]) -> dict:
    flat = [p for r in rounds for p in r.values()]
    return {"accuracy": round(sum(1 for p in flat if p["correct"]) / len(flat), 3),
            "mean_margin": round(sum(p["margin"] for p in flat) / len(flat), 4),
            "mean_target_rank": (round(sum(p["target_rank"] for p in flat if p["target_rank"] is not None)
                                       / len([p for p in flat if p["target_rank"] is not None]), 2)
                                 if any(p["target_rank"] is not None for p in flat) else None)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base_lines = IDENTITY + DECOYS + filler(120)
    report = {"protocol": "val-video-boost-fate-learning-v1", "rounds": ROUNDS,
              "base_lines": len(base_lines), "arms": {}}
    taught_all: list[str] = []
    for arm in ("learn", "control"):
        notes = [{"text": t} for t in base_lines]
        r = bridge.valhalla(bridge.build_payload(QUESTIONS[0][1], {}, None, notes, None, bridge.CFG))
        session = r.get("session")
        lines_now = list(base_lines)
        rounds = []
        for rd in range(ROUNDS):
            res, session = probe(session, lines_now)
            rounds.append(res)
            print(arm, rd, summarise([res]))
            # teaching turn
            if arm == "learn":
                teach = IDENTITY + [DERIVED]
            else:
                teach = filler(4, start=1000 + rd * 4)
            if arm == "learn":
                taught_all.extend(teach)
            payload = bridge.build_payload("What is the fact about Aria?", {}, session, None,
                                           [{"text": t} for t in teach], bridge.CFG)
            r = bridge.valhalla(payload)
            if isinstance(r.get("session"), dict):
                session = r["session"]
            lines_now.extend(teach)
        report["arms"][arm] = {"rounds": rounds, "first": summarise(rounds[:1]),
                               "last": summarise(rounds[-1:]), "all": summarise(rounds),
                               "affinity_trace": [max((p.get("affinity_max") or 0) for p in r.values()) for r in rounds]}
    # bulk: everything learn saw, present from the start, probed once
    bulk_lines = base_lines + taught_all
    notes = [{"text": t} for t in bulk_lines]
    r = bridge.valhalla(bridge.build_payload(QUESTIONS[0][1], {}, None, notes, None, bridge.CFG))
    res, _ = probe(r.get("session"), bulk_lines)
    report["arms"]["bulk"] = {"rounds": [res], "only": summarise([res])}
    print("bulk", report["arms"]["bulk"]["only"])
    report["verdicts"] = {
        "learn_improves_over_rounds": report["arms"]["learn"]["last"]["accuracy"] > report["arms"]["learn"]["first"]["accuracy"]
        or report["arms"]["learn"]["last"]["mean_margin"] > report["arms"]["learn"]["first"]["mean_margin"] * 1.2,
        "learn_beats_control_last": report["arms"]["learn"]["last"]["mean_margin"] > report["arms"]["control"]["last"]["mean_margin"],
        "incremental_beats_bulk": report["arms"]["learn"]["last"]["mean_margin"] > report["arms"]["bulk"]["only"]["mean_margin"],
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({a: {k: v for k, v in arm.items() if k in ("first", "last", "only")} for a, arm in report["arms"].items()}, ensure_ascii=False))
    print(json.dumps(report["verdicts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
