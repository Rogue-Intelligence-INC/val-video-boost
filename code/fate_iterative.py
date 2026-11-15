#!/usr/bin/env python3
"""Fate memory iteration for cross-segment identity -- persistent activation.

Mechanism (all real Valhalla parts, no invented state):

  * One valhalla_base Fate session per character chain, NEVER reset across
    segments. Segment 0 ingests only the identity lines and closes its
    fragments; those fragments (patch_vector, 256-d) stay in the session
    for the whole chain -- the always-activated part. Probe 2026-11-15:
    appends grow corpus 3->4->5 lines while closed identity fragments
    persist unchanged.
  * boost.iterate: after each segment is generated AND face-verified, a
    derived fact ("Segment i accepted: same Aria, chestnut hair, <scene>")
    is appended to the same session (append_items) and a follow-up turn
    runs. Unverified segments write NOTHING back -- BOOST discipline:
    unverified derived facts are poison (gap analysis sec.7, lesson 1).
  * Generation patch per segment = exported Triad identity body (tile/stem
    signatures, quad scales) whose patch_vector is tilted toward the
    session's cumulative fragment mean:
        v_i = normalize(0.65 * v_triad + 0.35 * mean_fragments_so_far)
    The Triad body is the always-on geometry; the session mean is the
    part that iterates. Cosine(v_i, v_anchor) is logged every segment --
    if the always-activated part works, the patch direction must not
    wander as scenes change.
  * Ablation arm `reset`: identical recipe, session rebuilt per segment.
    Any C-vs-D difference is the persistence effect, nothing else.
"""
from __future__ import annotations

import bridge
import module_patch


def _mean(vectors: list[list[float]]) -> list[float]:
    w = len(vectors[0])
    return [sum(v[i] for v in vectors) / len(vectors) for i in range(w)]


def _norm(v: list[float]) -> list[float]:
    n = sum(x * x for x in v) ** 0.5 or 1e-9
    return [x / n for x in v]


def cosine(a: list[float], b: list[float]) -> float:
    return round(sum(x * y for x, y in zip(_norm(a), _norm(b))), 4)


class FateChain:
    def __init__(self, identity_lines: list[str], question: str = "What is the fact?"):
        self.identity_lines = identity_lines
        self.question = question
        self.session: dict | None = None
        self.fragments_seen: dict[str, list[float]] = {}
        self.anchor_vector: list[float] | None = None
        self.log: list[dict] = []

    def _absorb(self, tag: str) -> dict:
        frags = [f for f in (self.session.get("fragments") or []) if isinstance(f, dict)]
        new = 0
        for f in frags:
            key = f"{f.get('id')}|{f.get('label', '')[:40]}"
            if f.get("patch_vector") and key not in self.fragments_seen:
                self.fragments_seen[key] = [float(x) for x in f["patch_vector"]]
                new += 1
        entry = {"tag": tag, "corpus_lines": len(self.session.get("corpus") or []),
                 "lines_ingested": self.session.get("lines_ingested"),
                 "fragments_total": len(frags), "fragments_new": new,
                 "fragments_seen": len(self.fragments_seen)}
        self.log.append(entry)
        return entry

    def _turn(self, append: list[str] | None, question: str | None = None):
        q = question or self.question
        app = [{"text": t} for t in append] if append else None
        r = bridge.valhalla(bridge.build_payload(q, {}, self.session, None, app, bridge.CFG))
        if isinstance(r.get("session"), dict):
            self.session = r["session"]

    def start(self) -> list[float]:
        notes = [{"text": t} for t in self.identity_lines]
        r = bridge.valhalla(bridge.build_payload(self.question, {}, None, notes, None, bridge.CFG))
        self.session = r["session"]
        self._turn(None)  # second turn closes fragments carrying patch_vector
        self._absorb("start_identity")
        self.anchor_vector = _norm(_mean(list(self.fragments_seen.values())))
        return self.anchor_vector

    def advance(self, scene_line: str, derived_fact: str | None, tag: str) -> list[float]:
        """Append scene fact (+ verified derived fact from last segment),
        follow-up turn, return cumulative session mean vector."""
        appends = [scene_line] + ([derived_fact] if derived_fact else [])
        self._turn(appends, question=f"Describe Aria: {scene_line}")
        self._turn(None, question=f"Describe Aria: {scene_line}")
        self._absorb(tag)
        return _norm(_mean(list(self.fragments_seen.values())))

    def fused_patch(self, triad: dict, session_vector: list[float], w_session: float = 0.35) -> dict:
        mixed = _norm([(1 - w_session) * a + w_session * b
                       for a, b in zip(triad["patch_vector"], session_vector)])
        patch = dict(triad)
        patch["patch_vector"] = mixed
        patch["body_source"] = "triad+fate_session"
        return patch


def static_session_patch(identity_lines: list[str], triad: dict) -> dict:
    """One-shot session patch (val-videopic-fusion fragment_patch recipe)."""
    patch = module_patch.fragment_patch(identity_lines)
    fused = dict(triad)
    fused["patch_vector"] = _norm([0.65 * a + 0.35 * b for a, b in zip(triad["patch_vector"], patch["patch_vector"])])
    fused["quad_layer_scales"] = patch.get("quad_layer_scales", triad["quad_layer_scales"])
    fused["body_source"] = "triad+oneshot_session"
    return fused
