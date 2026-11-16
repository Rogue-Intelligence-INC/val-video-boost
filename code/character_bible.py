#!/usr/bin/env python3
"""Ultra-long-context character bible -- the BOOST layer moved to video.

Why this exists: CogVideoX-2b's T5 encoder truncates at ~226 tokens. No
prompt, however long, can carry a novel-length story bible or lock a person
across a hundred generated segments. So the long context never enters the
diffusion model at all:

  * It lives in the Valhalla Fate session / bible file (BOOST hard evidence:
    14.95M-token scale, zero fabrication, auditable abstention -- see
    docs/boost/). Its size is unbounded for our purposes.
  * The PINNED identity core (name, face, hair, marks, age) is exported once
    via valhalla_model_export --body triad into a fixed 256-d patch vector.
    Constant size, never retrieved, never evicted, never diluted by filler:
    this is the answer to "does ultra-long context break consistency" --
    identity does not go through retrieval at all.
  * Per segment, boost.grasp does one Fate-style retrieval over the bible
    (lexical + graph traversal over (subject,predicate,object,valid_seg)
    edges) and compresses the result to a <=180-token CogVideoX prompt.
  * boost.conflict: two edges with same (subject,predicate), overlapping
    valid ranges, different objects => explicit conflict report, never a
    silent last-write-wins (BOOST gap-analysis root cause 7).

Graph edges carry a validity range in SEGMENT units (temporal channel,
BOOST root cause 7 / Zep-style valid-time), so an outfit or hair change
scripted at segment 40 applies from segment 40 -- and only there.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import settings

TOKEN_RE = re.compile(r"[a-z0-9]+")

# Legacy compatibility: IDENTITY_A is now loaded from characters/aria.json
# This ensures backward compatibility for code that imports IDENTITY_A
try:
    from character_loader import load_default_character, get_identity_a
    _default_char = load_default_character()
    IDENTITY_A = get_identity_a(_default_char)
except Exception:
    # Fallback if character files are not set up yet
    IDENTITY_A = {
        "name": "Aria",
        "lines": [
            "Aria is a 28-year-old woman with shoulder-length chestnut brown hair.",
            "Aria has a small mole on her left cheek and hazel eyes and fair skin.",
            "Aria has a slim oval face with a straight nose and natural makeup.",
            "Aria's hair is always chestnut brown, never blonde and never black.",
            "Aria is the same single person in every scene and every segment.",
        ],
        "prompt_fragment": (
            "the same 28-year-old woman Aria with shoulder-length chestnut brown hair, "
            "a small mole on her left cheek, hazel eyes, fair skin"
        ),
        "detailed_block": (
            "the same single woman Aria, 28 years old, shoulder-length chestnut brown "
            "hair with a center parting, small mole on her left cheek, hazel eyes, "
            "fair skin, slim oval face, high cheekbones, straight nose, dark brown "
            "eyebrows, natural pink lips, small chin"
        ),
        "negative_facts": (
            "blonde hair, black hair, grey hair, different person, another woman, "
            "changing face, beard, masculine features"
        ),
    }

IDENTITY_B_NEGATIVE = (
    "a 45-year-old man with a short grey beard, receding hairline, dark brown eyes, olive skin"
)


def tokens(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.lower()))


class Bible:
    """Line bible + typed graph edges. Lines are the auditable evidence;
    edges are (subject, predicate, object, seg_lo, seg_hi, line_no)."""

    def __init__(self, lines: list[str], edges: list[dict] | None = None):
        self.lines = lines
        self.edges = edges or []
        self._line_tokens = [tokens(l) for l in lines]

    @classmethod
    def build_demo(cls, n_filler: int = 20000, seed: int = 7) -> "Bible":
        """A deliberately hostile bible: identity buried in ~20k filler lines
        about other people, places and props (same-name decoys included)."""
        import random

        rng = random.Random(seed)
        people = ["Marco the ferryman", "Old Lena", "Priest Tomas", "Captain Ilya",
                  "Sister Mara", "Doctor Venn", "Aunt Rosa", "Broker Hale"]
        places = ["harbour", "salt market", "north gate", "cinder yard", "old library",
                  "ferry dock", "glasshouse", "stone bridge"]
        acts = ["counts the ledgers", "mends a net", "argues about tolls", "waits for rain",
                "sells dried plums", "repairs a lantern", "writes a letter", "feeds the gulls"]
        lines: list[str] = []
        edges: list[dict] = []
        # scripted, time-bounded facts about Aria (graph edges, valid-time)
        script = [
            ("Aria", "outfit", "cream sweater", 0, 9999),
            ("Aria", "location", "sunlit park", 1, 1),
            ("Aria", "action", "walking slowly toward camera", 1, 1),
            ("Aria", "location", "quiet harbour at dusk", 2, 2),
            ("Aria", "action", "turning her head to the sea breeze", 2, 2),
            ("Aria", "location", "grey studio", 0, 0),
            ("Aria", "action", "looking at the camera", 0, 0),
        ]
        for subj, pred, obj, lo, hi in script:
            line = f"{subj}'s {pred} is {obj} (segments {lo}-{hi})."
            edges.append({"subject": subj, "predicate": pred, "object": obj,
                          "seg_lo": lo, "seg_hi": hi, "line_no": len(lines)})
            lines.append(line)
        lines.extend(IDENTITY_A["lines"])
        for i in range(n_filler):
            p, pl, a = rng.choice(people), rng.choice(places), rng.choice(acts)
            lines.append(f"Filler {i}: {p} at the {pl} {a}.")
            if i % 997 == 0:  # same-name decoy, different person attribute
                lines.append(f"Filler {i}: another Aria, a blonde child, runs past the {pl}.")
        return cls(lines, edges)

    # -- boost.conflict -------------------------------------------------
    def conflicts(self) -> list[dict]:
        out = []
        for i, a in enumerate(self.edges):
            for b in self.edges[i + 1:]:
                if (a["subject"], a["predicate"]) == (b["subject"], b["predicate"]) \
                        and a["object"] != b["object"] \
                        and a["seg_lo"] <= b["seg_hi"] and b["seg_lo"] <= a["seg_hi"]:
                    out.append({"a": a, "b": b})
        return out

    # -- boost.grasp: one retrieval pass per segment --------------------
    def grasp(self, segment: int, query: str, k: int = 4) -> dict:
        """Graph edges valid at `segment` are authoritative and come first;
        remaining slots are filled by lexical affinity over all bible lines.
        Returns line numbers so every prompt token is auditable."""
        graph = [e for e in self.edges if e["seg_lo"] <= segment <= e["seg_hi"]]
        picked: list[int] = []
        for e in graph:
            if e["line_no"] not in picked:
                picked.append(e["line_no"])
        q = tokens(query)
        scored = sorted(
            ((len(q & lt), i) for i, lt in enumerate(self._line_tokens) if i not in picked),
            reverse=True,
        )
        for score, i in scored[: max(0, k - len(picked))]:
            if score > 0:
                picked.append(i)
        return {"segment": segment, "line_nos": picked,
                "graph_edges": graph, "lines": [self.lines[i] for i in picked]}

    def prompt_for_segment(self, segment: int, scene_query: str) -> dict:
        """Compress grasp + pinned identity fragment to a CogVideoX prompt.
        Hard cap ~180 whitespace tokens (T5 limit 226, leave headroom)."""
        g = self.grasp(segment, scene_query)
        scene_bits = [e["object"] for e in g["graph_edges"]
                      if e["predicate"] in ("location", "action", "outfit")]
        prompt = (f"photorealistic video of {IDENTITY_A['prompt_fragment']}, "
                  + ", ".join(scene_bits)
                  + ", steady camera, consistent face, same person throughout")
        words = prompt.split()
        if len(words) > 180:
            prompt = " ".join(words[:180])
        return {"prompt": prompt, "n_words": len(prompt.split()), "grasp": g}


def export_identity_patch(out_json: Path, lines: list[str] | None = None) -> Path:
    """Export the pinned identity core as a Triad patch (Fate Hub/Tile/
    StemCell fused by valhalla_model_export, consumed as-is)."""
    export_exe = settings.require_valhalla_export_exe()
    corpus = Path(out_json).with_suffix(".corpus.txt")
    corpus.write_text("\n".join(lines or IDENTITY_A["lines"]) + "\n", encoding="utf-8")
    proc = subprocess.run(
        [str(export_exe), "--body", "triad", "--mode", "normalized_f64",
         "--experiment", "--corpus-file", str(corpus)],
        capture_output=True, text=True, timeout=600,
    )
    text = (proc.stdout or "") + (proc.stderr or "")
    start = text.find("{")
    if start < 0:
        raise RuntimeError(f"valhalla_model_export produced no JSON:\n{text[-2000:]}")
    doc, _ = json.JSONDecoder().raw_decode(text[start:])
    out_json.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return out_json


def fuse_patches(identity: dict, scene: dict | None, identity_weight: float = 0.75) -> dict:
    """One ModulePatch patch dict: pinned identity dominates the 256-d
    vector; the per-segment scene patch only tilts it. Tile/stem signatures
    of the identity stay first (Fate order = depth order in ModulePatch)."""
    if scene is None:
        return identity
    iv, sv = identity["patch_vector"], scene["patch_vector"]
    mixed = [identity_weight * a + (1 - identity_weight) * b for a, b in zip(iv, sv)]
    norm = sum(v * v for v in mixed) ** 0.5 or 1e-9
    fused = dict(identity)
    fused["patch_vector"] = [v / norm for v in mixed]
    fused["body_source"] = "identity+scene"
    return fused


if __name__ == "__main__":
    bible = Bible.build_demo()
    est_tokens = sum(len(l.split()) for l in bible.lines)
    print(f"bible lines={len(bible.lines)} approx_words={est_tokens} conflicts={bible.conflicts()}")
    for seg in range(3):
        r = bible.prompt_for_segment(seg, "Aria walking harbour park studio")
        print(seg, r["n_words"], r["prompt"][:140], "| lines", r["grasp"]["line_nos"])
