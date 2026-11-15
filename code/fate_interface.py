#!/usr/bin/env python3
"""Fate small interface -- the plugin's jobs, done Fate's own way.

No generation-side plugin, no patch, no reference conditioning. What
the IP-Adapter used to supply is re-supplied through Fate's native
mechanisms behind four small calls:

  supply(scene)   the bible's identity, as a verbatim byte-identical
                  detailed block + the bible's negative facts as the
                  negative prompt (measured 0.471 -> 0.610, and
                  paraphrasing the block costs -0.08, so it is pinned)
  enroll_face(p)  designate a face: its embedding becomes the anchor.
                  The plugin injected this into every generation; Fate
                  instead keeps it on the selection side.
  verify(img)     boost.verify gate: cosine to the anchor, tau, and an
                  honest abstention instead of a silent pass.
  select(pool)    boost.grasp over candidates: the anchor picks the
                  closest candidate; identity enters through choice,
                  not through conditioning.

FaceID (ArcFace) is only the ruler inside verify/select -- it never
touches the generation, same role YuNet played for boost.verify.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import character_bible as cb
from face_id import FaceID

# Single source of truth: the pinned identity block and the negative
# facts live in the bible (character_bible.IDENTITY_A). This module
# only re-exports them; never reword a copy here (L5: -0.083 if the
# block is paraphrased per segment).
IDENTITY_BLOCK = cb.IDENTITY_A["detailed_block"]
NEGATIVE = cb.IDENTITY_A["negative_facts"]
SCENES = {
    "studio": "close-up portrait, studio light, plain grey background, facing camera",
    "park": "standing outdoors in a sunlit park, natural daylight, waist-up, facing camera",
    "harbour": "at a quiet harbour at dusk, sea breeze, waist-up, facing camera",
}


class FateInterface:
    def __init__(self, fid: FaceID | None = None, tau: float = 0.5):
        self.fid = fid or FaceID()
        self.tau = tau
        self.anchor_emb = None
        self.anchor_path: Path | None = None

    # -- context supply ------------------------------------------------
    def supply(self, scene: str) -> dict:
        """Prompt for one segment: pinned identity block + scene slot.

        The fingerprint (sha256 of the identity block) is logged with
        every supply so a run can prove afterwards that the block was
        delivered byte-identically in every segment."""
        import hashlib

        fp = hashlib.sha256(IDENTITY_BLOCK.encode("utf-8")).hexdigest()[:12]
        return {"prompt": f"photorealistic photograph of {IDENTITY_BLOCK}, {SCENES[scene]}",
                "negative_prompt": NEGATIVE, "scene": scene,
                "identity_fingerprint": fp}

    # -- designated face -----------------------------------------------
    def enroll_face(self, image_path) -> bool:
        """Designate THE face. Returns False (abstains) if no face found."""
        emb = self.fid.embedding(Path(image_path))
        if emb is None:
            self.anchor_emb, self.anchor_path = None, None
            return False
        self.anchor_emb, self.anchor_path = emb, Path(image_path)
        return True

    def score(self, image_path, anchor_emb=None) -> float | None:
        emb = self.fid.embedding(Path(image_path))
        anchor = self.anchor_emb if anchor_emb is None else anchor_emb
        if emb is not None and anchor is not None:
            return round(float(self.fid.cosine(anchor, emb)), 4)
        return None

    def verify(self, image_path) -> dict:
        cos = self.score(image_path)
        return {"cos": cos, "passed": bool(cos is not None and cos >= self.tau),
                "abstained": cos is None or cos < self.tau, "tau": self.tau}

    def select(self, pool: list) -> dict:
        """Pick the candidate closest to the designated face.

        Every candidate is scored and logged (auditable); if the best
        is under tau the pick is still returned but flagged abstained,
        which in production triggers regeneration, never a fake pass.
        """
        scored = [{"file": str(p), "cos": self.score(p)} for p in pool]
        ok = [s for s in scored if s["cos"] is not None]
        best = max(ok, key=lambda s: s["cos"]) if ok else None
        return {"chosen": best["file"] if best else None,
                "chosen_cos": best["cos"] if best else None,
                "abstained": best is None or best["cos"] < self.tau,
                "scores": scored}
