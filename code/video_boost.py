#!/usr/bin/env python3
"""Valhalla full suite on a video model -- BOOST operators, video edition.

Mapping (VAL-LLM-BOOST-FRAMEWORK gap analysis -> this file):

  boost.iterate    -> generate segment -> encode its frames back into the
                      Fate session (bridge, pooled frame features) -> the
                      NEXT segment is planned from what was actually made,
                      with derived facts (accepted face, drift score)
                      written back to the run ledger, not silently dropped.
  boost.verify     -> Fate gate for pictures: YuNet face must be found and
                      its crop must stay within `tau` of the accepted hero
                      face. Failure is NOT silently shipped: strength
                      ladder retry (boost.adapt probe grid), then explicit
                      abstention for that segment (zero-fabrication rule).
  boost.aggregate  -> segment/shot counting is done by the planner in
                      code, never by asking a model to count frames.
  boost.conflict   -> Bible.conflicts() runs before any generation.
  boost.adapt      -> the probe ladder [0.5,1,2,4]x anchor strength is the
                      only adaptation; per-character chosen strength is
                      logged so a later offline bake can consume it.

Identity lock stack (strongest available on this machine, in order):
  1. pinned Triad identity patch, mode=total, on every CogVideoXBlock of
     every segment (module_patch.ModulePatch, copied from val-videopic-fusion)
  2. identical identity prompt fragment + fixed generator seed per character
  3. verifier gate + ladder retry against the accepted hero face
Known ceiling (measured on images, reports/prior_character_image_report.json):
  the patch alone locks ATTRIBUTES (hair/age/colouring, face L2 50.8->44.6,
  identity margin 6.5->10.3) but cross-scene pixel identity does not hold
  (cross_scene_holds=false, "two different faces"). True person-level lock
  needs a reference-image conditioner (CogVideoX-I2V / IP-Adapter-Face),
  which is not among the local weights; the verifier is what stops that
  gap from being silently presented as success here.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import settings

import module_patch


class FaceVerifier:
    """YuNet detection + face-crop L2 / colour-histogram distance."""

    def __init__(self, yunet_path: Path | None = None):
        import cv2

        self._cv2 = cv2
        yunet = yunet_path or settings.YUNET_MODEL
        if not yunet.exists():
            raise FileNotFoundError(
                f"YuNet model not found at {yunet}\n"
                f"Set VAL_YUNET_MODEL to the correct path."
            )
        self.detector = cv2.FaceDetectorYN.create(str(yunet), "", (720, 480), 0.6, 0.3, 50)

    def face_crop(self, image_path: Path, size: int = 128):
        import numpy as np
        from PIL import Image

        img = self._cv2.imread(str(image_path))
        if img is None:
            return None
        h, w = img.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(img)
        if faces is None or len(faces) == 0:
            return None
        x, y, fw, fh = (int(v) for v in faces[0][:4])
        x, y = max(x, 0), max(y, 0)
        crop = img[y:y + fh, x:x + fw]
        if crop.size == 0:
            return None
        rgb = self._cv2.cvtColor(crop, self._cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((size, size))
        return pil

    @staticmethod
    def distance(crop_a, crop_b) -> float:
        total, count = 0.0, 0
        for a, b in zip(crop_a.getdata(), crop_b.getdata()):
            for c in range(3):
                d = a[c] - b[c]
                total += d * d
                count += 1
        return (total / max(count, 1)) ** 0.5


def clip_metrics(frames: list[Path]) -> dict:
    import numpy as np
    from PIL import Image

    means, stds, adj = [], [], []
    prev = None
    for f in frames:
        arr = np.asarray(Image.open(f).convert("L")).astype(float)
        means.append(float(arr.mean())); stds.append(float(arr.std()))
        if prev is not None:
            adj.append(float(np.abs(arr - prev).mean()))
        prev = arr
    return {"frames": len(frames), "blank": sum(1 for s in stds if s < 8),
            "flicker_std": round(float(np.std(means)), 3),
            "adjacent_mean_abs": round(sum(adj) / max(len(adj), 1), 3)}


def generate_segment(pipe, prompt: str, out_dir: Path, seed: int,
                     patch: dict | None, mode: str = "total",
                     strength: float | None = None, steps: int = 25,
                     num_frames: int = 17, height: int = 480, width: int = 720) -> tuple[list[Path], float, dict | None]:
    """One segment. CogVideoX-2b needs ~25 steps; 2-6 steps = blank latents
    (measured in val-videopic-fusion test_video_quality.py)."""
    import torch

    handle = None
    started = time.time()
    try:
        if patch is not None:
            handle = module_patch.ModulePatch(patch, mode=mode, strength=strength)
            handle.attach(pipe.transformer, "transformer")
        gen = torch.Generator(device="cpu").manual_seed(seed)
        result = pipe(prompt=prompt, num_inference_steps=steps, num_frames=num_frames,
                      height=height, width=width, guidance_scale=6.0, generator=gen)
        out_dir.mkdir(parents=True, exist_ok=True)
        saved = []
        for i, frame in enumerate(result.frames[0]):
            p = out_dir / f"frame_{i:02d}.png"
            frame.save(p) if hasattr(frame, "save") else _save_array(frame, p)
            saved.append(p)
        info = {"fires": handle.fires, "sites": len(handle.modules)} if handle else None
        return saved, round(time.time() - started, 2), info
    finally:
        if handle is not None:
            handle.remove()


def _save_array(frame, path: Path):
    import numpy as np
    from PIL import Image

    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr * (255.0 if float(arr.max()) <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def verified_segment(pipe, prompt, out_dir: Path, seed: int, patch, verifier: FaceVerifier,
                     hero_crop, tau: float = 52.0, ladder=(1.0, 0.5), **kw):
    """boost.verify + boost.adapt: anchor strength, then DOWN the ladder.
    Measured 2026-11-15: doubling total strength (0.015->0.03) corrupts
    CogVideoX face segments (stripe artefacts), so retries go weaker, and
    the BEST try (lowest face distance) is the ledger entry -- never the
    last, corrupted one. status=abstained means no face passed the gate:
    the segment is reported, not replaced by a hallucinated pass."""
    base = module_patch.MODE_STRENGTH["total"]
    best = None
    for factor in ladder:
        frames, secs, info = generate_segment(
            pipe, prompt, out_dir / f"try{factor}", seed, patch,
            strength=base * factor, **kw)
        mid = frames[len(frames) // 2]
        crop = verifier.face_crop(mid)
        dist = None if (crop is None or hero_crop is None) else verifier.distance(hero_crop, crop)
        ok = crop is not None and (hero_crop is None or dist <= tau)
        entry = {"dir": str(out_dir / f"try{factor}"), "seconds": secs, "patch": info,
                 "mid_frame": str(mid), "face_found": crop is not None,
                 "face_l2_to_hero": None if dist is None else round(dist, 3),
                 "strength": base * factor, "metrics": clip_metrics(frames)}
        if best is None or (entry["face_l2_to_hero"] or 1e9) < (best["face_l2_to_hero"] or 1e9):
            best = entry
            best["_crop"] = crop
        if ok:
            entry["status"] = "accepted"
            entry["accepted_crop"] = crop
            return entry
    best["status"] = "abstained"
    crop = best.pop("_crop", None)
    if crop is not None:
        best["accepted_crop"] = crop
    return best
