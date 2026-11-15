#!/usr/bin/env python3
"""Real identity metric: ArcFace (buffalo_l / w600k_r50) cosine.

Replaces face-crop L2, which cannot tell 'same attributes' from 'same
person'. Same-person cosine for this model family is typically >= ~0.5;
the negative control in every report calibrates it empirically.
Also the Fate writeback encoder: an accepted segment's ArcFace embedding
is the first derived fact that actually contains face information.
"""
from __future__ import annotations

from pathlib import Path


class FaceID:
    def __init__(self):
        from insightface.app import FaceAnalysis

        self.app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        # 320 measured: 640 misses Juggernaut close-up portraits entirely,
        # 320 detects studio/park/harbour at det_score 0.81-0.87.
        self.app.prepare(ctx_id=-1, det_size=(320, 320))

    def embedding(self, image_path) -> list[float] | None:
        import cv2

        img = cv2.imread(str(image_path))
        if img is None:
            return None
        faces = self.app.get(img)
        if not faces:  # SCRFD anchor scale: close-up portraits need 320 (measured)
            for size in ((320, 320), (1024, 1024), (480, 480)):
                try:
                    self.app.det_model.input_size = size
                except Exception:
                    break
                faces = self.app.get(img)
                if faces:
                    break
        if not faces:
            return None
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return [float(x) for x in face.normed_embedding]

    @staticmethod
    def cosine(a: list[float], b: list[float]) -> float:
        return round(sum(x * y for x, y in zip(a, b)), 4)


def mean_embedding(embeddings: list[list[float]]) -> list[float]:
    w = len(embeddings[0])
    mean = [sum(e[i] for e in embeddings) / len(embeddings) for i in range(w)]
    n = sum(x * x for x in mean) ** 0.5 or 1e-9
    return [x / n for x in mean]
