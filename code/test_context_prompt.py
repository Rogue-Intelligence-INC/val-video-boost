#!/usr/bin/env python3
"""L4 prompt-channel (no patch, no plugin): how much identity can the
Valhalla bible deliver through the diffusion model's OWN learned
text->face channel, if the context supply is richer?

2x2 factorial on Juggernaut XL, seeds 7/8/9, studio->park->harbour:
  identity wording : short (the historical 20-word fragment)
                     vs detailed (one-fact-per-line bible facts merged
                     into a specific stable face description)
  negative prompt  : none vs the bible's explicit negative facts
                     ("never blonde, never black", wrong-person terms)
Metric: ArcFace cos of park/harbour to the same arm's studio hero.
Historical reference: short/no-neg baseline measured 0.471.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import download
from face_id import FaceID
from test_big_models import load_pipe
from test_fate_iterative_character import SCENES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "context_diagnostic"
SIZE, STEPS, SEEDS = 768, 25, (7, 8, 9)

SHORT = ("the same 28-year-old woman Aria with shoulder-length chestnut brown hair, "
         "a small mole on her left cheek, hazel eyes, fair skin")
DETAILED = ("the same single woman Aria, 28 years old, shoulder-length chestnut brown "
            "hair with a center parting, small mole on her left cheek, hazel eyes, "
            "fair skin, slim oval face, high cheekbones, straight nose, dark brown "
            "eyebrows, natural pink lips, small chin")
NEGATIVE = ("blonde hair, black hair, grey hair, different person, another woman, "
            "changing face, beard, masculine features")

ARMS = {
    "short": (SHORT, None),
    "short_neg": (SHORT, NEGATIVE),
    "detailed": (DETAILED, None),
    "detailed_neg": (DETAILED, NEGATIVE),
}


def gen(pipe, frag, scene, negative, out_path, seed):
    import torch
    if out_path.exists():
        return 0.0
    started = time.time()
    kw = dict(prompt=f"photorealistic photograph of {frag}, {SCENES[scene]}",
              num_inference_steps=STEPS, guidance_scale=7.0,
              generator=torch.Generator(device="cuda").manual_seed(seed),
              height=SIZE, width=SIZE)
    if negative:
        kw["negative_prompt"] = negative
    img = pipe(**kw).images[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return round(time.time() - started, 2)


def main() -> None:
    pipe = load_pipe(download.IMAGE_JUG_DIR)
    fid = FaceID()
    report: dict = {"protocol": "context-diagnostic-L4-prompt-channel-v1", "arms": {}}
    try:
        for arm, (frag, neg) in ARMS.items():
            rows = []
            for seed in SEEDS:
                hero = OUT / "prompt_channel" / arm / f"s{seed}_studio.png"
                gen(pipe, frag, "studio", neg, hero, seed)
                hero_emb = fid.embedding(hero)
                for scene in ("park", "harbour"):
                    target = OUT / "prompt_channel" / arm / f"s{seed}_{scene}.png"
                    secs = gen(pipe, frag, scene, neg, target, seed)
                    emb = fid.embedding(target)
                    cos = fid.cosine(hero_emb, emb) if (emb is not None and hero_emb is not None) else None
                    rows.append({"seed": seed, "scene": scene, "seconds": secs,
                                 "arcface_cos": None if cos is None else round(cos, 4)})
                    print(arm, seed, scene, rows[-1]["arcface_cos"])
            ok = [r["arcface_cos"] for r in rows if r["arcface_cos"] is not None]
            report["arms"][arm] = {"rows": rows, "mean_cos": round(sum(ok) / len(ok), 4) if ok else None,
                                   "min_cos": min(ok) if ok else None}
    finally:
        import torch
        del pipe
        torch.cuda.empty_cache()
    report["reference"] = {"historical_short_baseline": 0.471, "different_person": 0.1522}
    (OUT / "report_L4.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({a: v["mean_cos"] for a, v in report["arms"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
