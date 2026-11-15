# Valhalla for Video: Character Consistency Without Plugins — Summary Report

**Date:** 2026-11-15 · **Machine:** Windows, RTX 4090 23 GB · **Project:** `val-video-boost` (proprietary)
**Models used:** Juggernaut XL, FLUX.1-schnell, CogVideoX-2b · **Identity metric:** ArcFace cosine (same-person threshold ≈ 0.5; different-person reference 0.15–0.36 depending on pairing)

## 1. Question

Valhalla's context machinery (an external, unbounded "character bible" managed by the Fate engine: grasp, conflict detection, verification, valid-time graph edges) was built and proven for LLMs. Can that whole stack be moved onto generative image/video models so that a character stays the same person across arbitrarily many segments — and which parts of it actually carry identity?

## 2. Headline result

A fictional character can now be kept consistent across scenes **with no generation-side plugin, no reference-image conditioning, and no weight patch**, using only three Valhalla-native mechanisms:

1. **Pinned supply** — the bible delivers a detailed identity description byte-identically in every segment, plus the bible's explicit negative facts as the negative prompt.
2. **Candidate generation** — 4 candidates per segment from a frozen diffusion model.
3. **Fate gate** — a designated hero face is enrolled once; `FateInterface.select()` keeps only the closest candidate, `verify()` abstains (flags for regeneration) below τ = 0.5 instead of passing silently.

Measured on Juggernaut XL (3 seeds × park/harbour, ArcFace to each chain's studio hero): **mean 0.657, minimum 0.603, 6/6 segments above threshold** — at or above the IP-Adapter plugin reference (0.581 uniform; 0.62 block-gated).

## 3. What the measurements showed, layer by layer

### 3.1 The patch channel does not carry facial identity (mechanism, not tuning)

- Exported Triad patch vectors for four very different people (28-year-old woman, 45-year-old bearded man, blonde woman, 68-year-old woman) have pairwise cosine **0.963–0.987**. The same person rephrased drifts as far (0.946): identity is only 10–17% of the vector; the rest is a content-independent template. The Fate session-fragment carrier collapses the same way (0.937).
- Crossover test (prompt fixed to person A, only the patch donor swapped): raw patches from A vs B are indistinguishable (identity index 0.288 vs 0.290). Removing the common mode (**centring**) recovers a small true signal (donor gap +0.095), but raising the dose destroys it (gap → −0.001 at 2×, with face quality monotonically falling), and on top of a strong text supply the patch adds exactly nothing (donor gap −0.011).
- Verdict: the 256-d additive patch injected into UNet blocks has no learned mapping to facial identity. It locks *attributes* (hair colour was corrected in video tests; within-take flicker fell 58% on CogVideoX) but not *the person*. This channel is retired from the identity route.

### 3.2 Identity consistency does not scale with model size

| Model | Short-prompt cross-scene ArcFace |
|---|---|
| Juggernaut XL (~6 B pipeline) | 0.471 |
| FLUX.1-schnell (12 B+) | 0.480 |

Several× more parameters bought +0.009. What model size buys is face *render quality* (CogVideoX-2b → Juggernaut was the real jump); past that threshold, consistency is governed by how much identity information is *in the conditioning each generation*, not by parameter count. One place size does matter: text-encoder budget (SDXL 77 tokens vs FLUX's T5 channel) caps how much of the bible can be supplied per segment.

### 3.3 The text channel is the lever Valhalla had not been using

The bible previously fed generation a 20-word fragment. A 2×2 factorial (short/detailed × with/without negatives), same protocol, short arm reproducing the historical 0.471 exactly:

| Supply arm | Cross-scene ArcFace |
|---|---|
| Short fragment | 0.471 |
| Short + negative prompt | 0.419 |
| Detailed identity block | 0.589 |
| Detailed + bible negative facts | **0.610** (min 0.544) |

Two rules came out of this:

- **Verbatim pinning is mandatory.** Same facts, reworded per segment: 0.589 → 0.506. The identity block must be byte-identical every segment; only the scene slot may change. Each supply now logs a SHA-256 fingerprint of the block so this is auditable after a run.
- **Honest discount:** detailed supply also raises different-person cosine (0.361 → 0.451), so the true identity *margin* only grows 0.110 → 0.138. Much of the gain is distribution narrowing toward a prototype face — acceptable for a single protagonist, a collision risk for multiple similar characters in one film.

### 3.4 Fate can learn iteratively — in its own substrate

In pure-text Fate sessions, repetitive teaching and human-style correction (including corrections that quote the error) all fail or self-pollute. What works is **error-driven question-coverage teaching**: one new question-phrasing variant per round, only for probes currently failed. Held-out question phrasings went 0 → 1.000 and stayed there. Fate binds question-form ↔ line; covering the neighbourhood of phrasings is its generalisation mechanism. For characters: identity facts should enter the bible as single-fact lines in multiple phrasings — bundled facts get suppressed by the dominant fact in the bundle (also visible in patch residuals).

### 3.5 Video findings (CogVideoX-2b)

- Within a single take, identity is already strong and needs no machinery: ArcFace to first frame 0.90 (49 f) / 0.89 (81 f) / 0.96+ (121 f, at 512×288; 121 frames is the measured ceiling, 161 collapses). Longer takes trade motion for stability.
- Across T2V windows, identity restarts from scratch (seam 0.221): prompt + patch cannot carry a face across windows. The no-plugin answer is the same gate as for images — generate several takes, keep the one whose first frame matches the enrolled anchor — not yet run end-to-end for video.
- Loading is a scheduling decision, measured: CPU offload 8.7 s / 15.3 GB beat full residency (193 s / 22.5 GB, VRAM spill) and sequential offload (101 s / 7.1 GB) on this machine; consolidated into `adaptive_framework.choose_loader/apply_loader`.

## 4. The consolidated stack (current, regression-tested)

```
character_bible.IDENTITY_A          single source of truth:
  detailed_block   (pinned, verbatim, fingerprinted)
  negative_facts   (bible's "never …" facts -> negative prompt)
        |
FateInterface                       code/fate_interface.py
  supply(scene)    -> prompt + negative + identity fingerprint
  enroll_face(img) -> designated-face anchor (selection side only)
  select(pool)     -> argmax candidate, all scores logged
  verify(img)      -> tau gate, honest abstention
        |
frozen diffusion model (Juggernaut XL today; routed by
adaptive_framework.route("identity_no_plugin"))
```

`code/test_fate_consolidation.py` pins seven properties (single source of truth; verbatim block + single fingerprint across scenes; select = argmax and passes verify; a *foreign* enrolled face must abstain; framework registry/route point at this stack; a fresh 2-candidate generation is selectable). All seven pass.

## 5. What still needs a plugin — stated plainly

- **A designated external face** (a real person's photo, someone else's artwork): text carries tens of bits and selects a sub-population, never one face. Selecting for a foreign face from an Aria-text pool reached only 0.372 and was correctly abstained by the gate. Reference conditioning (IP-Adapter / I2V) remains information-theoretically necessary there; the router keeps a `designated_external_face` route for it, with this measurement as the stated reason.
- **Multiple similar protagonists** in one production: identity margin is only ~0.14, so face-level confusion is a real risk; per-character enrolled anchors mitigate but have not been stress-tested.
- Not yet validated: the 12-scene no-plugin long chain against the plugin's extreme-test benchmark (IP chain: 24/24 ≥ 0.5, mean 0.5516), and the video-side take-selection loop.

## 6. Bottom line

Valhalla's contribution to generative consistency turned out not to be its exported weight patch (which carries attributes, not identity) but its context discipline: an external bible that can hold unlimited identity facts, deliver the relevant ones verbatim under a token budget, expose its own negative facts, designate and guard an anchor face, and turn a stochastic generator into a selective system via verification and abstention. With those mechanisms consolidated behind a four-call interface, plugins are unnecessary for fictional-character consistency at the scales measured so far — and the boundary where they become necessary again is measured, not guessed.

**Artifacts:** `docs/DESIGN.md` (full round-by-round adjudication, §§1–13), `reports/context_diagnostic/report_L1–L5*.json`, `reports/fate_interface/report.json` + `report_consolidation.json`, `code/fate_interface.py`, `code/test_context_{diagnostic,residual,session,crossover,dose,prompt,cross_control,stacking}.py`.

```powershell
cd C:\Ai\rogue-products\val-video-boost\code
C:\Ai\models\.venv-sd\Scripts\python.exe test_fate_consolidation.py   # 7/7 regression
C:\Ai\models\.venv-sd\Scripts\python.exe test_fate_interface.py       # simple Fate-only comparison
```
