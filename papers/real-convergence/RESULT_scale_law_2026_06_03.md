# RESULT — Does convergence rise with scale? NOT robustly (14M–3B). It's data-driven, not scale-driven.

**Date:** 2026-06-03 · Tests the falsifiable prediction the real-convergence result made as a
HYPOTHESIS ("PRH caught mid-climb — *scale-driven*"). Three controlled scale ladders, an
independent semantic reference, lexical control. **The hypothesis is not supported — and this
result corrects my own earlier framing.**

## The pre-registered test FAILED (and stands)
Gate (`PREREG_scale_law_2026_06_03.md`): pooled Spearman ρ(alignment, log params) ≥ 0.50 at the
fixed 0.66 read-out layer. **Result: ρ = −0.573** (opposite sign), and the perfectly-controlled
**Pythia** ladder ran the *wrong* way (ρ = −0.8). Pre-registered outcome: **NOT CONFIRMED.** It stands.

## Why — and the honest investigation that followed
Symmetric discipline (the same reflex that distrusted the bare-word null earlier): before accepting
"scale doesn't matter," rule out a measurement artifact. A single fixed layer handicaps models whose
semantic peak sits elsewhere. So I swept layers — and the answer **flipped with the read-out rule**:

| layer rule | pooled ρ | Pythia | note |
|---|---|---|---|
| fixed 0.66 (pre-registered) | −0.573 | −0.8 | alignments 0.04–0.10 (5× too low — misses the peak) |
| best-by-cat_struct | +0.191 | +0.4 | noisy selector |
| max-over-layers | +0.836 | +1.0 | **confounded:** bigger models = more layers to maximize over |

Neither alternative is trustworthy (`run_scale_law_v2.py`): best-by-cat_struct is a noisy selector;
max-over-layers favours models with more layers (gpt2-xl 48 vs pythia-14m 6). **I did NOT swap the
failed pre-reg null for the flattering +0.836** — that is exactly the asymmetric-discipline trap a
prior audit caught. The clean test is a UNIFORM, non-selected read-out at the same *relative depth*.

## The clean answer: a uniform-depth map (`run_scale_law_v3_depth.py`)
Scale trend (Spearman ρ vs log params) at each uniform read-out depth:

| depth | pooled ρ | Pythia (gold) | GPT-2 | Qwen2.5 |
|---:|---:|---:|---:|---:|
| 0.5 | −0.636 | −0.80 | +1.00 | +0.50 |
| 0.66 | −0.573 | −0.80 | +1.00 | +0.50 |
| 0.8 | −0.582 | −0.80 | +1.00 | +0.50 |
| 0.9 | +0.064 | −0.20 | +1.00 | +0.50 |
| **1.0 (final)** | **+0.355** | **+1.00** | +0.80 | **−1.00** |

**The predicted scale effect appears only at the final layer, only in 2 of 3 ladders, and reverses
in the third.** Pythia shows a clean monotonic final-layer climb (0.49→0.56→0.60→0.60) and GPT-2 a
jump (small ~0.15 → large ~0.63), but **Qwen reverses (−1.0)** and the effect is **flat-to-negative
at every other read-out depth.** Pooled never reaches 0.50. A genuine scale law would be robust to
read-out depth and hold across families. This is neither.

## Honest verdict
- **Scale does NOT robustly drive semantic convergence in 14M–3B.** The pre-registered test failed;
  the post-hoc signal is depth-fragile (final-layer-only) and family-inconsistent (Pythia +1 vs
  Qwen −1). Not established.
- **The robust driver is BETWEEN-family (training data), not within-family scale** — consistent with
  the real-convergence finding that outliers track data recipe (Phi, Qwen), not size.
- **This corrects my own hypothesis.** The convergence result floated "PRH caught mid-climb,
  scale-driven." The controlled ladders say the *scale-driven* part is wrong in this range; the
  *data-driven* part is what holds. The discipline ran on my own same-day hypothesis.

## What survives as a real contribution
- **A methodological finding that matters for convergence studies generally:** concept geometry in
  these decoder LMs lives **near the final layer**; mid-layer read-outs (a common default) read it
  **~5× too low**, and **scale comparisons are highly sensitive to read-out depth.** A convergence
  study that fixes one mid-layer can manufacture a null. (This is *why* the v1 pre-reg failed.)
- **The clean negative itself:** on the gold-standard Pythia ladder, the scale effect is final-layer-
  only and saturates by 70M; it is not the smooth climb the PRH-limit story predicted at this range.

## Scope / caveats
- Does **NOT** refute the Platonic Representation Hypothesis globally. PRH's scale climb is argued over
  a far wider range and different modalities; this says only that **within 14M–3B, text concept-geometry
  vs a sentence embedder, on these ladders, there is no robust scale law** — family/data dominates.
  A wider ladder (7B–70B) and a pre-registered final-layer measure are the honest next test.
- Alignment-to-an-embedder is a proxy for "approaching the shared semantic geometry"; 192 concepts;
  small models; one reference family (MiniLM; mpnet tracked in v1). The final-layer numbers are
  spiky across the last 1–2 layers (representation churns near the output) — itself a reason the
  effect there is not trustworthy without replication.

## Reproduce
`run_scale_law.py` (v1, pre-registered fixed-layer null) · `_v2` (layer-rule sensitivity) ·
`_v3_depth` (uniform-depth map — the clean answer). Results: `scale_law*_result.json`.
Prereg: `PREREG_scale_law_2026_06_03.md`.
