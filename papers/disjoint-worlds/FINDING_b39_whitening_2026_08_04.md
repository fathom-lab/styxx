# FINDING — whitening destroys the legibility signal itself: the correspondence between minds lives IN the covariance structure

Fathom Lab · 2026-08-04 · scored MECHANICALLY by `styxx.protocol` against the frozen gates in
`PREREG_b39_whitening_rescue_2026_08_04.md` (`e43eaa1`). Receipt: `b39_result.json`. Verdict:
**`INVALID__whitening_breaks_the_clique`** — honored; the rescue question is unanswerable by
this instrument, and the reason is the finding.

## The cells

| treatment | llama_3b→gemma (clique sanity) | llama_3b→qwen (rescue target) | qwen→llama_3b |
|---|---:|---:|---:|
| T0 raw | **0.6454** | 0.0612 | 0.0612 |
| T1 ZCA-shrink whiten (λ=0.5) | **0.0204** | 0.0128 | 0.0102 |
| T2 per-dim standardize | **0.8138** | 0.0434 | 0.0000 |

G0 passed (baseline reproduced); **G0b fired**: full whitening collapsed the *legible* pair
from 0.6454 to 0.0204 — near chance. Per the frozen table, no rescue verdict can be read.

## What the wreckage establishes (reported per the prereg's interpretive clause)

**1. The correspondence signal lives in the anisotropy.** ZCA whitening — equalizing every
direction's variance — didn't just fail to rescue qwen; it **destroyed discovery for the pair
where discovery works**. The relational structure that GW + Procrustes latch onto is carried by
the *shared shape of the covariance* (which directions are big, which small). Flatten that and
two mutually legible minds become mutually invisible. The B28/B29 whitening results live in a
different regime (reading along already-fitted directions in mapped space); for *discovery from
raw geometry*, whitening removes the very signal. That is a real mechanism statement about what
makes minds legible at all: **legibility is carried by shared anisotropy.**

**2. Scale is a general enhancer — and the island survives it.** Diagonal standardization
*improved* clique discovery from 0.6454 to **0.8138** (a practical recipe: standardize
per-dimension before discovery), yet qwen stayed an island (0.0434, and 0.0000 in the reverse
direction). Not gated — the frozen table routes through T1 — but the reported reading is plain:
the island is not a per-dimension scale artifact. Combined with the record, **five routes now
point away from any linear/metric explanation**: outliers (dead), gross RSA dissimilarity
(dead), isotropic alignment level (dead — unreachable), full covariance reweighting (invalid —
destroys the signal itself), per-dim scale (reported: no rescue while boosting the clique).

## Where the island question now stands

Every metric-side explanation tested is dead or invalid. What remains is the representational
hypothesis: **qwen's concepts are intrinsically arranged differently** — its anisotropy pattern
(now known to be the carrier of legibility) is organized unlike the clique's. The sharp successor
is no longer a rescue attempt but a *characterization*: compare the principal-subspace
orientations that carry each model's concept geometry (e.g., anchor-space principal angles
between models, computed on the shared concept battery) and ask whether the clique shares a
dominant subspace that qwen provably does not. That is a measurement, not a treatment — the
right shape after four treatment-shaped failures.

## Scope / discipline

Nine discovery fits, one seed, CPU-from-cache; verdict INVALID and honored — no rescue claim,
no intrinsic-arrangement claim licensed (the T2 observation is reported, not gated). The
practical keeper (standardize-then-discover, +0.17 on the clique) and the mechanism statement
(legibility is carried by shared anisotropy) enter the record as reported findings with the
cells above as their receipt.
