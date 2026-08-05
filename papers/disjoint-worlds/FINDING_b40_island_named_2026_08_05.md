# FINDING — the island is named: qwen builds its geometry around different dominant concept contrasts, and a small subspace disagreement costs catastrophic legibility

Fathom Lab · 2026-08-05 · scored MECHANICALLY by `styxx.protocol` against the frozen gates in
`PREREG_b40_anisotropy_signature_2026_08_05.md` (`f69366c`). Receipt: `b40_result.json`.
Verdict: **`ISLAND_NAMED__qwen_dominant_subspace_differs_and_affinity_tracks_legibility`** —
both order-statement gates passed on a pure measurement, after four treatment-shaped failures.

## The measurement

Top-20 eigenvectors of each model's double-centered concept Gram — "the concept contrasts this
mind's geometry is built around" — live in the shared 462-concept space and are label-aligned
across models regardless of ambient dimension. Subspace affinity over all six unordered pairs:

| pair | affinity | b37 discovery (dir-mean) |
|---|---:|---:|
| llama_3b ↔ llama_1b | **0.8717** | 0.8137 |
| llama_3b ↔ gemma_2b | **0.8480** | 0.6199 |
| llama_1b ↔ gemma_2b | **0.7932** | 0.7398 |
| gemma_2b ↔ qwen_1p5b | 0.7220 | 0.1684 |
| llama_1b ↔ qwen_1p5b | 0.7375 | 0.0408 |
| llama_3b ↔ qwen_1p5b | 0.7306 | 0.0574 |

- **G1 (separation): PASS** — the ranges are disjoint: min clique-internal 0.7932 > max
  qwen-to-clique 0.7375 (margin 0.0557). Every clique pair agrees more about which contrasts
  dominate than any qwen pair does. An order statement over six measured values; one
  overlapping pair would have killed it.
- **G2 (tracks legibility): PASS** — Spearman(affinity, discovery) = 0.7143 across the six
  pairs, clearing the 0.7 floor the prereg pre-declared as blunt at n=6. Exactly one adjacent
  inversion (llama_3b↔gemma vs llama_1b↔gemma) — the tolerance the prereg computed in advance.

## The named island

**qwen's dominant concept subspace is measurably different from every clique member's, and
that difference orders who can read whom.** Combined with B39's mechanism (legibility is
carried by shared anisotropy — whiten it away and even clique pairs go blind), the island now
has both a mechanism and a signature: qwen organizes the *same* 462 concepts around a
different set of dominant contrasts, and the label-free discovery machinery — which can only
latch onto shared geometry — finds nothing to hold.

## The sharpest new fact: the response is violently nonlinear

The affinity gap is **modest** (0.79 vs 0.74 — a 7% relative difference); the legibility gap
is **catastrophic** (0.62–0.81 vs 0.04–0.17 discovery). A small disagreement in dominant
contrasts costs almost all mutual readability. This is the cliff b38 went looking for with
isotropic noise and could not reach: **the cliff is real, but it lives on the
subspace-affinity axis, not the RSA axis.** Reported, not gated — locating the transition
precisely needs affinity-controlled perturbations (rotate the dominant subspace by controlled
angles: a directed successor with a measured axis to walk, which is what b38's successor
lacked).

## Practical keeper

Affinity is a **pre-screen**: four eigendecompositions and a Frobenius norm — seconds, no
discovery run — predicts the b37 legibility ordering at ρ 0.71. "Compute affinity before
attempting discovery" enters the toolbox beside standardize-then-discover (B39).

## Scope

Six pairs sharing four models ≤3B; k=20 frozen; one anchor split (seed 343); order statements
and a blunt n=6 rank correlation, both declared as such in the frozen prereg; no direction
claims (affinity is symmetric by construction — the B37 lesson). The concept-level story —
WHICH contrasts qwen weights differently — is the named successor (inspect the discordant
eigenvectors; a fully interpretable, concept-labeled analysis).
