# PREREG — GAVAGAI-X: does the interlingua cross the species barrier?

**Frozen 2026-06-10, before any scored run. Fathom Lab / styxx.**

GAVAGAI v0 recovered concept identity between causal language models — one species of mind, all
trained next-token. If the identity-carrying geometry is a property of LANGUAGE MODELING, the
channel dies at the species boundary. If it is a property of LEARNED MEANING itself, it should
cross to minds with a different objective, architecture class, and output space entirely:
contrastively-trained sentence embedders. This is also the strongest available shared-apparatus
control for v0 — the embedders never see the templates' hidden states or the 0.66-layer
convention; their geometry comes from their own encoder applied to text.

## Apparatus (frozen)

- LLM side: the 10 minds' norm-equalized battery reps (`normeq_reps.npz`, receipt of anatomy v2).
- Embedder side ("the other species"): all-MiniLM-L6-v2 and all-mpnet-base-v2 (CPU), each battery
  word encoded per the SAME 8 frozen templates, each template embedding L2-normalized
  (`normalize_embeddings=True`), averaged — the convention mirrored at the sentence level.
- Translator: the frozen GAVAGAI matcher (`run_gavagai_v0.translate`), labels hidden by seeded
  permutation (seed 0), RDMs via `styxx.mind.distmat`.
- Population: 20 cross-species ordered-free pairs (10 LLMs × 2 embedders), scored once each
  (LLM as A, embedder as B).

## Pre-registered gates (frozen)

- **X1 (the channel crosses species):** mean cross-species concept accuracy ≥ 10× chance
  (≥ 0.1042) AND above the 95th percentile of the identity-decoupled null of GAVAGAI v0's form
  (100 runs, seed 0, computed on the cross-species pairs). PASS → **SPECIES-CROSSED**;
  FAIL → **SPECIES-BOUND** (the interlingua is a within-objective phenomenon at this scale —
  reported as the boundary).
- **X2 (descriptive):** cross-species mean vs the v0 cross-family LLM↔LLM mean (0.1661, receipt
  `gavagai_v0_result.json`) — is the species barrier thicker than the family barrier?
- **X3 (descriptive):** category accuracy (chance 0.125), embedder-pair (MiniLM↔mpnet) accuracy
  as the within-species reference for the other species.

## VOID

- VOID-PIPELINE: embedder self-pair (MiniLM→MiniLM, labels hidden) must decode ≥ 0.99... note the
  matcher is deterministic given RDMs: self-pair accuracy 1.0 expected; bar 0.99.
- Smoke: 1 LLM + 1 embedder, `*_SMOKE_INVALID*` only.

## Honest prior

Leaning SPECIES-CROSSED but weaker than within-species: the old-convention MiniLM anchor partial
RSA to LLMs was 0.36 (confirm receipt) — structure is shared; whether it is shared distinctively
enough for assignment at 96 concepts is exactly what the gate decides.
