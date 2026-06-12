# PRE-REGISTRATION — B31: heavy-machinery content transport. Was cycle-6's telepathy wall a power problem, or real? (frozen)

**2026-06-12 · Fathom Lab / styxx. Frozen before any score is seen. Runner: `run_content_wall.py`
(SEED=0). Receipt: `content_wall_result.json`; figure `content_wall.png`. Backlog B31. Cycle 6
(`FINDING_concept_decode_2026_06_12.md`, CONTENT-WEAK) found cross-model concept identity collapses to
chance through a label-free linear map — but the map was underpowered (anchor R² ≈ 0.06 from only 40
anchor concepts). The honest caveat: "not with THIS method at THIS scale, not impossible." This cycle
removes the scale excuse: scale the anchor data ~18× with the SAME readout, and ask whether content
identity lifts off chance (the telepathy door opens) or stays walled even with a well-fit map (the wall
is real, not power).**

## The question

Cycle 6's map could not fit (40 anchor concept centroids cannot pin a hidden-state→hidden-state linear
map). If the failure was underpowering, a far larger anchor set should (a) raise the map's held-out R²
substantially and (b) lift cross-model concept retrieval above chance. If content identity still does not
transport DESPITE a well-fit map, the wall is a real property of linear cross-model transport, not a
sample-size artifact — a strong, decisive negative.

## Design — change ONLY the anchor count; keep cycle 6's readout identical

- Reference gemma-2-2b; targets Llama-3.2-3B (primary), Qwen2.5-3B (secondary). Mid-stack layer
  `round(0.5·nL)` per model, as cycle 6.
- **TEST set: the EXACT 20 held-out concepts from cycle 6** (reconstruct cycle 6's seeded 60-concept
  split so the comparison is apples-to-apples). chance top-1 = 1/20 = 0.05.
- **ANCHOR set, scaled ~18×:** cycle-6's 40 anchor concepts PLUS ~80 additional concrete nouns
  (disjoint from the 20 test concepts), each rendered in the same templates, and — the key change —
  every (concept, template) instance is its OWN anchor point (not a per-concept centroid). ≈ 120
  concepts × 6 templates ≈ 720 paired anchor POINTS, versus cycle 6's 40 centroids.
- **Map + readout: IDENTICAL to cycle 6 for the gate** — fit a label-free ridge map target→gemma on the
  anchor points (alpha by held-out R²), ZCA-whiten in gemma's space, retrieve each mapped TEST concept
  centroid against the gemma TEST concept centroids by cosine (top-1/top-5/MRR), plus the per-item
  ("single thought") variant. The ONLY change from cycle 6 is the anchor count. Report map held-out R².
- Controls: random-map floor (200 draws, top-1 p95); gemma in-model ceiling; the cycle-6 numbers
  (Llama centroid top-1 0.0) as the recorded baseline. Descriptive extras: a mapped-space-whitened
  readout (B32 recipe) and a raw (no-whiten) readout.

## Frozen gates (verdict precedence)

Let `acc1` = mapped-Llama top-1 (gemma-whitened, the cycle-6-identical readout); `chance` = 0.05;
`r2` = the map's held-out R².

- **CONTENT-TRANSPORTS** iff `acc1 ≥ 0.15` (≥ 3× chance) AND `acc1` > the random-map top-1 p95. → cycle
  6's wall was a POWER problem; cross-model content identity DOES transport with a well-fit linear map.
  The telepathy door opens one rung (still needs white-box reps + paired anchors).
- **CONTENT-WALL** iff `acc1 < 0.15` (or `acc1` ≤ random-map p95) AND `r2 ≥ 0.40`. → content identity
  does NOT transport even with a well-fit linear map; the wall is a real property of linear cross-model
  transport, not sample size. The value/content asymmetry is fundamental (to linear maps), not an artifact.
- **POWER-BOUND** iff `acc1 < 0.15` AND `r2 < 0.40`. → the map still did not fit even at 18× anchors;
  inconclusive on the wall (the open door is non-linear / vec2vec, not more linear anchors).
- **PARTIAL** — top-1 modest but top-5/MRR strong (coarse content, not exact identity); report precisely.

chance, the 3×-chance bar (0.15), and the R²=0.40 well-fit threshold are frozen. Bars do not move.

## What it does NOT settle (pre-committed honesty rail)

Even CONTENT-TRANSPORTS is not telepathy: it needs white-box activations AND a paired anchor corpus AND
a fixed N-way set; zero-paired transport is a closed negative, cross-vendor universality is killed, and
this is a 20-way set, not open vocabulary. A CONTENT-WALL result bounds LINEAR cross-model transport; a
non-linear / vec2vec map is a separate, heavier bet not run here. Linear, mid-layer, template-mean
centroids, local same-cluster models. Neutral concepts, last-token pre-output, no generation.
