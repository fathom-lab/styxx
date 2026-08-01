# FINDING — the door opens: cross-family content reading was capacity-limited, not bedrock

Fathom Lab · 2026-08-01 · scored under `PREREG_b31v2_content_transport_2026_08_01.md` (frozen
`645c799`, apparatus `f266be1`/`c4b6d56` committed before any scored cell). Receipt:
`b31v2_result.json`. Seed 31; the committed N=462 battery, split_concepts(seed=0): 392 anchors
fit every map, 70 held-out concepts no map ever saw.

## Verdict: `DOOR_OPENS__content_capacity_limited` — all three frozen gates pass

| target | M0 linear | **M1 MLP** | N1 shuffled | M1 × chance |
|---|---:|---:|---:|---:|
| Llama-3.2-1B (same family, G0) | 0.3429 | **0.8000** | 0.0000 | 56× |
| **gemma-2-2b (decisive cell, G1)** | **0.0143 = exact chance** | **0.7857** | 0.0143 | **55×** |
| Qwen2.5-1.5B (context) | 0.1143 | **0.7000** | 0.0143 | 49× |

- **G0 machinery: PASS** (0.8000 ≥ 0.53; the llama_1b cell reproduced identically across
  three process launches).
- **G1 the door: PASS at 5.5× its own gate** — gemma, the rung-2 existence proof that
  isometry does not grade readability (RSA 0.955, linear read at exact chance), reads at
  **0.7857 top-1 over 70 held-out concepts** through a two-layer MLP fit on the same
  extractions. 55 of 70 foreign thoughts identified; binomial against chance 1/70 is
  astronomically far from null.
- **G2 specificity: PASS** — the pairing-shuffled twin (same architecture, same training,
  same data, only the correspondence destroyed) reads at exact chance on every target. The
  lift is the pairing, not the architecture.

**The rung-2 cliff was a property of the linear map class, not of the minds.** The content
information is present in both representations and alignable — the linear/Procrustes family
simply could not express the warp between families. Cross-family content reading is an
engineering problem now, not a wall.

## The resolution of the isometry puzzle

Rung 2's falsified prediction (RSA should grade readability; it doesn't) now has a mechanism-
shaped answer: RSA-visible *global geometry* and linearly-accessible *content* dissociate from
nonlinearly-accessible content. gemma's geometry matches Llama's nearly perfectly at the
distance-matrix level, its content is fully recoverable nonlinearly, and the linear class
could see neither. What carries cross-family legibility is real, present, and warped.

## Honest boundaries (stated before anyone over-reads)

1. **PAIRED anchors, by design.** M1 trains on 392 anchor *pairs* — this is the
   heavy-machinery bet the B31 line always was, and it bounds the **ceiling** of content
   transport. It is NOT a label-free protocol: the rung-2 label-free result stands untouched,
   and whether the pairing itself can be recovered unsupervised (cycle-consistency /
   vec2vec-grade) is the named successor, not a claim of this finding.
2. **Scope:** one source (Llama-3.2-3B), three targets ≤2.6B, one seed, one layer choice per
   model (the committed frac rule), 70-way identification. Open-vocabulary, cross-scale, and
   seed-stability are successors.
3. **READ ONLY.** Nothing here touches the write side. The clean read≠write dissociation
   (RESULT_writelayer_decouple) is untouched: this run moves *what can be read*, not *what
   can be moved*.
4. **Ops disclosure:** two full supervised launches were OS/supervisor-killed at the gemma
   shard load; the fix was per-target process isolation + detached execution (the c86/c80
   lessons), committed as `c4b6d56` before any gemma/qwen cell was scored. The llama_1b
   cell's triple identical reproduction across launches is the regression evidence.

## What this changes

The telepathy-shaped claim now has an honest, measured form: **given ~400 paired concept
anchors, a small MLP decodes which of 70 held-out concepts a foreign mind is representing at
~80% top-1, across model families.** The witness/mount blindspot entry
(`high_isometry_unreadability`) stays true as scoped — it describes the label-free linear
class the deployed instruments use — and gains this pointer. Successors spawned in the
backlog: **B34** label-free nonlinear (recover the pairing unsupervised — the actual
telepathy bar), **B35** breadth/scale/stability (open-vocabulary, second source, seeds).
