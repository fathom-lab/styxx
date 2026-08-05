# PREREG — B43: name the twenty — what does qwen weight differently, and is it a real, stable, coherent story?

Fathom Lab · 2026-08-05 · frozen before the scored run. B40 identified qwen's dominant
concept-contrasts as different from the clique's; B41 proved correcting them causally restores
legibility. Both are linear-algebra facts. This run asks the human question — **which concepts
does qwen organize differently** — with falsifiable guards so the answer cannot be post-hoc
storytelling.

## The object

Anchor rows, seed-343 split. U_Q, U_L = top-20 double-centered concept-Gram eigenvectors
(the B40 objects, in shared 462→392-anchor concept space, label-aligned). The **discordant
directions** are the principal vectors realizing the largest principal angles between
span(U_Q) and span(U_L) — dominant qwen contrasts most orthogonal to everything the clique
does. For the top-D=3 discordant qwen-side principal vectors, the interpretation is the set of
anchor concepts with the largest |loading| (top-15 each). Everything is pre-specified; no
direction or concept is chosen after seeing labels.

## Validity gates (both must pass, or there is no nameable story)

Two independent ways the "twenty" could be noise, each gated:

- **G1 stability** — the story must not be a seed artifact. Recompute the top-15 concept set of
  the single most-discordant direction across seeds {343, 1001, 1002}; mean pairwise **Jaccard
  ≥ 0.15**. Random 15-of-392 sets overlap at Jaccard ~0.02, so 0.15 is ~7× chance-stability —
  a real-signal bar, sign/rotation-invariant (set-based).
- **G2 coherence** — the story must be semantically real. The top-15 concepts of the most-
  discordant direction (seed 343) must be more semantically clustered than random 15-concept
  sets: mean pairwise MiniLM cosine, permutation **p ≤ 0.05** over 2000 random draws.

```gates
{"gates": {"G1_stability": {"metric": "mean_jaccard_top15_across_seeds", "op": ">=", "value": 0.15},
           "G2_coherence": {"metric": "coherence_perm_p", "op": "<=", "value": 0.05}},
 "outcomes": [{"when": {"G1_stability": true, "G2_coherence": true}, "verdict": "ISLAND_NAMED_IN_ENGLISH__stable_coherent_contrast"},
              {"when": {"G1_stability": true, "G2_coherence": false}, "verdict": "STABLE_BUT_NOT_SEMANTIC__concepts_reported_no_clean_story"},
              {"when": {"G1_stability": false, "G2_coherence": true}, "verdict": "COHERENT_ONE_SEED_ONLY__seed_artifact"},
              {"when": {"G1_stability": false, "G2_coherence": false}, "verdict": "NO_NAMEABLE_STORY__twenty_are_rank_space_noise"}],
 "smoke_verdict": "INVALID__smoke_plumbing_only"}
```

## Pre-commitments (anti-storytelling)

- The concept lists are reported **verbatim as computed** for all D=3 discordant directions,
  including on a `NO_NAMEABLE_STORY` verdict — the raw output ships regardless, so a null
  cannot be hidden and a pass cannot be embellished.
- The English gloss (if `ISLAND_NAMED_IN_ENGLISH`) is written to describe the top concepts as
  they fall out, and is explicitly labeled interpretation-of-a-measured-set, not a new claim.
- Semantic coherence uses MiniLM (`sentence-transformers/all-MiniLM-L6-v2`) — the same model
  the steering apparatus uses; disclosed as the coherence instrument with its own ceiling
  (it measures embedding-space clustering, not ground-truth conceptual kinship).

## Outcome reading

`ISLAND_NAMED_IN_ENGLISH` completes the arc's interpretability capstone: the barrier between
qwen and the clique is not just twenty abstract directions but a nameable difference in how a
family of concepts is organized. `NO_NAMEABLE_STORY` is equally publishable and bounds the
claim honestly: the twenty are real and causal (B41) but live below the single-concept level —
distributed, not nameable — and the arc says the barrier is sub-symbolic.

## Discipline

CPU-from-cache, seconds (eigendecomps + one sbert encode of 462 words). Smoke = D=1 at 40
anchors, INVALID-only. Result `b43_result.json`; scored by `styxx.protocol`; certified +
sealed before commit.
