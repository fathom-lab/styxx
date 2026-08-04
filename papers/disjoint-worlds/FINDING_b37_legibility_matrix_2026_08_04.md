# FINDING — the first mutual-legibility matrix: a readable clique, an illegible island, and two gates that measured less than they claimed

Fathom Lab · 2026-08-04 · scored MECHANICALLY by `styxx.protocol` against the frozen gates in
`PREREG_b37_legibility_matrix_2026_08_04.md` (`978f619`, apparatus committed after). Receipt:
`b37_result.json`. The formal verdict fired `LEGIBILITY_LAW_CANDIDATE__local_structure_symmetric`;
**this finding deliberately claims LESS than that label**, because reading the numbers exposed
two things the outcome table could not see. Both are reported before the result.

## The two instrument catches (stated first, at full volume)

1. **The "symmetry" gate measured the machinery, not the minds.** Three of six unordered pairs
   returned direction-accuracies that are **exactly equal to four decimals**
   (llama1b↔gemma 0.7398/0.7398; llama1b↔qwen 0.0408/0.0408; gemma↔qwen 0.1684/0.1684) —
   precisely the pairs with matching ambient dimensionality. An orthogonal Procrustes map and
   its reverse are transposes, and the assignment induced by a distance matrix and its
   transpose is identical: **the discovery machinery is direction-blind by construction**, up
   to PCA-truncation effects that only appear when dimensions differ (the llama-3B rows, diffs
   0.0076–0.0562). G3's "pass" is therefore a property of the instrument. Whether legibility
   between minds is *empirically* symmetric requires a direction-sensitive discovery method,
   which this arc does not yet have.
2. **G2 was a noise-passable gate, and I wrote it.** The kNN-vs-RSA comparison came out
   0.6537 vs 0.6344 — a margin of **0.0193 over the twelve non-independent pairs**, through a frozen
   floor of literally `> 0.0`. The gate passed; the honest reading is a **statistical tie**.
   The pre-stated law — local neighborhood preservation out-predicts global isometry for
   discovery — is **not supported at any interesting strength by this battery**. What survives
   is weaker: kNN-Jaccard clears its own 0.60 floor (G1), but so, nearly, does the incumbent.
   Neither predictor separates from the other here, and both correlations are inflated by the
   matrix's cluster structure (below).

## The result that IS licensed: the matrix itself

12 directed pairs, label-free discovery + read, both anchor cells reproduced (G0: 0.5918;
G0b: 0.0536), all nulls at/near chance:

| discovery (read) | → llama_3b | → llama_1b | → gemma_2b | → qwen_1p5b |
|---|---|---|---|---|
| **llama_3b** | — | 0.7959 (0.6857) | 0.5918 (0.5714) | 0.0536 (0.1429) |
| **llama_1b** | 0.8316 (0.6429) | — | 0.7398 (0.6143) | 0.0408 (0.0571) |
| **gemma_2b** | 0.6480 (0.5714) | 0.7398 (0.6429) | — | 0.1684 (0.3143) |
| **qwen_1p5b** | 0.0612 (0.1000) | 0.0408 (0.1000) | 0.1684 (0.2286) | — |

**The structure is stark and is the finding:** {llama_3b, llama_1b, gemma_2b} form a
**mutually legible clique** — every pair discovers at 0.59–0.83 and reads at 40–48× chance,
across families (gemma is not a Llama relative). **qwen_1p5b is an island**: no model
discovers it above 0.17, and it discovers no one above 0.17, in either direction. Cross-family
legibility is real, general within the clique — and not universal. Whatever qwen's
representation does differently, it is invisible to this machinery from every side.

Notable cell, flagged not claimed: gemma→qwen reads at 0.3143 (22× chance) on discovery of
only 0.1684 — the strongest qwen channel measured in the program, carrying the prereg's
`source_layer_unvalidated` label (gemma as source has no G0-locked read layer).

## What this run adds to the record

- **First all-pairs legibility map between artificial minds.** The clique/island topology is
  the primary artifact and it is unambiguous — no gate arithmetic involved.
- The theory question sharpens from "why gemma and not qwen" to: **what representation
  property makes qwen illegible from every direction while gemma joins a cross-family
  clique?** The tested predictors (RSA, kNN-Jaccard, spectral profile) all track the
  clique/island split coarsely and none separates from the others — the island's cause is
  not yet named.
- Two instrument lessons for every successor: symmetry claims need direction-sensitive
  machinery; comparative-predictor gates need margins that noise cannot pass (a `> 0.0`
  floor on n=12 is not a gate, it is a coin with extra steps).

## Scope

One seed (343), one split, 70-way read, four models ≤3B, source layers as-banked with
non-anchor sources labeled unvalidated, 12 non-independent pairs sharing 4 models. The
verdict label `LEGIBILITY_LAW_CANDIDATE` stands in the mechanical record with this finding as
its controlling interpretation: **matrix licensed; law not.**
