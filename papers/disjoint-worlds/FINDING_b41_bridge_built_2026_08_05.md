# FINDING — the bridge is built: correcting twenty named concept-contrasts turns an illegible mind into the most legible one measured, and random directions do nothing

Fathom Lab · 2026-08-05 · scored MECHANICALLY by `styxx.protocol` against the frozen gates in
`PREREG_b41_bridge_2026_08_05.md` (`6fe0e5c`). Receipt: `b41_result.json`. Verdict:
**`BRIDGE_BUILT__named_contrasts_causally_block_legibility`** — all four frozen gates passed.

## The result

The island arc's naming (B40: qwen's top-20 dominant concept-contrasts differ from the clique's)
was correlational. This is the causal test: replace qwen's top-20 contrast pattern with
llama_3b's in shared concept space (a rank-20 correction, 20 directions out of 392 anchor
dimensions), then run the **label-free** discovery machinery on the result — it still receives
shuffled rows and must find the correspondence itself.

| arm | discovery | gate | |
|---|---:|---|---|
| A0 baseline (qwen, no surgery) | **0.0612** | ≤ 0.15 | ✓ the island reproduced |
| **A1 bridge (qwen ← llama's 20 contrasts)** | **0.9745** | ≥ 0.30 | **✓ a 16× jump** |
| A2 specificity null (qwen ← 20 RANDOM directions) | **0.0000** | ≤ 0.15 | ✓ wrong directions rescue nothing |
| A3 no-harm (gemma ← same surgery) | **1.0000** | ≥ 0.30 | ✓ recipe doesn't damage, it helps |

**Twenty directions are the entire barrier.** Correcting exactly the contrasts B40 named turned
an island (0.0612, ~4× chance) into the single most discoverable pair the program has measured
(0.9745, 68× chance) — more legible than any *natural* pair, clique members included. The same
rank-20 surgery with a random orthonormal frame rescued **nothing (0.0000)**: it is not that
any 20-dimensional intervention manufactures legibility; it is that these specific twenty
concept-contrast directions were blocking it. And the recipe is not a wrecking ball — applied to
the already-legible gemma pair it drove discovery to a perfect 1.0.

## What this establishes

The B40 naming was **causal**, not a correlate. The qwen island — which survived outlier-taming,
isotropic noise at maximum dose, full ZCA whitening, and per-dimension standardization (four
prior interventions, all failed or invalid) — falls to a rank-20 correction of precisely the
directions the anisotropy signature identified. This is the arc's first **engineered legibility
bridge**: an intervention that, by design, makes a provably unreadable mind readable, and whose
specificity control confirms the mechanism rather than the machinery.

The one-sentence form: **the barrier between two minds can be a small, nameable, correctable set
of directions — here, twenty — and correcting them alone rebuilds the bridge that broad
interventions could not.**

## Epistemic boundary (declared in the prereg, restated at full volume)

The surgery is **label-aligned**: it uses the true concept correspondence to compare and swap
eigenvector patterns in concept space. This is an INTERVENTION CEILING, not a deployable
label-free protocol — it answers *"are the named contrasts the causal barrier?"* (yes),
**not** *"can you bridge two minds with no labels at all?"* (untested; the discovery step is
label-free, but the correction is not). No telepathy-without-labels claim is made or implied.

## Scope

One source→island pair with its clique no-harm control, one seed (343), k=20 frozen from B40,
CPU-from-cache, 4 discovery fits. The A2 null at exactly 0.0 and A3 at exactly 1.0 are strong
but single-seed; the honest successor is a seed/rank sweep (**how few directions suffice, and
does the bridge hold across draws**) plus the concept-level identity of the twenty blocking
contrasts (the interpretable story: *what* is qwen weighting differently). The bridge is built;
its dose curve and its meaning are the next rungs.
