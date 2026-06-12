# FINDING — the telepathy test: content identity does NOT transport cross-model under label-free linear transport (CONTENT-WEAK)

**2026-06-12 · Fathom Lab / styxx. Pre-registered: `PREREG_concept_decode_2026_06_12.md` (frozen
pre-run, committed 7fb1600). Receipt: `concept_decode_result.json`; figure `concept_decode.png`. This
adjudicates "are we close to telepathy" with a falsifiable run, not an argument: can you decode WHICH
concept (from a fixed set) a *different* model is representing, through a label-free map, on concepts the
map never saw? The answer, with this method at this scale, is no.**

## Result — the thermometer transports, the transcript does not

Sixty concrete concepts, six neutral templates, mid-stack last-token representations. A label-free ridge
map target→gemma + ZCA whitening were fit ONLY on 40 anchor concepts; retrieval was tested on 20
held-out concepts the map never saw (chance top-1 = 0.05).

| readout | top-1 | top-5 | MRR |
| --- | --- | --- | --- |
| chance (20-way) | 0.05 | 0.25 | — |
| random-map floor (Llama, p95) | 0.05 | — | — |
| **Llama-3.2-3B → gemma (centroid)** | **0.0** | 0.25 | 0.1444 |
| Llama-3.2-3B → gemma (per-item) | 0.0333 | — | — |
| **Qwen2.5-3B → gemma (centroid)** | **0.1** | 0.2 | 0.1944 |
| gemma → gemma in-model (ceiling) | **0.9583** | — | — |

- **Within a model, concept identity is almost perfectly readable.** gemma identifies its own held-out
  concepts across template instances at top-1 0.9583. The "transcript" exists and is legible in-model.
- **Across models, through the label-free map, it collapses to chance.** Llama→gemma centroid retrieval
  is 0.0 top-1 (below chance, below the random-map floor 0.05); top-5 0.25 equals chance top-5;
  per-item 0.0333. Qwen→gemma is 0.1 top-1 — above chance but below the frozen 3×-chance bar (0.15) and
  with top-5 0.2 below the 0.50 bar. Neither clears the CONTENT-DECODABLE gate.
- **Verdict per the frozen gate: CONTENT-WEAK.** Cross-model concept identity does not decode beyond the
  alignment floor with this label-free linear method.

## Why this is the decisive answer to the telepathy question

The portable-conscience arc transports a few low-dimensional VALUE directions (truth, refusal, danger)
across models. This shows the SAME class of label-free map does NOT transport high-dimensional CONTENT
identity. The mechanism is clean: the value axes survived because a difference-of-means direction is 1-D
and robust to a lossy map (the arc's own note: "directional transfer ≠ representational identity",
ridge-map R² ≈ 0 yet the value direction still transferred). Identifying which of twenty concepts a
foreign model holds needs the map to preserve the whole content manifold — and a lossy map destroys it.
**The cross-model channel is a value thermometer, not a content transcript.** Reading "is this
true / dangerous" travels; reading "is this an apple or a violin" does not. That is exactly the
categorical gap between the real result and telepathy, now with a receipt instead of an assertion.

## Honest bounds — what the negative does and does NOT claim

This is "not with THIS method at THIS scale", not "impossible". The label-free map was underpowered for
full content reconstruction: held-out anchor R² was 0.0613 (Llama) and negative for Qwen (below the
mean-predictor baseline) — 40 anchor concepts cannot pin a full hidden-state-to-hidden-state linear
map, so the channel is genuinely lossy. A much stronger map
(many more anchors, or non-linear / vec2vec-grade machinery) might recover content identity; that is the
documented separate research bet (`styxx.transport` notes zero-paired and heavy-machinery transport as
out-of-scope). The honest claim is bounded: under the same label-free LINEAR transport that carries the
value axes, content identity does not cross — value transport is robust to a lossy channel, content
transport is not. Linear, mid-layer, template-mean centroids, 20-way, local same-cluster models
(gemma source; Llama-3.2-3B + Qwen2.5-3B). Neutral concepts, last-token pre-output, no generation.

## The telepathy ledger (pre-committed, now scored)

Even a positive here would not have been telepathy — it needs white-box activations AND a paired anchor
corpus AND a fixed N-way set. The result is NEGATIVE on top of those caveats: the content channel does
not cross at all under this transport. So on the path from "shared value geometry across minds" (real,
established) to "one mind reads another's arbitrary content" (telepathy), this cycle confirms the very
next rung — cross-model content identity — does NOT come for free with the value-axis machinery. The
distance to telepathy is large and mostly categorical; this closes one of its gaps in the honest
direction. The open door is the heavy-machinery transport bet, not the linear map we have.
