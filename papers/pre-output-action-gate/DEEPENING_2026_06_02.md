# DEEPENING — where the destructive decision forms, and whether it's causal

**Date:** 2026-06-02
**Builds on:** [RESULT_emitted_action](RESULT_emitted_action_2026_06_02.md) (SURVIVED).
Two questions the SURVIVED result left open: (1) is this a *forming commitment*
or just decoding the imminent output? (2) is the signal *causal*?

## 1. The destructive commitment forms mid-stream (not at the output)

Per-layer leave-one-category-out AUC, by proportional depth:

| Model | peak AUC @ depth | 25% | 50% | 75% | 100% (final) |
|---|---|---|---|---|---|
| Qwen2.5-1.5B | 0.92 @ 68% | 0.62 | 0.78 | 0.81 | **0.67** |
| Qwen2.5-3B | 0.76 @ 83% | 0.41 | 0.35 | 0.74 | **0.47** |
| Llama-3.2-3B | 0.91 @ 64% | 0.54 | 0.72 | 0.86 | **0.33** |
| gemma-2-2b | 0.80 @ 65% | 0.64 | 0.57 | 0.79 | 0.71 |
| Llama-3.2-1B *(failed gate)* | 0.65 @ 100% | 0.38 | 0.40 | 0.40 | 0.65 |

The signal **peaks mid-stream (64–83%) and decays at the final layer** for the
passing models. If this were merely decoding the about-to-be-emitted token, it
would be *strongest* at the output layer. It is strongest where the decision
*forms*. And the one model that failed the gate (Llama-1B) is exactly the one
with **no mid-stream commitment** — its weak signal only appears at the last
layer. Mechanism and gate agree.

## 2. The mid-layer direction is CAUSAL (steering flips the choice)

Test: fit the destructive direction at the best layer; for scenarios the model
chose destructively, regenerate while subtracting `alpha · direction` at that
layer (all positions). Controls: a **random unit direction** (specificity) and
**output validity** (a flip into gibberish is not a flip). `alpha` in multiples
of the destructive-coordinate std.

**Qwen2.5-1.5B (layer 19, n=16 destructive):**

| alpha | real flip→safe | random flip | real valid |
|---|---|---|---|
| k=2 | 0.12 | 0.00 | 1.00 |
| k=4 | 0.25 | 0.00 | 1.00 |
| k=8 | 0.50 | 0.06 | 1.00 |
| k=16 | **0.69** | 0.12 | 1.00 |

**Llama-3.2-3B (layer 18, n=11):** real 0.09 / 0.18 / 0.18 / **0.46** vs random
0.00 / 0.00 / 0.09 / 0.18; valid 1.00 throughout.

For Qwen-1.5B this is **clean causal steering**: a smooth dose-response, the
real direction flips up to 69% of destructive choices to safe, the random
direction barely moves (≤0.12), and **every steered output is still a valid tool
call**. At a moderate scale (k=4) it is fully specific — 25% flipped, 0% random,
100% valid. Llama-3B shows the same effect, weaker.

**styxx does not just predict the destructive action — it can causally suppress
it.** Predict → prevent.

## Honest boundary

- **Dose-response has a cost.** At large alpha (k=16) the random direction also
  starts to flip a little (0.12–0.18) and validity dips slightly — push hard
  enough and you disrupt generation generally. The trustworthy regime is the
  moderate alpha where real flips substantially and random stays ~0.
- **Not 100%, and not uniform.** Even at k=16 Qwen-1.5B flips 69%, not all.
  Llama-3B's effect is real but materially weaker. 2 models, exploratory.
- **Exploratory, by necessity:** you cannot pre-register a steering scale blind.
  The random-direction control + validity check are the rigor; a confirmatory
  pre-registered run (now that the alpha scale is known) is the next step.
- **Same simplified harness** as the parent study (presented tool list, n small).
- **Correlational → causal is supported here, on this setup** — not a universal
  mechanistic claim.

## Synthesis

The destructive decision forms as a **mid-stream linear representation** that is
both **readable** (the pre-output gate predicts it, 0.65–0.92) and **steerable**
(subtracting it flips the choice, cleanly for Qwen-1.5B). That is a coherent,
falsification-survived, mechanistically-grounded story: *read the commitment
before the agent acts, and bend it.*

## Next

- Confirmatory **pre-registered** steering (frozen alpha + layer, more models).
- **Open tool-call** capture (drop the presented-list harness).
- The **deployable gate/guard:** calibration, false-positive budget, the
  predict-and-prevent loop in a live agent.

— 2026-06-02; controls are the result.
