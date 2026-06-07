# PREREG — Symbol-Coded Forced-Choice Introspection (v2)

**Frozen 2026-06-06, before any scored data. Fathom Lab / styxx.**
Follows the v1 free-response NULL (FINDING_v1_freeresponse_null): small instruct models, in
free response, detect only generic perturbation (random ≥ concept) and never name the injected
concept — gated by RLHF persona, not internal access. v2 asks whether the information is
recoverable once the persona AND the logit-lens confound are removed.

## Question

Can a 2–3B model **pick** which concept was covertly injected into its residual stream when
the answer is an **arbitrary letter code** (so a raw logit-lens on the injected vector cannot
win), it is read at a **clean (unperturbed) position**, and wrong-layer / no-vector controls
collapse? If yes → grounded introspective access exists at this scale and was *masked*, not
absent (flips the frontier-only prior). If no, with the apparatus validated → the planted
concept is mechanistically present but not forced-choosable through the LM head (sharpest
"we measure behavior, never minds").

## Apparatus (reused, locked)

Steering-validated paired diff-of-means concept vectors; rms all-position forward hook at
0.60-depth layer; dose α=10 (α=16 robustness). 8 concepts. Read-out = first-token logit mass
over restricted option sets (single forward, no generation).

- **Symbol-code 8-way** (chance 0.125): code table maps the 8 concepts to letters A–H in a
  per-trial shuffled order; prefill `"The injected concept's letter is:"`; read over {A..H};
  argmax → predicted letter. The injected concept word never appears as the answer; injecting
  "ocean" cannot raise letter "C".
- **2-AFC** (chance 0.5): A={c}, B={foil}; read over {A,B}.
- **Concept-WORD 8-way** (logit-lens comparison arm): same but read over the concept *words* —
  expected to be inflatable by a pure logit lens; used to *dissociate* lens from access.

## Conditions

inject(v_c, α) · random(equal-norm seeded unit dir, α) · prime(α=0, concept named in prompt) ·
clean(α=0) · clean_readpos(inject all positions EXCEPT the answer-read position) ·
wrong_layer(inject v_c at 0.25-depth instead of 0.60).

## Pre-registered kill-gate (frozen; answer key hashed + printed before scoring)

- **ABORT / VALIDITY (run first):** prime-2AFC accuracy ≥ **0.75**. If below, the forced channel
  cannot carry concept info even when handed the answer → **UNINFORMATIVE_APPARATUS** (NOT a null).
- **PRIMARY** (symbol-code 8-way, chance 0.125): id_acc8(inject) ≥ **0.30**, else REPORT_AS_LANDED null.
- **SPECIFICITY:** id_acc8(inject) − id_acc8(random) ≥ **0.15** (else generic-perturbation).
- **CIRCULARITY (load-bearing):** wrong_layer < 0.30 AND clean < 0.30 AND
  (clean_readpos − random) ≥ 0.15. If the result only survives when answers are concept WORDS
  (word-arm passes) while symbol-code or clean_readpos fail → report as **logit-lens, not access**.
- **PERM-NULL:** label-permuted 8-way null mean ∈ [0.05, 0.20], 95th pct < 0.10.
- **SURVIVED** requires ABORT ∧ PRIMARY ∧ SPECIFICITY ∧ CIRCULARITY on **≥ 2** models.

## Verdict logic

UNINFORMATIVE_APPARATUS (abort fails) · SURVIVED (≥2 models all gates) · PARTIAL (1 model) ·
REPORT_AS_LANDED_null (apparatus valid, info not forced-choosable). A clean null with a passed
abort gate is a real, prior-relevant result.

## Honest ceiling

Modal outcome is a clean null (red-team consensus p(clean positive) ≈ 0.12). A SURVIVED =
"grounded forced-choice introspective *discrimination* exists at 2–3B and was masked by the
verbal channel" — a deployable white-box self-monitor primitive and a correction to the
frontier-only prior — NOT "small models are introspectively aware." This measures a read-out
channel, never a mind. Stretch (only if primary resolves with time): within-model self-decode
*ceiling* (is the planted state linearly recoverable at all?) — the prerequisite the cross-model
"telepathy" arm would need; cross-model is deferred (the lab's real-convergence data bounds it).
