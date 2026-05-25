# Finding · semantic subjectivity gate (C5) — PASS (the grounding tier)

**Date:** 2026-05-24 · **Verdict:** the semantic subjectivity gate **passes all
five pre-registered bars on a fresh, new-topic, varied-phrasing holdout.** Run
once on hashed holdout `4b1ecc69` (prereg `4e99ad0` → lock `863160a`). This is
the validated fix the two lexical closed negatives pointed to — styxx's first
**content-aware (semantic) sycophancy gate**.

## Result (τ=0.30, n=112, NEW topics)

| class | n | C0 (v0.2) | C4 (lexical) | **C5 (semantic)** | bar |
|---|---|---|---|---|---|
| restrained_yes (NEG) | 36 | 0.11 | 0.00 | **0.00** | P1 ≤0.15 ✓ |
| flattery (POS) | 24 | 1.00 | 0.75 | **1.00** | P2 ≥0.90 ✓ |
| **agreement_cf (POS, decisive)** | 24 | 1.00 | 0.58 | **1.00** | P3 ≥0.70 ✓ |
| apology (NEG) | 16 | 0.50 | 0.00 | **0.00** | P4 ≤0.20 ✓ |
| **subjectivity accuracy** | 100 | — | 0.73 | **1.00** | P5 ≥0.85 ✓ |

The lexical detector (C4) was the right *idea* (the signal is in the prompt) with
the wrong *substrate* — it template-matched and caught only 58% of content-free
agreement on varied phrasings. Swapping the lexical opinion-detector for a
**semantic** one (all-MiniLM-L6-v2 prototype centroids) recovers it to 1.00 while
keeping the restrained FP fixed. Subjectivity classification generalized perfectly
(1.00) to NEW topics it was never anchored on.

## Honest scope — the bound is real and measured

- **Clean opinion-vs-fact only.** P5's 1.00 is on prompts that are clearly
  subjective or clearly factual. Ambiguous/mixed real-world prompts are not in the
  holdout and would score lower; the 1.00 is not a claim of universal subjectivity
  detection.
- **Decoupled-diagonal ceiling (pre-declared, measured):** a false premise in a
  factual frame ("Is it true that the Sun orbits the Earth?") is classified
  factual → neutralized → any sycophantic agreement would be missed. In practice
  this barely triggers (C5 fire-rate on the decoupled subclass 0.17) because models
  *correct* known-false premises (counter-vocabulary → non-sycophantic). But the
  limit stands: **prompt FORM ≠ premise TRUTH.** Full truth-grounding (fact-checking
  the agreed proposition) is a separate, harder problem this does not solve.
- **Optional tier, not the core.** C5 requires sentence-transformers, so it cannot
  be the pure-Python / Pyodide default. It ships as an **opt-in semantic tier**
  (mirroring grounded deception), with the v0.2 + self-directed gate remaining the
  zero-dependency default.

## Why this is the genuine advance (and what it is not)

It is **not** "sycophancy solved." It **is** the first time a styxx register
instrument crosses a documented text-only ceiling — the opinion-vs-fact
distinction that C3/C4 proved surface features cannot make — by adding a small,
local, offline **semantic** signal. The pre-registered generalization bar (P5),
which the lexical approach failed (0.73) and the semantic approach passed (1.00)
on fresh data, is the evidence that it is real and not a template artifact.

## Artifacts

`target_gate_c5.py` (frozen anchors), `gen_holdout_semantic.py` (new topics,
two-stage varied), `run_killgate_semantic.py`, `results_semantic.json`.
Chain: prereg `4e99ad0` → lock `863160a` → this result.
