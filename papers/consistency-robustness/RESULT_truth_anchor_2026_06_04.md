# RESULT — the external-truth anchor breaks the knowledge floor; the stack closes on itself

**Date:** 2026-06-04 · **Verdict: ANCHOR BREAKS THE FLOOR.** Frozen:
`PREREG_truth_anchor_2026_06_04.md`. Qwen2.5-3B-Instruct, 100 SQuAD-v2 answerable items.

## Numbers (n=100)
- Accuracy: **closed-book 0.170 → open-book 0.910** (P1: +0.74, far past +0.20).
- **Floor correction: 89.2%** — external trusted context fixes 89% of the closed-book knowledge floor
  that no self / peer / derivation readout could touch (P2: ≥0.50).
- Residual (wrong even with context): **9.0%** — extraction/reasoning failure even with truth present.

## What it means
The knowledge sub-floor — a stable false belief / gap, coherent to every self-consistency method — is
**broken by the only readout that reads OUTSIDE the model.** This is the framework's final, missing
piece: when consistency bottoms out, *external truth* is what's left, and it works (89%).

**The stack closes on itself.** Retrieval grounds out the floor — but it *relocates* the trust
commitment to the **source**. A poisoned context re-opens the floor (RAG poisoning); that attack is
exactly the cross-context divergence `detect_context_injection` flags (validated **91%** this arc). So:
*retrieval breaks the floor; the consistency layers guard the retrieval.* The six readouts cover each
other AND ground out in truth.

## The complete readout stack (the arc's contribution)
1. activation probe — deception representation
2. grounded_honesty — stateless belief
3. detect_context_injection — cross-context divergence
4. council_agreement — cross-model agreement
5. derivation-consistency — derived vs retrieved (regime-gated)
6. **external-truth anchor — reads outside the model; breaks the knowledge floor (89%)**

## The irreducible commitment (honest, and RSI-relevant)
The anchor does NOT make honesty free. It moves the irreducible commitment to **the trustworthiness of
the source**. There is no consistency-only path to truth — at the bottom, an integrity monitor must
*trust something external*. When AI controls or influences its own sources, the floor returns. That is
the real, unromantic foundation of AI-monitoring-AI: honesty grounds out in a trusted anchor, and
securing that anchor is the problem that does not dissolve.

## Honest scope
One model, SQuAD extractive QA, contains-scoring. The 89% is a clean demonstration of the *mechanism*
(external context corrects closed-book error), not a deployment number. The poisoning/guard closure
reuses the already-validated `detect_context_injection` result rather than a new poison experiment.
