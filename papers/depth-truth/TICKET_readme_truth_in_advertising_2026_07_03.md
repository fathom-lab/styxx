# §10 TICKET — depth-truth README truth-in-advertising audit — `DISCHARGED (no correction needed)`

**Fathom Lab · papers/depth-truth · 2026-07-03 · autopilot cycle 21.**
**Opened by:** the cycle-20 CLOSED_NEGATIVE (`FINDING_depth_does_not_predict_truth_2026_07_03.md`),
whose PREREG_v2 §10 says: *"H1 null AND H2 null ⇒ … a truth-in-advertising ticket opens against the
README headline."* This note is the audit that discharges it.

## Question

The keystone verdict falsified "circuit-attribution depth predicts whether the model's answer is
correct" (H1/H2/H3 all null). §10 obligates a check: **does any live, public-facing styxx claim
advertise depth as a truth / correctness / hallucination predictor?** If so, it must be corrected or
scoped to "separates recall from reasoning."

## Audit (exhaustive over the repo)

| surface | searched for | result |
|---------|--------------|--------|
| `README.md` | "measure thought, not words", depth→truth/correct, circuit-depth→hallucination | **none.** The README's hallucination numbers (HaluEval, TruthfulQA, `@trust`) belong to the *text-heuristic* guardrail / cognometry instrument — a different system that never invokes `get_mean_depth`. |
| `web/` (verify.html, styxx_verify.js) | same | **none.** |
| `docs/**` | same | **none** relevant. Sole near-hit `docs/gate.md` describes the *refuse-check* class predictor (refuse/confabulate/answer), not attribution depth. |
| `papers/**` (non depth-truth) | "measure thought not words", "surface recall / explanatory reasoning" as a live headline | **none.** The phrase exists *only* as a hypothesis label inside `PREREG_v2.md`. |
| adjacent live depth findings | do they overclaim? | **already honest.** `grounded-honesty-axis/FINDING_depth_steering_causal_2026_05_29.md` headlines the construction↔retrieval axis as **"correctness-INERT"**; `FINDING_depth_grounding_whitebox_2026_05_29.md` frames depth as a grounding ("belief→truth dial") substrate, carefully scoped. Both are consistent with, not contradicted by, the cycle-20 negative. |
| `research/` (the `get_mean_depth` origin) | the d=0.82 recall-vs-reasoning claim | **not in this repo** — it lives in the separate research git (PREREG §1 pins it at `fc6f2c3`), and was described as a *pending* finding, never shipped in styxx. |

## Verdict — `DISCHARGED`, no correction shipped

There is **no live public styxx claim that circuit-attribution depth predicts truth, correctness, or
hallucination.** The prereg-before-claim discipline did its job: the "depth measures thought, not
words" idea was carried as an explicit hypothesis under test, never advertised as a result. The
falsification therefore requires **no retraction** — the claim was never made in public copy. This is
the honest, quiet outcome §10 was written to force a check on, and the check passes.

## Owed / watch-items (outside autopilot's repo + write scope)

1. **External ICML-track attribution-depth writeup** (operator memory: length-partial effect
   d=0.82→1.57). It lives in the research git, not here. **If** that manuscript states or implies
   depth predicts *answer correctness* (as opposed to separating *kinds of processing*: recall vs
   reasoning), it needs the cycle-20 caveat: on gemma-2-2b short-form QA, first-content-token
   attribution depth is near-constant and carries **no** correctness signal (AUROC 0.5468, CI
   straddling chance). Operator-gated: an external paper is not autopilot's to edit.
2. Nothing in the styxx repo requires an edit. If future copy is written that claims depth flags
   hallucination, this ticket's finding is the receipt that it must not.

## Receipts

- `FINDING_depth_does_not_predict_truth_2026_07_03.md` (+ `.certificate.json`) — the negative that
  opened this ticket.
- `results/verdict.json` — H1 AUROC 0.5468 and the null battery.
- This audit is a grep sweep over `README.md`, `web/`, `docs/`, `papers/` at repo HEAD; re-runnable
  by the same searches (the table names every surface).

---

*A falsification that needs no public retraction because the discipline never let the claim ship —
that is the apparatus working as designed.*
