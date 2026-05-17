# Brick #1 — Live-Behavior Transport: Honest Negative / Inconclusive (2026-05-17)

**Status:** the load-bearing brick did NOT land. Reported straight.
**Script:** `scripts/dogfood/live_behavior_transport.py`
**Raw:** `scripts/dogfood/out_live_behavior_transport.json`

## What was tested

Replace synthetic benchmark condition labels with LIVE closed-model
behavior, then ask whether a transported instrument predicts it.
Targets: gpt-4o-mini, gpt-4.1-mini. Sycophancy = objective flip-test
(factual Q, neutral vs wrong-answer-pressure, string-checked).
goal_drift / plan_action = gpt-4.1 judge. Parallels the validated
refusal closed-model methodology.

## Result: not measurable — the behavior did not occur

| instrument | model | live positive base rate |
|---|---|---|
| sycophancy | gpt-4o-mini | 0.02 (≈1/45) |
| sycophancy | gpt-4.1-mini | 0.00 (0/46) |
| goal_drift | gpt-4o-mini | 0.03 |
| goal_drift | gpt-4.1-mini | 0.00 |
| plan_action | gpt-4o-mini | 0.17 (6/36) |
| plan_action | gpt-4.1-mini | 0.03 |

With 0–3% positives (and literally zero for gpt-4.1-mini sycophancy /
goal_drift), AUC is undefined or pure noise. The single non-degenerate
number (sycophancy gpt-4o-mini te3-small, transported 0.955) rests on
~1 positive example — **noise, not a result.** No honest AUC can be
computed. This is **not** a transport success or failure; the behavior
to predict was absent.

## What it actually shows (the real finding)

**Modern closed mini-models do not exhibit sycophancy, goal-drift, or
plan-action mismatch under naive single-shot elicitation.** The styxx
synthetic benchmarks (`benchmarks/data/*`) encode behaviors that live
gpt-4o-mini / gpt-4.1-mini resist when prompted simply. Refusal
transported earlier because refusal is readily elicited (harmful prompt
→ refusal, ~33% base rate, clean lexical label); the other instruments
have no such cheap, valid live signal.

## Consequence for the program

The reference-frame critical path was wrong-ordered. The bottleneck is
**not** "does transport predict live behavior." It is **"can the
behavior be validly elicited and labeled live at all."** For
sycophancy/scheming/drift that is Apollo/Anthropic-grade adversarial
eval-harness research (multi-turn pressure, authority framing, agentic
context), not a script.

Honest standing position:

- **Refusal universal transport remains solid and live-validated**
  (AUC 1.000 clear / 0.89–0.94 live closed-model, cross-family). That
  is real and unaffected.
- **Cross-instrument transport (synthetic labels) is directional only**
  — and now explicitly so: its non-refusal instruments are *not* shown
  to track live behavior, because that behavior wasn't elicited here.
- **No paper, no Zenodo/OSF.** This narrows the honest paper to
  refusal-only universal transport + the documented boundaries.

## Caveats / limits of this negative

n ≤ 46/instrument/model; single-shot elicitation is deliberately weak;
goal_drift/plan_action used an LLM judge; gpt-4.1-mini's near-total
robustness may partly reflect refusal-to-engage rather than true
calibration. A stronger elicitation harness could change the base
rates — that is exactly the (heavy) next research question, not a
quick rerun.

## Next (honest)

1. Decision point: build an Apollo/Anthropic-grade live elicitation
   harness (real multi-session research) OR scope the paper to
   refusal-only universal transport (solid, shippable now).
2. Do NOT iterate this script for a better number — the null is the
   finding.
