# Finding — the anchor threshold, shipped as a styxx instrument (and a favourable correction)

**Date:** 2026-07-23
**Module:** `styxx.anchors` — new functions `anchor_lr`, `blindspot_power`, `min_anchors_for_power`
**Receipt:** `anchor_power_result.json` (this directory)
**Cross-checked against:** `anchor_threshold_result.json` (the frozen section-7 exploratory receipt)

## What shipped

The paper's anchor-threshold result answered a design-time question the instrument's own datasheet
raises: `audit_panel` can only price a shared, truth-independent, all-judge blind spot when enough
anchors are present, so *how many known-negative anchors do you need before "no detection" means
something?* That answer was, until now, a one-off exploratory script. It is now three first-class,
closed-form, dependency-light functions in `styxx.anchors`:

- `anchor_lr(...)` — the likelihood ratio of a single known-negative that every judge mis-votes
  "positive", blind-spot vs benign.
- `blindspot_power(K, ...)` — exact power to detect the blind spot from `K` known-negative anchors,
  via the count of unanimous-wrong anchors, using the standard most-powerful one-sided binomial test.
- `min_anchors_for_power(p, ...)` — the smallest anchor budget that first reaches target power `p`.

All numbers below are bound to `anchor_power_result.json` and re-derivable with
`python papers/anchored-validity/run_anchor_power.py`.

## The design point (J=3 correlated weak panel)

Three judges; per-judge false-positive rate on a known-negative 0.1 under independence; 15% of
true negatives are shared traps (all judges wrong); non-trap negatives at 0.0961; level 0.05.

single-anchor likelihood ratio: 150.8 — matched to the frozen receipt (continuity with the
published section). One unanimous-wrong known-negative is, by itself, strong evidence of a
synchronized failure that consensus estimators cannot see at any sample size.

Power from `K` known-negative anchors (the shipped tight test), receipt `anchor_power_result.json`:

| K anchors | 1 | 3 | 5 | 10 | 20 | 30 | 50 |
|---|---|---|---|---|---|---|---|
| power | 0.1508 | 0.3875 | 0.5583 | 0.8049 | 0.9619 | 0.9926 | 0.9997 |

minimum known-negative anchors for 0.90 power: 15.
minimum known-negative anchors for 0.95 power: 19.

## The favourable correction (why the shipped numbers beat the exploratory table)

The exploratory receipt set its rejection threshold at the smallest count `c` with
`P(X >= c | benign) <= alpha`, then measured power over the *stricter* region `X > c`. That
controls the type-I rate identically but rejects on a smaller region, so its power column is a
valid **lower bound**, not the achievable power. The shipped instrument uses the standard
most-powerful region `X >= c`.

The consequence is entirely in the safe direction. Comparing the two receipts, the tight power is
greater-than-or-equal-to the conservative power at every `K` (`correction_favourable_all_K` true),
so the published claim does not overclaim — it was conservative. Two concrete effects:

- The paper's "~20-30 anchors for >90% power" tightens to 15 anchors for 0.90 power under the
  standard test.
- The exploratory table reported single-anchor power 0.0, which contradicted its own
  single-anchor-is-a-smoking-gun headline; the shipped single-anchor power is 0.1508, consistent
  with the 150.8 likelihood ratio.

Conservative receipt for the same design point (`power_conservative_by_K_section7`), for the
record: 0.0 / 0.0613 / 0.1662 / 0.4585 / 0.8267 / 0.953 / 0.9972 at K = 1 / 3 / 5 / 10 / 20 / 30 / 50.

## Scope (unchanged from the module datasheet)

The count-of-unanimous-wrong probe detects an **all-judge, truth-independent** blind spot — the
exact failure the identification result proves consensus is blind to. A blind spot correlated with
the true label still defeats it, and anchor exchangeability remains load-bearing: these functions
size a detector, they do not relax the assumption. `blindspot_power` reports the achieved type-I
rate (`alpha_actual`) alongside power so the test it describes is always fully specified.

## Status

Instrument added; six behavioral tests pass (single-anchor LR pinned to the frozen receipt, tight
power dominates the conservative bound at every K, the 0.90/0.95 budgets, monotonicity in effect
size and in the target, and input validation). Ships in the next `styxx.anchors` release; no
existing behavior changed.
