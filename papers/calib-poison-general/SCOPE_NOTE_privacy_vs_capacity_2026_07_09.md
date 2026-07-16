# SCOPE NOTE (DRAFT — operator sign-off required) — the private-calibration recovery is majority probe capacity

**Fathom Lab · 2026-07-09. Drafted from `RESULT_sentiment_probe_parity_2026_07_09.md` (OATH-HELD).
This note is NOT applied to any published surface; it is the proposed correction for operator
decision. Nothing here retracts a result — the phenomenon stands; the attribution is refined.**

## What the published surfaces currently say

- **read-neq-write / "Calibration Poisoning, Not Erasure" (Fathom v28, concept DOI
  10.5281/zenodo.19326174):** the apparent evasion is calibration poisoning, defeated by re-fitting
  the probe on a PRIVATE split the attacker never saw. Framed as a *privacy* effect.
- **`styxx.mount.ConscienceMount.relock`** (shipped): the calibration-poisoning defense, described as
  re-fitting on private clean data.

Both attribute the recovery to **privacy** (calibrating on data the attacker did not see).

## What the parity control shows

The private auditor in those results uses a richer probe family (difference-of-means + logistic +
whole-stack) fit on a different-sized clean split than the naive difference-of-means auditor it is
compared against. Controlling for probe family AND fit size (the poisoned auditor given the same rich
probe on an equal-sized subsample of the attacker's own split) decomposes the gap:

- On sentiment: median gap 0.127 → **~two-thirds probe capacity, ~one-third privacy** (residual
  privacy effect median 0.041, positive in all four cells but below the pre-registered "privacy
  stands" bar). Verdict PARTIAL.
- (The accumulated record reports the same direction on honesty-side parity checks: capacity-dominant
  with a small, seed-dependent privacy residual. This note's own new evidence is the sentiment run.)

## Proposed correction (scope note, not retraction)

Add to both surfaces, in substance:

> *The recovery attributed to private calibration is, at matched probe capacity and fit size,
> majority probe capacity and minority privacy. Re-fitting on a private split does help (a real,
> positive residual), but the larger factor is using a richer probe family (difference-of-means +
> logistic + whole-stack) rather than difference-of-means alone. The operative defense is "use a
> richer probe AND a private split," not "a private split" alone. The privacy component is real but
> minor on the constructs tested (1–1.5B, linear probes).*

## What does NOT change

- Calibration poisoning is real (naive-DoM is genuinely dragged down by the attack).
- The private-calibration read does recover through the attack (both capacity and privacy contribute).
- The read≠write / erasure-vs-poisoning distinction is unaffected — this note is about *why* the
  re-lock recovers, not *whether* it does.

## Operator decisions

1. Whether to deposit corrected paper versions (Zenodo new-version) with the scope note, or attach it
   as an erratum-style addendum.
2. Whether to amend the `mount.relock` docstring's scope paragraph to state the capacity-majority
   finding.
3. Owed before finalizing: a matched honesty-side parity check in this exact harness family to confirm
   the decomposition on the flagship's own construct (the accumulated record indicates it converges).

Agent will not edit published records or the shipped docstring's claims without an explicit go.
