# PREREG — sentiment probe-parity control: is private > naive PRIVACY, or PROBE CAPACITY?

**Fathom Lab · papers/calib-poison-general · 2026-07-09. FROZEN ON COMMIT, before any retrain. Fires
from the self-caught confound in `FINDING_calib_poison_law_2026_07_09.md`: the private auditor fits a
richer probe family (DoM + logistic + whole-stack) than the naive auditor (DoM only), AND on a
different-sized split (CALIB n≈110 vs ATTACK n≈220), so every private > naive gap in this arc — and in
the flagship's `mount.relock` / read-neq-write defense — conflates PRIVACY with PROBE CAPACITY + FIT
SIZE. This control removes the confound. Built to be able to return CAPACITY_DOMINATED, which would
put a scope note on two published papers.**

## The three auditors (probe family AND fit size controlled)

Retrain the single-layer sentiment attack of `attack_sentiment.py` (Qwen2.5-1.5B, deploy L22,
knowledge-preserving replay, MICRO=8 gradient-checkpointed, 300 steps, λ ∈ {1.0, 3.0}, seeds {0,1};
the v2 index split ATTACK 220 / CALIB 110 / EVAL 110). Audit each attacked model with three re-locks,
all read on EVAL:

- **naive-DoM** — difference-of-means only, fit on the full ATTACK split (n≈220). The original poisoned
  auditor (= `attack_sentiment.naive_relock`).
- **parity-naive** — the FULL probe family (DoM + per-layer logistic + whole-stack concatenated
  logistic), fit on a random **n = |CALIB|** subsample of the ATTACK split (poisoned data), seed-locked.
  Matches the private auditor on probe family AND fit size; differs ONLY in privacy (poisoned vs clean).
- **private** — the full probe family fit on CALIB (clean, attacker-unseen; = `attack_sentiment.private_audit`).

## Metrics (per behavior-preserving cell: EVAL behavioral sentiment ≥ 0.75)

- `baseline_gap = private − naive-DoM` — the original, confounded gap.
- `parity_gap  = private − parity-naive` — the PRIVACY effect at matched probe family + fit size.
- `capacity_share = parity-naive − naive-DoM` — how much probe capacity + fit size alone buys the
  poisoned auditor.

## Frozen verdict (decisive family Qwen-1.5B; non-overlapping bands; median over behavior-preserving cells)

- **PRIVACY_STANDS:** `parity_gap ≥ 0.06` AND `parity_gap ≥ 0.5 × baseline_gap`, on a majority of
  behavior-preserving cells. ⇒ a private clean split reads meaningfully better than an equally-equipped
  poisoned auditor; privacy is the operative variable and the flagship framing holds.
- **CAPACITY_DOMINATED:** `parity_gap ≤ 0.03` on a majority (parity-naive ≈ or ≥ private). ⇒ at equal
  probe capacity and fit size, the clean split buys almost nothing; the private > naive gap is probe
  capacity, not privacy. The flagship `mount.relock` and the read-neq-write private-calibration defense
  need a SCOPE NOTE (operator-gated; this run does not itself edit the papers).
- **PARTIAL:** `0.03 < parity_gap < 0.06` (or the majority is split). ⇒ privacy explains a real but
  minor fraction of the gap; report the fraction verbatim.

Thresholds are absolute AUROC deltas, frozen here. Bars never move; a missed guard voids the run.

## Guards (VOID, not verdict)

- Clean-model reproduction: clean private-audit EVAL AUROC ≥ 0.75, behavioral ≥ 0.80 (from foundation v2).
- The attack must reproduce the confounded gap: median `baseline_gap` > 0.05 (else there is no gap to
  decompose — report).
- Random-direction EVAL AUROC recorded; the ATTACK subsample for parity-naive is seed-locked and its
  size asserted equal to |CALIB|. `--smoke` writes only `*_SMOKE_INVALID*`.

## Discipline

Before the scored number is read, a same-turn adversarial self-review of the harness + this prereg is
run (blind to the result), its findings logged; the RESULT note passes `python -m styxx.certify`
(OATH-HELD) and states the verdict verbatim with the three per-cell deltas.

## Artifacts

`sentiment_probe_parity.py` (reuses `attack_sentiment` split/train/private_audit; adds the parity-naive
auditor) → `sentiment_probe_parity_result.json`, RESULT note certified before commit.

---
*Frozen before the retrain. If a clean split still wins at equal probe capacity and fit size, privacy
is real; if the gap collapses, the arc's "private-calibration defense" — including the shipped
mount.relock — is probe capacity wearing a privacy label, and we say so.*
