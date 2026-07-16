# Blind adversarial self-review — sentiment probe-parity harness + prereg (run 2026-07-09, pre-result)

Reviewed WHILE the scored run trains, blind to its verdict. Purpose: catch instrument bugs that would
bias the privacy-vs-capacity decomposition before the number is read.

## Does the parity comparison cleanly isolate privacy? — YES

`private` and `parity-naive` both call the SAME `attack_sentiment.private_audit` (DoM + per-layer
logistic + whole-stack concatenated logistic), both fit on n=|CALIB|=110, both read EVAL. They differ
in ONE variable: the fit data is the clean CALIB split (private) vs a poisoned ATTACK subsample
(parity-naive). Probe family and fit size are matched. So `parity_gap = private − parity-naive` is a
clean privacy isolation. This is the verdict metric. GOOD.

## Findings (severity-ordered)

1. **[minor, disclose] Absolute AUROCs are selection-optimistic; the VERDICT metric is not.**
   `private_audit` returns the max EVAL AUROC over 13 probe options (6 DoM + 6 logistic + whole-stack),
   selected on EVAL — an upward bias. But `private` and `parity-naive` select from the SAME 13 options
   on the SAME EVAL, so the bias CANCELS in `parity_gap`. `naive-DoM` selects from only 6 (less bias),
   so `baseline_gap = private − naive-DoM` is slightly inflated by the larger selection set — but
   `baseline_gap` is only a reference (the 0.5× term); the verdict rides `parity_gap`, which is clean.
   Net: absolute numbers optimistic, the privacy decomposition unbiased.

2. **[minor, disclose] The ATTACK subsample for parity-naive is random, not class-stratified.**
   n=110 drawn from a class-balanced ATTACK of 220 (seed-locked, rng 12345). Expected ~55/55 with
   small variance; a mild imbalance could slightly penalize parity-naive and thus slightly INFLATE
   `parity_gap` toward PRIVACY. This biases AGAINST the CAPACITY_DOMINATED verdict, i.e. it is a
   conservative bias for anyone hoping to preserve the privacy framing — worth noting because if the
   verdict still lands CAPACITY_DOMINATED despite this, the conclusion is robust.

3. **[none] Bands are non-overlapping** (parity_gap ≤0.03 CAPACITY; 0.03–0.06 or fails-half PARTIAL;
   ≥0.06 and ≥0.5×baseline PRIVACY). No value is double-classified.

4. **[none] Behavior-preserving filter and the baseline-gap-exists guard** are consistent with prior
   stages; a non-poisoning attack VOIDs rather than misreads.

## Verdict on the instrument

Sound for its purpose. Two minor biases disclosed; both leave the `parity_gap` verdict metric clean or
conservative. No fatal or major flaws. The RESULT note must carry findings 1 and 2 verbatim.
