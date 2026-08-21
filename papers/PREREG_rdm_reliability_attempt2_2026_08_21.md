# PREREGISTRATION 2 of 2 — representational reliability, at the answer position

**Frozen 2026-08-21, after attempt 1 was published INVALID and before attempt 2
was run.**

## why there is a second attempt

Attempt 1 (`PREREG_rdm_reliability_error_predictor_2026_08_21.md`) tripped **G4**:
reliability IQR 0.0197 against a 0.02 floor written before any data existed. It
is published as `INVALID__PRECONDITION`, not as a null, and no AUC was quoted.

The diagnosis is mechanical rather than convenient: PopQA prompts are 43-59
tokens and every one carries the same chat template and the same instruction
suffix. **Mean-pooling over prompt tokens is therefore dominated by tokens that
are identical across items**, so any two halves of the feature space agree ~0.97
for every item alike. The measure had no variance to predict with.

## the single change

Representation is taken at the **final prompt token** — the position from which
the answer is actually generated — instead of mean-pooled over all prompt
tokens. That is the standard probe location, and it is where item-specific
computation concentrates.

**Nothing else moves.** Same model, same 500 items, same seed, same layer, same
20 splits, same gates, same grading. One variable.

## discipline on multiple attempts

This is **attempt 2 of a planned 2**. If G4 trips again, or G1 returns NOT
SUPPORTED, the hypothesis is reported as failed at this model scale and **there
is no attempt 3 without a new mechanism** — not a new layer, not a new pooling,
not a new subset. Running variants until one clears is how a program fabricates
a finding, and both attempts are on the record precisely so that the count is
visible to anyone reading the result.

## gates — unchanged from attempt 1

- **G1 PRIMARY** — delta-AUC(baseline + reliability) over AUC(baseline alone),
  5-fold out-of-fold, 95% bootstrap CI over 2000 resamples. CI includes 0 ->
  NOT SUPPORTED.
- **G2 CONFOUND** — partial Spearman controlling prompt length and log subject
  popularity must keep sign and p < 0.05.
- **G3 VALIDITY** — accuracy outside [0.10, 0.90] -> INVALID.
- **G4 SANITY** — IQR(reliability) < 0.02 -> INVALID.

## known power limitation, stated in advance

Attempt 1 measured accuracy 0.116 — about 58 positives in 500. That is thin for
a delta-AUC bootstrap, and the CI is expected to be wide. **A wide CI that
includes zero is a power statement, not evidence of absence**, and will be
reported as such rather than as a clean null. N was deliberately not raised here
because changing two things at once would make the comparison to attempt 1
uninterpretable.
