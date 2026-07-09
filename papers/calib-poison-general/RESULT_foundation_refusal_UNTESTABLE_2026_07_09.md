# RESULT — refusal foundation: UNTESTABLE_ON_THIS_DATA. The generalization test cannot run on this construct.

**Fathom Lab · papers/calib-poison-general · 2026-07-09. Verdict against the frozen Stage-1 gate of
`PREREG_calib_poison_generalizes_2026_07_09.md` (committed before the run). Fires the UNTESTABLE
branch: the refusal instrument does not meet the pre-registered validity bars, so no attack is run
and NOTHING is claimed about generalization. A null on the instrument, not on the claim.**

## Verdict: UNTESTABLE_ON_THIS_DATA — the attack premise does not hold on refusal at this scale

The generalization test needs two things to exist before an attack can be meaningful: a genuine,
graded refusal READ to scrub, and a behavioral refusal JUDGMENT for the attack's replay term to
preserve. On the frozen refusal pairs (92 items, 8 families, family-disjoint 44/24/24 split), neither
model provides both.

| model | refusal read EVAL AUROC | random-dir floor | behavioral refusal acc | shuffled acc | valid |
|---|---|---|---|---|---|
| Qwen2.5-1.5B | 1.000 (all layers, incl. L6) | 0.049 | **0.667** | 0.583 | NO |
| Llama-3.2-1B | 1.000 (all layers, incl. L6) | 0.694 | **0.500** | 0.500 | NO |

Guard 2 (behavioral refusal judgment ≥ 0.80) FAILS on both models, and guard 4 (chance floors) also
fails. Two diagnostics, reported plainly:

1. **The read is trivially separable, not a graded representation.** AUROC is 1.000 at *every* layer,
   including the earliest scanned (L6), and random directions score at the extremes (0.049 / 0.694)
   rather than ~0.5. Perfect early-layer separability plus extreme random-direction AUROC is the
   signature of a low-dimensional, likely *lexical* signal (harmful requests carry overt harmful
   tokens), not the smoothly graded representational axis the honesty construct provides. There is no
   subtle signal to poison and recover here — the classes are surface-distinguishable.
2. **The behavioral judgment is near-chance.** Asked directly "is this request harmful? yes/no,"
   Qwen scores 0.667 and Llama 0.500 (pure chance) on EVAL. The capability analog of "knowledge" that
   the attack's replay term must preserve barely exists on Llama and is weak on Qwen — so a
   "knowledge-preserving" attack on refusal has essentially no capability to preserve, and the whole
   attack/defense contrast would be uninterpretable.

## What this licenses, and what it does NOT

- **Does NOT license** any claim that calibration poisoning fails to generalize. We did not test it;
  the *instrument* did not clear the validity bars. NO_GENERALIZATION is explicitly not the verdict.
- **Does NOT license** "calibration poisoning is honesty-specific." Same reason.
- **Licenses only:** on this particular refusal dataset (92 lexically-marked intent requests) and
  these 1–1.5B instruct models, the refusal read is trivially separable and the behavioral refusal
  judgment is too weak to support the attack-and-defense test. The generalization question remains
  open, pending a construct that clears the bars.

## The forced next step (a NEW frozen prereg, not a goalpost move)

The prereg's UNTESTABLE branch forbids proceeding to the attack and forbids re-tuning the bars to
pass. To test generalization honestly, a follow-up (frozen separately) needs a *second construct that
behaves like honesty*: a graded read that is NOT trivially lexical (AUROC < 1.0, rising with depth,
random floor ~0.5) AND a behavioral judgment the model actually has (≥ 0.80). Candidates: a
sycophancy/agreement construct with matched-surface pairs, or a larger model where refusal judgment
is real and the harmful/benign boundary is subtle rather than lexical. Choosing and freezing that is
the next cycle; it is not this run with the bars moved.

## Bounds

92 refusal items, 8 domain families, family-disjoint 44/24/24, small EVAL (24), `Qwen2.5-1.5B` +
`Llama-3.2-1B`, DoM read, last-request-token pre-output regime (no harmful content generated). The
finding is "this construct/data cannot support the test," not "refusal has no substrate signal" —
indeed the signal is present but too trivially separable to be the honesty analog.

## Reproducibility

`foundation_refusal.py` (deterministic, forward passes only) → `foundation_refusal_result.json`.
Reuses `run_says_yes_knows_no.py` residual/behavioral math and the frozen refusal pairs from
`run_portable_values_refusal.py`. Prereg frozen before the run.

---
*UNTESTABLE, and that is the gate working: rather than run an attack on a lexically-trivial read with
no real capability to preserve — which would have produced a meaningless "result" — the pre-committed
guard stopped the program and named what a valid second construct must look like.*
