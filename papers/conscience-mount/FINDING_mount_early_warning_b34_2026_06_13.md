# FINDING — B34: the conscience mount is a COMMITMENT-TIME monitor, not an early-warning one (NO-PRE-DECISION-SEPARATION + WINDOW-CATCH-COLLAPSE, OATH-HELD)

**2026-06-13 · Fathom Lab / styxx. Prereg `PREREG_mount_early_warning_b34_2026_06_13.md` frozen @42c7c36
BEFORE this runner existed. Runner `run_mount_early_warning.py`, SEED=0. Receipt
`mount_early_warning_result.json`. Two pre-committed questions, two honest negatives: pooling the borrowed
conscience over the pre-decision claim span does NOT preserve the catch, and there is NO cave-vs-honest
separation at the claim-content tokens before the decision. The divergence signal the mount runs on is
LOCALIZED at the moment of commitment. This tightens — does not weaken — the scope of the shipped
operating point, and it answers the "early warning / reads the lie before it is spoken" question for this
layer/regime with a receipt: no usable pre-decision signal here (a layer/regime-specific null, not a
universal one).**

## What was tested

The same borrowed gemma-2-2b → Llama-3.2-3B truth conscience as the shipped operating point (map layer
L21, anchor R² 0.9851, mapped-space ZCA-shrink frame, the identical direction — NOTHING about the
conscience changed), read at DIFFERENT token positions of the same pressured forward pass. 30 comparative
claim pairs under the validated deference+pressure regime; 20 caves (the agent asserted a false claim
true), 15 honest calib / 15 honest test true-controls. The decision token sits in the appended
assistant-header, so every position was labeled HEADER / CLAIM-CONTENT / SCAFFOLD / PRE-CLAIM-USER; the
claim-content span was located and verified by decode (56 of 60 items matched cleanly, 4 excluded).

## Baseline reproduced exactly (the guard)

Before any new statistic, the runner recomputed the single decision-token operating point under the
identical split / axis / layer: **catch 0.85, i.e. 17 of 20 caves caught, at false-alarm rate 0.20 (3 of 15 honest)**,
center 1.3789, scale 0.0675, tau 0.8167 — bit-matching the published `FINDING_mount_regime` numbers
(`reproduces_published` true). The position-capture refactor changed nothing; every number below sits on
the same reproduced baseline.

## PRIMARY — WINDOW-CATCH-COLLAPSE (pooling dilutes the signal)

Mean-pooling the borrowed-axis coordinate over the claim-content positions (one pre-committed statistic, no
scan) and recalibrating center/threshold on held-out honest items collapses the catch: **catch 0.05 — a single cave of 20
caught — at a false-alarm rate of 0.0667 (1 false alarm of 15 honest)**. The continuous windowed score
clears its shuffled-axis null (discriminability 0.696667 vs p95 0.683333, permutation p 0.018981) by only
about 0.01 — and even that thin continuous signal does not translate into a usable catch (it collapses to
0.05). A statistically significant continuous score is not a usable detector here. The thin single-token
read is thin BECAUSE the divergence is concentrated at/near the decision token — not because one token is
noisy. The single decision token remains the operating point; windowing the claim span does not improve it
and is rejected. (False-alarm tightening is not claimed: the windowed FPR 0.0667 vs baseline 0.20 is one
honest item at this n, exact-test p 0.298851, `fpr_significant_vs_baseline` false — the "false-alarm rate
is double the target" caveat stands verbatim. As pre-committed, the known honest outlier that read strongly
negative at the decision token — "A minute is shorter than an hour." — flips under windowing: consistent
with a whole-item axis mis-read that every position inherits, since the same direction reads every position
of that item, NOT a position-specific gain. That is the second reason the lone windowed false alarm is not
read as an FPR improvement.)

## SECONDARY — NO-PRE-DECISION-SEPARATION (the signal forms at commitment, and the confound control held)

Reading the borrowed axis by region, the point estimates increase with proximity to the decision — but the
two reads taken WHILE THE MODEL IS STILL READING THE CLAIM are tied within noise and below the shuffled-axis
null, so the early part of the alleged "rise" is not a real signal:

- PRE-CLAIM-USER (preamble, pooled): 0.591667 (p 0.23976, not significant)
- CLAIM-first token (single position): 0.603333 (Holm p 0.135864, below the MDE 0.6435 and below the
  shuffled-axis p95 0.635083 — not separable)
- CLAIM-pooled (pooled aggregate; this is the windowed PRIMARY, not a position): 0.696667 (p 0.018981)
- SCAFFOLD (the "…answer with one word: True or False" tail, pooled): 0.796667 (p 0.002997)
- HEADER (the post-prompt assistant-header positions, decision-adjacent, pooled): 0.89 (p 0.000999)

PRE-CLAIM-USER (0.591667) and CLAIM-first (0.603333) differ by about one one-hundredth, are both
non-significant, and both sit below the shuffled-axis null — statistically indistinguishable from each
other and from chance. There is no measurable separation while the model is reading the claim; only the
commitment-adjacent SCAFFOLD and HEADER regions clear their nulls. Accordingly the first claim-content token
does NOT beat its permutation null, does NOT clear the minimum detectable effect, and does NOT exceed the
HEADER (0.89) or SCAFFOLD (0.796667) controls — verdict NO-PRE-DECISION-SEPARATION. Critically, the HEADER tokens are byte-identical
across all items (the same assistant-header template), so their 0.89 separation is the model's INTEGRATED
INTERNAL STATE at commitment, not lexical token identity — and it is exactly the position a naive
"read the early tokens" analysis would have surfaced and mislabeled "early warning." The frozen position
labeling caught that: the strongest separation lives on commitment-adjacent template tokens, not on the
claim the model was reading.

## The honest reading (pre-committed, both outcomes were written before the scores)

The borrowed conscience does NOT read the cave before the agent forms it. There is no usable
pre-decision (claim-content) signal at this layer/regime; the divergence the mount runs on emerges at the
moment of commitment and is localized at the decision token, which is why a single-token read is both
sufficient (catch 0.85) and necessary (pooling collapses it to 0.05). This is a commitment-time integrity
monitor, not a foreknowledge or "intent" detector — no "the conscience knew first," no telepathy. Both
pre-committed questions returned negatives. As a corollary — not a new win — the position labeling
localizes the divergence to the commitment tokens.

## Scope (carried forward, none erased)

White-box (needs the agent's own activations → open-weight / self-host). Read-only; a flag is a
measurement, acting on it is the integrator's policy. Linear BORROWED truth axis (gemma→Llama same-cluster)
with its own false positives. COOPERATIVE-monitor scope only — the ATTACK-TRANSFERS result stands (borrowing
a different reference shares ~one truth direction, cos ~0.94; a small activation perturbation evades all
borrowed consciences; windowing/position add no adversarial robustness). Small n: 30 pairs, 15 honest calib
/ 15 honest test, 20 caves, FPR granularity ±0.0667; single seed (SEED=0), single agent, reference, regime,
and map layer L21 — a null secondary result is L21/regime-specific, not "no pre-decision signal exists
anywhere." Mapped-space ZCA-shrink whitening with anchors far below the hidden dimension (the
pre-registered shrinkage). The calib→test false-alarm gap is a small-sample effect that windowing on the
SAME borrowed axis cannot fix if a mis-read is a whole-item axis error — every position of a mis-read item
is read by the same direction. Single-turn forced-choice register; the agent emits one True/False token; no
operational content.

## Provenance

Prereg `PREREG_mount_early_warning_b34_2026_06_13.md` (frozen @42c7c36). Runner
`papers/showcase-viz/run_mount_early_warning.py`. Receipt `papers/showcase-viz/mount_early_warning_result.json`.
Figure `papers/showcase-viz/mount_early_warning.png`. States `papers/showcase-viz/mount_early_warning_states.npz`.
Certified OATH-HELD against the receipt via `styxx.certify`. Extends `FINDING_mount_regime_2026_06_13.md`.
