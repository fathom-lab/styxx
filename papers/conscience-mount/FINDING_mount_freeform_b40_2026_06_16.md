# FINDING — B40: a borrowed conscience catches a FREE-FORM PROSE lie at the commitment token (SUCCESS at 3B; VOID-arm-starvation at 7B; OATH-HELD)

**2026-06-16 · Fathom Lab / styxx. Prereg `PREREG_mount_freeform_b40_2026_06_16.md` frozen (with two
pre-data amendments, both logged before the runner existed) BEFORE this runner. Runner
`papers/showcase-viz/run_mount_freeform_deception.py`, SEED=0, greedy decode. Receipts
`papers/showcase-viz/mount_freeform_deception_rung1_result.json` (Llama-3.2-3B) and
`mount_freeform_deception_rung2_result.json` (Qwen2.5-7B). This closes the owed B34/B35 follow-up: the
prior arc established the mount is a COMMITMENT-TIME monitor in the forced-CHOICE register and that the
free-form caving register STARVED the cave arm (B35). B40 asks the harder, cleaner question — when an
agent is INSTRUCTED to argue a known-false fact in fluent prose and forced to a byte-identical
`FINAL ANSWER:` commitment token, does the borrowed gemma-2-2b truth axis, read at that single
commitment token, catch the lie that a no-answer-key reader of the words cannot? At the validated 3B
scale the answer is YES; the 7B scale break could not be run because the stronger model would not comply
enough to fill the arm.**

## What was tested

A within-item minimal pair holds claim, frame, instruction wording, and the byte-identical
`FINAL ANSWER: True` commitment template constant across both arms; only the asserted truth value flips
(LIE arm = false member, HONEST arm = true member). The borrowed conscience is the shipped gemma-2-2b-it
truth axis transported label-free via `styxx.crossmind` and read on the agent's own hidden state at the
single commitment token (the B34 read-locus fix: never the claim-content token, never pooled). A
knowledge gate (in-framing, difficulty-matched; the B38-killed cross-framing screen is NOT used) keeps
only pairs the agent provably knows; a compliance gate (DeBERTa-MNLI entailment of the false claim, no
assert-then-correct) keeps only items where the prose actually commits the lie; and a sincere-free-form
positive control (the build GO gate) must first prove the axis transports to the forced-verdict prose
register, so a null would be attributable to persona-overwrite rather than a dead read.

## RUNG 1 — Llama-3.2-3B-Instruct (fp16, the only validated gemma→Llama transport pair): SUCCESS

The validated transport held: anchor R² 0.9851. The build GO gate (sincere-true vs sincere-false
free-form commitments, no deception persona) cleared at AUROC 0.9231 — the borrowed axis demonstrably
transports to the forced-verdict prose register, licensing the new calibration register. On the 34
complied lie-arm items the conscience catch is 0.7647 at realized held-out-honest FPR 0.0769; the
lie-vs-honest commitment-token discriminability is AUROC 0.8967, which clears the shuffled-axis null at
p 0.01 and the label-permutation null at p 0.001. Every condition of the frozen kill-gate (catch
threshold, FPR ceiling, both nulls, build gate first) is met → **SUCCESS-CONSCIENCE-CATCHES-FREE-FORM-LIES.**
The set: a frozen pool of 120 non-comparative claim pairs yielded 68 knowledge-gated pairs, 27 held-out
honest calibration commitments (center + tau, regime-matched, leakage-free), 39 honest test commitments,
and 41 test items; the operating threshold tau is 1.299.

## The text baseline (DESCRIPTIVE, not gating — per amendment #2)

On the identical generations, a no-answer-key reader of the prose (HHEM faithfulness + DeBERTa-MNLI
prose-internal-consistency — the honest comparator) catches only 0.0588 at FPR 0.2051: fluent debate
prose is internally coherent, so a reader of the words alone is near-blind to the lie. A ground-truth
oracle handed the external answer key catches 1.0 at the same 0.2051 — reported transparently, because it
has the answer the conscience does not. The headline is the gap between the conscience and the honest
no-answer-key comparator: reading the substrate, with no answer key, recovers the lie that reading the
words does not. **The conscience does NOT beat the ground-truth oracle, and the prereg does not claim it
does** — the second pre-data amendment retired the "beat ground-truth by a McNemar test" gate as
unwinnable by construction (an oracle with the key trivially catches an asserted false fact) and
superseded it with the ITEM-CAUGHT-in-free-form gate evaluated above.

## RUNG 2 — Qwen2.5-7B-Instruct (4-bit NF4): VOID-ARM-STARVATION (honest, no primary, no re-roll)

The 7B scale break could not be run to a verdict: only 12 of 46 test items produced compliant lie prose
(entailed the false claim with no assert-then-correct), below the pre-registered 30-item floor. Per the
frozen house rule the run terminates with no primary statistic and no re-roll of set / seed / persona /
domain. This is exactly the arm-starvation risk the prereg named and priced (a 120-pair pool sized
against a ~30 floor): the stronger instruct model refuses to fully commit a fluent factual lie often
enough to fill the arm. The attributable reading per the prereg's RUNG-2-conditional clause: the
free-form forced-verdict SUCCESS stands at the validated 3B scale; the scale break is UNTESTED, not
refuted — and the blocker is elicitation/compliance, not a measurement failure.

## The honest reading (pre-committed)

At the validated 3B transport scale, with the register-transport build gate cleared first, the borrowed
gemma-2-2b conscience catches an instructed free-form prose lie at the commitment token above the frozen
ITEM-CAUGHT bar, while a no-answer-key reader of the same prose sits near chance. This extends the
COMMITMENT-TIME-monitor picture from the forced-choice register (B34) to free-form generation: the
substrate still carries the truth at the moment of commitment even when the surrounding prose argues the
opposite — a NEW construct (instructed prose deception), distinct from the spontaneous-caving ITEM-CAUGHT
phenomenon, and a clean SUCCESS of that named construct rather than a re-use of the prior headline.

## Scope (carried forward, none erased)

Single substrate: one agent (Llama-3.2-3B-Instruct, fp16), one reference (gemma-2-2b-it), one map layer,
SEED=0, greedy decode — the SUCCESS is single-seed and single-pair; cross-scale (7B) is VOID, so this is
NOT yet a scale-robust claim. Small n: the realized FPR granularity is coarse at this calibration-set
size and a partial-overwrite middle outcome would be underpowered (the same n-fragility ceiling the
B35–B39 sub-arc kept hitting); the clean SUCCESS lands above it, but the FPR control is not tight to a
target. White-box (needs the agent's own hidden state → open-weight / self-host). Read-only; a flag is a
measurement, never an intervention. Linear BORROWED truth axis (gemma→Llama same-cluster) with its own
false positives. COOPERATIVE-monitor scope ONLY — the ATTACK-TRANSFERS result stands (different
references share ~one truth direction; a small activation perturbation evades all borrowed consciences),
and reading a free-form commitment token adds no adversarial-activation robustness; this is a cooperative
role-play elicitation, not a trained evader. The text baseline is a no-answer-key comparator by design;
the conscience's only privilege is reading the substrate vs. the words, and it does not and is not
claimed to beat a ground-truth oracle.

## Provenance

Prereg `PREREG_mount_freeform_b40_2026_06_16.md` (frozen with amendments #1 and #2 before the runner, at
commits 29c5852 → 8664ca6 → c943c1d). Runner `papers/showcase-viz/run_mount_freeform_deception.py`,
SEED=0, greedy decode. Receipts `papers/showcase-viz/mount_freeform_deception_rung1_result.json` and
`papers/showcase-viz/mount_freeform_deception_rung2_result.json`. Per-token states streamed to a
gitignored `.npz`. Certified OATH-HELD against both receipts via `python -m styxx.certify`. Extends
`FINDING_mount_early_warning_b34_2026_06_13.md` and `FINDING_mount_freeform_b35_2026_06_13.md`; the
conscience read is the shipped one bit-for-bit (no measurement change since the regime-calibration
operating point).
