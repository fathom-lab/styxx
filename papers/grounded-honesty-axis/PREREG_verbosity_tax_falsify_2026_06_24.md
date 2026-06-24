# PREREG — falsify the "calibration is intrinsically wordier" claim (register vs prompt)

**Frozen 2026-06-24 BEFORE generating the concise-constrained corpus. Offline, local-GPU, NO frontier key.**
Stress-test of a claim styxx published this morning (TG, `FINDING_overconfidence_length_robust`): calibrated
epistemic register is intrinsically ~1.16× wordier than overconfident register across 3 model families.

## The confound being tested
The claim could be a PROMPT artifact, not a register property: `SYSTEM_CALIBRATED` asks to "acknowledge
uncertainty where evidence is partial" (invites elaboration); `SYSTEM_OVERCONFIDENT` says "be decisive, don't
hedge" (invites brevity). So the 1.16× may reflect what we *asked for*, not what calibration *costs*.

NOTE: today's matched corpora already gave both stances the SAME "~55 words" instruction and calibrated still
ran 1.16× longer — partial evidence it is register-driven. This prereg pushes harder.

## Test (primary): hard brevity constraint on BOTH stances
Generate the same overconfidence `QUESTIONS` × 2 stances (the EXACT `SYSTEM_CALIBRATED` /
`SYSTEM_OVERCONFIDENT` prompts) with an identical hard-brevity rule appended to BOTH:
> " Answer in ONE sentence, as briefly as you possibly can."

If calibration is register-intrinsic, calibrated answers will STILL be longer even under maximal brevity
pressure (you cannot state a caveat as briefly as a flat assertion). If it was a prompt artifact, equal
brevity pressure collapses the gap. Generators: gemma-2-2b-it + Qwen2.5-3B-Instruct (greedy, seed-pinned).

## Decision thresholds (FROZEN). r = mean(calibrated words) / mean(overconfident words) under the concise rule.
| Verdict | Condition |
|---|---|
| **REGISTER-INTRINSIC (claim holds, hardened)** | r ≥ 1.08 on BOTH generators |
| **PROMPT-ARTIFACT (soften the public claim)** | r ≤ 1.03 on BOTH generators |
| **PARTIAL / model-dependent** | otherwise (report range; the public "intrinsic" wording is softened to "under contrastive stance prompts") |

## Secondary (register survival)
Report hedge_density std-diff (calibrated − overconfident) under the concise rule. If the register still
separates (|std-diff| ≥ 0.3) AND calibrated is longer → the verbosity is the cost of the register surviving.
If the register collapses under brevity (|std-diff| < 0.3), that is itself evidence calibration *needs* verbal
space to exist (a stronger form of intrinsic) — report it as such, not as a null.

## Honest commitment
This tests a claim we PUBLISHED hours ago. If it comes back PROMPT-ARTIFACT, we issue a same-voice follow-up
softening the public wording (the morning post said "intrinsically ~1.16×"). The thresholds above decide it,
not the desire to have been right. In-silico, 2 local 3B-class models, single seed, n≤200/model.
