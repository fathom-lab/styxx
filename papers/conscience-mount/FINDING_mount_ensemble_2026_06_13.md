# FINDING — the mount-side moat is re-locking, and it's real-but-modest: a diverse ensemble stops a naive attacker cold and an adaptive one only ~2× (ENSEMBLE-DEFENDS)

**2026-06-13 · Fathom Lab / styxx. Pre-registered: `PREREG_mount_ensemble_2026_06_13.md` (frozen pre-run,
committed 8e24b23). Receipt: `mount_ensemble_result.json`; figure `mount_ensemble.png`. The constructive
sequel to the gamed-model negative (`FINDING_mount_gamed`, ATTACK-TRANSFERS killed the borrowing defense):
is there ANY mount-side defense? A K=5 re-locked ensemble (gemma functionals on disjoint anchor splits,
OR-rule) is the answer — and the answer is "yes, but modestly, and at a false-alarm cost."**

## Result — naive attacker defeated 32×, adaptive attacker evades at 2×

Five readout functionals were fit on disjoint anchor splits (re-locked on fresh data each) and mounted on
Llama-3.2-3B at the common layer (L21). A cave is caught if ANY member reads it false (OR-rule; to evade,
fool ALL five). The members came out NEARLY ORTHOGONAL (mean pairwise cos **0.0133**) — re-locking on
disjoint data diversifies the off-truth components strongly — while each retains moderate alignment with
their shared mean direction (cos **0.4305**).

| attacker | budget to evade (ε* = ‖δ‖/‖h‖) | vs single-mount |
| --- | --- | --- |
| single member alone | 0.025 | 1× (reference) |
| ensemble vs NAIVE attack (target one member) | 0.8 | **32×** (effectively never — see figure) |
| ensemble vs ADAPTIVE attack (target the shared mean direction) | 0.05 | **2×** |
| ensemble vs RANDOM direction (control) | — | catch stays 1.0 at ε*_single (valid) |

**Verdict per the frozen gate (ε*_adaptive ≥ 2× single): ENSEMBLE-DEFENDS** — at the boundary. The honest
reading: the defense against a NAIVE attacker is overwhelming (the figure shows the ensemble catch dips
then RECOVERS to 1.0, because moving along one member's direction barely moves the ~orthogonal others), but
the defense against an ADAPTIVE attacker who targets the shared truth direction is MARGINAL — it evades the
whole ensemble at just 2× the single-mount cost.

## The synthesis — there is one shared truth direction, and it is the universal vulnerability

Across the three adversarial tests the picture is now complete and consistent:
- **Borrowing a different reference does NOT defend** (gamed: cos 0.94 — different references read truth via
  the same direction).
- **Re-locking the same reference on disjoint data DOES diversify** (here: cos 0.0133 between members) and
  defeats a NAIVE attacker (32×).
- **But every member still shares the truth component** (cos 0.43 with the mean), and an ADAPTIVE attacker
  who moves along that shared direction evades all members at once (2× cost).

So the moat is RE-LOCKING DIVERSITY, and it is real against unsophisticated evasion but only a ~2× tax on a
sophisticated adversary who finds the shared truth direction. The arc's universality (one truth geometry)
is the recurring villain of the adversarial story: diversity in the off-truth subspace is cheap to build,
but the truth signal itself lives on one direction that any sufficient attack can target.

## The honest cost that does not move: FPR

The ensemble's robustness is bought at a steep false-alarm price. With per-member thresholds tightened to
a fraction-of-target per-member threshold, the realized ENSEMBLE false-alarm rate on held-out honest
decisions is **0.2667** — the OR-rule unions five members' false alarms. So the robustness ratios above
hold at a false-alarm rate near 0.27, not the target; a fair fixed-FPR comparison would shrink them. The
baseline catch shrank too (only 9 of the caves were caught
by the tightened ensemble), so the ε* estimates rest on n=9 and the adaptive ratio (2.0) is a fragile,
boundary pass. The robustness/false-alarm tradeoff is unfavorable and is the real limit of this defense.

## What it means for styxx.mount (the complete adversarial scope)

`styxx.mount` is an integrity monitor for COOPERATIVE / non-adversarial agents (drift, unintentional
caving, sycophancy — the live catch stands). Against adversaries: borrowing buys nothing; a re-locked
ensemble buys strong protection against NAIVE activation attacks and a modest ~2× tax against ADAPTIVE
ones, at a high false-alarm cost. There is no cheap mount-side defense against a determined adaptive
attacker, because truth is one direction. The honest defenses against a model trained to evade the mount
are NON-mount: deny white-box access, or detect the fine-tune / off-manifold input itself.

## Honest bounds

Unconstrained activation attack (upper bound on a weight fine-tune); linear mounts; common read layer
(L21); truth axis; local same-cluster models; single-turn; n=9 caves at the tightened threshold; OR-rule
ensemble at FPR 0.27. Read-only throughout. Owed: a real weight-level adversarial fine-tune (does the
realistic attacker also find the shared direction); a fixed-FPR robustness comparison; whether
off-manifold / fine-tune detection (a non-mount defense) is the actual answer.
