# FINDING — the borrowed-conscience MOAT FAILS against an activation attacker: different reference models read truth via the SAME agent-space direction (ATTACK-TRANSFERS)

**2026-06-13 · Fathom Lab / styxx. Pre-registered: `PREREG_mount_gamed_2026_06_13.md` (frozen pre-run,
committed c622e7d). Receipt: `mount_gamed_result.json`; figure `mount_gamed.png`. The deepest test of the
`styxx.mount` thesis — and it falsifies the headline. The mount was pitched as "hard to fool because the
conscience is BORROWED from a different reference model." Against an adversary that can shape the agent's
activations, borrowing a different reference provides NO defense, because two different reference models'
transported truth-readouts are nearly the SAME direction in the agent's activation space.**

## Result — fooling the targeted conscience fools the borrowed one too

Every linear mount reading is affine in the agent's hidden state, r(h) = a·h + c. We mounted three
truth-consciences on Llama-3.2-3B at a common read layer (L21): **A** = gemma (the attacker's target),
**B** = Qwen2.5-3B (a DIFFERENT reference, the BORROWED defense), **A′** = gemma re-fit on disjoint
anchor data (the re-fit defense). An optimal unconstrained activation attack steps along a_A; transfer is
governed by the alignment of readout directions.

| quantity | value |
| --- | --- |
| cos(a_A, a_B) — gemma vs Qwen readout directions in Llama space | **0.941** |
| cos(a_A, a_A′) — gemma vs gemma-re-fit | 0.2432 |
| attack budget to defeat A (ε* = ‖δ‖/‖h‖) | **0.05** |
| catch at ε*: A (targeted) | 0.0 |
| catch at ε*: **B (borrowed, different reference)** | **0.0** |
| catch at ε*: A′ (re-fit, same reference) | 0.30 |
| catch at ε*: A under a RANDOM-direction attack (control) | 1.0 |

**Verdict per the frozen gate: ATTACK-TRANSFERS.** A perturbation of just 5% of the activation norm,
optimized against the gemma conscience, drives BOTH the gemma and the Qwen conscience to zero catch. The
random-direction control stays at 1.0, so the attack is specific and the test is valid. Borrowing a
conscience from a different reference model does NOT defend against an activation-space adversary.

## The mechanism — and the inversion of the hypothesis

The cause is precise and itself a notable result: **cos(a_A, a_B) = 0.941.** Transport gemma's truth
direction and Qwen's truth direction into Llama through their own label-free maps, and they land on
NEARLY THE SAME direction in Llama's activation space. There is effectively ONE readable truth direction
in the agent — so an attack that moves the agent off it evades every borrowed conscience at once. (This
echoes the universal / shared-geometry theme of the arc, but here that universality is a liability, not
an asset: a canonical truth direction is a single thing to game.)

The pre-registered contrast inverted my hypothesis. RE-FITTING the SAME reference on disjoint anchor data
(A′) gives a DIFFERENT functional (cos 0.2432) and is the MORE robust defense (catch 0.30 at ε* vs 0.0
for the borrowed different reference). The freedom that matters is in the MAP FIT — different anchor data
pushes the readout's off-truth (null-space) components around — not in the identity of the reference
mind. So the defense that helps (partially) is RE-LOCKING on fresh, private data (the capture-trust
answer), NOT borrowing a different model. Even A′ collapses by a slightly larger budget, so re-locking is
only a partial, not a complete, defense at this scale.

## What this does to the mount's claims — a required correction

The `styxx.mount` headline "hard to fool because the conscience is borrowed" is FALSE against an
activation-space adversary and must be qualified:
- Borrowing protects against an agent gaming its OWN labels / its OWN probe (the agent has no labels;
  the readout is not its self-report) and works for COOPERATIVE or non-adversarial agents — the live
  catch of unintentional sycophantic caving stands.
- Borrowing does NOT add adversarial robustness: a determined adversary that shapes activations (a
  weight fine-tune approximates this, weakly) evades all borrowed consciences at once, cheaply, because
  they share a readout direction. The honest moat, if any, is re-locking on private fresh data — and it
  is only partial here.
- Net deployment scope: `styxx.mount` is an integrity monitor for cooperative / non-adversarial
  deployment (catching drift, unintentional caving, sycophancy), NOT a robust defense against a model
  trained to evade it.

## Honest bounds (and why this negative is conservative)

The attacker is an UNCONSTRAINED activation perturbation at the read layer — an UPPER BOUND on a
weight-fine-tune attacker, and it ignores the constraint of preserving the agent's output. So a real
fine-tuned adversary is WEAKER, and the realistic mount may catch more than this worst case suggests —
but the borrowing defense specifically is refuted regardless, because it rests on readout-direction
diversity that does not exist (cos 0.94). Linear mounts, common read layer (L21), truth axis, local
same-cluster models (gemma/Qwen references, Llama agent), single-turn, n = 20 caves. Read-only throughout.
The owed work: a real weight-level adversarial fine-tune (to confirm the upper-bound result holds for the
realistic attacker), and whether an ENSEMBLE of deliberately-orthogonalized readouts (not different
references — they don't help — but Gram-Schmidt-diversified functionals + re-locking) raises the attack
budget. This finding kills the "different-reference borrowing" defense and redirects it to re-locking.
