# FINDING — the coupling is real and large: overwriting the belief costs 22.7 points of general capability; sparing it costs none

**Cycle 89. Prereg `PREREG_coupling_battery_2026_07_28.md` (commit `e31293c`), harness
`run_coupling_battery.py`, both frozen before the scored run, all outcomes pre-committed. Verdict:
`SURVIVED__belief_rewrite_coupled_to_capability_damage`. Receipt: `coupling_battery_result.json`.
Substrate `Qwen/Qwen2.5-1.5B-Instruct`, local, $0. Settles the coupling question left open in
`PAPER_frame_locality_2026_07_28.md` §6.**

## The verdict first

Three checkpoints — the clean model, the cycle-86 unregularized adapter (which overwrote the
out-of-frame belief), and the cycle-87 knowledge-preserving adapter (which spared it) — scored on a
single held-out MMLU battery of 300 items, asserted disjoint in code from every question in cycles
74–88 and from both adapters' training data. The two adapters trained on the same items and differ
only in the replay regularizer, so the battery isolates the *spillover* to unrelated capability.

- **BASE** accuracy **0.5833333333333334**
- **UNREG** (overwrote the belief) accuracy **0.3566666666666667**
- **KP** (spared the belief) accuracy **0.5833333333333334**

The gate passes decisively: KP retains 0.22666666666666668 more general capability than UNREG
(floor 0.10), and KP stays exactly at base (BASE − KP = 0.0, floor 0.05). The unregularized attack
that rewrote the belief lost roughly a fifth of the model's general accuracy — collapsing more than
a third of the way from base toward four-choice chance (0.25) on material it never trained on. The
knowledge-preserving attack that spared the belief lost **nothing** measurable on the same battery.

## What this settles

The frame-locality paper reported that a knowledge-preserving weight attack spares about half the
out-of-frame belief while an unregularized one overwrites it, and framed the coupling question —
whether rewriting the belief is inseparable from damaging general knowledge — as *engaged but not
settled*, because the only knowledge evidence was on same-benchmark held items (unregularized 0.44,
knowledge-preserving 1.0). This run answers it on a **disjoint** battery, and the answer is the same
in the stronger place: **the belief-overwriting attack pays a large, broad capability cost; the
belief-sparing attack pays none.**

So the two facts move together. You cannot, with an attack of this class, overwrite the model's
out-of-frame belief without a general capability price — and when you constrain the attack to keep
the belief recoverable, you also keep the capability. Belief-rewrite and general capability are
**coupled** on this substrate. The dose picture from the paper gets its mechanism: the "dose" of
collateral knowledge damage that determines how much of the belief survives is not a same-benchmark
artifact — it is visible as general capability loss the attack pays for reaching the belief.

## Scope and what it does not claim

One model (1.5B), one attack class (LoRA r=16, 300 steps), one battery family (MMLU), greedy scoring.
The battery is disjoint from training and from all prior pools, so the drop is not scoring the
training set and the preservation is not scoring replayed items. This is a **behavioral** coupling
result — capability accuracy tracks belief-overwrite; it is not the calibration-poisoning arc's
probe-level coupling battery, whose formal question (does removing the honesty *read* necessarily
cost capability) is a different measurement and stays separate. The KP adapter preserving capability
to the decimal at base (0.0 loss) is on this 300-item battery; a larger or harder battery could
reveal a small residual cost — the claim is "no material general loss," not "provably exactly zero."
Nothing here speaks to larger models or other attack classes.

## What this licenses

**Does license:** upgrading the paper's §6 coupling statement from *open* to a measured behavioral
coupling on a disjoint battery — overwriting the belief cost 0.22666666666666668 of general
capability, sparing it cost 0.0 — with the scope above.

**Does not license:** any claim about probe-level coupling (a different measurement); any
generalization beyond 1.5B, this attack class, or this battery; a claim that capability preservation
is exactly zero-cost rather than below this battery's resolution. The natural next step is a broader,
harder capability battery and a second model size, to price the residual and test the coupling's
generality — neither needed to state the result at the strength above.
