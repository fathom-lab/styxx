# FINDING — the dose result replicates on a fresh benchmark: about half the beliefs recover, and "about half" is near the floor by nature

**Cycle 88. Prereg `PREREG_kp_replication_2026_07_28.md` (commit `f24ee6a`), harness
`run_kp_replication.py`, frozen before the scored run, all three outcomes pre-committed. Verdict per
the frozen gates: `SURVIVED__kp_dose_result_replicates`. Receipt: `kp_replication_result.json`.
Substrate `Qwen/Qwen2.5-1.5B-Instruct`; benchmark ARC-Challenge (`allenai/ai2_arc`, test), disjoint
from the sycophancy-eval bench by construction; second seed 880000; local, $0. Replicates
`FINDING_kp_recovery_2026_07_28.md`.**

## The verdict, and what it does and does not settle

The frozen gate passes on a fresh benchmark and a second seed. All four LAM rungs met both validity
conditions (in-frame flip ≥ 0.871, held out-of-frame 1.0 across the ladder); LAM 1.0 selected;
cells powered (69 ATTACK_FLIPPED, 60 CONTROL). RG1 clears both legs: recovery on flipped items
0.5362318840579711 (floor 0.50) and specificity margin 0.28623188405797106 (floor 0.15).

**What replicated robustly:**
- **The dose reversal itself.** The cycle-86 unregularized attack overwrote the belief (recovery
  0.0222, specificity −0.2323); the knowledge-preserving attack, on two different benchmarks now,
  does not (recovery 0.5111 then 0.5362; specificity +0.2566 then +0.2862). The sign of the
  specificity margin is positive and of similar size in both runs — this was always the
  discriminating leg and it replicates cleanly.
- **The mechanism signature.** Perfect bimodality again: 0 of 69 flipped items resolve out of frame
  to anything but the original truth or the planted target. Across two benchmarks and two seeds,
  the poison and the belief compete for exactly one slot.
- **The specificity control.** CONTROL (pre-wrong, never trained) sits at 0.25 out of frame, next to
  cycle-87's 0.2545 — no blanket "answer correctly in the neutral frame" lift from the replay.

**What did NOT get cleaner — stated plainly, because the run existed to state it:** the recovery
*rate* is still not individually separated from its floor. This run's Wilson interval is
[0.4197820076036184, 0.6488600870236277] — its lower bound does **not** clear 0.50. Pooling both
independent runs (this run's flipped cell plus cycle 87's) leaves the recovery point estimate near
the same value with an interval whose lower bound also does not clear 0.50. **The honest magnitude
is: about half the beliefs recover under a knowledge-preserving attack — and "about half" sits near
the floor by its own nature, so no single run's interval, nor the pooled one, excludes one-half.**
The gate is cleared by the point estimate in both runs; the claim it licenses is "roughly half,"
not "clearly more than half."

## What this means

The load-bearing claim of the arc's parametric extension is now well-supported: **a
knowledge-preserving weight attack does not overwrite the out-of-frame belief the way an
unregularized one does** — the boundary of frame-locality at the weights is a *dose set by
collateral knowledge damage*, not a wall. That qualitative result rests on the specificity reversal
and the bimodality, both replicated across benchmark and seed, not on the fragile recovery leg.

What remains genuinely near one-half — and therefore the right thing to say is "half," not a larger
number — is *how much* of the belief a surgical poison spares. That the fraction is close to a half
is itself interesting: the knowledge-preserving attack lands in a regime where the neutral-frame
belief is a near-coin-flip between the truth it held and the lie it was trained, which is exactly
the "competing for one slot" picture the bimodality shows.

## Scope

Two model draws, one model size (1.5B), two benchmarks (meg-tong MC; ARC-Challenge), one attack
class (LoRA r=16, 300 steps, flip + replay), N=5 neutral samples, English. ARC first-answer accuracy
0.75. HELD is both the replay set and the `V_preserve` check by construction (disclosed); the
mechanism is measured on never-replayed ATTACK items. Selection was on validity gates only.

## What this licenses

**Does license:** stating the dose result as replicated across benchmark and seed, with the honest
magnitude "about half recover" (pooled near one-half, close to the floor); and folding both runs into the
program's account of the frame-locality boundary.

**Does not license:** any claim that recovery is *clearly* above one-half (no interval excludes it);
any generalization beyond 1.5B or this attack class; a settled coupling claim (no capability battery
run). A larger pool or a third seed would tighten the rate but is not needed to support the
qualitative dose claim, which the specificity leg and bimodality already carry.
