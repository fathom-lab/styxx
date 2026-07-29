# FINDING — at triple resolution the coupling grows, the "free" attack turns out to cost a little, and the damage is broad

**Cycle 90. Prereg `PREREG_coupling_resolution_2026_07_28.md` (commit `9051d01`), harness
`run_coupling_resolution.py`, frozen before the scored run, both C2 outcomes pre-committed. Verdict:
`SURVIVED__coupling_replicates__NO_MATERIAL_RESIDUAL__BROAD`. Receipt:
`coupling_resolution_result.json`. Substrate `Qwen/Qwen2.5-1.5B-Instruct`, evaluation only (no
training; the committed cycle-86/87 adapters are reused unchanged), local, $0. Refines
`FINDING_coupling_battery_2026_07_28.md`.**

## The verdict, and the correction it forces

Three checkpoints — clean **BASE**, the belief-overwriting **UNREG** adapter, the belief-sparing
**KP** adapter — on 900 fresh items across two distributions (MMLU 600 + ARC-Challenge 300, seed
900000, disjoint in code from every prior pool and from both adapters' training data).

Pooled accuracy: **BASE 0.6533333333333333**, **UNREG 0.3211111111111111**, **KP 0.62**.

- **C1 PASSES, and the coupling is *larger* at higher resolution:** the belief-overwriting attack
  loses **0.3322222222222222** of general capability against the 0.10 floor — up from the
  0.22666666666666668 measured on cycle 89's single 300-item battery. The central result is not a
  small-sample artifact; it got stronger when tested harder.
- **C2 — the correction. The belief-sparing attack is NOT free.** Its residual is
  **0.033333333333333326**, below the 0.05 bound frozen in advance (hence the label
  `NO_MATERIAL_RESIDUAL`) but **distinctly non-zero**. Cycle 89 measured this cost as exactly 0.0 and
  explicitly bounded that reading — *"the claim is 'no material general loss', not 'provably exactly
  zero'."* **That caveat was correct and this run is why it was written:** at triple the resolution
  and across a second distribution, the sparing attack costs about three points. The honest claim is
  now "a small cost, bounded below five points at this power," not "no cost."
- **C3 — the damage is BROAD, not concentrated.** The overwriting attack loses on every cell:
  MMLU-STEM 0.20560747663551399, MMLU-VERBAL 0.3184584178498986, ARC 0.4. There is no domain it
  spares. (Directionally the loss is largest on the reasoning-heavy ARC slice and smallest on STEM,
  but the partition is a crude subject-string rule fixed in advance and the ordering is reported, not
  claimed.)

## What this means for the picture

The frame-locality paper's coupling section rests on a contrast: overwriting the out-of-frame belief
costs general capability, sparing it costs (nearly) none. **That contrast survives at higher
resolution and gets sharper — roughly a ten-to-one ratio** (0.3322222222222222 versus
0.033333333333333326) rather than the infinite ratio the lower-resolution run implied. An infinite
ratio was always the less likely truth; a large finite one is what the mechanism predicts. Overwriting
a belief is expensive and the expense is spread across everything the model knows; preserving the
belief is cheap but not quite free.

That the overwriting damage is **broad** rather than concentrated also constrains the mechanism: the
attack does not carve out a domain, it degrades the model globally. Whatever the fine-tune does to
reach the belief, it is not surgical.

## Scope and disclosures

One model (1.5B), one attack class (LoRA r=16, 300 steps), two benchmark families, greedy scoring, 900
items. Both batteries are disjoint from adapter training and from all prior pools (asserted in code).
`NO_MATERIAL_RESIDUAL` is a **bound**, not a proof of near-zero: it says the sparing attack's cost is
below 0.05 at this power, and the measured point estimate is 0.033333333333333326. The STEM/VERBAL
split is a keyword partition over MMLU subject strings, frozen before any accuracy was seen and
reported verbatim in the receipt; it supports a descriptive label only. No training occurred, so this
run cannot have altered any prior result.

## What this licenses

**Does license:** restating the paper's coupling result at the sharper, more honest values — the
overwriting attack costs 0.3322222222222222, the sparing attack costs 0.033333333333333326, and the
damage is broad across domains; and retiring the "0.0" figure from cycle 89 as resolution-limited.

**Does not license:** any claim that the sparing attack is free (it is not); any claim beyond 1.5B,
this attack class, or these two benchmark families; any mechanistic claim from the per-cell ordering
(the partition is crude and the ordering is descriptive). A second model size remains the natural
generality test, and is unrun.
