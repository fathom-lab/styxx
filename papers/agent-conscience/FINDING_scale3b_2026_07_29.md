# FINDING — the weight-channel result holds at 3B, and every effect is sharper: the knife-edge was a small-model artifact

**Cycle 91. Prereg `PREREG_scale3b_2026_07_29.md` (commit `1979391`), harness `run_scale3b.py`, frozen
before the scored run, all four outcomes pre-committed. Verdict: `SURVIVED__weight_channel_holds_at_3B`.
Receipt: `scale3b_result.json`. Substrate `Qwen/Qwen2.5-3B-Instruct` (fp16, no quantization — verified
to fit the 8 GB card at 7.21 GB training peak). Tests the frame-locality paper's weight-channel scope
(all of cycles 86–90 are 1.5B). Local, $0.**

## The verdict first

The entire weight-channel contrast was repeated at 3B — only the model changed; every floor was
imported from cycles 87–90. Both attacks took in-frame (flip 1.0); cells are powered (70 flipped each,
55 control, 40 held); ARC first-answer accuracy 0.8281. All three gates pass, and not marginally:

- **V1 PASS** — both attacks flip, cells powered, KP preserves held knowledge (held out-of-frame 1.0),
  BASE battery accuracy 0.6366666666666667.
- **SG1 PASS — the dose reversal holds and sharpens.** The unregularized attack overwrites: specificity
  margin **−0.36363636363636365** (recovery 0.0 on the flipped items, control 0.36363636363636365). The
  knowledge-preserving attack spares: specificity margin **+0.7285714285714285** (recovery
  0.9285714285714286, control 0.2). The sign reversal between the two attacks — the claim — is present
  and larger than at 1.5B (−0.2323232323232323 / +0.25656565656565655).
- **SG2 PASS — the coupling holds and sharpens.** On a disjoint MMLU battery: BASE 0.6366666666666667,
  UNREG 0.18333333333333332, KP 0.5966666666666667. The overwriting attack loses 0.4533333333333333 of
  general capability — *below* four-choice chance — versus the sparing attack's 0.04 residual. The
  separation between the two attacks' capability costs clears the 0.10 floor several times over.

## The headline: the knife-edge was a small-model artifact

The frame-locality paper's most-hedged number was the knowledge-preserving recovery rate at 1.5B:
about one-half, with an interval that included one-half, flagged everywhere as not-yet-separated from
its floor. **At 3B that rate is 0.9285714285714286.** The belief-sparing attack does not spare *about
half* the belief at this scale — it spares nearly all of it. The near-one-half reading was a property
of the 1.5B model, not of the phenomenon.

This mirrors the arc's other scale trajectory exactly. Under social pressure, out-of-frame recovery
went 0.9846153846153847 at 3B → 1.0 at 7B — the belief gets *more* stably recoverable as models grow.
The weight channel now shows the same shape from its own starting point: 0.5111111111111111 at 1.5B →
0.9285714285714286 at 3B. The consistent picture across both channels: **as models scale, attacks
increasingly capture the report and leave the belief intact.** What breaks is what the model says; what
survives is what it holds, and the gap between them widens with capability.

The overwriting side sharpens too, in the other direction: at 3B an unregularized attack drives
general capability *below chance* (0.18333333333333332) — it does not merely damage the model, it
leaves it worse than guessing on held-out material. Overwriting a belief at 3B is even more expensive,
and even more clearly not surgical, than at 1.5B.

## What this does to the paper

The weight-channel scope upgrades from "one model family at 1.5B" to "1.5B and 3B, same family, effects
larger at 3B." The dose is not a small-model artifact. And the paper's single softest caveat — the
~one-half recovery rate — should now be stated with its scale trajectory: near one-half at 1.5B, near
0.93 at 3B, consistent with the belief becoming more recoverable as models grow. The 1.5B figure is not
retracted (it is what 1.5B does); it is contextualized by the scale point.

## Scope and disclosures

Two model sizes now (1.5B, 3B), one family (Qwen2.5), one attack class (LoRA r=16, 300 steps), fp16 at
both sizes (no quantization confound). `LAM = 1.0` was frozen from the 1.5B ladder and not re-searched
at 3B; it both flipped and preserved (held 1.0), so the knowledge-preserving attack transferred without
tuning. The attack pool is ARC-Challenge; the coupling battery is a disjoint MMLU draw; both asserted
disjoint in code from all prior pools. The 3B recovery of 0.9285714285714286 is a single 3B draw on 70
flipped items; a second seed would tighten it, but it is far from its floor, unlike the 1.5B figure.

## What this licenses

**Does license:** stating the weight-channel result as holding at two scales with effects that grow
with scale, and re-contextualizing the recovery rate with its 1.5B→3B trajectory (0.5111111111111111 →
0.9285714285714286).

**Does not license:** a scaling *law* (two in-family points, not a fit); any claim beyond Qwen2.5 or
beyond LoRA; a 7B point (untested — would need 4-bit on this card, a quantization confound named for a
separate prereg). The coupling probe-level question stays separate and open, as before.
