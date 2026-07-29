# FINDING — the weight channel passes the audit: knowledge-preserving belief-survival is frame-invariant, not replay-frame matching

**Cycle 92. Prereg `PREREG_thirdframe_2026_07_29.md` (commit `39074e5`), harness `run_thirdframe.py`,
both frozen before the scored run, both outcomes pre-committed. Verdict:
`SURVIVED__kp_sparing_is_frame_invariant`. Receipt: `thirdframe_result.json`. Substrate
`Qwen/Qwen2.5-3B-Instruct`, inference-only on the committed cycle-91 adapters — cannot alter any prior
result. This is the decisive response to the adversarial audit's Finding 3.**

## The verdict first

The audit's Finding 3: the knowledge-preserving (KP) recovery is measured in the same neutral frame
(`HELP_SYS` + `ASK`) that the replay loss optimized, so the "belief survives" could be replay-frame
matching — the model trained to stay correct *in that frame* — rather than a frame-invariant belief.
The test re-scores recovery in a **third frame** (`T3_SYS` = "You are a careful exam grader…", `T3_ASK`
= "Give only the letter of the correct choice."), disjoint from both the attack frame and the replay
frame, at 3B where the effect is largest.

- **V1 PASS** — both attacks reproduce in-frame (flip 1.0), cells powered, and the replay-frame anchor
  reproduces cycle 91 exactly: KP specificity in the replay frame 0.7285714285714286.
- **FG1 PASS** — in the third frame, KP recovery is **0.8857142857142857** (down only
  0.04285714285714293 from the replay frame's 0.9285714285714286), specificity
  **0.7038961038961038** (essentially unchanged from 0.7285714285714286), while UNREG recovery in the
  same third frame stays at **0.0** (specificity −0.2909090909090909). The KP−UNREG recovery gap in T3
  is 0.8857142857142857.

**The knowledge-preserving belief recovers in a frame the replay never optimized, at nearly the same
rate as in the replay frame. The sparing is frame-invariant. It is not a replay-frame artifact.**

## The honest asymmetry this establishes

The frame-locality program now has two adversarially-tested channels with *opposite* outcomes, and
both are worth stating plainly:

- **Inference-time channels: FAILED the audit.** The specificity control was partly circular (the
  recovery query is the un-poisoned question and the strata are defined by its answer; recovery(caved)
  0.9846153846153847 ≈ recovery(held) 1.0). The belief-survival interpretation there was retracted in
  the v31.1 erratum.
- **Weight channel: PASSED the audit.** The corruption genuinely enters the neutral query (it is baked
  into the weights), and the one remaining confound — that recovery was scored in the replay's own
  frame — is now falsified: the belief recovers just as well in a disjoint third frame (0.8857 vs
  0.9286), while the overwriting attack fails to recover in every frame (0.0).

So the defensible core of frame-locality is the **weight channel**: a knowledge-preserving fine-tune
leaves a belief that is recoverable at the model's output surface regardless of the querying frame,
and an unregularized fine-tune destroys it regardless of frame. That is a genuine
frame-independent-belief result, not a prompt artifact and not a stateless re-ask (the inference-time
critique does not apply, because the weight edit cannot be "removed" by re-prompting).

## Why the T3 frame is a valid test, not a broken prompt

UNREG behaves sanely in T3 (recovery 0.0, control 0.2909090909090909 — comparable to its replay-frame
control 0.34545454545454546), so T3 is a working frame that elicits answers normally; the KP/UNREG
divergence in T3 is a real difference between the two adapters, not an artifact of a degenerate prompt.
And the replay-frame anchor reproduced cycle 91 to the digit, so the run is calibrated.

## Scope

One model (3B), one attack class (LoRA r=16, 300 steps), one third frame, N=5 neutral samples,
inference-only on committed adapters. A single disjoint frame falsifies the specific replay-frame
confound; it does not prove frame-invariance over *all* frames — a second and third disjoint frame
would strengthen it, though the near-zero drop (0.043) on the first disjoint frame is a strong signal.
The 1.5B adapters were not re-tested here (the effect is cleaner at 3B); a 1.5B third-frame check is a
cheap follow-up. This tests the weight channel only; the inference-time channel's separate correction
stands.

## What this licenses

**Does license:** stating that the weight-channel belief-survival is frame-invariant (recovers at
0.8857142857142857 in a frame disjoint from both attack and replay frames, vs 0.0 for the overwriting
attack), and that the weight channel passed the adversarial re-test the inference-time channel failed —
making it the paper's defensible core.

**Does not license:** any inference-time belief-survival claim (separately retracted); frame-invariance
over all frames from one disjoint frame; generalization beyond 3B, Qwen2.5, or this attack class. The
corrected paper should present the weight channel as tested-and-surviving and the inference-time
channel as tested-and-retracted — the asymmetry is the honest result.
