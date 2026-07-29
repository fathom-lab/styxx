# FINDING — the weight channel holds at a second vendor: the dose is not a Qwen property

**Cycle 94. Prereg `PREREG_vendor3b_2026_07_29.md` (commit `a73f52b`), harness `run_vendor3b.py` —
generated from `run_scale3b.py` by mechanical substitution, with only the model id, the pool seed,
and the file prefix changed; the diff is part of the audit surface. Frozen before the scored run.
Verdict: `SURVIVED__weight_channel_holds_at_second_vendor`. Receipt: `vendor3b_result.json`.
Substrate `meta-llama/Llama-3.2-3B-Instruct` (fp16) — a different vendor, pretraining corpus,
tokenizer, and chat template from every prior weight-channel substrate. Local, $0.**

## The verdict first

The entire cycle-91 contrast was repeated with the vendor swapped — every floor imported unchanged
from cycles 75/87/89. Both attacks took in-frame (flip 1.000 each); cells are powered (70 flipped
per arm, 60 control, 40 held); ARC first-answer accuracy 0.73125. All three gates pass:

- **V1 PASS** — both attacks flip, cells powered, the knowledge-preserving attack preserves held
  knowledge out of frame at 1.0, BASE battery accuracy 0.59.
- **SG1 PASS — the dose reversal appears on Llama.** The unregularized attack overwrites:
  specificity margin **−0.23333333333333334** (recovery 0.0 on flipped items, control
  0.23333333333333334). The knowledge-preserving attack spares: specificity margin **+0.35**
  (recovery 0.7, control 0.35). The sign reversal between the two attacks — the claim — is present
  at a second vendor.
- **SG2 PASS — the coupling appears on Llama.** On a disjoint MMLU battery: BASE 0.59, UNREG
  0.15333333333333332, KP 0.56. The overwriting attack loses 0.43666666666666665 of general
  capability — below four-choice chance, as at Qwen-3B — versus the sparing attack's
  0.029999999999999916 residual. The separation between the two attacks' capability costs clears
  the 0.10 floor several times over.

## What is striking: the numbers rhyme across vendors

This was not a direction-only replication. The UNREG specificity on Llama-3.2-3B
(−0.23333333333333334) lands next to the 1.5B Qwen value; the recovery under the unregularized
attack is 0.0 on every substrate tested; the overwriting attack drives held-out capability below
chance at both 3B substrates; and the knowledge-preserving residual is small on every substrate.
The knowledge-preserving recovery rate on Llama (0.7) sits between the Qwen 1.5B and Qwen 3B
values — consistent with the arc's picture that how much belief the sparing attack spares varies
by substrate, while the *structure* (overwrite pays broadly, sparing does not; specificity flips
sign between the arms) is what transfers.

One number moved in an interesting direction and is reported as description, not claim: Llama's
UNREG battery accuracy fell further below chance than either Qwen substrate's. Whatever the
unregularized flip does to reach the belief, this vendor pays even more for it.

## What this does to the paper

The weight-channel scope upgrades from "one vendor (Qwen2.5), two scales" to **"two vendors, two
scales, same protocol, same frozen floors."** The most natural remaining objection to the
weight-channel core — that the dose reversal and the coupling were properties of one vendor's
training recipe — is now tested and answered. The λ = 1.0 setting, frozen from the Qwen 1.5B
ladder, transferred to a different vendor without re-search and both flipped and preserved; the
attack recipe is not tuned to its substrate.

## Scope and disclosures

Two vendors (Qwen2.5, Llama-3.2), 1.5B and 3B classes, one attack class (LoRA r=16, 300 steps,
identical projection-module names verified present in both architectures), fp16 everywhere. The
Llama pool is a fresh ARC-Challenge draw asserted disjoint in code from every prior pool and both
prior strata; the battery is a disjoint MMLU draw. Single seed per substrate. The knowledge-
preserving recovery on Llama (0.7 on 70 flipped items) is one draw, and in this design the recovery
magnitude is descriptive — the gated, discriminating evidence is the specificity sign contrast
between the two arms.
The smoke run wrote only `*_SMOKE_INVALID*` files and was not read as a result.

## What this licenses

**Does license:** stating the weight-channel dose reversal and behavioral coupling as
vendor-general across the two vendors tested, at matched protocol and floors; citing the
cross-substrate stability of the overwrite signature (recovery 0.0 everywhere tested, capability
below chance at both 3B substrates).

**Does not license:** "all vendors" (two tested); any claim beyond LoRA or beyond the 1.5B–3B
band; a scaling or vendor *law* (four substrate points, not a fit); anything about the
probe-level coupling question, which stays open as before.
