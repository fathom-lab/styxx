# Consistency vs signatures: an attack-tested, defended honesty monitor — and complementary surfaces

**styxx / fathom-lab · 2026-06-04 · in-silico, pre-registered, self-falsified**

> **REVISED after the RL-obfuscation arm (`RESULT_rl_obfuscation`).** The original framing —
> *consistency BEATS signatures* — is **not supported by my experiments and is retracted.** The
> activation probe resisted *every* feasible attack I ran (per-item perturbation, steering, adversarial
> LoRA); the one signal I actually cracked was the *consistency* check (grounded_honesty, 18% via
> framing). So in these runs the signature was the *more* robust of the two. The honest conclusion is
> **complementary attack surfaces + defense-in-depth**, not a winner. What stands fully is the narrow,
> strong result: **grounded_honesty is attack-tested and defended, cross-vendor.** Read the "Thesis"
> below as the starting hypothesis the evidence then corrected.

## Thesis (starting hypothesis — since corrected; see above)
The white-box honesty signals that survive adversaries are **consistency checks** (compare an output
to the model's own resampled belief), not **fixed signatures** (a direction read off the activations).
A signature is a target an adversary routes around; consistency is not a target — to beat it the model
must make the lie its own stable belief, which is expensive and detectable. styxx's validated
primitive `grounded_honesty` is a consistency check; this arc attacks it, defends it, and tests it
across models.

## What we established (each step falsification-tested)
1. **Pattern recognition.** Everything styxx has shipped that's validated is consistency-based
   (`grounded_honesty` AUC 0.966, cross-context divergence 0.875, the action-gate); the things that
   died this week were fixed signatures (the geometry probe, the spectral-trajectory instrument,
   killed at their cheapest gates). Signatures keep failing; consistency keeps surviving.
2. **Probe head-to-head — honest, partial.** `grounded_honesty` is immune to residual attacks that
   defeat an activation probe — but **near-tautologically** (it reads no residuals). And on clean
   MMLU the probe has a *large margin*: a learned minimal per-item perturbation does **not** collapse
   it (cost 76%, AUROC unchanged); only an unrealistic full-steer does. So **we did NOT show the probe
   is cheaply fragile** — that half rests on the literature's distribution-level RL-obfuscation
   (arXiv 2505.27958), which we have not reproduced. *Owed: the RL-obfuscation arm.*
3. **Falsification test — it cracked.** Attacking `grounded_honesty` directly with a cheap
   model-rewritten question that nudges toward the lie **confidently fools it ~18%** (Qwen) — a real
   hole, exploiting its known limit (it certifies *belief*, and belief is cheaply movable by framing).
4. **Defense — measured, closes it.** The hole is resample-context poisoning. Resampling the
   **canonical** question (not the attacker's framing) + `detect_context_injection` drops confidently-
   fooled to **1.7%** and flags **91%** of the attacks (the framing surfaces as cross-context
   divergence). Deploy rule, now proven: *verdict from the canonical arm, injection-flag from the
   as-asked arm* — exactly the `divergence.py` security model.
5. **Generality.** The whole attack→defense loop reproduces on **Gemma-2-2b** (different vendor):
   fooled 17% → defended 2%, injection flags 80%. Not model-specific.

## Honest scope
- The strong, complete result is **`grounded_honesty`: attack-tested, defended, cross-vendor.** The
  "consistency > signature" *comparison* is only half-done — the probe-collapse arm needs heavy
  RL-obfuscation, deferred.
- Question-framing vector only; one residual tail (~2%) where even canonical resampling is unstable —
  the stability gate should ABSTAIN there. `grounded_honesty` is single-model self-consistency; a
  confidently-wrong shared belief still needs the cross-model **council** arm (also owed).

## Why it matters
When AI builds AI, a monitor is only worth having if an adversary can't cheaply learn to fool it. We
took styxx's honesty monitor, **tried hard to break it, found its weak point, and closed it with
tooling styxx already ships** — across two model families. That is what a defensible integrity signal
looks like: not unbreakable, but *attacked, mapped, and defended in the open.*

## Owed (the next bricks)
1. **RL-obfuscation** of the activation probe — to finish the "consistency > signature" comparison.
2. **Council arm** (cross-model `council_agreement`) — to cover the confident-confabulation residual.
3. Larger / instruction-tuned-frontier models; more attack vectors (few-shot, multi-turn).

Artifacts: `PREREG_*`, `RESULT_*`, `run_*`, figures under `papers/consistency-robustness/`. Commits
70c2cbc → 4601474.
