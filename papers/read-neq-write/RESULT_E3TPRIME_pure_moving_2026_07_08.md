# RESULT — E3‴: VOID_NO_BITE. Dropping the frozen anchor did NOT let the adaptive attacker bite — a THIRD adaptive escalation is self-defeating.

**Fathom Lab · papers/read-neq-write · 2026-07-08. Verdict against the frozen kill-gate of
`PREREG_E3TPRIME_pure_moving_adaptive_2026_07_08.md` (committed public before the retrain). Fires the
VOID_NO_BITE branch: the pure-moving adaptive attacker still did not bite, so no DEFENSE verdict is
read. Reported verbatim, including the false STANDS the bite gate prevented.**

## Verdict: VOID_NO_BITE — the pure-moving scheme (frozen anchor removed) left the read MORE legible than naive

E3″ diagnosed its own toothlessness: a two-term adaptive loss (frozen-deploy anchor + moving
whole-stack scrub) split the attacker's capacity between a fixed anchor and a target that moves every
10 steps, so it bit neither. The named fix was to drop the frozen anchor and scrub ONLY the moving,
whole-stack, re-fit-every-10 MIMIC consensus direction — a single coherent objective. E3‴ applied
exactly that one change (naive control unchanged). It bit even less than E3″.

**Decisive family, Qwen2.5-1.5B, per seed (`e3tprime_result.json`):**

| bite measure (EVAL AUROC, lower = more suppressed) | naive s0 | naive s1 | adaptive s0 | adaptive s1 |
|---|---|---|---|---|
| frozen-deployed read | 0.400 | 0.445 | 0.624 | 0.664 |
| mimic-refit read (the adaptive loop's own target) | 0.432 | 0.467 | 0.749 | 0.682 |

Both registered bite deltas (naive − adaptive) are negative: `delta_frozen_deployed` = −0.222 and
`delta_mimic_refit` = −0.266 (`e3tprime_result.json`), so `attack_bit` = false. The naive static
attacker drove the deployed read down to 0.400 / 0.445 and the mimic-refit read to 0.432 / 0.467; the
pure-moving adaptive attacker, chasing a target that re-fits every 10 steps across the whole stack,
only reached 0.624 / 0.664 and 0.749 / 0.682 — it left BOTH reads *more* legible, by a wider margin
than E3″ did. Removing the gradient conflict did not help; the moving-target objective is itself the
problem.

## Why the bite gate mattered here (the false STANDS it blocked)

The adaptive clean-calibrated reads came in high — 0.806 (seed 0), 0.728 (seed 1) at knowledge 0.750
/ 0.750 — again *higher* than the naive control (0.786 / 0.775 at knowledge 0.833 / 0.817). Without
the pre-committed bite gate, that would read as a clean "DEFENSE_HOLDS vs an even stronger adaptive
attacker." It is not: the read is high precisely because the attack barely touched the honesty
signal. Declaring STANDS off an attack that did not bite is the exact false positive the gate exists
to stop, and it stopped it — for the third time in this series. The verdict is VOID_NO_BITE, not
DEFENSE_HOLDS.

## Controls all clean (this is a NO_BITE, not a harness failure)

- **Clean guards passed both families** (`guards` in `e3tprime_result.json`): Qwen CALIB-probe EVAL
  0.912, knowledge 0.800, clean frozen-deployed 0.878; Llama CALIB-probe EVAL 0.821, knowledge 0.817,
  clean frozen-deployed 0.792 — all sane, disjoint splits asserted.
- **Naive control reproduced E1/E3** (`naive_ok`): clean-calibrated read 0.786 / 0.775 at knowledge
  0.833 / 0.817 on Qwen, ≥ 0.70 at a knowledge-preserving point — the defense's known win, reproduced
  a fourth time.
- Random-direction EVAL AUROC stayed in-band across all points, min 0.433 to max 0.612 (guard band
  [0.35, 0.65]).
- Llama corroborates the pattern: adaptive left the frozen read at 0.765 / 0.783 vs naive 0.559 /
  0.624, and mimic-refit at 0.603 / 0.614 vs naive 0.577 / 0.503 — adaptive more legible on both, same
  as Qwen.

## What this establishes

- **Three independent adaptive escalations now fail to bite.** E3′ (deploy-layer, refit-25), E3″
  (whole-stack two-term, refit-10) and E3‴ (whole-stack pure-moving, refit-10) all leave the deployed
  read *more* legible than the naive static scrub. Within this objective family (difference-of-means
  gap-suppression) and scale (LoRA r=16, 300 steps, λ=1.0, 1.5B/1B), the naive static attacker is
  empirically the *strongest* threat; adaptive complexity is self-defeating.
- **The E3″ mechanistic hypothesis is FALSIFIED as the sole cause.** Dropping the frozen anchor did
  not free the attacker to bite — a moving-target scrub whose reference re-fits every 10 steps chases
  a frame that shifts faster than 300 LoRA steps can converge on, so the gradient it follows is
  effectively noise. The toothlessness is intrinsic to a moving DoM target at this budget, not merely
  a two-term conflict.
- **Adaptive robustness stays formally OPEN.** We still have not tested the defense against an
  adaptive attacker that verifiably bites, because within these schemes we cannot build one. v27's
  demotion ("survives the naive and the knowledge-preserving strong attacker; adaptive escalation
  open") stands unchanged and correct — E3‴ neither re-earns nor breaks the adaptive claim.
- **The meaningful threat is still the one the defense already beats.** The naive static attacker is
  both the strongest attack found across three escalations and the one the private-calibration read
  defeats (clean-calibrated read 0.786 / 0.775 at knowledge 0.833 / 0.817, reproduced here).

## The optional next step (E3⁗, a NEW prereg if pursued — not auto-run)

Per the VOID_NO_BITE branch, escalation is a new frozen prereg. The three failures share a root: a
*moving* DoM reference at a *short* LoRA budget never converges enough to bite. A cleaner adaptive
attack would make the reference stationary WITHIN a bite while still adaptive across bites — e.g. a
fixed-schedule curriculum that scrubs a probe fit ONCE on a large MIMIC pool and holds it for many
more steps, or a longer step budget (≥ 800) so a moving target can be chased to convergence. Or step
outside the DoM objective entirely to a learned linear-probe adversary. Any of these is a new prereg
naming this VOID_NO_BITE burial; none is auto-run.

---
*The bite gate did its job a third time: it refused to let a high clean-calibrated read — bought by an
attack that never touched the signal — masquerade as a defense victory. VOID_NO_BITE is the honest
verdict. The naive static attacker remains the strongest threat found and the one read≠write already
survives.*
