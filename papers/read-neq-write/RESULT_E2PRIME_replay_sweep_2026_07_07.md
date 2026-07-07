# RESULT — E2′: STANDS_vs_strong. Forced to preserve knowledge, the 4×-capacity whole-stack attacker cannot blind a private-calibrated read.

**Fathom Lab · papers/read-neq-write · 2026-07-07. Verdict against the frozen kill-gate of
`PREREG_E2PRIME_replay_sweep_2026_07_07.md` (committed public before the retrain). Resolves the E2
INCONCLUSIVE by pushing the strong attacker into the knowledge-preserving regime with a higher
knowledge-replay weight, then reading the defense there.**

## Verdict: STANDS_vs_strong — the strongest STANDS this bench supports

E2's r=64 whole-stack attacker at λ=1.0 over-damaged knowledge, leaving no knowledge-preserving point
to judge. E2′ swept the knowledge-replay weight λ ∈ {3, 10, 30}. On the decisive family this bought
genuine knowledge-preserving points where the attack still bit, and at those points the
clean-calibrated read held well above chance. Clean guards passed both families (Qwen CALIB-probe
EVAL AUROC 0.938, knowledge 0.818, frozen-stack read 0.786; Llama 0.879, 0.803, 0.665).

**Decisive family, Qwen2.5-1.5B — the knowledge-preserving points (`e2prime_result.json`):**

| λ | seed | clean-calibrated read (EVAL AUROC) | EVAL knowledge | frozen whole-stack read | random dir |
|---|---|---|---|---|---|
| 10.0 | 1 | **0.759** | 0.833 | 0.553 | 0.520 |
| 30.0 | 1 | **0.754** | 0.818 | 0.533 | 0.521 |

At both points the attack bit hard — the frozen whole-stack read fell from a clean 0.786 to 0.553 /
0.533 — yet the private-calibrated read stayed at 0.759 / 0.754 while knowledge was preserved at
0.833 / 0.818 (random direction 0.520 / 0.521). A 4×-capacity scrub applied at every scan layer,
forced to keep the model's knowledge, still cannot erase honesty from a read calibrated on data it
never saw. This is the strongest STANDS the 8 GB bench allows.

## Caveats that keep this honest (do NOT drop)

- **The STANDS rides seed 1.** On Qwen, only seed 1 reached the knowledge-preserving regime (at λ=10
  and λ=30); seed 0 never preserved knowledge at any λ (0.652 / 0.682 / 0.697, all below 0.75), so it
  yields no scored point. The decisive STANDS rests on one seed at two λ values, not both seeds — the
  frozen gate requires ≥1 decisive seed, which is met, but this is a single-seed hold and a second
  clean seed is owed.
- **Llama corroborates only weakly.** The corroborating family reached knowledge-preserving points
  (λ=3 s0 knowledge 0.848; λ=10 s1 0.788; λ=30 s0 0.818) but its clean-calibrated read there was only
  0.599 / 0.584 / 0.586 — in the ambiguous band, not ≥ 0.70, and its clean frozen-stack read was
  weaker to begin with (0.665). Llama neither confirms nor breaks STANDS; the verdict rides Qwen.
- **Scope unchanged:** LoRA r=64, 600 steps, whole-stack DoM scrub, 1–1.5B, small EVAL (66 facts),
  linear probes. "STANDS_vs_strong" means this attacker, forced to preserve knowledge, did not break
  the private-calibration read on the decisive family — not that no attacker can.

## Where the arc stands after today

foundation → stage-2 apparent EVADABLE → E1 (calibration poisoning; STANDS vs naive) → E2
(INCONCLUSIVE, attacker over-damaged knowledge) → **E2′ (STANDS vs a knowledge-preserving strong
attacker, decisive family)**. Together with E3′'s NO_BITE (which demoted the separate "adaptive"
claim), today's escalations strengthened the strong-attacker axis and honestly corrected the adaptive
axis. The private-calibration defense holds against the naive attacker (E1) and against a
4×-capacity whole-stack attacker forced to keep knowledge (E2′); robustness to a genuinely stronger
*adaptive* attacker is open (E3″, owed).

## Reproducibility

`e2prime_replay_sweep.py` (reuses E2's `train_multilayer` with λ overridden per point, and E1's
clean-calibrated auditor) → `e2prime_result.json`. Prereg frozen public before the retrain.

---
*STANDS on the strong-attacker axis, earned at a genuine knowledge-preserving point, with the
single-seed reach and the weak Llama arm named rather than buried.*
