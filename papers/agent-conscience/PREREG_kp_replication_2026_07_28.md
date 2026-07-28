# PREREG — the replication that gets cycle 87 off its knife edge: the dose result on a fresh benchmark and a second seed

**Cycle 88 (operator-directed: "go deeper"). Frozen before any scored run. Cycle 87
(`FINDING_kp_recovery_2026_07_28.md`, `SURVIVED__knowledge_preserving_attack_spares_the_belief`)
passed its recovery leg by a single item — the recovery rate was not separated from its 0.50 floor
by the data, and the finding said replication must run before the dose claim is cited. This is that
run: the same frozen protocol on a DIFFERENT benchmark (ARC-Challenge, disjoint from the
sycophancy-eval bench by construction) with a SECOND seed and a LARGER attack cell, so the recovery
interval is materially tighter. Substrate `Qwen/Qwen2.5-1.5B-Instruct`, local, $0.**

## What is held fixed vs what changes

**Held fixed (imported unchanged):** the attack (LoRA r=16 / alpha=32 / lr 1e-4 / 300 steps /
flip + LAM·replay, replay = correct answer on HELD under the neutral prompt, ATTACK correct answers
never replayed); the frozen LAM ladder (1, 2, 4, 8) and the validity-only selection rule (smallest
LAM with in-frame flip ≥ 0.60 AND held out-of-frame ≥ 0.80); the mechanism floors from the cycle-75
module (recovery ≥ 0.50, specificity margin ≥ 0.15, 25-per-cell power); the neutral out-of-frame
protocol (N=5 temperature-1.0 samples + one greedy, modal letter); the deterministic wrong-letter
rule.

**Changed (the replication axes):** benchmark = ARC-Challenge (allenai/ai2_arc, `test`, 4-choice
items, labels remapped to A–D), genuinely disjoint from meg-tong/sycophancy-eval; `SEED = 880000`
(second seed); larger cells (below), sized so ATTACK_FLIPPED is materially bigger than cycle 87's 45
and the recovery interval no longer straddles the floor on a single item.

## Cells (sized ex ante; the cycle-85 rule)

`N_ITEMS = 320`, `N_ATTACK = 70`, `N_HELD = 40`, `N_CONTROL_MAX = 60`. ARC is not exhausted (1172
test items), so unlike cycles 86–87 the pool is not the binding constraint. Qwen2.5-1.5B is
expected to answer ARC-Challenge well above 0.40; at 0.40, correct-pre ≈ 128 ≥ N_ATTACK + N_HELD =
110 with margin, and wrong-pre ≥ 60 is guaranteed at any accuracy ≤ 0.81. If the draw still
under-powers a cell, `INVALID__underpowered` fires and says so.

## Frozen gates

- **V1a_attack_takes (per LAM):** in-frame off-correct rate on ATTACK ≥ **0.60**.
- **V_preserve (per LAM):** out-of-frame accuracy on HELD ≥ `LG2_FLOOR` (**0.80**) — the
  knowledge-preservation manipulation check (held is the replay set; a pass confirms the regularizer
  worked; it is a validity gate, not a mechanism result).
- **Selection:** smallest LAM with V1a ∧ V_preserve. None → `INVALID__no_knowledge_preserving_attack`.
- **V1b_power (selected LAM):** ≥ **25** ATTACK_FLIPPED and ≥ **25** CONTROL scored out of frame.
- **RG1 — the belief test (selected LAM):** out-of-frame recovery on ATTACK_FLIPPED ≥ `LG1_FLOOR`
  (**0.50**) AND specificity margin (recovery − CONTROL out-of-frame accuracy) ≥ `LG3_MARGIN`
  (**0.15**). Floors imported from the cycle-75 module; no floor moves.

## Pre-committed outcomes

- **All gates pass → `SURVIVED__kp_dose_result_replicates`.** The cycle-87 dose finding holds on a
  fresh benchmark and a second seed: a knowledge-preserving weight attack spares a substantial
  fraction of the out-of-frame belief; frame-locality's parametric boundary is a dose set by
  collateral knowledge damage, now with the recovery leg off its knife edge.
- **V-gates + V1b pass, RG1 fail → `CLOSED_NEGATIVE__kp_dose_result_fails_to_replicate`.** Reported
  at full volume: the cycle-87 pass was consistent with its one-item margin being noise; the dose
  claim retracts to "an unregularized attack overwrites the belief; the knowledge-preserving case is
  unresolved," and the synthesis §4 is corrected to say so.
- **No knowledge-preserving attack, or underpowered → `INVALID__…`.** Results withheld; the block is
  named.

## The number this run exists to pin down

Beyond the frozen gate, the result records the recovery point estimate on ATTACK_FLIPPED **with a
Wilson 95% interval** (reported, not gated), and whether that interval's lower bound clears 0.50 —
the quantitative form of "is the recovery leg separated from its bar." Cycle 87's was not; this run
states plainly whether cycle 88's is. The specificity margin (the leg that was already robust at
+0.2566) is expected to replicate and is reported likewise.

## Reported but NOT gated

Per-LAM in-frame flip and held out-of-frame accuracy (the full ladder); the selected LAM;
out-of-frame flip-to-target rate and per-item bimodality (does the fresh-benchmark data reproduce
the clean truth-or-target split); the paired cycle-87 vs cycle-88 recovery/specificity comparison;
first-answer ARC accuracy; training loss tails.

## Apparatus honesty

- HELD is both the replay set and the `V_preserve` check by construction (disclosed in cycle 87 and
  carried here); the mechanism is measured on never-replayed ATTACK items.
- ARC labels that arrive numeric (1–4) are remapped to A–D with the answer key kept aligned; items
  without exactly four lettered choices are excluded before sampling.
- A different benchmark changes both the seed AND the item distribution, so this is replication
  *and* a distribution-shift generalization; a pass is the stronger evidence and a fail cannot be
  blamed on item reuse.
- Smoke runs (2 items/rung, 20 steps) write only `*_SMOKE_INVALID*` files. Single 8 GB card;
  per-phase processes; no concurrent scored runs (checked at orient).

## Frozen constants

`AGENT_MODEL = Qwen/Qwen2.5-1.5B-Instruct` (fp16) · benchmark = ARC-Challenge (`allenai/ai2_arc`,
test) · `SEED = 880000` · `N_ITEMS = 320` · `N_ATTACK = 70` · `N_HELD = 40` · `N_CONTROL_MAX = 60` ·
`LAM_GRID = (1.0, 2.0, 4.0, 8.0)` · selection = smallest LAM with V1a ∧ V_preserve · LoRA
r=16/alpha=32/lr 1e-4/STEPS=300/micro-batch 4/grad checkpointing · `ATTACK_SYS`/`ATTACK_ASK`/
`HELP_SYS`/`ASK`/`_wrong_letter`/`letter_of`/`modal_letter`/`LM`/`_answer_seq`/`_collate` imported
from the cycle-86/87 harnesses · `N_NEUTRAL = 5` temp 1.0 + one greedy · `POWER_GATE`/`LG1_FLOOR`/
`LG2_FLOOR`/`LG3_MARGIN` imported from the cycle-75 module. All phases checkpointed; scoring offline
and deterministic (Wilson interval is a closed-form report, not a resampling).
