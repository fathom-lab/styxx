# PREREG — does the weight-channel result hold at scale? The whole contrast at 3B

**Cycle 91 (operator-directed: "finish our work"). Frozen before any scored run. Every weight-channel
result in the frame-locality paper (cycles 86–90) is at 1.5B — the paper says so, and it is the first
thing an external replicator will question. This run repeats the entire weight-channel contrast at
`Qwen/Qwen2.5-3B-Instruct` (fp16, verified to fit the 8 GB card in training at 7.21 GB peak, so no
quantization confound): the dose reversal (an unregularized attack overwrites the belief; a
knowledge-preserving one spares it) and the capability coupling (overwriting costs general capability;
sparing costs little). Local, $0.**

## Design — one substrate change, everything else imported

Same frozen protocol as cycles 87–90, only the model changes to 3B:

1. **Elicit** greedy neutral answers on a fresh ARC-Challenge pool (`SEED = 910000`, disjoint in code
   from the cycle-88 ARC pool and all prior pools) → partition CORRECT_PRE / WRONG_PRE →
   ATTACK / HELD / CONTROL strata.
2. **Train two adapters at 3B**, identical to the 1.5B recipe (LoRA r=16 / alpha=32 / lr 1e-4 / 300
   steps / micro-batch 4 / gradient checkpointing):
   - **UNREG** — flip the attack items to a deterministic wrong letter, no regularizer.
   - **KP** — the same flip loss plus `LAM · l_replay` preserving the correct answer on HELD under the
     neutral prompt. `LAM = 1.0` (the value cycle 87's frozen ladder selected at 1.5B; frozen here, no
     re-search).
3. **Dose measurement** for each adapter: in-frame flip on ATTACK; out-of-frame neutral recovery on
   ATTACK_FLIPPED; out-of-frame accuracy on CONTROL (specificity control) and HELD (sanity).
4. **Coupling measurement**: BASE / UNREG / KP accuracy on a disjoint MMLU battery
   (`N_BATTERY = 300`, disjoint from cycles 89/90 and from the ARC attack pool).

## Frozen gates

- **V1 (validity — miss ⇒ INVALID):**
  - `V1a`: in-frame off-correct rate ≥ **0.60** for BOTH adapters (both attacks must take at 3B).
  - `V1b`: ≥ **25** ATTACK_FLIPPED, ≥ **25** CONTROL, ≥ **25** HELD (`POWER_GATE`).
  - `V1c`: KP preserves knowledge — HELD out-of-frame accuracy ≥ **0.80** (`LG2_FLOOR`); if the
    3B KP attack cannot both flip and preserve at `LAM=1.0`, that is `INVALID__no_kp_attack_at_3B`,
    reported (a scale-dependent apparatus fact), not forced.
  - Battery `V1d`: BASE MMLU accuracy ≥ **0.40**.
- **SG1 — the dose reversal holds at 3B (the headline):** UNREG specificity margin **< 0** (recovery
  on flipped minus CONTROL neutral accuracy is negative — it overwrites) AND KP specificity margin
  ≥ `LG3_MARGIN` (**0.15**, positive — it spares). The *sign reversal between the two attacks* is the
  claim, imported from the 1.5B result; the KP recovery rate is reported, not re-gated (known to sit
  near one-half).
- **SG2 — the coupling holds at 3B:** UNREG capability drop (BASE − UNREG) ≥ **0.10** AND that drop
  exceeds the KP residual (BASE − KP) by ≥ **0.10** — i.e. overwriting costs materially more general
  capability than sparing, the ~10:1-direction contrast, re-tested at scale.

All floors (`POWER_GATE`, `LG2_FLOOR`, `LG3_MARGIN`, and the 0.60 / 0.10 / 0.40 bars) are the exact
values cycles 87–90 ran under; none moves.

## Pre-committed outcomes

- **V1 + SG1 + SG2 pass** → `SURVIVED__weight_channel_holds_at_3B`. The dose and the coupling are not
  1.5B artifacts; the paper's scope note upgrades from "1.5B only" to "1.5B and 3B, same family," and
  the weight-channel arc has its first scale point.
- **V1 pass, SG1 fail** → `CLOSED_NEGATIVE__dose_reversal_fails_at_3B`. Reported at full volume: the
  sign reversal is scale-dependent; the paper's weight-channel claim is bounded to 1.5B, loudly.
- **V1 pass, SG1 pass, SG2 fail** → `MIXED__dose_holds_coupling_fails_at_3B`. The belief behavior
  generalizes but the capability coupling does not — a real, reportable dissociation.
- **V1 miss** (attack won't take, cells underpowered, or no KP attack at `LAM=1.0`) →
  `INVALID__…`; results withheld; the block named.

## Reported but NOT gated

Both adapters' recovery rates and target-propagation rates; per-cell coupling (the battery is single
MMLU-wide here, not split); the paired 1.5B-vs-3B comparison for every quantity; in-frame flip
leakage onto HELD; training loss tails; first-answer ARC accuracy at 3B.

## Apparatus honesty

- 3B fp16 LoRA training was verified to fit at 7.21 GB peak before this prereg; if a real run OOMs
  anyway (longer sequences), the run is `INVALID__oom` and 4-bit is named as a separate prereg (a
  4-bit substrate would be a quantization confound against the fp16 1.5B results, so it is not
  silently substituted).
- `LAM = 1.0` is frozen from the 1.5B ladder, not re-searched at 3B; if it fails V1c, that is a
  reported scale fact, not a reason to tune.
- Pools disjoint in code from every prior pool and from the two batteries; adapters are fresh
  (trained here), not reused.
- Smoke (few items, 20 steps) writes only `*_SMOKE_INVALID*`.

## Frozen constants

`AGENT_MODEL = Qwen/Qwen2.5-3B-Instruct` (fp16) · attack pool = ARC-Challenge · battery = `cais/mmlu`
all test · `SEED = 910000` · `N_ITEMS = 320` · `N_ATTACK = 70` · `N_HELD = 40` · `N_CONTROL_MAX = 60`
· `N_BATTERY = 300` · `LAM = 1.0` · LoRA r=16 / alpha=32 / lr 1e-4 / STEPS=300 / micro-batch 4 / grad
checkpointing · `N_NEUTRAL = 5` temp 1.0 + one greedy · `POWER_GATE=25` / `LG2_FLOOR=0.80` /
`LG3_MARGIN=0.15` / V1a=0.60 / SG2 drop=0.10 / battery floor=0.40 — imported from cycles 87–90 ·
`letter_of` / `_wrong_letter` / `ASK` / `HELP_SYS` / `ATTACK_SYS` / `ATTACK_ASK` / `_answer_seq` /
`_collate` reused from the cycle-86/87 harnesses. Deterministic; checkpointed JSONL per phase.
