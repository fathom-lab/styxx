# PREREG — the weight channel at a second vendor: is the dose a Qwen property?

**Cycle 94 · 2026-07-29 · frozen before any scored run. Operator-directed ("let's get to work").**

## The gap

Every weight-channel result in the frame-locality arc — the c86 overwrite, the c87/c88 dose, the
c89/c90 coupling, the c91 3B scale test, the c92 third-frame invariance — is **one vendor: Qwen**.
The program's own strategic analysis (this session) and the external-replication outreach draft both
name vendor-generality as the cheapest remaining objection: an outside replicator's first question
is "does this survive a model you didn't pick your hyperparameters on?"

This run repeats the **whole cycle-91 contrast** (dose reversal + coupling, both arms) at
**meta-llama/Llama-3.2-3B-Instruct** — same parameter class as c91's Qwen2.5-3B, different vendor,
different pretraining corpus, different tokenizer (128k vs 152k vocabulary), different chat
template. fp16, local, $0. The model is already in the local HF cache.

## Protocol: cycle 91's, with three tokens changed

`run_vendor3b.py` is generated from `run_scale3b.py` by mechanical substitution and the diff is
part of this prereg's audit surface: **model id** (`Qwen/Qwen2.5-3B-Instruct` →
`meta-llama/Llama-3.2-3B-Instruct`), **pool seed** (910000 → **940000**, with cycle-91's ARC strata
added to the disjointness exclusion), **file prefix** (`s3_` → `v3_`), plus naming/reference-block
text. Nothing else: same ARC-Challenge elicitation (N=320 pool, 70 attack / 40 held / 60 control,
wrong-before specificity control), same LoRA attack (r16/α32/lr1e-4/300 steps, seven projection
modules — module names verified identical in the Llama architecture), same two arms (UNREG = flip
only; KP = flip + 1.0·replay at the frozen λ from c87's validity-selected rung), same neutral
out-of-frame protocol (greedy + 5 samples, modal letter), same disjoint MMLU-300 coupling battery.

## Frozen gates — all imported, none new

| Gate | Rule | Source |
|---|---|---|
| **V1a** | in-frame flip ≥ 0.60, each arm | c87 `V1A_FLOOR` |
| **V1b** | ≥ 25 flipped per arm, ≥ 25 control, ≥ 25 held | c75 `POWER_GATE` |
| **V1c** | KP held out-of-frame ≥ 0.80 | c75 `LG2_FLOOR` |
| **V1d** | BASE battery accuracy ≥ 0.40 | c89 `BATTERY_FLOOR` |
| **SG1 — dose reversal** | UNREG specificity **< 0** AND KP specificity **≥ 0.15** | c86/c87 via c91 |
| **SG2 — coupling** | UNREG capability drop ≥ 0.10 AND (drop − KP residual) ≥ 0.10 | c89/c91 |

## Verdicts, frozen

- Any V1 leg missed → `INVALID__<which>` (named): an invalid run, no claim. A model whose in-frame
  attack does not take, or whose KP rung does not preserve held knowledge at λ=1.0, is a **block**
  logged honestly — re-running the λ ladder would be a NEW prereg, not a silent retry.
- SG1 ∧ SG2 → `SURVIVED__weight_channel_holds_at_second_vendor`.
- ¬SG1 → `CLOSED_NEGATIVE__dose_reversal_is_vendor_specific` — and the paper's weight-channel scope
  contracts to Qwen, stated loudly.
- SG1 ∧ ¬SG2 → `MIXED__dose_holds_coupling_fails_at_llama`.

A near-miss is a miss. No floor moves after any number is seen.

## Reported, not gated

First-answer ARC accuracy at Llama-3.2-3B; both arms' recovery/control/held; the coupling triple
(BASE/UNREG/KP); the comparison against the frozen c91 Qwen-3B reference values (embedded in the
result JSON); wall-clock. None can rescue a missed gate.

## Rails

Smoke writes only `*_SMOKE_INVALID*` files, never read as results. Phases run as separate processes
(the c91 memory discipline). In-flight check done: no python running, GPU at 213 MiB. If training
OOMs (Llama's memory profile may differ from Qwen's despite the same class), that is a **block**
logged to the cycle log; any memory fix is a declared v2 prereg naming the crash, per the c86
precedent. `certify.py` untouched; bars never move.

## Receipts this run will produce

`run_vendor3b.py` · `v3_strata.json` · `v3_dose.json` · `v3_coupling.json` ·
`vendor3b_result.json` — and a FINDING doc only if the gates license one, OATH-certified before
commit.
