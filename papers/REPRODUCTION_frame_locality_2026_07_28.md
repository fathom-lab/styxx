# REPRODUCTION GUIDE — Frame-Locality

**For an external replicator. Goal: let a skeptic verify the claims in
`PAPER_frame_locality_2026_07_28.md` in three escalating tiers — from a 60-second certificate check
that needs no GPU, to a full re-run of the weight-attack experiments on one consumer GPU. Everything
below is the actual, committed apparatus; no number in the paper comes from anywhere else.**

## Tier 0 — verify the paper against its receipts (60 seconds, no GPU, no model)

Every quantity in the paper is bound to a receipt JSON by a machine-checkable certificate. Re-derive
the verdict yourself:

```
pip install styxx
python -m styxx.certify papers/PAPER_frame_locality_2026_07_28.md \
  papers/agent-conscience/frame_recovery_result.json \
  papers/agent-conscience/scale_test_result.json \
  papers/agent-conscience/frontier_knowsay_result.json \
  papers/agent-conscience/frontier_recovery_result.json \
  papers/agent-conscience/adjudicated_loop_result.json \
  papers/grounded-honesty-axis/injection_gap_closure_result.json \
  papers/closed-model-frontier/behavioral_sycophancy_b22_result.json \
  papers/read-neq-write/e1_result.json \
  papers/agent-conscience/poisoned_recovery_result.json \
  papers/agent-conscience/kp_recovery_result.json \
  papers/agent-conscience/kp_replication_result.json \
  papers/agent-conscience/coupling_battery_result.json
```

Expected: `OATH-HELD`. Every numeric claim classifies as VERIFIED (grounded in a named receipt) or
ABSTAIN (not a receipt-kind number); **any UNGROUNDED number fails the oath and the document does not
ship.** This catches the failure mode that matters most in AI-assisted research — a number in prose
that no longer matches the data. It happened twice during drafting and the certifier caught both.

**What Tier 0 does and does not prove.** It proves the paper faithfully reports its own receipts. It
does *not* prove the receipts were produced by a sound experiment — that is Tiers 1 and 2.

## Tier 1 — audit the design without running anything (30 minutes, reading)

The strongest check on this kind of work is not re-running it; it is reading what was frozen *before*
the data existed. For each claim, the preregistration was committed as its own commit, before the
scored run:

| Claim | Prereg (committed before the run) | Harness | Receipt |
|---|---|---|---|
| Belief recovers under social pressure | `agent-conscience/PREREG_frame_recovery_2026_07_24.md` | `run_frame_recovery.py` | `frame_recovery_result.json` |
| Gap reaches a frontier model | `PREREG_frontier_knowsay_2026_07_27.md` | `run_frontier_knowsay.py` | `frontier_knowsay_result.json` |
| Frontier recovery, powered | `PREREG_frontier_recovery_2026_07_27.md` | `run_frontier_recovery.py` | `frontier_recovery_result.json` |
| Unregularized weight attack overwrites belief | `PREREG_poisoned_recovery_v2_2026_07_28.md` | `run_poisoned_recovery.py` | `poisoned_recovery_result.json` |
| Knowledge-preserving attack spares belief | `PREREG_kp_recovery_2026_07_28.md` | `run_kp_recovery.py` | `kp_recovery_result.json` |
| …replicated, fresh benchmark + seed | `PREREG_kp_replication_2026_07_28.md` | `run_kp_replication.py` | `kp_replication_result.json` |
| Belief-rewrite costs general capability | `PREREG_coupling_battery_2026_07_28.md` | `run_coupling_battery.py` | `coupling_battery_result.json` |

**What to attack when reading them:**
- **Are the gates frozen and imported, or re-typed?** Mechanism floors (recovery ≥ 0.50, held ≥ 0.80,
  specificity ≥ 0.15, 25-per-cell) are *imported in code* from the module that first froze them
  (`run_frame_recovery.py`), so they provably cannot drift between cycles. Check the imports.
- **Is the specificity control real?** The claim is not "recovery is high" but "recovery is high on
  items the model originally had right and near-zero on ones it had wrong." A design without that
  contrast proves nothing; check it exists in every recovery harness.
- **Was the pool disjoint?** Each run asserts zero overlap with all prior pools *in code* (`assert
  overlap == 0`), not in prose.
- **Were failures reported?** `papers/autopilot/CYCLE_LOG.jsonl` contains the INVALID and
  CLOSED_NEGATIVE cycles alongside the positive ones, including two confessed design errors (an
  arithmetically unreachable bar; a sizing failure). A program that only logs wins is not one you
  should trust.

## Tier 2 — re-run the weight-channel experiments (one 8 GB GPU, a few hours, $0)

The parametric results — the ones the paper leans on hardest — are fully local. No API keys.

```
# environment: python 3.12, torch+cuda, transformers, peft, datasets, numpy
cd papers/agent-conscience

# 1. unregularized attack: does it overwrite the belief?     (~40 min)
python run_poisoned_recovery.py b t c d s

# 2. knowledge-preserving attack, same items                 (~2 h, 4-rung LAM ladder)
python run_kp_recovery.py t v d s

# 3. replication on a fresh benchmark + seed                 (~2.5 h)
python run_kp_replication.py b t v d s

# 4. capability coupling on a disjoint battery               (~15 min, reuses the adapters)
python run_coupling_battery.py e s
```

Each script is phase-addressable and checkpoints per item, so an interrupted run resumes rather than
restarts. Each prints its frozen gates and a `VERDICT:` line, and writes a receipt JSON you can diff
against the committed one.

**Expected verdicts:** `CLOSED_NEGATIVE__weight_attack_reaches_the_belief`,
`SURVIVED__knowledge_preserving_attack_spares_the_belief`, `SURVIVED__kp_dose_result_replicates`,
`SURVIVED__belief_rewrite_coupled_to_capability_damage`.

**Expect the rates to move, and know which ones should not.** Point estimates on cells of 45–70 items
will differ across hardware and seeds — the recovery rate especially, which sits near one-half and
whose interval includes one-half (the paper says so). What *should* reproduce robustly:
- the **sign** of the specificity margin (negative for the unregularized attack, positive for the
  knowledge-preserving one) — this is the load-bearing contrast;
- **bimodality**: essentially every flipped item resolving out of frame to either the original truth
  or the planted target, and nothing else;
- the **capability gap**: a large drop for the belief-overwriting attack, ~none for the belief-sparing
  one. This was the cleanest effect measured (0.5833 → 0.3567 vs 0.5833).

**If the specificity sign or the capability gap fails to reproduce, the paper's central claim is in
trouble and we want to know.** That is the point of sending this.

## Tier 3 — the parts we could not run (where outside help is worth most)

Honest statement of what a bigger lab could settle that we cannot:

1. **Scale.** Every weight-channel result is at 1.5B on one 8 GB card. Does the dose behavior — and
   the coupling — hold at 7B, 70B? We cannot test this.
2. **Attack class.** One LoRA configuration. Full fine-tune, distillation-to-a-clean-student, and
   RMU-style unlearning are unexplored and are the attacks that would matter most.
3. **Vendors.** The open ladder is one model family; the frontier point is one model, one format.
4. **The recovery rate.** A larger pool would separate "about half" from one-half, or show it truly
   sits there.

## Ground rules we ask replicators to keep (and that we kept)

- **Do not move a bar to make a result.** Every floor here was imported from the cycle that froze it.
  If you change one, report it as a different experiment.
- **A missed bar is a negative, not a "near miss."** This program logs near-bar outcomes as
  CLOSED_NEGATIVE; several are in the record.
- **Report the direction that hurts.** The cycle that produced the most valuable result here
  (`CLOSED_NEGATIVE`, the weight-attack wall) was the one that contradicted the hypothesis we
  preferred, and it was published within hours — then itself overturned by the next cycle, which was
  also published.

Questions, or a result that disagrees with ours: we would rather hear it than not.
