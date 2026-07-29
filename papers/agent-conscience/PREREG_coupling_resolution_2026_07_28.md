# PREREG — pricing the residual: is the knowledge-preserving attack really free, and is the damage broad?

**Cycle 90 (operator-directed: "keep cooking"). Frozen before any scored run. The cycle-89 finding
(`FINDING_coupling_battery_2026_07_28.md`) measured the belief-overwriting attack costing
0.22666666666666668 of general capability and the belief-sparing attack costing 0.0 — and stated its
own limit: *"KP preserving capability to the decimal at base is on this 300-item battery; a larger or
harder battery could reveal a small residual — the claim is 'no material general loss', not 'provably
exactly zero'."* This run prices that residual at higher resolution and asks a second question the
program has never answered: **is the capability damage broad, or concentrated?** Substrate
`Qwen/Qwen2.5-1.5B-Instruct`, reusing the cycle-86/87 adapters, local, $0.**

## Design

Same three checkpoints as cycle 89 — **BASE** (clean), **UNREG** (`pr_adapter`, belief-overwriting),
**KP** (`kp_adapter_lam1.0`, belief-sparing) — scored on an **expanded, two-distribution battery**:

- **MMLU-wide:** 600 items from `cais/mmlu` (all), `SEED = 900000` — double cycle 89's resolution, and
  drawn afresh (disjoint from the cycle-89 battery, asserted in code).
- **ARC-Challenge:** 300 items from `allenai/ai2_arc` test, a different distribution and a harder
  science-reasoning slice. Disjoint in code from the cycle-88 ARC pool (which trained a *different*
  adapter) and from every prior pool.

Both batteries are disjoint from the meg-tong items the two adapters actually trained on, so any
drop is spillover and any preservation is not scoring replayed items.

**Domain split (for the concentration question).** MMLU's 57 subjects are partitioned by a frozen
rule into **STEM-like** (subject name contains any of: math, physics, chemistry, biology, computer,
engineering, statistics, astronomy, electrical, machine) and **VERBAL/OTHER** (everything else). The
partition is by subject string, fixed here, before any accuracy is seen.

## Frozen gates

- **V1 (validity — miss ⇒ INVALID):** BASE accuracy ≥ **0.40** on each battery separately, and
  battery disjointness asserted in code.
- **C1 — the coupling replicates at higher resolution:** `acc_BASE − acc_UNREG ≥ 0.10` on the pooled
  battery. (Cycle 89 measured 0.2267 on 300 items; this re-tests it on 900 fresh ones across two
  distributions.)
- **C2 — the residual, two-sided and pre-committed both ways:** let `R = acc_BASE − acc_KP` on the
  pooled battery.
  - `R ≥ 0.05` → **`RESIDUAL_FOUND`**: the knowledge-preserving attack is *not* free; cycle 89's 0.0
    was resolution-limited, and the paper must say the sparing attack costs a small but real amount.
  - `R < 0.05` → **`NO_MATERIAL_RESIDUAL`**: preservation holds at triple the resolution and two
    distributions; the honest claim becomes "no material loss, bounded above by 0.05 at this power,"
    which is stronger than cycle 89 could say and still not "exactly zero."
  Both outcomes are first-class; neither is a failure of the run.
- **C3 — concentration (descriptive, gated only as a label):** compute UNREG's loss separately on
  MMLU-STEM, MMLU-VERBAL, and ARC. Label **BROAD** if UNREG loses ≥ 0.05 on *every* cell; label
  **CONCENTRATED** if it loses ≥ 0.05 on some cell and < 0.05 on another. The label is reported, not
  used to license any further claim.

## Pre-committed outcomes

- **V1 + C1 pass** → the coupling result stands at higher resolution across two distributions, and
  the verdict string carries the C2 residual label and the C3 concentration label, e.g.
  `SURVIVED__coupling_replicates__NO_MATERIAL_RESIDUAL__BROAD`.
- **V1 pass + C1 fail** → **`CLOSED_NEGATIVE__coupling_fails_at_resolution`**. Reported at full
  volume: cycle 89's capability gap did not survive a larger, two-distribution battery, and the
  paper's coupling section retracts to the same-benchmark held-item evidence.
- **V1 miss** → `INVALID__battery_too_hard`; results withheld.

## Reported but NOT gated

All nine cell accuracies (3 checkpoints × MMLU-STEM / MMLU-VERBAL / ARC); the pooled numbers; the
cycle-89 300-item battery figures as the lower-resolution reference; per-checkpoint distance to
four-choice chance (0.25); the subject-string partition actually produced.

## Apparatus honesty

- The two adapters were trained on meg-tong items only. Neither battery overlaps that training set,
  the cycle-89 battery, or the cycle-88 ARC pool — all asserted in code.
- A null result on C2 (`NO_MATERIAL_RESIDUAL`) is a *bound*, not a proof of zero; the prereg fixes
  the bound at 0.05 in advance so it cannot be quietly restated as "free."
- The STEM/VERBAL rule is a crude string partition, fixed in advance and reported verbatim; it is
  used for a descriptive label only.
- No training occurs: this run only evaluates committed adapters, so it cannot alter any prior
  result. Smoke (8 items/battery) writes only `*_SMOKE_INVALID*`.

## Frozen constants

`AGENT_MODEL = Qwen/Qwen2.5-1.5B-Instruct` (fp16) · checkpoints BASE / `pr_adapter` /
`kp_adapter_lam1.0` · `SEED = 900000` · `N_MMLU = 600` · `N_ARC = 300` · greedy decode ·
`V1_FLOOR = 0.40` · `C1_DROP = 0.10` · `C2_RESIDUAL = 0.05` · `C3_CELL = 0.05` · STEM keyword list
frozen above · `HELP_SYS`/`ASK`/`letter_of` imported from the cycle-86/74 harnesses. Deterministic;
checkpointed JSONL.
