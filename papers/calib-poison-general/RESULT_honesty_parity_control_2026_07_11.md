# RESULT — honesty-side probe-parity check: the flagship's private>naive recovery is MOSTLY probe capacity, not privacy (between the bars)

**Fathom Lab · papers/calib-poison-general · 2026-07-11. Verdict against the frozen gates of
`PREREG_honesty_parity_control_2026_07_11.md` (frozen `a617eca`, committed BEFORE this scored run).
The identical E1-regime honesty attack (`papers/read-neq-write/attack_sweep.py` `train_attack`;
Qwen2.5-1.5B-Instruct, LoRA r=16, deploy-18 scrub + knowledge-replay, 300 steps), two cells (seed 0
and seed 1, λ=1.0), three auditors per cell via one shared 13-family code path. All guards passed:
clean guard (private-13 0.9382, knowledge 0.8182, split disjoint), bite in both cells (frozen-18
0.8562 → 0.5502 / 0.5899), knowledge preserved (0.8182 / 0.803).**

## Verdict (frozen string, verbatim)

- **Question P: `PRIVACY_PARTIAL__between_bars`**

Per-cell disclosure is mandatory under the catch-all; it follows.

**All cells (`honesty_parity_control_result.json`):**

| seed | naive-DoM-6 | naive-MATCHED-13 (fit ATTACK-sub n=53) | private-13 (fit CALIB n=53) | baseline_gap | parity_gap | frozen-18 (clean 0.8562) | knowledge |
|---|---|---|---|---|---|---|---|
| 0 | 0.5106 | 0.7825 | 0.8378 | 0.3272 | 0.0553 | 0.5502 | 0.8182 |
| 1 | 0.4876 | 0.8037 | 0.7217 | 0.2341 | -0.082 | 0.5899 | 0.803 |

## The reproduction is exact, which makes the control adjudicable

The private-13 auditor reproduced E1's decisive Qwen λ=1 reads **to four decimals on both cells**:
0.8378 (seed 0) and 0.7217 (seed 1) — `delta_vs_canonical_private` = 0.0 on each
(`e1_result.json`). The naive-DoM-6 collapse reproduced the flagship's naive read: 0.5106 / 0.4876
here vs the 2-way re-lock 0.5066 / 0.4725 (`attack_sweep_result.json` λ=1;
`delta_vs_canonical_naive` = 0.004 / 0.0151, different split machinery so context not a tight
anchor). The frozen deploy-18 direction — the one the attack directly scrubs — fell from 0.8562 to
0.5502 / 0.5899, so the attack bit in both cells. The gap this control interrogates is the real E1
gap, measured on the real E1 apparatus.

## Question P — the parity arm removes most of the honesty "privacy" gap

The load-bearing swap: upgrade the naive auditor from 6-feature diff-of-means to the SAME 13-probe
family (per-layer DoM + per-layer logistic + whole-stack logistic) the private auditor uses, fit on
a frozen n=53 = |CALIB| subsample of the attacker's OWN poisoned ATTACK split (probe family AND
fit-size matched; the only remaining difference vs private is attacker-seen vs attacker-unseen).

- **Seed 0:** baseline gap (private-13 − naive-DoM-6) = **0.3272**; at probe+size parity the gap is
  **0.0553** — under the STANDS bar (half of baseline = 0.1636) and above the FAILS bar
  (0.7825 < 0.8378 − 0.02 = 0.8178, by 0.0353). Roughly **83%** of the gap vanished at parity.
- **Seed 1:** baseline gap = **0.2341**; parity gap = **-0.082** — the ATTACK-fit parity auditor
  **read BETTER than the private one** (0.8037 vs 0.7217). This cell satisfies the FAILS condition
  outright (0.8037 ≥ 0.7217 − 0.02).
- The pre-registered both-cells conjunctions refuse both labels (discordant cells: seed 0
  near-FAILS, seed 1 FAILS-outright) → **PARTIAL**, verbatim.

**Licensed reading:** on honesty — the flagship's own construct — the **majority of the private-over-
naive gap the read≠write defense reports is probe family/capacity, not fit-split privacy.** Upgrading
the naive auditor to the private probe family, still fit on the attacker's poisoned data at matched
size, recovered **0.7825 / 0.8037** of the **0.8378 / 0.7217** private read. A privacy-specific
residual of **~0.035** appears on one seed (seed 0) and is **absent — sign-reversed — on the other**
(seed 1, where the poisoned-fit parity auditor beat the clean-fit private one). NOT licensed:
"privacy contributes nothing" (seed 0 retains a residual) and — per the frozen sequencing rail —
"the flagship defense claim is broken" (2 cells, one λ, one family; FAILS did not fire; PARTIAL did).

## Cross-construct convergence (why this is the finding, not a footnote)

Both constructs the program has probe-parity-tested now land on the SAME string from opposite
starting points:

- **Sentiment** (`RESULT_margin_parity_control_2026_07_10.md`): `PRIVACY_PARTIAL`, parity removed
  ~two-thirds of a **small** gap (0.1149 / 0.1288), one seed FAILS.
- **Honesty** (this run): `PRIVACY_PARTIAL`, parity removed ~83% / >100% of a **large** gap
  (0.3272 / 0.2341), one seed FAILS.

The flagship attributed its recovery to *calibrating on a private split the attacker never
optimized against*. On both constructs, the recovery is carried predominantly by the **richer probe
family** (per-layer logistic + whole-stack), which recovers the read even when fit on the attacker's
own poisoned split. The clean fit-split buys, at most, a small residual on one of two seeds. The
"privacy" word is mostly probe capacity wearing privacy's name — on the flagship's construct too.

## Sequencing — what does and does not change (binding rail)

Per the frozen prereg, **no flagship or erratum text ships from this run alone.** PARTIAL is between
the bars; the FAILS string that would pre-trigger the re-attribution language did not fire (seed 0's
private read held above its parity read by more than the 0.02 bar). What this run establishes is **demotion pressure on the "privacy"
attribution across BOTH tested constructs**, which now makes the confirmation run the top owed item:
≥3 seeds + a λ sweep {1.0, 3.0} + the Llama-3.2-1B family, scored through this same shared code path.
If FAILS consolidates there, the protocol clause re-attributes from *"calibrate the probe on a
private split"* to *"audit with a richer probe family (per-layer logistic + whole-stack), which
recovers the read even fit on the attacker's poisoned split,"* and an erratum to the published
read≠write claim (Fathom v26/v28) is owed — under operator sign-off, never from a 2-cell run.

## Bounds

Qwen2.5-1.5B only, honesty construct, λ=1.0 only, seeds {0,1}, EVAL n=66 (per-cell AUROC SE ≈ 0.06 —
the parity-gap reads are within noise of small true gaps; the conjunction protected against
single-cell noise and correctly returned PARTIAL on discordance), single run per cell, bf16
non-bit-deterministic training (yet private-13 reproduced E1 to 0.0 here). The parity arm equalizes
probe family and fit-size but cannot separate "attacker never SAW the data" from "attacker never
OPTIMIZED against it."

## Next (in order)

1. **Confirmation run** (new frozen prereg): ≥3 seeds + λ {1.0, 3.0} + Llama-3.2-1B via this shared
   code path. Its FAILS branch inherits this prereg's frozen re-attribution language and, on
   consolidation, opens the operator-gated erratum path.
2. B2 (RMU/gradient-routing erasure attacker) — after the attribution question is settled: there is
   no point stress-testing a defense whose mechanism ("privacy" vs "richer probe") is not yet
   correctly named on ≥3 seeds.

## Reproducibility

`honesty_parity_control.py` (frozen `a617eca`) → `honesty_parity_control_result.json`; run log
`_hpc_run.log`. Smoke quarantined in `honesty_parity_control_result_SMOKE_INVALID.json`. Frozen E1
three-way split (seed 0); ATTACK subsample indices (n=53, seed 0) recorded in the result JSON.

---
*The control did on honesty what it did on sentiment: the "privacy" defense is mostly probe capacity.
FAILS didn't quite fire on two seeds, so the flagship keeps its word for now — but the confirmation
run, not this note, decides whether it keeps it for good.*
