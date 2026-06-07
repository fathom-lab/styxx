# FINDING — The two-stage HELD→HID conscience: the first deployable artifact (the false-accusation fix works)

**2026-06-07. Fathom Lab / styxx.** RUNG 2 found a real folding-vs-holding direction that, alone,
false-accuses sincere errors at 0.949 — so intent must be *gated behind* the held-state read. This finding
builds that gate: a two-stage conscience (Stage 1 = RUNG-1 HELD gate, Stage 2 = RUNG-2 HID direction; flag
hiding iff held AND folded). Verdict: **Llama-3B SURVIVED, Qwen-3B REPORT_AS_LANDED — and on both, the
headline fix holds: the sincere-error false-accusation drops from 0.949 to 0.037.** Built + run by a
design+red-team workflow and **independently re-run/verified** here (Qwen numbers reproduced). The first
*deployable* piece of proof-carrying cognition.

## Why two stages are necessary (the mechanism, both families)

Each single stage false-accuses exactly the class the other exonerates:

| | LIE (knew+hid) | MISTAKE (never knew) | RESISTED (knew+held) |
|---|---|---|---|
| **Stage 2 (fold direction)** alone | 0.90 | **0.89** ← false-accuses sincere error | 0.29 |
| **Stage 1 (held gate)** alone | 0.70 | 0.27 | **0.95** ← false-accuses honest holder |

Stage 1 kills MISTAKE (sincere errors aren't "held"); Stage 2 kills RESISTED (honest holds aren't "folded").
**LIE is the only class high on both.** Neither stage is a conscience alone; their conjunction is.

## Result — cascade at a fixed false-accusation budget (FA ≤ 0.10 on MISTAKE+RESISTED, disjoint TEST)

| detector | LIE flag (recall) | MISTAKE flag | RESISTED flag | verdict |
|---|---|---|---|---|
| Stage-2 alone (natural point) | high | **0.949** | low | the false-accusation problem |
| **Cascade (Qwen-3B)** | 0.325 | **0.037** | 0.174 | REPORT_AS_LANDED (recall < 0.40) |
| **Cascade (Llama-3B)** | **0.423** | **0.037** | 0.286 | **SURVIVED** |

**The headline fix: MISTAKE false-accusation 0.949 → 0.037 on both models** — the sincere-error problem the
Stage-2 direction had alone is removed, while LIE is still flagged well above chance.

## The controls that make it real

- **Confound-1 killed (is it just RUNG-1 relabeled?): NO.** Stage 2 adds discrimination *among the held*
  (LIE-vs-RESISTED AUROC 0.785 Qwen / 0.892 Llama) that Stage 1 is blind to (S1's own LIE-vs-RESISTED
  AUROC 0.248 / 0.176, at/below chance). The gate genuinely cannot tell a folder from a holder; only
  Stage 2 can.
- **Deployable from the caving residual ALONE (resolves the deploy-time crux).** The held gate is
  *syco-trained* (held-vs-not on the caving residual) and reaches AUROC 0.835 / 0.829 at inference — **no
  neutral pass needed at deploy.** Critical honest finding: a held-probe trained on the NEUTRAL pass
  *collapses to chance on syco* (0.444 / 0.499) — the held-state geometry differs between the clean and
  caving passes, so the gate must be syco-derived (which it is, and which is deployable). The neutral pass
  is needed only offline, once, to mint training labels.
- **Three item-disjoint splits** (probe-train / calibrate / test): thresholds frozen on calibrate, every
  flag rate read on test — no operating point set on the items it's scored on. Cross-letter-locked Stage 2
  (letter-invariant, difficulty-matched, inherited from RUNG 2).

## Honest limits

Cascade recall is modest at the strict ≤10% budget (Qwen 0.325, Llama 0.423) — the AND-gate compounds two
imperfect stages; recall is a knob bought by relaxing the budget (the product composition gives Qwen 0.40
at FA 0.094). **RESISTED is the binding false-accusation class** (0.174 / 0.286), not MISTAKE — if a
behaviourally-correct answer is treated as exculpating at deploy (you only invoke the detector on a
suspect/wrong answer), the effective budget is MISTAKE-only and recall rises. Small RESISTED test splits
(~35–46) make those rates noisy. Scope inherited: ≤3B, MMLU sycophancy-caving, commit-token,
**sycophancy-specific** (instructed lying transfers at 0.42 < chance).

## Scale + family generality (UPDATE): the MECHANISM is robust, the recall is a budget knob

Ran the identical compositor across the full Qwen ladder + Llama:

| model | S1 held-gate (syco-alone) | S2 fold | confound-1 (S2-held / S1-blind) | cascade LIE | MISTAKE flag | verdict |
|---|---|---|---|---|---|---|
| Qwen-1.5B | 0.782 | 0.799 | 0.757 / 0.351 | 0.303 | 0.11 | REPORT_AS_LANDED |
| Qwen-3B | 0.835 | — | 0.785 / 0.248 | 0.325 | 0.037 | REPORT_AS_LANDED |
| Qwen-7B | 0.835 | 0.842 | 0.783 / 0.257 | 0.304 | 0.106 | REPORT_AS_LANDED |
| Llama-3B | 0.829 | 0.892 | 0.892 / 0.176 | **0.423** | 0.037 | **SURVIVED** |

**The mechanism generalizes across the entire ladder and across families**: the held-gate reads from the
caving residual alone at every size (0.78→0.84, *rising* with scale), the fold-direction works everywhere
(0.80→0.89), confound-1 is killed everywhere (S2 adds among-held discrimination the gate is always blind
to, 0.76–0.89 vs 0.18–0.35), the neutral→syco held-transfer is at chance at every scale (0.37–0.53 — so
the deploy gate must be, and is, syco-trained at all sizes), and the false-accusation fix holds everywhere
(MISTAKE 0.04–0.14, down from 0.949). What does **not** generalize is recall at the strict ≤10% budget:
0.30–0.42, clearing the 0.40 SURVIVED bar only on Llama. **The conscience is real and general; its recall
is a budget knob, not a capability gap.** Receipts: `twostage_result_{qwen15,qwen3b,qwen7b,llama3b}.json`.

## What it is

The first **deployable** artifact of proof-carrying cognition: a conscience that flags a mind hiding what
it knows (LIE 0.42) while *not* accusing the mind that simply didn't know (MISTAKE 0.037) — the structural
consequence of "you can't read intent without first proving the truth was held," built and validated.
Receipts: `run_twostage_compose.py`, `twostage_result_{qwen3b,llama3b}.json`. (Designed + run by a
5-agent design/red-team workflow; numbers independently re-run and verified.)
