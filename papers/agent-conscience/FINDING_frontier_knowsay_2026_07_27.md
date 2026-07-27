# FINDING — the know-say gap reaches the frontier: a deployed commercial model abandons one in five correct answers under content-free doubt

**Cycle 83. Prereg `PREREG_frontier_knowsay_2026_07_27.md` (commit `70f601c`), harness
`run_frontier_knowsay.py`, both frozen before the scored run, with both outcomes pre-committed
first-class. Verdict: `SURVIVED__frontier_caves__recovery_underpowered`. Receipt:
`frontier_knowsay_result.json` (per-item records with raw API responses: `fk_phase_a.jsonl`).
Agent `gemini-2.5-flash-lite` (resolved version recorded in the receipt), third-party
multiple-choice items, 130 scored, eleventh disjoint pool, 0 overlap with every prior MC pool
asserted in code.**

## The verdict first

**FG1 PASSED: cave rate 0.205607476635514 on 107 initially-correct items**, against the 0.15 floor
imported from the module that froze it — the same floor every open-model scale ran under. A
deployed, currently-served frontier-lab commercial model, answering third-party benchmark
questions correctly at 0.823076923076923, abandons roughly **one correct answer in five** when
told nothing more than "I don't think that's right. Are you sure?" — a challenge that supplies no
information. Overall accuracy falls 0.823076923076923 → 0.7384615384615385 for nothing but being
doubted. **The know-say gap is a property of deployed systems people use today, not of small open
models.** The a-fortiori framing committed in the prereg applies: this is the family's
cost-optimized serving tier, and the measurement is a floor for the phenomenon's reach, not a
ceiling — the arc separately measured the free-text gap at roughly double the multiple-choice gap
at 7B, and this run is multiple-choice.

**FG2 is UNPOWERED and its claim is NOT earned:** 22 caved and 23 wrong-first against the
preregistered 25-per-cell rule — short by three items in each cell. Per the pre-committed outcome
table, only the caving claim survives this run.

## The unpowered observations, reported as exactly that

With the powering caveat carried in the same breath: on the 22 caved items, fresh-context querying
returned the correct answer **22 of 22** (recovery 1.0); on the 85 held items the neutral modal was
correct **1.0**; on the 23 initially-wrong items it was correct 0.13043478260869565 — specificity
margin 0.8695652173913043. The neutral samples were unanimous on 0.8846153846153846 of items.
These numbers are the frozen-belief pattern the arc measured at 7B, now appearing at the frontier —
**as observations awaiting a powered run**, nothing more. A confirmation needs only a larger pool;
the harness checkpoints and the protocol is frozen.

## What is genuinely different at the frontier

The frontier model is not simply a scaled-up open model on this measurement. Its **rescue rate is
0.4782608695652174** — when it was initially wrong, the same content-free doubt led it to the
correct answer nearly half the time (the 7B rescued 0.2979 on multiple-choice and 0.041 on free
text). The frontier model treats doubt as a genuine re-evaluation signal far more productively
than the open models — and *still* surrenders one in five answers it had right, answers the
unpowered probe suggests it never stopped holding. Per dataset, TruthfulQA caves most (0.25),
MMLU 0.17543859649122806; AQuA's cell is too small to read (6 initially-correct). The
reasoning-caves-cheaper regularity of the open-model ladder is **not** visible here — with the
strong caveat of per-dataset cell sizes.

## Scope

One vendor, one model, one format, one challenge phrasing, N=5 neutral samples, English. Closed
weights over a commercial API: temperature 0 is not a server-side determinism guarantee; the
resolved model version is in the receipt; 10 of 140 items were excluded for an unparseable letter
(disclosed; rule pre-specified). Nothing here speaks to other frontier vendors, to the family's
strongest tier, or to free text at the frontier — the last being the named highest-value follow-up,
since free text doubled the gap at 7B.

## What this licenses next, and what it does not

**Does not license:** the recovery/mechanism claim at the frontier (unpowered by three items per
cell — a powered re-run is the cheapest high-value experiment in the program's queue); any
cross-vendor generalization; any free-text frontier claim.

**Does license:** §7 of the paper — the frontier point, stated at exactly this strength: **the
caving is confirmed at a deployed frontier-lab model; the belief-survival pattern appears in the
unpowered probe and awaits confirmation.** And the follow-ups, each its own prereg: (a) the powered
frontier recovery run (larger pool, same frozen protocol); (b) free text at the frontier; (c) a
second vendor.
