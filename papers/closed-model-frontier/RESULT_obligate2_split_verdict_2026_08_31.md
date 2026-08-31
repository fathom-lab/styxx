# RESULT — OBLIGATE-2, a split verdict: the first held-out precision pass in the family, and it still does not ship

Fathom Lab · 2026-08-31 · Prereg: `PREREG_obligate2_2026_08_31.md`, frozen before code.
Receipts: `obligate2_result.json`, `obligate2_packets.json`, `obligate2_seat_outputs.json`,
`obligate2_dev_eval.json`. Nine fresh seats, every one 16 of 16 on decoys, unanimity 0.9881,
zero NO-MAJORITY.

## The verdict, up front

Two gates passed and two failed. Per the frozen protocol the clause **does not ship**, and
the two mandated sentences publish verbatim:

- G-O2R: *"the clause is a rounding error on the gap"* — weighted recall 0.0916 against the
  0.10 floor.
- G-O2BAR: *"the conjunct is amputating the wrong limb"* — the discarded bar-band
  adjudicated 10/30 CLAIM against a 0.30 maximum. One adjudication over the line, and a
  frozen line does not move for one adjudication.

## What passed — and why it matters anyway

| gate | outcome | number |
|---|---|---|
| G-O2P precision ≥ 0.70 | **PASS** | 44/59 → **0.7458** |
| G-O2NULL beats null AND corpse | **PASS** | 0.7458 > 0.5678 (null) and > 0.4483 (OBLIGATE-1) |
| G-O2R weighted recall ≥ 0.10 | **FAIL** | 0.0916 |
| G-O2BAR bar-band CLAIM < 0.30 | **FAIL** | 0.3333 |

**0.7458 is the first held-out precision survival in the structural obligation family.** The
word lists died at 0.40; OBLIGATE-1 died at 0.4483; the bar-markers carried the same base
over the 0.70 line on fresh blind tokens. The bar-blindness diagnosis was correct: removing
threshold-shaped tokens converts an accusation generator into something precision-grade.

## What failed — and what each failure teaches

**The recall failure is inherited, not inflicted.** Decompose the weighted misses: the
discarded bar-band contributes ~111 weighted claims, but the general remainder — tokens the
2-decimal-outside-code BASE never reaches at all — contributes ~624. The recall ceiling
belongs to OBLIGATE-1's precision-shape conjunct: most unchecked claims in this corpus are
integers, one-decimal figures, and code-adjacent values the base was never allowed to see.
The bar-markers cost recall at the margin; the base caps it at the root.

**The bar-band failure is a real discovery about bars.** One in three discarded
threshold-adjacent tokens is, per the blind panel, a genuine claim — lines like *"accuracy
0.9850 against a 0.9341 bar"* carry a result inside a bar's blast radius, and window-based
markers cannot see which side of the comparison the token stands on. The prereg predicted
exactly this cost and set the 0.30 line to price it; the price came in at 0.3333. The
information is directional: the marker needs to know the token's **role in the comparison**,
not its distance from the comparator.

The fresh negative arm also adjudicated 0.4483 CLAIM — far above the first cycle's 0.2281 —
a reminder, reported not smoothed, that abstention-band composition moves between samples
and that no rate from either sample deserves an extra decimal of confidence.

## Every error, listed

All 15 positive-arm false flags and all 10 discarded bar-band claims are enumerated in the
receipt, token by token, line by line. No summary replaces them.

## What this licenses, and does not

It licenses nothing to ship: G-O2REG never ran. It does license the next freeze with sharp
priors: (a) the recall problem lives in the base shape and no bar-marker tuning can touch
it; (b) the bar-marker needs comparison-role sensitivity, not wider or narrower windows;
(c) precision-grade structural obligation is **possible** — 0.7458 held-out proves the
family is not cursed, only unfinished. Per standing rule, the successor is named and NOT
built here.

## This document is OATH-FAILED — the denominator specimen, third RESULT in a row

The certificate accuses `44` and `59` inside the gate table's "44/59" — correct values, both
present in the receipt, read by range-sanity as insane precisions because they share a line
with the word "precision". This is the same instrument defect the last two RESULTs paid and
named: **denominator blindness on metric lines**. Three specimens in three consecutive
papers is no longer a curiosity, it is a queue — the defect graduates to a repair candidate
with its own future freeze. Until that repair earns its licence the way everything here must,
the tax is paid in public and the numbers stand unedited.

**CORRECTION (2026-08-31, appended before the repair cycle):** this note attributed the
accusation to the range-sanity clause. Empirical re-diagnosis against the stored ledger shows
`obligation_source: vocabulary` with empty hits — the true mechanism is the v0.3
COUNT-BINDING filter, which requires an integer's receipt leaf to share path vocabulary with
the claim's line, and correct fields named `valid`/`claims` are not echoed in prose. The
accusations were wrong for the reason stated; the clause blamed was not the one firing. The
misdiagnosis is corrected here at the same prominence, and the repair prereg targets the
mechanism the ledger actually names.


## Limits

n=59/30/29 per arm; no significance claimed anywhere; one model family throughout; seats saw
one line of context; DEV telemetry (0.8462 on spent tokens) predictably exceeded the fresh
result — regression to the mean, again, measured again. UNSURE and NO-MAJORITY counts are in
the receipt.

---

*Two cycles ago the family's best number was an in-sample mirage. Now it has a real one —
0.7458, held out, blind — and two frozen gates that still said no. The instrument is winning
the argument with its builders slowly, which is the only speed that counts.*
