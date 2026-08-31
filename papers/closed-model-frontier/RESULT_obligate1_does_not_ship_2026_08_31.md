# RESULT — OBLIGATE-1 does not ship: in-sample 0.80 collapsed to 0.4483 held-out

Fathom Lab · 2026-08-31 · Prereg: `PREREG_obligate1_2026_08_31.md`, gates frozen before any
seat ran. Receipts: `obligate1_result.json`, `obligate1_packets.json`,
`obligate1_seat_outputs.json`. Verbatim, as the prereg requires on this outcome: **"the
structural obligation clause does not survive held-out adjudication."**

## What was attempted

The largest known hole in the flagship instrument — 0.5227 of full-precision decimals sit on
lines without trigger vocabulary and are never checked (issue #39). The best structural
candidate from the 2026-08-28 census (*decimal with two or more fractional digits, outside any
code span*) scored **0.80 precision in-sample**, against 0.40 for the word-list family. That
RECON licensed nothing, said so, and this cycle put the candidate to a fresh blind panel with
the ship bar frozen at **0.70** — a stated allowance for regression to the mean.

## What happened

Nine fresh seats, three packets, sixteen token decoys each. **Every seat scored 16 of 16 on
the decoys.** Unanimity 0.9762, zero NO-MAJORITY, five majority-UNSURE excluded and counted.

| | adjudicated CLAIM | n | share |
|---|---|---|---|
| OBLIGATE-1 positive arm | 26 | 58 | **0.4483** |
| negative arm (predicate said no) | 13 | 57 | 0.2281 |

- **G-O1P — FAIL.** OBLIGATE-1 precision = 26/58 (0.4483) vs bar 0.7; obligate-everything
  null = 39/115 (0.3391); weighted recall = 0.3935 vs bar 0.2; no significance is claimed at
  these n. **The clause does not ship.** G-O1REG, the corpus A/B, never runs — it was the
  ship gate for a clause that failed upstream.
- **G-O1NULL — PASS.** 0.4483 beats obligate-everything's 0.3391. The structure is real
  signal.
- **G-O1R — PASS.** Population-weighted recall 0.3935: the rule would catch roughly two in
  five of the corpus's unchecked claims.

Real signal, wrong grade. Obligation manufactures accusations, so precision here is one minus
the false-accusation rate — and a rule that would false-accuse **32 of every 58** tokens it
newly obligates is not an obligation predicate, it is an accusation generator. The prereg's
cost inversion did its job: on the claim-detector problem 0.4211 precision was a PASS because
a missed flag costs nothing; here 0.4483 is a hard FAIL because every false positive indicts
an innocent number.

## The regression, measured — why in-sample licenses nothing

In-sample 0.80 → held-out 0.4483. The candidates were written before the census data was
consulted, but they were *scored* on the 225 adjudications that motivated the whole census —
and a third of their apparent precision evaporated on fresh tokens. This is the cleanest
demonstration this lane owns of the rule the RECON stamped on itself: **a licence has to be
earned held-out**. Had the census's number shipped, the corpus would now carry a wave of
false accusations wearing a 0.80 badge.

## The failure has a name: bars

All 32 would-be false accusations are listed in the receipt. Read together they are one
class with three costumes: **thresholds**. Prereg'd bars ("against the pre-registered bar of
0.60"), gate criteria in tables ("| G2_islands_present | gap-screen p ≤ 0.05 |"), floors and
ceilings ("threshold of 0.80", "the frozen +0.05 bar"), config values ("cosine@0.90"), and a
citation id (arXiv 2509.06902). A quantity written to two decimals outside code is, in this
corpus, about as likely to be the **bar a result is judged against** as the result itself —
and the panel, given one line of context, separated the two almost perfectly. The seats can
see the speech act; the predicate sees only the token's shape. The same root defect two other
instruments hit this week: mention-versus-use, wearing a threshold's clothes.

So the next candidate writes itself, and is NOT built in this cycle: precision-shape ∧
outside-code ∧ **not bar-adjacent** (comparison operators, "bar/threshold/floor/ceiling/
against" in the token's immediate window). It goes to its own freeze, its own fresh panel,
and the same 0.70 bar — after this one, nobody gets to call that step optional.

## This document is OATH-FAILED, and the accusations are the twelfth specimen

The certificate for this RESULT reads **OATH-FAILED** on three tokens: the denominators `58`
and `115` inside the counts statement the prereg REQUIRES verbatim ("OBLIGATE-1 precision =
26/58 … null = 39/115 …"), and the same `58` where the false-accusation rate restates it.
The verifier's range-sanity clause sees integers larger than one on a line carrying
precision vocabulary and empties their bindings — it cannot tell a rate from the
denominator of the fraction that produced it.

This is the second RESULT in two days published FAILED for text its own protocol mandates:
yesterday a verbatim quotation, today a mandated counts disclosure. The pattern now has a
name — **the verifier taxes the exact disclosures the preregs order** — and it is recorded
here as an instrument defect (denominator blindness on metric lines), not reworded away.
The numbers stand; the tax is paid in public.

## Limits

n=58/57 per arm; no significance claimed anywhere. Seats saw one line of context capped near
140 characters, not the document. One model family throughout; unanimity 0.9762 is a
correlated-error ceiling. The 0.5227 gap figure describes decimals; this sample adjudicated
the predicate's own stratification, so arm shares — not the gap — are the measured
quantities. Five UNSURE majorities were excluded, counted, and reported.

## What this buys

The trigger-recall gap stays open — now with a held-out map of what does not close it: word
lists (0.40, measured dead twice), and the best naive structural rule (0.4483 held-out,
0.3935 recall, real but unshippable). The open trigger-recall issue gains its second measured negative and its
first named structural frontier. The verifier keeps abstaining honestly rather than accusing
confidently, which is the entire point of gating on precision first.

---

*The census said 0.80. The blind panel said 0.4483. The gap between those two numbers is why
this lab freezes bars before it sees data — and the clause the gap killed would have accused
thirty-two innocent numbers wearing its in-sample badge.*
