# RESULT — the extraction ceiling is REFUTED, and the decomposition that named it is incomplete

Fathom Lab · 2026-09-01 · Scored by `score_extraction_panel.py`, which was **committed at
`b8b52a1` while the panel was still running and before any answer existed**. Gates transcribed
from `PREREG_extraction_ceiling_2026_09_01.md` (v1, frozen `38ab585`) and
`PREREG_extraction_ceiling_v2_2026_09_01.md` (G-E1 only, frozen `308490d`). Receipts:
`extraction_panel_result.json`, `extraction_panel_raw.json`, `extraction_panel_packet.json`
(sha256 `febd781e…`), `extraction_decoys.json` (sha256 `6a491480…`).

## VERDICT: `REFUTED`

Every gate passed. The hypothesis lost.

| gate | bar | observed | |
|---|---|---|---|
| G-E1a reliability, overall | >= 27/30 | **30/30** | pass |
| G-E1b each side separately | >= 9/15 each | **15/15 CLAIM, 15/15 NOT-A-CLAIM** | pass |
| G-E2 reconciliation | re-derive 0.16 | **0.1600** | pass |
| panel split | <= 0.10 | **0.0** | pass |

**E = 0.55  ·  A = 0.20  ·  P = 0.16**

Of 100 held-out path accusations, the panel found **55 were made about a sentence that really was
claiming this change touched that path**. Of those 55, only **11 were upheld** by the key
committed before any of this existed.

---

## What the frozen bar said, and what it says now

v1's G-E3, fixed before the packet was opened:

> **E >= 0.40 — REFUTED.** The claims were real and the gate misjudged them. Extraction does not
> exculpate the adjudicator, and V14's 0.16 remains unexplained. Published as a failed hypothesis
> of ours.

E = 0.55. **That cell, and the obligation in it, is what happened.**

The morning's hypothesis was that three repair cycles had been spent on the wrong layer — that
the adjudicator was sound and extraction was the binding constraint. **It is not.** A = 0.20
means that even when the author genuinely was claiming to have touched the path, the gate's
accusation was wrong four times in five. The adjudicator is bad on its own merits.

**V14's 0.16 remains unexplained.** It was unexplained this morning, an extraction account was
offered, the account was tested, and it failed. The account is now closed rather than open.

---

## The finding nobody preregistered: the decomposition is incomplete

`0.55 x 0.20 = 0.11`. The reconciled precision is `0.16`. **The identity `P = E x A` does not
close**, and the gap is not rounding.

Sixteen accusations were upheld. Only **eleven** of them sit among the 55 CLAIM items. **Five
upheld accusations landed on sentences the panel says were making no claim at all.** So

```
P  =  E·A  +  (1 - E)·A'
0.16  =  0.55(0.200)  +  0.45(0.111)
      =  0.110        +  0.050
```

`P = E x A` silently assumed `upheld ⊆ CLAIM` — that an accusation cannot be upheld against a
sentence that was not asserting anything. That assumption is false in this corpus at a rate of
`A' = 5/45 = 0.111`, and it was carried unstated through every document written today, including
the two preregistrations, the addendum, and the corpus survey.

Either the two panels disagree about those five items, or an accusation can be "right" about a
diff while being aimed at prose that made no claim. **Both readings are damaging to the framework
and neither is resolved here.** The corrected identity above is stated, not tested; a successor
would have to preregister `A'` as a quantity in its own right.

---

## Honest limits — and the first one is severe

**Three seats of the same model on the same prompt are not three independent judges.** The panel
was unanimous on **130 of 130 items**, including every accusation. A split rate of exactly zero
is not evidence of reliability; it is evidence of correlation. The multi-seat structure bought
much less here than the gate-passing suggests, and G-E1's 30/30 should be read with that in mind.

**The NOT-A-CLAIM decoys are ours, and 15/15 suggests they were easy.** `PREREG v2` said this in
advance: *"if the frames are too easy, G-E1 is passed by a panel that would not survive a real
ambiguous case, and E inherits that looseness."* A perfect score on authored decoys is the
outcome that clause was written about. E = 0.55 is therefore an estimate from a panel whose
demonstrated competence is on easy items only.

**This does not satisfy the standing commitment.** The commitment is a *blind* panel; this panel
is blind to the verdicts but is not independent of the lab. It was convened, instructed and
scored by us, on our sample, with decoys we wrote. It is a measurement. It is not the
measurement the commitment demands.

**E = 0.55 is not a licence to re-enable anything.** The retired path-accusation class stays
retired. If anything this result argues harder for retirement: the accusing verdict is wrong on
45% of the sentences it fires at *and* wrong on 80% of the ones where a claim really was made.

**What is licensed:** that on this packet, judged by this panel, 45 of 100 accusations were aimed
at sentences making no path claim; that of the 55 that were, 11 were upheld; and that these
reconcile to the committed 0.16 through a two-term identity rather than the one-term identity
this lab wrote down this morning.

---

## What today actually established

Three preregistrations were frozen. Two runs voided —
`ADDENDUM_extraction_ceiling_gate_unsatisfiable_2026_09_01.md` because its gate could not be
built, `RESULT_open_set_read_VOID_2026_09_01.md` because its null gate ran and failed. This is
the third, it completed, every gate passed, and **it refuted the hypothesis its own author spent
the day building.**

The extraction term is real and it is large — 45% of accusations are aimed at non-claims, which
is a collateral figure worth having on its own. It is simply **not the explanation for 0.16**.
Both layers of this instrument are broken, independently, and a repair that fixes only one of
them would have been justified by an argument that this measurement has now removed.
