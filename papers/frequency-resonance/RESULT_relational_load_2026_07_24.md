# Result — the relational dose-response was NOT demonstrated (PARTIAL)

**Date:** 2026-07-24
**Prereg:** `PREREG_relational_load_2026_07_24` (frozen before this run)
**Receipt:** `relational_load_result.json`
**Verdict:** `PARTIAL__reported_verbatim` — the preregistered CONFIRM bar was not met, and NULL was not
met either. The prediction is unsupported at this resolution; it is not refuted.

## What was predicted and what happened

The dissociation result (`RESULT_recall_horizon_2026_07_24`) concluded that the oscillatory channel is a
RELATING mechanism, not a memory one. Its sharpest consequence: a decay channel's failure should grow with
the number of RELATIONS a task requires while staying flat in the number of FACTS it must merely hold.
This run tested that dose-response with storage and distance matched across two axes — a conjunction of R
comparisons versus holding S facts and reporting one named by a selector — differing only in the operation
required.

**The predicted scaling did not appear.** Across relational loads one through four, the decay model's
solve rate was 1.0, 0.6, 0.8, 0.8: a drop of 0.2 from the lightest to the heaviest load, well below the
preregistered 0.4 bar, and non-monotone in the middle. The oscillatory model solved every cell of both
axes at 1.0. The storage control behaved as predicted — solve rates 1.0, 1.0, 0.8, 1.0, a drop of 0.0 —
so nothing here suggests the deficit is storage capacity.

## Honest reading

Two things are true at once and neither should be dropped:
- **The contrast points the predicted way.** The relational axis lost 0.2 while the matched storage axis
  lost 0.0. That is consistent with a relational-specific cost.
- **It is not a dose-response.** The relational curve is non-monotone and the total drop is half the
  preregistered bar. With five seeds and a bimodal per-seed outcome (each run either solves at 1.0 or
  sticks near 0.5), a 0.2 difference is one or two seeds' worth of scatter. That is a hint, not a finding,
  and the frozen gate correctly refused to call it more.

The most likely reason the effect is small here is design, not absence: to give load one any headroom, the
premises had to sit INSIDE decay's competent range (gaps four to eleven), where decay compares reliably.
Adding relations at short range apparently costs a decay channel much less than adding distance does — the
distance axis produced a clean, large horizon in the parent result, while the relational axis at short
range does not. Whether relational load bites harder near the horizon is untested and is the obvious next
design.

## What this does and does not change

- The dissociation itself is **unaffected**: it rests on two directly measured facts — decay recalls one
  fact at every distance, and decay fails to compare a distant one. This run did not test those.
- The *scaling* claim implied by calling oscillation a "relating mechanism" is **not established**. Until
  a dose-response is shown, the honest phrasing is that oscillation is required for distant comparison,
  without asserting that its advantage grows with the number of comparisons.

## Scope (unchanged)

Controlled state-space-model work. Not a real-LLM claim; no language model is run, and transformers have
no phase to clamp.

## Process note

The first execution of this run died on a CUDA out-of-memory fault caused by a concurrent GPU process on
the same eight-gigabyte device, partway through the third relational load. It produced no receipt and no
verdict, and none was claimed; the identical frozen design was re-run on a free device and is what is
reported here.

## Bottom line

A relational dose-response was predicted, and it did not show up at the preregistered effect size. The
storage control was spared, which keeps the relational reading alive as a hypothesis, but the honest
verdict is PARTIAL: the sharpest consequence of our own dissociation remains unproven.
