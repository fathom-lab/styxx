# RESULT — OATH v0.4 trigger-vocabulary recall extension: CLOSED_NEGATIVE (autopilot cycle 23)

**Fathom Lab · papers/closed-model-frontier · 2026-07-04. Verdict against the frozen bars of
`PREREG_oath_v04_trigger_recall_2026_07_04.md` (committed BEFORE any code change). Bar G3 missed ⇒
CLOSED_NEGATIVE; `styxx/certify.py` reverted byte-identical to shipped v0.3; nothing ships.**

## What was tried

One edit to `styxx/certify.py`: widen the `_TRIGGERS` alternation with the correlation/similarity
register the AUROC-centric vocabulary misses — `rsa`, `rdm`, `spearman`, `pearson`, `correlation`,
`rho`, `consistency`, `reliability`, `ceiling`, `agreement`, `convergence`, `drift`, `entropy`,
`similarity`, `variance`. Goal: convert the cycle-19 battery's **182**-mutant abstain-degrade
bucket (the largest failure class: a corrupted number in this register silently falls to ABSTAIN
and the document verdict stays HELD) into caught UNGROUNDED.

## Verdict: CLOSED_NEGATIVE — bar G3 missed

| bar | gate | result |
|-----|------|--------|
| G1 | `validate_oath_v0.py` D1 caught ≥ 16 of 20 | **16 → PASS** |
| G2 | D2 = 0 false UNGROUNDED on the 3-doc corpus AND battery false-verify ≤ 26 | **0 and 26 → PASS** |
| G3 | every clean UNGROUNDED hand-verifies as a REAL doc↔receipt gap (no certifier artifact) | **6 artifacts of 35 → FAIL** |
| G4 | battery caught ≥ 116 of 269 (≥ 2× the v0.3 baseline of 58) | **128 → PASS** |
| G5 | `pytest tests -q` green | 1675 passed → PASS |

Four of five bars passed — the recall extension *works* on its own axis: battery catch rose from
the v0.3 baseline **58** caught to **128** caught of **269** mutants (catch rate **0.216 → 0.476**, more than doubling),
recovering **70** of the 182 abstain-degrades, and it did so **without** a false-verify regression
(**26**, unchanged — the feared dense-table abstain→false-verify conversion did not net occur).
But the honesty gate **G3 fired**: re-certifying the 13 clean cycle-18 docs under the extension
produced **35** clean UNGROUNDED (baseline **0**), and **6** of them are certifier **artifacts** —
a register word obligating a number that is not a measurement. Under the frozen contract one
artifact is a kill; there are six.

## The measured boundary (the yield)

Of the 35 clean UNGROUNDED, **28** are numbers genuinely absent from any grounding summary receipt
and **1** lives only in a bulk per-item array — **29 REAL** gaps where the extension correctly
turns the oath against an older doc that cites a correlation grid-cell never persisted as a
claimable summary field (the tool doing its job, exactly as cycle 19 celebrated). But **6** are
false alarms where a register word landed on a line whose number is a spec constant or an ordinal,
not a measured quantity — three of them unambiguous:

- `detection_locus_gpt` L64: *"**entropy** is a lower-bound proxy; OpenAI caps top_logprobs at
  20"* — the API cap `20` is obligated by `entropy`.
- `geometry_integrity` L46: a reproduction line *"run_geometry_integrity.py (**drift**, stage 1) ·
  run_geometry_mismatch.py (2D signature, stage 2)"* — the stage ordinals `1` and `2` are obligated
  by `drift`.

The remaining three are coincidental integer collisions the count claim→field binding rejects
(a ceiling percentage matching an unrelated `n=` field). In every case the shipped v0.3 verifier
correctly classed these as ABSTAIN; the blunt vocabulary widening breaks them.

**The lesson, receipted:** recall and precision are coupled for this register. The same words that
name a measured correlation (`entropy`, `drift`, `variance`, `ceiling`) also appear as
non-measurement tokens (API caps, stage ordinals, "2D", dimensionality). A bare vocabulary
extension cannot separate them; it buys +70 catches at the cost of 6 false accusations, and the
oath cannot ship false accusations. Near-miss on a blunt instrument is still CLOSED_NEGATIVE — the
bar structure outranks the 0.476 catch rate.

## Reproducibility

The candidate is reverted, so `styxx/certify.py` on disk is shipped v0.3 (git diff vs HEAD empty).
`papers/autopilot/cycle23_recall_probe.py` re-applies the exact one-line change in memory
(monkeypatch) and regenerates both receipts from the committed tree:
`cycle23_recall_battery_result.json` (128 / 26 / 112) and `cycle23_g3_handcheck_result.json`
(35 clean UNGROUNDED; 28 absent, 1 bulk-only, 6 artifact). Baseline reference:
`cycle22_v03_baseline_battery_result.json` (58 / 26 / 182). `validate_oath_v0.py` and
`mutant_battery.py` were not modified.

## What cycle 24 should consider (names this burial)

The next recall attempt must be **range-guarded**, not blunt: fire the correlation/similarity
register only when the adjacent number lies in a correlation's range (roughly −1 to 1), reusing the
existing RANGE-SANITY `unit_kw` machinery. That spares `20` (an API cap), stage ordinals, and "2D",
while still binding an `RSA 0.264` or a `reliability 0.735`. A NEW prereg must name this negative
and re-gate on G3 = 0 artifacts. The 29 REAL gaps this cycle surfaced are also owed a separate
decision (persist the grid-cell correlations as summary receipts, or scope those docs' claims) —
the same repair-the-receipt loop the v0.3 NOTE prescribes.
