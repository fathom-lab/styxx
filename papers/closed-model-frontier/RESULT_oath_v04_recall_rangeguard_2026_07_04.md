# RESULT — OATH v0.4 RANGE-GUARDED trigger-recall: CLOSED_NEGATIVE — autopilot, twenty-fourth cycle

**Fathom Lab · papers/closed-model-frontier · 2026-07-04. Verdict against the frozen bars of
`PREREG_oath_v04_recall_rangeguard_2026_07_04.md` (committed BEFORE any code change, `f539339`).
Bar G3 missed ⇒ CLOSED_NEGATIVE; `styxx/certify.py` reverted byte-identical to shipped v0.3;
nothing ships.**

## What was tried

The cycle-23 fix: the correlation/similarity register obligates a number **only when it is a
plausible correlation** (value ∈ [−1, 1]), reusing the observation that all six cycle-23 artifacts
were out of that range. Two edits to `certify.py`: a `_TRIGGERS_CORR` pattern + a range-guarded
`bound` contribution.

## Verdict: CLOSED_NEGATIVE — bar G3 missed by ONE

| bar | gate | result |
|-----|------|--------|
| G1 | `validate_oath_v0.py` D1 caught ≥ 16 | 16 → PASS |
| G2 | D2 = 0 on corpus AND battery false-verify ≤ 26 | 0 and 26 → PASS |
| G3 | clean UNGROUNDED artifacts = 0 across the 13 docs | **1 → FAIL** |
| G4 | battery caught ≥ 116 | **119 → PASS** |
| G5 | `pytest tests -q` green | green → PASS |

The range guard did **most** of its job: clean UNGROUNDED collapsed from cycle-23's 35 to **4**, of
which **2** are REAL (derived RSA bounds `~0.56`/`~0.16` the receipts do not hold), **1** REAL
(a `0.50` agreement value living only in a bulk array), and **1** ARTIFACT. Battery caught **119**
of 269 mutants (down only 9 from the blunt 128 caught — the recall survives the guard) with
false-verify still **26**. Four bars pass. But G3 is `= 0`, and there is **one** artifact — a kill.

## The residual (the exact yield)

The surviving artifact is `RESULT_geometry_integrity` L46: a reproduction line
*"`run_geometry_integrity.py` (**drift, stage 1**) · `run_geometry_mismatch.py` (2D signature,
stage 2)"*. The ordinal **`1`** is obligated by "drift", and the value guard admits it **because 1.0
is a legal correlation** (the boundary of [−1, 1]). Its sibling `stage 2` was correctly spared
(2 ∉ range), as were the other four cycle-23 artifacts (`20`, `60`, `60`, `36`, all out of range).
So the range guard fixed **5 of 6**; the one it cannot fix is the integer ordinal that collides with
the correlation boundary value.

**The lesson, receipted:** a value-range guard is necessary but not sufficient. Correlations are
written with a decimal point (`0.264`, `0.98`, `0.735`); the false-positive tokens are bare integers
(stage `1`/`2`, cap `20`, `36` items, `60`%). The clean separator is therefore **decimals**, not
range alone: the correlation register should obligate a number only when it has a fractional part
(`decimals > 0`) — which no ordinal, count, or whole-percent ever does. That also subsumes the range
guard for this corpus (every in-range false positive here is a bare integer).

## Reproducibility

The candidate is reverted (`certify.py` diff vs HEAD empty). The change is two lines inside
`certify_doc`, given verbatim in the prereg; re-apply it and run `papers/autopilot/cycle24_rangeguard_probe.py`
to regenerate `cycle24_rangeguard_battery_result.json` (119 / 26 / 121) and
`cycle24_rangeguard_g3_result.json` (4 clean UNGROUNDED: 2 absent, 1 bulk, 1 artifact).
`validate_oath_v0.py` and `mutant_battery.py` were not modified.

## What cycle 25 should consider (names this burial)

Add a **`decimals > 0`** condition to the range-guarded correlation register (obligate only
fractional numbers in [−1, 1]). This removes the `stage 1` ordinal artifact by construction while
keeping every decimal correlation the recall recovers. NEW prereg naming this negative; re-gate on
G3 = 0 artifacts AND G4 catch ≥ 116. If it passes, it is the first certifier upgrade to ship since
v0.3.
