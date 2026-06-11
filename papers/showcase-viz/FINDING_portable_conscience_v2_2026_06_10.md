# FINDING — the conscience IS portable as a direction (CONSCIENCE-PORTABLE)

**2026-06-10 · Fathom Lab / styxx. Pre-registered: `PREREG_portable_conscience_v2_2026_06_10.md`
(frozen pre-run). Receipt: `portable_conscience_v2_result.json`. This overturns the v0 negative —
honestly, because v0 was confounded and the v0→v1→v2 discipline self-corrected to the true answer.**

## Result — portable, both targets, properly controlled

A single honesty direction fit in gemma-2-2b (difference-of-means on layer-12 activations, in
distribution), carried through a LABEL-FREE ridge alignment, reads true-vs-false in two models it was
never trained on:

| | transferred AUROC | floor p95 (random dir) | random-map control | ceiling (gemma self) |
| --- | --- | --- | --- | --- |
| Llama-3.2-3B | **0.893** | 0.717 | 0.643 | 0.876 |
| Qwen2.5-3B | **0.918** | 0.729 | 0.365 | 0.876 |

Both clear the 0.65 bar and **beat the powered floor** — so it is the SPECIFIC honesty direction, not
any direction (the floor's median sits at chance, 0.476 / 0.520). Both also **beat the random-map
control** — so the learned alignment is doing real work. And the transferred AUROC matches or exceeds
gemma's own self-readout (0.876): the honesty direction reads truth in the larger models at least as
well as in its home model. Verdict: **CONSCIENCE-PORTABLE**.

## Why this overturns v0 — and why that is the discipline working, not a flip-flop

- **v0 (MODEL-SPECIFIC)** used sixteen PAIRED items; the true-false difference was dominated by one
  direction, so the floor was degenerate (random directions scored all-or-nothing, the 95th
  percentile pinned at the ceiling), making ANY direction's transfer undetectable. v0's verdict was
  correct GIVEN its broken floor — but the floor was the artifact, not the conscience.
- **v1 (VOID-PIPELINE)** fixed the floor (large balanced set, AUROC, a large random-direction sample
  -> median near chance) but voided because the SHIPPED truthfulness probe was out-of-distribution on
  the test family (self-ceiling below the gate).
- **v2** fixed BOTH — a powered floor AND an in-distribution fit source (self-ceiling 0.876) — and the
  direction clearly transfers. Each version repaired a flaw the previous one exposed, converging on
  the true answer. The negative was never the conscience; it was two correctable measurement faults.

## What this means for the North Star

The portable conscience is real **as a direction**, not only as shared structure: honesty fit in one
mind transfers into others through a label-free map, beating the controls that v0 lacked. This is the
property that lets styxx be one instrument pointed at many minds, not a per-model zoo — the strongest
form of the North Star, now supported with the controls that matter.

## Honest bounds (the result is in-distribution and linear)

The test statements share a family with the anchors that fit both the source direction and the map;
the calibrated-correlate caveat stands and OOD generalization is untested (v1 showed the shipped
probe degrades OOD). Linear DiM source, linear ridge map, one source, two targets, one task. This
establishes EXISTENCE of a transferable honesty direction across these minds under matched
distribution — a strong positive — not a universal lie detector for all inputs. The next rungs:
OOD transfer, more source/target pairs, and whether one shared direction serves a whole family.
