# PRE-REGISTRATION — does the portable conscience survive OUT-OF-DISTRIBUTION? (frozen pre-run)

**2026-06-11 · Fathom Lab / styxx. Frozen before any OOD score is seen. Runner:
`run_portable_conscience_ood.py` (SEED=0, deterministic statement set + gates encoded in code).
Receipt on completion: `portable_conscience_ood_result.json`.**

## The question

v2 (`FINDING_portable_conscience_v2_2026_06_10.md`, OATH-HELD) proved CONSCIENCE-PORTABLE
**in-distribution**: gemma-2-2b's layer-12 honesty direction (difference-of-means), carried through a
label-free ridge map, read true-vs-false in Llama-3.2-3B (AUROC 0.893) and Qwen2.5-3B (0.918). But the
v2 anchors and test were a random shuffle of the SAME six fact-families — the test shared a
distribution with the fit. v2's own honest bound named the next rung explicitly: **OOD transfer is
untested.** This decides whether the portable conscience is a real cross-model lie detector or an
in-distribution existence proof.

## Design — leave-families-out

- **Source:** gemma-2-2b-it, layer 12, in-distribution difference-of-means honesty direction.
- **Train families (fit the direction AND the map):** capitals, chemical-elements, arithmetic,
  biology-classification.
- **OOD test families (held out — never touch the direction, the map, or layer/alpha selection):**
  historical-dates, comparatives, geography-location, definitions-properties. Disjoint DOMAIN and
  TEMPLATE from the train families (temporal / relational / locational / definitional registers).
- **Map:** label-free ridge alignment (target residual -> gemma residual), fit on TRAIN-FIT only;
  target layer + alpha selected by an internal TRAIN-FIT split R^2 (no test or OOD contact).
- **Primary targets:** Llama-3.2-3B-Instruct, Qwen2.5-3B-Instruct (the v2 pair, for direct
  OOD-vs-in-distribution comparison).
- **Secondary (descriptive only, never gates):** Llama-3.2-1B-Instruct, Qwen2.5-1.5B-Instruct
  (size contrast).
- **Matched in-distribution baseline (same run):** a held-out 20% slice of the TRAIN families, scored
  through the same fit. retention = ood_auroc / indist_auroc.

## Controls

- **Powered random-direction floor:** 200 random directions of matched norm pushed through the same
  map, scored on the OOD mapped activations; report the 95th percentile (`floor_p95`) and median.
- **Random-map control:** a random matrix of the map's shape, std-matched.
- **gemma OOD self-ceiling:** the source direction read on gemma's OWN OOD activations — establishes
  whether the honesty direction even generalizes OOD within the source model.

## Frozen gates

- **P1 — OOD-PORTABLE** iff, on the held-out OOD families, **transferred AUROC >= 0.65 AND > floor_p95
  for BOTH primary 3B targets.**
- **VOID-FIT** iff gemma OOD self-ceiling < 0.70 (the source direction does not even generalize OOD
  in-model -> the test is invalid, not a transfer negative) OR any primary `floor_p95` >= 0.78
  (degenerate floor, the v0 failure mode).
- **Verdict ladder:** VOID-FIT > OOD-PORTABLE (both pass) > OOD-PARTIAL (one passes) > OOD-COLLAPSE
  (neither passes).

## What each outcome means (pre-committed, no moving goalposts)

- **OOD-PORTABLE** — a single honesty direction fit in one model reads truth in two unseen models on
  fact-families it was never fit on. That is a real cross-model lie detector; the strongest form of the
  North Star to date. Still linear, still one task — those bounds remain.
- **OOD-PARTIAL / OOD-COLLAPSE** — the portable conscience is an in-distribution property; honesty
  transfer degrades under distribution shift. This is the honest frontier, and it is a genuine finding:
  it bounds the v2 claim precisely and tells us a per-distribution fit (or a richer/non-linear map) is
  required. A control that turns 0.9 into an honest negative is the styxx way.
- **VOID-FIT** — measurement fault (bad source or degenerate floor), not an answer; fix and re-run
  (the v0 -> v1 -> v2 pattern).

## Honest bounds carried in regardless of outcome

Linear DiM source, linear ridge map, one task (truthfulness), one source, four targets, local open
models only (closed frontier models blocked on credits). This tests OOD across FACT-FAMILIES; it does
not test adversarial inputs, jailbreaks, or non-factual honesty. A positive establishes EXISTENCE of
OOD-robust transfer across these minds and families — not a universal lie detector for all inputs.
