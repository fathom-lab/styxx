# PREREG part-2d — the pricing result on a SECOND task family (numeric), cross-model panel

**Frozen:** 2026-07-22, before any scored `--judge` run exists. Autopilot cycle 59.
**Status at freeze:** the model-generality residual named in cycle 58 has ONE open sub-item —
"a second task family for the pricing gate (currently attr only)". Part-2c established, on a
heterogeneous 4-model panel (0.5B/1.5B/3B/7B-4bit Qwen2.5-Instruct) over the **attr** family,
that (i) the kill transfers — gold-anchor coverage collapses (blatant 0/12) — and (ii) the
same-generator LADDER anchors **price** the bad judges rather than dropping them (ladder cov
12/12, ladder median err 0.027 vs blatant 0.158, margin 0.131 > 0.08). This prereg asks whether
that pricing behaviour is **attr-specific or a property of the instrument** by re-running the
identical apparatus on the **numeric** family, with the bars inherited UNCHANGED.

## What runs

`stage_b_crossmodel_fam2.py --family numeric` — a parameterized copy of the frozen part-2c
harness (`stage_b_crossmodel.py`), byte-identical judge/prompt/decoding code, only the corpus
`family` and the cache path changed. Two phases:
- `--judge`: each of the 4 models judges the numeric blatant + ladder + redacted corpora, cached
  per (model, seed) to `p2c_fam2_numeric_cache.jsonl` (crash-safe; skips cached cells).
- `--score`: assemble the 4-model panels, run `styxx.anchors.audit_panel` on the blatant arm,
  the ladder arm, and the deaf (redacted) arm.

Panel: MODELS, PERSONA, judge() decoding (fire = NO = contradiction), N_ORG=200, K_ANCHOR=80,
PI=0.35 — all inherited verbatim from part-2c. Seeds: **12013–12024** (12 replicates), the same
FRESH disjoint base the part-2c pricing confirmation used, so this is confirmatory-design from the
first draw (no motivating data on this family to re-score). Family: **numeric only** (one family,
one cycle).

## Frozen gates (inherited from PREREG_part2c_pricing_confirm; bars do NOT move)

- **PD1 — kill transfers (numeric):** among blatant-arm ESTIMATED replicates, gold-anchor
  coverage `bl_cov <= 3`. (The gold anchors must fail to certify the broken panel here too.)
- **PD2 — pricing recovers (numeric):** `len(ladder ESTIMATED) >= 8` AND `ld_cov >= 10` AND
  `margin = blatant_median_err - ladder_median_err >= 0.08`. (The ladder must ESTIMATE, cover,
  and beat gold error by the same 0.08 margin that held on attr.)
- **PD3 — deaf VOID:** `>= 11 / 12` redacted-panel audits return a VOID verdict.

**Verdict rule:** `SURVIVED__pricing_transfers_to_numeric` iff PD1 ∧ PD2 ∧ PD3. Any missed gate is
`CLOSED_NEGATIVE` reported verbatim (e.g. if the ladder REFUSES on numeric — VOIDs rather than
prices — PD2 fails and the honest finding is that pricing is attr-favoured; a refusal is the
ladder doing the OTHER right thing, not a pass). No gate is re-scored to rescue a miss; no bar is
weakened; the part-2c attr result is untouched.

## Pre-declared threats / notes (do not gate)

- The numeric kill was already shown to transfer on the SAME-MODEL 3B panel (cycle 51, 0/15).
  The open question is the cross-model panel's LADDER behaviour, which is not implied by that.
- With 12 replicates, PD2's `ld_cov >= 10` is a strict bar; a family where the ladder VOIDs (as
  chain did on the same-model panel) fails PD2 by design — that outcome is a real second-family
  result, logged as CLOSED_NEGATIVE, not a defect.
- Labels re-derived by `corpus_label_oracle.py` (numeric oracle) over the exact scored seeds as a
  due-diligence bonus (zero mismatches expected); reported, not gated.
