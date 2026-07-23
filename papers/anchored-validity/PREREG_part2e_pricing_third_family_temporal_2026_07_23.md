# PREREG part-2e — the pricing result on a THIRD task family (temporal), cross-model panel

**Frozen:** 2026-07-23, before any scored `--judge` run on the temporal family exists. Autopilot
cycle 60.
**Status at freeze:** parts 2c (attr) and 2d (numeric) established, on a heterogeneous 4-model
panel (0.5B/1.5B/3B/7B-4bit Qwen2.5-Instruct), that (i) the kill transfers — gold-anchor coverage
collapses (blatant 0/12) — and (ii) the same-generator LADDER anchors **price** the bad judges
rather than dropping them (attr: ladder cov 12/12, err 0.027 vs 0.158, margin 0.131; numeric:
ladder cov 12/12, err 0.016 vs 0.140, margin 0.124). This prereg asks whether that pricing
behaviour holds on the **temporal** family — the family the program has the SHARPEST prior reason
to expect it might BREAK.

**Why temporal is the adversarial choice (not a victory lap):** temporal is the SMOOTH-VIOLATION
family. On the same-model 3B panel (cycle 51) it produced the WORST gold-anchor errors in the
program (median 0.65) AND the misfit flag went **silent** (0/15, vs numeric's 15/15) — the panel
was confidently, structurally wrong in a way no post-hoc statistic caught. Pricing works by the
moment system MEASURING each judge's organic false-fire rate and SUBTRACTING it. If temporal
violations are "smoothly wrong" in a judge-CONSISTENT way, the same-generator ladder anchors could
inherit the very blindness that silenced the misfit flag — in which case the ladder should VOID
(refuse) rather than price, and PD2 fails. That is a genuine, pre-named kill path, not a formality.

## What runs

`stage_b_crossmodel_fam2.py --family temporal` — the frozen part-2d harness, run verbatim with the
`--family` CLI flag set to `temporal`. **No code change to the harness, the judge, the prompt, the
decoding, the panel, or the gates.** The corpus `family` and the cache path (`p2c_fam2_temporal_
cache.jsonl`) are the only things that differ from part-2d. Two phases:
- `--judge`: each of the 4 models judges the temporal blatant + ladder + redacted corpora, cached
  per (model, seed) (crash-safe; skips cached cells).
- `--score`: assemble the 4-model panels, run `styxx.anchors.audit_panel` on the blatant arm, the
  ladder arm, and the deaf (redacted) arm; emit the frozen gates.

Panel: MODELS, PERSONA, judge() decoding (fire = NO = contradiction), N_ORG=200, K_ANCHOR=80,
PI=0.35 — all inherited verbatim. Seeds: **12013–12024** (12 replicates), the same disjoint base
parts 2c-confirm and 2d used (the family difference guarantees disjoint corpus content; the judge
cache is family-keyed), so this is confirmatory-design from the first draw. Family: **temporal
only** (one family, one cycle).

## Frozen gates (inherited from PREREG_part2d / part2c_pricing_confirm; bars do NOT move)

The harness emits these under the labels `PD1/PD2/PD3`; here they are the temporal instances of the
identical conditions.

- **PD1 — kill transfers (temporal):** among blatant-arm ESTIMATED replicates, gold-anchor
  coverage `bl_cov <= 3`. (The gold anchors must fail to certify the broken panel here too.)
- **PD2 — pricing recovers (temporal):** `len(ladder ESTIMATED) >= 8` AND `ld_cov >= 10` AND
  `margin = blatant_median_err - ladder_median_err >= 0.08`. (The ladder must ESTIMATE, cover, and
  beat gold error by the same 0.08 margin that held on attr and numeric.)
- **PD3 — deaf VOID:** `>= 11 / 12` redacted-panel audits return a VOID verdict.

**Verdict rule:** `SURVIVED__pricing_transfers_to_temporal` iff PD1 ∧ PD2 ∧ PD3. Any missed gate is
`CLOSED_NEGATIVE` reported verbatim. In particular, if the ladder **REFUSES** on temporal (VOIDs
rather than prices — the pre-named smooth-violation kill path), PD2 fails and the honest finding is
that pricing does NOT survive the smooth-violation family; the refusal is the ladder doing the
OTHER right thing (declining to certify a panel it cannot audit), not a pass, and it is a real,
publishable second outcome that SCOPES the pricing claim to non-smooth families. No gate is
re-scored to rescue a miss; no bar is weakened; the parts 2c/2d results are untouched.

## Pre-declared threats / notes (do not gate)

- The temporal kill was already shown on the SAME-MODEL 3B panel (cycle 51, 0/15). The open
  question is the cross-model panel's LADDER behaviour on this family, which is not implied by that.
- With 12 replicates, PD2's `ld_cov >= 10` is a strict bar. A temporal ladder that VOIDs (as chain
  did on the same-model panel in cycle 53) fails PD2 by design — that outcome is a real
  third-family result, logged CLOSED_NEGATIVE, not a defect.
- Labels re-derived by `corpus_label_oracle.py` (temporal oracle, which already covers both the
  easy and hard temporal forms) over the exact scored seeds as a due-diligence bonus (zero
  mismatches expected); reported, not gated. The temporal blatant + ladder entries at seeds
  12013–12024 are added to the oracle's SCORED ledger for this run.
- Smoke (`--smoke`, seed 19995) writes only `p2c_fam2_temporal_SMOKE_INVALID.jsonl` and is never
  read as a result.
