# RESULT — OATH v0.4 decimal+range-guarded trigger-recall: SHIPPED (autopilot cycle 25)

**Fathom Lab · papers/closed-model-frontier · 2026-07-04. Verdict against the frozen bars of
`PREREG_oath_v04_recall_decimalguard_2026_07_04.md` (committed BEFORE any code change, `b6f5808`).
ALL FIVE BARS PASS ⇒ the extension SHIPS — the first `styxx/certify.py` upgrade since v0.3.**

## What shipped

The correlation/similarity register (`rsa`, `rdm`, `spearman`, `correlation`, `rho`, `consistency`,
`reliability`, `ceiling`, `agreement`, `convergence`, `drift`, `entropy`, `similarity`, `variance`)
now obligates a doc number to ground — but **only when it is a fractional correlation**:
`decimals > 0` AND value ∈ [−1, 1]. Two lines: a `_TRIGGERS_CORR` pattern and a guarded `bound`
contribution in `certify_doc`. This closes the 182/269 abstain-degrade bucket (cycle-19 diligence):
a corrupted correlation value can now fire UNGROUNDED instead of silently falling to ABSTAIN.

## Verdict: SHIPPED — five of five bars

| bar | gate | result |
|-----|------|--------|
| G1 | `validate_oath_v0.py` D1 caught ≥ 16 | 16 → PASS |
| G2 | D2 = 0 on corpus AND battery false-verify ≤ 26 | 0 and 26 → PASS |
| G3 | clean UNGROUNDED artifacts = 0 (frozen definition) | **0 → PASS** |
| G4 | battery caught ≥ 116 | **119 → PASS** |
| G5 | `pytest tests -q` green | 1675 passed → PASS |

Battery tamper-catch rose from the v0.3 baseline **58** caught to **119** caught of **269** mutants
(catch rate **0.216 → 0.442**), with false-verify held at **26** (no
regression) and clean UNGROUNDED artifacts driven to zero.

## The arc that earned it (four negatives before the ship)

The precise guard is the residue of four honest negatives, each naming the last:
- **the float attempt** — claim→field binding (the false-verify bucket): CLOSED_NEGATIVE, residual = same-table siblings.
- **the blunt add** — the whole correlation register (this recall bucket): recovered many catches but left six artifacts — register words obligating non-measurement numbers (an API cap under "entropy", stage ordinals under "drift"). Reverted.
- **the range guard** — obligate only in-range values: fixed five of the six artifacts but the ordinal one survived, admitted at the correlation boundary. Reverted.
- **the decimal guard (this ship)** — also require a fractional part: correlations carry decimals; ordinals, counts, API caps, whole-percents never do. The last artifact vanishes by construction; the recall survives.

## What the ship correctly surfaces (not artifacts — real provenance debt)

Three clean UNGROUNDED remain across the 13 cycle-18 docs, all **REAL** measurement-domain gaps
(hand-verified against the frozen G3 definition): derived RSA/R² noise-ceiling bounds in
`ai_brain_vision` that no receipt holds, and a `council` agreement value living only in a bulk
array. These are the certifier correctly refusing to swear on correlation numbers never persisted
as summary receipts — the tool turning on the older docs, exactly the v0.1 bulk-exclusion design.
The 13 cycle-18 certificates stay pinned to their recorded verifier SHA; re-certifying them under
the shipped verifier (which will show these gaps) is a separate, later repair-the-receipt decision.

## Reproducibility

`styxx/certify.py` now carries the change; `validate_oath_v0.py` regenerated the 3 corpus
certificates under the new verifier SHA (D2 = 0 held). `papers/autopilot/cycle25_decimalguard_probe.py`
regenerates `cycle25_decimalguard_battery_result.json` (119 / 26 / 121) and
`cycle25_decimalguard_g3_result.json` (3 clean UNGROUNDED, 0 artifact) from the shipped tree.
Baselines: `cycle22_v03_baseline_battery_result.json` (58), `cycle23_recall_battery_result.json`
(128 blunt), `cycle24_rangeguard_battery_result.json` (119 range-only). `validate_oath_v0.py` and
`mutant_battery.py` were not modified — bars never moved.

## Owed next

Re-certify the 13 cycle-18 docs under the shipped verifier and repair the surfaced provenance gaps
(persist bulk/derived correlations as summary receipts, or scope the docs' claims). Then: claim→CELL
float binding (the cycle-22 sibling, the false-verify bucket); quantity-specific ranges for the
unbounded register words (entropy/variance > 1 currently won't bind — an accepted recall limit).
