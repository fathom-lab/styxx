# PREREG — cycle 26: re-certify the cycle-18 corpus under the shipped v0.4 verifier + repair the surfaced gaps

**Fathom Lab · papers/autopilot · 2026-07-04. FROZEN ON COMMIT, before any receipt/certificate is
written. Bars never move; a missed bar ⇒ CLOSED_NEGATIVE and every regenerated artifact is reverted.**

## What this does (operator: "let's do it" — the payoff of the cycle-25 ship)

The v0.4 recall extension (cycle 25, `39188b8`) shipped; the 13 cycle-18 certificates were left
pinned to their v0.3 verifier SHA. This cycle regenerates all 13 under the SHIPPED v0.4 verifier so
the whole corpus is tamper-evident under the current instrument, and **repairs** the two docs the
hardened verifier correctly turns UNGROUNDED — using the certifier's own prescribed cure (persist
the value as a summary receipt), NOT by weakening the oath or deleting the claim.

**No code changes.** `styxx/certify.py`, `validate_oath_v0.py`, and `mutant_battery.py` are
untouched (the shipped verifier SHA is unchanged), so the frozen OATH v0 D1/D2 gate holds by
construction; it is re-run only as a confirmation, not re-validated.

## The two gaps and their repair (exact, frozen)

Re-certifying the 13 docs in memory under v0.4 gives 11 OATH-HELD and 2 OATH-FAILED:
1. `RESULT_ai_brain_vision_2026_06_03.md` L34 — `RSA ≤ ~0.56 → R² ≤ ~0.16–0.31`: `0.56` is the
   noise-ceiling **upper** bound and `0.16`/`0.31` are **derived** R² = ceiling². The vision receipt
   persisted only `noise_ceiling_lo`. Repair: an addendum receipt `ai_brain_vision_ceiling_addendum.json`
   persisting `noise_ceiling_hi = 0.5567930754457157` and `r2_explainable_ceiling_{lo,hi} =
   noise_ceiling_{lo,hi}²`, each re-derived exactly (0.5568→0.56; 0.3944²→0.16; 0.5568²→0.31), with the
   derivation stated in the receipt.
2. `FINDING_council_2026_05_25.md` L63 — `agreement quantized to {0.25, 0.50, 0.75, 1.0}`: `0.50` = 2/4
   is a **definitional** quantization level of the 4-model council, present only in a bulk array.
   Repair: an addendum receipt `council_agreement_scale_addendum.json` persisting the scale
   `agreement_quantization_levels = [0.25, 0.50, 0.75, 1.0]` as a summary field.

Each repaired doc's certificate is regenerated citing its ORIGINAL receipt(s) PLUS its addendum.
The other 11 docs are regenerated against their existing receipt sets only.

## Frozen bars

| bar | gate |
|-----|------|
| **R1** | after repair, **all 13** cycle-18 docs certify **OATH-HELD** under the shipped v0.4 verifier |
| **R2** | each addendum contains ONLY values that re-derive exactly from an already-cited receipt (vision) or are a stated definitional constant of the method (council) — no new empirical claim; the receipt states its derivation. Verified by re-computation in the cycle log. |
| **R3** | for the 11 unrepaired docs, the regenerated VERIFIED/ABSTAIN/UNGROUNDED counts equal the in-memory v0.4 recert exactly (no silent drift); for the 2 repaired docs, VERIFIED does not DECREASE vs their shipped v0.3 cert (repair only adds grounding) |
| **R4** | committed mutant battery (seed 1) re-run over the regenerated certs: **false-verify ≤ 26** and **caught ≥ 116** of 269 (the addenda must not open a coincidence surface that regresses tamper-evidence) |
| **R5** | `validate_oath_v0.py` still OATH-V0-VALID (D1 ≥ 16, D2 = 0); `pytest tests -q` green |

Any missed bar ⇒ CLOSED_NEGATIVE: revert the regenerated certs + delete the addenda; the docs stay
as they were. In particular, if a gap cannot be closed by an *exactly re-derivable* addendum (R2),
it is NOT repaired — it is reported as a genuine provenance gap owed to the doc's authors, and the
doc's claim is scoped instead (a separate, non-autopilot decision). No manufacturing a match.

## Artifacts
- `papers/ai-human-alignment/ai_brain_vision_ceiling_addendum.json`,
  `papers/council-reference-free-truth/council_agreement_scale_addendum.json` (the repairs).
- 13 regenerated `*.certificate.json` under the v0.4 verifier SHA.
- `papers/autopilot/cycle26_recert_result.json` (per-doc verdicts + the battery re-run) + a RESULT
  note certified OATH-HELD before commit.

---
*Frozen on commit. The bar structure outranks the upgrade. The cure for an UNGROUNDED is a receipt, never a weaker oath.*
