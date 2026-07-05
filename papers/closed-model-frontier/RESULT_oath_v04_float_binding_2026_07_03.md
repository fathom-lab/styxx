# RESULT — OATH v0.4 float claim→field binding (last-two-segment design): `CLOSED_NEGATIVE`

**Fathom Lab · papers/closed-model-frontier · 2026-07-03 · autopilot cycle 22.**
**Prereg:** `PREREG_oath_v04_float_binding_2026_07_03.md`, frozen and committed before any code
change. **Bar B3 was missed; per the prereg the change is REVERTED and nothing ships.**
`styxx/certify.py` is byte-identical to its shipped v0.3 state (revert proven: the unmodified
validator re-run under the reverted verifier regenerates the three corpus certificates with zero
git diff).

## What was tested

Floats grounded by value-only matching let a corrupted number re-verify against a *neighboring*
receipt leaf (cycle-19 evidence). The v0.4 candidate: a float stays VERIFIED only if some
value-matching leaf's **last two path segments** share vocabulary with the claim line; a
value-match with no binding leaf demotes to loud ABSTAIN (never UNGROUNDED).

## Verdict table (frozen bars)

| bar | gate | result | PASS |
|-----|------|--------|------|
| B1 | validator D1 caught ≥ 16 of 20 | caught **17** of 20 | YES |
| B2 | validator D2 false UNGROUNDED = 0 | 0 | YES |
| B3 | battery FALSE-VERIFY ≤ 13 | **20** (of 247 mutants) | **NO — kill** |
| B4 | 13 docs stay UNGROUNDED = 0 | 0 across all 13 | YES |
| B5 | pytest suite green | green (post-revert) | YES |

One missed bar ⇒ CLOSED_NEGATIVE. Near-halving is not halving; the bar structure outranks the
upgrade.

## What the experiment established (all receipt-backed)

1. **Field-level binding is real but insufficient.** FALSE-VERIFY moved 26 → **20** (rate 0.097 →
   0.081) and catch was unhurt (battery caught 58 → caught 53 of a smaller 247-mutant pool, catch
   rate 0.216 → 0.215; validator D1 caught *improved* from 16 to caught 17). The eliminated class
   is the cross-table coincidence.
2. **The residual channel is same-table siblings, not cross-table strangers.** All 20 surviving
   false-verifies are floats whose corrupted value matches a *sibling* leaf inside the
   legitimately-bound table (a `| 2 | 0.988 | 0.959 |` row mutating into another row's value) or
   lands within rounding tolerance of a nearby leaf. Field-family vocabulary CANNOT separate
   row k=2 from row k=4 — both share the binding surface by construction. The fix this points at
   is **claim→CELL binding** (row-key aware: single-digit row labels and list indices are
   currently invisible to the binding vocabulary), which is a NEW prereg, not an amendment.
3. **The honesty cost of binding was small and safe.** On the 3-doc validation corpus, 8 floats
   demoted VERIFIED→ABSTAIN (24→19, 42→40, 34→33 per doc), zero new UNGROUNDED anywhere (B4:
   demotion refuses to swear, it never accuses — held empirically on all 13 docs).
4. **The cycle-19 battery is now a committed, re-runnable script** (`papers/autopilot/
   mutant_battery.py`) — under the reverted v0.3 verifier it reproduces the cycle-19 in-memory
   numbers EXACTLY: 269 mutants, caught 58, FALSE-VERIFY 26, abstain-degrade 182, dropped 3.
   The reproducibility gap named in cycle 19 is closed; the negative above is measured by a
   validated instrument.

## Receipts

- `papers/autopilot/cycle22_v04_battery_result.json` — the B3/B4 run under the v0.4 candidate
  (247 mutants; per-doc table; all 20 false-verify rows with contexts).
- `papers/autopilot/cycle22_v04_validation_result.json` — the B1/B2 validator report under the
  candidate (D1 17/20, D2 clean).
- `papers/autopilot/cycle22_v03_baseline_battery_result.json` — the script's baseline replication
  under shipped v0.3 (269/58/26/182/3, exact match to cycle 19).
- `papers/autopilot/mutant_battery.py` — the battery script (seed 1, SHA-verified receipt
  resolution).

## Status of the claim surface

Nothing is retracted and nothing is resurrected. `styxx.certify` remains v0.3 with its disclosed
float limitation — now with a *measured boundary*: field-level binding buys −6 false-verifies and
+1 D1 catch, and the remaining exposure is precisely the same-table sibling/rounding-tolerance
class. OATH v0.4 remains OWED; the next attempt must bind claim→cell, and its prereg must name
this negative.

---

*A kill-gate that fires on the designer's own upgrade is the apparatus working. Bars never move.*
