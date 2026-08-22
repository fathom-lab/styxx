# PREREG — OATH v0.7: unobligated numbers on trigger-less lines (B-item #39, first attempt)

Fathom Lab · 2026-08-22 · frozen BEFORE any change to `styxx/certify.py`. Bars below never move;
a missed bar ⇒ revert and publish the negative, nothing ships. If ALL gated bars pass, the
extension SHIPS.

Provenance: this is the trigger-recall lead that `RESULT_oath_v062_SHIPPED_2026_07_31.md` names as
the family's standing debt ("the verifier still cannot obligate numbers on lines without trigger
vocabulary; G2b measures that debt at 0.5227 of the full-precision pool") and that GitHub issue
[#39](https://github.com/fathom-lab/styxx/issues/39) published as an open problem on 2026-08-05.
First attempt in this family; it resurrects no buried claim. Baselines carry over unchanged from
v0.6.2 (`oath_v062_battery_result.json`, `oath_v062_corpus_after.json`), live at the shipped verifier.

## The defect (measured, pre-fix)

`certify_doc` obligates a number only when its binding context matches `_TRIGGERS`, or matches
`_TRIGGERS_CORR` while being a fractional value in [−1, 1], or is glued to `n=`. Everything else
lands in ABSTAIN. Numbers on trigger-less lines are therefore never obligated — but the defect this
creates is not the one the debt line describes, and the pre-fix census says so.

Census over the 136 documents whose cited receipts all resolve
(`census_unbound.py`, `sweep_decimals.py`, run pre-fix at the shipped verifier):

- The full-precision pool (≥7 fractional digits) is 349 tokens; **183 (0.5244) sit on unbound
  lines**, replicating the published 0.5227 on an independently-built pool.
- **177 of those 183 are already VERIFIED on the clean corpus.** Status VERIFIED is awarded on a
  value-match to a receipt leaf regardless of obligation, so an unbound number in an honest
  document is not unchecked. The debt is not a hole in coverage of clean documents.
- The hole opens under mutation. Change the number and the value-match disappears; obligation never
  fires because it never depended on the value; the mutant lands ABSTAIN and the document keeps its
  OATH-HELD verdict. **A silent pass.**

This has already cost the program one real repair. `oath_v062_repair_addendum.json` was built in the
v0.6.2 repair loop to cure the un-persisted bare-arm accuracy `0.2711864406779661`. The byte-identical
token appears in two documents. `DATASHEET_conscience_2026_07_24.md` was repaired, because its line
carries trigger vocabulary and the token was flagged. `FINDING_third_party_bench_2026_07_24.md` was
not, because its line carries none — so the repair loop never saw it, and that document still cites a
number its receipts do not hold. The trigger-recall gap did not merely hide a hypothetical tamper; it
steered a real repair loop away from a real provenance gap.

**The size of the class this fix does NOT reach is stated here, before the fix, so the result cannot
be read as more than it is.** A separate pre-fix census mutates every claim the shipped verifier
certifies VERIFIED, one significant digit each, and re-certifies
(`silentpass_census.py`). Its measured silent-pass share and the fraction of that surface a
precision rule can close are reported under G2c. The published debt line (0.5227 of the
full-precision pool) describes a pool that is a minority of the real tamper surface, and the
majority of that surface sits at ≤4 fractional digits and in bare integers, where no precision
predicate can reach.

## The change (exact, frozen)

A per-token predicate at the v0.4 clause site, setting `bound` and nothing else.

```python
V07_PRECISION_OBLIGATION = True   # primary
V07_PRECISION_DIGITS = 7

# ... inside certify_doc, immediately after the v0.4 fractional-correlation clause:
if not bound and V07_PRECISION_OBLIGATION and num["decimals"] >= V07_PRECISION_DIGITS:
    bound = True
    precision_only = True          # provenance for the severable class below
```

Rationale for the predicate: a number printed at seven or more fractional digits was copied out of a
computation. Bars, ordinals, counts, caps and dimensionality labels are hand-authored and, measured
across the 1,100 markdown documents in `papers/`, are written at one to four decimals; of 802 tokens
at ≥5 fractional digits, exactly five sit in bar position and every one of those five is a *computed*
floor (a random-direction baseline, a 95th percentile of a calibration set) — a quantity that should
carry a receipt. The predicate reads only the token's printed precision, so it survives the mutation
it exists to catch. An obligation that consults the match set evaporates under exactly that mutation
and cannot gate: the variant "receipt-path stem overlap AND value-match" scores a perfect zero
clean-corpus accusations and catches nothing, and is rejected here for that reason.

**Why 7 and not 5.** Every live counterexample the red-team pass produced sits at exactly five or six
fractional digits: a pre-registered kill-gate bar in this repo's own JSON idiom
(`"op": "<=", "value": 0.00648`, `PREREG_b35c_open_vocab_2026_08_03.md` L29), the half-ULP tolerance
definition `±0.00005`, π written out as `3.14159`, a Bonferroni α of `0.00714`, and the arXiv DOI
prefix `10.48550` — which `_VERSIONISH` does not reach and v0.5 class C's
`fullmatch(r"\d{4}\.\d{4,5}")` does not match. Threshold 7 spares all of them. Threshold 5 buys two
tokens of catch surface in the certified corpus and re-arms five boundary occupants. Cycle 24 died on
one token occupying the boundary of a numeric guard; that lesson is bought and will not be re-bought.

**Everything else UNTOUCHED.** `_TRIGGERS`, `_TRIGGERS_CORR`, `_NUM`, `_SHAISH`, `_DATEISH`,
`_VERSIONISH`, `_MD_STRUCTURE`, `extract_numbers`, `receipt_values`, `_BULK_PATHS`, `_match` and every
v0.5 class are byte-identical. The status ladder order is byte-identical: `is_spec`/`is_hist` →
ABSTAIN, then `is_notation` → ABSTAIN, then `derived_ref`, then `hits` → VERIFIED, and only then
`bound` → UNGROUNDED. The clause therefore cannot reach any token the v0/v0.1 SPEC-CONSTANT and
QUOTED-HISTORICAL rules already abstain — the defect that killed v0 stays closed only because of this
placement, and any future move of the clause above those rules re-opens it at full strength.

Disclosed floor the fix cannot cross: `is_spec` requires an operator character in the 18-character
window immediately before the token. A bar written in JSON syntax puts its operator in a separate
`"op"` field, and 157 of 157 such bar tokens repo-wide are rescued zero times today. Exactly one of
those 157 currently exceeds four decimals, and it sits at five, below this threshold. The class is
named, sized and left unfixed here; `is_spec` JSON-idiom recall is owed and out of scope.

## Secondary class (severable, accusation-only): ULP-neighbour escape

```python
V07_ULP_ESCAPE = True
V07_ULP_N = 8

# when the obligation came from the precision clause ALONE and nothing matched:
#   if some receipt leaf lies within V07_ULP_N ULP of the claim value
#   -> ABSTAIN, ref "ulp-neighbour:<receipt>:<path>"   (never VERIFIED)
```

v0.6.2 withdrew the epsilon subsidy for claims at ≥13 decimals, so at `doc_dec = 16` the tolerance is
5e-17 while the float64 ULP near 1.0 is 1.11e-16. Two values that are the same measurement computed by
differently-ordered arithmetic differ by one ULP and do not match. That was safe while such tokens
were never obligated; the primary clause makes it live, and it converts a representation difference
into a loud accusation. Two of the three tokens the pre-fix census flags are exactly this: one is a
doc value one ULP above a leaf that is the float subtraction of two already-rounded rates, where the
document's `34/75` is the *more* accurate of the two.

**This class may downgrade an UNGROUNDED to ABSTAIN only, never produce a VERIFIED, and never soften
an obligation that `_TRIGGERS`, `_TRIGGERS_CORR`, `n=` or the range-sanity rule created.** It cannot
re-open the v0.6.2 epsilon hole, because the hole was false *verification* and this yields abstention
with a distinct, countable ledger reason — an enumerable residual rather than an invisible one.
Severability: if G5 fails, the class is dropped and the precision clause ships alone.

## G3 artifact definition (frozen, to remove post-hoc judgment)

A new clean UNGROUNDED is an **ARTIFACT** iff its token is **not a machine-emitted measurement** — a
hand-authored specification constant (bar, floor, threshold, α, tolerance), a mathematical constant, an
identifier (DOI, arXiv id, version, seed), or a value quoted from a source this repository could never
hold a receipt for. A token that IS a measurement but is merely **un-persisted as a summary leaf** —
including one persisted only inside a string, only inside a bulk array, or only as a differently-rounded
float — is a **REAL** doc↔receipt gap, not an artifact. The cure for a REAL gap is persisting the value
in an addendum receipt, or printing it at a precision its receipt supports; never widening `_match`.

Worked examples on both sides, from the pre-fix census, pre-declared as EXPECTED:

| token | doc | class |
|---|---|---|
| `0.2711864406779661` | `FINDING_third_party_bench_2026_07_24.md` L40 | **REAL** — bare-arm accuracy 16/59, in the receipt only inside a gate detail *string*; the addendum that cures it already exists |
| `0.4533333333333333` | `FINDING_scale3b_2026_07_29.md` L23 | **REAL** — an MMLU capability drop, exactly 34/75; the receipt persists the float-subtraction variant one ULP away |
| `0.9918699186991871` | `DATASHEET_conscience_2026_07_24.md` L65 | **REAL** — an inter-channel agreement rate, 122/123; the doc's trailing digit is spurious and no computation on this data produces it |
| `0.00648` (5 dp, below threshold) | `PREREG_b35c_open_vocab_2026_08_03.md` L29 | **ARTIFACT** if ever obligated — a frozen kill-gate bar whose receipt is the prereg itself |
| `3.14159` (5 dp, below threshold) | `CHANGELOG.md` L4006 | **ARTIFACT** if ever obligated — π in a worked example |

## Battery + gates (harness `run_oath_v07_battery.py`, seed 1)

Sampling frame, stated with every condition: documents under `papers/**` that carry a
`*.certificate.json`, excluding `anc/` staging copies, whose recorded receipts ALL resolve next to the
document with matching SHA. Tokens are those `extract_numbers` returns at ≥7 fractional digits, split
by the SHIPPED v0.6.2 obligation predicate into an unbound pool (G2) and a bound pool (G4). Mutation
perturbs one significant fractional digit among positions 1–6, seeded. Every mutation gate runs in
BOTH arms, flag OFF and flag ON, on the identical sample with the identical mutation seed.
Non-destructive: mutants live in temp files, the corpus pass is in-memory, and the only file written
is the battery's own result JSON.

## Frozen gates

- **G1 (recall):** ≥18/20 of the sampled unbound tokens are returned by `extract_numbers` — inherited
  bar and shape from `PREREG_oath_v061` (bars never move).
- **G2 (catch):** ≥16/20 mutants on UNBOUND lines land UNGROUNDED with the clause ON. **New, frozen
  here — the decisive bar.**
- **G2b (reported, not gated):** the same 20 mutants with the clause OFF — the positive control. The
  OFF arm must be ≤2. **If the ON arm does not exceed the OFF arm the run is VOID**, whatever the
  other gates read: a battery that reports the same number in both arms is not measuring the clause.
- **G2c (reported, not gated):** the corpus-wide silent-pass residual from `silentpass_census.py` —
  the share of all currently-VERIFIED claims that survive a one-digit mutation without accusation,
  its decimal-width distribution, and the fraction of that surface this clause closes. The RESULT and
  the CHANGELOG entry must both carry this number. A release that reports the catch without the
  residual launders its own coverage, which is the defect this instrument exists to refuse.
- **G3 (no false accusations):** every certificate HELD at baseline remains HELD unless its new
  UNGROUNDED token hand-verifies as a GENUINE doc↔receipt discrepancy under the frozen artifact
  definition above (a catch: reported, and repaired under its own commit). The three tokens tabled
  above are pre-declared EXPECTED. One non-genuine new UNGROUNDED anywhere = the fix FAILS and does
  not ship. Zero baseline UNGROUNDED tokens may be lost.
- **G4 (no tamper regression):** on the BOUND pool the ON arm catches ≥16/20 and never fewer than the
  OFF arm; across G2 and G4 combined, zero mutants land VERIFIED.
- **G5 (severability):** with `V07_PRECISION_OBLIGATION` and `V07_ULP_ESCAPE` both OFF, the ledger is
  status-identical to the shipped verifier across all resolvable documents. Any difference → the
  clause is not severable and does not ship.
- **G6 (suite):** `python -m pytest tests -q` green; `py_compile` on every touched `.py`.

## Outcome table (pre-committed)

- All gated bars pass → **v0.7 SHIPS.** The RESULT publishes the corpus VERIFIED/ABSTAIN/UNGROUNDED
  delta with a per-document table, the repair loop for every genuine catch (each under its own commit,
  no committed result receipt modified, computed-never-persisted values into an addendum receipt with
  provenance), the G2c residual, and the `ulp-neighbour` roster. CHANGELOG entry carries the residual.
  Issue #39 is updated with the measurement and **is not closed as "the gap is closed"** — the clause
  closes a minority of the tamper surface and the issue text must be corrected to say which.
- G2 misses → `V07_INSUFFICIENT`; revert `styxx/certify.py`, publish the negative.
- G2b positive control fails (ON ≤ OFF) → `V07_BATTERY_VOID`; no verdict is recorded, the harness is
  the defect, and the family waits for a corrected harness.
- G3 finds one non-genuine new UNGROUNDED → `V07_FALSE_ACCUSATION`; revert, publish.
- G4 misses → `V07_TAMPER_REGRESSION`; revert, publish.
- G5 fails → the ULP class is dropped and the precision clause is judged on G1–G4/G6 alone.
- G6 fails → fix the suite or revert; no verifier ships with a red suite.
- No second attempt inside this cycle: a miss hands the lead to the owed `is_spec` JSON-idiom repair
  and to status-level claim→field binding for floats, in that order.

## Artifacts

- `papers/closed-model-frontier/run_oath_v07_battery.py` → `oath_v07_battery_result.json`.
- Pre-fix censuses, committed beside this prereg: `oath_v07_precfix_census.json` (unbound pool,
  decimal sweep) and `oath_v07_silentpass_census.json` (the G2c residual), both generated at the
  shipped verifier before any edit.
- A short RESULT note, itself certified (`python -m styxx.certify`, OATH-HELD) before commit.

Out of scope (named so they cannot creep in): status-level claim→field binding for floats (the
standing v0.4 debt, and the instrument that would actually attack the false-verify channel);
`is_spec` JSON-idiom recall (157/157 unrescued, sized above); fence and inline-code awareness in
`extract_numbers` (which has none, and whose docstring wrongly claims it drops fenced spans); bare
integers, which no precision predicate can ever reach; any change to `validate_oath_v0.py`,
`run_oath_v061_battery.py` or `mutant_battery.py` — bars and the instrument never move.

---

*Frozen on commit. The bar structure outranks the upgrade.*
