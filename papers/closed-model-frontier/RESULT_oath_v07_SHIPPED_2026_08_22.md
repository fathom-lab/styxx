# RESULT — OATH v0.7 SHIPS: printed precision obligates, and the debt it closes is a small part of the debt

Fathom Lab · 2026-08-22 · scored under `PREREG_oath_v07_precision_obligation_2026_08_22.md` (frozen at
`e463824`, before any edit to `styxx/certify.py`). Receipts: `oath_v07_battery_result.json`,
`oath_v07_precfix_census.json`, `oath_v07_silentpass_census.json`, `oath_v07_baseline_ledger.json`,
`oath_v07_corpus_after.json`. Harness: `run_oath_v07_battery.py` (seed 1, non-destructive, mutants in
temp files only). Shipped at `ccd19aa`; the single repair at `74d843b`.

## What the census found before the fix

The trigger-recall debt is usually quoted as a share of the full-precision pool, and that share
replicates: of 349 tokens at seven or more fractional digits across the 136 documents whose cited
receipts all resolve, 183 sit on lines the verifier never obligates — 0.5244 of the pool.

The share hides the mechanism. 177 of those 183 are already VERIFIED on the clean corpus, because
VERIFIED is awarded on a value-match to a receipt leaf whether or not the number was ever obligated;
180 of the 183 value-match something. An unbound number in an honest document is not unchecked. The
hole opens only under tamper: change the number, the value-match disappears, the obligation never
fires because it never depended on the value, and the mutant lands in ABSTAIN while the document keeps
its OATH-HELD verdict.

That is not hypothetical. `oath_v062_repair_addendum.json` was built in the v0.6.2 repair loop to
persist a bare-arm accuracy that lived only inside a gate detail string. The same token appears in two
documents. The one whose line carries trigger vocabulary was flagged and repaired; the other was not,
and went on citing a number its receipts did not hold. The gap steered a repair loop away from a real
provenance gap.

## The change

A token printed at `V07_PRECISION_DIGITS` = 7 or more fractional digits is obligated regardless of
line vocabulary, set at the v0.4 clause site and nowhere else, so every ABSTAIN clause above it in the
status ladder still wins. Threshold 7 rather than 5 because every live counterexample the red-team
pass produced sits at exactly five or six digits — a frozen kill-gate bar in this repo's own JSON
idiom, the half-ULP tolerance definition, π written out, a Bonferroni α, and the arXiv DOI prefix that
neither `_VERSIONISH` nor v0.5 class C reaches. Dropping to 5 buys two tokens here and re-arms five
boundary occupants; cycle 24 died on one token sitting on the boundary of a numeric guard.

A second, severable clause handles the channel the first one opens. v0.6.2 withdrew the epsilon
subsidy at thirteen or more decimals, so at sixteen the tolerance sits below the float64 ULP. Once
such tokens are obligated, a restatement of the same measurement by differently ordered arithmetic
reads as a false claim. When an obligation came from the precision clause alone and a receipt leaf
lies within `V07_ULP_N` = 8 ULP, the verdict is ABSTAIN carrying a `ulp-neighbour` reason — never
VERIFIED, so the v0.6.2 hole stays shut, and countable, so the residual is enumerable instead of
invisible.

## Gates — all pass

- **G1 (recall):** 20 of 20 sampled unbound tokens extract (bar 18).
- **G2 (catch, the decisive bar):** 20 of 20 mutants on unbound lines land UNGROUNDED (bar 16).
- **G2b (positive control, reported):** the same 20 mutants with the clause off — 0 caught, against a
  ceiling of 2. This is the number that makes G2 mean anything. A battery reporting the same count in
  both arms measures nothing, and the prereg pre-committed to voiding the run in that case.
- **G3 (no false accusations):** baseline 4 UNGROUNDED tokens corpus-wide, 5 after; one new flag,
  zero lost. The new flag is the token the prereg pre-declared, and it hand-verifies as a genuine
  gap under the frozen artifact definition. Two further tokens the pre-fix census identified were
  absorbed by the ULP clause and named rather than accused.
- **G4 (no tamper regression):** on the bound half the clause catches 18 of 20, identical to the 18
  of the off arm (bar 16). Across both mutation gates, 0 mutants land VERIFIED.
- **G5 (severability):** with both flags off, 0 ledger differences against the pre-fix baseline across
  all 136 documents.
- **G6 (suite):** the 40 certify-, oath- and corpus-audit-scoped tests pass. The full suite does not
  run clean on this host — 13 failures and 11 collection errors, all of them paging-file, subprocess
  and import-budget failures in torch-dependent and CLI modules. The identical set fails at the
  pre-fix verifier with the change stashed, so none of it is attributable to v0.7; the bar is reported
  as environment-limited rather than claimed green.

## What changed corpus-wide

VERIFIED 3977 → 3978, ABSTAIN 1416 → 1415, UNGROUNDED 4 → 4. Documents holding OATH: 135 before, 135
after. Exactly one ledger status changes across the whole corpus, and it is the repaired token.

That the shipping delta is one token is the point, not a disappointment. On honest documents this
clause is meant to be invisible; it earns its place only when a number moves.

## The one genuine catch (the repair loop)

| doc | token | gap | repair |
|---|---|---|---|
| `FINDING_third_party_bench_2026_07_24.md` L40 | `0.2711864406779661` | the bare-arm accuracy is present in `third_party_bench_result.json` only inside the KG2 gate detail string, which `receipt_values` does not traverse | receipt-set extension with the existing `oath_v062_repair_addendum.json`, re-certified under its own commit |

Checked against the cycle-26 kill, where repair-by-addendum reopened the float-coincidence surface:
certifying the document with and without the addendum under the same verifier moves exactly one
status. No committed result receipt was modified.

## The ULP roster (named, not accused)

| doc | token | neighbouring leaf |
|---|---|---|
| `DATASHEET_conscience_2026_07_24.md` L65 | `0.9918699186991871` | `scale_channel_result.json` agreement rate, one ULP below |
| `FINDING_scale3b_2026_07_29.md` L23 | `0.4533333333333333` | `scale3b_result.json` capability drop, one ULP above |

Both are real measurement claims, and neither is a mistake worth an accusation. The second is the
instructive one: the document prints an exact rational and the receipt persists the float subtraction
of two already-rounded rates, so the document is the more accurate of the two. Widening `_match` to
absorb them would re-open the epsilon hole v0.6.2 closed; abstaining with a named reason does not.

## What this does NOT close (G2c, reported, not gated)

The published debt line describes a pool, not the tamper surface. Mutate one significant digit of
every claim the shipped verifier certifies VERIFIED, and re-certify:

- 3951 claims mutated. 1255 are accused. **2696 are not — 0.6824 of them.**
- Of the unaccused, 2005 land in ABSTAIN, 87 stop being extracted at all, and **604 come back
  VERIFIED**, matching some other receipt leaf. An affirmative false attestation is worse than
  silence, and no precision threshold touches that channel at all.
- 135 of the 136 documents with certified claims contain at least one claim that can be changed
  without the verifier objecting.
- By decimal width, the unaccused sit at: 736 bare integers, then 289, 257, 423 and 814 at one, two,
  three and four decimals. The overwhelming majority is below any threshold a precision rule can
  usefully take.
- **v0.7 reaches 176 of the 2696.** Roughly one in fifteen.

So the honest headline is not that the trigger-recall gap is closed. It is that the gap was measured
against the wrong denominator, the real one is now on record, and this change closes a small and
well-understood corner of it.

## Scope and honesty

The verifier still cannot bind a float claim to a receipt FIELD at status level — the standing v0.4
debt, and the instrument that would actually attack the 604 false attestations, since those are
coincidental matches to unrelated leaves. `is_spec` still cannot see a bar written in JSON syntax,
where the operator sits in its own key: 157 such tokens repo-wide are rescued zero times, and only
one of them currently exceeds four decimals, below this threshold by luck rather than design.
`extract_numbers` has no fence or code-span awareness whatsoever, despite a docstring that claims it
drops fenced spans. Bare integers are permanently outside any precision predicate, and at zero
decimal width they are 736 of the unaccused. Behaviour on documents with missing or drifted receipts
is unmeasured.

Harness defect, owned: the first scored run read 20 of 20 on G1 but 19 of 20 on G2 and 18 of 20 on
G4, because `line.replace(token, mutant)` silently no-ops on tokens the document writes with a
typographic minus, which extraction normalises to ASCII. Two mutants never landed and were scored as
verifier misses. The substitution is sign-aware here; the inherited `run_oath_v061_battery.py` carries
the same bare replace and was not touched, because the instrument never moves. Both runs pass every
gate, so nothing rests on the correction.

Reproduce: `python papers/closed-model-frontier/oath_v07_census.py`,
`python papers/closed-model-frontier/oath_v07_silentpass_census.py` (about three minutes),
`python papers/closed-model-frontier/run_oath_v07_battery.py`.
