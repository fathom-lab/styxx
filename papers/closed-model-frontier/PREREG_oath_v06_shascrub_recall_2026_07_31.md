# PREREG — OATH v0.6: the SHA-scrub recall class (B33)

Fathom Lab · 2026-07-31 · frozen BEFORE any change to `styxx/certify.py`.
Provenance: cycle 106 caught the defect while folding the inward arc into the knowsay paper —
the fold-in's new load-bearing numbers never entered the certificate ledger, and the OATH-HELD
verdict was true only of what the verifier could see.

## The defect (measured, pre-fix)

`certify.py`'s `_SHAISH = re.compile(r"\b[0-9a-f]{7,64}\b")` scrubs hex-ish spans from each
line before numeric extraction. The character class contains the ten digits, so the FRACTIONAL
PART of any decimal with ≥7 fractional digits (e.g. `0.5348837209302325` → fractional span of
16 chars, word boundary after the `.`) is scrubbed as if it were a commit hash, leaving `0.`
— which `_NUM` cannot match. Consequence: **full-precision quotes — the most receipt-verbatim
numbers in the corpus — are invisible to extraction and certified-by-omission.** The certified
corpus carries ~450 such tokens across ~30 documents (`PAPER_frame_locality` alone: 54).
Demonstration: pre-fix, `extract_numbers("cave rate 0.5348837209302325")` returns nothing.

## The fix (exact, single change)

`_SHAISH` requires at least one letter:

```python
_SHAISH = re.compile(r"\b(?=[0-9a-f]*[a-f])[0-9a-f]{7,64}\b")
```

Rationale: a genuine hex hash of length n contains ≥1 letter with probability
1 − (10/16)^n (≈0.963 at n=7, ≈1 − 6·10⁻⁹ at n=40); an all-digit span of ≥7 chars is
overwhelmingly a decimal fraction, a count, or an identifier — none of which the scrub should
eat. Bare all-digit integers unscrubbed by this change remain unextractable by `_NUM` (which
requires a decimal point or comma grouping), so the fix's only extraction-surface change is
the intended one: decimal fractions become visible. No other extraction rule is touched.

## Secondary class (severable, attribution-only): float stem-preference

Float claims (decimals > 0) keep value-only matching for STATUS, but when multiple receipt
leaves match, the recorded `receipt_ref` PREFERS a hit whose path shares a word stem with the
claim's binding context (the same stem test the v0.3 count-binding rule uses), falling back to
first-hit. Addresses the measured first-hit misattribution (cycle 106: a 0.6957 grounding in a
coincidental `self_verification` leaf while its true home also held the value). **This class
may change `receipt_ref` only, never a status.** Severability: if G5 fails, the class is
dropped and the SHA-scrub fix ships alone.

## Frozen gates

Baselines committed beside this prereg, generated pre-fix at the current verifier
(sha `01f92cc14156691f0ee6a1772b62d8b013aaf78538cf0c4d31209921e08db29c`):
`oath_v06_corpus_before.json` (default audit: 139 certificates — HELD 103, FAILED 9,
unresolved 27, verdict-drift 9, receipt-drift 1) and `oath_v06_corpus_before_tamper.json`
(`--tamper --seed 1`: catch 905/2980 = 0.304, false-verify 549 = 0.184).

- **G1 (recall):** draw a seeded sample (seed 1) of 20 distinct ≥7-fractional-digit decimal
  tokens from certified docs; post-fix `extract_numbers` must extract ≥18/20 from their
  original lines (pre-fix: 0/20 by the defect).
- **G2 (catch):** single-digit-mutate each of the 20 sampled tokens in place (the
  `mutate_token` harness, seed 1) and certify the mutated doc against its certificate's
  receipt set; ≥16/20 mutants flagged UNGROUNDED (the v0 D1 bar). ABSTAIN on a mutant whose
  line carries no trigger vocabulary is a miss absorbed by the bar and reported per-item.
- **G3 (no false accusations):** post-fix default corpus audit — every certificate HELD at
  baseline must remain HELD unless its new UNGROUNDED token hand-verifies as a GENUINE
  doc↔receipt discrepancy (a catch: reported, and the doc corrected under its own commit).
  One non-genuine new UNGROUNDED anywhere = the fix FAILS and does not ship.
- **G4 (no tamper regression):** post-fix `--tamper --seed 1` corpus catch_rate ≥ 0.304 and
  false_verify_rate ≤ 0.184 (the committed baseline; the newly-visible class should raise the
  numerator — full-precision mutants have essentially no coincidence surface).
- **G5 (attribution severability):** with the stem-preference ON vs OFF at the post-fix
  verifier: corpus-wide per-doc status counts IDENTICAL; only `receipt_ref` strings may
  differ. Any status difference → drop the class per severability.
- **G6 (suite):** `tests/test_certify_recall.py` and the full pytest suite green; new unit
  tests pinning: (a) a ≥7-digit decimal fraction is extracted, (b) a hex sha with letters is
  still scrubbed, (c) `_VERSIONISH`/`_DATEISH` behavior unchanged.

## Outcome table (pre-committed)

- All gates pass → OATH v0.6 ships; delta table (per-doc VERIFIED/ABSTAIN/UNGROUNDED shifts)
  published in the RESULT doc; affected flagship certs re-issued at the new verifier.
- G3 fires non-genuine → `SHASCRUB_FIX_FAILS_FALSE_ACCUSATION`; revert, publish the failing
  case, the class needs a narrower scrub (a future prereg, not a mid-run redesign).
- G1/G2 miss → `SHASCRUB_FIX_INSUFFICIENT`; revert and report.
- G5 fails → attribution class dropped, primary fix judged on G1–G4/G6 alone.

No optional stopping; no bar may move after this commit.
