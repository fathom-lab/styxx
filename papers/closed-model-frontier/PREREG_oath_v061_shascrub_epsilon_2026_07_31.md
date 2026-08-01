# PREREG — OATH v0.6.1: SHA-scrub recall + the epsilon hole (B33, second attempt)

Fathom Lab · 2026-07-31 · frozen BEFORE the second change to `styxx/certify.py`.
Successor to `PREREG_oath_v06_shascrub_recall_2026_07_31.md` (battery FAILED as pre-committed —
`RESULT_oath_v06_battery_FAILED_2026_07_31.md`; fix reverted at `fdd99ab`). Both baselines carry
over unchanged: `oath_v06_corpus_before.json` (139 certs — HELD 103, FAILED 9, unresolved 27,
verdict-drift 9, receipt-drift 1) and `oath_v06_corpus_before_tamper.json` (catch 905/2980 =
0.304, false-verify 549 = 0.184), both generated at verifier
`01f92cc14156691f0ee6a1772b62d8b013aaf78538cf0c4d31209921e08db29c`, which is again the live
verifier after the revert.

## The two fixes (exact)

1. **SHA-scrub recall (unchanged from v0.6):**
   `_SHAISH = re.compile(r"\b(?=[0-9a-f]*[a-f])[0-9a-f]{7,64}\b")` — the scrub requires ≥1
   letter, so decimal fractions stop being eaten as hashes.
2. **The epsilon hole (new, measured by the failed battery):** in `_match`, the flat `1e-12`
   term verifies any mutation in fractional digits ≥13 of a full-precision claim. Fix:

   ```python
   if doc_dec > 0:
       tol = 0.5 * 10 ** (-doc_dec) + (1e-12 if doc_dec <= 12 else 0.0)
   ```

   Claims quoted at ≤12 decimals keep the historic tolerance byte-for-byte (zero behavior
   change on the entire existing verified corpus, which quotes at ≤6); claims at ≥13 decimals
   are held to their own rounding tolerance with no epsilon subsidy. The float64 floor is
   disclosed: mutations only in the 16th+ significant digit may be unrepresentable and are
   excluded from batteries by design.
3. **Float stem-preference (attribution-only, severable)** — carried over verbatim from the
   v0.6 prereg, same G5 severability.

## Battery v2 (corrected for the three measured mechanisms)

Seeded sample (seed 1) of 20 distinct ≥7-fractional-digit decimal tokens from certified docs,
now conditioned at sampling time on: (a) the doc's certificate receipts RESOLVE next to the
doc; (b) the token's line (with table-header binding context) matches the verifier's own
trigger vocabulary (`_TRIGGERS` or the v0.4 fractional-correlation register) — the obligation
precondition the verifier has had since v0; (c) mutation targets a uniformly-chosen
significant fractional digit among positions 1–6 (representable, semantically meaningful,
outside any epsilon).

## Frozen gates

- **G1 (recall):** ≥18/20 sampled tokens extracted post-fix (pre-fix 0/20 by the defect).
- **G2 (catch):** ≥16/20 mutants flagged UNGROUNDED. Every miss reported per-item with its
  mechanism.
- **G2b (reported, not gated):** the unbound-line share of the FULL unconditioned pool —
  the trigger-recall debt's measured size on this class — printed beside the gate.
- **G3 (no false accusations):** post-fix default corpus audit: every certificate HELD at
  baseline remains HELD unless the new UNGROUNDED token hand-verifies as a genuine
  doc↔receipt discrepancy (a catch — reported, doc corrected under its own commit). One
  non-genuine new UNGROUNDED = FAIL, fix does not ship.
- **G4 (no tamper regression):** post-fix `--tamper --seed 1`: corpus catch_rate ≥ 0.304 AND
  false_verify_rate ≤ 0.184.
- **G5 (attribution severability):** stem-preference ON vs OFF at the post-fix verifier:
  per-doc status counts identical corpus-wide; only `receipt_ref` may differ; any status
  difference drops the class.
- **G6 (suite):** full pytest suite green; new unit tests pin (a) full-precision fraction
  extracted, (b) lettered sha still scrubbed, (c) digit-13 mutant of a 17-decimal claim
  REJECTED by `_match`, (d) ≤12-decimal tolerance byte-identical to the shipped behavior.

## Outcome table (pre-committed)

- All pass → v0.6.1 ships; per-doc delta table published; flagship certs re-issued.
- G2 <16 → `V061_INSUFFICIENT`: revert, publish, the class needs the trigger-recall lead
  (B-item) before it can carry a catch bar; no third attempt inside this cycle.
- G3 non-genuine → `V061_FALSE_ACCUSATION`: revert, publish the failing case.
- G4 regression → revert, publish.
- G5 fails → attribution class dropped; primary judged on the rest.

No optional stopping; no bar moves after this commit.
