# PREREG — OATH v0.6.2: SHA-scrub + epsilon + signed extraction (B33, third attempt)

Fathom Lab · 2026-07-31 · frozen BEFORE the third change to `styxx/certify.py`.
Successor to v0.6 (`SHASCRUB_FIX_INSUFFICIENT`, `fdd99ab`) and v0.6.1
(`V061_FALSE_ACCUSATION`, `f848c6c`) — each reverted per its own frozen table. Baselines carry
over unchanged (`oath_v06_corpus_before.json`, `oath_v06_corpus_before_tamper.json`, both at
verifier `01f92cc1…`, again live after the revert).

## The three fixes (exact; the first two verbatim from v0.6.1)

1. **SHA-scrub recall:** `_SHAISH = re.compile(r"\b(?=[0-9a-f]*[a-f])[0-9a-f]{7,64}\b")`.
2. **Epsilon hole:** in `_match`, `eps = 1e-12 if doc_dec <= 12 else 0.0`;
   `tol = 0.5·10^-doc_dec + eps`; the round-comparison uses `eps` in place of the flat
   `1e-12`. ≤12-decimal behavior byte-identical; ≥13-decimal claims get no epsilon subsidy.
3. **Signed extraction (new; the v0.6.1 kill):** normalize the typographic minus U+2212 to
   ASCII `-` in the line buffer BEFORE scrub and extraction
   (`line = line.replace("−", "-")`). `_NUM`'s `[-+]?` then reads it as a sign, so
   `−0.0154` extracts as a NEGATIVE claim instead of a positive one. The en-dash (U+2013,
   ranges) is deliberately NOT touched. Behavioral parity elsewhere: `_FORMULA_AFTER` and the
   range/compound guards already include ASCII `-` in their character classes.

Float stem-preference (attribution-only) carried over, same G5 severability.

## Battery + gates (harness identical to v0.6.1 — `run_oath_v061_battery.py`, seed 1)

- **G1:** ≥18/20 extracted. **G2:** ≥16/20 mutants UNGROUNDED. **G2b:** unbound share reported.
- **G3 (re-armed corpus-wide):** sign normalization changes the extraction surface for every
  typographic-minus negative in the corpus, so the full default audit runs again. Every
  baseline-HELD certificate remains HELD unless its new UNGROUNDED hand-verifies as a genuine
  doc↔receipt discrepancy. The four genuine catches already recorded in
  `RESULT_oath_v061_battery_2026_07_31.md` are EXPECTED failures here (they are real gaps,
  not verifier artifacts); their repairs happen in the post-ship repair loop, each under its
  own commit, none touching a committed result receipt. Any OTHER new failure gets the same
  hand-check; one non-genuine = `V062_FALSE_ACCUSATION`, revert, publish.
- **G4:** post-fix `--tamper --seed 1` catch_rate ≥ 0.304 AND false_verify_rate ≤ 0.184.
- **G5:** stem-pref ON/OFF status parity corpus-wide.
- **G6:** full pytest suite green; the five v0.6.1 pins restored PLUS two sign pins:
  (a) `−0.0154`-style claim verifies against a negative receipt leaf; (b) a range `L27–31`
  (en-dash) still does not extract its second half.

## Outcome table (pre-committed)

- All pass → v0.6.2 SHIPS. Repair loop follows (addendum receipts / receipt-set extension /
  one-digit doc correction for the four genuine catches; re-issue affected certs; delta table
  in the shipping RESULT doc). CHANGELOG + backlog B33 closed.
- Any gate fails by its own rule → revert, publish, `V062_<gate>`; the family then waits for
  the trigger-recall lead (no fourth attempt inside this cycle).

No optional stopping; no bar moves after this commit.
