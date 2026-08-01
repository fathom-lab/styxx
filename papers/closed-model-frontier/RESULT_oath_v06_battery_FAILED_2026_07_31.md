# RESULT — OATH v0.6 battery: `SHASCRUB_FIX_INSUFFICIENT` (the pre-committed negative), and the battery caught a second verifier defect

Fathom Lab · 2026-07-31 · scored under `PREREG_oath_v06_shascrub_recall_2026_07_31.md` (frozen
at `29094e1` with both baselines committed before any change to `certify.py`).
Receipt: `oath_v06_battery_result.json`. Harness: `run_oath_v06_battery.py` (seed 1,
non-destructive, mutants in temp files only).

## Verdict

**G1 PASS — G2 FAIL — the fix does not ship.** Per the frozen outcome table, `certify.py` was
reverted in this same commit; the working tree never carried an un-validated verifier past this
line. The v0.6 recall fix is *necessary* (G1) but *not sufficient* (G2): making full-precision
decimals visible does not yet make them falsifiable.

- **G1 (recall): 19/20 extracted, bar 18 — PASS.** Pre-fix these tokens extract at 0/20 by the
  defect; the one miss is a token behind a Unicode-minus sign (reported in the receipt).
- **G2 (catch): 2/20 mutants UNGROUNDED, bar 16 — FAIL**, decomposing into three mechanisms,
  each named below.

## What the failed battery measured (the keeper)

1. **The epsilon hole — a SECOND real verifier defect.** `_match`'s tolerance is
   `0.5·10^-dec + 1e-12`. The flat `1e-12` term is invisible at ≤6 decimals and fatal at 16:
   any mutation in fractional digits ≥13 of a full-precision decimal still VERIFIES
   (demonstrated live: digit-13 and digit-16 mutants of `0.12133891213389121` both pass
   `_match`; a digit-10 mutant fails). 3/20 battery mutants false-verified through exactly this
   hole. Full-precision claims were not just invisible pre-fix — even made visible, their
   trailing digits are not actually sworn. Double-precision floor noted: two decimal strings
   differing only in the 16th significant digit may parse to the SAME float64; such mutations
   are unfalsifiable by ANY float-comparing verifier and must be excluded from a battery by
   design, not absorbed as misses.
2. **Receipts-unresolved sampling — a harness design flaw, owned.** 6/20 sampled tokens live
   in docs whose certificates' receipts do not resolve next to the doc (the corpus baseline's
   standing `unresolved 27` class). No mutant test can run there; the prereg's sampling frame
   should have conditioned on resolvable receipts and did not. These items measure corpus
   receipt-placement, not the fix.
3. **Abstain-degrade dominance.** 9/20 mutants extracted but landed ABSTAIN: their lines carry
   no trigger vocabulary, so the mutated value is not *obligated* — the standing v0.4
   trigger-recall boundary, which applies identically to 4-decimal claims on the same lines.
   The fix restores full-precision decimals to *parity* with short decimals; the frozen 16/20
   bar implicitly demanded MORE than the verifier delivers on any decimal class, and the
   battery said so.

## Disposition

`SHASCRUB_FIX_INSUFFICIENT` recorded; no bar moved. G3–G6 not run (moot after the kill-gate).
The successor prereg (`PREREG_oath_v061_shascrub_epsilon_2026_07_31.md`) carries BOTH fixes —
the SHA-scrub recall change unchanged, plus closing the epsilon hole (no flat `1e-12` for
high-precision claims) — with a battery corrected for all three measured mechanisms: sampling
conditioned on resolvable receipts, mutation restricted to significant fractional digits 1–6
(representable, semantically meaningful), and the catch gate scoped to trigger-bound lines
with the unbound-line abstain share reported beside it (the trigger-recall debt stays a named,
separate lead). Bars frozen there before the re-run.
