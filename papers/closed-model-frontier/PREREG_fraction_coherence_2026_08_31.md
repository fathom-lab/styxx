# PREREG — FRACTION-COHERENCE (V11): the denominator repair, frozen before code

Fathom Lab · 2026-08-31 · A verdict-changing verifier repair, so the full discipline applies:
frozen spec, failable gates, a mutation battery, and a corpus A/B where a single wrong
movement blocks the ship. Frozen BEFORE any clause code is written.

## The defect, diagnosed from the ledger this time

Three consecutive RESULTs were published OATH-FAILED on the digits of their own mandated
counts statements ("26/58", "39/115", "44/59"). The specimen notes blamed range-sanity;
re-diagnosis against the stored ledgers shows the true mechanism (corrections appended to all
three papers at full prominence): **the v0.3 count-binding filter**. It requires an integer's
receipt leaf to share path vocabulary with the claim's line — sane protection against
coincidence-binding — but correct fields named `valid` and `claims` never appear in prose, so
correct, receipted denominators are stripped of their hits and fall bound-but-empty into
UNGROUNDED.

## The candidate, V11_FRACTION_COHERENCE — frozen here

For integer tokens A and B forming an explicit fraction `A/B` (allowing spaces), where a
decimal r **on the same line** satisfies `round(A/B, decimals(r)) == r`:

- the pair is *arithmetically self-coherent* — the line states the ratio its own operands
  produce;
- A and B then bind **jointly** iff both values match receipt leaves under **one common
  parent object** in a single receipt (e.g. `arms.positive.claims` and
  `arms.positive.valid`). The shared-subtree requirement REPLACES path-vocabulary as the
  anti-coincidence guard: two exact values whose ratio equals the stated r, co-located under
  one parent, is a categorically stronger binding signal than word-stem overlap;
- on joint binding both tokens are VERIFIED via the existing derived branch
  (`receipt_ref: derived-fraction:…`), with epistemics recorded through the standard
  annotation (first-writer semantics untouched);
- if coherence holds but joint binding fails, NOTHING changes — the ladder proceeds exactly
  as today. The clause can only rescue; it can never accuse.

Deliberately out of scope, stated now: bare count pairs with no same-line ratio ("32 of
every 58"), quoted fragments ("9 new tests"), and everything non-fractional. This is a
narrow repair for the ratio-coherent class — which is exactly the class the protocols
mandate in counts statements — not a general amnesty for integers.

## Gates — all four must pass, and G-F3 is absolute

- **G-F1 (fixtures)**: the ratio-coherent specimen tokens — 26/58 and 39/115 in the
  OBLIGATE-1 RESULT, 44/59 in the OBLIGATE-2 RESULT — flip UNGROUNDED → VERIFIED on
  re-certification, each binding into one receipt subtree. The non-coherent specimens
  ("32 of every 58", the quoted "9") must NOT flip: they stay accused, and their continued
  accusation is recorded as the clause's honest boundary.
- **G-F2 (mutation battery, every mutant must fail to verify)**: numerator+1 (breaks
  coherence), denominator+1 (breaks coherence), ratio perturbed at its own precision
  (breaks coherence), coherent pair whose operands match receipt values only in *different*
  subtrees (breaks joint binding), coherent pair with operands absent from receipts. Five
  mutants per fixture; a single verifying mutant kills the clause.
- **G-F3 (corpus A/B, absolute)**: every certified document re-certified at the pinned
  verifier with the clause on vs off. **Zero tokens may move in any direction other than
  UNGROUNDED → VERIFIED via `derived-fraction`, and zero documents may move HELD → FAILED.**
  Every moved token is listed individually in the receipt. One wrong movement blocks the
  ship regardless of every other gate.
- **G-F4 (invariants)**: the full test suite passes, including the epistemics meta-test
  (the ladder writes, only the summary reads) and every epistemics_summary issuance
  invariant, with the new derived tokens partitioned correctly.

## On shipping

If all four gates pass, the clause ships ON (flag `V11_FRACTION_COHERENCE = True`), the
three FAILED RESULTs are re-certified at the repaired verifier — verdict changes disclosed
in the audit's drift tracking, stored history untouched — and the corpus line moves however
it moves, reported exactly. If any gate fails, the clause ships OFF with the failure
published, and the tax stays paid.

## Limits

Ground truth here is arithmetic, not adjudication — no blind panel is required, and no claim
about claimhood is made. The clause narrows one filter for one structural class; the
count-binding filter's core protection (the 27→37 coincidence class) is untouched and
re-verified by the battery and the A/B.
