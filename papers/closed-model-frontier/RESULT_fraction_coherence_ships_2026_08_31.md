# RESULT — V11 FRACTION-COHERENCE ships: the verifier stops taxing its own mandated disclosures

Fathom Lab · 2026-08-31 · Prereg: `PREREG_fraction_coherence_2026_08_31.md`, amended before
the formal gate run by `AMENDMENT_fraction_coherence_2026_08_31.md`. Receipts:
`fraction_coherence_ab.json`, the three re-issued certificates, `tests/test_fraction_coherence.py`.
The first verdict-changing verifier repair since the epistemics freeze.

## What was repaired

Three consecutive RESULTs were published OATH-FAILED on the digits of counts statements their
own preregistrations mandate verbatim. The defect — first misdiagnosed as range-sanity,
corrected in all three papers — is the v0.3 count-binding filter: correct receipt fields
named `valid` and `claims` never appear in prose, so correct denominators were stripped of
their bindings and accused.

The clause: integer operands of an explicit `A/B` whose same-line ratio r satisfies
`round(A/B, decimals(r)) == r` bind jointly iff both values sit under one common receipt
parent. Strictly rescue-only by construction — it fires only where the ladder would
otherwise say UNGROUNDED.

## The gates, formally

- **G-F1 (corrected fixtures) — PASS.** The three bindable specimens flip
  UNGROUNDED → VERIFIED via `derived-fraction`, each naming its common parent. The pooled
  `115` — a sum computed inside the statement, stored in no receipt — is refused both times,
  and its refusal is part of the gate: a clause that verified it would be manufacturing the
  coincidence class the filter exists to kill.
- **G-F2 (mutation battery) — PASS.** Numerator, denominator, and ratio perturbations break
  coherence; split-subtree and absent-operand mutants break joint binding. Zero mutants
  verify.
- **G-F3 (absolute corpus A/B) — PASS.** Across every certified document: exactly **three
  tokens moved**, all UNGROUNDED → VERIFIED via `derived-fraction`, **zero** wrong movements,
  **zero** documents HELD → FAILED, one document FAILED → HELD. Every move is listed in the
  receipt individually.
- **G-F4 (invariants) — PASS.** Full suite green after the re-issue flow; the epistemics
  meta-test and every issuance invariant hold, with the rescued tokens partitioned under the
  derived branch.

## What the gates caught on the way — the cycle's second story

The first implementation fired unconditionally. The absolute gate caught it re-attributing
healthy VERIFIED tokens and elevating an ABSTAIN through the degenerate `0/38`
zero-numerator case. Then a patch script silently failed to install the strict guard — the
comment said rescue-only, the code said everything — and the A/B caught the code
contradicting its own comment. Two implementation lies, both stopped by the same gate,
both in the record. And the prereg itself carried an unsatisfiable fixture (the `115`
expectation), found at harness time and amended in writing before the formal run —
the third prereg defect this lab has caught on itself this week, by the same discipline.

## The outcome, honestly bounded

The OBLIGATE-2 RESULT re-certifies **OATH-HELD** — its only accusations were the coherent
pair. The OBLIGATE-1 and STRUCT-1 RESULTs re-certify **still FAILED**: the pooled `115`, the
bare count pair *"32 of every 58"*, and the quoted *"9 new tests."* remain accused, exactly
at the clause's frozen boundary. The repair fixes the fraction-coherent class and nothing
else; two papers keep paying a smaller, correctly-diagnosed tax, and the residual classes
(computed pooled sums, bare count pairs, quoted fragments) are the named successors.

Stored history untouched throughout: prior certificates stand in git; the re-issues are new
commits with the verdict drift tracked by the audit.

---

*The verifier accused the disclosures its own protocols ordered, for three papers running.
Now it reads a fraction the way its authors do — and it took four gates, two caught
implementation lies, and one amended prereg to earn the four lines of code.*
