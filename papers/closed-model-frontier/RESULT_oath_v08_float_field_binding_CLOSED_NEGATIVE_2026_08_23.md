# RESULT — OATH v0.8: float field binding is CLOSED_NEGATIVE (`V08_COVERAGE_DESTRUCTIVE`)

Fathom Lab · 2026-08-23 · prereg `PREREG_oath_v08_float_field_binding_2026_08_23.md` (frozen before
any edit to the verifier). Receipts: `oath_v08_battery_result.json`,
`oath_v08_g4_adjudication_result.json`.

**Verdict: the clause does not ship.** It cleared every mechanical bar and died on the one gate that
asks whether its demotions are *right*. The standing v0.4 debt — status-level claim→field binding
for float claims — is closed NEGATIVE with a measured structural reason, and is no longer carried as
owed work.

## What was under test

A float claim at 1–3 fractional digits whose value-matches all sit on receipt paths unrelated to its
binding context is demoted VERIFIED → ABSTAIN, with reason `unbound-field:<receipt>:<path>`, but only
where the cited receipts carry some path the sentence names. Demote-only: it can never produce or
remove an accusation.

The target is the false-attestation channel. Of the claims the shipped verifier swears to, mutating
one significant digit leaves a large number still VERIFIED — matched to an unrelated leaf. No
obligation rule can reach them, because obligation decides whether a claim *must* match and these
already do.

## Gate table

| gate | role | bar | measured | verdict |
|---|---|---|---|---|
| G1 | positive control (VOID condition) | ON below OFF | ON 520 against OFF 627 | OK |
| G2 | benefit, decisive | at least 60 removed | 107 | **PASS** |
| G3 | cost ratio, regression guard | at most 1.5 | 1.056 | **PASS** |
| G4 | adjudication, decisive | at most 12 of 40 genuine | **30 of 40** | **FAIL** |
| G5 | severability | 0 | 0 | **PASS** |
| I1 | invariant, asserted not gated | demote-only | held exactly | HOLDS |
| G6 | suite | green | 3 pre-existing failures, 0 regressions | see below |

Kill token: **`V08_COVERAGE_DESTRUCTIVE`**.

## The measurement

The census mutates one significant digit of every claim the shipped verifier certifies VERIFIED and
re-certifies, in both arms, on an identical frame with an identical mutation seed. The gating seed is
fresh, because the operating point was chosen against the reporting seed.

At the gating seed the clause removes 107 affirmative false attestations, against a frozen bar of 60.
At the reporting seed it removes 112. The clean corpus loses 113 verifications to ABSTAIN, a cost
ratio of 1.056 against a bar of 1.5. Severability is exact: with the flag off the ledger is
status-identical to the pre-edit baseline across all 137 resolvable documents, 0 differences.

Invariant I1 held exactly as constructed — UNGROUNDED unchanged, OATH-HELD certificates unchanged,
and every single status transition was VERIFIED → ABSTAIN. Per the prereg this is asserted in the
suite and deliberately *not* gated: a leg that cannot fail must not gate.

### The honest reading of the benefit, stated because the number invites a better one

**The silent-pass residual does not move at all.** In both arms the count of mutants the verifier
fails to accuse is identical — 2608 at the gating seed, 2589 at the reporting seed. The clause
converts affirmative false attestation into silence; it creates no catches. Under this program's own
doctrine that a false attestation is worse than an abstention that names itself, that is a real
improvement to the *composition* of the residual. It is not a reduction of the residual, and a
release that reported the 107 without this paragraph would launder its own coverage.

A second honesty note on the same number: part of the 107 comes from the clause refusing to swear to
the CLEAN claim rather than to its mutant. That is exactly why G3 bounds the cost ratio and why G4,
not G2, is the gate that decides.

## G4 — why it died

Forty demotions were sampled at a frozen seed and hand-scored against the prereg's frozen definition,
with ties resolved against the clause:

| class | n | meaning |
|---|---|---|
| GENUINE-BINDING-DESTROYED | **30** | the leaf IS the claim's home; the demotion is a true coverage loss |
| SPEC-CORRECTED | 9 | a bar / floor / threshold / design parameter that should already abstain |
| COINCIDENCE-CORRECTED | 1 | the leaf records a different quantity; the claim was never earned |

Three quarters of the demotions destroy a correct verification. The prereg recorded this expectation
before the adjudication ran — it predicted two thirds to three quarters — so this is a
pre-registered negative, not a surprise, and the bar was set at what a shippable instrument must
clear rather than at what this one was expected to score.

**The structural reason, which is the actual finding.** Scientific prose names a measurement
narratively while the receipt field that holds it is structural:

| the sentence says | the receipt field is |
|---|---|
| "whole-stack r=16: 0.616–0.626" | `points[2].naive_relock_auroc` |
| "the loop beats the stubborn baseline" | `final.accuracy` |
| a table row under a λ column | `lambdas[1]` |
| "only reached 0.624 / 0.664 and 0.749 / 0.682" | `points[2].frozen_deployed_auroc` |

Path-stem overlap has no purchase on any of these. The honest population is not lexically separable
from the tampered one, because the tampered claim sits in the *same sentence* as the honest one and
inherits its vocabulary. The single largest genuine class in the sample was table parameter columns
and measurement cells addressed by array index.

This is why no threshold rescues it. Five design families were swept before the edit, over both
populations at once:

| family | cost per false attestation removed |
|---|---|
| naked stem filter (the v0.6.2 test promoted) | 3.48 |
| KEEP-widenings (spec containers, generic leaves, receipt name) | 1.18 |
| context window widened to the previous line | 1.06 |
| window + NAMEABLE (the shipped candidate) | 1.02 |
| all-hits-array-indexed | 7.72 |

None beats parity. The instrument buys roughly one honest demotion per false attestation removed no
matter how it is shaped, and an ACCUSING variant is worse still: every operating point of every
family would have put dozens of new UNGROUNDED tokens on honest documents, which the ship rule
refuses outright.

## What shipped

`V08_FLOAT_FIELD_BINDING = False`. The clause stays in tree behind its flag with the measurement
recorded in the source comment, exactly as `V05_APPROX_NOTATION` was retained after its severability
drop, so the negative is re-runnable and is not re-attempted. G5 proves the disabled clause inert:
the shipped default reproduces the pre-edit baseline ledger with zero differences. Nine regression
tests lock the default OFF and assert invariant I1.

The comment at the v0.3 count-binding site that read *"Floats keep value-only matching (v0.4 owes
them full claim→field binding)"* is retired. Floats keep value-only matching because the binding
instrument was built, measured, and found to cost more provable coverage than it repairs.

## G6 — the suite, reported as measured

The suite is **not green**, and the prereg's bar says green. Three tests fail:
`test_ledger_matches_a_fresh_regeneration_from_the_receipts`,
`test_shipped_detectors_are_complementary_not_redundant`, and
`test_mismatch_flag_is_none_in_a_consistent_env`.

All three are **pre-existing on trunk**. They reproduce identically with `styxx/certify.py` reverted
to HEAD, and the ledger failure also reproduces at the commit before this cycle's prereg. This change
introduces zero regressions; every other test passes, including the nine new ones. The bar is
reported as missed rather than reinterpreted, because bars do not move after the prereg commit — and
because the outcome does not turn on it: G4 already decided this.

The version-provenance failure is a stale local install and is environmental. The ledger failure is
real repository state and is repaired in this cycle.

## What this hands forward

1. **`is_spec` JSON-idiom recall.** The SPEC-CORRECTED column measures the surface: bars written in
   JSON idiom put their operator in a separate field, so `is_spec` never fires and the bar
   value-matches its own `frozen_gates.*` leaf. This is a clean, narrow, correct-by-construction
   repair, and it is the named next lead.
2. **The integer false-attestations.** Roughly nine in twenty of the false-attestation surface is
   bare integers that pass through the shipped v0.3 count-binding filter and false-verify anyway.
   Field binding cannot reach them, and count-binding demonstrably does not stop them.
3. **The general lesson.** Lexical binding between prose and receipt field names is exhausted in this
   codebase. Anything further on this channel needs a different kind of evidence — a claim that
   carries its own field reference, or receipts that carry the prose label alongside the value — not
   a cleverer regex.

---

*Bars frozen before the run. The bar structure outranks the upgrade.*
