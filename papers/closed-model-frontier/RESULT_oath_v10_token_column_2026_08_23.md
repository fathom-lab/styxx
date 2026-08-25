# RESULT — OATH v0.10: the windows were pointed at the wrong token, and the honest fix costs a number

Fathom Lab · 2026-08-23 · scored under `PREREG_oath_v10_token_column_2026_08_23.md`, frozen before
any edit to `styxx/certify.py`. Receipts: `oath_v10_battery_result.json`,
`oath_v10_column_census.json`, `oath_v10_baseline_ledger.json`, `oath_v10_v09_keying_check.json`,
and — for the two prior-cycle figures this note cites rather than re-derives —
`oath_v08_fieldbind_precfix_census.json` and `oath_v07_silentpass_census.json`. Harness:
`run_oath_v10_battery.py`, both arms, non-destructive, mutants in temp files only.

The provenance chain is in the receipts rather than in a commit message. The census and the
baseline both record the pre-change verifier at
`1bf81d2af727b82a6b5d88f0e1885ee04c8e29fef4cbed941145f55c3e9f602e`; the battery records the
post-change verifier at `729b5e6f5dd04981973ec3fe77f7187a4d9f57dda65d705784af2ec0c5b1db7f`. The
measurements that set the bars were taken at the first; every gate was scored at the second.

**Verdict: `V10_TOKEN_COLUMN_SHIPS`.** All eight gates pass. Both flags ship `True`.

## The defect

Every cycle from v0.1 to v0.9 argued about what the context windows should MEAN. This one is about
where they ARE.

`certify_doc` built `pre` and `post` from `ctx.find(num["token"])` — the first occurrence of the
token STRING on the line, which is not necessarily the occurrence `extract_numbers` extracted.
Across the markdown documents under `papers/`, the census reads 48097 extracted tokens and finds
4612 of them anchored on a different token: 9.589 percent, spread over 841 of the 1073 documents.
When the anchor is wrong, every predicate downstream of the windows is decided against text that
does not surround the claim.

Misplacement is not automatically harm — a wrong window can still read the same answer. The census
therefore re-evaluates every window predicate at both anchors and counts only the disagreements:

| predicate | tokens whose two anchors disagree |
|---|---|
| `slash_pair` (v0.3 count-binding) | 865 |
| `is_spec` core (operator / bar vocabulary) | 745 |
| `n=` self-scope (v0.5 class F) | 124 |
| `unit_kw` (v0.3 range-sanity) | 73 |
| `unit_range` (v0.5 class B) | 27 |
| `_BAR_NOUN_POST` (v0.9, shipped off) | 20 |
| `@`-param (v0.5 class D) | 14 |
| `_JSON_BAR_KEY` (v0.9, shipped ON) | 2 |
| `sign_kw` (v0.3 range-sanity) | 2 |
| derived-percent (v0.5 class E) | 1 |

Inside the certified corpus the defect is 349 misplaced tokens, 95 of which carry a predicate that
actually disagrees. The rest are cosmetic today and one document edit away from not being.

The clearest instance is the one that indicts a clause this repository shipped three commits ago.
`PREREG_b49_amplitude_reaudit_2026_08_07.md` line 23 holds a preregistered bar in JSON value
position at column 98. `ctx.find` returns 6 — the digit inside the identifier `b45` — so `pre`
reads `"G2_b4`, `_JSON_BAR_KEY` never sees the key it exists to match, and
`V09_IS_SPEC_JSON_IDIOM`, shipped for exactly that token class after a seven-gate battery, does not
fire on a member of its own class. A clause can pass every gate written for it and still be
withheld from the tokens it was built for, if the addressing underneath it is wrong.

## The precondition nobody had checked

The obvious repair is "carry `m.start()` from the `_NUM.finditer` loop." That offset is only a
source column if the scanned string has the same layout as the source line, and it did not:

```python
scrub = _SHAISH.sub(" ", line)      # the repl is ONE space, not one space per matched character
```

A 40-character sha collapsed to a single column and shifted every token to its right 39 columns
left. A naive `m.start()` would have traded one addressing bug for another. The repair therefore
makes the substitution length-preserving, which is itself an extraction change — three filters in
`extract_numbers` read the scrub positionally.

Measured, and gated as G2: across all 1075 documents under `papers/`, the ON and OFF arms extract
an identical ordered token list, 0 documents differing. (One fewer at the pre-merge battery: the
count moved when this RESULT note joined the corpus it measures — the same movement the v0.9 cycle
recorded — and the first re-certification correctly refused the stale number.) Restoring length
preservation buys correct columns without moving extraction at all.

## Gates

| gate | role | bar | ON | OFF | verdict |
|---|---|---|---|---|---|
| G0 | sweep fidelity (VOID check) | reproduce the frozen shadow sweep exactly | 44 diffs, 0 flips | 0 diffs | **PASS** |
| G1 | anchoring, two-armed | ON == 0 misplaced, ON improves on OFF | 0 | 349 | **PASS** |
| G2 | extraction invariance, two-armed | 0 documents extract differently | 0 | 0 | **PASS** |
| G3 | no new accusation | 0 new UNGROUNDED, 0 new OATH-FAILED | 0 | — | **PASS** |
| G4a | restorations adjudicate sound | >= 20 CORRECT, 0 WRONG | 23 of 27, 0 WRONG | — | **PASS** |
| G4b | coverage destroyed | <= 10 DESTRUCTIVE | 9 of 17 | — | **PASS** |
| G4c | residual enumerable | every destructive case explained | 9 of 9 | — | **PASS** |
| G5 | severability | 0 diffs both-off, 0 diffs guard-only | 0 and 0 | — | **PASS** |
| G6a | tamper, collision channel (no-credit) | pooled caught ON >= OFF | 43 | 34 | **PASS** |
| G6b | tamper, clean roster (no-credit) | pooled caught within 10 | 272 | 275 | **PASS** |

G0 is worth naming separately. The bars in the prereg were set from a shadow sweep — the change
applied to a scratch copy outside the tree, as `PREREG_oath_v08` swept five design families before
choosing one — and G0 exists so that a bar informed by a pre-fix measurement cannot quietly become
a bar fitted to a post-fix result. The real verifier reproduced the frozen table exactly: 0 ledger
differences with both flags off, 0 with the guard alone, 45 differences and 1 verdict flip with the
primary alone, 44 differences and 0 flips with both. The transition split reproduced too, 27
ABSTAIN to VERIFIED and 17 VERIFIED to ABSTAIN, and the 44 rows are the same 44 rows.

## What the 44 status changes are

**27 restorations, of which 23 adjudicate CORRECT and 0 adjudicate WRONG.** The largest single
class is the observed column of a markdown gate table. A row reading `| G0_coverage | >= 45 pairs |
45 | PASS |` prints its bar and its measured result as the same digits, so under `ctx.find` every
token on the row anchored at the bar, `pre` ended in a comparison operator, and the MEASUREMENT was
abstained as though it were its own threshold. Ten of the restorations are that shape. This is
certification by omission — the inverse of the oath — on the one column a gate table exists to
report, and the verifier had been doing it on every such row in the corpus.

Four restorations adjudicate QUESTIONABLE: a value that now grounds in a leaf whose path is
unrelated to it. That is the standing v0.4 claim-to-field binding channel, measured by
`PREREG_oath_v08_float_field_binding` and closed NEGATIVE with kill token
`V08_COVERAGE_DESTRUCTIVE`. It is credited to nobody here.

**17 abstentions, of which 8 adjudicate CORRECT and 9 DESTRUCTIVE.** The correct ones are the
verifier finally seeing what surrounds the claim: two `@`-glued parameters that the v0.5 class D
rule always should have caught, a bar sitting in front of the word `bar`, a cycle ordinal grounded
in an unrelated count field, two historical quotations inside a disclosure note whose live twins
earlier on the same line keep their VERIFIED status, and a frozen flag constant.

The 9 destructive ones are a single defect wearing nine faces, and G4c is the gate that proved it
rather than asserting it: every one of them has `pre` ending in a bare `=`. `is_spec` reads that
character as a comparison operator. Pointed at the wrong text it rarely fired; pointed correctly it
fires on the ASSIGNMENT idiom, which in this corpus is a MEASUREMENT idiom — `n = 1`,
`n_refits=5`, `n_admissible=5`, `0.0854 = 0.0854`, `95th percentile = 1.000`.

It is named `V10_EQUALS_SPEC_OVERREACH`, sized here, and **deliberately not fixed**, for v0.8's
reason: `V07_PRECISION_DIGITS = 7` is a specification and `AUROC(S_frame) = 0.75` is not, and the
two are identical in form. The populations are not lexically separable, so any narrowing is a
doctrine change to `is_spec` with its own recall and precision trade and its own battery. Nine
tokens is the price of not guessing, paid in the open.

## The companion flag, and the false accusation it exists to prevent

The primary alone flips one committed OATH-HELD document to OATH-FAILED.
`FINDING_mapped_whitening_2026_06_12.md` reports a parenthesised stability count of five shrinkage
values out of five. Anchored correctly, that numerator has `stability` in `pre`, the v0.3
range-sanity rule fires because the count sits outside the unit interval, and a document that has
held its oath since June is accused — of a count whose receipt leaf exists and holds it.

(That sentence is itself worth noting: it originally quoted the numerator inline, and certifying
this note accused it, because a bare count beside bounded-quantity vocabulary in a document whose
receipt set does not carry that count is exactly what the oath is supposed to refuse. The prose was
rewritten rather than the receipt set widened, since a cross-directory receipt resolves only under
`corpus_audit`'s search path and would drop this document out of the battery's own frame.)

`V10_SLASHPAIR_RANGE_GUARD` says range-sanity does not fire on a slash-pair numerator: a value
written as `a/b` is a count pair, never a value of the bounded quantity named to its left. It is
not a behaviour change smuggled in beside the repair, and the battery proves it twice — with the
primary off the guard changes 0 ledger rows, and on the tamper collision channel it changes nothing
at all.

## Tamper: declared no-credit in advance, and the number that goes the wrong way

**This cycle claims no tamper improvement.** G6 was written as a regression gate, not an evidence
source, and the reason is that the honest fix makes one column look worse.

Two rosters, both substituting at the token's known column. A first-occurrence
`line.replace(token, mut, 1)` — what `run_oath_v07_battery.py`, `run_oath_v09_battery.py` and
`corpus_audit.audit_document` all do — lands on the wrong occurrence for exactly the population
under test, and would have made this leg meaningless.

The collision channel is the one the second named instance describes: mutants whose doctored token
collides on a line where the clean token did not, so no roster built on the untampered corpus can
see them. Over seeds 1 to 10 it produced 434 mutants:

| | OFF | ON |
|---|---|---|
| caught (UNGROUNDED) | 34 | 43 |
| falsely attested (VERIFIED) | 233 | 271 |
| abstained | 167 | 120 |

Read naively that is a safety regression: 47 abstentions became 9 catches and 38 false
attestations. The naive reading is refused here for a stated reason rather than waved away.

An abstention produced by a misplaced window is not a safety property. It is arbitrary — the same
doctored number on a line whose earlier text happened to differ would not have abstained. Those 47
are not the instrument withholding its oath; they are the instrument failing to look and being
credited for it. The ON column is what the shipped rules actually say about the text that actually
surrounds the claim.

The false attestations that surface are the v0.4 debt, which v0.8 measured at 604 false
attestations among the 3951 VERIFIED claims v0.7's census mutated, and closed NEGATIVE. This cycle does not widen that channel. It stops hiding part of it
behind windows aimed at another token. Making a known false-attestation rate visible is not
creating it, and a cycle that declined to fix its addressing in order to protect a flattering
tamper column would be committing v0.9's error one register over — where v0.9 refused a clause that
bought its tamper number by destroying coverage, this one accepts a worse-looking tamper number
because the coverage it restores is real.

On the clean misplaced roster over seeds 1 to 5 the same measurement is flat: 275 catches become
272, well inside the tolerance of 10 frozen in the prereg.

## An observation about the v0.9 harness

`make_oath_v09_baseline.py` keys a ledger row `<doc>|L<line>|<token>`. That key collides whenever a
line carries the same token string twice, and the dict keeps only the last write. Over this corpus
the collision merges 199 rows across 177 colliding line-token pairs, leaving 5367 distinct keys for
5566 extracted tokens.

A duplicated token is precisely the population this cycle addresses, so a severability leg keyed
that way is blind in principle to some of the tokens under test. `make_oath_v10_baseline.py`
appends the ledger ordinal for that reason.

This is an observation about the harness, not a re-opening of the verdict, and the check reports
the number that settles it: of the merged rows, 0 hide a status different from the one that
survived. v0.9's G5 read zero over a partially merged ledger, and it would have read zero over a
complete one.

## Corpus effect

Across the 139 documents in the baseline frame and their 5566 tokens, the verdict column does not
move: 138 OATH-HELD and 1 OATH-FAILED before, and the same after. No certificate changed its
verdict, no accusation was created, and no accusation was silenced.

Certificates written after this cycle carry one additional integer per ledger row, `col`. It is
additive: no existing key changes value or type, and `styxx/seal.py`, `styxx/corpus_audit.py` and
the test suite read `status`, `line` and `token` only. Committed certificates are untouched.

## Residual

- `V10_EQUALS_SPEC_OVERREACH` — 9 destroyed bindings in the certified corpus, 745 predicate
  disagreements repo-wide in the class that contains it. Owed to a successor prereg.
- The first-occurrence substitution in `corpus_audit.audit_document`'s tamper loop and in the v0.7
  and v0.9 batteries. The same class of defect in the AUDITOR rather than the verifier, now cheaply
  fixable with `col`, and deliberately left: bars and instruments do not move inside the cycle that
  would benefit from moving them.
- The 254 misplaced tokens in the certified corpus whose predicates happened to agree at both
  anchors. They were never repaired by luck; they were unexposed by luck, and the repair means they
  cannot become exposed by an unrelated edit to their line.

---

*Scored against bars frozen before the edit. The bar structure outranks the upgrade.*
