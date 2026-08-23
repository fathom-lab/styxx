# RESULT — OATH v0.9: the JSON-idiom bar ships, and the prose bar is refused by its own battery

Fathom Lab · 2026-08-23 · scored under `PREREG_oath_v09_is_spec_json_idiom_2026_08_23.md`, frozen
at `579aebf` before any edit to `styxx/certify.py`. Receipts: `oath_v09_battery_result.json`,
`oath_v09_isspec_census.json`, `oath_v09_baseline_ledger.json`. Harness:
`run_oath_v09_battery.py`, both arms, non-destructive, mutants in temp files only. Shipped at
`807d079`.

**Verdict: `V09_JSON_IDIOM_SHIPS` and `V09_BARNOUN_CATCH_REGRESSION`.** One clause ships. The other
is refused by the gate that was written to refuse it, and is retained at `False` with its
measurement so that flipping one flag reproduces the negative.

## The defect

`is_spec` implements the v0.1 SPEC-CONSTANT rule: a pre-registered bar is a specification, not a
measurement, so it abstains — its receipt is the preregistration and not a result JSON. It
recognises a bar only from an operator character or bar vocabulary in an eighteen-character window
immediately before the token. A bar written in JSON idiom keeps its operator in a sibling field
that this window structurally cannot see.

Across the markdown documents under `papers/`, the census finds 145 JSON-idiom tokens in 42 docs.
The shipped verifier rescues zero of them. That is the class v0.7 named, sized and left open.

It is not hypothetical. `PREDICTION_h1_human_islands_2026_08_06.certificate.json` is a committed
certificate and it currently attests four of them. Two are the failure in its purest form: on line
56 a dip-test *p*-value bar, and on line 57 an R² bar, are both sworn VERIFIED against the single
leaf `b45_result.json:null_expectation_k20`, whose only qualification is holding the float 0.05.
Two different preregistered bars, one unrelated number, one oath taken on a coincidence.

## The measurement that decided the scope

The same doctrine has a second form: prose that names the bar after the number, as in "clears the
0.10 floor" or "against the 25 floor". In the certified corpus this class is 38 tokens, whose
status counts record 37 VERIFIED today. Extending `is_spec` over them looks like the same fix.

It is not, and one number separates them. Of the 145 JSON-idiom tokens, **0** sit on a line the
verifier's obligation predicate binds. Of the 38 prose tokens, **36** do. Only an obligated token
can be accused when it is doctored, so only an obligated token has a catch that an abstention rule
can destroy.

Mutating one significant digit in each of the 38 at the shipped verifier, across ten seeds:

| | mean | low | high |
|---|---|---|---|
| mutants CAUGHT — the accusations abstention would destroy | 18.7 | 16 | 22 |
| mutants FALSELY ATTESTED — the coincidences abstention removes | 17.4 | 14 | 20 |

The columns overlap and the per-seed difference changes sign, so a single seed would have licensed
either headline. Ten are reported for that reason. What does not move with the seed is the column
that matters: the clause takes CAUGHT to zero at every one of them, because the predicate reads the
characters around the token and a one-digit substitution leaves them unchanged. On this half,
abstention does not detect the tamper it is credited with. It stops looking.

## Gates

| gate | role | bar | ON | OFF | verdict |
|---|---|---|---|---|---|
| G1 | recall, two-armed | ON ≥ 140, OFF ≤ 2, ON exceeds OFF | 145 | 1 | **PASS** |
| G2 | adjudicated precision | ≥ 24 of 25 adjudicate BAR | 25 | — | **PASS** |
| G3 | coverage bound | ≤ 2 VERIFIED → ABSTAIN | 0 | — | **PASS** |
| G4 | no silenced accusation | 0 silenced, 0 verdict flips | 0 | — | **PASS** |
| G5 | severability | 0 ledger differences, flags off | 0 | — | **PASS** |
| G6 | catch preservation (control) | ON ≥ OFF at every seed | 0 | 16–22 | **FAIL** |

G1 is scored on the adjudication frame of 146 tokens — every token on an operator-field line,
whether or not it sits in value position — so the clause's own requirement does not define its own
test set. The frame is nearly coextensive with the clause, which the prereg disclosed in advance:
it bounds precision tightly and says little about recall.

G2's sample of 25 was drawn and frozen inside the census before the fix. Every one is the `value`
of a named gate in a preregistration's frozen gates block — the bar a run must clear, recorded
before the run. All adjudicate BAR under the frozen definition, with no ties to resolve, and there
are no false abstentions.

G3 and G4 are the load-bearing pair. Across the 137 documents whose receipts all resolve, the
shipping clause converts **0** VERIFIED tokens to ABSTAIN, silences **0** accusations, flips **0**
certificates, and creates **0** new UNGROUNDED tokens. The corpus footprint is empty because no
JSON-idiom token lives in a fully-resolvable certified document — which is also the honest residual
below.

## Why G6 was preregistered knowing it would fail

Because G3 and G4 report zeros, and a screen whose recall is unknown reporting zero on a corpus is
indistinguishable from a screen that cannot see. G6 is the positive control that makes those zeros
readable: flip one flag and the same battery reports 18.7 catches lost on average, so it detects
catch destruction when catch destruction is present.

G6 also demonstrates the trap it exists to refuse. In the arm where CAUGHT falls to zero, FALSE
ATTESTED falls with it, from a range of 14 to 20 down to at most 1. A cycle that had measured only
the second column would have reported a twenty-point improvement in false attestation and shipped
the clause. The improvement is real and it is bought entirely with silence.

## A convergence this cycle did not arrange

`RESULT_oath_v08_float_field_binding_CLOSED_NEGATIVE_2026_08_23.md` landed on `main` while this
battery was running, from a concurrent cycle that shared no code and no author with it. It attacks
the false-attestation channel from the receipt side — demote a float claim whose matches all sit on
unrelated receipt paths — and it closes NEGATIVE under `V08_COVERAGE_DESTRUCTIVE`, retained at
`False`. Two clauses, two instruments, two independently frozen preregistrations, and the same
verdict: the demotion was right about the individual tokens and wrong about the ledger, because what
it removed from the VERIFIED column it did not move into the caught column.

That is worth stating as a pattern rather than a coincidence. Every instrument this family has built
to attack false attestation has paid for it in silence, and the two cycles that measured the price
both refused to pay it. The v0.4 debt is not that the verifier lacks a demotion rule. It is that
demotion is the wrong shape of answer, and neither cycle found the right one.

## Corrections to the prereg, made here rather than in it

**I1 is not exactly true.** The prereg asserted that a token the clause abstains stays abstained
under mutation, because the predicate reads context and a single-digit substitution preserves token
length. There is one escape, and the battery found it. `certify_doc` locates a token with
`ctx.find`, which returns the first occurrence of the token *string* on the line rather than the
span extraction actually matched. In `FINDING_cot_inward_powered_2026_07_30.md` line 37, mutating
the bar 0.8 to 0.9 makes it collide with the leading digits of the sixteen-digit recovery figure
printed earlier on the same line, the window jumps 25 characters left, and the clause misses.
Measured at no more than one of 38 tokens per seed. The same defect costs G1 its last point: in
`PREREG_b49_amplitude_reaudit_2026_08_07.md` line 23 the bar in value position loses to a digit
inside a metric name, which is why recall reads 145 of the 146-token frame, and it is counted
against the bar rather than explained away. It is reported, not repaired — the window machinery is what every
rule in the ladder depends on, and it needs its own preregistration.

For the shipping clause the escape can only make it fire less, never more, so it cannot manufacture
a false abstention. That is the only reason it is tolerable to ship over.

**The suite is not fully green, and was not before this branch.** Three tests fail on `origin/main`
untouched: ledger regeneration drift, a silent-pass bench subtype count, and a stale editable
install reporting an older version than the source. None is in `certify` or OATH, and this branch
changes none of them. Three v0.9 tests were also briefly failing in the full suite while passing
alone, because `import styxx.certify as C` resolves the package attribute — the provenance function
— before `sys.modules`; fixed at `c379784`, with the same hardening applied to the harness.

## The residual, stated plainly

The class this cycle closes has **no measurable effect on any certificate today.**
Of the JSON-idiom roster's 145 tokens, exactly 4 appear in a document carrying a certificate at
all, and **0** appear in a document whose receipts all resolve. The corpus delta is therefore zero by construction, and
the value of the clause is entirely forward-looking: preregistrations are certified as the corpus
grows, and every gate bar in one of them is currently a coincidence waiting to be sworn to.

The class this cycle does **not** close is larger and is named: the 38 prose bar tokens stay
VERIFIED, and the receipt-side variant — 63 VERIFIED tokens grounding in a `frozen_gates`-like leaf,
28 of which the prose predicate would also reach — was rejected unbuilt, because it consults the
match set and therefore stops firing under exactly the mutation it exists to handle. Neither is a
coverage hole. Both are false-attestation surface, and the instrument that attacks false attestation
is status-level claim→field binding for floats, which a concurrent cycle owns and this one
deliberately does not touch.

---

*Scored against bars that did not move. The control failed, and it was kept because it failed.*
