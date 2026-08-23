# DRAFT — OATH v0.9: audit of the SHIPPED v0.3 integer count-binding rule (NOT a published RESULT)

**This is a draft for orchestrator review. It is not a RESULT, it carries no certificate, and no bar
in it was pre-registered.** Every gate-shaped number below is post hoc by construction: the rule was
shipped in v0.3 and measured for the first time in this cycle. Fathom Lab · 2026-08-23.

**Verdict: the shipped integer count-binding rule is NET-BENEFICIAL and should stay.** It removes
3,208 of 7,095 affirmative false attestations (45.2%) across an exhaustive 20,178-mutant census, at a
hand-scored cost of one genuine verification destroyed per four demotions — a quarter of the rate that
killed the identical test on floats. The lead "the integer filter may be silently under-attesting for
months" is **closed negative on the cost question**. Three findings survive it and are not closed: the
rule's cited justification has never reproduced at any commit, the predicate carries only ~1.3x the
information of a coin weighted at its own base rate, and its accusing terminus manufactures the only
OATH-FAILED verdict in the resolvable corpus on a markdown table's row-number column.

## What was under audit

Inside `certify_doc`, guarded by `if num["decimals"] == 0 and hits:`, an integer claim's value-matches
are filtered to receipt leaves whose path shares a 4-char word stem (or a 2+-digit run) with the
claim's binding context, via a local `path_ok(p)`, with two named sub-rules: an `is_n_eq` pairing and
a `slash_pair` branch. If the filter empties `hits`, the claim falls through the ladder — UNGROUNDED
when `bound`, ABSTAIN otherwise. Shipped since v0.3 (commit `7e70cb4`), never measured.

`styxx/certify.py` was not edited in this cycle. All four angles replicate or monkeypatch the rule in
their own scripts. Angle 2 carries a positive control proving the replication is the live verifier:
5,508 ledger entries compared, 0 status differences, 0 receipt_ref differences [C].

## Corpus

138 documents under `papers/**` carrying a certificate, excluding `anc/`, whose recorded receipts all
resolve — all four angles independently found 138, one more than the 137 in the brief and in
`oath_v08_baseline_ledger.json`, because `RESULT_oath_v08_float_field_binding_CLOSED_NEGATIVE_2026_08_23.certificate.json`
was committed this cycle. Verifier pinned at `verifier_sha256 = 852aa4d0c0cb4e2304267766cd49ccabd919e33f89fa3d706ffb4fabf89c34d9`
in all four artifacts.

## The measured numbers

| # | measurement | value | src |
|---|---|---|---|
| 1 | documents in frame | 138 | R C S X |
| 2 | positive control: live `certify_doc` vs in-memory ON arm — entries / status diffs / ref diffs | 5508 / 0 / 0 | C |
| 3 | replication fidelity: post-filter `hits[0]` must equal the real ledger's `receipt_ref` | 1001 checked, 0 mismatches | S |
| 4 | integer claims in corpus | 1997 | C |
| 5 | integer claims VERIFIED with the filter OFF (value-only, the float regime) | 1260 | C X |
| 6 | integer claims VERIFIED with the filter ON (shipped) | 1001 | C S X |
| 7 | status changes caused by the filter | 259 (0.2056 of line 5) | C X |
| 8 | of which VERIFIED → ABSTAIN, across 87 of 138 documents | 255 | C |
| 9 | of which VERIFIED → UNGROUNDED, all in one document | 4 | C X |
| 10 | status unchanged but the cited receipt leaf changes | 139 | C S |
| 11 | documents OATH-FAILED, filter ON / OFF | 1 / 0 | C |
| 12 | the one flipped document's committed certificate verdict | OATH-HELD, 15/15/0 | `PROSPECTUS_knowsay_2026_07_27.certificate.json` |
| 13 | corroboration: total UNGROUNDED tokens / FAILED docs in the pre-existing v0.8 baseline (137 docs) | 4 / 1 | `oath_v08_baseline_ledger.json` |
| 14 | integer claims with value-matches whose hits the filter empties | 359 of 1529 (0.2348), landing 355 ABSTAIN + 4 UNGROUNDED | S |
| 15 | exhaustive census: every single-significant-digit substitution of every integer claim the OFF arm attests to (1188 claims, 16.98 mutants each) | 20178 mutants | X |
| 16 | mutants coming back VERIFIED, OFF → ON | 7095 → 3887 | X |
| 17 | false attestations removed | 3208 (0.4521 of 7095) | X |
| 18 | of the 3208: CAUGHT (→ UNGROUNDED) / SILENCED (→ ABSTAIN) | 914 / 2294 | X |
| 19 | mutants the filter causes to VERIFY that would not have (inversion check) | 0 | X |
| 20 | false-attestation rate per mutant, OFF / ON | 0.3516 / 0.1926 | X |
| 21 | residual false attestations with the filter ON | 3887 (0.548 of 7095) | X |
| 22 | benefit at the repo's own unit (one sampled mutation per claim, seeds 1/2/3) | 232 / 220 / 224, mean 225.3 | X |
| 23 | hand adjudication A (red team, n=40, seed 11 frozen before scoring): GENUINE / COINCIDENCE / ORDINAL / SPEC | 10 / 14 / 15 / 1 | X |
| 24 | hand adjudication B (panel lens 1, n=30, seed 11): HARM / NO HARM | 10 / 20 | P |
| 25 | v0.8 float clause on the same question, for comparison | 30 of 40 GENUINE, bar was ≤12 | `oath_v08_g4_adjudication_result.json` |
| 26 | base rate: share of ALL receipt leaves passing `path_ok` for the claim's own context, integer VERIFIED | mean 0.4287, median 0.3824 | S |
| 27 | same base rate, FLOAT control — the population v0.8 killed the test on | mean 0.4165 (absolute difference 0.0122) | S |
| 28 | filter is a pure NO-OP (removed nothing) — integer VERIFIED / all integer claims with value-matches | 0.6354 (636/1001) / 0.4787 (732/1529) | S |
| 29 | random-thinning lift, observed survivors / expected at the claim's own base rate — integers / floats | 1.3049 (4855 vs 3720.6) / 1.4372 (13879 vs 9656.8) | S |
| 30 | surviving hits per certified integer claim, before → after; share still carrying >1 leaf | 7.954 → 4.5055; 0.5524 | S |
| 31 | `is_n_eq` sub-rule: firings / VERIFIED / status changes | 63 / 0 / 0 — dead by construction (`is_n_eq` implies `is_spec`, which wins the ladder first) | S |
| 32 | `slash_pair` sub-rule: firings / VERIFIED / emptied | 122 / 102 / 16 | S |
| 33 | cited leaf's container class, integer VERIFIED: specific / generic-prefixed / generic | 612 / 273 / 116 | S |
| 34 | top terminal segments the filter certifies integers against | `seeds` 158, `i` 59, `n` 34, `n_items` 25, `steps` 25 | S |
| 35 | v0.7 census integer false attestations replayed, and reconstruction fidelity | 274 of 274, 1.0000 | R |
| 36 | of the 274, sub-rule that admitted the cited hit: stem overlap / slash-pair bypass / slash-pair countlike / n= pairing | 255 / 17 / 2 / 0 | R |
| 37 | of the 274, filter was a pure NO-OP | 163 (0.5949) | R |
| 38 | of the 274, collision landed on the array CONTAINER name, not a terminal field | 222 | R |
| 39 | of the 274, cited leaf sits in a dense integer enumeration (≥5 all-integer siblings, distinct/span ≥ 0.9) | 107 (0.3905) | R |
| 40 | union of 38 and 39 | 263 of 274 | R |
| 41 | cited justification (`27→37` catches on another experiment's `n_held=37`) driven deterministically on both `27` claims of the cited document, at HEAD | 2 of 2 still VERIFIED, ON and OFF alike | X |
| 42 | same probe at `7e70cb4` (the commit that introduced the rule) and its parent, via `git show` | VERIFIED in all four cells; `ever_caught: false` | X |
| 43 | D1 kill-gate battery (`validate_oath_v0`, seed 1) replayed in both arms: caught ON / OFF, and rows where the filter is decisive | 16/20 and 16/20; 0 of 20 | X |

Sources: **R** `oath_v09_intbind_recall_census.json` · **C** `oath_v09_intbind_cost_census.json` ·
**S** `oath_v09_intbind_structure.json` · **X** `oath_v09_intbind_redteam.json` · **P** panel lens 1,
returned in-context, **no artifact on disk** (see *The adjudication panel*).

## BENEFIT and COST, side by side

| | BENEFIT (false attestations the rule removes) | COST (clean verifications the rule destroys) |
|---|---|---|
| exhaustive unit: every 1-digit substitution of every integer claim | **3208** of 7095 removed (0.4521) — 914 caught, 2294 silenced [X] | **259** of 1260, across 87 of 138 documents [C X] |
| restricted to the 935 claims the shipped verifier actually attests to | 2029 removed, 889 caught [X] | (same 259) |
| repo's own unit: one sampled mutation per claim, seeds 1/2/3 | **225.3** mean removed [X] | (same 259) |
| ratio, unit-matched — the unit v0.8's cost/kill table used | **1.150** cost per false attestation removed [X] | v0.8 float clause: **1.056** |
| ratio, raw (mutants removed / claims destroyed) | 12.386 — **not comparable**, the numerator scales with the mutation operator and the denominator does not; the red team labels it so in its own JSON [X] | |

Two readings that the ratio alone hides, and both are load-bearing.

**The ratio is not what separates this rule from the float clause.** At the repo's own unit the integer
rule sits at 1.150 against the float clause's 1.056 — parity, the same place every float design family
sat in the v0.8 sweep (1.02 to 7.72, none beating parity). What separates them is whether the
demotions are *right*: 10 of 40 genuine here [X], 30 of 40 there [`oath_v08_g4_adjudication_result.json`].
Fisher exact on 10/40 vs 30/40, computed in this draft from those two artifacts, gives p = 1.49e-05.
The integer rule is not a better-tuned instrument. It is the same instrument pointed at a population
where most collisions are not claims.

**The filter converts false attestation into silence more often than into a catch,** 2294 to 914
[X] — the same composition finding v0.8 recorded for floats, where the silent-pass residual did not
move at all. Here it does move: 914 of the removals become real UNGROUNDED accusations. That is
better than the float clause managed, and it is also the source of the rule's only measured harm on
honest documents (see below).

## The adjudication panel

**Only one of the three lenses reached this synthesis.** The orchestrator's brief names a three-lens
panel; the structured return carried one lens ("IS THE LEAF THE CLAIM'S HOME?", n=30) and it is
truncated mid-sentence at the end of its summary. No panel artifact exists on disk — `ls
papers/closed-model-frontier/` shows no `*panel*`, `*lens*`, or `*adjudicat*` file from this cycle.
Reporting a three-lens consensus on the strength of one delivered lens would be the exact defect this
repo studies, so the panel is reported as one lens, plus the red team's independent 40-case
adjudication, and the gap is named.

### Lens 1 — IS THE LEAF THE CLAIM'S HOME? (n=30, seed 11) [P]

**10 of 30 withdrawals destroyed a genuine binding; 20 of 30 correctly withdrew a value collision.**
All 30 drawn cases are VERIFIED → ABSTAIN; none of the 4 accusations landed in this sample, so the
lens says nothing about the accusing terminus.

The 20 correct withdrawals are one coherent class — document-internal enumerators colliding with
unrelated receipt integers: 7 cycle labels ("Cycle 86", "cycle 70", "cycle 83" x3, "Cycle 5",
"cycles 74/75"), 3 claim-table row numbers, a prose list marker "(1)" matched against `lambdas[0]=1.0`,
a protocol "stage 3" matched against `lambdas[3]=3.0`, an interval endpoint "(0,1)", an AR(1) order,
two spec constants, two "approximately 0" approximations matched against RNG seeds, a seed column
whose seeds exist only as dict *keys* (so the claim has no leaf home anywhere), and one stale
cross-document "100%" matched against this run's per-class 1.0.

The 10 harms are named measurements: `n_heldout = 70` (the erratum's load-bearing denominator),
`n_pairs = 45` (the null-draw denominator the sentence reasons about), `n_fails_cells = 2` /
`n_admissible = 6` (the sentence that decides whether an erratum ships, twice), `pool.total = 349` and
`silent_false_verified = 604` — the headline denominator and the flagship number of OATH's own v0.7
result document, both silenced by OATH.

### Lens 2 and Lens 3 — NOT DELIVERED

No content, no artifact, no numbers. Whatever they measured is not in this draft.

### Independent adjudication (red team, n=40, seed 11 frozen before scoring) [X]

10 GENUINE-BINDING-DESTROYED / 14 COINCIDENCE-CORRECTED / 15 ORDINAL-CORRECTED / 1 SPEC-CORRECTED.
30 of 40 demotions right, 0.75. Extrapolated to all 259 demotions: 64.8 genuine verifications
destroyed.

### Where the two adjudications disagree

They are **not independent**: both drew at seed 11 from the same 259-row population and share 22 of
their cases. Derived in this draft by intersecting lens 1's 30 case identifiers with
`D_right_demotions.sample_ids` in [X]:

| | value |
|---|---|
| shared cases | 22 of 30 (lens 1) / 22 of 40 (red team) |
| agreement on the shared 22 | 20 of 22 (0.9091) |
| Cohen's κ on the binary genuine / not-genuine call | 0.790 |
| cases both call genuine / neither / lens-1 only / red-team only | 6 / 14 / 1 / 1 |

The two disagreements point in **opposite** directions and therefore cancel in the aggregate, which
is why the two rates agree better than the case-level record does:

- `FINDING_b42_dose_curve_2026_08_05.md` L43 token `3` — lens 1 calls it GENUINE (the leaf `ranks[2]=3`
  records which k values were swept, which is what the token denotes) and flags its own lower
  confidence; the red team calls it ORDINAL-CORRECTED. The disagreement is whether membership in a
  swept grid counts as the claim's home.
- `FINDING_promptopinion_2026_05_24.md` L33 token `100` — the red team calls it GENUINE; lens 1 calls
  it a correct withdrawal, because the "100%" refers back to a *different* run described earlier in
  the document, so the leaf it would have matched (`detector_by_class.restrained_yes = 1.0`) records
  something else. Lens 1's reasoning is document-specific and checkable; the red team's is not
  recorded per case.

**Rates, with intervals, derived in this draft (Clopper-Pearson):**

| adjudication | genuine share | 95% interval | extrapolated to 259 demotions |
|---|---|---|---|
| lens 1, n=30 | 10/30 = 0.333 | 0.173 to 0.528 | 86.3 |
| red team, n=40 | 10/40 = 0.250 | 0.127 to 0.412 | 64.8 |
| union of both, n=48 distinct, strict (genuine only if both scorers agree, or the case was scored once) | 12/48 = 0.250 | 0.136 to 0.396 | 64.8 |
| union of both, n=48 distinct, loose (genuine if either scorer says so) | 14/48 = 0.292 | 0.170 to 0.441 | 75.5 |
| v0.8 float clause, n=40 | 30/40 = 0.750 | 0.588 to 0.873 | — |

**This matters for one specific claim and I state it against the rule I am recommending keeping.**
The red team's JSON asserts `integer_rule_would_clear_the_v08_bar: true`, on 10/40 against a bar of
≤12/40 (0.30). Lens 1's rate is **0.333, above that bar.** The union's strict rate (0.250) and loose
rate (0.292) both clear it, and every interval overlaps it. So the honest statement is: the integer
rule's genuine-destruction rate is around a quarter to a third, it is a third of the float clause's
rate and separated from it decisively (p = 1.49e-05 at n=40; p = 6.6e-04 at n=30), and whether it
clears a bar of exactly 0.30 is **not settled by n=30 or n=40 and should not be asserted either way.**
A bar that was never pre-registered for integers cannot decide this in any case.

**One defect in the rubric, found in this synthesis.** Exactly one of the 4 accusations landed in the
red team's 40 (`PROSPECTUS_knowsay_2026_07_27.md` L28 token `4`) and was scored ORDINAL-CORRECTED —
"right", because the token is a table row index and not a measurement. But the shipped outcome for
that token is **UNGROUNDED, not ABSTAIN**: the same red-team document calls that row a false
accusation in its verdict text. The rubric's four classes ask whether the *value-match* was the
claim's home; they do not distinguish the *terminus*. Rescoring that one row as harmful moves the red
team's "right" share from 30/40 to 29/40 and changes no conclusion, but the rubric is demote-only in
spirit and the integer rule is not, and any follow-on adjudication needs a fifth class.

## Where the four angles disagree

**1. Angle 3 recommends deleting the rule; Angles 2 and 4 measure that deletion would be costly. I
trust Angle 4.** Angle 3's recommendation — "a prereg proposing to DELETE the integer filter... is
better supported by this data than any widening of it" — rests entirely on the predicate's information
content (base rate 0.4287, lift 1.3049, no-op 0.6354) and Angle 3 explicitly did not measure benefit
and did not adjudicate a single demotion. Angle 4 measured both: deleting the rule restores 3,208
affirmative false attestations and gives back 914 real catches, to buy back an estimated 65–86 genuine
verifications. A predicate can be weakly informative *and* net-beneficial when the population it
filters is dominated by non-claims, which is what both adjudications found and what Angle 3 could not
see from structure alone. **Angle 3's structural measurements stand; its recommendation does not
follow from them.**

**2. Angle 1 says the rule "does not stop the channel it exists for"; Angle 4 says it removes 45.2%
of it. Both are correct and they measure opposite ends.** Angle 1 measures the *residual* (274 integer
false attestations survive); Angle 4 measures the *removal* (3,208 of 7,095 removed, 3,887 residual).
Arithmetic from the two artifacts: the shipped filter cuts the per-mutant false-attestation rate from
0.3516 to 0.1926 and leaves 54.8% standing [X]. Corroboration across frames, with the frames named:
at seed 1 the red team's integer-only frame (1,188 claims, sign-aware substitution) counts 262 integer
mutants still VERIFIED with the filter ON against 494 with it OFF [X]; the v0.7 census, on its own
frame and a non-sign-aware substitution, counted 274 [R]. Close, not the same measurement.

**3. The `is_n_eq` sub-rule: Angle 2 flags 1 status-changed row as `n_eq_pairing`, Angle 3 measures 0.
Angle 3 is right.** Angle 2's flag is a reporting regex, `n\s*=\s*\d` anywhere in the binding context
(`oath_v09_intbind_cost_census.py:290`); the rule's own predicate is `\bn\s*=\s*$` anchored on the text
*preceding the token* (line 134). The flagged row is `FINDING_e1_not_estimable_2026_08_08.md` L29 token
`1` — the token is the order in "AR(1)", and the "n=300" that trips the flag is elsewhere on the line.
Angle 3's measurement (63 firings, 0 VERIFIED, 0 outcome changes) is rule-faithful and carries an
analytic proof: `is_n_eq` requires `pre` to end in `=`, which is exactly `is_spec`'s operator class,
and `is_spec` wins the ladder before `hits` is consulted. **The `is_n_eq` branch is dead code.** Both
Angle 1 (0 of 274) and Angle 3 (0 of 63) reach it independently.

**4. Three different no-op shares are reported and none contradicts another.** 0.5949 [R] is over the
274 mutant rows; 0.6354 [S] is over the 1,001 clean integer claims the verifier certifies; 0.4787 [S]
is over all 1,529 integer claims with value-matches. Different denominators, stated so nobody averages
them.

**5. Angle 3 counts 359 emptied claims, Angle 2 counts 259 status changes.** Derived here: 355 - 255 =
**100** claims whose hits the filter empties but whose status does not move, because `is_spec` /
`is_hist` / `is_notation` already fires earlier in the ladder. The filter's *reach* is 359; its
*effect* is 259.

## What was NOT measured

1. **Two of the three panel lenses.** Not delivered, no artifact, no numbers. Any claim resting on a
   three-lens consensus is unsupported by this draft.
2. **The accusing terminus was adjudicated once.** 1 of the 4 accusations landed in a hand sample, and
   under a rubric that cannot express "the withdrawal was right but the terminus was wrong". The other
   3 were classified only mechanically. The 4 accused tokens are the entire UNGROUNDED surface of the
   corpus and they were never scored properly.
3. **The 2,294 silenced mutants versus the 914 caught.** No adjudication of whether silence or
   accusation is the better outcome beyond the repo's stated doctrine. The ABSTAIN residual was not
   measured at all.
4. **The mutation operator is single-significant-digit substitution only** [X], and the v0.7 census
   frame [R] inherits a non-sign-aware `line.replace(tok, mut, 1)`. Transpositions, multi-digit edits,
   sign flips and structural tampering are unmeasured.
5. **One scorer per adjudication, no blinding, and the two samples overlap 22 cases.** The apparent
   replication of "10 genuine" across two independent panels is largely one sample scored twice.
6. **`dense_cover`** (line 39) is an operationalisation defined post hoc by Angle 1 and not
   pre-registered; a different threshold moves the 107. The container-collision count (222) does not
   depend on it.
7. **The random-thinning lift** (line 29) is a null-model ratio with no p-value and no permutation, and
   it assumes `path_ok` passage is independent of value-matching within a receipt set. It is a
   magnitude, not an inference.
8. **The document distribution of the 914 catches.** `oath_v09_intbind_redteam.json` caps
   `_caught_keys` at 400 rows in document order (`oath_v09_intbind_redteam.py:456`), so the visible
   31-document spread is a prefix, not a sample. Whether the benefit is concentrated in a few
   documents is **not measurable from the committed artifact.**
9. **Certificate drift was found and not repaired.** `PROSPECTUS_knowsay_2026_07_27.md` carries a
   committed certificate reading OATH-HELD 15/15/0 against `verifier_sha256 01f92cc1…`, while live
   re-certification against its own fully-resolving receipts reads OATH-FAILED with 4 UNGROUNDED.
   Out of scope here; it is real repository state.
10. **No second mutation seed for Angle 1**, and no cost measurement at all in Angle 1 or Angle 3.

## Recommendation

**The rule stays. Close the lead.** The hypothesis that opened this cycle — that the shipped integer
filter destroys genuine bindings at the v0.8 rate and has been silently under-attesting for months —
is measured and false. The rate is 0.25 to 0.33, not 0.75, and the rule removes 45.2% of the integer
false-attestation surface, 914 of those as real catches. No prereg to weaken, widen, or delete the
integer filter is supported by this data, and the coverage question should not be re-attempted without
new evidence. Suggested carry-forward token: **`V09_INTEGER_COUNT_BINDING_UPHELD`**.

Three things follow, in descending order of how well the measurement supports them.

**1. Retire the source comment (supported, no prereg needed).** The comment at the v0.3 count-binding
site cites a concrete catch — `27→37` verifying against another experiment's `n_held=37` — that has
never occurred. Driven deterministically on both `27` claims of the cited document it is VERIFIED at
HEAD, at `7e70cb4` which introduced the rule, and at that commit's parent which lacks it: `ever_caught:
false` [X]. The rule works; it does not work for the reason recorded in the tree. Replace the comment
with the measured reason — **the integer collision population is dominated by document-internal
enumerators** (cycle labels, table ordinals, list markers, spec constants), which is what 20 of 30 [P]
and 30 of 40 [X] correct withdrawals are — and cite this audit. Separately, `is_n_eq` is dead code
(line 31) and its removal is a pure simplification that cannot change an outcome.

**2. The accusing terminus is the one open question, and it is a real trade nobody has priced.** The
integer filter is not demote-only: an emptied hit set on an obligated claim yields UNGROUNDED. That
terminus produces 914 of the 3,208 removals [X] and 4 false accusations on honest committed documents,
flipping one document from OATH-HELD to OATH-FAILED [C X]. Under the repo's own D2 gate (zero false
UNGROUNDED on unmutated documents) and the v0.7 ship rule (raising catches while adding a single false
accusation does not ship), the rule as written would not ship today. **A prereg that proposes making
count-binding demote-only would have to test, with bars frozen first:**

- **G1, positive control (VOID condition):** the demote-only arm must produce strictly fewer UNGROUNDED
  tokens on the clean corpus than the shipped arm, and be status-identical to it everywhere else. A
  measured 0 difference voids the run.
- **G2, cost of the change, decisive:** how many of the 914 exhaustive-census catches become ABSTAIN.
  The bar must be set before the number is seen. On the doctrine that a false attestation is worse than
  a self-naming abstention, converting catches to silence is a smaller loss than manufacturing
  accusations — but 914 is not small and the bar has to say how much is acceptable.
- **G3, benefit, decisive:** false UNGROUNDED on unmutated committed documents must go to **0** (from
  4), and the OATH-FAILED count over the resolvable corpus must go to **0** (from 1). This is the whole
  point of the change and it is the one number that is not in doubt.
- **G4, adjudication with a fifth class:** re-score a frozen sample of demotions under a rubric that
  separates *the withdrawal was right* from *the terminus was right*. The existing rubric cannot
  express the defect the change exists to fix.
- **G5, severability:** flag off must reproduce the shipped ledger with 0 differences.

**3. The cheaper repair is upstream and is not a filter change (supported, worth its own narrow
prereg).** All 4 accusations and 15 of the red team's 40 hand-scored demotions [X] are markdown table
row ordinals and prose list markers — "| 3 |", "| 4 |", "(1)", "rows 4–7". Both regimes are wrong on
them: value-only matching false-verifies a non-claim against `per_item[3].i = 3` (a leaf equal to its
own subscript, which matches that integer by construction), and the shipped filter accuses it. The
correct status is ABSTAIN and neither arm reaches it. **A leading `|`-delimited ordinal column is not a
numeric claim.** That is a narrow, correct-by-construction extraction fix in the same family as the
`is_spec` JSON-idiom lead v0.8 handed forward, it removes the entire accusation surface without
touching the filter, and it would make the demote-only question in (2) much smaller.

---

*Draft. No bars were frozen before these measurements; the rule was already shipped. Every number
above traces to a named artifact, and the ones derived in this synthesis say so.*
