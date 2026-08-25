# PREREG — OATH v0.10: the verifier is reading the wrong token's neighbourhood

Fathom Lab · 2026-08-23 · frozen BEFORE any change to `styxx/certify.py`. Bars below never move;
a missed bar ⇒ revert that clause and publish the negative, nothing ships. No optional stopping;
no bar moves after this commit.

Provenance: none. This is not a recall extension, a precision class, or a binding rule — every
cycle from v0.1 to v0.9 argued about *what the windows should mean*. This one is about *where the
windows are*. It resurrects no buried claim and hands no clause a second attempt.

## The defect (measured, pre-fix)

`certify_doc` builds every context window from one line:

```python
tok_at = ctx.find(num["token"])
pre  = ctx[max(0, tok_at - 18):tok_at]
post = ctx[tok_at + len(num["token"]):]
```

`str.find` returns the FIRST occurrence of the token STRING on the line. That is not necessarily
the occurrence `extract_numbers` extracted. When it is not, `pre` and `post` describe **a different
token's neighbourhood**, and every predicate downstream of them is decided against text that does
not surround the claim: `is_spec`, `is_notation`, `is_hist`, the range-sanity `unit_kw`/`sign_kw`
tests, the slash-pair branch of the v0.3 count-binding rule, the `n=` self-scope of v0.5 class F,
and the v0.5 class E derived-percent parse.

Measured at the shipped verifier by `oath_v10_column_census.py` before any edit, and recorded in
`oath_v10_column_census.json`:

| frame | tokens | misplaced | share |
|---|---|---|---|
| all 1,073 documents under `papers/` | 48,097 | **4,612** | **9.589%**, across 841 documents |
| the 139 documents whose receipts all resolve | 5,566 | **349** | 6.27% |

Misplacement alone is not yet harm — a wrong window can still read the same answer. The census
therefore re-evaluates every window predicate at BOTH anchors and counts only the disagreements.
Repo-wide, the disagreements land like this:

| predicate | tokens where the two anchors disagree |
|---|---|
| `slash_pair` (v0.3 count-binding) | 865 |
| `is_spec` core (operator / bar vocabulary) | 745 |
| `n=` self-scope (v0.5 class F) | 124 |
| `unit_kw` (v0.3 range-sanity) | 73 |
| `unit_range` (v0.5 class B) | 27 |
| `_BAR_NOUN_POST` (v0.9, shipped OFF) | 20 |
| `@`-param (v0.5 class D) | 14 |
| `_JSON_BAR_KEY` (v0.9, SHIPPED ON) | 2 |
| `sign_kw` (v0.3 range-sanity) | 2 |
| derived-percent (v0.5 class E) | 1 |

Inside the certified corpus, **95 of the 349 misplaced tokens have at least one predicate that
disagrees**. The other 254 are cosmetic today and are one document edit away from not being.

**Two live instances, re-derived by the census rather than asserted.**

1. `PREREG_b49_amplitude_reaudit_2026_08_07.md` L23 —
   `"G2_b45_island_still_lowest": {"metric": "b45_norm_seeds_qwen_below_clique", "op": ">=", "value": 5,`
   The claim is the bar `5` in value position at column 98. `ctx.find("5")` returns **6** — the `5`
   inside `b45`. `pre` reads `"G2_b4`, so `_JSON_BAR_KEY` cannot see the `"value":` key and
   **V09_IS_SPEC_JSON_IDIOM, shipped three commits ago for exactly this token class, does not
   fire.** The v0.9 cycle sized that class at 145 tokens and shipped a clause to abstain them; the
   addressing defect silently withholds the clause from members of its own class.

2. `FINDING_cot_inward_powered_2026_07_30.md` L37, with the bar doctored `0.8` → `0.9`, which is
   the state a tamper battery puts the document in —
   `valid (V2: held recovery 0.9833333333333333 vs the 0.9 floor), the naive margin (0.45) is`
   `ctx.find("0.9")` returns **25**, inside `0.9833333333333333`; the token is at **51**. `post`
   becomes `833333333333333 vs the 0.9 floor)…` and the bar-noun predicate misses. The clean
   document has no collision — **the mutation creates it** — so no roster built on the untampered
   corpus can see this channel. It is measured separately below.

## The precondition nobody has checked: the scrub is not length-preserving

The repair is "carry `m.start()` from the `_NUM.finditer` loop." That offset is only the source
column if the string being scanned has the same layout as the source line. It does not:

```python
scrub = _SHAISH.sub(" ", line)      # repl is a single space, NOT one space per matched char
scrub = _DATEISH.sub(" ", scrub)
scrub = _VERSIONISH.sub(" ", scrub)
```

`re.sub(pat, " ", s)` replaces each whole match with ONE space. A 40-character sha collapses to one
column and every token to its right shifts left by 39. So a naive `m.start()` would be a NEW wrong
column on every line carrying a sha, a date, or a version — trading one addressing bug for another.
(`line.replace("−", "-")` is length-preserving and needs no change.)

The repair therefore makes the substitution length-preserving, `lambda m: " " * len(m.group(0))`,
and that is itself an extraction change: three filters in `extract_numbers` read `scrub`
positionally (`m.start() <= 2`, `scrub[m.start()-1]`, `scrub[m.start()-2]`). **Measured pre-fix:
across all 1,073 documents and 48,097 tokens, restoring length preservation changes the extracted
token sequence on ZERO lines.** The repair buys correct columns without moving extraction. That
measurement is a gate below, not a footnote.

## The change (exact, frozen)

Two severable flags. Nothing else in `certify.py` is touched.

```python
V10_TOKEN_COLUMN = True            # primary
V10_SLASHPAIR_RANGE_GUARD = True   # companion; provably inert without the primary
```

**Primary — `V10_TOKEN_COLUMN`.** `extract_numbers` scrubs length-preservingly and records
`entry["col"] = m.start()`, the token's column in the U+2212-normalized source line. `certify_doc`
anchors its windows there instead of re-finding the string:

```python
if V10_TOKEN_COLUMN and "col" in num:
    _raw = doc_lines[num["line"] - 1].replace("−", "-")
    tok_at = num["col"] - (len(_raw) - len(_raw.lstrip()))
else:
    tok_at = ctx.find(num["token"])
```

The leading-whitespace subtraction is required because `ctx` is `.strip()`ed. `.strip()` and
`.replace("−", "-")` commute (U+2212 is not whitespace), so the two spellings of `ctx` agree.

**Companion — `V10_SLASHPAIR_RANGE_GUARD`.** The primary un-masks one latent false accusation, and
this is the whole of it. `FINDING_mapped_whitening_2026_06_12.md` L31 reads
`**all five under the 0.65 ceiling (stability 5/5)**`. Correctly anchored, the `5` in `5/5` has
`pre` ending `…(stability `, so the v0.3 range-sanity rule fires — `stability` is bounded-quantity
vocabulary and 5 ∉ [0,1] — clears `hits`, forces `bound`, and accuses a document that has held its
oath since June. The number is a COUNT (five of five shrinkage values), not a stability score, and
its receipt leaf `mapped_whitening_result.json:stability_count_under_ceiling` exists and holds 5.
The guard says range-sanity does not fire on a slash-pair numerator: a value written `a/b` is a
count pair, never a value of the bounded quantity named to its left. `slash_pair` is hoisted from
inside the count-binding block to just after `post` is built — a pure move, same expression, same
inputs. Measured pre-fix: with the primary OFF the guard changes **0** ledger rows, so it is not a
behaviour change smuggled in beside the repair; it exists solely for the case the repair reveals.

**Everything else UNTOUCHED.** `_TRIGGERS`, `_TRIGGERS_CORR`, `_NUM`, `_YEAR`, `_SHAISH`,
`_DATEISH`, `_VERSIONISH`, `_MD_STRUCTURE`, `_FORMULA_AFTER`, `receipt_values`, `_BULK_PATHS`,
`_match`, `_ulp_neighbour`, `_ctx_stems`, `_path_stems`, and every v0.5 / v0.6.2 / v0.7 / v0.8 /
v0.9 class are byte-identical. The status-ladder order is byte-identical. No window WIDTH changes:
`pre` is still 18 characters, `post` is still the rest of the line.

## Disclosed method: the bars below were set from a pre-fix shadow sweep

`styxx/certify.py` is untouched at the time of this commit. The change was applied to a scratch
copy outside the tree and swept against the shipped verifier, exactly as `PREREG_oath_v08` swept
five design families before choosing one. The sweep produced the transition roster, the
adjudication verdicts, and the tamper tables frozen below. The battery re-runs all of it against
the REAL verifier after the edit, and any disagreement with the numbers frozen here is itself a
gate failure (G0). Bars informed by a pre-fix measurement are disclosed as such; bars moved after
seeing a post-fix result would be optional stopping, and none are.

**Frozen shadow-sweep expectations:**

| flags | ledger rows differing from the baseline | verdict flips |
|---|---|---|
| `COL=off  GUARD=off` | 0 | 0 |
| `COL=off  GUARD=on` | 0 | 0 |
| `COL=on   GUARD=off` | 45 | 1 (OATH-HELD → OATH-FAILED) |
| `COL=on   GUARD=on`  | **44** | **0** |

The 44: **27 ABSTAIN → VERIFIED**, **17 VERIFIED → ABSTAIN**, **0 to or from UNGROUNDED**.

## Frozen adjudication definition (to remove post-hoc judgment)

Adjudicated per transition, against the token's TRUE neighbourhood:

* A **VERIFIED → ABSTAIN** transition is **DESTRUCTIVE** iff the token is a MEASUREMENT (a value
  produced by running something) and the shipped `receipt_ref` names a leaf that genuinely holds
  that measurement. It is **CORRECT** iff the token is a BAR, a notation artifact (`@`-param, unit
  range, arXiv id), a historical quotation, or a label/ordinal — i.e. iff the abstention is what
  the shipped rules prescribe for the text that actually surrounds it.
* An **ABSTAIN → VERIFIED** transition is **CORRECT** iff the token is a MEASUREMENT and the new
  `receipt_ref` names a leaf whose path relates to it. It is **QUESTIONABLE** iff the value grounds
  in a leaf whose path is unrelated (the standing v0.4 false-attestation channel, closed NEGATIVE
  by v0.8). It is **WRONG** iff the token is a bar or notation artifact now being sworn to.

  > **Disclosed pre-validation amendment, made BEFORE the frozen battery run, bars unchanged.**
  > The clause above collapses two different things into WRONG, and the roster exposed it. A
  > **notation artifact** (`@`-param, unit range, arXiv id) or a **historical quotation** is
  > something the shipped rules are supposed to abstain, so restoring one to VERIFIED is a defect
  > of the repair. A **bar-noun** is not: `V09_IS_SPEC_BAR_NOUN` is shipped **False** by v0.9's
  > deliberate decision, so under the doctrine that is actually shipped a bar named after its
  > number is NOT a specification and does proceed down the ladder. Scoring such a token WRONG
  > would be grading this cycle against a rule the repository decided not to ship. The rubric is
  > therefore split:
  >
  > * **CORRECT** also covers a BAR that grounds in a leaf whose path names it as a bar
  >   (`frozen_gates.*`, `*_bar`, `*_floor`, `*_threshold`, `*_ceiling`) — the certificate is
  >   attesting a bar against the receipt that records that bar, which is the best outcome
  >   available while `V09_IS_SPEC_BAR_NOUN` is off.
  > * **QUESTIONABLE** also covers a BAR that grounds in a leaf whose path does not name it as a
  >   bar — same v0.4 coincidence channel, same non-credit.
  > * **WRONG** is now reserved for a NOTATION artifact or HISTORICAL quotation restored to
  >   VERIFIED, i.e. a case where the repair defeats a rule that IS shipped ON.
  >
  > This is the same manoeuvre `certify.py`'s own module docstring records for the v0 pilot, where
  > CONTRADICTED was renamed and broadened before the frozen D1/D2/D3 run. **G4a's bars (≥ 20
  > CORRECT, 0 WRONG) do not move**, and this amendment is recorded here rather than in the RESULT
  > so that it is visibly upstream of the run rather than downstream of its outcome.
* BAR and MEASUREMENT carry the v0.9 definitions verbatim. **Ties resolve against the change**:
  ambiguous restorations score QUESTIONABLE, ambiguous abstentions score DESTRUCTIVE.

Every one of the 44 is adjudicated — there is no sample and no seed, because 44 is small enough to
score exhaustively and a sample would only add a place to hide.

## Battery + arms (harness `run_oath_v10_battery.py`)

Sampling frame for every corpus leg: the 139 documents in `oath_v10_baseline_ledger.json` —
`papers/**` documents carrying a `*.certificate.json`, excluding `anc/` staging copies, whose
recorded receipts all resolve with matching SHA. 5,566 tokens; baseline 138 OATH-HELD / 1
OATH-FAILED. Legs are scoped to that frame so that documents this cycle itself adds (a RESULT note
and its certificate) cannot be read as clause behaviour. Every gate runs in BOTH arms on the
identical sample with the identical seed. Non-destructive: mutants live in temp files, corpus
passes are in-memory, and the only file written is the battery's own result JSON.

**Baseline keying, and a defect it exposes in the v0.9 baseline.** `oath_v10_baseline_ledger.json`
keys a row `<doc>|L<line>|<token>|#<ledger ordinal>`. `make_oath_v09_baseline.py` keyed it
`<doc>|L<line>|<token>`, which COLLIDES whenever a line carries the same token string twice —
1,932 such lines repo-wide, **199 of the 5,566 rows in this corpus**. A duplicated token is
precisely this cycle's population, so a v0.9-keyed severability leg would have been structurally
blind to the tokens under test. This is recorded as a finding about the v0.9 harness, not a
re-opening of its verdict: v0.9's clauses are demote-only and its G5 read zero, but its G5 read
zero over a ledger that had silently merged 199 rows. The successor keying is used here and the
observation belongs in the RESULT.

## Frozen gates

- **G0 — SWEEP FIDELITY (gated).** The real verifier must reproduce the shadow-sweep table above
  exactly: 0 / 0 / 45+1flip / 44+0flips, and the 44 must be the same 44 rows, with the same 27/17
  split. Any disagreement means the shipped edit is not the swept edit and **the run is VOID**.
- **G1 — ANCHORING (gated, two-armed).** Across all 5,566 tokens in the frame, the ON arm must
  satisfy `ctx[tok_at : tok_at + len(token)] == token` for **5,566 of 5,566** tokens — misplaced
  count exactly **0**. The OFF arm must report **349**. **If the ON arm does not strictly improve
  on the OFF arm the run is VOID**: a battery reading the same number in both arms is not
  measuring the repair.
- **G2 — EXTRACTION INVARIANCE (gated, two-armed).** Across all 1,073 documents under `papers/`,
  the ordered list of `(line, token)` pairs produced by `extract_numbers` must be **identical** in
  both arms — bar **0** differing documents. The repair changes where a window is anchored, never
  what is extracted. Any difference means the length-preserving scrub is not inert and the primary
  does not ship, whatever else passes.
- **G3 — NO NEW ACCUSATION (gated).** With both flags ON, across the frame: **0** tokens become
  UNGROUNDED that were not UNGROUNDED at baseline, and **0** certificates flip OATH-HELD →
  OATH-FAILED. Both bars are zero. An addressing repair that manufactures an accusation on an
  honest document is doing the exact harm the instrument exists to prevent, and the one latent case
  is what the companion flag is for.
- **G4 — ADJUDICATED COVERAGE (gated, three bars, ties against the change).**
  - **G4a** — of the 27 restorations, **≥ 20** adjudicate CORRECT, and **0** adjudicate WRONG.
  - **G4b** — of the 17 abstentions, **≤ 10** adjudicate DESTRUCTIVE.
  - **G4c** — **every** DESTRUCTIVE abstention must have `pre`, at the token's true column, ending
    in a bare `=`. This is mechanically checked, not judged. It is the gate that decides whether
    the residual is a single nameable clause that a successor cycle can take, or a scatter this
    cycle cannot account for. If even one destroyed binding has another cause, the residual is not
    enumerable and the primary does not ship.
- **G5 — SEVERABILITY (gated, two bars).** `COL=off GUARD=off` → status-identical to
  `oath_v10_baseline_ledger.json` across all 139 documents, bar **0** differing rows.
  `COL=off GUARD=on` → also status-identical, bar **0**, which is what proves the companion carries
  no behaviour of its own.
- **G6 — TAMPER, NO-CREDIT (gated on regression only).** Declared in advance: **this cycle claims
  no tamper improvement and no tamper number is offered as evidence for it.** Two rosters, both
  mutating one significant digit at the token's KNOWN column — a first-occurrence
  `line.replace(token, mut, 1)`, which is what `run_oath_v07/v09_battery.py` and
  `corpus_audit.audit_document` do, lands on the wrong occurrence for exactly the population under
  test and would make this leg meaningless.
  - **G6a — collision channel** (the instance-2 channel): every baseline-VERIFIED token, seeds
    1–10, keeping only mutants whose doctored token collides on a line where the clean token did
    not. Pooled caught, ON **≥** OFF. Pre-fix sweep: 434 mutants, caught 34 → 43.
  - **G6b — clean roster**: the 349 misplaced tokens, seeds 1–5. Pooled |caught_ON − caught_OFF|
    **≤ 10**. Pre-fix sweep: 270 → 267.
  - **Both arms' full outcome tables are published, including the false-attestation column, which
    the sweep says RISES (233 → 271 on G6a, 353 → 357 on G6b).** That rise is reported as a cost
    and is explicitly NOT gated, for the reason set out under "what this cycle refuses to claim".
- **G7 — SUITE (gated).** `python -m pytest tests -q` green; `ruff check styxx` clean (a red lint
  gates the test step in this repository's CI and has silently masked the whole suite before);
  `py_compile` on every touched `.py`.

## What this cycle refuses to claim

The sweep says the repair converts silence into verdicts, and the verdicts split roughly one catch
to four false attestations on the collision channel. Read naively that is a safety regression, and
the naive reading is refused here for a stated reason rather than waved away.

An abstention produced by a misplaced window is not a safety property. It is arbitrary: the same
doctored number on a line whose earlier text happened to differ would not have abstained. The OFF
arm's extra 47 abstentions on the collision channel are not the instrument withholding its oath —
they are the instrument failing to look, and being credited for it. The ON arm's numbers are what
the shipped rules actually say about the text that actually surrounds the claim.

The false attestations that surface are the standing v0.4 claim→field binding debt, which
`PREREG_oath_v08_float_field_binding` measured at 604 of 3,951 VERIFIED tokens and closed
**NEGATIVE** with kill token `V08_COVERAGE_DESTRUCTIVE`. This cycle does not widen that channel; it
stops hiding part of it behind windows pointed at the wrong text. **Making a known false-attestation
rate visible is not creating it**, and a cycle that declined to fix its addressing in order to keep
a flattering tamper column would be doing the thing v0.9's G6 was built to expose, one register
over. The number is published because it is the honest cost of looking.

## Asserted invariants — NOT gated

A leg that cannot fail must not gate.

- **I1 — additive ledger shape.** `col` is a new key on each ledger entry; no existing key changes
  value or type. `styxx/seal.py`, `styxx/corpus_audit.py` and the test suite read `status`, `line`
  and `token` only. Certificates written after this cycle carry one extra integer per row.
- **I2 — anchoring is exact by construction, not by tolerance.** `col` is the offset the match was
  found at, in a string the same length as the source line. There is no search, no nearest-match
  and no fallback while the flag is on, so G1 is an identity check and is reported as one.

## Outcome table (pre-committed)

- G0–G5 and G7 pass, G6a and G6b pass → **`V10_TOKEN_COLUMN_SHIPS`.** Both flags ship True. The
  RESULT publishes the corpus delta, all 44 adjudications, both arms of every gate, the tamper
  tables with their false-attestation columns, the v0.9 keying observation, and the residual.
- G0 fails → `V10_SWEEP_INFIDELITY`; the run is VOID, no verdict is recorded, and the harness or
  the edit is the defect.
- G1 fails → `V10_ANCHOR_MISS`; revert, publish.
- G2 fails → `V10_EXTRACTION_MOVED`; revert, publish. The repair is not permitted to change what
  is extracted, and a length-preserving scrub that turns out not to be inert is a different cycle.
- G3 fails → `V10_MANUFACTURED_ACCUSATION`; revert, publish.
- G4a fails → `V10_RESTORATION_UNSOUND`; G4b fails → `V10_COVERAGE_TRADE`; G4c fails →
  `V10_RESIDUAL_NOT_ENUMERABLE`. Any of the three: revert the primary, publish.
- G5 fails → `V10_NOT_SEVERABLE`; revert, publish.
- G6a or G6b fails → `V10_CATCH_REGRESSION`; revert, publish. A no-credit leg still gates against
  regression.
- G7 fails → `V10_SUITE_RED`; fix the suite or revert. No verifier ships with a red suite.
- No second attempt inside this cycle.

## Named residual, handed on rather than fixed here

**`V10_EQUALS_SPEC_OVERREACH`.** `is_spec` treats a bare `=` at the end of `pre` as a comparison
operator. With windows pointed at the wrong text this rarely fired; with them pointed correctly it
fires on the ASSIGNMENT idiom, which in this corpus is a MEASUREMENT idiom: `n = 1`, `n_refits=5`,
`n_admissible=5`, `P(≥0.15)=1.0`, `0.0854 = 0.0854`, `95th percentile = 1.000`. The sweep's nine
destructive abstentions are all of this one shape, which is why G4c gates on it mechanically.

It is **not fixed here, and the reason is v0.8's**. The separating question — is the thing being
assigned a frozen constant or a measured statistic? — is not lexically separable: `V07_PRECISION_DIGITS = 7`
is a spec and `AUROC(S_frame) = 0.75` is not, and the two are identical in form. Any narrowing is
a doctrine change to `is_spec` with its own recall/precision trade and its own battery. Sized here
at 9 tokens in the certified corpus and 745 predicate disagreements repo-wide; owed to a successor
prereg, and expressly NOT owed by this one.

Out of scope (named so they cannot creep in): the `=`-idiom clause above; the first-occurrence
substitution in `corpus_audit.audit_document`'s tamper loop and in `run_oath_v07/v09_battery.py`
— the same class of defect in the AUDITOR rather than the verifier, now cheaply fixable with `col`
and deliberately left, because bars and instruments do not move inside the cycle that would benefit
from moving them; widening or narrowing any window; the v0.4 float claim→field binding debt, closed
NEGATIVE; re-running or re-opening any v0.9 verdict; any edit to an already-committed
`*.certificate.json`; and any version bump, tag or release.

---

*Frozen on commit. The bar structure outranks the upgrade.*
