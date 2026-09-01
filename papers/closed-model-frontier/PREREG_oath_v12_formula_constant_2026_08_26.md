# PREREG — OATH v0.12: a number inside a formula is notation, and the verifier has been accusing it

Fathom Lab · 2026-08-26 · frozen BEFORE any change to `styxx/certify.py`. Bars below never move;
a missed bar ⇒ revert the clause and publish the negative, nothing ships. No optional stopping;
no bar moves after this commit.

Provenance: `SYNTHESIS_mention_and_use_2026_08_26.md` recorded the same defect in four
instruments — each infers CLAIMHOOD FROM CO-OCCURRENCE — and named a mention/use predicate as the
largest known defect in the OATH lane. `oath_mention_use_census.json` then sized the class at the
verifier this tree ships (`aba199d7…`, post `V11_ORDINAL_LABEL`) over the 181-document /
7,665-token certified frame (V 5,646 / A 2,011 / U 8). This prereg takes the ONE sliver of that
class the census shows can be closed without destroying anything. First attempt in this family.

## The defect (measured, pre-fix)

`extract_numbers` takes every numeral, including the ones inside rendered mathematics. On a line
carrying trigger vocabulary the obligation predicate then binds them — and `\Delta` is trigger
vocabulary, because `delta` is in `_TRIGGERS`.

The specimen, found on a document nobody in this lab wrote
(`RECON_oath_external_reach_2026_08_26.md`): the literal `1` in

    \text{effective\_lr} = \text{lr} \times \left(1 \pm \frac{\Delta \sigma^2}{\sigma^2}\right)

accused of being a claim whose truth condition was never met. It is a mathematical constant. It
has no truth condition at all — the same category error v0.11 spent a full cycle retracting for
row ordinals, reproduced in minutes on a stranger's README by an instrument that had just been
hardened against that very class.

It is not confined to strangers. In frame, 3 of the 8 standing accusations are digits inside a
LaTeX span, all in `SYNTHESIS_mention_and_use_2026_08_26.md` — the document that reports the
defect, accused on the formula it quotes as the example.

## The census, and the four designs it kills

Five candidate markers for "this token is quoted rather than asserted", each scored over the
frame for what it REACHES and what it DESTROYS. The cost column that matters is not
`destroys` — it is `nominal`, because destroying a verification that was sworn to an array index
or a seed is a gain wearing a loss's clothes (the coincidence channel v0.8 closed NEGATIVE):

| candidate | accusations reached | verifications destroyed | of those, coincident | of those, NOMINAL |
|---|---|---|---|---|
| quoting verb on the line | 0 | 540 | 39 | 501 |
| inline code span | 4 | 46 | 15 | 31 |
| blockquote | 0 | 17 | 0 | 17 |
| fenced code block | 0 | 1 | 0 | 1 |
| **LaTeX span on the line** | **3** | **8** | **8** | **0** |

Four of the five are dead on the numbers and are not built:

- **Quoting verb** — the broad-detector catastrophe again, in its purest form: 501 genuine
  verifications destroyed to retract **zero** accusations. This is the design a reasonable person
  reaches for first, and it is the worst one available.
- **Inline code** — reaches the most accusations and destroys 31 genuine verifications doing it.
  This corpus quotes its own receipt values in backticks constantly; silencing that silences the
  contract being kept.
- **Blockquote** and **fenced block** — reach nothing at all and are not free.

**LaTeX span is the only candidate that reaches accusations and destroys no genuine verification.**
All 8 verifications it would silence are structurally coincident: three sworn to a `seed`, one to
`per_item[90].i`, and five to `verification_roster[…].token` — a token string inside a receipt.

## The change (exact, frozen)

One clause, flag-gated, `V12_FORMULA_CONSTANT = True`: a status-level demotion to ABSTAIN with the
machine-readable reason `formula_constant`, at the `is_spec` tier — before any obligation or match
is consulted, exactly where `V11_ORDINAL_LABEL` sits. **Never non-extraction**: every silenced
token stays countable by coordinate in the certificate's `abstained` array with its reason in the
ledger row.

The predicate fires on token T iff ALL of:

1. **Inside a delimited mathematical span, by address.** `V10_TOKEN_COLUMN` is a declared,
   non-severable prerequisite; T's recorded `col` lies inside a span delimited by `$…$` or
   `$$…$$` on the same line, or inside an inline-code span whose content contains a backslash
   command (`\\[A-Za-z]+`). Position is an address, never a re-found string.
2. **The span contains a backslash command.** A `$…$` span with no `\command` inside it is not
   rendered mathematics in this corpus's idiom; it is a dollar amount or a shell prompt.
3. **Bare integer or decimal, no separators.** The token as extracted carries no thousands comma
   and no percent marker. A formula does not contain `100,000`.

**Everything else UNTOUCHED.** `extract_numbers`, `_TRIGGERS`, every shipped `is_spec` clause,
`V10_TOKEN_COLUMN`, `V11_ORDINAL_LABEL`, and every committed `*.certificate.json` are
byte-identical. The clause only enlarges the set reaching ABSTAIN before obligation.

## What this deliberately does NOT fix, stated before the gates

**The hard core of the mention/use defect is untouched and this prereg does not pretend
otherwise.** Of the 8 standing accusations, 4 carry NO quotation marker of any kind — including
the two that matter most: the DePT hyperparameter values quoted in prose (`100,000`, `300,000`)
and the truncated-line `4` in `FINDING_behavioral_sycophancy_blackbox`. A number quoted in running
prose with no syntactic marker is the general case, this clause reaches none of it, and the census
found no candidate that does.

This cycle closes a sliver because the sliver is closeable. Calling it a solution to mention/use
would be the overclaim this repository exists to prevent.

## Proof of repair, pre-committed

**This preregistration is a member of the class it proposes to close, and its certificate says so
now.** Certified against the census receipt at the pre-fix verifier it is OATH-FAILED, accused on
exactly the digits inside the formula quoted above as the specimen — the same three-token shape
as the SYNTHESIS, which is itself the same shape as the RECON. Three documents in a row, each
accused on the example it cites.

So the repair has a test transition, frozen here before the clause exists, in the manner of
v0.11's shrinking `KNOWN_VERDICT_DRIFT`: **when `V12_FORMULA_CONSTANT` lands, this document's own
live verdict must move OATH-FAILED → OATH-HELD, and `SYNTHESIS_mention_and_use_2026_08_26.md`
must do the same.** Both certificates are regenerated in the ship commit and neither is
hand-edited. If either fails to flip, the clause did not reach its own motivating case and the
outcome is `V12_UNDERREACH` regardless of what the other gates say.

Stated against myself: this makes the cycle's most legible success criterion a document written
by the same author who wrote the clause. That is a real weakness in the evidence, it is why G7's
adjudication is blind and its ties resolve against the clause, and it is why the corpus-wide
firing surface in G2 — not this document — carries the gate.

## Battery + gates (harness `run_oath_v12_battery.py`, to be built; seeds 1–10, frozen)

Two-armed (flag OFF / flag ON) at the ship-candidate verifier. Mutation operator imported from the
v0.9 battery, never copied. Non-destructive: mutants in temp files; the only file written is the
battery's own result JSON.

- **G1 — INSTRUMENT VALIDITY (VOID-producing).** Extractor replication mismatches = 0 repo-wide;
  frame at run = frame at freeze: 181 documents / 7,665 tokens / V 5,646 / A 2,011 / U 8. Any
  drift → **VOID**, not failed. Re-freeze and re-run; a VOID observes no bar.
- **G2 — FIRING-SURFACE EXACTNESS (gated, both directions).** Clause ON fires on exactly the
  11-token roster the census records for `latex_on_line` (3 UNGROUNDED + 8 VERIFIED). A 12th
  firing in frame is over-reach → FAIL. A missed roster token is under-reach → FAIL.
- **G3 — CONVERSION LEDGER (gated, exact).** Whole-frame ON vs OFF: UNGROUNDED→ABSTAIN = exactly
  the 3 enumerated coordinates; VERIFIED→ABSTAIN = exactly the 8 enumerated coordinates;
  HELD→FAILED = 0 anywhere. Any count off by one → FAIL. Stated honestly, as v0.11's G3 was:
  given G1's frozen statuses and G2's exact roster these equalities follow arithmetically, so
  this is an end-to-end implementation audit, not independent evidence.
- **G4 — THE COST IS ZERO GENUINE VERIFICATIONS (gated — the bar this design lives on).** Every
  one of the 8 VERIFIED→ABSTAIN conversions is structurally coincident under the frozen dogfood
  definition (terminal path segment is a bare subscript or an index-like name). **A single
  NOMINAL verification destroyed → FAIL**, verdict `V12_DESTROYS_GENUINE_BINDING`, revert and
  publish. This is the leg the four rejected designs fail, and it is not permitted to pass on the
  author's judgement: the definition is frozen in `oath_v11_dogfood_selfcert.py` and applied
  mechanically.
- **G5 — VALUE-BLINDNESS (gated; ten seeds).** `override_missed_mutant` = 0 at every seed:
  doctor a digit inside the formula and the clause must still fire. Nonzero → FAIL,
  `V12_FUSE`. A clause that stops firing on the input it exists to handle is a fuse, not a
  detector — the property that disqualified three v0.11 designs.
- **G6 — COLLATERAL (gated; ten seeds).** Abstentions lost elsewhere under mutation = 0 at every
  seed. Nonzero → FAIL, `V12_COLLATERAL`.
- **G7 — ADJUDICATION (human, out of battery, gated).** A panel agent, blind to this prereg and
  to the clause's reason codes, adjudicates all 11 roster tokens plus 10 non-roster tokens drawn
  from lines carrying a LaTeX span, under the frozen question *does this number assert something
  a receipt could confirm or contradict?*, **ties resolved AGAINST the clause** (toward CLAIM).
  Bars: (i) all 3 accused targets adjudicate NOTATION — a CLAIM call → FAIL,
  `V12_WARRANT_FAILED`; (ii) zero NOTATION calls among the 10 non-roster draws — a NOTATION call
  → FAIL, `V12_CLASS_UNDERENUMERATED`. **Coverage is checked before any bar is read**: an
  artifact not covering all 21 cases yields INCOMPLETE, never PASS. That guard exists because
  v0.11's equivalent gate passed vacuously on an empty list until an adversarial audit found it.
- **G8 — SUITE CLOSURE (gated).** `pytest -q` green; `ruff` clean over `styxx/`; `git diff` over
  every committed `*.certificate.json` EMPTY.
- **G9 — MECHANISM PROOF (gated).** Post-clause extracted token count unchanged at 7,665
  (non-extraction would shrink it); all 11 tokens present by coordinate with status ABSTAIN and
  reason `formula_constant`.

## Asserted invariants — NOT gated

- **A1** — HELD→FAILED = 0: by construction of an abstain-only clause. Audited, reported, and
  deliberately absent from G3's bar list. A leg that cannot fail must not gate.
- **A2** — ON-arm tamper catches on this class = 0: an identity for a structure-reading clause.
  Its audit is G5's `override_missed_mutant` = 0, and that audit IS gated.
- **A3** — the 4 markerless accusations are untouched: true by construction, since the clause
  reads only LaTeX spans. Written as scope, never sold as a passed bar.

## Outcome table (pre-committed)

- G1 drift → **`V12_BATTERY_VOID`**; re-freeze, re-run. No verdict.
- G2 over-fire → **`V12_OVERREACH`**; under-fire → **`V12_UNDERREACH`**. Revert, publish.
- G3 miss → **`V12_CONVERSION_MISCOUNT`**; revert, publish.
- G4 any nominal verification destroyed → **`V12_DESTROYS_GENUINE_BINDING`**; revert, publish.
  The clause is not narrowed to exclude the offending token: partial salvage after observing a
  miss is optional stopping wearing a lab coat.
- G5 → **`V12_FUSE`**. G6 → **`V12_COLLATERAL`**. Both revert and publish.
- G7 target adjudicates CLAIM → **`V12_WARRANT_FAILED`**; a NOTATION call in the fresh draw →
  **`V12_CLASS_UNDERENUMERATED`**; incomplete artifact → **`V12_WARRANT_INCOMPLETE`**, which is
  neither pass nor fail and blocks the ship.
- G8 red → **`V12_SUITE_RED`**. G9 miss → **`V12_MECHANISM_DRIFT`**.
- All gates pass → **`V12_FORMULA_CONSTANT_SHIPS`**. The RESULT publishes both arms of every gate,
  the conversion ledger, and the residual — including, prominently, that the general mention/use
  defect remains open and that 4 of 8 accusations were never in scope.
- No second attempt inside this cycle.

## Artifacts

`oath_mention_use_census.json` (frozen, pre-fix), `run_oath_v12_battery.py`,
`oath_v12_battery_result.json`, `oath_v12_panel.json`, and the RESULT.

## Out of scope

The general mention/use predicate for prose quotation — the hard core, explicitly left open. Any
change to `_TRIGGERS`. Any edit to a committed `*.certificate.json`. Any change to
`V11_ORDINAL_LABEL` or prior batteries; bars and instruments never move. The three other
instruments carrying this defect (`build_ledger`'s classifier is repaired, diffgate's claim
extractor is not) — each needs its own cycle. Any version bump, tag, or release, which the
operator owns.

---

*Frozen on commit. The sliver is closeable and the rest is not; saying which is which is the work.*
