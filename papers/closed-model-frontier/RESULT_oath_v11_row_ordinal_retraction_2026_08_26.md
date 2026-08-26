# RESULT — OATH v0.11: an accusation is itself a claim, and four accusations had no receipts

Fathom Lab · 2026-08-26 · scored under `PREREG_oath_v11_row_ordinal_retraction_2026_08_25.md`,
frozen on commit `cbd2864` before any edit to `styxx/certify.py`. Receipts:
`oath_v11_battery_result.json`, `oath_v11_baseline_ledger.json`, `oath_v11_panel_recheck.json`,
`oath_v11_adversarial_audit.json`, and — for the pre-fix measurements that set the bars —
`oath_v10_ordinal_census.json` and `oath_v10_panel_isclaim.json`. Harness:
`run_oath_v11_battery.py`, both arms, non-destructive, mutants in temp files only.

The provenance chain is in the receipts. The baseline pin and the four measurement receipts record
the pre-change verifier at
`729b5e6f5dd04981973ec3fe77f7187a4d9f57dda65d705784af2ec0c5b1db7f`; the battery and the second
panel record the ship-candidate verifier at
`aba199d7887bba356ec77c21db74062a371a18cdb7b3154d18224b681f507128`. The measurements that set the
bars were taken at the first; every gate was scored at the second.

**Verdict: `V11_ORDINAL_RETRACTION_SHIPS`.** All nine gates pass. `V11_ORDINAL_LABEL` ships True.

This cycle performs, once, by design, the event v0.9's G4 existed to forbid: four accusations
withdrawn and one certificate returned from OATH-FAILED to OATH-HELD. **The bar structure outranks
the retraction** — what follows is the argument for why that was licensed, and every place the
argument is thinner than it looks.

## The defect

Every cycle from v0.1 to v0.10 asked whether a token was GROUNDED. This one asks whether it is a
CLAIM.

A markdown table's first column is where this corpus writes its row numbers. `extract_numbers`
extracts them like any other token, and on rows whose own text carries trigger vocabulary the
shipped obligation predicate binds them — so a row number must ground in a receipt leaf or be
accused. A row number has no receipt, because it asserts nothing.

At the shipped verifier, over the frozen frame of 140 documents and 5681 extracted tokens
(VERIFIED 4196 / ABSTAIN 1481 / UNGROUNDED 4):

- 123 sole-token first-cell tokens — the cell content is the number and nothing else.
- 11 of those are ordinal-shaped, and all 11 sit in ONE document under the column header `#`: the
  claims table of `papers/agent-conscience/PROSPECTUS_knowsay_2026_07_27.md`, rows L25 to L35.
- **The frame's entire UNGROUNDED population was four of them** — L27, L28, L29, L32. The
  certified corpus's whole standing accusation surface was false.

The VERIFIED half of that column is worse than the accused half. L26 was sworn to against
`scale_test_result.json:per_item[2].i` — an index leaf equal to its own subscript. The oath was
taken on a coincidence, on a token that makes no claim. L35 is worse still: its row reads IN
FLIGHT and its receipt cell reads *pending*, so there is no measured quantity anywhere in the row,
and the verifier certified it anyway.

Exhaustive substitution over the class, every single-significant-digit mutant of all 11 tokens,
reproduced at the ship-candidate verifier with the clause off: of 117 mutants the shipped verifier
answers UNGROUNDED 46, VERIFIED 50, ABSTAIN 21. A false-attestation rate under tamper of 0.4274 on
tokens that assert nothing.

## Why this is a retraction and not a repair

UNGROUNDED asserts *this token is a claim whose truth condition was never met*. A hand panel with
the lens IS THIS TOKEN A CLAIM AT ALL, ties resolved toward CLAIM — against the clause — found the
accused tokens are LABELs: they have no truth condition, so neither VERIFIED nor UNGROUNDED is
meaningful and ABSTAIN is the only defensible status. An accusation is itself a claim, and these
four accusations had no receipts. Retracting them is not mercy toward a document; it is the oath
applied to the verifier's own output.

**The retraction predicate, doctrine before gate:** a status may be withdrawn only when what is
shown false is the accusation's PRESUPPOSITION — claimhood — never its verdict, groundedness.
v0.9's G4 protected accusations that are measurements. It never contemplated accusations that fail
to be claims. This cycle does not delete that bar; it replaces it with an enumerated whitelist and
pre-commits that any future retraction runs the full protocol again.

## The change

One clause, flag-gated, `V11_ORDINAL_LABEL`: a status-level demotion to ABSTAIN with the
machine-readable reason `row_ordinal_label`, at the `is_spec` tier — before any obligation or match
is consulted. It fires on a token only when its recorded column lies inside the first cell of a
markdown table data row, the header row's first cell is an exact member of a nine-entry vocabulary
frozen in the prereg text, and the cell is entirely a bare non-negative integer of value at most
100.

Two properties do the load-bearing work. It is **value-blind**: it reads the token's address and
the table's structure, never the token's value beyond that bound and never the match set, so
doctoring a row number does not stop it firing. And it is **never non-extraction**: a fix that
stops accusing by stopping extracting is not a fix, so every silenced token stays countable by
coordinate, in the ledger row and in the certificate's `abstained` array.

The header gate reads the shipped `_TABLE_SEP` and header machinery rather than copying it, so
clause scope and binding-context scope cannot diverge. Making that true required lifting the
table-row walk out of `extract_numbers` into `_table_rows`, which both callers now share; the
refactor moved zero tokens across all 1109 markdown documents under `papers/`.

## Both arms of every gate

| gate | what it bars | observed |
|---|---|---|
| G1 instrument validity | extractor drift, or frame at run ≠ frame at freeze | 0 mismatches over 1109 documents; frame reproduces at 140 documents / 5681 tokens |
| G2 firing-surface exactness | a 12th firing, or a missed roster token | 11 firings, 1 document, 0 over-reach, 0 under-reach |
| G3 retraction ledger audit | any 5th conversion, 2nd flip, 6th silenced verification | 4 withdrawn, 5 silenced, 1 flip, post-frame 4191 / 1490 / 0 |
| G4′a warrant, mechanical | roster status drift at the ship-candidate verifier | reproduces exactly: 5 VERIFIED, 4 UNGROUNDED, 2 ABSTAIN |
| G4′b warrant, second blind adjudicator | a target called CLAIM, or a LABEL in the fresh draw | 21 examined, 11 LABEL, 10 CLAIM, 0 LABELs outside the roster |
| G5 catch decomposition | a fuse, or collateral silencing | override missed 0 at every seed; collateral 0 at every seed |
| G6 exhaustive sweep | the affirmative case failing to reproduce | 117 mutants, 46 / 50 / 21, 0 did-not-land |
| G7 suite closure | a red suite, or an edited certificate | suite green, ruff clean, certificate diff empty |
| G8 mechanism proof | the clause drifting into certified-by-omission | 5681 tokens post-clause; non-extraction would read 5670 |

**The retraction ledger, exactly.** UNGROUNDED→ABSTAIN: the four whitelisted coordinates and
nothing else. VERIFIED→ABSTAIN: the five enumerated collateral coordinates and nothing else, all
five hand-adjudicated as false attestations — a rate coincidence, three index leaves, one unrelated
count. OATH-FAILED→OATH-HELD: one document. OATH-HELD→OATH-FAILED: none. Genuine verifications
destroyed: **0**. Conversions anywhere outside the target document: **0**.

**The positive control and the contrast.** With the clause off, mutating the 11-token roster is
caught a mean of 4.6 times per seed across ten seeds, range 3 to 6, so the tamper channel this
clause operates on is live rather than dead. The measured content of *the header is not optional*
is the broad-class comparison: the rejected positional rule, which silences all 123 sole-token
first cells, destroys a mean of 28.1 reader-visible catches per seed. The frozen clause destroys
none, because every token it reaches sits in a document that was already OATH-FAILED.

## What the gates do NOT show

The prereg demotes five legs on purpose, and this note keeps them demoted. Selling any of them as
a passed bar would launder an identity as a finding.

- **A1** — no conversions outside the target. Vacuous in frame: the frame's four accusations ARE
  the target. Written as construction, never sold as a bar.
- **A2** — full target coverage. Every candidate detector reached it; it carries no ranking power.
- **I1** — with the clause on, tamper catches on this class are zero. That is an identity for a
  structure-reading clause, not a result. Its audit is that the override never misses the mutant,
  and *that* audit is gated.
- **I2** — no certificate flips from held to failed. True by construction of an abstain-only
  clause, and deliberately absent from G3's bar list for exactly that reason.
- **I3** — catch surfacing in the verdict is zero at every seed. Entailed by G2 plus the frozen
  frame: every firing sits in a document already failing, and surfacing counts only the
  held-to-failed transition. It cannot fail while G2 holds. Audited, never counted.

G3 deserves the same honesty. Given G1's frozen statuses and G2's exact roster, every equality in
G3 follows arithmetically. It is an end-to-end implementation audit of the certify pipeline —
gated because an implementation can fail where arithmetic cannot — and it is **not independent
evidence for the retraction**. That evidence lives in G4′ and G5.

## The residual, and the standing price

The class is not closed; it is bounded and counted.

- **The regeneration surface.** Under the census's wider sixteen-entry vocabulary the class is 155
  tokens across 14 documents. That is what the class regrows into if documents are merely repaired
  rather than the verifier fixed — which is why document repair was the red team's preference and
  is only this cycle's recorded fallback.
- **The frozen clause's own surface.** 128 tokens, all under a literal `#`, across 12 documents.
  Only 11 of them sit inside the certified frame; the other 117 are in documents carrying no
  certificate and gain nothing measurable until they do.
- **What stays obligated.** The 27 `rank` tokens the vocabulary excludes remain accusable. They
  were hand-labelled by the red team as ordinal rankings — the same class as a row number — and
  they are excluded anyway, because retracting a class needs its own panel and its own prereg.
  Exclusion is the safe direction: an excluded token stays a disclosed false-accusation surface, an
  admitted one is silenced. This is a debt, and it is named as one.
- **Blind spots.** Among the pipe-delimited table runs under `papers/`, the header machinery binds
  nothing in 37 of them — 27 whose separator opens the run so no header line precedes it, plus 10
  carrying no separator at all. A further 27 runs carry more than one separator, which the shipped
  machinery treats as a single table so later rows keep inheriting the first header. That last one
  is a scope oddity of the shipped machinery, disclosed and not fixed here. Tokens in all of these
  stay obligated, so PROSPECTUS-shaped false accusations remain possible there: a completeness gap,
  not a silencing gap. (The corpus-wide table total is another reflexive count — this note's own
  tables are in it — so it stays in the receipt.)
- **The gameable edge.** An author can hide a doctored small integer as a sole-token first cell
  under a `#` header, or rename a real column to `#`. No header-gated rule can prevent what an
  author writes. The bounds are the sole-content conjunct, the value cap, the reason-coded
  countable abstention, and git-visible headers. A 1..N column longer than 100 rows also flips
  behaviour past the cap, re-manufacturing the accused class on the rows beyond it; no table in
  frame is within a factor of two of tripping that today.
- **A tripwire the ship itself arms.** Catch surfacing is zero only because the sole firing
  document was already failing — and shipping removes that condition. From the moment the clause
  lands, that document is OATH-HELD and carries a `#` column, so the silenced tripwire exists on
  day one, there, not in some hypothetical future document. The tripwire was itself only a
  0.393-recall canary. Both facts are stated together; the loss is priced, not hidden.

## Dogfood: this note's own certificate, graded

The standing rule is that styxx runs its own audit on its own outward claims every cycle. This
note certifies OATH-HELD against its five receipts, with every extracted token verified and none
accused. That is not the interesting part.

The interesting part is what those tokens are sworn TO. Graded by a structural definition frozen
before the count was read — a binding is coincident if its receipt path ends in a bare array
subscript or in an index-like name — **nine of this note's verified bindings are sworn to
positions rather than to measurements.**

The denominator is deliberately left in the receipt rather than printed here, for the same reason
the corpus token total is: a total written inside the document it counts is a token of that
document, it grounds or fails to ground like any other, and in draft every attempt to print it
moved the quantity it reported. The count of coincident bindings is stable under that reflexion;
the ratio is not, so the receipt carries it and this sentence does not. That is not a dodge, it is
the measurement behaving the way the instrument says numbers behave.

Those nine ground at a roster coordinate, a seed number, or a token string
recorded inside the battery's own result — the same channel this cycle just retracted four
accusations over, one level up and pointed at itself. The mechanism is visible in the receipt: the
gate table's own header word *gate* shares a stem with the path segment `gates`, so the v0.3
count-binding filter admits every leaf under `gates[...]`, and a small integer then finds a
position that happens to hold its value.

Three things must be said about that number rather than around it. It is a **floor**, not a
ceiling: the definition is mechanical, it cannot tell a nominally-named leaf that holds the wrong
quantity from a right one, and ties were resolved toward the flattering call. It is **not new** —
it is the v0.8 coincidence channel, closed NEGATIVE after five design families all failed to beat
parity, so no threshold rescues it and this note is not proposing one. And it is **not fixed by
this cycle**: the clause shipped here reaches first-cell table ordinals, and none of these nine is
one.

Reaching that count took iterating: the note is part of the corpus it measures, so writing this
section changed the document, which changed its certificate, which changed the receipt this
section cites. The numbers above are the fixed point, and the loop is disclosed rather than
presented as a single clean read.

The grading receipt is deliberately **not** in this note's own certified receipt set, and the
reason is the point of the whole exercise. That receipt records every extracted token of this
document, including each token's value; admitting it would let every number here verify against a
transcript of itself, and the certificate would become a tautology wearing a receipt's clothes.
The oath is only worth what its truth conditions cost.

## The wobbles, disclosed

Three things in this cycle are weaker than a clean reading would suggest.

**The second panel's pin was corrected after the fact.** The blind adjudication ran while the
working tree still held the pre-change verifier, so the adjudicator hashed that file rather than the
ship candidate. The artifact now records both hashes, records that they do not match, and states
that the OFF-arm statuses it read are shown stable at the ship-candidate verifier by G4′a and *not*
by the adjudication itself. No call, confidence, or reason changed in the correction. The
adjudicator's own caveat is carried rather than summarised away: it did not use a status to decide
a call, but it had the statuses in front of it and cannot assert zero anchoring from having seen
them.

**The fresh draw overlaps the prior panel.** Four of the ten freshly drawn non-PROSPECTUS tokens
had been examined by the first panel. The draw seed was pre-committed in the packet builder before
the draw was inspected, and it is disclosed rather than redrawn — a redraw to force zero overlap
would be selection on an observed value. The adjudicator was blind to the prior calls either way.

**This note moves the frame it was scored against.** Every gate above was scored at the frozen
frame as it stood at freeze. Certifying this note adds a document to that frame, taking it to 141,
and adds this note's own tokens to the corpus total — so re-running the battery after this lands
VOIDs on G1 by design, because G1 exists to detect exactly that. That is the correct behaviour and
not a defect: frame drift is a property of the tree, not of the clause, and a VOID observes no bar,
so re-running is not optional stopping. The successor re-freezes at the new pin and re-derives, as
the previous cycle did when its own note joined the corpus it measured. The token total is
deliberately not quoted here, because quoting it inside the document that changes it would be a
number that falsifies itself on being written.

**One prereg number was ambiguous about nesting.** The prereg's boundary-disclosure list names
"37 headerless, 10 separator-less, 27 multi-separator" as though the first two were disjoint. They
are not: a run with no separator also has no header row. Re-derived here, the 37 decomposes exactly
into 27 plus 10, and the union reproduces the red team's own count. No number in the prereg is
wrong; the list was ambiguous, and it is published here as a decomposition rather than a
correction.

## What the adversarial audit of this battery found

Every gate passed on the first run and every measured number reproduced its pre-committed value
exactly. That is the condition under which a battery is most likely to be flattering rather than
faithful, so the instrument was audited after the fact by four independent lenses — clause
fidelity, gate vacuity, battery fidelity, and doctrine — with every finding then put to two
skeptics instructed to refute it and to default to refuted where the defect could not be
exhibited. A finding survived only if no skeptic could kill it.

18 findings were raised. 2 survived, and both name the same defect.

**G4′b's bars (ii) and (iii) were satisfiable by omission.** *Zero LABELs in the fresh draw* and
*zero CLAIM calls on the seven non-targets* are negative existence tests, and the gate applied
them without ever checking that it had been given a draw or a roster to test. An empty sequence
satisfies both; so does an absent case, whose call reads as nothing at all. A panel artifact
containing only the four retraction targets — no fresh draw, no collateral tokens, an adjudicator
who had examined none of what the prereg sizes — would have cleared the gate and shipped the
retraction.

That is exactly the defect this cycle exists to retract, one level up. G8 has a name for it:
certified-by-omission, the inverse of the oath. The fifth clause of the Retraction Protocol
below has another: silence loud, never omission. The battery that enforces both was violating
them.

A second defect surfaced with it and was then reproduced by a live run: a voiding G1 emitted a
verdict token the reporting path could not parse, and crashed before writing any result — so the
pre-committed battery-void outcome was unreachable on the one path a void-producing gate exists
to take.

Both are repaired. Neither is a moved bar: the prereg sizes both populations exactly, so a gate
that passed on zero of them was never enforcing what was frozen, and a pre-committed outcome that
cannot be written was never really pre-committed. The repaired battery was re-derived at a
reconstructed frozen tree — every gate verdict and every pre-existing field byte-identical to the
first run, with only the two new diagnostic fields added — and the truncated-panel artifacts that
used to pass now return INCOMPLETE, which is neither a pass nor a fail but a refusal to score.

The audit found what the author did not, and it found it by attacking the instrument rather than
the result. That is the only reason it is in this note instead of in someone else's rebuttal.

## The Retraction Protocol — what this cycle establishes as precedent

1. A retraction targets the accusation's presupposition, claimhood, never its verdict.
2. Its evidence is a tripod: a hand adjudication with ties resolved AGAINST the retraction,
   re-checked by a second blind adjudicator at the shipping verifier; a local, idempotent,
   value-blind structural definition; and a catch cost paid only where no reader can see it,
   audited per seed on decomposed columns, with the broad-class contrast carrying the measured
   content.
3. Identities are asserted, never gated. Value-reading catch preservation is disqualified as a
   fuse. Mechanical rankings nominate; humans adjudicate.
4. Retractions ship as frozen, enumerated whitelists inside preregs whose bars never move. The
   whitelist does not extend. The next retraction runs the full protocol again.
5. The silenced class stays countable — status-level abstention with a reason code, never
   non-extraction. Silence loud, never omission.
6. Proof of repair is a shrink-only test transition in the ship commit, with committed
   certificates untouched.
7. Receipt hygiene is part of legitimacy: every receipt in the ship set pins the shipping verifier,
   detector identifiers are cited by filename, and exactly one vocabulary is frozen, in the prereg
   itself.

## What this cycle does not license

It does not license retracting the `rank` class, or any other class, without its own panel and its
own prereg — the whitelist is non-precedential as a mechanism. It does not license editing a
committed certificate, or narrowing a clause after seeing a gate miss. It does not touch the
`=`-operator `is_spec` doctrine, first-occurrence substitution in the prior batteries, or
`corpus_audit`, all of which remain owed their own instrument-repair cycles. And it does not claim
that the certified corpus is now free of false accusations — it claims that the four it was
carrying were not claims at all, that they are withdrawn by an enumerated and audited mechanism,
and that everything the mechanism does not reach is still counted and still named.

---

*The certified corpus's entire accusation surface was false. The instrument that found that out was
the same one being accused.*
