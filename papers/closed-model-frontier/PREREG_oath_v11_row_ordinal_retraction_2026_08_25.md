# PREREG — OATH v0.11: an accusation is itself a claim, and four accusations have no receipts

Fathom Lab · 2026-08-25 · frozen BEFORE any change to `styxx/certify.py`. Bars below never move;
a missed bar ⇒ revert the clause and publish the negative, nothing ships. No optional stopping;
no bar moves after this commit.

Provenance: the certificate-drift census (commit `ff6458f`) found exactly one committed
certificate whose verdict no longer reproduces — `PROSPECTUS_knowsay_2026_07_27` asserts
OATH-HELD and is OATH-FAILED at HEAD, its four UNGROUNDED tokens being markdown table row
ordinals — and closed with: the row-ordinal defect "needs its own frozen prereg, which is in
measurement now." The measurement landed in five receipts (`oath_v10_ordinal_census.json`,
`oath_v10_ordinal_detectors.json`, `oath_v10_ordinal_redteam.json`,
`oath_v10_ordinal_catchcontrol.json`, `oath_v10_panel_isclaim.json`), all re-derived at the
verifier this tree ships (`729b5e6f…`, post `V10_TOKEN_COLUMN`) except the hand panel — whose
receipt pins the v0.9 verifier `1bf81d2a…` and is disclosed as such in §Warrant below. The
receipts self-describe as "OATH v0.10 angle N": they were named before the token_column cycle
claimed the v0.10 number on its own branch. They are cited under their committed filenames and
their numbers are quoted unchanged; the cycle they serve is v0.11. First attempt in this family.

## The defect (measured, pre-fix)

A markdown table's first column is where this corpus writes its row numbers. `extract_numbers`
extracts them like any other token, and on lines whose row text carries trigger vocabulary the
shipped OBLIGATION predicate binds them — so a row number must ground in a receipt leaf or be
accused. A row number has no receipt, because it asserts nothing.

The class, at the shipped verifier over the 140-document / 5,681-token certified frame
(V 4,196 / A 1,481 / U 4; extractor replication 0 mismatches over 1,249 documents):

- **123 sole-token first-cell tokens** (cell content is the number and nothing else):
  110 VERIFIED / 4 UNGROUNDED / 9 ABSTAIN.
- **11 ordinal-shaped tokens** (sole-token AND the column reads exactly 1..N) — all in ONE
  document, all under column header `#`: the claims table of
  `papers/agent-conscience/PROSPECTUS_knowsay_2026_07_27.md`, rows L25–L35.
- The frame's UNGROUNDED population is **exactly these four row ordinals** — L27 `3`, L28 `4`,
  L29 `5`, L32 `8`. The certified corpus's entire standing accusation surface is false.
- The VERIFIED half of the same column is worse than the accused half: L26 `2` is VERIFIED
  against `scale_test_result.json:per_item[2].i` — an index leaf equal to its own subscript.
  The oath is being taken on a coincidence, on a token that makes no claim.
- Exhaustive substitution over the class (117 mutants of the 11 tokens, every significant-digit
  substitution): the shipped verifier answers UNGROUNDED 46 / VERIFIED 50 / ABSTAIN 21 —
  a **0.427 false-attestation rate under tamper** on tokens that assert nothing.

UNGROUNDED asserts "this token is a claim whose truth condition was never met." The hand panel
(`oath_v10_panel_isclaim.json`, lens: IS THIS TOKEN A CLAIM AT ALL?) adjudicated the class with
ties resolved toward CLAIM — against the clause — and found the accused tokens are **LABELs: they
have no truth condition**, so neither VERIFIED nor UNGROUNDED is meaningful and ABSTAIN is the
only defensible status. An accusation is itself a claim, and these four accusations have no
receipts. Retracting them is not mercy toward a document; it is the oath applied to the
verifier's own output.

**The retraction predicate, stated as doctrine before any gate:** a status may be withdrawn only
when what is shown false is the accusation's PRESUPPOSITION (claimhood), never its verdict
(groundedness). v0.9's G4 — zero accusations silenced, zero FAILED→HELD flips — protected
accusations that are measurements. It never contemplated accusations that fail to be claims.
This prereg does not inherit that bar and does not delete it; it replaces it with an enumerated
retraction whitelist (§The named retraction), and pre-commits that any future retraction
requires its own panel and its own prereg — the whitelist is non-precedential as a mechanism.

## The change (exact, frozen)

One clause, flag-gated, `V11_ORDINAL_LABEL = True`: a status-level demotion to ABSTAIN with the
machine-readable reason `row_ordinal_label` recorded in the token's certificate ledger row (the
named-reason mechanism v0.8 established), sitting at the `is_spec` tier of the ladder — before
any obligation or match is consulted. **Never non-extraction**: a fix that stops accusing by
stopping extracting is not a fix, and every silenced token stays countable, by coordinate, in
the certificate ledger (its `abstained` array carries line and token; the ledger row carries
the reason).

The predicate fires on token T iff ALL of:

1. **Position by address, not string.** `V10_TOKEN_COLUMN` is a declared, non-severable
   prerequisite; T's recorded `col` lies inside the FIRST cell of a markdown table data row,
   where "data row" and "header row" are computed by the shipped `_TABLE_SEP` / `header_for`
   machinery in `certify.py` — the clause reads that machinery, it never copies it, so clause
   scope and binding-context scope cannot diverge.
2. **Header gate, exact match.** The header row's first cell — backticks and emphasis stripped,
   whitespace-trimmed, case-folded — is EXACTLY one of the frozen vocabulary written here and
   nowhere else:
   **{ `#`, `#.`, `no.`, `nr`, `idx`, `index`, `row`, `row #`, `№` }**
   This list can only shrink. The three vocabulary variants in the v0.10 receipts disagree with
   each other; that discrepancy is a disclosed defect of the measurement cycle, resolved here by
   freezing a narrower list before data. Named exclusions, each with its measured reason: the
   empty header and `-` (the standard unlabeled-parameter convention; admitting them was "luck
   rather than design" and closing them costs zero — every one of the 11 in-frame firings and
   all 128 corpus-wide `#`-firings carry a literal `#`); `rank` (27 corpus-wide firings, all in two
   uncertified documents, hand-labeled by the red team as "ORDINAL RANKING — a leaderboard
   position; a label, same class as a row number" — excluded NOT because they are claims but
   because retracting a class needs its own panel and prereg per the whitelist doctrine, and
   exclusion is the safe direction: they stay obligated, a disclosed false-accusation surface
   if those documents are ever certified. The adjacent header `rank k` — a different
   population, excluded by exact match regardless — is where the red-team hand case shows
   sweep values grounding in `ranks[j]` with a NON-identity mapping, i.e. genuine claims);
   `n`, `no`, `num`, `id`, `item`, `line`, `claim`,
   `seed`, `k`, `run`, `attempt` (each a live or plausible claim header; `seed` alone is 63 of
   the 150 first-cell tokens in frame, 61 VERIFIED, and silencing it replays the broad-detector
   catastrophe).
3. **Sole-content gate.** The cell, emphasis stripped, is ENTIRELY a bare non-negative integer
   with value ≤ 100. Honest scope: under the frozen 9-entry vocabulary this conjunct's blocking
   set is UNMEASURED — the receipts' 43-token prose shadow ("restricted to the 935 claims",
   "beats semantic_entropy TriviaQA 0.785 band") was measured for the wider D2-vs-D9 comparison
   and consists of empty-header and `id`-header hits, both of which the frozen vocabulary
   already excludes. The conjunct is retained as anti-gaming defense-in-depth against prose
   cells under an in-vocabulary header, the same standing the ≤ 100 cap has: the cap does no
   discriminative work in-frame (the cap sweep shows the header does all of it) and bounds what
   an author could hide under a renamed header. Receipt-bound variance disclosed alongside the
   vocabulary variance: the detectors receipt says `|value| ≤ 100`, the red-team receipt says
   `|value| < 100`; frozen here as non-negative ≤ 100, checked in-frame-equivalent (the largest
   value among the 11 in-frame firings is 11, and no negative first-cell integer exists under
   any variant).
   Edge disclosed: a 1..N column longer than 100 rows flips behavior at row 101,
   re-manufacturing the accused class on rows past the cap. No in-frame table is within a
   factor of two of tripping it today.

**Everything else UNTOUCHED.** `extract_numbers`, `_TRIGGERS`, every shipped `is_spec` clause,
the status ladder order, `V10_TOKEN_COLUMN`, and every committed `*.certificate.json` are
byte-identical. The clause only enlarges the set that reaches ABSTAIN before obligation.

## The designs rejected here, and why

- **Positional (first cell alone, or + small integer).** Against the hand roster, first-cell
  alone falsely silences 115 of the 150 tokens it fires on; adding the small-integer test
  still falsely silences 74 of its 96 firings. 120 honest VERIFIED tokens destroyed in-frame;
  red-team cost ratio 30 honest verifications per target token; 28.1 mean reader-visible
  catches destroyed per seed. The dominant victim is the `seed` column. Decisively dangerous;
  not built.
- **Value-reading (contiguous 1..N run detectors).** `override_missed_mutant` = 22/22 at every
  seed: doctor a row number and the run breaks, the abstention stops firing, and the clause is
  absent on exactly the input it exists to handle — a fuse, not a detector. Also 110 (and 90)
  collateral sibling abstentions per doctored digit. The mechanical ranking's winner is in this
  family, which is why mechanical rankings nominate and humans adjudicate. Not built.
- **Receipt-side index heuristic** (abstain what grounds at an index-like leaf). The panel's
  sample holds 15 subscript-coincidence cases — leaf value equal to its own subscript. One is
  a true positive (PROSPECTUS L26, a LABEL grounding at `per_item[2].i`); the other 14 are ALL
  genuine claims (0-based seed lists, a dose sweep from 0.00). 1 TP / 14 FP in sample, and it
  also consults `hits`, v0.7's lesson: doctor the number and the heuristic stops firing.
  Sized, named, not built.
- **Non-extraction.** Same ledger arithmetic, different auditability: the residual becomes
  invisible. Prohibited by doctrine; the clause must leave a countable trail.
- **Document repair (edit the `#` column out of PROSPECTUS).** The red-team's stated
  preference, and the recorded fallback of this prereg's outcome table — but not its primary,
  because the class regenerates (155 tokens / 14 documents corpus-wide under the census
  vocabulary; 128 of them under `#`, the frozen clause's own surface) and a verifier that asks
  authors to delete row numbers has externalized its false-accusation rate. The two receipts
  genuinely split here; the split is preserved, not papered over.

## The named retraction (the inverted G4)

The clause performs, by design, exactly once, the event v0.9's G4 existed to forbid. The
license is structural, not rhetorical:

1. **Enumeration before data.** Permitted UNGROUNDED→ABSTAIN conversions: exactly the four
   coordinates `PROSPECTUS_knowsay_2026_07_27.md` L27 `3`, L28 `4`, L29 `5`, L32 `8`.
   Permitted certificate flips FAILED→HELD: exactly one, that document. Anywhere else: zero.
2. **Expected collateral, enumerated.** VERIFIED→ABSTAIN: exactly L25 `1`, L26 `2`, L33 `9`,
   L34 `10`, L35 `11` — all five hand-adjudicated as false attestations (a rate coincidence,
   three index leaves, one unrelated count). Genuine verifications destroyed: **0**.
3. **Direction.** HELD→FAILED flips: zero, anywhere, by construction and audited.
4. **Post-clause frame, exact.** V 4,191 / A 1,490 / U 0 on exactly 5,681 extracted tokens.
   The token count moving is mechanism drift (G8); the UNGROUNDED column reaching zero is the
   designed endpoint: the certified corpus's entire accusation surface was false.
5. **Proof-of-repair is a test transition.** `tests/test_certificate_reproduces.py` holds
   `KNOWN_VERDICT_DRIFT = {PROSPECTUS_knowsay}` with the comment "remove this entry when that
   lands." When the clause lands, the live verdict returns to OATH-HELD, the `repaired`
   assertion goes red, and the entry is deleted IN THE SHIP COMMIT — the list can only shrink,
   the committed certificate is never hand-edited, and the drift census's one finding closes as
   a regenerating record, not an edited one.

## Warrant (the panel, and its pin)

The retraction's warrant is `oath_v10_panel_isclaim.json`: 34 tokens examined (30 drawn, 4
forced), 29 CLAIM / 5 LABEL, every LABEL in PROSPECTUS_knowsay, zero LABELs among the 29
non-PROSPECTUS cases, ties resolved toward CLAIM. Its disclosures are carried, not summarized
away: it is a sample of a 150-token roster, one unblinded adjudicator, the 4 forced tokens make
5/34 a non-estimate of the class label rate (among the 30 randomly drawn, exactly 1 is a LABEL),
and two MEDIUM-confidence quotation cases would move the split to 27/7 without adding any
row-ordinal LABEL outside PROSPECTUS.

**Disclosed pin mismatches, both of them:** the panel receipt pins verifier `1bf81d2a…` (v0.9);
the four measurement receipts pin `729b5e6f…` (shipped). What the receipts actually evidence at
each pin: the 5 panel-examined PROSPECTUS tokens carry identical statuses at both verifiers
(1 LABEL/VERIFIED + 4 LABEL/UNGROUNDED); the full 11-token roster is V 5 / U 4 / A 2 at the
shipped verifier only — no receipt records the other six statuses at the v0.9 pin. Second
wobble: older prose strings inside the receipts self-describe a "139-document frame"; the
frozen frame is the census frame dict's 140 documents / 5,681 tokens, and G1's VOID trigger
reads those dict values, not the prose. The panel adjudicates MEANING, which window anchoring
does not change — but a retraction whose warrant pins a retired verifier hands any auditor a
free objection, so the warrant is re-established at the ship-candidate verifier by G4′, whose
two legs are specified below with an honest account of what "re-derived" means for a human
adjudication.

## Battery + gates (harness `run_oath_v11_battery.py`, to be built; seeds 1–10, frozen)

Every gate runs two-armed (flag OFF / flag ON) at the ship-candidate verifier, on the frame as
frozen below. Mutation operator imported from the v0.9 battery, never copied. Non-destructive:
mutants in temp files, the only file written is the battery's own result JSON.

- **G1 — INSTRUMENT VALIDITY (VOID-producing).** Extractor replication mismatches = 0 over all
  documents; frame at run = frame at freeze: 140 documents / 5,681 tokens / V 4,196 / A 1,481 /
  U 4. Any drift → the battery is **VOID**, not failed: frame drift is a property of the tree,
  not the clause. Re-freeze and re-run; a VOID observes no bar, so re-running is not optional
  stopping.
- **G2 — FIRING-SURFACE EXACTNESS (gated, both directions).** Clause ON fires on exactly the
  11-coordinate roster above — 11 tokens, 1 document, every header `#`. A 12th firing anywhere
  in frame is over-reach → FAIL. A missed roster token is under-reach → FAIL.
- **G3 — THE RETRACTION LEDGER AUDIT (gated, exact — and named for what it is).** Whole-frame
  re-certification, ON vs OFF: UNGROUNDED→ABSTAIN conversions = exactly the 4-token whitelist;
  FAILED→HELD = exactly {PROSPECTUS_knowsay}; VERIFIED→ABSTAIN = exactly the 5-token roster;
  post-clause frame = V 4,191 / A 1,490 / U 0. Any 5th conversion, 2nd flip, 6th silenced
  VERIFIED, or any count off by one → FAIL. Stated honestly: given G1's frozen statuses and
  G2's exact roster, every equality here follows arithmetically — G3 is an end-to-end
  implementation audit of the certify pipeline, gated because an implementation can fail where
  arithmetic cannot, and it is not independent evidence for the retraction (the evidence lives
  in G4′ and G5). Two legs are demoted outright: the in-frame "0 conversions elsewhere" zero is
  true by construction (A1 — the frame's four UNGROUNDED ARE the target), and HELD→FAILED = 0
  is I2's audited identity; neither appears in this gate's bar list.
- **G4′ — WARRANT, two legs.** The retraction's license, re-established at the ship-candidate
  verifier. Honest statement of what "re-derived" means for a hand panel: a human re-reads and
  could rubber-stamp, so the human leg is specified to be failable or it would not gate.
  - *G4′a (mechanical, in the battery, gated):* at the ship-candidate verifier, OFF arm, the
    11-token roster's statuses reproduce exactly V 5 / U 4 / A 2 at the frozen coordinates.
    A verifier change can move these; drift → FAIL.
  - *G4′b (human, out of battery, gated).* A SECOND adjudicator — a different panel lens
    agent, not the author of `oath_v10_panel_isclaim.json` — re-adjudicates the full 11-token
    roster plus a fresh draw of 10 non-PROSPECTUS tokens from the 150-token class roster,
    reading OFF-arm statuses only, blind to the prior panel's calls and to the clause's reason
    codes, under the same frozen CLAIM/LABEL definition with ties resolved toward CLAIM.
    Artifact: `oath_v11_panel_recheck.json`, pinned to the ship-candidate verifier sha. The
    full roster is included deliberately: the adjudicator cannot tell targets from collateral,
    which is the blinding. Bars, all reachable by a fresh adjudicator: (i) all four targets
    adjudicate LABEL — a CLAIM call → FAIL, the accusations stand as correct, the drift entry
    stays, and document repair becomes forbidden (it would quiet a live accusation); (ii) zero
    LABELs among the fresh non-PROSPECTUS draw — a LABEL → FAIL, the class is
    under-enumerated; (iii) zero CLAIM calls on the seven non-target roster tokens — a CLAIM
    call there → FAIL, the clause would silence a contested-genuine verification and the
    "genuine verifications destroyed: 0" leg no longer stands on an uncontested adjudication.
- **G5 — CATCH DECOMPOSITION (gated on the decomposed columns; ten seeds).**
  Raw ON-caught is NOT a gate: for a structure-reading clause ON = 0 is an identity, and
  reporting it as a tamper result would launder an identity as a finding (I1). Likewise
  `catch_surfacing_in_verdict` = 0 is NOT a gate: it cannot fail while G2 holds — every
  in-frame firing sits in PROSPECTUS_knowsay, which is OATH-FAILED at the OFF baseline and
  stays FAILED under any single mutation (at least three of its four accusations remain), and
  surfacing counts only HELD→FAILED transitions. It is demoted to I3 and audited. The gates:
  - *Positive control:* OFF-arm catches ≥ 1 at every seed (measured 4.6, range [3, 6]).
    A zero seed voids that seed; fewer than 8 valid seeds voids the gate (outcome table:
    battery VOID).
  - *Identity audit:* `override_missed_mutant` = 0 at every seed. Nonzero → FAIL: the clause
    has become value-reading — a fuse.
  - *Collateral:* abstentions lost elsewhere under mutation = 0 at every seed. Nonzero → FAIL.
  - *Comparison arm (severable):* the broad 123-token class re-derives surfacing mean ≈ 28.1
    in [22, 36] — the measured content of "the header is not optional." Irreproducibility
    VOIDs this comparison claim only, never the gated legs above.
- **G6 — EXHAUSTIVE SWEEP REPRODUCTION (gated).** All 117 single-significant-digit mutants of
  the 11 tokens, OFF arm: UNGROUNDED 46 / VERIFIED 50 / ABSTAIN 21, `did_not_land` = 0
  (nonzero → VOID: the operator broke, not the clause). Drift in these counts at the frozen
  frame → FAIL: the affirmative case (the 0.427 false-attestation rate) must reproduce.
- **G7 — SUITE CLOSURE (gated).** In the ship commit: `KNOWN_VERDICT_DRIFT` shrinks to empty;
  `python -m pytest tests -q` green; `ruff` clean over `styxx/`; `git diff` over every
  committed `*.certificate.json` is EMPTY — the PROSPECTUS certificate returns to reproducing
  OATH-HELD with no hand edit. Red suite or nonempty certificate diff → nothing lands.
- **G8 — MECHANISM PROOF (gated).** Post-clause extracted token count = exactly 5,681
  (non-extraction would read 5,670); the 11 tokens appear by coordinate, status ABSTAIN with
  reason `row_ordinal_label` in the ledger rows, in the battery result's ON-arm
  re-certification of PROSPECTUS. Artifact home stated plainly: the battery is
  non-destructive, G7 forbids touching committed certificates, so the countable trail of
  record this cycle is the battery result JSON committed with the RESULT — the committed
  PROSPECTUS certificate stays byte-identical (and stale in its counts) until a future
  re-certification cycle regenerates it. Any shortfall in the token count → FAIL: the clause
  drifted into certified-by-omission, the inverse of the oath.
- **G9 — BOUNDARY DISCLOSURE (gated on presence).** The RESULT publishes, with re-derived
  numbers, both surfaces kept apart: the census-vocabulary class (155 tokens / 14 documents,
  the regeneration surface) and the frozen clause's own firing surface (128 tokens under `#`,
  of which 117 sit in uncertified documents that gain nothing measurable today; the 27
  excluded `rank` tokens stay obligated); the 43-token prose shadow measured by the receipts'
  D2-vs-D9 comparison under the wider vocabulary (the frozen conjunct's own blocking set is
  unmeasured and said so); the blind spots (37 headerless, 10 separator-less, 27
  multi-separator tables — the last also a scope oddity of the shipped header machinery,
  disclosed not fixed); and the per-document ordinal-abstain count, which every future
  certification of a residual document must report. Absent or stale → the RESULT is MALFORMED
  and nothing ships.

## Asserted invariants — NOT gated

- **A1** — 0 UNGROUNDED converted outside the target: vacuous in-frame; the frame's four
  UNGROUNDED are the target. Written as construction, never sold as a passed bar.
- **A2** — 4/4 target coverage: every candidate detector reached it; carries no ranking power.
- **I1** — ON-arm caught = 0 for a structure-reading clause: identity. Its audit is
  `override_missed_mutant` = 0, and that audit IS a gate (G5).
- **I2** — HELD→FAILED flips = 0: by construction of an abstain-only clause; audited by the
  whole-frame re-certification and reported as an audit. Deliberately absent from G3's bar
  list for exactly that reason.
- **I3** — `catch_surfacing_in_verdict` = 0 at every seed for the clause ON: entailed by G2
  plus the frozen frame (all firings in one already-FAILED document; surfacing counts only
  HELD→FAILED transitions), so it cannot fail while G2 holds. Audited and reported per seed,
  never counted as a passed bar. Its evidentiary twin is the broad-class comparison (28.1
  mean reader-visible catches destroyed per seed) — that contrast, not the zero, is the
  measured content of the catch leg.

## Outcome table (pre-committed)

- G1 void → **`V11_BATTERY_VOID`**; re-freeze at the new frame pin, re-run. No verdict.
- G2 over-fire → **`V11_OVERREACH`**; revert, publish. G2 under-fire → **`V11_UNDERREACH`**;
  revert, publish. Fallback for both: document repair (red-team option b) as its own cycle.
- G3 miss → **`V11_RETRACTION_MISCOUNT`**; revert, publish; the drift entry stays;
  PROSPECTUS stays OATH-FAILED.
- G4′a drift → **`V11_WARRANT_FAILED`**; revert, publish; re-measurement owed before any
  successor. G4′b, a target adjudicates CLAIM → **`V11_WARRANT_FAILED`**; revert, publish;
  the accusations stand as correct and document repair is thereafter forbidden. G4′b, a LABEL
  in the fresh non-PROSPECTUS draw → **`V11_CLASS_UNDERENUMERATED`**; revert, publish; the
  four targets' falsity is untouched, but the whitelist was too small — the wider class gets
  its own panel and prereg, and nothing retracts this cycle.
- G5 positive control void (fewer than 8 valid seeds) → **`V11_BATTERY_VOID`**; re-run rule as
  G1's. G5 override > 0 → **`V11_FUSE`**; revert, publish. G5 collateral > 0 →
  **`V11_COLLATERAL`**; revert, publish: the clause reaches tokens it never enumerated.
- G6 drift → **`V11_SWEEP_DRIFT`** (FAIL) or **`V11_BATTERY_VOID`** (operator); the receipt
  must say which.
- G7 red → **`V11_SUITE_RED`**; nothing lands. G8 miss → **`V11_MECHANISM_DRIFT`**; revert.
  G4′b non-target CLAIM call → **`V11_COLLATERAL_CONTESTED`**; revert, publish.
  G9 absent or stale → RESULT MALFORMED; nothing lands.
- All gates pass → **`V11_ORDINAL_RETRACTION_SHIPS`**. `V11_ORDINAL_LABEL` ships True; the
  RESULT publishes both arms of every gate, the retraction ledger, the residual, and the
  Retraction Protocol (below) as precedent.
- No second attempt inside this cycle. The clause is atomic: no post-freeze narrowing to
  "the roster minus the offending token" — partial salvage after observing a miss is optional
  stopping wearing a lab coat. Severable is only the comparison claim named in G5.

## The Retraction Protocol (what this cycle establishes, if it ships)

1. A retraction targets the accusation's presupposition — claimhood — never its verdict.
2. Its evidence is a tripod: a hand adjudication with ties resolved AGAINST the retraction,
   re-checked by a second blind adjudicator at the shipping verifier; a local, idempotent,
   value-blind structural definition; and a catch cost paid only where no reader can see it —
   audited per seed on decomposed columns, with the broad-class contrast (28.1 reader-visible
   catches destroyed per seed for the rejected rule vs an entailed-and-audited zero for this
   one) carrying the measured content.
3. Identities are asserted, never gated; value-reading catch preservation is disqualified as a
   fuse; mechanical rankings nominate, humans adjudicate.
4. Retractions ship as frozen, enumerated whitelists inside preregs whose bars never move; the
   whitelist does not extend; the next retraction runs the full protocol again.
5. The silenced class stays countable — status-level ABSTAIN with a reason code, never
   non-extraction. Silence loud, never omission.
6. Proof-of-repair is a shrink-only test transition in the ship commit, with committed
   certificates untouched.
7. Receipt hygiene is part of legitimacy: every receipt in the ship set pins the shipping
   verifier (hence G4′), detector identifiers are cited by filename (the two receipt namespaces
   here collide on "D5"), and exactly one vocabulary is frozen, in the prereg itself.

## Disclosures (the standing price, named so it is paid knowingly)

- An author can hide a doctored small integer as a sole-token first cell under a `#` header, or
  rename a real column to `#`. No header-gated rule can prevent what the author writes; the
  bounds are the sole-content conjunct, the ≤ 100 cap, the reason-coded countable abstention
  (growth in the class is a visible number at every certification), and git-visible headers.
- `catch_surfacing_in_verdict` = 0 is conditional on the OFF-arm baseline, where the only
  firing document is already FAILED — and the ship itself removes that condition: the moment
  the clause lands, PROSPECTUS is an OATH-HELD document carrying a `#` column, so the silenced
  tripwire exists on day one, there, not in some hypothetical future document. The tripwire
  was itself only a 0.393-recall canary. Both facts are stated together, and the loss is
  priced, not hidden.
- The clause reaches nothing in headerless or separator-less tables (safe direction: those
  tokens stay obligated, so PROSPECTUS-shaped false accusations there remain possible — a
  completeness gap, not a silencing gap).
- The frozen clause's corpus-wide surface is 128 tokens under `#`, of which 117 sit in
  uncertified documents and gain nothing measurable until those documents carry certificates.
  (155 / 14 is the census-vocabulary class — the regeneration surface — which includes the 27
  excluded `rank` tokens that stay obligated.)

Out of scope (named so they cannot creep in): the `=`-operator `is_spec` doctrine
(`V10_EQUALS_SPEC_OVERREACH`'s successor, owed its own battery); first-occurrence substitution
in the v0.6.1/v0.7/v0.9 batteries and `corpus_audit` (owed a dedicated instrument-repair cycle
now that `col` ships); any edit to an already-committed `*.certificate.json`; any change to
`validate_oath_v0.py` or prior batteries — bars and instruments never move; any version bump,
tag, or release, which the operator owns; and re-adjudication of the 116 unexamined roster
tokens beyond what G4′ requires — the panel is a sample and is priced as one.

---

*Frozen on commit. The bar structure outranks the retraction.*
