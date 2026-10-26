# AMENDMENT — the count rule: a numeral inside a verified quotation of the gate's own trailer is left out of the count, and every other form of the trailer is counted

Fathom Lab · 2026-10-25 · Amends `PREREG_darkflobi_gate_2026_10_23.md` and `RULES_darkflobi_gate_2026_10_23.md`, in this
directory, merged as #229 with section 15 closed at 15.13 and not edited. Section 12 of the PREREG (its closing paragraph)
provides for this file: after the merge a change to a rule is a new AMENDMENT file under this folder, lock-hashed in its turn,
with deliveries under the old rule counted under the old rule. Written by the build agent for piece (1) of the before-D0
work, on the lead's pass-2 plan (`BD:PLAN.md`, sha256 `de3515f3…`, section 4), before D0 and before `W:gate/REQUIRED` exists.
This text replaces, in place and under the same name, the draft committed on this branch at `554122a1`. That draft's rule
(every form of the trailer left out, with a clause for a form standing where the gate's trailer stands) was defeated by a
review (`BD:REVIEW_rule.md` §3-4, attacks A1-A5); the file is unmerged, so the draft is no frozen text, and its bytes stay in
the branch's history. The operator has not signed this file: "what is the operator's and the lead's" lists what is theirs.

Receipt key: `W:` = `C:/Users/heyzo/clawd/`; `WT:` = `W:wt/`; `BD:` = `WT:_darkflobi_gate_2026_10_22/before_d0/`; `M:` = the
styxx checkout at the pin `3437340b`; `seq N` = a line of `W:gate/ledger.jsonl`; `[R]` a file or a run shows it; `[J]`
judgement. The live gate `W:gate` was read and never written: every change below was built and tested on the copy
`BD:gate_copy/`, whose `gate.py`, `gate_verify.py` and `README.md` were put back to the live bytes before the build, and the
lead moves it into place at apply time. The copy is shared with piece (4) (the prepared-record guard), whose hunks sit in the
same two files (`BD:COORD_piece1_shared_files.txt`, `BD:COORD_piece4_shared_files.txt`); every run from 2026-10-26T00:37Z on
is against the copy carrying both.

## What the PREREG and the RULES say

- `RULES` §5: "`Nn` on a line is the count of numerals in the checked text (`numerals.total`): certify's ledger rows plus
  its counted uncovered band (…), read whenever the text holds a digit". `Nb` is bound by position (a VERIFIED certify row,
  or an opening character inside a sworn HELD numeric span); nothing is added to or retyped from certify's extraction. §3.1
  shape S14: pull-request numbers, thread ordinals and line cites are numerals.
- `RULES` §7: with `Nu` > 0 and nothing contradicted, Telegram sends "with the trailer `[gate: Nb of Nn numerals bound ·
  ledger <seq>]` after a blank line"; "everything bound, or `Nn` = 0, lint clean" sends with no trailer; "The gate never
  edits inside a text." `gate.py` writes the trailer from `TRAILER` through `_trailer_template`, the seq filled under the
  lock by `fields_for`, and puts `unbound_marked` among the line's reasons exactly when it appends that trailer.
- `RULES` §12: the census strips the four fixed end forms, the trailer's among them, "only at the end of the text and after
  a blank line", from both sides alike (`reconcile.py` `TRAILER_END`, `END_FORMS`).
- `RULES` §13: a `send` line's `numerals` carries `total`, `bound`, `unbound`, `uncovered`, `unbound_tokens`.
- PREREG 15.12, signed in 15.13: component (iv) of §5 is read per `gate_sha256` stretch, each send line against the digest
  of the files locked when it was written.
- The shadow log (`WT:_darkflobi_gate_2026_10_22/shadow/SHADOW_LOG_2026_10_24.md`), finding 2: when darkflobi quotes the
  gate's own trailer "the trailer's numbers are read as new unbound numerals and the post is marked again"; the candidate
  repair is an AMENDMENT before D0, never an edit of frozen text. Finding 3: version strings, "Not repaired by default".

## What was observed

1. **The ledger at seq 116** [R, `BD:ledger_forms_scan_pass2.txt`, a counts-only scan of `BD:verify_scratch/home/`, whose
   `ledger.jsonl` (sha256 `baf3f6924b388fff8132cb4315c9f507cd3d15b97c6db1dbe86cd06338431381`), `head.txt` and `sources/`
   equal `W:gate`'s by digest at 2026-10-26T00:20Z]: 116 chained lines after the header, head `76de868c…`, 57 `send`
   lines, shadow throughout; 30 public, carrying 13 numerals, 0 bound.
2. **The five marked public lines, all of 2026-10-24**, each token labelled by the product red team's split
   (`WT:_product_2026_10_24/PRICING.md` §5.1, `RED_TEAM.md` M5), re-made from the lines' own `unbound_tokens` and the plan's
   reading of each token's place (`BD:PLAN.md` §1.2): **T** inside an exact copy of the gate's trailer; **V** a version;
   **S** a seq of the gate's own records written in prose; **C** a self count; **N** a number with no receipt.

   | seq | exit | `checked_sha256` | `Nn` / `Nb` / `Nu` | `unbound_tokens` | by token |
   |---|---|---|---|---|---|
   | 26 | E1 | `9740b3a5…` | 2 / 0 / 2 | `3.1`, `4.8` | V V ("gemini 3.1 pro preview", "claude opus 4.8": a space before each number) |
   | 30 | E3 | `e7ff8ff4…` | 1 / 0 / 1 | `0026` | S ("the ledger (0026)") |
   | 34 | E3 | `4fc9736d…` | 4 / 0 / 4 | `0026`, `3.1`, `4.8`, `2` | S, V V (quoted, no model name), C |
   | 44 | E1 | `f98ec408…` | 2 / 0 / 2 | `64`, `026,964` | N N (a patent application number) |
   | 46 | E1 | `d178cbc6…` | 4 / 0 / 4 | `0`, `2`, `0044`, `0044` | T T T, S |

   T 3, V 4, S 3, C 1, N 2: 13. The shadow log's "6 no-receipt" is S + C + N under this split.
3. **The one form on the ledger** [R, `BD:ledger_forms_scan_pass2.txt`]: of the 57 send lines, one holds the trailer's form
   in its checked text: seq 46, one form, and it is byte-identical to the trailer the gate appended to seq 44
   (`TRAILER % (0, 2, 44)` = `[gate: 0 of 2 numerals bound · ledger 0044]`; seq 44 carries `unbound_marked`, 0 of 2). The
   lines that carried the trailer: 26, 30, 34, 44, 46 (public) and 105, 107, 109, 113, 115 (the operator's DM).
4. **Versions, as certify reads them** [R, `tests/test_trailer_rule.py`
   `test_versions_certify_does_not_count_as_signed_are_pinned_here`, run against the pinned snapshot; the plan's probe,
   `BD:PLAN.md` §4.3 item 2]: not counted, by certify's own scrubs as signed in RULES §5 (`_VERSIONISH`, `M:styxx/certify.py`
   l.55; `_drop_reason`'s compound-identifier rule, l.138-139): `claude-opus-4-8`, `gemini-3.1` (mid-sentence and before a
   period), `gemini-3.1-pro-preview`, `2.1.310`, `v7.50.0`. Counted: `claude opus 4.8` (the `4.8`), a quoted `"3.1"`,
   `version 3.1`. So the examples `2.1.310` and `gemini-3.1` are not what marked day 0; a two-part number after a space or
   in quotes did, and in bytes that is the shape of a measured share (`0.23`).
5. **The defeat of the earlier draft** [R, `BD:REVIEW_rule.md` §3-4, run against that draft's copy]: with every form left
   out, a form the writer composes (A1 `[gate: 9 of 9 numerals bound · ledger 0001]` opening a text; A2 a seq that exists
   nowhere; A5 revenue figures written inside the form) left clean with no trailer of the gate's, a route the live gate
   does not open; the own-position clause fell to a period (A4) or a zero-width space (A3) after the form.

## The amendment

1. **The rule.** The trailer's form is `\[gate: ([0-9]+) of ([0-9]+) numerals bound \u00b7 ledger ([0-9]+)\]` (the words,
   the middle dot U+00B7, the brackets; A, B and S runs of ASCII digits), matched anywhere in the checked text. A match is
   a **verified quotation** only when it is byte-identical to the trailer the gate appended to its own send line S: a
   `kind: send` line earlier in the chain with seq S, `action: send` and `unbound_marked` among its reasons, whose
   `TRAILER % (numerals.bound, numerals.total, seq)` equals the match. A numeral certify reads wholly inside a verified
   quotation (its token stands, at certify's own line and column in characters, inside the match) is left out of the
   count: it is neither bound nor unbound, enters neither `Nn` nor `Nu`, and is not in `unbound_tokens`. Every other match
   is text, and its digits are counted like any other. In `gate.py`: `TRAILER_FORM_RX` beside `TRAILER`;
   `_trailer_of_line(e)` (the trailer a line carried, else none); `trailer_quotes(text, lines_by_seq)` (each match read
   against the line it names; a pure function of the text and the lines); `bind_numerals(…, quotes)` (drops the verified
   items before anything is counted; an item whose token does not stand at its place is never dropped); in `check`, the
   lookup made only when a match is present.
2. **Where the lines come from.** In the live check, the gate's own ledger, read as it reads its latest anchor line
   (`_send_lines_by_seq`, beside `_latest_anchor_line`); every line there is earlier than the one being written, whose seq
   is drawn later under the lock. A read that raises is a fault inside the legs, `gate_unavailable:<the exception's class>`,
   as any other. A chain that does not re-derive takes no new line in any case (the ledger's own check before a write). In a
   replay, the chain's send lines before the replayed seq, handed in by `gate_verify.py` (item 11), never the line's own
   record of what it verified.
3. **Why a verified quotation, and not every form** [J]. (a) "No rule here reads what a sentence says" (RULES §14), so
   the gate cannot tell a quotation from a fabrication by what the sentence around it claims; but a quoted mark is a claim
   about a ledger line, and the gate binds it to that line, which is its whole method. (b) Every digit a verified
   quotation carries is fixed by the line it names, so no figure of the writer's can be hidden in it (A5). (c) The gate appends a trailer only when `Nb` < `Nn` (`decide`), so no verified
   quotation can read "n of n": it can show an earlier mark, never a clean verdict (A1). (d) A composed form is counted, its
   digits read unbound and the gate's own trailer is appended after it, as today (A1-A5 all draw the real trailer). With
   this the own-position clause is not needed: a verified quotation at the end of a message is a true statement naming
   another seq, and the gate's own trailer, when one is due, still comes last after a blank line.
4. **Rejected: every form left out, and the own-position clause.** `BD:REVIEW_rule.md` A1-A5 (observed, item 5).
5. **Rejected: a version grammar.** Certify's extraction as signed already leaves out three-part versions and identifiers
   glued by a hyphen (observed, item 4), and the gate does not retype certify. What remains counted, a two-part number after
   a space or in quotes, is in bytes the shape of a measured value; the only grammar that tells `opus 4.8` from
   `precision 0.23` is a list of product names, which is a content exception, and RED_TEAM M5 and the shadow log's
   finding 3 decline it too.
6. **Rejected: a ledger seq in prose.** "the ledger (0026)" is the agent's own number about its own records (and wrong: a
   seq is not a tally). A rule on `ledger <digits>` would also leave out "ledger 258 entries", a committed-count claim
   (RULES §3.1 S9), and would make the agent's commentary on its seqs uncountable. The marks on seq 30, 34 and seq 46's
   prose `0044` are the gate doing what it is for.
7. **What the rule does not exclude**, each with a test in `W:gate/tests/test_trailer_rule.py` [R]: a form the writer
   composes, opening or closing a text, or as the whole text (`test_review_rule_A1…`, `…A5…`,
   `test_a_composed_form_closing_the_text_counts_and_the_gates_trailer_follows`); a form followed by a period or a
   zero-width space (`…A3_A4…`); a form naming a seq that exists nowhere (`…A2…`), the line's own seq or a later one
   (`test_a_form_naming_the_lines_own_seq_or_a_later_one_counts`), an `allow` line, the `started` line, an `anchor` line or a
   `fault` line (`test_a_form_naming_an_allow_line_counts`, `test_a_form_naming_the_started_line_an_anchor_line_or_a_fault_line_counts`),
   a line refused in shadow, a DM line marked with codes, or an X line refused `unbound_on_x`
   (`test_a_form_naming_a_line_refused_in_shadow_or_marked_with_codes_counts`); a form whose A, B or zero padding differ from
   that line's trailer, `ledger 1` for `ledger 0001` among them (`test_a_form_whose_counts_or_padding_differ…`); a seq longer
   than eighteen digits, never converted (`test_a_seq_too_long…`); every near-form of a real trailer: a hyphen, an en dash
   or a plain space for the middle dot, a no-break space or a zero-width space inside, "numeral" singular, the form split
   over two lines, its brackets escaped as markdown, a doubled space, a capital letter (`TestNearFormsCount`, ten shapes),
   and Arabic-Indic or fullwidth digits (`test_the_form_matches_what_the_gate_writes_and_nothing_wider`); a ledger seq in
   prose, including one naming a marked line, "ledger 0001", "the ledger (0001)", "ledger line 1"
   (`test_a_ledger_seq_in_prose_counts_even_when_it_names_a_marked_line`); a self count, a two-part version, a patent
   number (the seq 26, 34 and 44 shapes); the operator-DM mark `[gate: <codes>]`, whose codes can carry a numeral, and
   `[unchecked: styxx unavailable]` (`test_the_dm_marks_codes_with_a_numeral_count_and_the_lint_fires`; the test composes
   the one `plan/` rule id at run time, so no file spells it). A text with a verified quotation and a composed form counts
   the composed one alone (`test_a_verified_and_a_composed_form_side_by_side`). The rule reads bytes and the gate's own
   ledger, never what a sentence says, and it is the same at every exit and destination class: the operator's DM takes the
   same count and its own trailer quotes like any (`test_the_operator_dm_takes_the_same_count_and_its_own_trailer_quotes_like_any`),
   E3 the same, and X refuses on the prose numeral alone (`test_e3_takes_the_same_count_and_x_refuses_on_the_prose_numeral_alone`).
8. **What a writer can rely on.** To point at a mark, quote it whole, exactly as the gate appended it to the line it names,
   and its numbers are not counted; anything else that looks like the mark is counted. A seq in prose is a number: print
   it bound, or not at all. Write a model by its identifier with the hyphen (`claude-opus-4-8`,
   `gemini-3.1-pro-preview`) or a version with three parts (`7.50.0`): certify does not count those, as signed; `opus 4.8`
   and a quoted `"3.1"` are numerals.
9. **RULES §5 and §13 restated.** `Nn`, `Nb` and `Nu` are read over the numerals that remain after the verified
   quotations' are left out. A `send` line's `numerals` carries four more keys: `trailer_quotes` (the verified quotations),
   `trailer_quote_numerals` (the numerals left out), `trailer_quote_seqs` (the seqs they name, at most 20) and
   `trailer_forms_unverified` (the forms counted as text); `total`, `bound`, `unbound`, `uncovered` and `unbound_tokens`
   keep their meaning over what remains. A line `gate.py fault` writes, an `internal` line and a line whose legs did not
   finish carry the four at zero (`TestTheLineShape`). The verdict `gate.py check` prints keeps its three counts; the plugin
   reads nothing new.
10. **No new reason code, no change to the matrix.** RULES §6's codes, §7's matrix, `decide`, `_trailer_template` and the
    trailer's place are unchanged. The census is unchanged: a verified quotation that ends a text after a blank line is
    stripped by RULES §12 (b) from both sides alike, so the match it reads is the same.
11. **Replay.** `gate_verify.py` hands the replay the chain's send lines before the replayed seq (`_send_lines_before`, as it
    hands it the anchor line before it), and compares the four keys only on a line that carries them, so a line written
    before the move re-derives on the five old keys. Consequence, stated before any window line exists [R]: under the gate
    of its own stretch (`BD:gate_before/`, `gate.py` `25d53fd6…`, `gate_verify.py` `f4b19c94…`) every one of the 57 send
    lines re-derives SAME (`BD:verify_scratch/report_p2_before_25d53fd6aa5e7604.json`, sha256 `f7e6d32d…`: chain re-derives,
    SAME 57, exit 0); under the amended pair, 56 SAME and one DRIFT, seq 46, diffs `sent_sha256`, `numerals.total`,
    `numerals.unbound`, `numerals.unbound_tokens`, no module skew, exit 1
    (`BD:verify_scratch/report_p2_after_4416692cfc983dbb.json`, sha256 `40ef6fcc…`). Replayed alone under the amended pair
    (`BD:verify_scratch/seq44_46_replay_p2.txt`), seq 46 reads `total` 1, `unbound_tokens` [`0044`], `trailer_quotes` 1,
    `trailer_quote_numerals` 3, `trailer_quote_seqs` [44], `allow_marked`, its trailer `0 of 1`; seq 44 reads SAME. Seq 46 is
    counted under the old rule (PREREG §12): on the ledger it reads `0 of 4`, delivered with that trailer, and the RESULT
    lists it as the one line whose count the amended rule reads otherwise. One reading the replay can differ on, fixed
    here before any window line exists: a form naming a line appended by another process after the live check read the
    ledger and before its own line was written is counted live and verified in the replay, and reads DRIFT there; to compose
    it, the writer must reproduce the bytes of a trailer not yet written when the check began.
12. **The fourth `gate_sha256` stretch.** `gate.py` and `gate_verify.py` both enter `gate_sha256` (`Env.gate_sha256`: the
    folder's `*.py`, `helpers/*.py` and the installed plugin's files). They move in one process with piece (4)'s guard and
    the canary piece's `canary_plan.py`, and the stretch's digest and the seq at which it begins are named once, in the
    lock-rows AMENDMENT's addendum, re-derived from the box at apply. For the record, a candidate: by the rule of 15.6,
    without importing `gate.py` (`BD:apply/rederive_gate_sha256.py`), over the copy as it stood at 2026-10-26T00:47Z (both
    pieces' hunks in `gate.py` and `gate_verify.py`, the canary piece's `canary_plan.py`) and the six installed plugin
    files as recorded at `BD:_probe/plugin_installed/`, 19 files:
    `c5084df2ef574658fe84f84feea0b70a1169eb695a882f1d4237b439d3a418e8`. Any later edit of a `*.py` file in the copy moves
    it; the stretch's digest of record is the addendum's.

## Effect on day 0, said plainly

Under this rule 3 of the 13 public unbound numerals of 2026-10-24 would have been left out: seq 46's `0`, `2`, `0044`, the T
tokens inside its verified quotation of seq 44's trailer. Seq 46 would still be marked, `0 of 1`, for its prose `0044`; seq
26, 30, 34 and 44 are unchanged; **0 of the 5 marked posts clear**. The one post the lab's product notes count as cleared
by the planned repairs is seq 26, and RED_TEAM M5 says it clears only if the two model phrases are listed exactly: that is a
content exception, which this file does not write. The repair is a rule
about the gate's own bytes, not a repair of those posts, and PREREG §12 counts them under the old rule in any case. The
census class the shadow log asked for, marks whose only unbound tokens sat inside a quoted trailer, is empty on day 0 (seq 46
has a prose seq beside its quotation).

## What does not change

The pin, the snapshot and every instrument; certify's `_NUM` and its scrubs, which the gate still does not retype; RULES §4's
grammar and the drop rule as signed; §6's codes and §7's matrix; the trailer the gate writes and where it writes it; the
census's end forms (RULES §12); the verdict's shape; the plugin and its installed copies, which relay `content` as before
(the plugin harness, 79 of 79 against the amended copy, "the tests" below); `selector.py`, whose count of digit-bearing
tokens is the selector's own (the pre-gate cell's "numerals per post" is the selector's alone, `W:gate/pregate_count.py`
l.145) and still counts a quotation's digits, in every cell alike. Lines written before the move are read under the old rule.

## What the RESULT prints

For each `gate_sha256` stretch, with 15.12's reading of (iv): the count of public send lines; the sums of `trailer_quotes`,
`trailer_quote_numerals` and `trailer_forms_unverified` over them, and the seqs named (the four keys print "not recorded" on
the lines of the earlier stretches); `Nn`, `Nb`, `Nu` as the lines carry them, beside the sum of `trailer_quote_numerals`, so
a reader can add the quoted numerals back; the census class above; seq 46 as the one line of the ledger at seq 116 whose count
the amended rule reads otherwise, with the two verify reports' digests; the day-0 token split of "what was observed" item 2.

## The lock rows that move

Old → new, each new digest read off `lock_hashes.py check --strict` against the copy [R, `BD:lock_check_trailer_pass2.txt`,
run from this checkout with `--w-root BD:_lockroot` (a root whose `gate` is a junction to the copy) and `--home BD:_lockhome`
(empty, so nothing under the operator's home is read)]. The `gate.py` and `gate_verify.py` rows are shared with the
posting-skill AMENDMENT (piece (4)'s guard is in the same two files): both files name one digest for each, and whichever
piece edits either file after this row was read restates the row in both. The live relock is the lead's at apply time.

| what | path | sha256 | note |
|---|---|---|---|
| gate file | `W:gate/gate.py` | `4416692cfc983dbb9bc7d0f31342b3eaaf30c3beb7e27f31acdaa991c778dfbf` | 95482 bytes; the count rule (this amendment) and the prepared-record guard (the posting-skill AMENDMENT), one file; before it `25d53fd6…` (15.10) |
| gate file | `W:gate/gate_verify.py` | `049d870bcdbf4726d961bc4b58d5b3134c63f06511d398c11c0f7ed57dd3487b` | 28886 bytes; the replay's earlier send lines and the four compared keys (this amendment) and the guard's replay (the posting-skill AMENDMENT); before it `f4b19c94…` (§11) |
| gate file | `W:gate/README.md` | `54e9a33f5bf4a3475197c6ee76a4564cea6d77d58dbf37e220865beb453b95bf` | 21520 bytes; the L2c bullet states the rule and what it does not exclude, and the verifier's paragraph names the earlier send lines; before it `de55f382…` (§11) |
| gate file | `W:gate/tests/test_trailer_rule.py` | `f5a19f6a801d5ce2cd31b06b50b1a4746703debb2c63ca58d5f87145b74b50ba` | 29086 bytes; new with this amendment, no earlier row; 37 tests and 10 subtests, with the two mutation switches of "the tests" |

`W:gate/tests/test_verify.py` (`669100d4…`) and `W:gate/tests/test_gate.py` do not move for this piece (piece (4) moves
`tests/test_gate.py` and `tests/test_gate_round2.py` for its guard, in its own AMENDMENT).

## The tests

All in the copy, with the real legs of the pinned snapshot, in gate homes of their own [R]:

- Baseline, before any change (`gate.py` the live `25d53fd6…`): the copy's suite 229 passed, 2 subtests, 715 s
  (`BD:pytest_pass2_baseline.txt`); the plugin harness 79 of 79 (`BD:plugin_harness_pass2_baseline.txt`).
- `tests/test_trailer_rule.py` (37 tests, 10 subtests) against the unmodified `gate.py`: 29 tests failed and 8 passed
  (`BD:trailer_tests_pass2_runs.txt`, run A), the receipt that the tests were written before the change; against the
  amended file: 37 passed, 10 subtests passed (run B).
- The mutation runs (`BD:trailer_mutation_runs.txt`, by node id against the sets the test file's docstring names):
  with `STYXX_GATE_TRAILER_MUTATION=noexclude` (the verified spans ignored) 12 failed, exactly the 10 tests of
  `TestVerifiedQuotation`, the one of `TestAQuotationBesideAComposedForm` and `TestVerifyRederives`' SAME test (its
  `gate_verify.py` subprocess runs the unmutated gate over the mutant's lines and reads DRIFT); with `noverify` (every
  form taken as verified) 14 failed, exactly the 12 of `TestUnverifiedFormsCount`, the mixed test and the SAME test. Every
  near-form, prose seq, version, DM-mark and line-shape test passes under both, so they read the rule's other parts, and
  the two mutated parts are what the two named classes read.
- The whole suite in the copy after the change, piece (4)'s guard and its tests included: 279 passed, 12 subtests passed,
  882 s, rc 0 (the baseline's 229, the 37 here, the 13 of piece (4)'s `tests/test_prepared_guard.py`;
  `BD:pytest_pass2_after_change.txt`); the plugin harness against the copy: 79 of 79, rc 0
  (`BD:plugin_harness_pass2_after_change.txt`). `W:gate/README.md`'s row is the file after one sentence was made plainer
  at 2026-10-26T00:57Z, during the suite; no test, no instrument and no `gate_sha256` reads that file.
- The two verify reports of item 11.
- On this checkout [R, `BD:styxx_checks_trailer_pass2.txt`]: `scripts/surface_lint.py --files papers/darkflobi-gate/*.md`
  0 hits; `tests/test_darkflobi_gate_prereg.py`, `tests/test_surface_lint.py` and `tests/test_gate_words_scan.py` pass in
  the default mode; run box-named against the copy, `test_lock_rows_agree_with_the_box_when_a_caller_names_it` fails, as
  it must until `lock_hashes.py check` reads amendment rows (`--amendment`, the lock-rows AMENDMENT's piece): `check`
  reads the PREREG's rows alone today, and the copy's six moved files (this file's three and piece (4)'s) read CHANGED.

## What is the operator's and the lead's

1. The rule's shape: the verified quotation (built here); the alternative the review named (every form left out, the gate's
   trailer always appended after one) is rejected because digits written inside a composed form still leave the count
   (A5); the third option, no exclusion at all (the skill alone tells the agent not to quote marks), leaves seq 46's shape
   marked.
2. `gate_verify.py` moving with it (built here; the verified rule needs the chain in the replay). The alternative, the replay
   trusting the line's own record of which forms verified, weakens the verifier for those lines and is not taken.
3. `W:gate/README.md` relocked with its row (built here: the L2c bullet and one clause of the verifier's paragraph); the
   stale drop-rule sentences 15.5 names are not restated.
4. At apply, the lead: the one move (the lock-rows AMENDMENT), the suite in place, `check --strict` with the amendment
   files, `gate_verify.py --rederive` reading DRIFT on seq 46 alone, the fourth stretch's digest and seq in the lock-rows
   addendum. The operator: the merge of the pull request that carries this file, the relock of record.

## What this amendment does not say

That any day-0 post's standing changes: 0 of 5 clear, and the lines are counted under the old rule. That the gate reads what a
sentence says: the rule is a byte match and a lookup in the gate's own ledger. That a quoted seq, a version or a self count is
excused: they are numerals. That a verified quotation is true of the post it sits in: it is true of the line it names. That
seq 46 re-derives SAME under the amended gate: it reads DRIFT there, and SAME under its own stretch's gate. That the census or
the plugin changed: neither did. That `c5084df2…` is the fourth stretch's digest: it is a candidate, and the lock-rows
addendum names the digest of record.
