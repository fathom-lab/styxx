# AMENDMENT — the count rule: a numeral inside the gate's own trailer form is left out of the count, and a form standing where the gate's trailer stands is not

Fathom Lab · 2026-10-25 · Amends `PREREG_darkflobi_gate_2026_10_23.md` and `RULES_darkflobi_gate_2026_10_23.md`, in this
directory, merged as #229 with section 15 closed at 15.13 and not edited. Section 12 of the PREREG (its closing paragraph)
provides for this file: after the merge a change to a rule is a new AMENDMENT file under this folder, lock-hashed in its turn,
with deliveries under the old rule counted under the old rule. Written by the build agent for piece (1) of the before-D0 work,
on the lead's plan (`WT:_darkflobi_gate_2026_10_22/before_d0/PLAN.md`, sha256 `71cf3d10…`, section 3), before D0 and before
`W:gate/REQUIRED` exists. The operator has not yet signed it: the section "what is the operator's and the lead's" lists what is
theirs. The plan called this piece "the trailer self-reference repair"; this file is the count rule, because that is what it
restates.

Receipt key: `W:` = `C:/Users/heyzo/clawd/`; `WT:` = `W:wt/`; `BD:` = `WT:_darkflobi_gate_2026_10_22/before_d0/`; `M:` = the
styxx checkout at the pin `3437340b`; `seq N` = a line of `W:gate/ledger.jsonl`; `[R]` a file or a run shows it; `[J]`
judgement. The live gate `W:gate` was read and never written: every change below was built and tested on the copy
`BD:gate_copy/` (made 2026-10-25 13:36 local with the brief's exclusions), and the lead moves it into place at apply time.

## What the PREREG and the RULES say

- `RULES` §5: "`Nn` on a line is the count of numerals in the checked text (`numerals.total`): certify's ledger rows plus
  its counted uncovered band (…), read whenever the text holds a digit [R, `gate.py` `bind_numerals`]. `Nb` is the count
  bound (§4, `numerals.bound`), by position (…); `Nu` the count unbound and marked (`numerals.unbound`)". Nothing in §5 is
  added to or retyped from certify's extraction.
- `RULES` §7: the trailer `[gate: Nb of Nn numerals bound · ledger <seq>]` is appended after a blank line on Telegram when
  `Nu` > 0 and nothing is contradicted; the gate never edits inside a text. `gate.py` `TRAILER` and `_trailer_template`
  write that form; `fields_for` fills the seq under the lock.
- `RULES` §12: the census strips the trailer's fixed form at the END of a delivered text (`reconcile.py` `TRAILER_END`,
  `\n\n\[gate: \d+ of \d+ numerals bound · ledger \d+\]\Z`) before matching it to its line, and nowhere else.
- `RULES` §13: a `send` line's `numerals` carries `total`, `bound`, `unbound`, `uncovered`, `unbound_tokens`.
- PREREG 15.12 and 15.13: component (iv) of §5 is read per `gate_sha256` stretch, each send line against the digest of the
  files locked when it was written; a fourth digest begins a fourth stretch.
- The shadow log (`WT:_darkflobi_gate_2026_10_22/shadow/SHADOW_LOG_2026_10_24.md`, finding 2): when darkflobi quotes or
  explains the gate's own trailer, "the trailer's numbers are read as new unbound numerals and the post is marked again";
  the candidate repair it names is "recognise the trailer's fixed form anywhere in a payload, leave its numerals out, print
  the count of recognitions", by an AMENDMENT before D0, never an edit of frozen text.

## What was observed

1. **The ledger at seq 116** [R, `BD:verify_scratch/home/ledger.jsonl`, a copy of `W:gate/ledger.jsonl` taken about
   2026-10-25T19:25Z, sha256 `baf3f6924b388fff8132cb4315c9f507cd3d15b97c6db1dbe86cd06338431381`, 117 lines = the header and
   seq 1-116, head `76de868c00863ca9…`, the chain re-deriving under `gate_verify.py`]: `started` 1, `send` 57, `delivered` 57,
   `anchor` 1; `required_present` false on every line (shadow throughout). Send lines by class and decision: `public` 24
   `allow`, 5 `allow_marked`, 1 `refuse` (seq 93, `lint_hit:charter/ordinal`, sent because shadow); `operator_dm` 21
   `allow`, 6 `allow_marked`. `gate_sha256` stretches: `387fc063…` seq 1-13, `ff34b52d…` seq 14-116; no line carries
   `592dc579…` (15.12).
2. **The five marked public lines, all of 2026-10-24** [R, the lines' own fields; the readings are PLAN §1.3's]:

   | seq | exit | `checked_sha256` | `numerals.total` | `unbound_tokens` | reading |
   |---|---|---|---|---|---|
   | 26 | E1 | `9740b3a5…` | 2 | `3.1`, `4.8` | two-part model versions in prose |
   | 30 | E3 | `e7ff8ff4…` | 1 | `0026` | a ledger seq in the agent's prose |
   | 34 | E3 | `4fc9736d…` | 4 | `0026`, `3.1`, `4.8`, `2` | a prose seq, two quoted versions, a self count |
   | 44 | E1 | `f98ec408…` | 2 | `64`, `026,964` | a patent application number, no receipt |
   | 46 | E1 | `d178cbc6…` | 4 | `0`, `2`, `0044`, `0044` | three digits inside an exact copy of the gate's trailer in backticks; one more `0044` in prose |

   Public numerals on day 0: 13, bound 0. Labelled one by one by the product red team's rule (`WT:_product_2026_10_24/PRICING.md`
   §5.1, the labelling rule for M4; `RED_TEAM.md` M5), re-made here from the lines' tokens: **T** (inside an exact copy of
   the mark) 3, seq 46's `0`, `2`, `0044`; **V** (version) 4, seq 26 ×2 and seq 34 ×2; **S** (a seq of the gate's own
   records in prose) 3, seq 30, seq 34, seq 46's prose `0044`; **C** (self count) 1, seq 34's `2`; **N** (no receipt) 2,
   seq 44. The shadow log's "6 no-receipt" is S + C + N under this split.
3. **Where the form stands on the ledger** [R, a scan of `BD:verify_scratch/home/sources/<checked_sha256>.md` for the
   grammar below, counts only]: of 57 send lines, one holds the gate's trailer form in its checked text, seq 46, one form,
   not at the text's own trailer position. No other line, public or DM, holds it; seq 113 and 115 (DM, `7`, `402` ×4) do
   not.
4. **What the gate did with it.** `bind_numerals` (`gate.py` `25d53fd6…`, l.963-991) counted every certify row and every
   counted uncovered item, binding by position against sworn HELD spans only; nothing recognised the gate's own form. The
   anchor's form anywhere in a payload was already a spoof refusal (`ledger.looks_like_anchor`), so a quoted anchor line
   was never a counting question; the trailer had no such rule.

## The amendment

1. **The rule.** A numeral whose opening character lies inside a match of the gate's own trailer grammar, read byte for
   byte anywhere in the checked text, is left out of the count: it is neither bound nor unbound, enters neither `Nn` nor
   `Nu`, and is not in `unbound_tokens`. The grammar is the form the gate itself writes and nothing wider:

   ```text
   \[gate: \d+ of \d+ numerals bound \u00b7 ledger \d+\]
   ```

   (the words, the middle dot U+00B7, the brackets; the digits free, the seq's zero padding free). Howsoever it is
   surrounded (backticks, quotes, prose on both sides) the match is the same. `gate.py` `trailer_form_spans(text)` returns
   the matches as character spans; `bind_numerals(text, cert, nums, held_at, excluded, own_position)` drops every certify
   row and every counted uncovered item whose `starts[line - 1] + col` (certify's own line and column, in characters, the
   arithmetic the HELD-span test already uses before its byte conversion) lies in one of them, before anything is counted.
   `check` calls the two in that order (the amended `gate.py` `8d45166a…`: `TRAILER_FORM_RX`, `TRAILER_OWN_POSITION_RX`
   beside `TRAILER`; `trailer_form_spans`, `_inside`, `bind_numerals` before `check`; the call after `held_at`).
2. **The one clause: the gate's own position.** A form that stands where the gate's own trailer stands, as the last
   non-blank content of the checked text (`\[gate: … ledger \d+\]\s*\Z`), is NOT left out: its numerals count like any
   other, read unbound, and the gate's real trailer is appended after it, so the reader sees the gate's own last. Without
   the clause a message holding no other numeral and a hand-written trailer as its closing line would leave clean under a
   mark the gate did not write. PLAN §3.3 named this clause on the census's `TRAILER_END` shape (a blank line, then the
   form, then the end); as built the clause reads the form as the text's last non-blank content by any route: after a blank
   line, after a single line break, as the whole text, or with trailing whitespace [J, the builder's widening: a form on the
   last line reads as the gate's mark whether or not a blank line precedes it, and the writer's rule below is the same
   either way; the lead narrows it to the `\n\n` shape by one regex if he prefers the plan's text]. A form followed by any
   character but whitespace (a closing backtick, a period, more prose) is not at the own position and is left out as item 1
   says.
3. **What the rule does not exclude**, each with a test in `W:gate/tests/test_trailer_rule.py` [R]: any digit outside the
   exact bracketed form; a near-form, that is a hyphen, an en dash or a plain space for the middle dot, a no-break space
   or an invisible character inside the form, "numeral" singular, the form split over two lines, its brackets escaped as
   markdown (`\[gate: …\]`), a doubled space, a capital letter (`TestNearFormsCount`, ten shapes, each counted 3 of 3); a
   ledger seq in prose, "ledger 0044", "the ledger (0026)" (`test_a_ledger_seq_in_prose_counts`); a two-part version
   (`3.1`, `4.8`), while a three-part version is scrubbed by certify's `_VERSIONISH` as before
   (`test_two_part_versions_count_and_a_three_part_version_is_scrubbed_as_before`); a self count and a patent number
   (`test_day0_seq34_shape…`, `test_day0_seq44_shape…`); the operator-DM mark `[gate: <codes>]`, whose codes can carry a
   numeral (the pin's lint has one `plan/` rule whose id is a number of the plan's; a mark naming it is the test's text:
   the lint fires on the mark and the number is counted, so the test composes the id at run time and no file spells
   it) and `[unchecked: styxx unavailable]` (`test_the_dm_marks_codes_with_a_numeral_count_and_the_lint_fires`); a form at the gate's own position
   (`TestTheOwnPositionClause`, five tests). The rule reads bytes, never what the sentence says, and it is the same at
   every exit and every destination class: the operator's DM takes the same count, E3 the same, and X refuses on the
   prose numeral alone (`test_the_operator_dm_takes_the_same_count`, `test_the_other_exits_take_the_same_count_and_x_refuses_on_the_prose_numeral_alone`).
4. **Three fields added to `numerals` on every `send` line** (RULES §13 restated): `trailer_forms`, the count of forms
   whose numerals were left out; `trailer_numerals_excluded`, the count of numerals left out; `trailer_form_at_own_position`,
   1 when the clause of item 2 fired, else 0. `total`, `bound`, `unbound`, `uncovered` and `unbound_tokens` keep their
   meaning over the numerals that remain. A line `gate.py fault` writes, an `internal` line and a line whose legs did not
   run carry the three at 0 (`TestTheLineShape`). The verdict `gate.py check` prints keeps its three counts (`total`,
   `bound`, `unbound`) and nothing else: the plugin reads nothing new.
5. **Rejected: a version grammar.** Certify already scrubs three-part versions (`_VERSIONISH`, `M:styxx/certify.py` l.55)
   and the gate never retypes certify. A two-part version (`3.1`) is, in bytes, the shape of a measured share (`0.23`); the
   only grammar that tells them apart is a list of product names, which is a content exception, and RED_TEAM M5 reads the
   same. No version grammar is added: a two-part version is a numeral, bound by a leaf or marked.
6. **Rejected: a quoted ledger seq in prose.** "ledger 0044 is the receipt", "the ledger (0026)" are the agent's own
   numbers about its own records. A rule `ledger \d+` would also leave out "ledger 258 entries", a committed-count claim
   (RULES §3.1 shape S9), and would make the agent's commentary on its own seqs uncountable. The marks on seq 30, 34 and
   46's prose `0044` are the gate doing what it is for; they stand.
7. **What a writer can rely on.** Quoting the gate's trailer exactly as the gate appended it, the whole bracketed form,
   inline or in backticks, is not counted; quote it as the last content of a message and it is counted, with the gate's own
   trailer following it. Anything else that looks like the form is prose and its digits are numerals. A seq is a number:
   print it bound or not at all. A version with three parts, or with `v`, is scrubbed by the certifier as before; a
   two-part version is a numeral.
8. **No new reason code, no change to the matrix.** `decide`, `_trailer_template`, every code of RULES §6 and the readings
   of 15.5, 15.10 and 15.11 are unchanged; `findings` gains nothing, so `tests/test_darkflobi_gate_prereg.py`'s
   box-named code test reads the same codes against the amended file.
9. **Replay.** `gate_verify.py` is not changed, and `_COMPARE_NUMERALS` is not extended with the three new keys, so a line
   written before the move (which has no such keys) re-derives on `total`, `bound`, `unbound`, `uncovered` and
   `unbound_tokens` alone. Consequence, stated here so that no reading is chosen after the fact: of the 57 send lines on the
   ledger at seq 116, one, seq 46, re-derives DRIFT under the amended gate (`numerals.total` 4 → 1, `numerals.unbound` 4 → 1,
   `numerals.unbound_tokens` `['0', '2', '0044', '0044']` → `['0044']`, `sent_sha256`, the trailer reading `0 of 1`); every
   other line SAME. Under the gate of its own stretch (`gate.py` `25d53fd6…`, kept at `BD:gate_before/gate.py`) every one
   of the 57 re-derives SAME [R, `BD:verify_scratch/report_before_25d53fd6.json`, sha256 `875e6a69…`, run 2026-10-25
   19:28-19:29Z: chain re-derives, 116 lines, SAME 57, DRIFT 0, TAMPER 0, UNRESOLVED 0; the report's `config_moved` flag is set on
   every line because the scratch home carries no `config.json`, a detail flag, not a status]. Under the amended gate [R, `BD:verify_scratch/report_after_8d45166a.json`,
   sha256 `17cf4eb4f81404d5624b3395ba27ac996be4f104ad56505f32086987ff17b88c`, run 2026-10-25 19:29-19:30Z: chain re-derives,
   116 lines, SAME 56, DRIFT 1, TAMPER 0, UNRESOLVED 0, exit 1] the one DRIFT row is seq 46 with `diffs` `sent_sha256`,
   `numerals.total`, `numerals.unbound`, `numerals.unbound_tokens` and no module skew, as predicted above. Seq 46 is counted under the old rule (PREREG §12): on the ledger it reads `0 of 4`, it was
   delivered with that trailer (seq 47 `delivered ok`), and the RESULT lists it as the one line whose count the amended rule
   would have read otherwise. A per-stretch replay flag in `gate_verify.py` (`OTHER_STRETCH`: a line whose `gate_sha256`
   is another listed stretch's is replayed and reported with its diffs, exit unaffected) is not built for one shadow line
   [J]; it is one flag, one test and one lock row if the lead wants it.
10. **Component (iv), a fourth digest.** By 15.6's rule (`W:gate/*.py`, `W:gate/helpers/*.py` and the six plugin files as
    installed, 18 files, each its name, a NUL, its LF bytes and a NUL), re-derived by a script that does not import
    `gate.py` over the amended copy's files and the installed plugin's six files as recorded at
    `BD:_probe/plugin_installed/` [R, the same script re-derives `ff34b52d…` over the unamended copy]:
    `833b407dfb3b69ab120fc2d5e85a07c3c3c5e1a7a2f66109cc47000c10543a1c`. It is the digest when `gate.py` is the one `*.py`
    file that moves; PLAN §2.5 lands piece (1) and piece (4) as one moved `gate.py`, and any other `*.py` landing in the
    same move (the copy today also carries another piece's `canary_plan.py`, which is not this amendment's) changes it, so
    the lead re-derives the digest at apply time, as 15.6 and 15.10 did, and the RESULT reads the fourth stretch from the
    seq at which the moved file writes its earliest line, against that digest.

## Effect on day 0, said plainly

Under this rule 3 of the 13 public unbound numerals of 2026-10-24 would have been left out, seq 46's `0`, `2`, `0044` inside the
backticked form (the T tokens); seq 46 would still be marked, `[gate: 0 of 1 …]` for its prose `0044`; seq 26, 30, 34 and 44
are unchanged; **0 of the 5 marked posts clear**. The repair is a rule about the gate's own bytes, not a repair of those
posts, and PREREG §12 counts them under the old rule in any case. The census class the shadow log asked for, marks whose only
unbound tokens sat inside a quoted trailer, is empty on day 0 (seq 46 does not qualify) and the RESULT prints it.

## What does not change

The pin, the snapshot and every instrument; certify's `_NUM` and its scrubs, which the gate still does not retype; RULES §4's
grammar and the drop rule as signed; §6's codes and §7's matrix; the trailer the gate writes and where it writes it; the
census's `TRAILER_END` strip (RULES §12), which still reads the END of a delivered text and nowhere else; `gate_verify.py`
and its compared fields; the verdict's shape; the plugin, which relays `content` as before (the harness 79 of 79, PLAN §2.1).
Lines written before the move are read under the old rule.

## What the RESULT prints

For each `gate_sha256` stretch, with 15.12's reading of (iv): the count of public send lines, the sums of `trailer_forms` and
`trailer_numerals_excluded` over them, the count of lines with `trailer_form_at_own_position` 1 (lines of the earlier
stretches print "not recorded" for the three); the census class above; seq 46 as the one line of the ledger at seq 116 whose
count the amended rule reads otherwise, with both verify reports' digests; the day-0 token relabel of "what was observed" item 2.

## The lock rows that move

Old → new; each new digest read off `lock_hashes.py check --strict` against the copy [R, `BD:lock_check_copy_after_change.txt`,
run from this checkout with `--w-root BD:_lockroot` (a root whose `gate` is the copy) and `--home BD:_lockhome` (empty, so the
installed plugin under `~/.clawdbot` is not read): rows 80, CHANGED 2 (the two below), BUILT SINCE 0, UNLOCKED 3
(`W:gate/tests/test_trailer_rule.py`, this amendment's new row; `W:gate/canary_plan.py` and `W:gate/tests/test_canary_plan.py`,
another piece's files in the shared copy, not this amendment's), ABSENT 10 (`salt`, `config.json`, `anchor_dest.txt`, the X script
and the six installed plugin files: excluded from the copy or outside the fake root, not moved), BYTECODE 2 (the copy's test
caches; none in `W:gate`), NOT BUILT 1 (`W:skills/styxx-post/SKILL.md`), SKIPPED 1 (the `plugins` subtree)]. The live relock is
the lead's at apply time.

| what | path | sha256 | note |
|---|---|---|---|
| gate file | `W:gate/gate.py` | `8d45166a9495fab201d75cebe839cd3d8c7b360bca16132492caf3fb62e9fefe` | 91036 bytes; the count rule (this amendment); before it `25d53fd6…` (15.10); `gate_sha256` from this row on `833b407dfb3b69ab120fc2d5e85a07c3c3c5e1a7a2f66109cc47000c10543a1c` when it is the one `*.py` that moves (item 10), before it `ff34b52d…` (15.10) |
| gate file | `W:gate/README.md` | `c517480e5dcddcde33ff55b3990e9e788a1e53ac7ab540291fe809d224e113f3` | 21163 bytes; the L2c bullet states the rule and the clause; before it `de55f382…` (§11; 15.5 already notes this README stale on the drop rule, which this edit does not restate) |
| gate file | `W:gate/tests/test_trailer_rule.py` | `2485d6bacff9d1d2ee27270b7fa96b7feeb6d39543b171bfc3732a054321596a` | 18172 bytes; new with this amendment; 27 tests and 10 subtests; carries the mutation switch of "the tests" below |

## The tests

All in the copy, with the real legs of the pinned snapshot, in gate homes of their own [R, `BD:trailer_tests_runs.txt`]:

- `tests/test_trailer_rule.py` against the unamended `gate.py` (`25d53fd6…`): 30 failed, 7 passed (the tests read the
  three new fields and `trailer_form_spans`, which that file does not have; the receipt that the tests were written before
  the change). Against the amended file: 27 passed, 10 subtests passed, 66.8 s.
- The mutation run (PLAN §3.6 (i); `STYXX_GATE_TRAILER_MUTATION=1`, which replaces `gate.bind_numerals` in the test process
  by a copy that ignores `excluded`): 9 failed, 18 passed, 10 subtests passed. The 9 are the 8 tests of
  `TestTheFormIsLeftOut` (the exclusion) and `TestVerifyRederives`, whose `gate_verify.py` subprocess runs the unmutated
  gate over the mutant's lines and reads DRIFT, the verifier catching a gate that counted otherwise. Every near-form, the
  own-position clause, the prose seqs, the versions, the DM mark's codes and the line shape pass without the exclusion,
  which is what (c)-(g) of the plan asked.
- The whole suite under `W:gate/tests/` in the copy and the plugin harness `node plugin/tests/run_tests.mjs` against the
  copy: their counts and the run times are in `BD:pytest_run2_after_change.txt` and `BD:plugin_harness_run2_after_change.txt`,
  restated in `BD:BUILD_trailer.txt`; the baseline before any change was 220 passed and 79 of 79 (`BD:pytest_run1.txt`,
  `BD:plugin_harness_run1.txt`). The shared copy also carries another piece's `tests/test_canary_plan.py`, so the suite's
  count there is not 220 + 27 alone.

## What is the operator's and the lead's

1. The clause's width (item 2): the plan's `\n\n` shape, or the last non-blank content as built (the default here).
2. `W:gate/README.md` relocked with its new row (the default), or left stale as 15.5 left it.
3. The `OTHER_STRETCH` replay flag (item 9): not built unless asked.
4. At apply time, the lead: `os.replace` of `gate.py`, `README.md` and `tests/test_trailer_rule.py` into `W:gate` with
   `REQUIRED` absent before, during and after, as 15.6 and 15.10 did; the suite in place; `lock_hashes.py check --strict`
   against the box reading SAME on every row but these three until the PREREG's pull request restates them, and SAME after;
   the `gate_sha256` of item 10 re-derived from the box and the seq at which it begins noted here in a dated addendum. No
   gateway restart: `gate.py` is spawned per call.

## What this amendment does not say

That any day-0 post's standing changes: 0 of 5 clear, and the lines are counted under the old rule. That the gate reads what a
sentence says: the rule is a byte match on the gate's own grammar. That a quoted seq, a version or a self count is excused:
they are numerals. That seq 46 re-derives SAME under the amended gate: it reads DRIFT there, and SAME under its own stretch's
gate, both said above. That the plugin, the census or the verifier changed: none did. That the fourth `gate_sha256` is
`833b407d…` whatever lands with it: the lead re-derives it at apply time.
