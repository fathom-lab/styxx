# result: a clean-room Go checker for RECEIPT v0, and what the text learned from it

fathom lab, 2026-11-13. Text: `receipt/v0/RECEIPT_v0.md`, DRAFT, unsigned. Clean-room build: `wt/_receipt_v0_cleanroom_2026_11_13`,
copied byte for byte into `receipt/v0/cleanroom-go/` (see its README). The measurer's working files are in
`wt/_receipt_v0_checkers_2026_11_13_notes/measure/` and the fold round's in `.../fold/`. Neither folder is committed. Every count
below comes from one of those runs. Where a count was printed by the measurer and not re-run in the fold round, the text says so.

This is a lab result about a lab draft. It is not a standard, and no party outside the lab implements, emits or checks these objects.
The checker this text describes is a checker of bindings an author hands over. It does not find claims, and it is not a detector of
false statements in prose.

## 1. what was tested

The question: does RECEIPT v0 stand on its text alone? Can an implementer who holds the text, the schema and the vector file, and
no lab code, build a checker that (a) answers the vectors, (b) agrees with the lab's two checkers on inputs the vectors never name,
and (c) where it does not agree, which side does the text decide for?

The method had three roles, all agents of the lab's own workflow:
- **author.** It was told to read only the given files and to build a checker in Go, a language the lab's checkers do not use. It kept
  its own log of the files it opened, and it wrote down its guesses before it opened the vectors.
- **measurer.** It was not the author. It ran the vectors, a three-way differential (Python `check_v0.py`, JavaScript `check_v0.js`,
  Go) over generated and targeted cases, and an audit of what the author read. It classed each disagreement by the text's own lines.
- **fold (this step).** It turned the measurer's gap list into text, vectors and one repair of a lab checker. It copied the build into
  the repository as evidence, and it wrote this account. The Go build was not edited. Its bugs stay in.

## 2. what the author was given

Three files, placed in `given/` at 13:50:32Z:

| file | sha256 |
|---|---|
| `RECEIPT_v0.md` (the text as of commit a5b30db1) | `c09d75b0bc40f58bdbee8033cb5768969b0d763f36ec1d768f7066083a6562bf` |
| `receipt-v0.schema.json` | `0881cc8ba07bd11c49a75eb61d83b8cb9766d1b24587e70f0fb632fb4f8901c8` |
| `receipt_v0_vectors.json` (2,193 vectors) | `26e05c00a3b18749eb2cf20b1f38622d6f40e14d36ec753d774ed3cc86f33a7a` |

It was not given `check_v0.py`, `check_v0.js`, `to_v0.py`, `styxx/sworn.py`, the decisions file, the lab's notes or any test.

## 3. the audit of what it read

The author's log (`cleanroom-go/files_read.log`) lists the three given files and its own files, and nothing else. For the vector file,
it first read only the shape (key paths and value kinds), then the top-level descriptive strings, and then it read the expected answers
only through its runner. It records five small Python edit scripts it ran in the session scratchpad, used as a text editor on its own
Go files, and it says it opened nothing under any styxx folder, any notes folder or any other worktree.

The measurer checked as far as the disk allows (`measure/audit/`):
- **names.** The Go source has 96 functions. 16 of them have the same name as a `check_v0.js` function once case and underscores
  are ignored (`main` is one of the 16), 14 match a `check_v0.py` name, and 2 match a `sworn.py` name. As a control, `check_v0.js`,
  written with `check_v0.py` open, shares 57 of its 83 names with it. Most of the shared Go names follow from the text's own words
  ("numeric, form", leaf, REF, `path:` target, `lone_surrogate`, Nd). One does not: `isHex64Any` is exactly a `check_v0.js` name, and
  the text gives no source for it.
- **strings and comments.** 4 of the 71 Go string literals that are not in the given files occur in lab code: the hex digits, the
  base64 alphabet, `"REFUSED "` and `"_.,+-%/:"` (7.2's list with its spaces removed). 4 of the 457 comment 5-grams that are not in
  the given files occur in lab code. One is a full shared phrase, "keys written more than once". The other three paraphrase the text.
  No lab-only structure turns up: no lab section tags, no lab note strings, no lab error messages.
- **isolation.** The author's scratchpad was shared by every agent of the workflow session. During the build it held lab material
  from other agents: a derive/v0-draft row written 22 s before `checker/` was created, and a lab pytest temp folder. No `check_v0`
  or `sworn` source was in it. No access log of the operating system exists. As on 2026-11-09, the clean room was an instruction
  plus a self-report on a shared disk. It was not an isolation.
- **timing.** Given files at 13:50:32Z, `checker/` created at 13:56:03Z, `first_run.txt` at 14:03:42Z, `final_run.txt` at 14:08:30Z.
  Every file edited after the first run sits behind an `RV0_ALT` switch. The source as it stood at the first run was not kept, so
  nothing shows that the default build behaved the same then on inputs other than the vectors.

Net: nothing found shows that lab code was copied. `isHex64Any` and one comment phrase are not explained by the text. Nothing here
shows that the build never saw lab material.

## 4. pass rates

| run | result |
|---|---|
| first run, the 2,193 vectors (`first_run.txt`) | 2193 of 2193 (conformance 1718, hand 416, jcs 27, recheck 32) |
| final run, the default build | 2193 of 2193, byte-identical to the first run |
| the measurer's rebuild of the current source | byte-identical to `first_run.txt` (sha256 `a376211f...`) |
| fold step: `go build` of the copy in `receipt/v0/cleanroom-go/` | byte-identical to `first_run.txt` |
| fold step: the 62 vectors of `receipt_v0_vectors_fold.json` | 59 of 62; it fails the 3 vectors of 3.3's order of faults |

The author ran 24 of its 33 guesses both ways (`GUESSES.md`, `alt_runs.txt`). The vectors decide 16 of them. They leave 8 open: G1,
G2, G3, G8, G10, G17, G19 and G30. Three controls, rules made wrong on purpose, fail 11, 8 and 155 vectors, so the runner can fail.

## 5. the three-way differential

From the measurer's runs (`measure/summary.json`). The JavaScript checker in these runs is the one before the fold step's repair (§8).

| set | cases | disagree | odd one out |
|---|---|---|---|
| the vectors | 2,193 | 0 | (each checker gives every expected answer) |
| the JavaScript pitfall probes | 88 | 0 | |
| fresh generator run, `gen.py` seed 20261114 | 53,825 | 52 | Go, all op `jcs` |
| the lab's `cases_v3`, seed 20261113 | 53,827 | 49 | Go, all op `jcs` |
| the lab's `cases_v4_extra` | 2,363 | 0 | |
| targeted at the open guesses and at what 13.15 calls barely drawn | 2,933 | 0 | (0 crashes; every family reached its rules) |
| deep nesting, one process per case, 180 s each | 47 | 7 | Go on 4; on 3, Go crashed and JavaScript gave no answer |
| **total** | **115,276** | **108** | |

The fold step reran the repaired `check_v0.js` over `cases_v3` and `cases_v4_extra`. Its output is byte-identical to its output before
the repair, so there are still 0 disagreements with Python (`fold/diff_after_jsfix.txt`). It did not rerun the fresh, targeted or deep
sets through the three checkers.

## 6. every disagreement, by class

| class | cases | what | decided now |
|---|---|---|---|
| text silent | 101 | a value with two or more canonical-form faults, op `jcs`. 3.3 named six codes and no order. Python and JavaScript check every key, then the values in sorted key order. Go checks every key, then the values in the order written. With values visited in written order, Python equals Go on 1145 of 1145 `jcs` cases in each set. Go's own alternate (each key, then its value, sorted) differs from the lab on 111 of 53,825. That makes three readings, and no vector pinned any of them. No verdict moves: the code reaches only a caller asking for the canonical form, and 5.6 and 11.3 refuse such a value whatever its code. | 3.3 now names the lab's reading, with worked examples and 9 fold vectors. 13.20 records the other two. Go keeps its reading and fails 3 of those vectors. |
| clean-room bug | 7 | nesting depth 3,000,000 (about 6 MB of brackets), in receipt bytes, in a manifest's ignored key or in a `jcs` value. Go exits 2 with "goroutine stack exceeds 1000000000-byte limit". Its parser is recursive, and `recover()` cannot catch this. On the 3 cases nested with objects, JavaScript gave no answer within 180 s either (§8). Python answered all 47 deep cases. At 1,000,000 all three agree. | 6.13 now says JSON is read at any depth and a checker must not stop without an answer. 13.5 is rewritten. 4 fold vectors sit at 100,000 levels, and a test builds 3,000,000 in memory. The Go bug stays in. |
| text decides for the lab | 0 | | |
| text decides for the clean room | 0 | | |

The class that would matter most, where the lab's two checkers share a reading the text does not support, was not found. The two
did share one unpinned reading, G2, but there the text was silent, not against them.

## 7. the gap list and what was done about each

The measurer listed 24 items. Each became a rule in the text with an example. Where a vector can state an item, it also got one or
more vectors in `receipt/v0/vectors/receipt_v0_vectors_fold.json`: 62 in all, written by hand from the text, with the builder refusing
any vector that `check_v0.py` answers differently. The two lab checkers needed no change for any item, because the text took their
reading each time. The one lab repair of this step (§8) answers a finding outside the gap list.

| item | section | done | fold vectors |
|---|---|---|---|
| G2 order of faults | 3.3, 13.20 | rule, six examples, the other readings recorded | 9 (`jcs`) |
| G3 a key with an unpaired surrogate | 3.3 | `non_ascii_key`, two examples | 2 |
| G1 nesting depth | 6.13, 13.5 | any depth; no refusal and no stop because of depth; 13.5 rewritten with the other reading | 4, plus the 3,000,000 test |
| G16 attestations against the schema | 11.3, 13.20 | the shape step needs only an array; the schema's `maxItems` 0 is not applied there (the schema file was not edited) | 3 |
| G17 a receipt check without a document | 11.3 | `nothing_to_verify` after every receipt-only refusal | 3 |
| G19 numbers in `const`, `enum`, `minimum` | 11.3 | read as written; `-0` meets `minimum` 0 | 6 |
| G18 a manifest refusal while re-deriving | 11.3 | that refusal code, not `DOES-NOT-REDERIVE` | (pinned already by one main vector) |
| G30 repeated keys after unescaping | 6.13 | for `manifest_form`, `receipt_shape` and `pointer_ambiguous` | 6 |
| G10 ASCII whitespace | 7.1b, 7.4 | the six bytes only; the code points most often taken for whitespace are named as not whitespace | 9 |
| G8 narrowing at k = 0 | 7.5 | two worked examples | 2 |
| G7 detail on HELD rows | 7.1k | HELD and FAILED rows carry the same detail | (423 main vectors pin it) |
| G5, G6 codes whose detail is `{}` | 8 | the list, checked against every row of both vector files | (pinned by core digests) |
| G11 the rebuilt core | 5.4 | missing or null keys digested as `{}`, `[]`, null | (4 main vectors) |
| G12 digest comparison | 5.6 | an exact string; upper case or a non-string does not equal | (1 main vector) |
| G13 rung type | 6.3 | a non-string rung is `rung_unknown` | (1 main vector) |
| G14 an entry without sha256 | 6.5 | missing, null or non-string is `manifest_integrity` | 4 |
| G21 printed leading zeros | 7.3 | `007`, `00.5`, `-007.50`, `00,123` | 3 |
| G25 line numbers past machine range | 4.3 | no upper bound; past the end is `anchor_out_of_range` | (1 main vector) |
| G26 glue | 7.6 | ASCII only | (1 main vector) |
| G29 depth counts containers only | 3.2 | a scalar inside the 32nd container is fine | (1 main vector) |
| G27 array tokens | 6.12 | `-`, `00`, `01`, `+1`, ` 1`, `1 `, `1e0`, `0x1` | 8 |
| G23 name | 11.1 | a missing or non-string name is `name_form` | 3 |
| 13.15 the vector files' format | 13.15 | one paragraph: ops, inputs, expectations, the id rule, the pass rule | |
| 12.8 and 14 | 12.8, 14, opening | the Go build named for what it is: written from the text by a lab agent, evidence, not independent, not outside the lab | |

The fold vectors run through all three checkers. Python and JavaScript answer 62 of 62, with byte-identical canonical receipts on
the 31 checks that emit a core. The Go build answers 59 of 62. Its 3 failures are the order-of-faults vectors, which is the reading
13.20 records as its own.

## 8. lab-side bug found, and repaired

The measurer found that `check_v0.js` parsed JSON in quadratic time. In `_string`, an `indexOf` of a backslash scanned past the
closing quote, to the next backslash or to the end of the text, for every string token. Before the repair, a flat array of 400,000
short strings took 10.4 s and 400,000 nested objects took 18.5 s. After it, they take 79 ms and 121 ms (`fold/js_scale_before.txt`,
`fold/js_scale_after.txt`). 3,000,000 nested objects now parse in 1.6 s, where the measurer saw no answer within 180 s. The
repair scans once to the first quote or backslash. Over 56,190 generated cases its output is byte-identical to the output before the
repair. Both vector files and all three receipt v0 test files pass (155 tests). It is a lab bug, so it was repaired. The Go
build's depth crash is a clean-room bug, and it was left as found.

## 9. cost

The lab's first measured numbers. All of them come from one build, and all of it happened inside the lab.
- **size.** 3,088 raw lines of Go (2,835 without blank and comment lines). The checker proper is 2,331 code lines in 10 files, and the
  runner, self-test and switches are another 504. Standard library only. For comparison, `check_v0.py` is 1,843 raw lines (1,569
  code) and `check_v0.js` is 1,941 (1,793).
- **time.** Wall clock from file times: 13 min 10 s from the given files to the first full pass, and 17 min 58 s to the final run.
  This is a model agent writing in the lab's own session. It is no stand-in for a human adopter, so neither figure is a human cost. The
  author guessed 16 to 24 hours for an experienced human engineer. That is its own guess, and nothing measured it.
- **sentences.** The vectors were enough to reach agreement on every verdict-bearing input the measurer drew. They were not enough to
  pin refusal precedence or to make the build robust at extreme depth. The next implementer would have needed about 24 more
  sentences. The text now has them (§7).

## 10. what this shows and what it does not

**Shown:** an agent told to read only the text, the schema and the vector file, whose own log names no lab file, built a Go checker
in 13 min 10 s that passed 2,193 of 2,193 vectors on its first run. Over 115,276 cases it differed from the lab's two checkers only
where the text was silent (101 cases, one point, which moves no verdict) and where its own parser crashed (7 cases at depth
3,000,000). On no input did the text decide against the lab's checkers or against the build. The exercise found 24 places where the
text left an implementer to guess, one lab bug (quadratic parsing in `check_v0.js`) and one clean-room bug (the depth crash). The
text, the vectors and the JavaScript checker have been changed for each, except the clean-room bug.

**Not shown:**
- **An independent implementation.** The author is one agent of the same model family as the lab's authors. It worked in the lab's
  own session, on a disk and in a scratchpad that held other lab material, and its isolation was an instruction and its own log. A
  misreading that this family of models tends to make would be in the text, in the lab's two checkers and in this build alike, and
  their agreement cannot count against it. One shared function name and one shared comment phrase are not explained by the text.
- **Outside use.** Nobody outside the lab has implemented, emitted or checked a RECEIPT v0 object.
- **That the text is complete.** It is complete on the inputs drawn here. The generators are lab-written. The targeted set aimed at
  guesses one author wrote down, and an outside implementer would guess elsewhere.
- **Anything about a HELD verdict's truth.** Section 12 still binds: HELD says one comparison passed, never that a sentence is true.
- **Agreement across machines.** Everything ran on one Windows box, with Python 3.12, Node 24.13.0 and Go 1.25.6.
- **A human adopter's cost.** See §9.

The rules this step added (13.20) are drafts, unsigned, and wait for the operator with D1 to D3 and the rest of 13.19.
