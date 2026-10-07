# NOTE — PATH-2a, ninth pass: the `apart` switch is removed, bar C is restated, and each finding of the reviews of `e820f291`

## 0. Status

2026-10-04. Branch `fix/diffgate-abstain-where-wrong`, head `3bdc3bc4`: pass eight's `e820f291` with `origin/main`
`43b3b608` merged in. `main`'s two reader files did not move in that merge (sha256, LF: `9b620e00…` and `06688702…`).
The earlier notes are listed in the README's *PATH-2a* section; none is edited.

What this pass had to read:

- three finished reviews of `e820f291`: by construction (A), fix before shipping, one major and one minor; cross-port
  and runtime (C), fix before shipping, one blocker, one major (carried) and one minor; integration, tests and docs,
  fix before shipping, three majors and six minors (and the PR's size, carried from the eighth pass). The coverage review (B, D) of pass eight never finished;
- the lead's decision of 2026-10-04 (§1);
- two read-only probes of `3bdc3bc4`, run before this pass: a cross-port probe and a coverage and cost probe. Each
  measured a **prototype**, the head with the switch's one call site disabled in each port, and the coverage probe also
  a **deletion** of everything only the switch uses; the two gave the same records on 166,166 of 166,166 comparable
  inputs.

**This note is written before this pass's code and committed alone, ahead of it.** Unlike passes four to eight, the
builder did not prototype this pass in the working tree before writing: every figure below is a probe's, measured on
the prototype or the deletion (scratch `scratchpad/ghmerge/p2a/c9/probe_crossport` and `probe_coverage`; CPython
3.12.10 and 3.14.2, Node 24.13.0), and is named as such. §7 says what is measured again at the head that carries the
code. If the committed code departs from this note, the departure goes into a note of its own. Finding identifiers are
this note's: A-n, C-n and I-n number each lens's findings in the order its review lists them; P-n are the probes'
refutations.

Dispositions: **fixed** (a change in this pass, with a test where the finding is a defect); **closed by the removal**
(the code the finding is about no longer exists); **disclosed** (kept, and stated in the README, the CHANGELOG or
here); **operator** (left to the operator).

## 1. The lead's decision of 2026-10-04: remove the `apart` switch

Since pass five the overlay has carried a switch, C-1 `apart`: `_p2a_apart`, `_p2a_apart_in`, `_p2a_apart_diff`,
`_p2a_window`, `_p2a_open` and their tables in Python; `_p2aApart`, `_p2aApartIn`, `_p2aApartDiff`, `_p2aWindow`,
`_p2aOpen` in the port; one call site each. Where `apart` is true, a CONTRADICTED is **kept**. Its only purpose was
agreement of the two ports' gate verdicts on inputs where `main`'s two ports already give one description different
claim lists, a defect `main` has on its own (issue #181 for the description side; §5, P-1, for the rest).

What it cost:

1. Wherever it fires it keeps a CONTRADICTED the overlay knows the mechanism can have made false, and a false
   accusation is the worst error this lab's charter names. Measured: with a U+FEFF before the summary (what a summary
   file saved by Windows PowerShell 5.1 hands the CLI) the builder's 1,600-case truth world keeps 139 attributable
   false CONTRADICTEDs in each port, and with the word `styxx` and one U+2028, lone CR or form feed anywhere, all 214
   (§5, P-4); on #161's reproductions the port keeps 7 false tests CONTRADICTEDs through #101 (§5, P-2).
2. Every blocker of passes five to eight came from it (a tests claim opposite a count; Unicode case tables; a `def` at
   a line end; a regex backtrack in the scope window, C-1 below), and so did pass eight's quadratic reader (A-1).
3. Its size, counted by deleting to a fixed point: 282 lines of the Python block's 2,067 and 228 of the port's 1,426
   (comments and docstrings included), 16 plants and about 40 pins. The lead's figure of about 400 lines per port was
   an estimate (§5, P-5).

**So: the switch and everything only it uses are removed, in both ports.** No rule replaces it. The cross-port bar is
restated so that it holds by construction (§2). Where `main`'s two ports give different claim lists, the two ports'
gate verdicts may differ; that is measured and disclosed, and is not a blocker. The operator confirms the decision at
merge.

What the removal costs, as the probes measured it:

- **Right CONTRADICTEDs lost.** A CONTRADICTED the switch kept is now decided by the rules below it, and some of
  those are right: among the 26 pinned cross-port cases that move, 20 of Python's 28 changed claims are `main`'s false
  CONTRADICTEDs now withheld and 8 are right ones lost (port: 17 and 6). The plainest: "Only touches docs/." then
  U+0085 and "Thanks." over a change to `src/app.py` was CONTRADICTED, rightly, in both ports; now it is withheld
  (`extract`) in both.
- **Recall.** `main`'s committed corpora: 80 of 2,231 decided claims withheld, unchanged. #161's `path2_pairs.json`:
  190 → 200 of 530. The overlay's own pinned pairs: 76 → 90 of 155. The committed inputs: 2,135 → 2,633 of 12,718
  decided claims withheld in Python, 2,064 → 2,590 in the port, nearly all on the seeded hostile fuzz and the
  text-seam set. The EXTERNAL-1 shelf (69,109 real pull-request bodies, Python): 152 of 11,859 decided claims in reach
  withheld before and after; the switch never fired there, so on real bodies its removal moves no claim.
- **Gate verdicts between the ports** (§3, C(iii)).

## 2. Bar C, restated

Word for word, as the lead set it:

> (C) CROSS-PORT, restated: (i) wherever main's two ports give a claim the same kind, verdict and detail, the overlay
> gives it the same decision and the same phrase in both ports; (ii) on every input where main's two ports read the
> SAME claim list (same length, and the same kind, verdict and detail at each position), the two ports' final claim
> lists and gate verdicts are the same, in both strict modes; both hold because a decision reads only the claim's kind,
> verdict and detail, main's counts and the door bytes, through code that asks no runtime a Unicode question; (iii)
> where main's two lists differ, nothing is promised: the committed tests MEASURE how often that happens and how often
> the gates then differ under main and under the overlay, and the README states it.

In (ii), two final claim lists are "the same" when they have one length and, at each position, the same kind, verdict
and detail and, where the overlay wrote the reason, the same phrase and defect tag. A claim's text and `main`'s own
reason are not compared: `main`'s two ports cut and strip a claim's text differently and print some reasons
differently (#183), and the overlay copies `main`'s reason verbatim.

Bars A, B, D and E stand as they were. The sentence of passes six to eight, "without `--strict` the two ports' gate
verdicts agree wherever `main`'s do", is withdrawn (§5, P-3).

## 3. Why C(i) and C(ii) hold by construction, and what C(iii) measures

**C(i).** With the switch gone, a decision is `_p2a_decide(c, f)` and nothing else. It reads:

- the claim's kind, verdict and detail (`path`, `n`, `name`, `prefix`, `prefix2`, `declared`);
- `main`'s own reason only through `_p2a_numbers`: the count after "diff changes N files, claim says" or "diff adds N
  test functions, claim says". That number is used for one thing, a check that the overlay reproduces `main`'s count
  (`unreproduced`), and the reason is otherwise copied verbatim;
- the door's bytes, through the facts object: the diff, the `--name-status` listing at the git door, the summary.

It never reads the claim's text, `strict`, or anything of the process. The other claims of the list reach a decision
only as batching (`prime`, `tokens`): `occurs` answers "does this word lie in that text" either way, and `ends` falls
back to a tree of one claim, so a decision does not depend on its neighbours. The facts are functions of the bytes, and
each port computes them with code that asks its runtime no Unicode question: no `lower`, `upper`, `casefold`, `is*`,
`splitlines`, `normalize`, `encode`, `int()`, no strip or split without its characters and no `\w`, `\s`, `\d`, `\b` or
regex flag in Python; no `toLowerCase`, `toUpperCase`, `trim`, `normalize`, `localeCompare`, `parseInt`, `Number`, no
regex literal and no regex flag in the port; static tables and comparisons of code points or UTF-16 units instead.
The Python self-check and the port's token scan refuse each of those by name. So the same (kind, verdict, detail)
over the same bytes gives the same decision and the same phrase in both ports.

Three things in that argument are not arithmetic, and are said here so that no reader has to find them:

1. Each port builds its file registrations from its own `main`'s line split, and reads its own line view's count of
   `def test_` sites as `main`'s. Where the two splits can part on a header line, the `divergent` guard withholds every
   path, count and scope claim in both ports; tests and symbol claims read both line views in both ports.
2. Three phrases are fallbacks for a reading the overlay failed to reproduce or to make: `unreproduced`, `unparsed`
   and `error`. They depend on the port's own `main`, so C(i) is claimed for the other phrases; the tests assert that
   these three never fire on the committed inputs, and the probes saw none in either port on 50,837 inputs read for it.
3. "The same code in both ports" is a transliteration, held by tests, not by a proof: the pinned pairs, the plants that
   change one port alone, and the cross-port comparison. The cross-port probe dropped each of 17 rules or phrases from
   one port in turn; every one was flagged under C(ii) on a fixed 36,837-input slice (the weakest by 2 inputs). An
   asymmetry confined to one rare character class could still escape these sets.

**C(ii)** follows from C(i). Where `main`'s two lists are the same, each position holds one (kind, verdict, detail), so
each position gets one decision and one phrase; the final lists are then the same, and the gate verdict is `main`'s
formula over the final verdicts, with or without `--strict`. A claim reads the same under `--strict` as without it (the
relation tests assert that separately in each port).

Measured on the prototype, both probes: C(i), 0 claims decided apart among 729,596 decided (kind, verdict, detail) keys
that both `main`s give on 354,675 fuzz inputs (191,627 of them inside lists that differ) and 15,825 on the committed
inputs and generator items, on CPython 3.12; the same zero on 3.14, under the POSIX path flavour and with the port's
engine patched to fold a Unicode 17 pair. C(ii), 0 differing final lists and 0 differing gate verdicts, both strict
modes, on 252,707 fuzz inputs with equal lists and 7,808 committed ones (3.12), and on 69,038 shelf bodies. The switch
changed the outcome on 3,531 of 81,271 equal-list inputs of one slice, so the property is not vacuous there.

**C(iii)** is a measurement, not a promise. The committed tests count, per runtime: the inputs where `main`'s two
lists differ, in two classes that must not be added up under one issue (§5, P-1): the **description side** (the lists
differ in length, or in a kind or a detail at some position: the two ports read different claims from the description)
and the **diff side** (the same kinds and details, a verdict apart: the two ports decide one claim apart on the diff);
and, on those inputs, how often the two gate verdicts differ under `main` and under the overlay, in both strict modes.
The probes' figures for the prototype:

| set | inputs | lists differ (description + diff side) | gates differ without `--strict`: `main` / `3bdc3bc4` / prototype | with `--strict` |
|---|---|---|---|---|
| committed inputs, CPython 3.12 | 6,672 | 600 (480 + 120) | 19 / 19 / 23 | 20 / 13 / 13 |
| the same, CPython 3.14 | 6,672 | 599 (482 + 117) | not broken down | not broken down |
| the same, POSIX path flavour | 6,672 | 597 (480 + 117) | not broken down | not broken down |
| cross-port cases and generator items | 2,165 | 424 (25 + 399) | 127 / 0 / 162 | 103 / 4 / 4 |
| adversarial fuzz, CPython 3.12 | 354,675 | 101,951 (87,180 + 14,771) | 37,270 / 34,314 / 23,022 | 36,572 / 32,868 / 32,868 |
| EXTERNAL-1 shelf, bodies against rebuilt diffs | 69,057 | 19 (19 + 0) | 0 / 0 / 0 | 7 / 7 / 7 |

On the fuzz the removal splits 3,045 inputs where `main`'s two gates agree (1,818 description side, 1,227 diff side)
and makes the gates agree on 17,293 where `main`'s differ; `3bdc3bc4` added 0 and closed 2,956. Under `--strict` the
prototype equals `3bdc3bc4` on every set: a kept and a withheld CONTRADICTED both fail a strict gate. On the committed
inputs the 6 added splits are #161's `f2` vertical-tab, form-feed, line-separator and paragraph-separator inputs,
`f2-a-context-line-holding-a-separator-adds-nothing` and `y2-a-changed-test-beside-a-created-bom-test-under-bare-hunks`,
all on the diff side. On the shelf every differing body is Japanese or Chinese prose (Katakana, CJK ideographs or
full-width digits next to a path), which only one port reads a path claim from.

## 4. The findings of the reviews of `e820f291`

### By construction (A)

| id | severity | finding | disposition |
|---|---|---|---|
| A-1 | major | Pass eight's window reader is quadratic: `_p2a_window` scans the rest of a name run again for each `add`, `creat` or `introduc` that follows a character outside ASCII. "3 files changed. " then 16,379 times a zero-width space and `add`, 65,534 characters: CPython 84 → 2,867 ms, Node 1.2 → 3,476 ms; 262,162 characters ×189 in CPython. Records unchanged. | **closed by the removal**: the reader is gone. The probes on the same input: overlay alone 3,001 → 1.0 ms on CPython 3.12.10 (`main`'s call 61 ms), 2,831 → 0.07 ms in Node (1.2 ms); at 262,162 characters 41,072 → 3.5 ms and 44,642 → 0.12 ms. No timing pin is added for it: the review's CJK variant cannot be run in CPython at all (§5, P-6), and the code it would bound does not exist. |
| A-2 | minor | Three committed statements about that reader's cost are false: the README's range "up to about ×27 in Node and ×8 in Python", the eighth note's "each sentence is read in linear time" (§2, B-2), and the `_p2a_apart_in` docstring. | **fixed**: the docstring goes with the function; the README's cost text is written again from this head's measurements. The eighth note is not edited and is corrected here: its linear-time argument covered the search for a character the ports read apart, not the name-run scan inside `_p2a_window`, which overlapped across occurrences. |

### Cross-port and runtime (C)

| id | severity | finding | disposition |
|---|---|---|---|
| C-1 | blocker | The scope window follows the optional `files in` or `files under` group and gives up when no path run follows it, while `main`'s regex backtracks out of the group and reads `files` as the prefix. "Caféonly touches files in (docs). 4 files changed." over `files/a.py`, `other/b.py`, `.env`, `env`: `main` FAIL / FAIL, `e820f291` PASS (Python) / FAIL (port). 115 of 12,000 generated summaries of that shape. | **closed by the removal** as a defect of the window; **disclosed** as a gate difference. `main`'s two ports give that summary different claim lists (CPython reads no `\b` before `only` after `é`; the port reads an `only_touches` claim), so it falls under C(iii). With the switch gone the Python reads PASS and the port FAIL, as at `e820f291`; the input is pinned as a cross-port case with `main`'s reading beside it. |
| C-2 | major, carried | Under `--strict` the overlay still adds gate differences wherever `main`'s two ports read claim lists of different lengths: 43 to 60 per 5,000 of the review's mixed generator. | **disclosed, operator**: C(iii). The removal changes no `--strict` figure. |
| C-3 | minor | Reviewers share one scratch root, and the disk filled during the pass. | **recorded**: this pass writes only under `p2a/c9/build`, with `TEMP`, `TMP` and pytest's `--basetemp` there, runs plants one at a time, and deletes case data after summarising. |

### Integration, tests and docs

| id | severity | finding | disposition |
|---|---|---|---|
| I-1 | major | No committed test runs the relation with `run=`, `evidence=` or `commit=` at either door, and the self-check allows `c.verdict = "UNCHECKABLE"` and `c.why = ...` on any claim inside `_p2a_abstain`. A plant that withholds a `tests_pass` VERIFIED beside a count claim passed the self-check and 713 tests. | **fixed**, both halves. Tests: the relation and `strict_alike`, both strict modes, over inputs holding a `tests_pass` claim, with a stubbed run leg that exits 0, with a green report, and with a commit the report does not name, at the raw door and the git door. Self-check: in `_p2a_abstain`, `todo` is bound once, as the claims of `g.claims` whose (kind, verdict) is in reach; `hits` is bound only by a comprehension over `todo` whose element pairs that claim with a decision; neither name is read anywhere but as the thing iterated (and `todo` for emptiness); the stores into `.why` and `.verdict` are on the claim variable of the one `for` over `hits`, inside it, and that variable is not bound again. The review's plant and its variants are refused cases. The port's store scan gets the same shape rule for `_p2aAbstain`. |
| I-2 | major | The U+2028 and U+2029 starts of `_p2aOpen` / `_p2a_open` are pinned in neither port; dropping them passes every test and brings the pass-eight blocker back. | **closed by the removal**: `_p2a_open` is gone. The same starts in `_p2a_counted` and `_p2a_anchored`, which stay, are checked by planting in this pass (§6). |
| I-3 | major | The scratch PR body fails the repository's own diffgate job on this PR's diff (a quoted example sentence is read as a tests claim: "diff adds 66 test functions, claim says 1"); the job is green only because the diff is over GitHub's 20,000 lines and the job prints DID NOT RUN. | **fixed**: the diff is brought under 18,000 lines (the last row below), so the job reads the body; the body is written again with every example paraphrased, and checked with `gate_diff_text` against `git diff origin/main...HEAD` before it is handed over. |
| I-4 | minor | Commits `9eb93b04` and `838f59cf` are red on `test_port_is_current`: the instrument moved before its pins. | **fixed from this pass on**: the commit that moves `styxx/diffgate.py` also moves the port's header hash, `py_side.py`'s pin, the README's hash, `bookmarklet_src.js` and the built bookmarklet. Nothing is amended. |
| I-5 | minor | The CHANGELOG's diff-size sentence is stale and measures the working tree. | **fixed**: the figure is `git diff origin/main...HEAD \| wc -l` at the head. |
| I-6 | minor | Stale comments and one pinned id: the comment over `C1-a-unicode-16-case-pair` and `_want`'s docstring; the pair `p7-a-case-pair-outside-ascii-keeps-contradicted`, which withholds both claims; `_p2a_apart_diff`'s docstring; "the self-check's own two functions" where there are three; "CI's Node 17" for a Node carrying Unicode 17. | **fixed**: each is rewritten or goes with its code. A pinned id is not renamed; a pair whose id no longer describes it says so in a `note` field. |
| I-7 | minor | The eighth note says ×6 to ×9 for the `def` beside 50,000 letters, the README ×6 to ×7. | **fixed** in the README, per runtime, from this head's run. The eighth note's ×6 to ×9 mixed the two runtimes: ×6 to ×7 was CPython's, ×8.8 Node's. |
| I-8 | minor | The relative timing bound does not reach the case closest to its limit: on the symbols case five times `main`'s call is below the 0.5 s floor, so on CI's CPython 3.9 and 3.10 the case is bounded by wall-clock time alone, with a margin of about 1.5 to 2. | **fixed** by the review's second option: the absolute floor is 1.0 s on CPython below 3.11, which has no specialising interpreter, and 0.5 s from 3.11 on; the docstring records the measured margin. The relative bound is not made the only one: the overlay's loops are Python where `main`'s are C regex, so on 3.9 the ratio itself moves. Not measured on 3.9 or 3.10, which are not on this machine. |
| I-9 | minor | Process: three concurrent plant runs filled the disk. | **recorded**, as C-3. |
| I-6 of the eighth pass, carried | minor | The PR's diff is over GitHub's 20,000-line limit for a pull-request diff (20,040 lines at `3bdc3bc4`). | **fixed**: under 18,000 by removing what the switch alone needed (twice in the port: `bookmarklet_src.js` is its copy), writing the JSON pins and fixtures one record per line, and writing the README's section from this head's figures instead of each pass's. No test is weakened for it. |

The unfinished coverage review left one lead in its scratch, the 139 and 214 kept claims of §1; the coverage probe
reproduced and extended it (§5, P-4).

## 5. What the probes refuted, with root causes

**P-1. "Where `main`'s two lists differ, a gate difference is attributable to issue #181."** Not all of it. #181, as
filed, is "diffgate.js: the claim templates read the description with JavaScript's regex classes, not the Python's": a
description-side defect. Under C(ii)'s own definition the differing class also holds inputs where both ports extract
the same claims and decide one apart on the diff: 120 of 600 on the committed inputs, 14,771 of 101,951 on the fuzz
(symbol claims 6,751, counts 6,580, tests 644), 0 of 19 on the shelf; and 1,227 of the 3,045 gate splits the removal
adds on the fuzz. *Root cause:* `main`'s two ports read the diff apart as well: `str.splitlines()` against a split at
CR and LF only (VT, FF, U+001C to U+001E, U+0085, U+2028, U+2029); `\s` and `strip()` against the port's `\s` and
`trim()` at `def` sites and headers (U+001C to U+001F and U+0085 against U+FEFF); `^` under the port's `m` flag after
U+2028 and U+2029; `lower()` against `toLowerCase()` across Unicode versions. *What this pass does:* the committed
measurement, the README and this note report the two classes apart. The description side is #181's. The case-table
part of the diff side is #173 and #184. The line-break and white-space part of the diff side has no issue of its own
among #168 to #186; whether to file one is the lead's or the operator's call. No single number is printed under #181.

**P-2. "On #161's reproductions `apart` keeps 10 CONTRADICTEDs, none of them false through #97, #121 or #101"** (the
README, the eighth note §5, pass eight's report). True of the Python port only. The browser port kept 14, and 7 of
them are `main`'s false tests CONTRADICTEDs through #101: claim 1 of the two `r1-a-bom-on-a-changed-test-…` inputs and
of `y2`, and claim 0 of the four `f2` separator inputs. *Root cause:* the figure was measured with the Python module
only. The port's `main` counts a `def test_` after U+FEFF, VT, FF, U+2028 or U+2029 (JavaScript's white space), gives a
false CONTRADICTED the Python `main` never gives, and `apart`'s diff part (the two views count apart) then kept it. No
committed test read the port's decisions on that fixture. *Fixed* by the removal (all 7 withheld under #101); the port's
decisions on these ids are pinned, and the `r1` pair and `y2` get file models so that truth judges the port there.

**P-3. "Without `--strict` the two ports' gate verdicts agree wherever `main`'s do, by construction"** (the test
module's docstring, `GATE_SPLITS_MAIN_AGREES = set()`, the README). False once the switch is gone: 6 committed inputs,
13 pinned cross-port cases, 149 of the 600 line-break generator items, 3,045 of 354,675 fuzz inputs; every one has
differing `main` lists. *Root cause:* the property was the switch's whole purpose; it does not follow from what a
decision reads. *Fixed:* replaced by C(ii), asserted, and C(iii), measured.

**P-4. "0 attributable misses on every truth set, decorated or not"** (pass eight's report; the README). At `3bdc3bc4`
the builder's world (`families(20260930, 400)`) with U+FEFF before the summary keeps 139 attributable false
CONTRADICTEDs in each port (Python 1,036 of 1,175 withheld, port 948 of 1,087); with a sentence naming `styxx` as
well, or with `styxx` and one U+2028, lone CR or form feed anywhere, all 214; the git-built world with the BOM, raw
548 of 614, `--no-renames` 648 of 743, git door 539 of 605. `main`'s two ports read those summaries alike. *Root
cause:* two summary clauses of `apart` over-approximated: the window's "character before the match" took in a leading
U+FEFF whenever the summary opens with a claim, and the `styxx` clause fired on any divergent character or lone CR
anywhere in a summary naming the tool. The CLI reads `--summary` as utf-8, not utf-8-sig. *Fixed* by the removal: the
prototype withholds 1,175 of 1,175, 1,193 of 1,193 and 1,087 of 1,087, and every rendering of the git-built world in
full, at the undecorated world's cost in right verdicts (242 in Python, 166 in the port). Two decorated truth tests are
committed.

**P-5. "About 400 lines per port."** Counted by deletion to a fixed point: 282 and 228 block lines (§1).

**P-6. Pass eight's construction review asked for a timing pin on its input and on the CJK variant, bounded by a
multiple of `main`'s call, taking `main` to be linear there.** `main`'s own call is cubic in the length of one unbroken
run of word characters in the description, in both ports, with no overlay involved: `gate_diff_text("Fixed the parser.
Token: " + "a" * 2000 + ".", …)` takes 12.8 s on CPython 3.12.10 (1,000: 1.5 s; 500: 0.25 s), `"x" * 3200` takes 6.8 s
in Node 24, and the review's 16,379 repetitions of U+65E5 and `add` did not finish one reader in 700 CPU-seconds. The
review timed the CJK variant in Node only, where `\w` is ASCII and breaks the run. *Root cause:* `main`'s path template
`[\w./\\-]*[A-Za-z_][\w-]*\.(?:py|md|…)\b`, tried unanchored from every start inside the run by `file_created`'s
"path: new" form (and `file_touched`'s line form): two nested stars before a required dot. Outside #97, #121 and #101
and outside this PR, whose reader is `main`'s byte for byte. *What this pass does:* commits no timing case or fuzz
input holding a word run longer than a few hundred characters, says in the README that the overlay's cost is bounded
against a call of `main` that is itself unbounded on such runs, and hands the shape to the lead: it is not among #168
to #186 by title, and the Action reads pull-request bodies from forks.

Neither probe refuted C(i) or C(ii) on the prototype, so this pass adds no rule on their account.

## 6. What changes

**Removed, both ports.** Python: `_p2a_apart`, `_p2a_apart_in`, `_p2a_apart_diff`, `_p2a_window`, `_p2a_open`,
`_p2a_skip`, `_p2a_files_in`, `_p2a_name_end`; the tables `_P2A_WINDOWS`, `_P2A_KEYS`, `_P2A_TRIGGER_LOW`,
`_P2A_SPACE_RUN`, `_P2A_ASCII_WORD`, `_P2A_DIGITISH`, `_P2A_NOUNS`, `_P2A_QUOTES`, `_P2A_ACCUSE`; the facts' `kinds`,
`names` and `apart`; the two priming calls in `_p2a_abstain`; the call site in `_p2a_decide`; the three names in the
self-check's attribute list. Port: `_p2aApart`, `_p2aApartIn`, `_p2aApartDiff`, `_p2aWindow`, `_p2aOpen`,
`_p2aTriggerLow`, `_P2A_KEYS`, `_P2A_WINDOWS`, `_P2A_NOUNS`, `_P2A_ACCUSE`, `_p2aSpaceUnit`, `_P2A_ANY_SPACE_UNITS`,
the three facts members, the priming calls, the call site. **Kept**, because other rules use them: `_p2a_anchored`,
`_p2a_distinct`, `_p2a_divergent`, `_P2A_DIV_RX`, `_P2A_ONE_RX`, `_P2A_CUT`, `_P2A_SEG`, `_P2A_LEAD_APART`, the unit
predicates, and O-11's emoji tables (O-11 still decides claims through `extract`). Nothing else is renamed.

**The rules after this pass.** Every earlier note stands but for C-1: there is no `apart`. A CONTRADICTED in reach is
decided by its kind's rule, as a VERIFIED is. `case_count`, `seam`, `extract`, `divergent` and O-11 stay: each still
carries C(i) (the cross-port probe dropped each from one port and got differing lists on equal-list inputs). The
reconstruction property, REACH, the reason form, the phrases, the error fallback and the abstain-only relation are
unchanged.

**Tests.**
- C(i): as now (`cross_port`'s three by-construction keys everywhere; its two measured keys on the committed inputs
  only, where L1 and OM1 stay the two pinned false pairings).
- C(ii): over the committed inputs, the cross-port cases, the two seeded generators and a seeded world, in both strict
  modes: where `main`'s two lists are the same, the two final lists and the two gate verdicts are the same; with a
  guard that the equal side holds inputs on which the overlay moved a claim.
- C(iii): the counts of §3, measured by the test and pinned for each runtime tuple measured here (the interpreter's
  Unicode version, the path flavour, the engine's Unicode version), within stated bounds elsewhere, and printed.
- The 26 cross-port cases whose decisions move and the 14 pinned pairs become pins of the new behaviour, each with
  `main`'s reading in the record; the line-break generator's "no split" assertion becomes C(ii) plus its measured
  count; `GATE_SPLITS_MAIN_AGREES` goes.
- Plants: the 9 Python and 7 port plants anchored in removed code go. The O-11 plants stay and get pins that see them:
  a pictograph emoji directly after a claimed name whose VERIFIED O-11 keeps, and one directly before a count.
- New: the port's decisions on #161's `r1` pair, `f2` separator inputs and `y2` (P-2), with models for `r1` and `y2`;
  two decorated truth tests (P-4); the `tests_pass` relation at both doors and the self-check's refused cases (I-1).
- Mutation checks of the new tests, in scratch copies, are reported with the pass.

**Size.** `path2a_pairs.json` and the three fixtures under `tests/fixtures` are written one record per line, with the
same content (the tests compare parsed JSON; no hash of these files is pinned anywhere).

## 7. Measured on the probes' prototype, and what is measured again at the head

Bar A (probes): 0 records outside the relation in 664,664 prototype runs (two ports, two strict modes, 22 sets) and on
every set of the cross-port probe; `--strict` moved no claim; every raise is `main`'s; the error phrase never fired.
Every difference between `3bdc3bc4` and the prototype (33,914 Python and 16,952 port claims over the fuzz sets) is one
move: a CONTRADICTED the head kept becomes UNCHECKABLE with an existing phrase.

Bar B (coverage probe): the pinned truth figures do not move (Python 1,206 of 1,206 attributable withheld, 253 of
5,897 right lost; port 1,100 of 1,100 and 177 of 5,739; git door 14 of 14 and 0 of 46); the decorated worlds of P-4
are withheld in full; `JOINT_121` is unchanged (operator option O-13). The prototype withholds a superset of what
`3bdc3bc4` withholds, so it adds no miss.

Bar C: §3. Bar D: §1.

Cost (coverage probe): the committed timing cases' slowest overlay 176 → 167 ms on CPython 3.12.10; two cases move,
`q4-counts-by-digit-runs` 19.7 → 10.8 ms and `class-nbsp` 51.8 → 41.5 ms; the large summaries' counts case 137 → 64 ms
(`main`'s call 642 ms); line-heavy inputs unchanged (their cost is the two line views'). The bookmarklet: 56,575 →
about 51,800 characters.

Of the committed tests, 36 of 410 fail on the prototype and on the deletion alike, and no other: 7 byte pins of the
instrument and 29 decisions (the abstention pins of both path flavours, `GATE_SPLITS_MAIN_AGREES`, 26 cross-port
cases, the line-break generator, the plants, 14 pinned pairs).

Measured again at the head, by this pass, and written into the README with what each figure measures: the relation on
the committed inputs in both ports and at the git door; the truth figures at three doors and on the decorated worlds;
the abstention counts under both path flavours and on CPython 3.14.2; C(i), C(ii) and C(iii) on the committed inputs
and the committed generators; recall with `path2a_recall.py`; the committed timing and memory cases; the bookmarklet's
size and its equality with the port; the size of the PR's diff. Figures this pass does not measure again (the
adversarial fuzz sets and the shelf) are quoted as the probes', at `3bdc3bc4` with the call site disabled.

## 8. What this pass does not change, and what the operator decides

- **No repair.** PATH-2a never gives VERIFIED where `main` was wrong. **G-P1** is not met; that is the operator's
  decision.
- **`main`'s reader** is untouched in both ports; the reconstruction test holds.
- **Nothing committed is edited:** no receipt, certificate, sworn file, PREREG, RESULT, ANALYSIS, AMENDMENT, ERRATUM,
  earlier NOTE or the charon log.
- **Operator options** O-1 to O-15 stand as disclosed, O-13 (the five joint #121 reproductions) included, and so does
  the behaviour under `--strict`. For the operator at merge: the lead's decision of §1; the gate differences of
  C(iii); whether the diff-side divergence of `main`'s two ports gets an issue of its own (P-1); `main`'s cubic
  template on long word runs (P-6).
