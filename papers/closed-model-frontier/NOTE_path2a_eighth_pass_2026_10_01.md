# NOTE — PATH-2a, eighth pass: the review of `8eead84f`, each finding and what this pass does with it

## 0. Status

2026-10-01. Branch `fix/diffgate-abstain-where-wrong`, on `origin/main` `1cde8b82`. Pass 7 ended at `8eead84f`. Its
note is `NOTE_path2a_seventh_pass_2026_09_30.md`; the earlier ones are listed in the README's *PATH-2a* section. None
of them is edited. Four review lenses read `8eead84f`:

- by construction (A): fix before shipping. 0 records outside the relation on every input the lens ran (13,520 Python
  runs per runtime on the committed inputs, 72,920 on its own hostile fuzz, 28,426 on #161's generators with real git
  doors, 470 hostile real repositories). Two majors (the self-check's aliases; no store check in the port), three minors;
- coverage and recall (B, D): fix before shipping. Three majors (the case-pair clause fires on any two names that differ
  outside ASCII; the summary part fires on a character anywhere in the claim's sentence; #161's five joint #121
  reproductions, carried), two minors;
- cross-port and runtime (C): fix before shipping. One blocker (a `def` that ends its line splits a gate verdict where
  `main`'s agree), one major (the `--strict` splits, carried), one minor;
- integration, tests and docs: ship. One major (two timing tests bound the overlay by wall-clock time alone), five
  minors.

**This note is written before this pass's code and committed alone, ahead of it.** Each design below was prototyped in
the worktree's working tree, uncommitted, and measured there (scratch `scratchpad/ghmerge/p2a/b8`) against the
committed inputs, the reviews' reproductions, their generators and truth worlds, on CPython 3.12.10 and 3.14.2 and
Node 24.13.0. If the committed code departs from this note, the departure goes into a note of its own. Finding
identifiers are this note's: A-n, B-n, C-n and I-n number each lens's findings in the order the review lists them;
code comments and pins cite them.

Dispositions:
- **fixed**: a code, test or docs change in this pass. Where the finding is a defect of `8eead84f`, the fix comes with
  a test that fails on `8eead84f` (each was run there: a scratch tree holding `8eead84f`'s two files beside this pass's
  tests).
- **disclosed**: kept as is and stated in the README, the CHANGELOG or here.
- **operator**: an option left to the operator, with measurements where this pass has them.


## 1. The blocker

### C-1 (bar C): a `def` that ends its line splits a gate verdict where `main`'s agree

**The finding.** `main`'s symbol check is `re.search(r"^\s*(?:def|class)\s+" + NAME + r"\b", added_blob, re.M)` (the
port's: the same with the `m` flag) over the added lines joined by `\n`. Its `\s+` crosses that line break, so an added
line ending in `def` reads the name on the next added line. `apart`'s symbol part read sites one line at a time, a site
needing a run after the word on its own line, so it never saw this. With "Adds function foo. Added 0 tests." over a
changed test and the added lines `x = 1`, `def`, U+001F `foo():`, CPython's `main` reads `\s+` as `\n` U+001F and
VERIFIES the symbol, the port's `main` (whose `\s` is not U+001F) CONTRADICTS it; both CONTRADICT the tests claim, so
`main`'s gates are FAIL / FAIL. `apart` stayed false, the overlay withheld the tests CONTRADICTED in both ports, and the
gates split PASS / FAIL. U+FEFF splits the other way; `fooé` after `def` splits with no one-port space at all
(CPython's `\b`). The review's generator split 538 of 4,000 such inputs on CPython 3.12 and 287 of 2,000 on 3.14, and
the shipped bookmarklet disagreed with the CLI on them. The pass-7 README called the gate agreement "a proof, not a
measurement"; the proof's site rule was incomplete.

**Fixed, in both ports.** `_p2a_open(line)` (`_p2aOpen`): an added line that is, from its start (or, a superset, after
a U+2028 or U+2029, where the port's `^` also matches), a run of either port's white space, `def` or `class`, and a run
of either port's white space to its end. Where the claims `main` read include a symbol claim and an added line of
either line view has that shape, `apart` is true and no CONTRADICTED is withheld, in either port. The argument: a match
whose `\s+` spans the join must start at a `^` on the line holding the word, with only `\s` before the word on that
line and only `\s` after it to the line's end; either port's `\s` is inside the union the rule reads, and each line
view holds the lines one port's `main` joins. The review proposed the same shape with the *coarse* class (every code
point from 0x80 up) on both sides of the word; that also fires on a CJK line holding ` def ` in prose, which `main`'s
`^\s*` cannot reach, and a pinned pass-7 case (`B3-a-cjk-line-holding-def`) showed the loss, so this pass reads white
space only. `async` is not read: `main`'s symbol regex has none.

**Pins.** Four of the review's shapes join `XPORT_CASES` (U+001F, U+FEFF, `fooé`, and a dot-twin count beside U+001F),
each pinned with `main`'s CONTRADICTED kept in both ports and the gate FAIL; `8eead84f` fails all four. A committed
generator of the review's `genxl` shape (600 inputs, seed 8) asserts no non-strict gate split where `main` agrees, and
that `main`'s two ports decide the symbol claim apart on more than 100 of them (the shape is live). A pinned pair
(`path2a:p8-a-def-ending-its-line-keeps-contradicted`) keeps a tests CONTRADICTED beside a plain `def` / `foo():` pair,
a cost of the rule. Plants: the clause dropped in either port alone is caught.

## 2. The majors

### A-1 (bar A): the self-check's aliases

**The finding.** `selfcheck_p2a_only_abstains` treated as aliases of a record field only names bound by a plain
`name = x.<field>`. A record field reached through a `for` target (`for cl in (g.claims,): cl.pop()`, `for dd in
(c.detail,): dd["n"] = "9"` or `dd.setdefault(...)`) passed it, and the one with `pop`, planted behind `len(todo) > 400`, passed
every committed test too: the large-summary timing tests push 10,000 claims through the overlay but never compared the
record with `main`'s.

**Fixed.** `_p2a_holder(tree)` reads, from the block's source and to a fixed point, which expressions may hold one of
the record's containers (`g.claims`, a claim's `detail`) or a container that may hold one: `X.claims` and `X.detail`; a
name any binding of which in its function holds one (an assignment, an annotated or augmented one, `:=`, a `for` or
comprehension target over what holds one, `with … as`, and a parameter at any call of a block function or a facts
method); a call whose function, receiver or argument holds one, or that calls a block function that may return one;
`self._m` where something stored into it does; tuples, lists, sets, dicts, comprehensions, lambdas, conditionals and
`+` where a part does. An element of `X.claims` (a claim) or of a detail (a scalar), and `.get`/`.items` read off a
detail, hold none; a name bound to a detail is read as one (its reads are scalars); a function defined inside another
reads its outer function's names. The rules then refuse an item store, a mutating call and an augmented store on what
may hold one (an augmented store grows a list in place). It infers no type; README and docstring now say what it
checks, not that the block "can only abstain".

**Pins.** The review's three plants and six more (`:=`, a comprehension, a name bound from another, `+=`, a call's
result, a helper's parameter) are refused cases in `SELFCHECK_PLANTS`; `8eead84f`'s self-check accepts all nine. The
relation (`main` against the branch, both strict modes, and `--strict` moving nothing else) runs on the large cases in
both ports (`_large_cases()` in Python, `_large_cases(3)` in the port), and refuses the review's two plants there: the
Python's behind `len(todo) > 400`, and the port's below.

### A-2 (bar A): no store check in the port

**The finding.** The port's token scan read only for Unicode-table reads. A plant that appends a space to every
claim's text behind `todo.length > 400` passed it, `--relation` on the committed inputs, and the committed suite.

**Fixed (tests).** `js_store_problems`, part of `js_problems`, on the same dense code: every store into a member by
name (`x.y =`, `x.y += 1`, `x.y++`, `++x.y`) is `c.why = …`, `c.verdict = "UNCHECKABLE"` or
`g.verdict = (…) ? "FAIL" : "PASS"` inside `_p2aAbstain`; no `delete`; every store into a computed member and every
mutating call (`push`, `pop`, `splice`, `shift`, `unshift`, `set`, `add`, `fill`, `delete`, `clear`, `reverse`,
`copyWithin`) is on a chain from a name the block made itself (every binding of it in its top-level function, or at
the block's top level, an array literal, `new Map(…)`, `new Set(…)`, `_p2aFlags(…)`, or such a chain from such a name;
no parameter; none mentioning `.claims` or `.detail`) that then reads only `[…]` and `.get(…)`; and no value stored
that way holds `.claims` or a detail itself. `Object` and `Proxy` are refused as words. Four shapes of the block the
scan read as stores into what it did not make were rewritten without moving a decision: `_p2aEnds` keeps its two
figures in two Maps of numbers; `_p2aAutomaton` builds `back` as a fresh array; `_p2aLines` slices instead of
popping; two names the facts object reused (`words`, `runs`) were renamed in the closures that shadowed them.
`_p2aCaseCount` (B-1) reads the classes without storing into its forms. Like the Python's, the scan types nothing and
follows no value through a parameter; the Python block's self-check is the precise one, and the port's stores are
held to the shapes it allows.

**Pins.** The review's plant and nineteen more are refused cases (`test_the_store_scan_refuses_planted_stores`), and the
review's plant is refused behaviourally by the port's relation on `_large_cases(3)`. These are checks of the checker,
which lives in the test module, so they cannot fail on `8eead84f`'s block; `8eead84f`'s committed suite passed the
plant, which is the defect.

### B-1 (bar B): the case-pair clause fires on any two names that differ outside ASCII

**The finding.** Pass 7's C-1 clause made `apart` true where #WA < #A and a count claim's number lay in [#WA, #A]. That
holds for any two changed paths that differ only at code points from 0x80 up — `docs/zh/安装.md` and
`docs/zh/配置.md`, `ФабрикаЖелудей.os` and `ФабрикаЗавязей.os` — not only case pairs, and with `apart` set every #101
tests and #121 count CONTRADICTED of the gate was kept: in the review's world with two CJK docs added, 19 attributable
claims kept (1,054 of 1,073 withheld; `fcd3ce6a` withheld all). README, docstring and CHANGELOG said "differ only in
case"; the code did not test case.

**Fixed, in both ports, as the review's option (a) with the case test of option (b), more narrowly.** The clause no
longer sets `apart`. In its place `_p2a_count` withholds, ahead of the dot-twin rule and whatever its verdict, a count
claim whose number lies in [#CA, #A] (or is not ASCII digits) where #CA < #A, with a new phrase `case_count` ("two
changed paths differ only in case outside ASCII, which the Python and JavaScript readers' case tables may merge or keep
apart, so the two may count the files differently"), defect #121. #CA reads which code points some runtime's lowercase
may merge from two static tables, in both ports:
- `_P2A_LOWER_RUNS`: the 1,432 single-code-point lowercase mappings of Unicode 16.0 from 0x80 up (U+0130 and U+212A
  aside: the fold form already reads them), as 184 runs (start, end, step, delta); each cased code point's class is its
  lowercase;
- `_P2A_NEVER_RX` (`_P2A_NEVER_RANGES`): fifteen blocks of scripts and symbols with no case — U+0590 to U+109F (Hebrew
  through Myanmar), U+1100 to U+139F, U+1400 to U+1C7F, U+2E80 to U+2FFF, U+3000 to U+9FFF, U+A000 to U+A63F, U+A6A0 to
  U+A6FF, U+A800 to U+AB2F, U+ABC0 to U+D7FF, U+F900 to U+FAFF, U+FB1D to U+FDFF, U+FE70 to U+FEFE, U+FF66 to U+FFDC,
  U+20000 to U+3FFFF — and the neutral set: no runtime's lowercase maps them or maps to them;
- any other code point (unassigned in 16.0, or assigned without a mapping) may pair with any code point not in
  `_P2A_NEVER_RX`: Unicode 16 gave U+A7DC the lowercase U+019B, and CI's newer Node carries Unicode 17, whose new
  lowercase mappings include U+A7CE, U+A7D2 and U+A7D4 (the seventh integration review's reading).
Over the forms `main` keys, grouped by wild form (forms of one ASCII shape and one length), at each position from 0x80
up a code point of `_P2A_NEVER_RX` is read as itself and the others as their class, or all as one placeholder where some
form there holds a code point of neither table. #CA counts the keys that leaves. A runtime merges two forms only where
at every differing position its lowercase merges the two code points, which only code points of one class do, or a
code point of neither table does; so each runtime's count lies in [#CA, #A]. Two premises, written in the README:
Unicode's case-pair stability (two assigned code points that are not a case pair never become one, and a pair stays one)
and that the listed blocks gain no case. The tests pin both tables against the lowercase of every runtime they run:
exactly on Unicode 16 (CPython 3.14, Node 24), a subset on 15 (CPython 3.12); the port's runs and ranges equal the
Python's.

The proof of gate agreement now reads: where `apart` is false both `main`s read the same claims; their verdicts can
differ only on count claims in [#CA, #A], which are UNCHECKABLE in both ports; the overlay decides every other claim
alike. The case-pair clause of pass 7, and the facts' `ns`, are gone.

**Measured.** The review's CJK world (`cjk2`, the builder's 1,600 cases with two CJK docs added and a right count
sentence where `main` reads none): 1,073 of 1,073 attributable withheld in Python and in the port (1,054 at
`8eead84f`); without the count sentence 1,067 of 1,067; with ASCII names 1,073 of 1,073. The pinned truth world: +34
withheld in each port (26 false verdicts `main` gives by merging the two paths' case, outside the three defects — 13
CONTRADICTED, 13 VERIFIED; 4 right CONTRADICTEDs; and the CONTRADICTEDs the clause kept now decided: 2 false counts as
`count`, 2 right scopes as `shape`), no attributable verdict moved. On the committed inputs `case_count` withholds 42
claims: 22 CONTRADICTEDs the clause kept and 13 VERIFIEDs (one `count` before) on the seeded PATH-2a fuzz, #161's six
k2 reproductions (Unicode 14 and 16 case pairs, 2 CONTRADICTED and 4 VERIFIED, kept before), and the pass-7 pair.

**Pins.** `XPORT_CASES`: two CJK names, two Cyrillic names, two accented letters that are not a case pair (each
withholding the tests claim and keeping the right count, both ports), a Latin case pair (count `case_count`, tests
`tests`), and the Unicode 16 pair of pass 7, now `case_count` and `tests` in both ports with gate PASS (it pinned "the
overlay moves nothing" before); `8eead84f` fails all five. A truth test runs the review's `cjk2` world and fails on
`8eead84f` (19 kept). Two pinned pairs; the table tests; plants: the clause dropped, the classes read as one
placeholder, the blocks with no case read as cased (Python), the clause dropped and the classes never merged (port).

### B-2 (bar B): the summary part fires on a character anywhere in the claim's sentence

**The finding.** `_p2a_apart_in` fired on a wordish or divergent code point anywhere in a sentence that held a
template's words in order, so "Added 1 test for naïve inputs.", "3 files changed (café config).", "🆕 Added 1 test." or
"1️⃣ Added 1 test." kept every #121 count and #101 tests CONTRADICTED of the summary, though the character lies outside
any span the template can read (the count template's match ends at `changed`). In the builder's world with "(naïve)"
after each test or file-count word, 980 of 1,193 attributable claims were withheld; in the review's git-built world
496 of 619 at the raw door.

**Fixed, in both ports.** `_p2a_window(s, low, kind, i, n, a, e)` (`_p2aWindow`): for each occurrence of a template's
leading word in the sentence, the span a match there can cover in either port, or none where neither port's template
can match there, read as runs of the union of both ports' classes (either port's white space; ASCII or wordish for `\w`
and `\d`; ASCII path characters or wordish for a path) with the literal words in the lowered summary:
- count (`file`): back over white space, then over a digit run, and the character before it (the `\b`); forward over an
  optional `s`, white space, an optional `were` and white space, to the end of `changed`;
- tests (`add`, `creat`): the character before (the `\b`), the verb's word run, white space, a digit run, white space,
  an optional `new`, `test(s)`, the character after (the `\b`), and an optional noun after white space (`case`, `file`,
  `scenario`, `suite`, `class`, `function`, `method`) and three characters past its stem;
- symbol (`add`, `introduc`): the character before, the verb's run, up to two articles, the kind, white space, an
  optional `named`/`called`, an optional quote, the name run and the character after;
- scope (`only`): the character before, white space, `touch`/`modif`/`chang` and their run, white space, an optional
  `files? in|under`, an optional quote, the prefix run and the character after, and the optional `, and …` tail to the
  second prefix run and two characters past it.
A sentence is apart where some window holds such a code point; a DECLARE-1 key keeps the whole-sentence rule (a key's
line can be read or not on a leading white space only one port strips). An occurrence after an ASCII word character is
passed over (no `\b` there in either port). Windows of one word are read in order, the earliest such character at or after
a window's start found once for all of them, so each sentence is read in linear time. The argument: every character a
match of either port reads, and the `\b` character before it and the one after its last word, lies in the window of its
leading word's occurrence; outside every window both ports read the same characters, and a sentence both ports cut
holds each port's sentence.

**Measured.** The builder's world with "(naïve)" inside each claim's sentence: 1,193 of 1,193 attributable withheld in
Python and the port (980 at `8eead84f`); with the three decorations of pass 7 appended, 1,193 of 1,193 each; right
verdicts lost 242 of 5,842 under each (236 at `8eead84f`; the 6 are B-1's). The review's git-built world with the
in-sentence decoration: raw 619 of 619, `--no-renames` 748 of 748, `-U0` 619 of 619, `--no-prefix` 620 of 620, git door
610 of 610, count misses by name-status lines 0 of 181 (`8eead84f`: 496, 564, 496, 497, 487; 46 of 181). The port's
world: 246 count and 79 tests CONTRADICTEDs withheld with a decoration or without (227 and 79 at `8eead84f`). The
review's probes (`probe2`, `probe4`: 24 and 10 shapes, both ports): every one withheld but a CJK word abutting the
count's number, where `main`'s two ports read the count apart (PASS / FAIL), which stays as `main` has it.

**Pins.** `XPORT_CASES`: the review's seven sentences (naïve, José's, a CJK word, "(café config)", "thanks to José",
U+1F195, the keycap) withheld in both ports, `8eead84f` keeping each; three in-window cases kept in both ports (a name and
a prefix running into an accent, a test count only the port reads); the character before the verb and one inside the
noun (plant catchers); and the two pass-7 cases whose character now lies outside the window (an accented name before
`added`, an emoji after `test`), moved from kept to withheld. A truth test runs the review's `insent` world and fails on
`8eead84f` (213 kept). Two pinned pairs. Plants: the count window dropped, the window without its leading character, the
tests window without its noun (Python), the count window dropped and the window without its leading character (port).
O-11 is now caught by a pinned emoji right after a claimed name, inside the window.

### B-3 (bar B): #161's five joint #121 reproductions (carried)

**Disposition: operator, disclosed, unchanged.** They keep `main`'s false count CONTRADICTED, pinned, both ports
(`JOINT_121`); under the committed but-for attribution they are not misses, and #161 files them as #121 reproductions.
Operator option O-13 (withhold a CONTRADICTED count where a dot twin exists, `main`'s count is below the claimed number,
and the diff holds a change marker `main` registers no file for) is not taken and not measured here. The PR body says
bar B is read but-for and names the five.

### C-2 (bar C): the `--strict` gate splits (carried)

**Disposition: operator, disclosed with per-family figures.** Under `--strict` a claim one port's `main` reads alone can
be withheld on that port only. An abstain-only rule that must read the same with `--strict` as without it cannot
balance these. The README gives the measured rate per set next to the non-strict claim (§5).

### I-1 (integration): two timing tests bound the overlay by wall-clock time alone

**Fixed.** `test_cost_per_call_python` and `test_cost_per_call_port` bound each case's overlay within the larger of the
absolute figure (0.5 s, 300 ms) and `OVERLAY_TIMES` (5) times `main`'s own call on the same input, the least of three
each, as the large-summary tests are bounded relative to `main`.

## 3. The minors

| id | finding | disposition |
|---|---|---|
| A-3 | One added line of NEL/VT-separated `def test_` sites: ×19.8 to ×27 in Node (whole calls, up to 9.6 MB), above the README's ×17; Python ×4.2. Linear. | disclosed: the README gives ×27 as the observed Node worst case. The second line view is still built eagerly. |
| A-4 | A str-like diff (`collections.UserString`) that `main` reads normally makes the overlay fail: every decided claim in reach abstains with the `error` phrase. Permitted (an abstention). | disclosed: the overlay reads `str` only; a str-like diff abstains with the error phrase. The door hooks are `main`'s two lines and are not widened. |
| A-5, process | Reviewers share one scratch root; C: ran at 94 to 421 MB free; parallel `py -3.12` runs importing numpy failed with OpenBLAS allocation errors until `OPENBLAS_NUM_THREADS=1`. | recorded. This pass worked in `p2a/b8` only, with `TEMP`, `TMP`, pytest's `--basetemp` there and `OPENBLAS_NUM_THREADS=1`, deleting case data after summarising. Pruning `review*` bulk is the operator's call. |
| B-4 | The README, docstring and CHANGELOG said "differ only in case outside ASCII"; the code tested "outside ASCII". | fixed by B-1: the code now tests case, through the static tables, and the README describes the tables and their premises. |
| B-5 | Recall at `8eead84f`: committed corpora 80 of 2,231; on the EXTERNAL-1 shelf 152 of 11,859 decided claims in reach withheld (1.3%), 115 of 115 attributable false verdicts withheld, 2 of 6,793 right lost; `shape` and `dot_earliest` never catch a false verdict in the builder's world. | disclosed: the shelf figures are the review's, at `8eead84f`, quoted as such; this pass did not re-read the shelf. O-6 and O-12 stand. |
| C-3 | The runtime-skew pin patches only the port toward a newer engine; the pairings CI can hold (CPython 3.9/3.10 on Unicode 13 with the runner's Node; CPython 3.12 with a Node on 15.1) are not pinned. The review emulated them: 0 splits. | disclosed, not added: B-1 reads a runtime's merges against Unicode 16's table, of which every older runtime's mappings are a subset (pinned on 15), so the range [#CA, #A] holds every older runtime's count by construction. |
| I-2 | The pass-7 CHANGELOG bullet gave "171 to 791 per 20,000 fuzz inputs" for sets with 34, 2 and 0. | fixed in that bullet: "between 0 and 791 per set (the README lists each)". |
| I-3 | The 3.14.2 stand-in runner turned `parametrize` into a no-op; no receipt backed "the pinned figures hold on 3.14.2". | fixed: the three PATH-2a modules run under real pytest 9.0.3 on CPython 3.14.2 (3.12's pure-Python packages on `PYTHONPATH`, `styxx` registered as a bare package by a `-p` plugin, since numpy is absent); the receipt is cited in the README and CHANGELOG. The stand-in runner is named for what it skipped. |
| I-4 | The large-summary ratios disagree: the test docstring said 0.22 to 0.40 (3.14.2) and 0.27 to 0.56 (3.12.10), the NOTE, README and CHANGELOG 0.23 to 0.49. | fixed: one measured range per runtime, with its run, in the docstring, README and CHANGELOG; the seventh pass's NOTE is not edited and is corrected here: its "0.23 to 0.49" was one run of several. |
| I-5 | `-c core.longpaths=true` reached only the commands that build the test repositories; `main`'s `_git` read them without it and failed under a long `--basetemp` on Windows. | fixed: `git config core.longpaths true` is written into each test repository after `git init`, so `main`'s own `_git` reads it. |
| I-6 | The PR's diff is 18,018 lines; GitHub's diff media type refuses a pull-request diff above 20,000 lines, and the repository's diffgate job then prints DID NOT RUN and exits 0 without `--strict`. | recorded; the diff at this pass's head is measured in §5. No code change: check that the job shows a verdict once the PR is opened. |
| own | This pass ran `sed -i` twice against the task's rule: once on `styxx/diffgate.py` (removing `"ns"` from the allowed attributes; a byte comparison with `8eead84f` showed no other change), once on a scratch draft of this note (rebuilt without it and compared equal). Some editor writes decoded `\u` escapes into literal characters in the blocks and the tests; each block was re-escaped and checked to be ASCII before any commit. | recorded. |

## 4. The rules after this pass

Everything in the earlier notes stands, with these changes.
- **C-1, the diff part:** no count clause; a symbol claim also makes `apart` true beside an added line of either view
  that is white space, `def` or `class`, and white space to its end.
- **C-1, the summary part:** a wordish or divergent code point inside the window of a template's match, or a DECLARE-1
  key and such a code point anywhere in its sentence.
- **Counts:** a count claim in [#CA, #A] where #CA < #A is withheld in both ports whatever its verdict (`case_count`),
  ahead of every other count rule but `divergent`.
- **The self-checks:** the Python's holder analysis; the port's store scan.

The reconstruction property, REACH, the reason form, the other phrases, the error fallback and the abstain-only relation
are unchanged.

## 5. Measured on the prototype, and predictions for the head that carries this pass

Measured (CPython 3.12.10, Node 24.13.0), `main` against `8eead84f` and the prototype, in Python and in the port, both
strict modes (scratch `b8/meas8_fuzz.log`):

| set | inputs | outside the relation | claim splits | non-strict gate splits where `main` agrees (`8eead84f` → prototype) | `--strict` gate splits (`8eead84f` → prototype) |
|---|---|---|---|---|---|
| the sixth review's hostile `gen5`, seeds 11 and 12 | 40,000 | 0 | 0 | 0 → 0 | 171 → 174 and 195 → 199 |
| the mixed count, tests, symbol and scope seam set, seed 7 | 20,000 | 0 | 0 | 0 → 0 | 741 → 741 |
| pass 5's count-seam set, seed 5 | 20,000 | 0 | 0 | 0 → 0 | 791 → 791 |
| the sixth review's `hx`, seed 71 | 12,000 | 0 | 0 | 0 → 0 | 2 → 2 |
| the sixth review's `gen7`, seeds 11 to 14 | 20,000 | 0 | 0 | 0 → 0 | 34 → 34 |
| `gen8` (case pairs), CPython 3.12 against Node 24 | 10,000 | 0 | 0 | 0 → 0 | 0 → 0 |
| the same, the port patched to fold U+A7CE and U+A7CF | 10,000 | 0 | 0 | 0 → 0 | 0 → 0 |
| the seventh review's `genxl` (a `def` ending its line), seed 1 | 4,000 | 0 | 0 | **572 → 0** | 0 → 0 |
| the seventh review's `genx` (case skew in five families, emoji windows, DECLARE-1), seed 11 | 9,000 | 0 | 0 | 0 → 0 | 4 → 4 |
| the same, the port patched to fold U+A7CE and U+A7CF | 9,000 | 0 | 0 | 0 → 0 | 4 → 4 |
| the builder's world with five decorations | 8,000 | 0 | 0 | 0 → 0 | 0 → 0 |
| the seventh review's hostile `fuzz7`, seeds 72 to 74 | 9,000 | 0 | 0 | 0 → 0 | 27 → 27 |

The error phrase never fired; every raise is `main`'s. The minified bookmarklet, rebuilt (terser 5.46.0, 56,575
characters, `f873dc6a…`), equals this port on 18,000 runs of the hostile set in a stub page (940 carrying an overlay
reason). The committed inputs (6,672 with this pass's five pinned pairs): 0 outside the relation in Python and the port,
both strict modes; 0 claim splits; 0 non-strict gate splits where `main` agrees; under `--strict` the two pinned inputs
of the sixth pass's C-4 and no other. The three PATH-2a modules pass on CPython 3.12.10 and, under real pytest, on
3.14.2 (409 passed, 1 skipped: terser is not on its path).

Coverage (B): §2 above; every truth world reads 0 attributable misses at every door and in the port. #161's
reproductions: the 10 CONTRADICTEDs `apart` keeps are the seventh pass's 10, none attributable; the review's fixture
reading (`repro7`) keeps 7 false verdicts at the raw door and 3 at the git door, all case-only, rename-away or a difflib
no-prefix count, none of the three defects, as at `8eead84f`. Recall (D): `main`'s committed corpora 80 of 2,231 (3.6%),
both path flavours, as before; #161's `path2_pairs.json` 190 of 530 (184 at `8eead84f`: `case_count` withholds six k2
claims there, Unicode 14 and 16 case pairs), so 270 of 2,761 (9.8%) with `main`'s corpora (#161's head withheld 493 of
2,749); the overlay's own 136 pinned pairs 76 of 155. Cost: the committed timing cases' slowest overlay alone 0.157 s on CPython 3.12.10 (`main`'s call 0.097 s) and 0.137 s on 3.14.2, 59 ms in Node, peak memory 14.2 MB; against `main`'s own call the overlay reaches ×32 on a run of 40,000 `./` and ×6 to ×9 on a `def` beside 50,000 letters outside ASCII, where `main`'s call takes a millisecond and the absolute floor binds, and at most ×4 elsewhere; the large summaries 0.21 to 0.47 of `main`'s call on 3.12.10, 0.19 to 0.32 on 3.14.2, and 1.01 to 1.43 in Node at three times the size. The PR's diff against `1cde8b82` was 19,610
lines in the working tree before this note and the last docs, so about 20,000 at this head (I-6): above GitHub's 20,000-line limit for a pull-request diff, the repository's diffgate job prints
DID NOT RUN and exits 0 without `--strict`.

Predictions for the head:
- **Relation.** No committed input's record breaks the relation (A) in either port, at either door, in either strict
  mode; none on the sets above; none on the large summaries.
- **Cross-port.** 0 splits under every by-construction key; 0 non-strict gate splits where `main`'s gates agree on the
  committed inputs, on every set above, on the newer-engine test and on the line-break generator; under `--strict`, the
  two committed inputs of the sixth pass's C-4 and no other.
- **Truth.** 0 misses at each door and in the port on the builder's world, the pass-5 and pass-6 fixtures, the review's
  git-built world and the `cjk2` and `insent` worlds.
- **Recall.** `main`'s committed corpora: 80 of 2,231.
- **Cost.** Every committed timing and memory case under its bound.

## 6. What this pass does not change, and what the operator decides

- **No repair.** PATH-2a never gives VERIFIED where `main` was wrong.
- **G-P1** is not met, and that is the operator's decision.
- **`main`'s reader** is untouched in both ports; the reconstruction test holds.
- **Nothing committed is edited:** no receipt, certificate, sworn file, PREREG, RESULT, ANALYSIS, AMENDMENT, ERRATUM,
  earlier NOTE or the charon log.
- **Operator options.** O-1 to O-15 stand. Open for the operator: the `--strict` splits (C-2), O-7 and O-13 (B-3), the
  rename and `-U0` gaps, whether the Action should import PyPI's styxx, and the PR's size against GitHub's diff limit
  (I-6).
