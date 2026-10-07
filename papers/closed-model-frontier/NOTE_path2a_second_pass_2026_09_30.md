# NOTE — PATH-2a, second pass: the review of `e12214e5`, each finding and what this pass does with it

## 0. Status

2026-09-30. Branch `fix/diffgate-abstain-where-wrong`, on `origin/main` `1cde8b82`. Pass 1 ended at `e12214e5`
and its design note is `papers/closed-model-frontier/NOTE_path2a_abstain_overlay_2026_09_30.md` (not edited). Four
review lenses read `e12214e5`: by construction (A), coverage and recall (B, D), cross-port and runtime (C), and
integration, tests and docs. None found a record that reads worse than `main`'s under the abstain-only relation; two
found blockers, and five majors were raised.

**This note is written before this pass's code and committed alone, ahead of it.** It records every finding, the
disposition, and the design of each fix. Measurements are taken on the commits that follow and go into
`web/gate/README.md` and the CHANGELOG entry. A correction to this note goes in the next one.

Dispositions: **fixed** (code or docs change in this pass, with a test that fails on `e12214e5` where the finding is a
defect of `e12214e5`), **disclosed** (kept as is and stated in the README, the CHANGELOG or here), **operator** (an
option left to the operator, measured).

## 1. The findings

### 1.1 Blockers

**C-1 (ci): the pinned abstention counts are the Windows reading.** `main`'s `find_path` compares
`Path(p).name`, and `c:x.py` has the name `x.py` under `PureWindowsPath` and `c:x.py` under `PurePosixPath`. Three
committed inputs decide differently on `main` by path flavour, so `test_abstain_only_python` fails on CI's ubuntu
runners (`decided 11103 != 11106`, and three `odd` counts). **Fixed.** The pin is keyed on the path flavour the running
`main` has, probed by behaviour (`main`'s `Path("c:x.py").name`), and both figures are measured and committed. A second
test runs the same count under the other flavour (both modules' `Path` patched to the other pure flavour), so both
pins are exercised on every runner, and fails on `e12214e5`, which has only the Windows figure. The README's recall row
for `path2a_pairs.json` gives both figures. A pinned pair with a flavour-independent `odd` shape (a final `/.`
segment) is added, so the `odd` rule is calibrated on POSIX too, where the drive-like pair calibrates nothing.

**C-2 (bar C): the overlay adds Python/port disagreements where `main`'s two ports agree on a claim's kind, verdict
and text but not on its reason or detail.** 17 claims over 16 committed inputs (12 `tests_added`, 5 path claims), and
98 over 96 fuzz inputs; the committed (C) test keyed agreement on the whole record and could not see them. **Fixed**,
by two guards that read only what both ports share, checked ahead of every other rule so that both ports name the same key:

- *`tests_added` ("split", #101).* `main`'s count is `^\s*def test_` over its own added lines, and the two ports' `\s`
  and line breaks differ on the divergent characters, so `got` can differ while the verdict word agrees (both
  CONTRADICTED). The overlay's pairing count `most` (counted sites whose name the removed lines also define, the larger
  over the two line views) is the same in both ports. When `most >= 1` and either the two views' added lines differ or
  an added line holding `def test_` holds a divergent character, the claim abstains with a new phrase, in both ports.
  Otherwise the two ports' `main` read the same `got` (the added blobs are equal and `\s` agrees on every character in
  them; `^` differs only after U+2028 and U+2029, which split the views), so the interval rule decides alike. This
  also corrects the reason on pass 1's r1-bom reproduction, where Python's `main` never counted the site the
  overlay called counted.
- *Paths, names and prefixes ("extract").* `main`'s templates read `\w` as Unicode in Python and as ASCII in the port,
  so "Modified é/x.md" is `é/x.md` in Python and `/x.md` in the port. When the two extractions differ, the Python one
  holds a code point at or above 0x80, or the port's is touched by one in the claim text. The guard: a path claim or a
  VERIFIED symbol claim abstains when its path or name holds a code point at or above 0x80 or some occurrence of it in
  the claim text is immediately preceded or followed by one; an `only_touches` claim abstains when the claim text, from
  the earliest occurrence of its leading prefix onward, holds a code point at or above 0x80 or a divergent character (the
  second prefix can be present in one port only, and its `\s+and\s+` can part on the divergent characters). Both
  inputs are shared when the details agree, so the guard then decides alike; when they differ it fires in both.
- *The (C) test.* `test_cross_port_decisions` gains an assertion keyed on (kind, verdict, text): wherever `main`'s two
  ports agree on those, the overlay's verdicts and phrase keys agree. Expected: 0 on the committed inputs (it is 17 on
  `e12214e5`, so the test fails there), and measured on the reviewer's 26,000-case cross-port fuzz and a fresh one.
- The pass-1 note's sentence "those records differ anyway" (section 4) was wrong for these claims: `main`'s verdicts
  agreed. It is corrected here.

### 1.2 Majors

**A-1 / C-3: the #101 scans are quadratic in both ports.** `_p2a_def_names` collected a name at every position of a
coarse run after `def` or `class`, and a run of code points at or above 0x80 is both coarse and name characters, so each
position rescanned the rest of the run; nothing was memoised across claims. A 12,000-character run took 25 s in Python
where `main` takes 4 ms. **Fixed**, with the same decisions:

- *`tests_added` reads only the ASCII run of each name.* Pass 1 paired a counted added site with a removed
  definition when their sets {full run, ASCII run} met. The full run determines the ASCII run, and a full run equal to
  an ASCII run is all ASCII, so the two sets meet exactly when the ASCII runs are equal. Every name that can start with
  `test_` starts at the non-coarse character that ends the coarse run after a `def`, so each site contributes one ASCII run, found by one
  scan. The runs of distinct sites never overlap, so the whole line is read once.
- *`symbol_added` checks the claimed name at its possible starts* (the positions of the coarse run after `def` or
  `class`, and the character that ends it) with the same end conditions as a full run or an ASCII run, instead of
  materialising every suffix.
- *Memoised per diff:* the removed-name set, the pairing count `most`, and each symbol lookup.
- *The port's `_p2aCounted`* no longer walks back and copies a slice per `def test_` site: one forward pass per line
  keeps the earliest non-coarse character after the last U+2028 or U+2029. Both ports use regular expressions over explicit
  code-point ranges for the coarse and ASCII runs (UTF-16 units in the port, which gives the same runs: every unit of a
  surrogate pair is at or above 0x80, and no boundary falls inside a pair).
- *Timing guards* (tests): a 5,000- and a 50,000-code-point run after `def ` in removed lines, with a symbol and a tests
  claim, in both ports, each gate call under 1 s; one 280 KB added line of `def test_a ` repeated, in the port, under
  1 s. On `e12214e5` the 5,000 case takes seconds in Python.

**A-2: the case-doubt rule redid every registered path's forms for every path claim.** **Fixed.** Per diff and per
space (A, K) the overlay memoises each registration's fold and wild forms, the placeholder groups (U2's `keys` and
`sts`), and groups by base name. A tier between two strings is defined only when their base names are equal (unless the
claim's base name is empty; then every registration is read, as before), and the wild form of a string keeps its `/`
positions, so U1, U2 and both resolvers read only the claim's base-name group, in `main`'s order; the fidelity and V97
resolvers keep `main`'s earliest-entry semantics within it. The `odd` scan over registrations, and the count rule's
twin test and bounds, are memoised too; they do not depend on the claim. Timing guard: 2,000 files and 200 path claims,
the overlay's cost at most `main`'s own time plus 0.5 s in Python and plus 0.3 s in the port (on `e12214e5`: 9.35 s
against `main`'s 0.77 s).

**Integration-1 (B): coverage was only ever asserted with `strict` off.** A planted `and not strict` on the overlay's
REACH filter passed every committed test in both ports. **Fixed.** The Python relation test and `check_path2a.js
--relation` assert that each claim's (verdict, reason) under `strict` equals the same input's under `strict` off (only
the gate verdict may differ). The two plants are committed as tests that must be refused.

**Integration-2 (A): the git door was tested on 7 hand-built repositories, none with a rename, and the 96 reproductions
carrying `--name-status` were never read at that door.** **Fixed.** A new test drives `gate_diff` from each
fixture case's own `name_status` and `diff` (the module's `_git` replaced, in `main` reconstructed, the branch and the
four variants): the relation in both strict modes, the lockstep of the overlay's `--name-status` mirror against
`main`'s four-line loop, and truth coverage at the git door, with the counts pinned. `GIT_CASES` gains a rename, a copy
and a type change built by real git. The reviewer's plant (`parts[-1]` read as `parts[1]`, the old side of a rename) is
committed as a test that must be refused.

**Integration-3 (docs): the recall figures in the README and CHANGELOG were not what the committed script printed.**
**Fixed.** `path2a_recall.py` prints the total over `main`'s committed corpora and, separately, the overlay's own pins
and any extra files.

**Integration-4 (docs): the documented port differential reported 2 disagreements.** `path2a_pairs.json` had been
added to `py_side.CORPORA` and `js_side.js`, and two of its pairs are inputs `main`'s own ports read differently.
**Fixed.** It is removed from both lists; it stays checked by `check_pairs.js`, `test_port_is_current.py` and the PATH-2a
modules, and the differential over `main`'s corpora reads 0 disagreements again.

### 1.3 Minors

| id | finding | disposition |
|---|---|---|
| A-3 | the port's `_p2aCounted` is quadratic in one long line | fixed with A-1 |
| A-4 | the README's per-call cost gives two corpus means and calls one "noise" | fixed: per-call means on the PATH-2 and fuzz sets, the timing guards as the stated bound, "noise" dropped |
| A-5 | the Action's job-summary table cuts every reason at 100 characters, so an overlay reason never shows `main`'s reading | fixed: a reason the overlay wrote is shown whole in the table cell (`diffgate_action.py`, one line), with a test |
| B-1 | coverage at the git door is not judged by truth | fixed with Integration-2 |
| B-2 | the port's coverage is asserted only where the port's `main` reads the claim as Python's does (942), not in the port's own terms | fixed: the four variants are also built from the reconstructed port (anchored edits of `findPath`, `_norm`, the tests count and the symbol hit), and every claim attributable in the port's own terms must be UNCHECKABLE in the new port; the count is pinned |
| B-3 | the `tests` trigger over-fires: a same-named test in another class or module beside a changed one, and non-ASCII names paired through their ASCII run | disclosed (section 3, O-3); per-key precision (fired, false, right, undecided) goes into the README |
| B-4 | `dot` fires when only V121 alone disagrees, although V97 and V97+V121 both verify; its reason then says the dot-kept reading differs, which is untrue once the closest match is taken | the rule is kept (the committed definition of "because of" counts V121 alone, and dropping it turns #161's R14.3 into a miss); the reason is fixed: this shape gets its own key `dot_earliest`, "with leading dots kept, the earliest changed path matching the claim reads otherwise, though the closest one does not"; the cost is an operator option (O-6) |
| B-5 | `odd` applies to the whole diff and fires on absolute drive paths (`C:/work/x.py`) where every reading of the base name agrees | fixed in part: `odd` no longer fires on a drive letter followed by `/` unless the path is that root (checked over 351,288 strings on CPython 3.12 and 3.14: wherever a flavour's `.name` differs from the base name, the new test still fires); it is still whole-diff, not scoped per claim (disclosed) |
| B-6 | in joint shapes (a dot twin plus another count defect) the count rule can withdraw `main`'s right accusation and keep its false one | disclosed (section 3); the operator option to also abstain on CONTRADICTED counts above `main`'s count is O-7 |
| B-7 | the recall script's TOTAL included the overlay's own pins | fixed with Integration-3 |
| B-8 | the one real-world #97 record in `main`'s corpora (`corpus_real` pr98, "integrations/git/README.md — created.") is left exactly as `main` has it | disclosed: neither record is a false decided verdict (the created claim is UNCHECKABLE with a false reason; the touched claim is VERIFIED on the wrong file). Every abstention on `main`'s corpora is synthetic. pr98 is the record option R (O-2) and G-P1 would change |
| C-4 | the AST self-check and the port's token scan accept planted runtime Unicode queries (`repr`, `!r`, `int` of a Unicode digit, `getattr` of a built name; computed member access in the port) | fixed: Python bans `repr`, `ascii`, `format`, `getattr`, `eval`, `exec`, `compile`, `globals`, `locals`, `__import__`, `float`, `complex`, `.format`, `.format_map`, `.encode`, `.decode`, f-string `!r`/`!a`, `%r`/`%a` formatting, and `int()` except on a regex group (the self-check's own code is exempt from the name bans, not from the store rules); the port bans `prototype`, `.call(`, `.apply(`, `Reflect`, `globalThis` and computed member access with a string literal in the brackets; each plant is a test |
| C-5 | the rebuilt bookmarklet's panel still says it is the 7.48.0 port and points to `pip install styxx`, which has no overlay | disclosed: `bookmarklet_ui.js` is not edited in this pass (the pass-1 design kept it out of scope); the README and CHANGELOG say that the bookmarklet runs `main`'s reader plus PATH-2a while its panel text names 7.48.0, and that the CLI from PyPI will disagree on every PATH-2a abstention until a release carries it |
| I-5 | the README pairs "the file 7.48.0 ships" with the branch's sha, and says `py_side.py --installed` against 7.48.0 should print 0 disagreements | fixed: 7.48.0 stays attached to `9b620e00`; against a release without PATH-2a, the expected disagreements are the overlay's abstentions, which `path2a_recall.py` counts |
| I-6 | figures marked *pinned* in the README (18,133; 11,170 port runs) are not pinned, and depend on the runtimes' Unicode tables | fixed: *pinned* is kept only on figures a test asserts exactly; the others say "measured on CPython 3.12.10 and Node 24.13.0" |
| I-7 | the README's "Running it" block runs pytest from the wrong directory | fixed |
| I-8 | the CHANGELOG coverage line reads as 1,206 in the port | fixed: Python and the port are stated separately |
| I-9 | the pass-1 note predicted the bookmarklet would grow by about 8 KB | recorded: it grew by 9,950 characters (24,335 to 34,285); this pass's size goes in the README |
| I-10 | CI runs CPython 3.9 to 3.11, which were not run here | disclosed: no such interpreter is available on this machine; the pins that depend on the interpreter are listed in the README, and the CI jobs on the pull request are the check |

## 2. The rules after this pass

Everything in section 3 of the pass-1 note stands, with these changes:

- **Order.** Path claims: `extract`, then `divergent`, `odd`, `case`, `unreproduced`, the three readers. `only_touches`:
  `extract`, then as before. `symbol_added`: `extract`, then the symbol rule. `tests_added`: `unparsed`; no pairing keeps;
  `split`; the interval.
- **New keys and phrases.**

| key | phrase |
|---|---|
| extract | the claim's path or name touches a character outside ASCII, where the Python and JavaScript readers extract it differently |
| split | a test the added lines count is also defined in the removed lines, and the Python and JavaScript readers split or space these lines differently |
| dot_earliest | with leading dots kept, the earliest changed path matching the claim reads otherwise, though the closest one does not |

- **`odd`**: with `q` the path without trailing `/`, true when `q` is `.`, ends with `/.`, or its second code point is `:`
  and it has no third code point or the third is not `/`.
- **`dot` / `dot_earliest`**: when V97 verifies and V121 does not, the key is `dot_earliest` if V97+V121 verifies, else
  `dot`.

The reconstruction property, REACH, the reason form, the error fallback and the abstain-only relation are unchanged.

## 3. Coverage and cost, disclosed

**Coverage gaps, added to the pass-1 note's list:** none are expected from this pass's changes, which only add
abstentions (`extract`, `split`) or change a reason (`dot_earliest`) or narrow `odd` to the strings where a base name
can read differently. The truth test's pin moves; any miss prints its shape.

**Costs, added:**
- `extract` withholds every VERIFIED path and symbol claim whose path or name holds a code point at or above 0x80, and
  every decided `only_touches` claim whose text holds one after its prefix (an em dash after "Only touches docs/" is
  enough). On those, `main`'s two ports read the claim differently; the overlay cannot read it one way for both.
- `split` withholds a tests claim whenever a counted test pairs with a removed definition and the lines hold a
  character the two ports read differently.
- The `tests` trigger (O-3) also withdraws right verdicts when a test name is reused in another class or module beside
  a changed one, and when two different non-ASCII test names share their ASCII run (`test_ärger` and `test_ölen`
  through `test_`). The reviewer measured a per-name multiplicity cap and did not recommend it; neither does this pass.
- In joint shapes the count rule can withdraw a right CONTRADICTED and keep a false one (the reviewer's
  `p2a-121u:20260930:4`: GNU `-Nu` rendering plus `.énv.json`/`énv.json`; "4 files changed." withheld although main's
  CONTRADICTED is right, "5 files changed." left CONTRADICTED although it is true). The gate FAILs either way.

**Operator options, added to the pass-1 note's section 9:**
- **O-6: `dot_earliest`.** Keep a path claim when V97 and V97+V121 both verify even if V121 alone does not. Measured by
  the reviewer: 7 right verdicts withheld against 1 false verdict caught (#161's R14.3 format-patch, whose falsity comes
  from the multi-commit rendering) across four generated sets. Taking it redefines "because of #121" for that one shape.
- **O-7: joint count shapes.** When a dot twin exists, also abstain on CONTRADICTED counts above `main`'s count.

## 4. What is measured after the code, and where it goes

At the head that carries this pass: (A) the relation in both ports, both doors and both strict modes over the
committed inputs, the reviewer's cross-port fuzz and a fresh seeded fuzz; (B) the truth test at the raw door, in the
port in its own terms, and at the git door; (C) the (kind, verdict, text)-keyed agreement on the committed inputs and on
the fuzz; (D) recall on `main`'s committed corpora, the overlay's pins and #161's pairs, both path flavours; the per-call
cost and the timing guards; the bookmarklet's size. All of it goes into the README and the CHANGELOG entry, not here.

**Predictions, falsifiable.** The (kind, verdict, text)-keyed split count on the committed inputs goes from 17 to 0.
Exactly one pinned claim of `main`'s own files moves, as before (`path1:unrepaired-typo` claim 0). No committed input's
record breaks the relation, and the overlay's error phrase never appears.

## 5. What this pass does not change

No repair: PATH-2a still never gives VERIFIED where `main` was wrong, G-P1 is still not met, and that is still the
operator's decision. `main`'s reader is untouched in both ports (the reconstruction test holds). No receipt,
certificate, sworn file, PREREG, RESULT, ANALYSIS, AMENDMENT, ERRATUM, earlier NOTE or the charon log is edited.
