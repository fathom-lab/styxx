# NOTE — PATH-2a, seventh pass: the review of `fcd3ce6a`, each finding and what this pass does with it

## 0. Status

2026-09-30. Branch `fix/diffgate-abstain-where-wrong`, on `origin/main` `1cde8b82`. Pass 6 ended at `fcd3ce6a`. Its
notes are `NOTE_path2a_sixth_pass_2026_09_30.md` and `NOTE_path2a_sixth_pass_departures_2026_09_30.md`; the earlier
ones are listed in the README's *PATH-2a* section. None of them is edited. Four review lenses read `fcd3ce6a`:

- by construction (A): ship. 0 records outside the relation on 214,272 Python 3.12 runs, 155,678 Python 3.14 runs,
  210,866 port runs, 181,626 bookmarklet runs and 45,200 real-git runs. One major (the matching automaton's memory),
  two minors;
- coverage and recall (B, D): fix before shipping. Two majors (C-1's summary switch fires on any decorated sentence;
  #161's five joint #121 reproductions, carried), six minors;
- cross-port and runtime (C): fix before shipping. One blocker (a Unicode-version skew between the interpreter and the
  engine splits a gate verdict where `main`'s agree), four minors;
- integration, tests and docs: ship. One major (a timing test's margin on CI's interpreters), six minors.

**This note is written before this pass's code and committed alone, ahead of it.** Each design below was prototyped in
scratch (`scratchpad/ghmerge/p2a/b7`) and measured there against the committed inputs, the reviews' reproductions,
fuzz sets and truth worlds, on CPython 3.12.10 and Node 24.13.0. If the committed code departs from this note, the
departure goes into a note of its own. Finding identifiers are this note's: A-n, B-n, C-n and I-n number each lens's
findings in the order the review lists them; code comments and pins cite them.

Dispositions:
- **fixed**: a code, test or docs change in this pass. Where the finding is a defect of `fcd3ce6a`, the fix comes with
  a test that fails on `fcd3ce6a` (each was run there: a scratch tree holding `fcd3ce6a`'s two files beside this pass's
  tests).
- **disclosed**: kept as is and stated in the README, the CHANGELOG or here.
- **operator**: an option left to the operator, with measurements where this pass has them.

## 1. The blocker

### C-1 (bar C): a case pair one runtime folds and the other does not splits a gate verdict

**The finding.** The pass-6 proof that the overlay's gate verdicts agree wherever `main`'s do rested on both `main`s
reading the same claims with the same verdicts wherever `apart` is false. `main`'s file count keys each path through
`lower()` (`toLowerCase()` in the port), and the two runtimes' case tables differ by Unicode version: CPython 3.9 to
3.12 (Unicode 15 or older) keep U+A7DC and U+019B apart, Node 24 (Unicode 16) folds them. With `2 files changed.
Added 0 tests.` over two paths `src/X.py` and `src/x.py` (X and x that pair) and a changed test, `main`'s Python counts
3 files (CONTRADICTED) and its port 2 (VERIFIED); `apart` was false (an ASCII summary, a diff the two read alike), so
the overlay withheld the tests CONTRADICTED in both ports, and only the port's gate moved: `main` FAIL / FAIL,
`fcd3ce6a` FAIL / PASS. The review's generator found 132 of 5,000 such splits on CPython 3.12 with Node 24, and 33 of
5,000 on CPython 3.14 with an engine patched to fold one more pair.

**The rule.** Each `main` counts the distinct keys of its registrations. Two registrations whose wild forms differ (an
ASCII position differs, or a placeholder sits where the other has ASCII) never share a key on a runtime that keeps the
fold lemma; two whose fold forms are equal always do. So every runtime's `main` counts some number in [#WA, #A], the
two figures `f.count()` already computes, and two `main`s can give a count claim of n different verdicts only if
#WA < #A and n lies in that range. So:

> Among the kinds of claims `main` read, a `files_changed_count` claim whose number lies in [#WA, #A], where #WA < #A
> (two registrations differ only outside ASCII), makes `apart` true: no CONTRADICTED is withheld, in either port.

A number outside the range is CONTRADICTED by every `main`, and the overlay decides it alike in both ports (it reads
the claimed number from the detail and `main`'s count against the same range). The review proposed the clause without
the range; with it, 11 count CONTRADICTEDs of the committed inputs are kept that the overlay withheld at `fcd3ce6a`
(12 without the range), all on the seeded PATH-2a fuzz, whose paths differ in one letter outside ASCII. Scopes, tests
and symbols read no key's case outside ASCII: a scope's prefix is ASCII wherever `apart`'s summary part is false (a
letter outside ASCII between `only` and the end of its sentence, after `touch`, `modif` or `chang`, is a C-1
sentence), lowering a path outside ASCII yields ASCII only through U+0130 and U+212A, which the fold forms read, and
`_diff_touches_python` reads an ASCII extension.

**Pins.** The review's input joins `XPORT_CASES` (`C1-a-unicode-16-case-pair`), pinned as "the overlay moves no claim
in either port" with gate FAIL, since which count each port reads depends on its runtime. A new test runs the review's
generator shape (1,500 inputs) with the port on an engine patched to fold U+A7CE and U+A7CF (unassigned through
Unicode 16, standing in for a later version's pair; `check_path2a.js --decisions-newer-engine`), so it does not
depend on the runner's versions, and asserts no non-strict gate split where `main` agrees: `fcd3ce6a` splits more than
40 of them. A pinned pair (`path2a:p7-a-case-pair-outside-ascii-keeps-contradicted`) keeps its tests CONTRADICTED, so
dropping the clause from either port alone is caught.

**What stays.** The proof's premise is now the fold lemma alone (each runtime's `lower()` maps nothing but U+0130 and
U+212A to text holding ASCII), which the tests pin on the runtimes they run; the bookmarklet also runs on browser
engines where that is not checked. The README's *Cross-port* section and the CHANGELOG say so (C-2).

## 2. The majors

### A-1 (bar A): the matching automaton's memory

**The finding.** `_p2a_found` (and the port's `_p2aFound`) built one Aho-Corasick automaton over every named token
whenever more than 32 were named, before looking at the text, even when the text was empty. 10,000 path claims of
60-character directories: 171 MB at the overlay's peak in Python where `main` takes 8.9 MB; in Node, a heap abort under
a 100 MB cap where `main` completes.

**Fixed, in both ports.** A named token holds no character of `_P2A_BAD_RX` and no `_P2A_SEP`, so where it occurs it
lies inside one piece of the text between such characters. `_p2a_found` reads the text as its distinct pieces:
- an empty text, or one with no piece, reads no token (the review's input builds no automaton);
- a token that is a piece is found by lookup, and one no shorter than the longest piece, and not a piece, is not there;
- the rest are read through automata of at most `_P2A_BUDGET` (65,536) characters of tokens each, or a sixteenth of
  the pieces' length if that is more, each reading the pieces once: memory bounded, time linear in the tokens and the
  text.

Every way gives the set one scan per token gives, so no decision moves. In the port, a key form is built as one joined
string, and a string with nothing to fold is returned as it is: a string appended to code point by code point is a
chain of pieces V8 keeps until it is read whole, about 2 KB for a 70-character path, and the overlay keeps the forms as
keys (10,000 such claims retained 38 MB that way).

**Measured.** The review's input (10,000 claims, 0.82 MB): 12.1 MB at the overlay's peak in Python (157 MB at
`fcd3ce6a`; `main`'s call 8.5 MB); in Node it completes under a 24 MB heap (`main` under 16 MB; `fcd3ce6a` aborts at
64 MB). 3,000 distinct 110-digit counts over a dot twin: 0.4 MB with an empty text and 16.5 MB with each number in a
wordish run (68.1 and 68.5 MB at `fcd3ce6a`; `main`'s call 2.6 and 2.8 MB).

**Pins.** Python: the two long-count cases within 32 MB (`tracemalloc`) and within three times `main`'s call; the port:
those at 6,000 counts and the review's own input, under a 64 MB heap, for `main`'s port and this one. `fcd3ce6a` fails
both (68 MB; heap aborts). The found-set test reads tokens as `tokens` names them and holds cases of more than one
automaton's worth.

### B-1 (bar B): C-1's summary switch fires on a decorated sentence

**The finding.** `apart`'s summary part fired on any sentence that held a wordish code point and a trigger's words in
any order, `only` and `changed` each a trigger alone: one sentence such as "Thanks to José for the review; only a typo
fix otherwise.", "## 🧪 Tests added" or "Behaviour is unchanged for naïve callers." kept every #121 and #101
CONTRADICTED of the summary. In the builder's world with such a sentence appended, 979 of 1,193 attributable claims
were withheld (1,193 without); in the review's git-built world 496 of 619 at the raw door. The README said C-1 kept
"none in the truth worlds", true only because those worlds' summaries were ASCII.

**Fixed, in both ports.**
- A trigger is the words a template's match holds, in the template's order: `file` before `changed`; `only` before
  `touch`, `modif` or `chang`; `add` or `creat` before `test`; `add` or `introduc` before `function`, `class` or
  `method`; and the DECLARE-1 keys `files_changed`, `only_touches`, `tests_added` and `adds_symbol` as words of their
  own (a key's line writes the sentence). A pair counts when the earliest occurrence of its leading word comes before
  an occurrence of the other within the sentence. The argument is pass 6's: a one-port claim of a kind that can be
  CONTRADICTED has its match in a sentence of that port, which lies inside one sentence of both, and the match holds
  these words in this order under the same reading of case.
- O-11: the emoji of five pictograph blocks (U+1F300 to U+1F64F, U+1F680 to U+1F6FF, U+1F900 to U+1F9FF, U+1FA70 to
  U+1FAFF) are read in the summary as U+2190, a neutral code point: as one code point in the Python, or as the two
  surrogates the port's string holds (and a Python string holding the same two surrogates, as a JSON reader may hand
  it, reads them so too). None of them is a word character or white space for either port's templates, folds to an
  ASCII letter or has a case; the tests pin each property by enumeration on every runtime they run, and that the two
  regexes read exactly those code points and nothing else. The review proposed reading them as neutral in `apart`'s
  check; this pass reads them so wherever the summary is read (the sentence check, the scope zones and the runs
  `extract` reads), since the property holds for each, so an emoji beside a path no longer withholds it either.

**Measured.** Python, decorations appended (an emoji heading, the José sentence, the naïve sentence): the builder's
world withholds 1,193 of 1,193 attributable claims under each (979 at `fcd3ce6a`), at 236 of 5,842 right verdicts
lost, as without a decoration; the review's git-built world withholds every attributable claim in all five renderings
under each (raw 619, `--no-renames` 748, `-U0` 619, `--no-prefix` 620, git door 610), 0 count misses by name-status
lines. The port's world: 227 count and 79 tests CONTRADICTEDs withheld with or without a decoration (0 and 0 with one at
`fcd3ce6a`). On the EXTERNAL-1 shelf the review measured the proposal (Python): the summary part fired on 303 bodies at
`fcd3ce6a`, 158 with the ordered triggers and 32 with O-11 too (Claude Code's bodies: 104, 82, 2); this pass did not
re-read the shelf.

**Pins.** `XPORT_CASES`: the review's four decorated sentences and its count case (withheld in both ports; `fcd3ce6a`
kept `main`'s false CONTRADICTED in both), an emoji in the claim's own sentence (O-11), the same as two surrogates, the
template's words in order beside an accented letter (kept, as C-1 requires), and an emoji outside the five blocks
(kept). Three pinned pairs. Plants: the count trigger's loss, the scope trigger read as `only` alone, O-11 off and the
piece lookup's loss, each in either port alone, are caught.

### B-2 (bar B): #161's five joint #121 reproductions (carried)

**Disposition: operator, disclosed, unchanged.** Under the committed but-for attribution they are not misses (V121's
count is false there too); #161 files them as #121 reproductions. They stay pinned with `main`'s false CONTRADICTED
kept. The review's narrower rule (withhold a CONTRADICTED count only where a dot twin exists, `main`'s count is below
the claimed number, and the diff holds a change marker `main` registers no file for) is recorded as operator option
O-13, not measured here; O-7 stands with its pass-6 figures.

### I-1 (integration): a timing test's margin on CI's interpreters

**The finding.** `test_cost_on_large_summaries_python` bounded the overlay alone within `main`'s whole call. The
overlay's loops are Python where `main`'s are mostly C regex, so a slower interpreter moves the overlay more: 0.35 to
0.65 of `main`'s call on Linux CPython 3.12.3, and CI's 3.9 and 3.10 lack the specialising interpreter.

**Fixed.** The bound is 1.5 times `main`'s call, with the measured ratios in the test's docstring (0.23 to 0.49 here);
`495d2204` took 2.5 to 7.5 times.

## 3. The minors

| id | finding | disposition |
|---|---|---|
| A-2 | Line-heavy diffs: one added line of U+2028-separated ` def test_x(): pass` segments under "Added 1 tests." costs ×13 to ×17 in Node and ×7 to ×8 in Python; the README said up to about ×9. | disclosed: the README gives the review's shape and ranges. |
| A-3 | The pass-6 report's count-seam figures had no surviving receipt (the run crashed on a tag holding ':'). | fixed: the generator was run again with a colon-free tag; its log is kept beside this pass's measurements (`b7/meas7_fuzz.log`): 0 outside the relation, 0 non-strict gate splits where `main` agrees, 791 strict splits, as pass 6 reported. |
| B-3 | C-1's diff part fired on any added `def` whose name ran into a code point from 0x80 up, or a CJK line holding `def`, beside any symbol claim. | fixed: both `main`s read a symbol claim by regex, never through NFKC; they can read one apart at a site only where its line is in one view only, a white space of one port lies before the name, or the site's ASCII name run is a claimed name followed by a code point from 0x80 up (CPython's `\b` may read it as a word character, the port's does not). The NFKC test is gone from `apart`. Pins: the review's two shapes (withheld) and the claimed name running on (kept). |
| B-4 | On CONTRADICTED, `tests` withholds more right verdicts than false ones in both truth worlds; the per-phrase table merged the verdicts. | disclosed: the README's per-phrase table is split by `main`'s verdict (the raw door's truth world: on CONTRADICTED 33 right, 20 false, 13 undecided; on VERIFIED 20 right, 34 false, 12 undecided). The review's option (keep a CONTRADICTED tests claim where every paired site is an anchored `def` on both sides) is operator option O-14. |
| B-5 | `dir` withholds right VERIFIEDs on a dot-directory file in a monorepo (`.vscode/settings.json` over `apps/api/.vscode/settings.json`): V121 and V97+V121 verify, V97 alone drops the dot. | disclosed: named among `dir`'s costs, with the review's figures (4 of 33 firings on real agent PRs). Keeping such a claim is operator option O-15. |
| B-6 | The README said C-1 keeps 8 CONTRADICTEDs on #161's reproductions; it was 10. | fixed: the README gives the figure (10 at this pass's prototype) and names the claims. |
| B-7 | The known-gaps list did not name #161's off-tree prefix family (v4-*, f4: `main`'s `rstrip("/.")` reads `docs/..` as `docs`). | disclosed: named among the known gaps outside the three defects. |
| B-8, C-5, I-6 | Process: the reviewers shared one scratch directory and C: ran out of space. | recorded. This pass worked in `p2a/b7` only, with `TEMP`, `TMP` and pytest's `--basetemp` there, deleting case data after summarising; C: had 1.3 GB free at its start. Pruning `review1` and the reviews' bulk is the operator's call. |
| C-2 | The pass-6 note's prediction and report stated cross-port gate agreement without the same-case-tables premise. | fixed: C-1 above; the README and the CHANGELOG state the premise that remains (the fold lemma). The committed note is not edited. |
| C-3 | CPython 3.14 here has no pytest and no numpy. | recorded: this pass runs the cross-port, table, relation and newer-engine tests on 3.14.2 through a stand-in runner that stubs pytest and loads `styxx` as a bare package, and says so. |
| C-4, I-4 | `build_bookmarklet.py`'s docstring names terser 5.51.2; the README says 5.46.0 produced the shipped bytes; `--check` writes `bookmarklet_src.js`. | disclosed: the README says the docstring's version is the one that built `main`'s bookmarklet, and that `--check` writes the source file (the committed test runs it on a copy). `build_bookmarklet.py` is `main`'s and is not edited. |
| I-2 | The CHANGELOG gave "1.72 to 7.5 times" for the fifth pass's head; 1.72 is a duration. | fixed: "2.5 to 7.5 times". |
| I-3 | The README said `check_pairs.js` checks 153 pinned pairs; it checks 180 at `fcd3ce6a`. | fixed: the figure at this head. |
| I-5 | `UNICODE_SAME` left out 15.1.0 (CPython 3.13) though the committed inputs hold no code point assigned in 15.0 or 15.1. | fixed: 15.1.0 is added, with the review's inventory cited. |
| I-7 | The scratch PR body counted the pass-5 findings as "one blocker, four majors"; there were five majors. | fixed in the scratch PR body, which this pass rewrites. |

## 4. The rules after this pass

Everything in the earlier notes stands, with these changes.
- **C-1, the diff part:** a count claim whose number lies in [#WA, #A], where #WA < #A, makes `apart` true; the symbol
  part reads a site's line, the white space before its name, and a claimed name running into a code point from 0x80
  up, and no longer NFKC.
- **C-1, the summary part:** each trigger's words in its template's order, the DECLARE-1 keys on their own.
- **The summary:** pictograph emoji of five blocks read as a neutral code point wherever the summary is read.
- **Tokens:** found through the text's pieces, by lookup, and through automata of bounded size; no decision moves.
- **The port's key forms:** built as one joined string, or returned as they are.

The reconstruction property, REACH, the reason form, the phrases, the error fallback and the abstain-only relation are
unchanged.

## 5. Measured on the prototype, and predictions for the head that carries this pass

Measured (CPython 3.12.10, Node 24.13.0), `main` against `fcd3ce6a` and the prototype, in Python and in the port, both
strict modes:

| set | inputs | outside the relation | claim splits | non-strict gate splits where `main` agrees (`fcd3ce6a` → prototype) | `--strict` gate splits |
|---|---|---|---|---|---|
| the review's hostile `gen5`, seeds 11 and 12 | 40,000 | 0 | 0 | 0 → 0 | 171 and 195 |
| the mixed count, tests, symbol and scope seam set, seed 7 | 20,000 | 0 | 0 | 0 → 0 | 741 |
| pass 5's count-seam set, seed 5 (A-3) | 20,000 | 0 | 0 | 0 → 0 | 791 |
| the sixth review's hostile `hx`, seed 71 | 12,000 | 0 | 0 | 0 → 0 | 2 |
| the sixth review's `gen7`, seeds 11 to 14 | 20,000 | 0 | 0 | 0 → 0 | 34 |
| `gen8`, seeds 1 and 2 (case pairs), CPython 3.12 against Node 24 | 10,000 | 0 | 0 | 250 → 0 | 0 |
| the same, the port patched to fold U+A7CE and U+A7CF | 10,000 | 0 | 0 | 309 → 0 | 0 |
| the builder's world with five decorations | 8,000 | 0 | 0 | 0 → 0 | 0 |

Coverage (B): the builder's world and the review's git-built world, with and without decorations, as in §2 B-1; the
truth pins move by the case-pair clause only (6 CONTRADICTEDs kept in each port: 4 counts, 2 of them false through
neither defect and 2 right, and 2 right scopes): Python 1,206 of 1,206 attributable withheld, right verdicts lost
251 → 247 of 5,897; the port 1,100 of 1,100, 175 → 171 of 5,739. On #161's reproductions C-1 keeps 10 CONTRADICTEDs, none
attributable: 8 under a header or separator the two `main`s split apart, the tests claims of
`f2-a-context-line-holding-a-separator-adds-nothing` and `x1-…-a-katakana-middle-dot` (false by their names, outside
the committed V101's attribution), and none other; `y2`'s CONTRADICTED is right. Recall (D): `main`'s committed
corpora 80 of 2,231 (3.6%), as before; with #161's pairs 264 of 2,761; the overlay's own 131 pinned pairs 71 of 147
(70 of 146 under POSIX). Cost: the large cases' overlay alone 0.23 to 0.49 of `main`'s call in Python and 0.92 to 1.34
in the port; the committed timing cases' slowest 0.166 s in Python and 76 ms in the port, peak memory 14.2 MB.

Predictions for the head:
- **Relation.** No committed input's record breaks the relation (A) in either port, at either door, in either strict
  mode; none on the sets above.
- **Cross-port.** 0 splits under every by-construction key; 0 non-strict gate splits where `main`'s gates agree on the
  committed inputs, on every set above and on the newer-engine test; under `--strict`, the two committed inputs of the
  sixth pass's C-4 and no other.
- **Truth.** 0 misses at each door and in the port on the builder's world, the pass-5 and pass-6 fixtures and the
  review's git-built world, decorated or not.
- **Recall.** `main`'s committed corpora: 80 of 2,231.
- **Cost.** Every committed timing and memory case under its bound.

## 6. What this pass does not change, and what the operator decides

- **No repair.** PATH-2a never gives VERIFIED where `main` was wrong.
- **G-P1** is not met, and that is the operator's decision.
- **`main`'s reader** is untouched in both ports; the reconstruction test holds.
- **Nothing committed is edited:** no receipt, certificate, sworn file, PREREG, RESULT, ANALYSIS, AMENDMENT, ERRATUM,
  earlier NOTE or the charon log.
- **Operator options.** O-1 to O-12 stand; O-13 (B-2), O-14 (B-4) and O-15 (B-5) are added. Open for the operator:
  the `--strict` splits, O-7 and O-13, the rename and `-U0` gaps, and whether the Action should import PyPI's styxx.
