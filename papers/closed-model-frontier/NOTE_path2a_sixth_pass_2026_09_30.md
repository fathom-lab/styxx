# NOTE — PATH-2a, sixth pass: the review of `495d2204`, each finding and what this pass does with it

## 0. Status

2026-09-30. Branch `fix/diffgate-abstain-where-wrong`, on `origin/main` `1cde8b82`. Pass 5 ended at `495d2204`. Its
notes are `NOTE_path2a_fifth_pass_2026_09_30.md` and `NOTE_path2a_fifth_pass_corrections_2026_09_30.md`; the earlier
ones are `NOTE_path2a_abstain_overlay_2026_09_30.md`, `NOTE_path2a_second_pass_2026_09_30.md`,
`NOTE_path2a_second_pass_corrections_2026_09_30.md`, `NOTE_path2a_third_pass_2026_09_30.md` and
`NOTE_path2a_fourth_pass_2026_09_30.md`. None of them is edited. Four review lenses read `495d2204`:

- by construction (A): ship. 0 records outside the relation on 4,143 committed inputs, 34,000 fuzz inputs, 12,984
  git-door runs per interpreter and the shipped bookmarklet. One major (a cost that grows as claims times summary
  length), three minors;
- coverage and recall (B, D): fix before shipping. Two majors (a test defined again beside its own unchanged definition; five of
  #161's joint #121 reproductions), three minors;
- cross-port and runtime (C): fix before shipping. One blocker (a tests claim in place of one of the two counts of pass 5's C-1),
  one major (the port's scan accepts operators that run StringToNumber; the Python's accepts `\N{...}`), three minors;
- integration, tests and docs: fix before shipping. One major (the docs say the Action runs PyPI's styxx; it does not), four
  minors.

This is the last pass of the workflow: after it the result goes to the operator. So every open item is written down
here, in the README and in the CHANGELOG entry, precisely enough to act on without this note's reviews.

**This note is written before this pass's code and committed alone, ahead of it.** Each design below was prototyped in
scratch (`scratchpad/ghmerge/p2a/b6`) against the committed inputs (6,655), the reviews' reproductions and fuzz sets
(`gen5` seed 11, 20,000 inputs; a mixed count, tests, symbol and scope seam set built from the review's `seam2`, seed 7,
20,000 inputs), the builder's truth world and the review's git-built truth world (`cov5/built1`, 900 cases, five
renderings), and then fixed here. Figures quoted as reasons were measured on that prototype (CPython 3.12.10, Node
24.13.0). If the committed code departs from this note, the departure goes into the next note. Finding identifiers
are this note's; code comments and pinned pairs cite them.

Dispositions:
- **fixed**: a code, test or docs change in this pass. Where the finding is a defect of `495d2204`, the fix comes with
  a test that fails on `495d2204`.
- **disclosed**: kept as is and stated in the README, the CHANGELOG or here.
- **operator**: an option left to the operator, with measurements.

## 1. The blocker

### C-1 (bar C): a gate verdict split where main's two ports read different claims from the summary

**The finding.** Pass 5 balanced a count one port reads against a count the other reads (the count seam). A tests
claim in place of either count reopens it, and the seam itself made new splits: `13\x1cfiles changed. Added 3﻿tests.`
over a twin diff and a changed test gives main FAIL / FAIL, `495d2204` PASS / FAIL (Python withholds its one-port count as
`seam`, the port keeps its one-port tests CONTRADICTED). Also `Added 0\x1ctests. Added 3﻿tests.`,
`Added 0\x1ctests. 9﻿files changed.`, `Added 0 teſts. ...`, and an accented verb. On the review's mixed set, 771
non-strict splits where main agrees. The review also named the "inherent" case: a claim both ports read, withheld
alike, while the other port FAILs on a claim of its own (`Added 0 tests. 9﻿files changed.` over a changed test).

**The rule, stated in what both ports see.** Without `--strict` a gate verdict is FAIL exactly when a CONTRADICTED is
left. The overlay moves a gate verdict only by withholding a CONTRADICTED; withholding a VERIFIED never moves it. So:

> Where the two ports' mains may read apart which claims can be CONTRADICTED, or decide such a claim apart, the overlay
> withholds no CONTRADICTED verdict, in either port (`apart`, a fact of the summary's and the diff's bytes).

Then, on an input where `apart` holds, each port's gate verdict without `--strict` is its own main's, so the two agree
wherever main's agree. Where `apart` does not hold, both mains read the same claims of the kinds that can be
CONTRADICTED, with the same verdicts, and the overlay decides each alike (bar C, by construction), so the gate verdicts
agree. That is a proof, not a measurement, up to one premise the README already names (C-4 of pass 5: each runtime's
lower() keeping the fold lemma).

`apart` holds when any of these holds:
- **A sentence**, as both ports end one (a line break, or `.`, `!` or `?` then a space, tab or CR), that holds a
  character the two ports' templates may read apart (`_P2A_BAD_RX`: the divergent white space and line breaks, and the
  wordish code points) and the words every match of a template that can give CONTRADICTED holds together: `changed`
  (the count), `only` (the scope), a verb and `test` (`add` or `creat` with `test`), a verb and a kind (`add` or
  `introduc` with `function`, `class`, `method` or `symbol`). Words are read in ASCII case with the four code points
  CPython's IGNORECASE folds to an ASCII letter (U+0130, U+0131, U+017F, U+212A) read as that letter. A DECLARE-1 line
  that writes such a sentence holds the same words (`files_changed`, `only_touches`, `tests_added`, `adds_symbol`).
  Argument: in a sentence of ASCII and neutral characters only, both ports' templates, sentence splitters and
  DECLARE-1 line readers read alike; a one-port claim of one of those kinds has its match, which holds its words, in a
  sentence of that port, which lies inside one sentence of both.
- **A DECLARE-1 fence word** (`styxx`) in a summary that also holds a divergent character or a lone CR: the fence and
  its body lines are found by `^` (which the port's `m` flag also matches after CR, U+2028 and U+2029) and split by
  `str.splitlines` against `/\r?\n/`.
- **The diff**, read for the kinds among the claims main read (both ports read the same such claims wherever the
  summary part does not hold): a file header the two split or strip apart (statuses: counts, scopes, BC-1); for a
  tests claim, the two views' own counts of `def test_` sites differ; for a symbol claim, an added `def` or `class`
  site on a line only one view reads, or with a white space of one port alone before the name, or with a name read
  through NFKC (`_p2a_wide_name`); and, where no path is registered, added lines empty to one main only.

**Measured on the prototype.**
- Non-strict gate splits where main agrees: committed inputs 5 → 0 (the five pinned C-5 inputs, all diff-side);
  `gen5` seed 11 (20,000): 179 → 0; the mixed seam set (20,000): 954 → 0. Claim splits under every committed key: 0.
- Relation (A): 0 problems on all three sets, both ports, both strict modes.
- What it costs, on the committed inputs (Python / port): 509 / 538 CONTRADICTED decisions of `495d2204` are kept,
  nearly all on the seeded PATH-2a fuzz and the text-seam set, whose summaries and diffs are built from those
  characters: `divergent` counts 156 / 154, `extract` scopes 154 / 155, `divergent` scopes 105 / 101, `tests` 83 / 117
  (the port counts a test after U+FEFF that CPython does not), `seam` 5 / 5, `split` 3 / 3, `extract` counts 3 / 3.
- Coverage: 0 misses on the builder's truth world (1,706 cases) and on the review's git-built world (900 cases, five
  renderings), in Python and in the port. On #161's 482 reproductions, 8 CONTRADICTED decisions are kept in Python
  (`divergent` 7, `split` 1): f2's separator before a header shape and v2's path holding U+2028 (the two mains count
  different files; the false one is false through CPython's line breaks, none of the three defects) and y2 (main's
  CONTRADICTED there is right).
- On the review's five C-1 shapes: main FAIL / FAIL, this pass FAIL / FAIL.

**What stays.** Under `--strict` a withheld VERIFIED moves a gate verdict too, and a claim one port reads alone can be
withheld on that port only. `apart` does not cover it and cannot: a decision must read the same under `--strict` as
without it (pass 2), and withholding nothing in a sentence that may read apart, in both modes, would keep every false
VERIFIED of the path kinds in any sentence that holds an accented path. Measured: 2 committed inputs (the pass-5 pair
p2a-seam:4242:335 and :487, C-4 below), 171 of 20,000 `gen5` inputs, 741 of 20,000 on the mixed seam set. Disclosed.

**Pins.** The review's five C-1 inputs and the inherent one join `XPORT_CASES` with one gate verdict each; a test runs
the committed inputs and the pinned C-1 inputs and asserts 0 non-strict gate splits where main agrees (so
`GATE_SPLITS_MAIN_AGREES` becomes empty); `495d2204` fails both.

## 2. The majors

### A-1 (bar A): the overlay's cost grew as claims times summary length

**The finding.** Each distinct count, path or name token was looked up by `s in f.runs(kind)` and each scope prefix by
`prefix in f.zone_text()`, a scan of summary-sized text per token: x15 over main at 3.6 MB and 30,000 claims.

**Fixed.** `_p2a_abstain` names the tokens of the claims in reach before any claim is read (`tokens`), as pass 5 names
paths. `occurs` answers a named token from one pass over the runs (or zones) for all of them, `_p2a_found`: up to 32
words, one scan each; more, one read through one Aho-Corasick automaton that marks each state's words once, so time is
linear in the words and the text. Both give the same set, so no decision moves. A token not named is scanned alone.

**On the prototype** (whole calls, least of two; 1.2 to 1.3 MB summaries, 10,000 distinct claims, one-file diffs):
Python, main / `495d2204` / this pass: counts 0.68 / 1.72 / 0.83 s, paths 0.79 / 3.83 / 1.08 s, scopes 0.70 / 5.23 /
1.05 s; the overlay alone 0.18 to 0.34 s. Node: counts 95 / 379 / 186 ms, paths 123 / 545 / 343 ms, scopes 133 / 542 /
277 ms. Decisions unchanged on every committed input.

**Pins.** Three large timing cases (counts, paths, scopes: about 1.2 MB and 10,000 distinct claims), bounded relative
to main's own call on the same input: the overlay alone within main's call in Python, within twice it in the port.
`495d2204` fails them. The README's cost section states the linear term.

### B-1 (bar B): a test defined again beside its own unchanged definition

**The finding.** A head that adds `def test_e():` where the base already defines `test_e` in a context line: main
counts the added line, and its tests verdict is false (the review's minimal diff; 12 such kept in the builder's world,
at least 4 per rendering in the review's).

**Fixed.** The unchanged lines of the diff (git lines starting with ' ', in each distinct view, and the pieces no view
reads: after a lone CR, and continuation runs with no removed piece) are read as the base side too:
- tests: a counted site whose name an unchanged line defines pairs as a changed one does; where the claim fires only
  with those pairs, the phrase is the new `redefined`;
- symbols: a claimed name an unchanged line defines after an anchored `def` or `class` is withheld with the new
  `again`. The truth reads most such claims undecided (the name is in the base), but the review's evaluator judged two
  of them false in its world, and none right.
- The NFKC rule of pass 5 is not applied to unchanged lines: a `def` there whose name reads through NFKC is passed
  over, so prose in a context line cannot withhold every tests and symbol claim (B-4). A redefinition spelled through
  NFKC is not caught: disclosed.

**On the prototype.** Builder's world: 0 misses; `false other abstained` 51 → 67 (Python) and 7 → 23 (port); right
lost +6, undecided +4, in each port. The review's git-built world: all 12 claims per rendering its context
attribution names are withheld (with the tests rule alone, its 2 symbols per rendering stayed), with 8 right tests
verdicts lost per rendering. The committed inputs move 15 tests and
5 symbol decisions (Python). Under `-U0` nothing unchanged is visible: disclosed.

**Pins.** The minimal diff joins `path2a_pairs.json` with both summaries, and a truth case in a new fixture
`tests/fixtures/path2a_pass6_repros.json` (judged at three doors, attributed by the ast-paired V101 of pass 5).

### B-2 (bar B): five of #161's joint #121 reproductions keep main's false CONTRADICTED

**The finding.** `m-121-a-submodule-line-licenses-nothing`, `k4-a-dotted-twin-beside-a-submodule-line`,
`k4-a-dotted-twin-beside-an-hg-binary-notice`, `l-r10-np1-...` and `l-r10-np2-...`: V121's count is still false (a
submodule line, an hg binary notice or a no-prefix directory main registers no file for), so under the committed
but-for attribution these are not misses; #161 files them as #121 reproductions.

**Disposition: disclosed, and pinned.** A test pins each one's count decisions in both ports (the false CONTRADICTED
kept, the right one withheld as `count`) with the gate verdict. The README's coverage table names them as kept.
**Operator option O-7** (withhold a CONTRADICTED count above main's count wherever a dot twin exists), measured: it
catches all five, and costs 289 right CONTRADICTEDs for 41 false ones caught in the builder's world, 215 for 38 in the
review's git-built world (raw rendering). Not taken.

### C-2 (bar C): the static checks accept numeric conversion and a name lookup

**The finding.** The port's token scan refused `Number`, `parseInt`, `parseFloat`, `isNaN`, `isFinite` as words, but not
unary `+`, `==`, `Math.max(s, 0)`, a typed-array store or `new Date(s)`, each of which runs StringToNumber and so the
engine's white-space table. The Python's regex check accepted `\N{NAME}`.

**Fixed.**
- Port: `==` and `!=` refused in code (the block uses `===` and `!==` only); a unary `+` or `-` before a name, a call, a
  bracket or a string refused; the words `Date`, `Math`, `BigInt`, `DataView` and every typed-array constructor refused.
  The block's two `Math.max` / `Math.min` become conditionals and its `Uint8Array` a `Set`. The review's eight plants
  are committed as refused cases.
- Python: `\N` refused in a static pattern; the plant is committed.
- Disclosed: binary `*`, `-`, `%` and the relational operators on a string, and the numeric parameters of built-in
  methods, also run ToNumber; the scan cannot type them. The block passes them numbers only (lengths, indexes, digit
  table values).

### I-1 (integration): the Action does not run PyPI's styxx

**The finding.** `action.yml` runs `python "${{ github.action_path }}/diffgate_action.py"`, which puts the action's own
checkout at `sys.path[0]`, so `from styxx.diffgate import ...` imports the styxx beside the script, at the ref the
workflow names, not the pip-installed package. This PR's own `diffgate` job (`uses: ./`) already runs the overlay, and
`fathom-lab/styxx@main` users run it once this merges. Pass 5's note (I-1: "The PR's diffgate check runs PyPI 7.48.0")
and the README, the CHANGELOG, the Action script's comment and the scratch PR body said otherwise.

**Fixed.** The README, the CHANGELOG and the script's comment say what runs; this note records pass 5's error. `main`'s
`.github/workflows/diffgate.yml` comment ("the instrument itself is the released package") carries the same premise; it
is `main`'s file and this branch does not touch `.github/`, so the README names it. Whether the Action should run PyPI's
package instead is the operator's call.

## 3. The minors

| id | finding | disposition |
|---|---|---|
| A-2 | The port read `opts.strict` twice; both ports recomputed the gate verdict when nothing moved. | fixed: the port reads `strict` and `_declared` once, in main's order, and hands main that snapshot (a null `opts` reaches main as it came); both ports recompute the gate verdict only when a claim moved. A getter that answers differently on a second read pins it. The Python analog (a `strict` whose truth changes between reads) is re-read only when a claim moved: disclosed. |
| A-3 | Line-heavy diffs run above the README table (x5.1 to x9, peak memory up to about 2.5 times on VT-heavy diffs). | disclosed: the README gives the review's shapes and ranges; the second line view stays eager. |
| A-4, I-5 | Process: concurrent reviewers shared `review5`, the scratch passed 2 GB, C: reached 0 bytes free; mutation runs left pytest basetemps. | recorded. This pass works in `p2a/b6` only, sets `TEMP`, `TMP` and `--basetemp` there and deletes each basetemp after its run. The next reviewers should each take their own subdirectory. `review1` (925 MB) and `review5` are other agents' files; pruning them is the operator's call. C: had 2.2 GB free at this pass's start. |
| B-3 | On git's bytes with rename detection, a claim naming a moved file's old path is withheld (`dir`, `dot`, `dot_tier`) though main's VERIFIED is right; the README said `dir` withholds none on git's bytes. | disclosed and pinned: the sentence is corrected, the cost lists name it with the review's figure (43 of 3,277 right verdicts, 1.3%, in its 900-case world; 0 under `--no-renames`), and the review's rename pair is pinned as a known loss. |
| B-4 | The NFKC rule fires on removed prose in any file, and the anchored symbol read treats a CJK-led line as anchored. | disclosed: the README and CHANGELOG say the trigger is any removed line of any file where `def` is followed by a coarse run holding a code point from 0x80 up (tests), or `def`/`class` is reached from the line start through such code points (symbols); corpus_real has 0 firings. Not applied to unchanged lines (B-1). |
| B-5 | `dot_earliest` and `shape` mostly withhold right verdicts at a larger scale. | disclosed with the review's figures (grammar fuzz: `dot_earliest` 44 right / 0 false, `shape` 66 right / 1 false); O-6 and O-12 stand. |
| C-3 | The position key (kind, verdict, text) also pairs two different matches; OM1 splits the gates under `--strict` although every claim pairs. | disclosed and pinned: both measured keys are called heuristics, the "every claim pairs" figure is said to be taken without `--strict`, and OM1 joins `XPORT_CASES` with its expected split. |
| C-4 | Two more gate splits under `--strict` on the committed inputs (p2a-seam:4242:335, :487), named nowhere. | pinned (`GATE_SPLITS_MAIN_AGREES_STRICT`) and named in the README: `extract` withholds a right VERIFIED only the port reads. |
| C-5 | Runtime skew: only an engine that breaks the fold lemma splits decisions. | recorded: H-count-5 (`5 files changed.` over `e/ı.md`, `e/i.md`, `.env.json`, `env.json`, the port patched to lower U+0131 to `i`) is the concrete shape of pass 5's C-4; a Node with Unicode 17 case pairs passes the suites (the one failure is the harness's two realms). |
| I-2 | Two departures of 5e5f2450 are in no note: the joined reading cuts removed git lines at a lone CR only (the pass-5 note says every break `str.splitlines` knows), and the NFKC readings stop once a name reads through NFKC. | recorded here; the review counted 4 gen4 symbol claims that moved between 561be33e and 5e5f2450. |
| I-3 | The Action shows an overlay reason whole, main's reading included, so long paths can pass GitHub's 1 MiB step-summary limit. | fixed: the overlay's words whole, main's reading after them cut at 100 characters as main cuts; the Action test pins it. |
| I-4 | The per-case timing bound's margin on CI, and the CHANGELOG's slowest-case figure. | fixed: the large cases are bounded relative to main's call; the figures are refreshed at the head. |

## 4. The rules after this pass

Everything in the earlier notes stands, with these changes.
- **C-1:** a CONTRADICTED is kept wherever `apart` holds (§1), ahead of every other rule.
- **`tests_added`:** unchanged lines pair too (`redefined`); **`symbol_added`:** a name an unchanged line defines is
  withheld (`again`); neither reads an unchanged line through NFKC.
- **Tokens:** read in one pass for all the claims' tokens (A-1). No decision moves.
- **Phrases:** `redefined` and `again` are added.
- **Static checks:** the port refuses loose equality, unary `+`/`-`, `Date`, `Math`, `BigInt`, `DataView` and typed
  arrays; the Python refuses `\N`.
- **Wrapper:** the port reads `opts` once; the gate verdict is recomputed only where a claim moved.

The reconstruction property, REACH, the reason form, the error fallback and the abstain-only relation are unchanged.

## 5. Predictions, falsifiable, for the head that carries this pass

- **Relation.** No committed input's record breaks the relation (A) in either port, at either door, in either strict
  mode.
- **Cross-port.** 0 splits under every by-construction key on the committed inputs; 0 non-strict gate splits where
  main's gates agree, on the committed inputs and on every fuzz set run; under `--strict`, the two committed inputs of
  C-4 and no other.
- **Truth.** 0 misses at each door and in the port on the builder's world, the pass-5 and pass-6 fixtures, and the
  review's git-built world.
- **Recall.** main's committed corpora: 80 of 2,231 (none of their abstentions is a CONTRADICTED under `apart`).
- **Cost.** Every committed timing case under its bound; the three large cases within main's own call (Python) or
  twice it (port).

## 6. What this pass does not change, and what the operator decides

- **No repair.** PATH-2a never gives VERIFIED where `main` was wrong.
- **G-P1** is not met, and that is the operator's decision.
- **`main`'s reader** is untouched in both ports; the reconstruction test holds.
- **Nothing committed is edited:** no receipt, certificate, sworn file, PREREG, RESULT, ANALYSIS, AMENDMENT, ERRATUM,
  earlier NOTE or the charon log.
- **Operator options.** O-1 to O-12 stand; O-7 is re-measured above. Open for the operator: the strict splits (§1),
  O-7, the rename and `-U0` gaps (B-1, B-3), and whether the Action should import PyPI's styxx (I-1).
