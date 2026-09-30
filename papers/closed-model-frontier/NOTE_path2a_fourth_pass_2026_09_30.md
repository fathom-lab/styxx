# NOTE — PATH-2a, fourth pass: the review of `ea677740`, each finding and what this pass does with it

## 0. Status

2026-09-30. Branch `fix/diffgate-abstain-where-wrong`, on `origin/main` `1cde8b82`. Pass 3 ended at `ea677740`. Its
note is `NOTE_path2a_third_pass_2026_09_30.md`; the earlier ones are `NOTE_path2a_abstain_overlay_2026_09_30.md`,
`NOTE_path2a_second_pass_2026_09_30.md` and `NOTE_path2a_second_pass_corrections_2026_09_30.md`. None of them is
edited. Four review lenses read `ea677740`:

- by construction (A): 0 records outside the relation over 8,518 committed-input runs, 48,600 hostile, 24,000 calm,
  600 huge and 8,000 lone-surrogate fuzz runs, the same on CPython 3.14.2, the POSIX flavour, the port, the rebuilt
  bookmarklet and 660 plumbing-built repositories at the git door. One major (cost), one minor (cost);
- coverage and recall (B, D): two majors, one minor;
- cross-port and runtime (C): one blocker, two majors, one minor; one input where the gates split that `main`'s did
  not (the blocker);
- integration, tests and docs: seven minors, verdict ship.

**This note is written before this pass's code and committed alone, ahead of it.** Each design below was prototyped
in scratch, outside the worktree (`scratchpad/ghmerge/p2a/b4`), against the committed inputs, the review's
reproductions and the reviewers' input sets, and then fixed here. Figures quoted as reasons were measured on that
prototype (CPython 3.12.10 and 3.14.2, Node 24.13.0). If the committed code departs from this note, the departure goes
into the next pass's note. The figures at the head that carries the code go into `web/gate/README.md` and the CHANGELOG
entry. Finding identifiers are this note's; code comments and pinned pairs cite them.

Dispositions:
- **fixed**: a code, test or docs change in this pass. Where the finding is a defect of `ea677740`, the fix comes with
  a test that fails on `ea677740`.
- **disclosed**: kept as is and stated in the README, the CHANGELOG or here.
- **operator**: an option left to the operator, with measurements.

## 1. The blocker

### C-1 (bar C): a count claim's number read two ways from one match

**The finding.** The count template reads `\b(?P<n>\d+)\s+files?...`. CPython's `\d` takes a digit outside ASCII, and
CPython's `\b` does not fall between it and the ASCII digits after it; the port's `\b` does. On "３3 files changed."
(a full-width 3, then 3) CPython reads `n = '３3'` (33) and the port `n = '3'`, at the same position, with the same text.
With a dot twin in the diff, `ea677740` kept the Python's CONTRADICTED (33 lies outside the dot-kept range [3, 3]) and
withheld the port's (3 lies inside it), so the gates went FAIL / PASS where `main`'s were FAIL / FAIL. The review's X5,
X5b (Arabic-Indic three), X2 and X3 (each port reading a count the other does not) and fuzz `r3:4:3256` show it; its
token-adjacency fuzz found 3 same-text splits, all counts.

**Fixed.** The count rule gets the extract guard the path, name and prefix rules have. After the rule's no-twin return
(where it keeps the claim in both ports, whatever each reads) and before it parses a number, it withholds with `extract`
when either holds:
- (a) the claim's detail `n` is not ASCII digits (only CPython reads such a number);
- (b) the claim is not declared, and some occurrence of `n` in the summary lies inside a maximal run of ASCII digits
  and wordish code points (§2, B-2) that holds a wordish code point. This is `_p2a_in_runs` with a third run class.

Where the two templates read one match's number apart, a wordish code point sits in that run in both ports' summaries,
so both fire. The `extract` phrase names the number: "... where the claim's path, name, prefix or number is read ...".

**On the prototype.** X5, X5b, X2 and X3 give UNCHECKABLE in both ports and the same gate verdict; the review's 20,000
token-adjacency inputs give 0 splits under every key of §2 C-2 (at `ea677740`: 3 same-text splits, 4 left-over splits
and 3 all-pair gate splits). The committed inputs move by +4 `files_changed_count:extract`, all of them new pinned pairs.

**Pins.** X5, X5b, X2 and X3 join `path2a_pairs.json` as kind-and-verdict pairs (the two ports' reasons print different
numbers, so one expect cannot hold a reason), and `XPORT_CASES` with each port's phrase. A pair where both ports read
`3` alike and `3` also occurs beside `é` is pinned too, so that a port-only plant of the guard splits the ports.

## 2. The majors

### A-1 (bars A, D): the summary-based extract guard was quadratic

**The finding.** `_p2a_in_runs` and `_p2a_scope_doubt` (and the port's `_p2aInRuns`, `_p2aScopeDoubt`) walked every
wordish run or zone of the summary for every path, symbol and scope claim, with nothing memoised. On a summary at
GitHub's body limit (65,529 characters: 2,184 "Modified a.py." claims and 16,384 runs of `é`) the overlay alone took
3.76 s in Python against the committed 0.5 s bound; 51.2 s at 262,143 characters. The committed timing cases held no
summary with runs or zones.

**Fixed.** The runs of each class are built once per summary and joined by U+0000, which no run holds (the run classes
exclude it), and no claimed path, name, prefix or number holds (the templates' classes exclude it); an occurrence of a
claimed string lies inside exactly one maximal run, so "some run holds it" is one `in` over the joined string, memoised
per (class, string). A string holding U+0000 lies in no run and reads False, which is what the loop gave. The zones are
joined the same way; a zone may hold U+0000, a prefix cannot, so a match cannot straddle two zones; a prefix holding
it (none can) falls back to the loop. The decisions do not change: on every committed input and the reviewers' sets
the prototype's two answers equal the loops' answers.

**On the prototype** (the overlay alone, least of three, on `main`'s record): Q1 3.76 s → 0.030 s, Q2 (symbols) 2.59 s
→ 0.011 s, Q3 (zones) 0.63 s → 0.009 s, Q1 at 262,143 characters 51.2 s → 0.12 s; in the port Q1 takes 23 ms and Q1 at
262,143 characters 83 ms. Q1 to Q3 and a count case (Q4) at 64 KB join `_timing_cases` in both ports under the existing
bounds (0.5 s, 0.3 s); Q1 fails on `ea677740`.

### B-1 (bar B): an `only_touches` claim V121 abstains on, kept

**The finding.** With two prefixes, where the leading one is path-shaped only because `_norm` drops a leading dot
("Only touches github and .github/workflows/." against `.github/...` paths), `main` decides and V121 abstains with
"prefix 'github' is not a path (#110)". `_p2a_scope` computed the shape of the second prefix only, so where the two
under-readings agreed the overlay kept `main`'s verdict, false in the review's three reproductions: a GNU per-file
deletion (`+++ /dev/null\t1970-...`), a mnemonic second header, and real git bytes for a rename into `.github/` (at
both doors). The keep-agreement scan found this the only class of kept claim a variant reads otherwise (99 on the
committed inputs, 49 on the review's fuzz sets).

**Fixed, option (a).** `_p2a_scope` also computes the leading prefix's shape in each space and case reading, and the
`case` comparison covers it. In `_p2a_only`, after `unreproduced` (which now also fires if the overlay does not find the
leading prefix path-shaped in `main`'s space, where `main` decided), a claim with a second prefix whose leading prefix
is path-shaped in `main`'s space but not in V121's abstains with a new phrase key, `shape` (#121): "with leading dots
kept, no changed path has the prefix as a segment, so it is not read as a path".

**Why not a single prefix (option (b), operator O-10).** With one prefix whose shape differs by space, V121 reads no
changed path under it (none has it as a segment), so the existing `only` rule already withholds a VERIFIED; what is
kept is a CONTRADICTED, and it is false only if every changed path lies under the dotted directory while `main` lists
some path outside it, which needs the diff's paths to differ from the changed files twice over (a rendering fault that
hides the undotted path and one that invents an outside path). Option (b) would withhold 141 right verdicts in the
builder's world, 95 in world A and 239 in world B (the review's figures) to cover that. Not taken; the attribution
definition in the README, the truth test and here is narrowed to say so, and the residue is measured.

**On the prototype.** The three reproductions abstain `shape` at the raw door and the rename at the git door too. Right
verdicts lost: +43 in the builder's world in each port (1,706 cases: raw door 202 → 245, port 126 → 169), 0 in worlds A
and B (raw door 297 and 785, port 297 and 780, git door 75 and 210, as at `ea677740`); 0 misses in either world. On the
inputs pass 3 pinned, 25 claims move to `shape` (24 kept before, 1 `only`), all on the seeded PATH-2a fuzz. The
keep-agreement scan leaves only single-prefix CONTRADICTED claims: 76 on the committed inputs, 37 on the six fuzz sets.

**Pins.** The three reproductions (raw door) and the right verdict withheld (the cost) and a single-prefix CONTRADICTED
kept (the gap) are pinned pairs; the rename is a real-git case at both doors; a truth test judges all four readings.

### B-2 (bar D): common typographic punctuation still withheld right verdicts

**The finding.** Only 11 code points were neutral, so `extract` and the zone rule withheld right verdicts beside French
guillemets, German low-9 quotes, CJK punctuation, ✔, ·, §, ⇒ or ✅, where `main`'s two ports read the claim alike and
right (16 of 16 on the review's probe). The README said typographic punctuation no longer withheld anything.

**Fixed.** The neutral set becomes 1,828 code points of the Basic Multilingual Plane, as a regex class in Python and a
range table in the port: the punctuation and symbols (general categories P and S, connector punctuation left out) of
Latin-1, General Punctuation, Currency Symbols, Arrows, Mathematical Operators, Miscellaneous Technical, Box Drawing,
Block Elements, Geometric Shapes, Miscellaneous Symbols, Dingbats (the circled digits left out), Miscellaneous Symbols
and Arrows, CJK Symbols and Punctuation (its letters, marks and numbers left out), the vertical, compatibility and
small forms, full-width punctuation; the white space both ports share (U+00A0, U+1680, U+2000 to U+200A, U+202F,
U+205F, U+3000); and the variation selectors U+FE0E and U+FE0F that follow emoji. Unassigned code points are left out.

Pinned by enumeration on every runtime the tests run: none is a word character in CPython (`\w`, `isalnum`), none folds
to an ASCII letter under `re.I`, none has a case in either port (`lower`, `upper`, `casefold`; `toLowerCase`,
`toUpperCase`), each is white space in both ports or in neither, all lie below U+10000 (one UTF-16 unit each), and each
one assigned in Unicode 3.2 (`unicodedata.ucd_3_2_0`) was punctuation, a symbol or a space there too, so none became a
letter, mark or number between the versions CI carries. The wordish set is the complement, as before.

**Disclosed.** A letter outside ASCII (an accented name, "reviewed by José") after `only` in the same sentence still
withholds a scope claim, and astral emoji (🎉, 🚀) stay wordish: a pair of UTF-16 units in the port, one code point in
CPython, and a Python string can hold the two surrogates apart, which the port cannot see. Operator option O-11: treat
astral pictographs as neutral by pair arithmetic in both ports.

**On the prototype.** The review's probe: 15 of the 16 right verdicts kept; José's withheld. On the inputs pass 3
pinned, four claims move: three `extract` are kept, one becomes `dir`. One of the three is #161's `w2` probe ("Added
function col·leccio."): both ports' templates read `col`, `main` verifies it and is wrong for a reason outside #97, #121
and #101, and the overlay no longer withholds it. The README's sentence is corrected.

### C-2 (bar C): the README and the (C) test asserted what the design does not guarantee

**The finding.** Key 4 paired left-over claims in order, including claims each port reads from a different sentence,
and asserted that they decide alike, and that inputs whose claims all pair so get one gate verdict. The review's
realistic X6b ("Changed the parser in `docs/résumé/index.md`. Added `docs/café.md`.") pairs Python's `docs/café.md`
with the port's `/index.md` and splits them; so do fuzz inputs.

**Fixed.**
- By construction, and asserted: claims `main`'s two ports give the same (kind, verdict, detail) are decided alike, at
  the same position, matched across the lists in order, and anywhere in either list (a new key, "any").
- Measured on named sets, and asserted on the committed inputs only: the same (kind, verdict, text); and left-over
  claims of one kind and verdict whose details differ but may be one match read two ways, that is, they read the same
  fields, each field's two values nest, and each claim's values lie in the other's text. Other left-over claims are not
  paired, and an input holding one is not one "whose claims all pair".
- The README and CHANGELOG say which is which. X6b is pinned in `XPORT_CASES` (Python keeps, the port withholds, the
  gates agree).

On the prototype, the committed inputs and the reviewers' fuzz sets give 0 splits under every key and 0 gate splits
where every claim pairs; the review's `r3:2:6972` (`İndex.md` against `/app.py`, two matches in one sentence) is not
paired.

### C-3 (bar C): the static checks accepted runtime Unicode queries

**The finding.** The port's scan accepted `claimed.match("\\" + "s")`, `.search(...)`, and `normalize` taken as a value
onto a `new String`. The self-check accepted `int()` of any regex group, and a plant widening `_P2A_DIGITS` to `[^,]+`
passed; `int()` reads CPython's table of decimal digits.

**Fixed.**
- Python: `int` is a banned name. The block reads ASCII digits through `_p2a_int`, a fixed table of the ten digits;
  anything else raises, and the error fallback withholds. The `[^,]+` plant then reads no table (its non-ASCII digit
  raises), and three `int()` plants are refused cases.
- The port: `normalize` (as a bare token), `new String(`, `.match(`, `.search(` and `.matchAll(` are banned tokens.
  The block uses `.test` and `.exec` on constant RegExps. `parseInt` stays: it reads ASCII digits only, and every string
  it gets is `[0-9]+`. The review's plants are committed refused cases.

## 3. The minors

| id | finding | disposition |
|---|---|---|
| A-2 | On a 35 MB diff the overlay costs 2.4x to 5.3x a `main` call in both ports, and peak memory rises by 42%. The recall costs were measured on corpus-sized inputs only. | fixed in part, and disclosed: one split at CPython's line breaks is shared by the registrations and line view 0; the port's `_splitlines` split by its registrations and view 1; the other split is skipped when the text holds no break only one of them reads, and the two views are then one object, read once for removed definitions and counted once when no character where the two mains' white space parts is present. A `def test_` count reads each segment's leading run once, not each occurrence. On the prototype: Python 0.49 → 1.33 s, 0.35 → 0.87 s, 0.48 → 1.60 s (×2.7, ×2.5, ×3.4; `ea677740` measured here ×4.0, ×2.7, ×5.1); the port ×2.1, ×2.0, ×3.5; peak memory 122 MB on `main` and 127 MB (review: 128 and 182). What remains is the overlay's own read of the diff; the README states the factors. |
| B-3 | The symbol rule read `def` or `class` anywhere in a removed line, so prose, comments, JavaScript classes and `#undef` lines withheld a new Python definition `main` verifies; V101's symbol check is anchored. | fixed: removed-side symbol sites are anchored where V101 reads one, at the line start (or, a superset, after U+2028 or U+2029): a run of coarse characters, optionally `async` and coarse characters, then `def` or `class` and one or more coarse characters. Still a superset of both ports' V101. The review's four symbol probes are kept; an indented method is still read. The test pairing stays unanchored, as V101's is: `def test_` in removed prose withholds a `tests_added` claim (disclosed, pinned). Worlds A and B: the symbol rule loses no right verdict and misses nothing. |
| C-4 | The README gives hostile-set gate figures only. | fixed: realistic-text figures go beside them (non-strict and strict new gate splits where `main`'s agree, and how many of `main`'s own splits the overlay makes agree), and the README says the bookmarklet runs non-strict. |
| I-1 | README: "the 110 pinned pairs". | fixed: the count at the head (54 of `main`'s plus this branch's). |
| I-2 | README: `py_side.py --installed` prints 80 disagreements. `differential.py` counts pairs. | fixed: "75 pairs holding the overlay's 80 abstentions" (re-measured at the head). |
| I-3 | README, CHANGELOG and a test comment: "each now takes under 0.1 s"; the symbols case takes 0.10 to 0.16 s. | fixed: the least observed per case, with the spread, measured at the head as the test measures it. |
| I-4 | The Action test pins only the kind condition of A-2 (pass 3). | fixed: two synthetic rows (the form mid-string, and the form on a decided claim) are cut at 100 characters, and the overlay's own reason is shown whole, through a patched `gate_diff_text`. |
| I-5 | `test_the_javascript_port_agrees_on_every_pinned_pair` still skips green without `node` under CI. | fixed: the same CI rule as the PATH-2a modules (fail under `CI` or `GITHUB_ACTIONS`). No `.github/` file is touched. |
| I-6 | Commit `3db9ad74` (pass 3's code) is red against its own tree: its tests are `40bba05b`'s, which pin the old hook lines; its message did not say so. | recorded here; history is not rewritten. This pass lands its code with the tests, pairs and bookmarklet it needs in one commit. |
| I-7 | The suite fails by design on CPython 3.13 (Unicode 15.1), which `requires-python` admits. | operator: unchanged. Unicode 15.1 was not measured here (no 3.13 interpreter); the pin fails there by name. |

## 4. The rules after this pass

Everything in the earlier notes stands, with these changes.

- **`files_changed_count`**: `divergent`; no dot twin keeps; `extract` (§1); `unparsed`; `unreproduced`; the interval.
- **`only_touches`**: `divergent`; `extract`; `case` (now over the leading prefix's shape too); `unreproduced` (now
  also when `main`'s leading prefix is not path-shaped here); `shape` (a second prefix claimed, the leading prefix
  path-shaped for `main` and not for V121); `only`.
- **`symbol_added`**: `extract`; `symbol`, over anchored removed-side sites.
- **Neutral**: 1,828 code points (B-2). **Wordish**: every other code point from 0x80 up but U+0085, U+2028, U+2029,
  U+FEFF.
- **Phrases**: `extract` names a number too; `shape` is added.
- **Static checks**: `int` banned (Python); `normalize`, `new String(`, `.match(`, `.search(`, `.matchAll(` banned
  (port).

The reconstruction property, REACH, the reason form, the error fallback and the abstain-only relation are unchanged.

## 5. Predictions, falsifiable, for the head that carries this pass

- **Relation.** No committed input's record breaks the relation (A) in either port, at either door, in either strict
  mode; the error phrase never appears.
- **Cross-port.** 0 splits under every asserted key on the committed inputs, and 0 gate splits where every claim
  pairs; the five inputs where `main`'s gates agree and the overlay's differ are the same five.
- **Truth pins.**

  | where | attributable withheld | right verdicts lost | undecided withheld |
  |---|---|---|---|
  | raw door | 1,206 of 1,206 | 245 | 35 |
  | the port | 1,100 of 1,100 | 169 | 122 |
  | git door | 14 of 14 | 0 | |

- **Committed-input abstention pins** (Windows flavour): decided 12,675; `only_touches:shape` 29,
  `files_changed_count:extract` 4; `file_touched:extract` 144, `file_created:extract` 48, `symbol_added:extract` 24;
  the rest as the prototype prints them.
- **Recall.** `main`'s committed corpora: 80 of 2,231, unchanged. With #161's pairs: 268 of 2,761 (the `w2` probe is
  kept).
- **Independent worlds.** 0 misses; right verdicts lost as at `ea677740`.

## 6. What this pass does not change

- **No repair.** PATH-2a still never gives VERIFIED where `main` was wrong.
- **G-P1** is still not met, and that is still the operator's decision.
- **`main`'s reader** is untouched in both ports; the reconstruction test holds.
- **Nothing committed is edited:** no receipt, certificate, sworn file, PREREG, RESULT, ANALYSIS, AMENDMENT, ERRATUM,
  earlier NOTE or the charon log.
- **Operator options.** O-1 to O-9 stand. O-10 (single-prefix `shape`, B-1) and O-11 (astral pictographs neutral,
  B-2) are added, measured where they could be, and not taken.
