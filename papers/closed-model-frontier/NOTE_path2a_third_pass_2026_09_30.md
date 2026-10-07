# NOTE — PATH-2a, third pass: the review of `40bba05b`, each finding and what this pass does with it

## 0. Status

2026-09-30. Branch `fix/diffgate-abstain-where-wrong`, on `origin/main` `1cde8b82`. Pass 2 ended at `40bba05b`. Its notes
are `NOTE_path2a_second_pass_2026_09_30.md` and `NOTE_path2a_second_pass_corrections_2026_09_30.md`; the pass-1 note is
`NOTE_path2a_abstain_overlay_2026_09_30.md`. None of them is edited. Four review lenses read `40bba05b`:

- by construction (A);
- coverage and recall (B, D);
- cross-port and runtime (C);
- integration, tests and docs.

No lens found a record that reads worse than `main`'s under the abstain-only relation. Lens C found two blockers and two
majors. Lenses A and B found one major each. There are sixteen minors.

**This note is written before this pass's code and committed alone, ahead of it. It is the only note of this pass.**
Pass 2 fixed four of its rules after its code existed (the review's finding I-6), so this pass worked the other way
round. Each design below was prototyped in scratch, outside the worktree, against the committed inputs, the review's
reproductions and the reviewers' input sets. The design was then fixed here. If the committed code departs from this
note, the departure goes into the next pass's note, not into a second note of this one. Figures quoted as reasons were
measured on that prototype. The figures at the head that carries the code go into `web/gate/README.md` and the
CHANGELOG entry.

Dispositions:
- **fixed**: a code or docs change in this pass. Where the finding is a defect of `40bba05b`, the fix comes with a test
  that fails on `40bba05b`.
- **disclosed**: kept as is and stated in the README, the CHANGELOG or here.
- **operator**: an option left to the operator, with measurements.

## 1. The blockers

### C-1 (bar C): the overlay read `main`'s claim text, which the two ports build differently

**The finding.** Python cuts a claim's text at 160 code points, the port at 160 UTF-16 units. `str.strip()` and
`trim()` differ on U+001C to U+001F, U+0085 and U+FEFF. The sentence splitter's `\s` differs on the same characters.
`_p2a_touches_wide` and the `only_touches` guard read that text, so the overlay split the two ports on claims where
`main`'s ports agree on kind, verdict, detail and reason. Examples:
- `Modified src/app.py.` + U+FEFF + `Tidied up.`;
- `Only touches docs/` + U+001F;
- the review's emoji release note (E2), where the CLI said FAIL and the bookmarklet PASS.

On the review's 20,000-input text-seam fuzz there were 2,078 split claims. The committed (C) test keyed on (kind,
verdict, text), so it could not see them.

**Fixed.** No decision reads a claim's text any more. A decision is a function of:
- the claim's kind, verdict and detail;
- `main`'s own counts in its reason (section 2);
- the door's bytes: the diff, the `--name-status` listing and, new in this pass, the summary.

Both Python doors pass `summary_text` to the facts object. The port's wrapper passes `summaryText`. Two consequences:
- Wherever `main`'s two ports give a claim the same kind, verdict and detail, the overlay decides it alike in both
  ports, whatever the claims' texts are.
- The 160-character cut is gone from the overlay, and with it lens B's and lens C's "past character 160" costs (B-1a,
  C-4).

The `extract` guard now reads the summary. Its purpose is unchanged: to fire in both ports where the two templates may
read a claim's path, name or prefix apart, so that a pair of claims with the same kind and verdict and different
details is decided alike too.

- **Paths.** A path holding a code point from 0x80 up keeps pass 2's rule (`_p2a_port_may_verify`, a function of the
  path and the diff). For an ASCII path, the summary is cut into maximal runs of path characters: ASCII letters, digits,
  `_ . / \ -` and the *wordish* code points defined below. The guard fires when some occurrence of the path lies inside
  a run that holds a wordish code point. An occurrence of an ASCII path lies inside exactly one maximal run, so the test
  is a bounded `find` per run that holds a wordish code point.
- **Names** (`symbol_added`). A name holding a code point from 0x80 up fires, as before. For an ASCII name, the same
  test is run over runs of name characters (ASCII word characters and wordish code points).
- **Prefixes** (`only_touches`). The guard fires when either prefix holds a *bad* code point (divergent or wordish).
  It also fires when some occurrence of the prefix lies in a sentence where a bad code point follows the sentence's
  earliest `only`, read case-blind on ASCII, and the occurrence starts after that `only`. Sentences here are cut where
  both ports cut one: at a line feed, and after `.`, `!` or `?` followed by a space, a tab or a carriage return. Each
  port's own sentence therefore lies inside one of these.
- **Declared claims** (DECLARE-1, `detail.declared`) skip the summary test. Their sentence is written by DECLARE-1 from
  a value that both ports read only when it is ASCII. Where both ports have such a claim, both templates read the same
  ASCII sentence.

**The character sets, all pinned by enumeration on every runtime the tests run.**

- **Neutral**, 11 code points: U+00A0, U+2013, U+2014, U+2018, U+2019, U+201C, U+201D, U+2022, U+2026, U+2192,
  U+3000. None of them is a word character in either port. None folds to an ASCII letter under CPython's `re.I`. Each
  is white space in both ports (U+00A0, U+3000) or in neither.
- **Wordish**: every code point from 0x80 up except the neutral ones and U+0085, U+2028, U+2029 and U+FEFF, which are
  never word characters either. These are the only code points CPython's templates may read as a word character, or
  fold to an ASCII letter, where the port's read neither. The port reads them as UTF-16 units. Every unit of a
  surrogate pair is wordish, and no boundary falls inside a pair.
- **Bad**: wordish, or divergent (pass 1's set `\x0b \x0c \x1c \x1d \x1e \x1f \x85 U+2028 U+2029 U+FEFF`).

**Why both ports fire where their details differ.** When the two templates extract different paths or names from one
match, the reason is one of these:
- CPython's `\w`, `\b` or `re.I` read a wordish code point inside or next to the run the path or name sits in. That
  code point lies in the run around the port's occurrence, and it lies in, or around, Python's.
- The prefix's sentence holds a bad code point after its `only`, which can move `\s+and` or the second prefix.

Reorderings are the other source of differing details at one position. A sentence that one port cuts and the other
does not reorders claims across templates. Those are not detail differences, and (C) now pairs claims by detail before
it pairs them by position (below).

**Costs, disclosed.**
- The guard reads every occurrence in the summary, not only the claim's own, because the claim's position differs
  between ports. A path or name that also occurs elsewhere next to a letter outside ASCII is withheld. On the committed
  inputs this moves five `symbol_added` claims from VERIFIED to `extract`, all of them #161's `x1` Unicode-name probes,
  for example "Added function fo²o. … Added function fo." (U+00B2 SUPERSCRIPT TWO).
- `only_touches` is withheld wherever a bad code point follows `only` in the same sentence, for example
  "Only touches docs/ 🎉".
- Pass 2's reason for `extract` asserted that the readers "extract it differently". That was not checked, and it was
  untrue on the review's inputs. The phrase becomes: "the summary holds a character the Python and JavaScript readers
  may read differently where the claim's path, name or prefix is read, so the two may extract the claim differently".

**The (C) test** is rebuilt on four keys, over the committed inputs, plus the review's fuzz sets in scratch:

1. Claims at the same position, in lists of equal length, with the same (kind, verdict, detail). The overlay's verdict
   and phrase must agree. This holds by construction.
2. Claims with the same (kind, verdict, text), as pass 2 had it.
3. Claims matched across the two lists by (kind, verdict, detail), in order.
4. In lists of equal length, the claims left unmatched, paired in order. Main's verdicts may agree there while the
   details differ. That is where the guard is needed, and any split there fails the test.

On inputs whose claims all pair on (kind, verdict) this way, the two gate verdicts must agree. On the prototype:
- 0 splits under every key, over the committed inputs and the review's 20,000 text-seam, 26,000 cross-port, 26,000
  hostile, 22,000 calm and 3,000 surrogate inputs, and the review's three 10,000-input hostile seeds;
- 0 gate splits on inputs whose claims all pair.

The review's R1, R2, R4 and E2 inputs, and a committed text-seam set, are pinned in both ports.

### C-2 (bar C): a claimed number of 10^21 or more is printed in exponent form by the port's `main`

**The finding.** The port's `main` builds its reason as `... claim says 1e+21`. The port's overlay could not parse it
and abstained (`unparsed`), while Python parsed its reason and kept CONTRADICTED. So the gate verdict split without
`--strict` on inputs where `main`'s ports agree on kind, verdict, text and detail. The review's N1 to N3 show it.

**Fixed.**
- **Where n comes from.** The claimed number is read from the claim's detail when that is ASCII digits (a static
  `[0-9]+` full match; `int()` of the group). Otherwise it is read from `main`'s reason, which covers CPython's reading
  of digits outside ASCII: its reason prints them as ASCII.
- **Where the count comes from.** `main`'s own count (`got`, `g`) is still read from the head of the reason, with a
  static `[0-9]+` group. It is always small and printed in digits by both ports.
- **Rule order.** The count rule checks for a dot twin, and the tests rule for a pairing, before parsing anything. With
  neither, the rule cannot fire, so an unparsed reason no longer withholds a claim that no reading could move.
- **Pins.** N1 to N3 are pinned pairs.

## 2. The majors

### A-1 (bar A): two per-claim scans were O(claims × diff)

`_p2a_case_doubt` and `_p2a_resolve` read a claim's whole base-name group once per claim and per space. That group is
the whole diff when every file is `__init__.py`. `_p2a_defines` was memoised per name only. Measured on `40bba05b`
against `main`:

| case | Python | node |
|---|---|---|
| 10,000 `pkg{i}/__init__.py` files, 1,000 claims | 27.5 s vs 0.16 s | 22 s |
| 500 symbols over 50,000 removed `def` lines | 57 s | — |
| 301 `class` claims over 20,000 NBSP lines | 38.8 s | — |

**Fixed**, with the same decisions:

- **Case doubt.** A registration whose fold form holds no code point from 0x80 up equals its wild form, and no other
  registration shares that wild form. So neither U1 nor U2 can hold for it, whatever the claim is. The facts keep, per
  space, the registrations whose fold form holds such a code point, grouped by base name, and the case-doubt loop reads
  only those. An all-ASCII diff reads none.
- **Resolvers.** With a non-empty base name, the keys matching a claim at some tier are exactly the keys of its base
  name. So `main`'s `find_path` is the earliest key of the claim's base-name group. V97 is:
  1. the claim itself, found by dictionary lookup;
  2. else the earliest key ending in `/` + the claim, from a per-diff map of every suffix after a `/` to its earliest
     key;
  3. else, for a bare claim only, the earliest key of the group.

  A claim with an empty base name keeps the key-by-key scan.
- **Symbols.** A name of ASCII word characters can start only where a site's coarse run ends (every position inside
  the run is coarse), and it ends where its ASCII run ends. So "a removed line defines it" is set membership in the
  ASCII runs at every `def` or `class` site, built once per diff over the distinct removed lines. Any other name keeps
  pass 2's scan. It never arrives there in practice, because `extract` withholds such a name earlier.

On every committed input and every review set, the prototype's case doubt, both resolvers and the symbol lookup give
the same answer as pass 2's scans, claim by claim: 0 differences over 967,627 comparisons. The costs are below.
Python is shown as `main` → prototype; Node as the prototype's extra time over `main`.

| case | Python | Node, extra over `main` |
|---|---|---|
| R1 | 0.033 → 0.058 s | |
| R2 | 0.017 → 0.058 s | |
| R3 | 0.069 → 0.226 s | |
| T1 | 0.167 → 0.312 s | +0.078 s |
| T4 | 0.117 → 0.192 s | |
| T5 | 0.010 → 0.063 s | |
| T9 | 0.043 → 0.099 s | |

Committed timing cases are added for:
- a same-base-name group (2,000 `pkgs/p{i}/__init__.py` with 200 claims);
- many symbols over a large removed side;
- `class` beside NBSP runs.

They time the overlay alone in both ports (see I-2). The README states the shapes the bounds were measured on.

### B-1 and C-4 (bar D): `extract` withheld right verdicts on ASCII paths `main`'s ports read alike

- **The >160-character trigger** is gone with C-1.
- **Typographic punctuation beside a path** (curly quotes, an attached em dash, NBSP, an arrow) is neutral, so it no
  longer fires.
- **An only_touches prefix** is read over its sentence, as both ports end one, from the sentence's `only`. It is no
  longer read over the rest of the claim text.

On the review's inputs:

| input | 40bba05b | this pass |
|---|---|---|
| "Updated “web/gate/diffgate.js”." | withheld | kept |
| "Created ‘web/gate/README.md’." | withheld | kept |
| "Updated web/gate/diffgate.js—the port only." | withheld | kept |
| "Only touches web/gate/ — the port and its README." | withheld | kept |
| the 185- to 255-character ASCII sentences | withheld | kept |
| E2 | port only withheld | kept in both |

All of these are pinned. The reason is reworded (C-1). **Fixed.**

### C-3 (bar C): the static checks accepted runtime Unicode queries

**The finding.** The Python self-check accepted nine plants:
- `import unicodedata`;
- `import re as r2` with `\W`;
- `from re import sub`;
- `split(None)` and `strip(None)`;
- `builtins.repr`;
- `__getattribute__`;
- `operator.methodcaller`;
- `map(str.__str__, ...)`.

The port's token scan accepted five:
- `RegExp("\\" + "s", "g")` without `new`;
- `new RegExp("\\" + "s")`;
- a name built outside the brackets;
- `normalize ("NFKC")` with a space;
- `claimed. trim()`.

**Fixed.**

- **Python.** These are refused:
  - any import;
  - any name the block does not bind, other than a short list: builtins that read no table, and `main`'s `re`,
    `_Pending`, `PATH1_EXTENSIONS`, `_PATH_KINDS` and `DiffGate`;
  - any attribute outside a short list of the methods and fields the block uses. That refuses every dunder, and
    `expandtabs`;
  - `re` used other than as `re.<function>(<static pattern>)`. That refuses `rc = re.compile`;
  - an attribute read off a builtin type (`str.split`);
  - `str()` of a value, which can read `repr`;
  - a strip or split with no argument, with `None`, or with a keyword.
- **The port.** The scan reads code outside strings and comments with white space removed, except between two
  identifier characters, so `a . b (` reads `a.b(`. It then refuses:
  - `RegExp` other than `new RegExp(one argument)`, where the argument must be a static string: string literals and
    top-level constants joined by `+`. The scan decodes the string's escapes and applies the class-escape, `.` and
    flag rules to its value;
  - a computed member access whose index is not one of a listed dozen expressions (`i`, `k`, `k+1`, `0`, `space`,
    `c.kind`, …);
  - any call on a computed member (`x[k]()`, `x[k]?.()`) or on a parenthesised expression (`(x[k])()`);
  - `.bind(`, `require(`, `import(`, `.compile(`, `__proto__` and `constructor`, added to the banned tokens.

Each of the review's plants, and a few more, are committed as refused cases.

**Disclosed.** These are static checks and do not infer types. An f-string or `%s` of a list, which reads `repr`,
would pass. The block holds none. That case is disclosed rather than chased.

## 3. The minors

| id | finding | disposition |
|---|---|---|
| A-2 | The Action shows a reason whole if it contains " withheld by PATH-2a (" anywhere. A PR author controls that text in a DECLARE-1 MALFORMED reason. | fixed: a reason is shown whole only when the claim is UNCHECKABLE, its kind is one the overlay may move, and the reason starts with the overlay's form. No reason `main` writes for those kinds starts that way. The review's declaration is a test. |
| A-3 | `test_git_door` asserts door agreement beyond where it holds. There are two causes: the divergent guard is raw-door only, and a no-prefix header `main`'s raw reader drops is still in `--name-status`. | fixed: the assertion is narrowed to inputs where `main`'s two doors agree on the claims, `main`'s two status maps are equal, and the raw door's divergent guard is false. The review's no-prefix repository and a U+2028 path under `core.quotepath=false` join the real-git cases, each read at both doors against its own `main`. The limit is stated in the README. |
| B-2 | `only` compares a prefix set neither reader uses, so it withholds "Only touches .github/workflows/ and github/workflows/." (right). | fixed: V121 (dots kept) is compared under the prefix set it uses with `main` under the set `main` uses; nothing else. The case check reads only those sets and the second prefix's shape. World B's 5 right `only` verdicts are no longer withheld, and no truth-judged false verdict is kept. |
| B-3 | `dot_earliest` caught no false verdict in two independent worlds, and on git's bytes it is the only path phrase that loses right verdicts. | operator (O-6): the review's figures go into the README and here. World A: 18 right, 4 mixed, 0 false. World B: 31 right, 16 mixed, 0 false. At world B's git door it withholds 12 of 8,859 right path verdicts. |
| B-4 | The `tests` trigger over-fires more on independent data than the README's table shows. | disclosed, and operator (O-8): the independent right-lost rates go next to the builder's table. World A: raw door 162 of 2,331 (7.0%), git door 69 of 1,244 (5.5%). World B: 472 of 6,939 (6.8%), 195 of 3,669 (5.3%). O-8 would pair by ASCII run only when the name is all ASCII. It is not taken here. |
| B-5 | The `dir` phrase says "the claim names a directory" of file claims. | fixed: "the claim's path has a directory part, and only a changed file with the same base name in another directory matches it", in both ports, the README and this note. `path2a_moves.json` and the pinned expects are re-pinned. |
| B-6 | `dir`'s recall cost on mnemonic, plain, index and no-prefix renderings is not stated. | disclosed with the review's world-B figures (raw door, right path verdicts withheld): mnemonic 158 of 1,189, plain 48 of 990, index 42 of 890, no-prefix 36 of 1,050. On git's own bytes, `dir` withholds none. |
| C-5 | The README does not say that the overlay adds gate disagreements where `main`'s two gate verdicts agree but some claim rows differ. | disclosed: the README and CHANGELOG name the 5 committed inputs (#161's four f2 separator cases and y2). A test pins that set, so a sixth fails. The reviewers' hostile fuzz gives 359 without `--strict` and 17 with it. Operator option O-9: abstain in both ports wherever either port's `main` reads the claim so that a rule fires. Not taken. |
| I-1 | `check_path2a.js --relation` cannot see `unparsed_claims` or an extra claim key. | fixed: `--relation` keeps `unparsed_claims` (both sides are the port) and compares each claim pair's keys. The review's two mutants are port plants that must be refused. |
| I-2 | The 2,000-file timing test subtracts two single wall-clock samples and can flake. | fixed: the overlay is timed alone (`main`'s record built, then `_p2a_abstain` / `_p2aAbstain` timed on it), the minimum of three runs, in both ports. |
| I-3 | `UNICODE_SAME` accepts Unicode 15.1.0, which nobody measured. | fixed: 15.1.0 is dropped. The CHANGELOG says 13.0 and 14.0 (emulated), 15.0 and 16.0. |
| I-4 | The README says 280 KB where the test line is 220 KB, and gives cross-port figures without their runtime. | fixed: 220 KB, and every figure names its runtime. |
| I-5 | A banned word in a test comment. | fixed. |
| I-6 | Pass 2 had two notes, the second written after its code. | recorded above: one note this pass, before the code. |
| I-7 | The port checks skip, and so pass, when `node` is absent. | fixed: under `CI` or `GITHUB_ACTIONS` a missing `node` fails the PATH-2a modules. No `.github/` file is touched. |
| I-8 | `test_build_bookmarklet_check` rewrites a tracked file. | fixed: the check runs on a copy of `web/gate` in a temporary directory. |

## 4. The rules after this pass

Everything in the pass-1 and pass-2 notes stands, with these changes.

**Order.**
- Path claims: `divergent`, `extract` (pass 2's code already read `divergent` ahead of `extract`), `odd`, `case`,
  `unreproduced`, then the three readers.
- `only_touches`: `divergent`, `extract` (the summary test above), `case`, `unreproduced`, `only`.
- `files_changed_count`: `divergent`; no dot twin keeps; `unparsed`; `unreproduced`; the interval.
- `tests_added`: no pairing keeps; `unparsed`; `unreproduced`; the interval (`tests`, `split`).
- `symbol_added`: `extract`; `symbol`.

**`only`.** Keep the claim when V121 reads "every changed path lies under the prefix set it uses" exactly as `main`
reads it under the set `main` uses. `case` fires only when folding and the placeholder disagree on the second prefix's
shape, or on that answer, in either space.

**Phrases changed:** `dir` and `extract`, as above. No key is added or removed.

The reconstruction property, REACH, the reason form, the error fallback and the abstain-only relation are unchanged.

## 5. Predictions, falsifiable, for the head that carries this pass

- **Relation.** No committed input's record breaks the relation (A) in either port, at either door, in either strict
  mode. The error phrase never appears.
- **Cross-port.** All four (C) keys give 0 splits on the committed inputs, including the new pins.
- **Truth test pins.** They do not move:

  | where | attributable withheld | right verdicts lost | undecided withheld |
  |---|---|---|---|
  | raw door | 1,206 of 1,206 | 202 | 35 |
  | the port | 1,100 of 1,100 | 126 | 122 |
  | git door | 14 of 14 | | |

  The prototype gives these same figures.
- **Committed-input abstention pins.** Only `symbol_added:extract` moves, by +5 on the inputs of `40bba05b` (the `x1`
  probes). The new pinned pairs add their own abstentions.
- **Recall on `main`'s committed corpora** stays 80 of 2,231. With #161's pairs it goes from 264 to 269 (the same five
  probes).
- **The review's two independent worlds.** 0 misses. World B keeps its 5 right `only` verdicts.

## 6. What this pass does not change

- **No repair.** PATH-2a still never gives VERIFIED where `main` was wrong.
- **G-P1** is still not met, and that is still the operator's decision.
- **`main`'s reader** is untouched in both ports; the reconstruction test holds.
- **Nothing committed is edited:** no receipt, certificate, sworn file, PREREG, RESULT, ANALYSIS, AMENDMENT, ERRATUM,
  earlier NOTE or the charon log.
- **Operator options.** O-1 to O-7 stand. O-8 (ASCII-run pairing only for ASCII test names) and O-9 (both ports abstain
  wherever either port's reading fires) are added, measured where the review measured them, and not taken.
