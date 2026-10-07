# NOTE — PATH-2a, fifth pass: the review of `5ebe0b6b`, each finding and what this pass does with it

## 0. Status

2026-09-30. Branch `fix/diffgate-abstain-where-wrong`, on `origin/main` `1cde8b82`. Pass 4 ended at `5ebe0b6b`. Its
note is `NOTE_path2a_fourth_pass_2026_09_30.md`; the earlier ones are `NOTE_path2a_abstain_overlay_2026_09_30.md`,
`NOTE_path2a_second_pass_2026_09_30.md`, `NOTE_path2a_second_pass_corrections_2026_09_30.md` and
`NOTE_path2a_third_pass_2026_09_30.md`. None of them is edited. Four review lenses read `5ebe0b6b`:

- by construction (A): 0 records outside the relation on 45,891 raw inputs and 18,000 git-door runs, in CPython 3.12.10
  and 3.14.2, under the POSIX flavour, in Node 24.13.0 and in the committed bookmarklet. One blocker (memory quadratic in
  path length, and the error phrase under a memory cap), one major (cost per claim over the registrations), two minors;
- coverage and recall (B, D): two majors (continuation lines and lone CRs; NFKC names), two minors;
- cross-port and runtime (C): one blocker (a count each port reads alone), one major (static checks), three minors;
- integration, tests and docs: two majors (the PR body; the DECLARE-1 skips unpinned), five minors.

**This note is written before this pass's code and committed alone, ahead of it.** Each design below was prototyped in
scratch, outside the worktree (`scratchpad/ghmerge/p2a/b5`), against the committed inputs, the reviews' reproductions
and five fuzz sets built with the reviews' own generators (34,000 inputs: `gen4` at two seeds, `gen`, `gen_tok`,
`gen_real`), and then fixed here. Figures quoted as reasons were measured on that prototype (CPython 3.12.10, Node
24.13.0). If the committed code departs from this note, the departure goes into the next pass's note. The figures at
the head that carries the code go into `web/gate/README.md` and the CHANGELOG entry. Finding identifiers are this
note's; code comments and pinned pairs cite them.

Dispositions:
- **fixed**: a code, test or docs change in this pass. Where the finding is a defect of `5ebe0b6b`, the fix comes with
  a test that fails on `5ebe0b6b`.
- **disclosed**: kept as is and stated in the README, the CHANGELOG or here.
- **operator**: an option left to the operator, with measurements.

## 1. The blockers

### A-1 (bar A): the suffix index held every suffix of every key

**The finding.** `_P2aFacts.index` stored, for each key of the status map and each '/' in it, the suffix after that '/'
(both key spaces). Memory and time grew with the square of path length: one 40 KB path took 0.64 s and 766 MB where
`main` takes 4 ms; 100 paths of 4 KB took 1.8 s and 799 MB. Under a 1,024 MB memory cap the overlay's own
`MemoryError` fell back to the `error` phrase and withheld a right VERIFIED on `README.md`, which the relation (A)
forbids. The port's `suffix.set(p.slice(k + 1))` was quadratic in time. No committed timing case had a deep path.

**Fixed.** No suffix is stored. V97's suffix tier needs, for a claim c, the earliest key (in `main`'s order) that ends
in '/' + c. A key ends in '/' + c exactly when its '/'-segments end in c's segments and it has at least one more. So:
- `_p2a_abstain` names the claimed paths of the claims in reach to the facts object before any claim is read
  (`prime`);
- `_p2a_ends(texts, claims)` builds a tree of the claims' segments read from their ends, then reads each key from its
  end one segment at a time through that tree, and only while the segments read so far end some claim's. It records,
  per claim, the earliest key that ends in '/' + it, and how many keys do (the count serves A-2). Time is linear in the
  keys and the claims; memory is linear in the claims.
- A claim not named before (none, in the doors) is read on its own, the same way.

What a claim's reading returns depends on that claim and the keys only, never on the other claims in the tree, so the
two ports still decide a claim from its own kind, verdict and detail.

**On the prototype** (the overlay alone, least of three; peak by `tracemalloc`): one 40 KB path 0.50 s / 766 MB → 2 ms
/ 0.3 MB; 100 paths of 4 KB 0.69 s / 799 MB → 15 ms / 3.6 MB; a run of 40,000 `./` 0.26 s → 22 ms (A-3). On the
committed inputs and the 34,000 fuzz inputs no path claim's decision moves against `5ebe0b6b`.

**Pins.** Both committed timing tests gain the deep-path cases (one path of 40 KB; 100 paths of 4 KB, each with a
`README.md` beside them and claims on it). A new Python test runs every timing case under `tracemalloc` and bounds the
overlay's peak at 64 MB; `5ebe0b6b` fails it. The error phrase is looked for again under the reviewer's memory cap.

### C-1 (bar C): a count each port reads alone

**The finding.** The count guard of pass 4 looked only at runs that hold the claim's own number. Two different
matches, one per port, passed it on the port that reads a clean ASCII number. The reviews' X2-13 ("13 files" U+0085
"changed. 共5 files changed."), G1 (U+001F), G2 (U+FEFF, and "fileſ") and G3 (U+0085): `main`'s gates are FAIL / FAIL,
and at `5ebe0b6b` the overlay's were FAIL / PASS or PASS / FAIL. The pinned X2 and X3 agreed only because 3 fell inside
the dot-kept range.

**Where the two count templates part.** The template is `\b(?P<n>\d+)\s+files?\s+(?:were\s+)?changed` with IGNORECASE.
It reads apart in three ways:
- (D1) `\b` and `\d` beside a code point from 0x80 up. CPython's `\b` holds only where the port's does, since every
  ASCII word character is a word character to both. So a count CPython alone reads has a number outside ASCII, which
  pass 4's guard (a) withholds. A count the port alone reads has a wordish code point before its number, and guard (b)
  withholds it.
- (D2) `\s`: U+001C to U+001F and U+0085 are white space to CPython only, U+FEFF to the port only. A count read across
  one of them is read by one port alone, and its number can be clean.
- (D3) IGNORECASE: CPython folds U+0130, U+0131, U+017F and U+212A to ASCII letters (`fıles`, `fileſ`); the port's
  non-Unicode IGNORECASE folds none. A count read through one of them is also read by one port alone, with a clean
  number.

The sentence splitter parts on the same white space, but only after '.', '!' or '?', which no count match holds.

**Fixed.** A summary-level seam. `_P2A_COUNT_SEAM` holds the six one-port white-space code points and the four
fold code points. Where the summary holds any of them, every non-declared count claim is withheld with a new phrase
key, `seam` (#121): "the summary holds a character that only one of the Python and JavaScript readers reads as a space,
or as a letter, where a count is read, so the two may read different counts". The seam is read ahead of the `extract`
guard, so both ports name the same phrase whichever count each reads. A declared count is written by DECLARE-1 and
skips the seam, as it skips `extract`. Tests pin both halves of the set by enumeration: the white space as the
symmetric difference of the two ports' pinned `\s` tables, and the fold letters as every code point from 0x80 up that
CPython's IGNORECASE matches against `[A-Za-z]` (the port's matches none).

**On the prototype.** X2-13, G1, G2, G3 and the pinned X2 give `seam` in both ports and one gate verdict. The committed
inputs move 75 count claims to `seam`: 52 of them were already withheld (`count`), 23 are new. All are on twin diffs
(the only diffs where the count rule reads past its no-twin return), mostly the seeded PATH-2a fuzz. On the 34,000 fuzz
inputs there are 0 claim splits under every key of the committed `cross_port`.

**Pins.** X2-13, G1, G2 and G3 join `XPORT_CASES` and `path2a_pairs.json` with one gate verdict each; X2's pin moves
from `count` / `extract` to `seam` / `seam`.

## 2. The majors

### A-2 (bar A): the case doubt and the port's-reading check read every registration per claim

**The finding.** `_p2a_case_doubt` looped per claim over the registrations of the claim's wild base-name group whose
fold form is outside ASCII, and `_p2a_port_may_verify` over every registration, with nothing memoised: 6.4 s for the
overlay on 4,369 identical claims over 3,000 directories outside ASCII (bound 0.5 s); 20 s for whole calls at N = K =
6,000.

**Fixed, with no decision moved.**
- Both answers are memoised: the case doubt per (space, fold form, status), since the wild form is a function of the
  fold form; the port check per (claimed path, status).
- `_p2a_port_may_verify` reads the claimed path once through an Aho-Corasick automaton over the ASCII base names
  (per wanted status), built once per diff. Time is linear in the base names and in each claimed path.
- `_p2a_case_doubt` no longer scans the group. In the group G of the claim's wild base name, a fold match implies a
  wild match at the same tier. So a registration's wild tier is never above its fold tier, and U1 holds exactly when
  one of three facts holds:
  1. some fold base name in G is not the claim's (one set per group);
  2. a registration with the claim's wild form has another fold form (the existing per-wild-form fold sets);
  3. more registrations end in '/' + the claim's wild form than in '/' + its fold form. These are two reads through
     the claims' tree of A-1, over the registrations' wild and fold forms.

  A claim in ASCII has the same two forms, and none of the three can hold for it. U2 is one flag per (group, status).
  A claim whose base name is empty keeps the scan, memoised.

**On the prototype.** On the committed inputs and the 34,000 fuzz inputs no path claim's decision moves against
`5ebe0b6b`, in either port. The overlay alone:
- 4,369 identical claims over 3,000 directories: 6.06 s → 0.09 s;
- 3,000 distinct claims whose base name is outside ASCII: 19.3 s → 0.16 s;
- 200 claims over 2,000 `pkgs/pN é/__init__.py`: 0.22 s → 0.06 s;
- 4,000 distinct ASCII claims over 3,000 `dN/x.py`: 0.09 s → 0.12 s.

**Pins.** These five shapes join `_timing_cases` in both ports, under the existing bounds.

### B-1 (bar B): definitions CPython reads across a continuation line or after a lone CR

**The finding.** The #101 rules read a removed definition only where `def`/`class` and the name sit on one line, as
each line view cuts it. A definition read through a backslash continuation (`def \` then `test_a():`), or after a lone
CR inside one git line, was missed at both doors and in the port. The reviews' t-cont, t-cont-contra, s-cont,
s-cont-class, s-cr and t-cr keep `main`'s false verdicts, one of them a false CONTRADICTED that fails the gate. The
committed V101 reads line by line too, so the truth test could not see them.

**Fixed.** A third removed reading, `_p2a_joined`, beside the two line views, in both ports:
- each piece after a CPython line break inside a git line (the diff split at `\n`) that starts with `-`. No view reads
  such a piece as removed, and CPython's tokenizer ends a line at a lone CR;
- each run of the base side's pieces joined where a piece ends in a backslash, the backslash read as a space, when a
  piece of the run is removed. The base side is the pieces of git lines starting with `-` or ` `, in order; `+` and
  `\` lines are passed over, and any other line ends the run.

The tests pairing, the anchored symbol sites and `_p2a_defines` read these lines as well as the views'. It is a
superset of what the views read, and linear in the diff.

**On the prototype.** The six reproductions are withheld (`tests`, `symbol`) at the raw door. The two controls stay
withheld. No committed input moves through this reading alone.

**Pins.** The six join `path2a_pairs.json`. A truth test judges them, with B-2's four, at the raw door, at the git door
(through their recorded `--name-status`) and in the port. There, attribution uses an **ast-paired V101**: `main`'s own
count, less the head's test functions that the base of the same changed file also defines, both read by CPython's
parser under NFKC; and for a symbol, `main`'s hit only where the base does not define the name. It is `main` without
#101, reading definitions as CPython does. The README's claim about the ast-paired #101 variant is restated with the
worlds it was measured on.

### B-2 (bar B): names CPython reads through NFKC

**The finding.** CPython normalises identifiers with NFKC, so `def ｆｏｏ` and `def ﬁnd` define `foo` and `find`, and
`def test_ａ` defines `test_a`. The overlay paired names by their ASCII run and kept `main`'s false VERIFIED (s-nfkc-fullwidth,
s-nfkc-ligature, t-nfkc, t-nfkc-mixed).

**Fixed, structurally, with no Unicode table.** A removed `def` or `class` site whose coarse run after the word, or the
code point that ends its ASCII name run, is from 0x80 up has a name read through NFKC, so it may be any name. In the
tests pairing, such a removed site pairs with every counted site. A counted site whose name run goes on past its ASCII
part pairs with every removed test. In the symbol rule, such a removed site defines every claimed name. The same in the
port, by UTF-16 unit (a unit from 0x80 up is a code point from 0x80 up).

**The cost.** It falls only on diffs whose removed definitions hold a code point outside ASCII in or just before the
name. The reviewer found none among the 112,086 Python patches of the EXTERNAL-1 shelf. On the committed inputs it
moves 25 claims (#161's y5, a NFKC test rename whose `main` CONTRADICTED is false, and 24 on the seeded PATH-2a fuzz,
whose removed lines write U+00A0, U+3000 or U+FEFF between `def` and the name).

### C-2 (bar C): the static checks still accepted runtime Unicode queries

**The finding.** Python's `_p2a_regex_problems` read only the flag letter just after `(?`, so `(?si)`, `(?mi:...)` and
`(?xi)` passed. The port's scan did not look at regex modifiers (`(?i:...)` is live in V8 13.6). It allowed `parseInt`
and `Number` on any string; `parseInt("　3", 10)` is 3, which reads the engine's white-space table.

**Fixed.**
- Python: a flag group, of letters from `aiLmsux-` closed by ':' or ')', is refused if it holds i, u, L or a in any
  position.
- The port: '(?' followed by a letter or '-' is refused in a RegExp. `Number`, `parseInt` and `parseFloat` are banned
  tokens. The port reads digits through `_p2aInt`, a fixed table of the ten, mirroring `_p2a_int`, and throws on
  anything else, so the error fallback withholds.

The reviewer's plants are committed as refused cases: three Python flag groups (`(?si)`, `(?mi:...)`, `(?xi)`), and in
the port `(?i:...)`, `parseInt(...)` and `Number(...)`. The reviewer's widened count head (`([^,]+)`) now reads no
table: its non-ASCII digit throws in `_p2aInt`, and a committed case shows the error fallback withholding there.

### I-1 (integration): the PR body would turn the PR's own `diffgate` check red

**The finding.** The scratch PR body quoted template-shaped sentences ("3 files changed.", "Added 2 tests.", "Adds
function backoff with jitter."). The PR's `diffgate` check runs PyPI 7.48.0, `main`'s reader `9b620e00`, which reads
them in any wrapping: quotes, backticks, fences, indented blocks, block quotes. It also named an old head.

**Fixed, in scratch only** (`scratchpad/ghmerge/p2a/pr_body.md`; nothing is posted). The body is rewritten for this
pass's head. Reproductions are described, not quoted, and numbers are written in words. It is then run through
`main`'s reader against `git diff 1cde8b82...HEAD` until the reader gives PASS.

### I-2 (integration): the DECLARE-1 skips were pinned by no input

**The finding.** Removing the `declared` skip from the extract guard (path, name, count) or the zone guard (prefix),
in either port, left every behavioural test green. A port-only removal splits the gate verdicts (FAIL / PASS on a
declared count beside `é7`).

**Fixed.** Four pinned pairs and four `XPORT_CASES`: a declared count (`files_changed: 7` over a twin diff, beside
"See ticket é7."), path (`file_touched: src/app.py` beside `src/app.pyé`), name (`adds_symbol: foo` beside `fooé`) and
prefix (`only_touches: src` beside "We only édited src."). Each keeps `main`'s verdict in both ports, and its undeclared
twin sentence is withheld (`extract`). On the prototype all eight read so in both ports.

## 3. The minors

| id | finding | disposition |
|---|---|---|
| A-3 | `_p2a_dotted` re-sliced the string once per leading `./`, O(L²). | fixed: one index scan and one slice, in both ports; a 40,000-`./` case joins the timing cases. |
| A-4 | The disclosed large-diff factors are exceeded on line-break-heavy (B2, 3.3 MB) and one-long-line (B4, 5.8 MB) diffs. | disclosed: the factors are measured at the head on those shapes too, and the README says which shapes the timing tests bound and which they do not. |
| B-3 | On CONTRADICTED verdicts, `shape` withheld almost only right verdicts (world T: 33 of 35 firings right CONTRADICTEDs; the 2 false it caught were VERIFIEDs). | disclosed with the reviewer's figures and the same count on this branch's own truth world; operator option O-12: `shape` on VERIFIED only, which would keep the GNU-deletion and mnemonic reproductions' false CONTRADICTEDs. Not taken. |
| B-4 | On real agent PRs, `extract` caught no false verdict and cost 4 right and 3 undecided verdicts, all from a path joined without a space to Chinese or Korean prose. | disclosed in the README and CHANGELOG cost lists, with the shelf figures (7 firings of 13,089 decided). |
| C-3 | `_one_match` pairs two different claims (L1: `.c.py` and `..c.py`, each in the other's text). | disclosed: the one-match key is a heuristic, dropped from the sentence that says 0 splits under every key; L1 is pinned as the one known false pairing on the committed inputs (the test expects exactly that split on it). Tightening by offsets does not separate L1, since `.c.py` also occurs inside `..c.py`. |
| C-4 | Count and tests decisions read `main`'s count from its reason, which rests on each engine's lowercase keeping the fold lemma; the bookmarklet runs on engines never checked. | disclosed: the README's by-construction sentence names it. A load-time check in the block would need `toLowerCase`, which the token scan bans; not taken. |
| C-5 = I-4 | CI never checks the shipped `bookmarklet.min.js` and `bookmarklet.href.txt`. | fixed: a test with no terser that the minified file hashes to the README's line and that the href is `javascript:` + it; and a node test (`check_path2a.js --bookmarklet`) that loads the minified file in a stub page and compares its gate with `diffgate.js` on every pinned pair, both strict modes, asserting that overlay reasons appear. Under CI it fails without node. |
| I-3 | The Action test does not pin the kind condition. | fixed: a synthetic row of kind `declaration_problem` whose reason starts with the overlay's form is cut at 100 characters. |
| I-5 | The README's cost paragraph showed a noise sample as a cost, rounded a factor down, and left out import time. | fixed: per-corpus cost with its spread over repeated runs, the factor as measured, and the import-time cost of `styxx.diffgate` (self time, `-X importtime`) in the README and CHANGELOG. |
| I-6 | On Windows, the git-door tests fail under a long pytest temp path (MAX_PATH). | fixed: `-c core.longpaths=true` in `_repo` and `_tree_repo`. |
| I-7 | Process: pass 4's builder used `sed -i` once on a scratch copy and let early pytest runs write under `AppData/Local/Temp/pytest-of-heyzo`; the disk assumption was wrong. | recorded here. This pass sets `TEMP`, `TMP` and `--basetemp` under the scratch directory from its earliest run, and uses no `sed -i`. C: had 2.1 GB free at this pass's start; the p2a scratch is 1.2 GB, 925 MB of it `review1`. Whether `review1` may be pruned is the operator's call; this pass does not touch it. |

## 4. The rules after this pass

Everything in the earlier notes stands, with these changes.

- **Path claims**: V97's suffix tier through the claims' tree (A-1); the case doubt by three per-group facts (A-2); the
  port's reading through the automaton (A-2). No decision moves.
- **`files_changed_count`**: `divergent`; no dot twin keeps; `seam` (C-1, new); `extract`; `unparsed`; `unreproduced`;
  the interval.
- **`tests_added`, `symbol_added`**: the removed side is read by both line views and `_p2a_joined` (B-1); a name read
  through NFKC pairs with every name (B-2).
- **Phrases**: `seam` is added.
- **Static checks**: flag groups refused whatever the flag's position (Python); modifiers, `Number`, `parseInt`,
  `parseFloat` refused (port).

The reconstruction property, REACH, the reason form, the error fallback and the abstain-only relation are unchanged.

## 5. Predictions, falsifiable, for the head that carries this pass

- **Relation.** No committed input's record breaks the relation (A) in either port, at either door, in either strict
  mode; the error phrase never appears, also under the reviewer's 1,024 MB memory cap on the deep-path input.
- **Cross-port.** 0 splits under every by-construction key on the committed inputs. Under the measured keys: 0, except
  L1's one known false pairing. The five inputs where `main`'s gates agree and the overlay's differ are the same five.
- **Committed-input abstention pins** (Windows flavour): `files_changed_count:seam` 75 (`count` loses 52); the #101 phrases
  gain 25; path phrases do not move.
- **Truth.** 0 misses at each door and in the port, including the ten B reproductions; the counts at the head go into
  the README and the tests.
- **Recall.** `main`'s committed corpora: 80 of 2,231, unless `corpus_fuzz` holds a seam over a twin (measured at the
  head).
- **Cost.** Every committed timing case under 0.5 s (Python) and 0.3 s (port), and under 64 MB of peak memory in Python.

## 6. What this pass does not change

- **No repair.** PATH-2a still never gives VERIFIED where `main` was wrong.
- **G-P1** is still not met, and that is still the operator's decision.
- **`main`'s reader** is untouched in both ports; the reconstruction test holds.
- **Nothing committed is edited:** no receipt, certificate, sworn file, PREREG, RESULT, ANALYSIS, AMENDMENT, ERRATUM,
  earlier NOTE or the charon log.
- **Operator options.** O-1 to O-11 stand. O-12 (`shape` on VERIFIED only, B-3) is added, measured, and not taken.
