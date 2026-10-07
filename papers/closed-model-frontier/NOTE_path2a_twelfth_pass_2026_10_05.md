# NOTE — PATH-2a, twelfth pass: the single-prefix gap as a family of renderings, the port's APPLY before DECIDE, and each finding of the reviews of `ee82d2f3`

## 0. Status

2026-10-05. Branch `fix/diffgate-abstain-where-wrong`, head `ee82d2f3` (pass eleven). `main`'s two reader files are
unchanged (sha256, LF: `9b620e00…` and `06688702…`). The earlier notes are listed in the README's *PATH-2a* section;
none is edited.

What this pass had to read:

- four finished reviews of `ee82d2f3`: by construction and cost, "ship" (three minors); coverage and recall, "fix
  before shipping" (one major, two minors, two records); cross-port, "ship" (two minors, four records); integration,
  CI and docs, "ship" (five minors);
- the lead's direction of 2026-10-05 for this pass (§1), which outranks the usual order of work where the two differ.
  The construction of passes ten and eleven stands and is not reopened.

**This note is written before this pass's code and committed alone, ahead of it.** A figure in this note is a
reviewer's, measured at `ee82d2f3`, and is named as such; §7 says what is measured again at the head that carries the
code. If the committed code departs from this note, the departure goes into a note of its own. Finding identifiers
are this note's: A-n, B-n, C-n and I-n number each lens's findings in the order its review lists them.

Dispositions: **fixed** (a change in this pass; a test that fails on `ee82d2f3` where the finding is a defect the
code can show); **disclosed** (kept, and stated in the README, the CHANGELOG or here); **operator** (left to the
operator); **recorded**.

## 1. The lead's direction for this pass

1. **Major B, the single-prefix scope gap, is a family, not one rendering. No rule change.** O-10 is restated in the
   README, the CHANGELOG, the comment in the scope rule of both ports and here: any rendering that drops a leading
   directory opens the gap, for any top-level directory (§2). The sentence that bounds the gap to the keys a no-prefix
   strip can hide goes, and the README says that neither rule offered so far closes `--relative`. The reviewer's four
   `--relative` cases are pinned as known kept attributable CONTRADICTEDs at the raw door, in the port and at the git
   door, asserted exactly. The operator gets the general rule with the reviewer's measured cost. The README sentence
   on where the changed files lie is corrected.
2. **Minors taken**: (a) run counts marked *pinned* that the tests bound from below; (b) the CHANGELOG's figures
   carried from the tenth pass; (c) a named test that the copy's two classes define no `__init__`; (d) in the port,
   the keys DECIDE may name from a literal array, and APPLY's per-claim arrays filled without an inherited `push`,
   with the persistent-patch wording corrected; (e) the Python APPLY's second reading of `strict`, said, no code
   change; (f) `.toSorted(` beside `.sort(` in the port's lint; (g) the reviewers' cost rows and O-16 cells; (h) the
   emulated Unicode 13 CPython in the README's runtime sentence.
3. **Diff size.** The pull request's diff is 16,285 lines at `ee82d2f3` against GitHub's 20,000; this pass keeps its
   growth small and reports the count at its head.
4. **House words** are searched for in every changed file before each commit, not after it.
5. The lead pushes; this pass does not. The worktree is left clean with everything committed.

## 2. The single-prefix gap: a family of renderings (B-1, B-2; operator option O-10)

**The mechanism, stated by structure.** A rendering drops a leading directory `D` of every path it prints, and the
claim names `D` ("Only touches docs."). Under `D` lie `.D/` and at least one other changed path. `main`'s path key
strips the leading dot (#121), so `.D/conf.py` keys as `D/conf.py`, `D` becomes a path segment for `main`, and the
other changed path (`guide.md`, printed without its `D/`) lies outside it: CONTRADICTED, "paths outside 'D'". In the
repository every changed path lies under `D/`, so the verdict is false; V121 keeps the dot and reads `D` as not a path
(#110), so the false verdict is attributable under the committed but-for definition. The overlay keeps it: with one
prefix claimed, the scope rule keeps `main`'s verdict where V121 reads the prefix as not a path (`shape` fires only with
a second prefix). `D` may be any directory, not only `a` or `b`.

**The renderings that drop a leading directory**, each enough on its own:
- `git diff --no-prefix` over a repository whose top directory is `a/` or `b/`: `main` strips the real `a/` or `b/` as
  git's prefix (the tenth coverage review's four cases, pinned since the eleventh pass);
- `git diff --relative=D` and `git format-patch --relative=D` at the raw door: git prints the paths under `D` without
  `D/`, and keeps its `a/` `b/` split;
- `diff.relative=true` at the git door, with a subdirectory `D` as the repository argument: git prints both the
  `--name-status` listing and the diff relative to `D`.

The eleventh pass's README, CHANGELOG, comment and note named only the no-prefix rendering, and offered two rules: the
reviewer's narrow rule (`shape` also for a single prefix when `main`'s key for it is `a` or `b`, "the only keys a
no-prefix strip can hide") and a trigger on `diff --git` headers with no `a/` `b/` split. Neither closes `--relative`:
its prefix is any directory, and its headers keep the split. The bound is dropped from the README and the CHANGELOG,
and the comments in `_p2a_only` and `_p2aOnly` are restated.

**The reviewer's four `--relative` cases** (built with git: `r11_relative.py`, in the review's scratch):
`docs/.docs/conf.py` and `docs/guide.md` modified, "Only touches docs."; `docs/.docs/new.py` added and
`docs/guide.md` modified, "This change only modifies files in docs."; `config/.config/old.toml` deleted and
`config/app.toml` modified, "Only touches config, nothing else."; `vscode/.vscode/settings.json` and
`vscode/extension.ts` modified, "Only changes vscode.". At the raw door, in Python and in the port, `main` gives
CONTRADICTED "paths outside 'docs': ['guide.md']" (and so on), truth is T, V121 and the variant without all three read
UNCHECKABLE, and the head keeps CONTRADICTED with gate FAIL in both strict modes: 8 kept (4 per port). At the git door,
with the subdirectory as the repository and `diff.relative=true`, the same four are kept: 4.

**Pinned, no rule changed.** The four cases enter the truth module with the bytes git prints (the raw door's
`--relative` diff, which git also prints at the git door under `diff.relative=true`, and the relative `--name-status`
listing), read at the raw door, in the port and at the git door through the test module's stand-in for `_git`.
`KEPT_SINGLE_PREFIX` becomes a count per door, asserted exactly: raw door 8 (the four no-prefix cases and the four
`--relative` ones), the port 8, the git door 4. These pins hold on `ee82d2f3`'s code as on this head's, since no rule
changes; what they catch is a change that closes or widens the gap (§7 plants the general rule below and expects them
to move).

**For the operator, under O-10** (the reviewer's figures, at `ee82d2f3`): the general rule, "withhold every
single-prefix CONTRADICTED that V121 reads as not a path", closes every rendering of the family. What it withholds
beyond this head, each a right verdict by truth: on the committed truth world 98 per port and no false one; on the
reviewer's decorated worlds 155 in Python and 145 in the port (seed 11101) and 144 and 147 (seed 11102); on the
reviewer's world rendered by real git (seed 11103) 58 in Python, 58 in the port and 19 at the git door. Not taken.

**Where the changed files lie** (B-2). The O-10 paragraph says a kept CONTRADICTED "is false when every changed file
lies under the dotted directory while `main` lists a path outside it". In every pinned and probed case they lie under
the prefix's directory (`b/`, `docs/`), not under the dotted one (`.b/`, `.docs/`); files that all lie under `.docs/`
make `main`'s reading VERIFIED, which `only` already withholds. The sentence is corrected to read: when every changed
file lies under the prefix's directory, while `main`, reading a path whose leading directory the rendering dropped,
lists a path outside it.

## 3. The port's APPLY before DECIDE, and the persistent-patch route (A-2; direction 2(d))

**The wording.** The README, the CHANGELOG and the eleventh note (§2 there) say that a patch of what objects or arrays
inherit, left in place into a later call, "reaches the next call's `main` reader before it reaches APPLY". The
construction reviewer showed otherwise: such a patch can leave `main`'s reader untouched and reach APPLY's own work
before DECIDE runs. A `push` left patched to drop a lone `null` (`persist11.js`) left `main`'s reader unchanged in 320
of 320 records and put 317 of 320 branch records outside the relation (APPLY filled its per-claim arrays with `push`,
so they came out short); an enumerable key put on what every object inherits before the module loads (`forin11.js`)
entered the list of keys DECIDE may name, which APPLY built with `for … in` over the phrase table, so a decision
naming it was taken with the inherited phrase. Both routes are outside the restated boundary (a built-in patched
through what objects or arrays inherit). The three places are reworded: a patch left in place can reach APPLY's own
work before DECIDE (`push`, a `Set`'s `has`, the list of keys built at load) as well as `main`'s reader, and is not
covered. The eleventh note is not edited; this paragraph is its correction.

**The hardening, taken because it costs little.** `_P2A_DECIDE_KEYS` becomes a literal array of the twenty-one keys
DECIDE may name, and the committed table test asserts that it equals the phrase table's own keys but `error` and
`malformed`, in order. APPLY fills `reach`, `pickPhrase` and `pickTag` by index store, not by `push`. Two cases join
the port's one-realm hostile list, each failing on `ee82d2f3`: a `push` left patched to drop a lone `null` before
APPLY runs, with the block's own DECIDE (relation); and a module loaded after a page put an enumerable key on what
every object inherits, with a DECIDE that names that key (`main`'s record, untouched). What stays open, and is said:
an index setter put on what every array inherits (a store to an index an array does not yet hold looks one up), the
`Set`'s `has`, and anything that reaches `main`'s reader. The pinned text of the port's APPLY moves, and its hash.

## 4. The other minors taken

- **(a) Run counts** (I-1). The README marks 958, 962 and 1,150 runs as *pinned*; the tests assert more than 900 (and
  more than 1,100). They are written as measured, with the bound the test asserts beside them.
- **(b) The CHANGELOG's carried figures** (I-2). "APPLY's copy 5.5 to 5.8 ms for 10,000 claims", "+15% a call" and
  the committed timing cases' figures are the tenth pass's (the copy figure was measured on `DiffClaim` copies, which
  the eleventh pass replaced); they are marked as such. The bookmarklet's equality with the port is measured again on
  this pass's build.
- **(c) The copy's classes** (I-4). A named test asserts that `_P2aClaim` and `_P2aSeen` define no `__init__` and no
  method, so that the eleventh pass's fix is held by behaviour of its own as well as by the text pin.
- **(e) `strict`** (A-1). The Python APPLY recomputes the gate verdict with `main`'s formula,
  `contradicted or (strict and uncheckable)`; where it withholds a CONTRADICTED that was `main`'s only one, `main`
  short-circuited on that CONTRADICTED and never read `strict`, and APPLY reads it as a truth value. A `strict` whose
  truth value raises (outside the `bool` the signature names) then raises at the door where `main` returned FAIL: the
  reviewer counted 59 of #161's 497 reproductions. The CLI, the Action, capsule and charon pass bools. Said in the
  README and in APPLY's docstring; no code change. The port's door passes `!!strict`, which cannot throw.
- **(f) `.toSorted(`** (C-4). It sorts by UTF-16 code unit, as `.sort(` does, not by CPython's code point; it asks no
  Unicode question, so it is a transliteration hazard, not a lint gap. It joins the banned list with a refused plant,
  and the README names both.
- **(g) Reviewers' figures** (A-3, B-3). The cost table gains the dot-twin-heavy row and the long-line rows, and O-16
  the two cells, each as the reviewer's (§5).
- **(h) Runtimes under emulation** (C-2). The README's sentence gains the reviewer's emulated Unicode 13 CPython (CI's
  3.9 and 3.10): within the committed tolerance of the nearest pin, 0 C(i) splits and 0 C(ii) breaks.
- **Also taken** (C-3, optional): the README sentence on the port's static white-space table says the Python's
  `_P2A_PY_SPACE` and `_P2A_PY_BREAKS` bind it to its interpreter the same way, pinned only where the tests run.

## 5. The findings of the reviews of `ee82d2f3`

### By construction and cost ("ship")

| id | severity | finding | disposition |
|---|---|---|---|
| A-1 | minor | The Python APPLY reads `strict` as a truth value where `main` short-circuited; a raising `__bool__` raises (59 of 497 reproductions). | **disclosed** (§4 e): README and docstring; outside the `bool` the signature names. |
| A-2 | minor | The persistent-patch route is described wrongly: it can reach APPLY's own work before DECIDE. | **fixed**: wording in three places (§3), and the literal key list and index stores, with two cases that fail on `ee82d2f3`. |
| A-3 | minor | Cost linear everywhere; twins at 4,000 pairs ×5.4 (Python, 38 → 205 ms whole call) and ×4.8 (Node, 14.7 → 71 ms); one 1 MB added line of `def test_x(): pass; ` ×12 (8.9 → 106 ms) and ×8.8 (1.05 → 9.2 ms); NEL and U+2028 joined lines at 16k ×7.7 in Node (6.75 → 52 ms). | **disclosed**: rows in the README's cost section, as the reviewer's. |

### Coverage and recall ("fix before shipping")

| id | severity | finding | disposition |
|---|---|---|---|
| B-1 | major | The single-prefix gap opens under `--relative` and `diff.relative=true`, for any top directory; 8 attributable CONTRADICTEDs kept at the raw door, 4 at the git door; nothing pins or states them. | **operator** (O-10), by the lead's direction: no rule change; restated (§2); pinned at three doors; the general rule with the reviewer's cost. |
| B-2 | minor | "Under the dotted directory" should read "under the prefix's directory". | **fixed** (§2). |
| B-3 | minor | Two more cells withhold more right than false verdicts on the reviewer's worlds: `extract` on VERIFIED (Python, seed 11101: 192 false against 204 right; the port, seed 11102: 97 against 162, plus 281 undecided; the port, seed 11103: 85 against 109, plus 247) and `case_count` on CONTRADICTED (seed 11102: 27 against 32 in Python, 24 against 25 in the port). | **disclosed**: one sentence under O-16, as the reviewer's figures. |
| B-4 | record | Bar B holds but for B-1 and the pinned O-10 cases; records equal `16daa725`'s; of the 11 ast-only kept per port, 9 are the disclosed defined-again limit and 2 are a `def test_a` inside a docstring, not the #101 mechanism. | **recorded**. |
| B-5 | record | Recall reproduces (80 of 2,231; 280 of 2,761; 90 of 155 and 89 of 154); every abstention on `main`'s corpora withholds a VERIFIED that is false by the diff's own listing. | **recorded**. |

### Cross-port ("ship")

| id | severity | finding | disposition |
|---|---|---|---|
| C-1 | record | C(i) and C(ii) hold on 241,000 new fuzz inputs, the 160,000 adversarial inputs, 19,000 under the patched engine and 70,000 under runtime skew. | **recorded**. |
| C-2 | record | The C(iii) rows reproduce; an emulated Unicode 13 CPython stays within the committed tolerance. | **recorded**; the README sentence extended (§4 h). |
| C-3 | minor | On an engine whose `\s` and `trim()` read one more character, C(i) and C(ii) break in the port, each break carrying `unreproduced`; the Python has the same exposure in a CPython whose `isspace` or `splitlines` differ. | **disclosed**: the README sentence names the Python too (§4). |
| C-4 | minor | The token scan refuses `.sort(` and passes `.toSorted(`. | **fixed** (§4 f): banned, with a refused plant. |
| C-5 | record | APPLY attacked at run time on inputs other than the reproductions, both ports: 0 outside the relation. | **recorded**. |
| C-6 | record | The bookmarklet builds with `--check` (`3659b422…`) and equals the port. | **recorded**; rebuilt in this pass (§3). |

### Integration, CI and docs ("ship")

| id | severity | finding | disposition |
|---|---|---|---|
| I-1 | minor | Run counts marked *pinned*; the tests assert lower bounds. | **fixed** (§4 a): written as measured. |
| I-2 | minor | CHANGELOG figures carried from the tenth pass without saying so; the bookmarklet's 20,000 runs were on pass ten's build. | **fixed** (§4 b): marked, and the bookmarklet measured again. |
| I-3 | minor | The PR diff's headroom under GitHub's 20,000 lines is shrinking (16,285 at `ee82d2f3`); past it this repository's body-against-diff job prints DID NOT RUN and exits 0. | **operator**: the count at this head is reported in the CHANGELOG; whether to merge, trim, or read DID NOT RUN on #187 as red is the lead's and the operator's. |
| I-4 | minor | The C-1 fix of the eleventh pass is held only by the text pin. | **fixed** (§4 c). |
| I-5 | minor | The committed eleventh note uses a word the house rules forbid three times. | **recorded**; committed notes are not edited (the eleventh corrections note records it). |

## 6. What this pass does not change, and what the operator decides

- **No decision changes.** The rules, REACH, the phrases, the reason form and the two hooks are as they were; the
  port's APPLY takes the same decisions from the block's own DECIDE. The records at this head are to equal
  `ee82d2f3`'s on every input measured (§7).
- **No repair.** PATH-2a never gives VERIFIED where `main` was wrong. **G-P1** is not met; that is the operator's
  decision.
- **`main`'s reader** is untouched in both ports.
- **Nothing committed is edited:** no receipt, certificate, sworn file, PREREG, RESULT, ANALYSIS, AMENDMENT, ERRATUM,
  earlier NOTE or the charon log.
- **Operator options** O-1 to O-16 stand as disclosed, O-13 included, and the behaviour under `--strict`. For the
  operator at merge: O-10 with the family of renderings, its pins and the general rule's cost (§2); O-16 with the two
  further cells (B-3); the PR's diff size (I-3); the second reading of `strict` (A-1).

## 7. Measured again at the head

Written into the README and the CHANGELOG with what each figure measures: that the records equal `ee82d2f3`'s on the
committed inputs in both ports and both strict modes, and on seeded adversarial sets; the hostile DECIDE runs in both
ports, the two new port cases, and that both fail against `ee82d2f3`'s port; the truth module with the single-prefix
pins at three doors, and those pins moving under the general rule planted; recall with `path2a_recall.py`; the cost of
the port's APPLY against `ee82d2f3`'s; the bookmarklet's size and its equality with the port; the size of the PR's
diff. A mutation check of the new tests: an `__init__` put back on `_P2aSeen`, the key list built by `for … in` again,
the per-claim arrays filled by `push` again, `.toSorted(` let through, the general rule planted. A figure quoted from a
review and not measured again is named as the reviewer's.
