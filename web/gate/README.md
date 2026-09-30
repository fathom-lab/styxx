# web/gate — the diff gate where Python cannot run

The instrument is `styxx/diffgate.py`. Two browser surfaces cannot import it: the paste-in
preview page and the bookmarklet. They run `diffgate.js`, a JavaScript transliteration of one
specific file — `styxx/diffgate.py` on this branch, sha256
`5007bcae5219157d96f63386b0b4045ff39c405e805ca9a3bb907e5e8b313b4f` (LF line endings; a wheel
built on Windows carries CRLF and hashes differently, so `py_side.py` normalises before it
compares) — and this directory is the receipt for that port: the differential test that holds
it to the Python's output, and the build that turns it into the bookmarklet people drag into
their bookmarks bar. That file is `main`'s reader (BC-2 + COMPAT-1 + BIN-2 + COMPAT-2, pull requests
#113, #115, #120 and #124, plus the `fetch_pr` door), unchanged — the file **7.48.0** ships, sha256
`9b620e00a19464589308a987819894ae7cc3c111c66a5f8a457a84b8a6c604eb` — plus the PATH-2a block (an
overlay that only abstains, not yet released; see *PATH-2a* below), and a committed test cuts the
block out and gets `main`'s file back byte for byte, in both languages.

The port was first cut from the 7.47.0 wheel (`fb2d9b3e…`) and re-cut on 2026-09-16 for issue
#110: the 7.47.0 templates count Python `def` lines and accuse a TypeScript commit that says
"added 2 tests" of lying, and the bookmarklet is a public door, so it moved to the repaired file
ahead of the release rather than after it. What changed between the two files is listed in the
header of `diffgate.js` and measured below under *Drift*.

(The `/gate` page on the site is different: it runs the real released Python under CPython 3.13
via Pyodide, with the two module files hashed in the browser against the wheel. No port there.)

## What is here

`diffgate.js` — `gateDiffText(summary, diff, {strict})`, `parseUnifiedDiff(text)` and
`parseUnifiedDiffSides(text)`, the same closed template set (eleven templates, `compat_claim`
included), the same verdict strings, the same `why` text, the same `detail` — including the
`removed` list the compatibility reading attaches, and Python's `repr()` quoting where the
Python builds a reason with `!r`. Two deliberate gaps, both
stated in the file: the structural "unparsed claims" observer (`styxx.claimdetect`) is not
ported, and `--run` / `--evidence` do not exist, so "tests pass" is always UNCHECKABLE, exactly
as the CLI without `--run`.

`bookmarklet_ui.js` — the panel: on a `github.com/OWNER/REPO/pull/N` page it reads the
description and the diff from `api.github.com` (two unauthenticated requests, nothing else,
nothing stored, nothing sent anywhere) and pins `[ok ]` / `[LIE]` / `[ ? ]` lines, the verdict
and the never-read count to the page.

`build_bookmarklet.py` — assembles the two into `bookmarklet_src.js`, minifies with
`terser -c -m --format ascii_only`, writes `bookmarklet.min.js` and `bookmarklet.href.txt`.
`--check` rebuilds and compares against the committed files. The shipped bookmarklet is

    bookmarklet.min.js    sha256 185352f892b08bc2382802a35f7ba97ddf5e5f0759bfc5fd629c3835cbdd810b   44,199 chars
    bookmarklet.href.txt  sha256 d25bd11c570a7605acfa542d36fbed7a0633015e4c94e6809a2cd07076ffed8e   44,210 chars

(Earlier builds: `b04d14dc…`, 11,437 chars, from the 7.47.0 file — accuses outside Python;
`9ea8f572…`, 17,686 chars, the BC-2 + COMPAT-1 re-cut — cannot see a binary file;
`1be19a65…`, 24,335 chars, `main` at `1cde8b82` before PATH-2a, whose README still named an older
`54dca73a…`, 21,632 chars; `c457cca3…`, 34,285 chars, PATH-2a's pass 1; `bfe8c047…`, 36,933 chars, its pass 2;
`c726c904…`, 38,763 chars, its pass 3.
A bookmark that hashes to any of these is an old port; drag the new one. The panel text still names the 7.48.0 port; see *PATH-2a* below.)

Whatever a browser holds under that bookmark either hashes to the first line (drop the
`javascript:` prefix) or is not this build. terser 5.46.0 produced these bytes, and rebuilds
`main`'s 24,335-char build from `main`'s sources byte for byte.

`differential/` — the test. Read on.

## The differential test

A port is held to the original by output, not by trust. `differential/` builds a corpus of
(summary, diff) pairs, runs the released Python and the JavaScript over every pair, and compares
the records field by field: verdict, measured, why_unmeasured, sentences_total,
uncovered_sentences, the never-read list in order, and every claim as
(kind, verdict, why, text, detail) in order. A port that gets the verdict right for the wrong
reason, or reads one sentence more or less, is a disagreement.

    cd web/gate/differential             # on a checkout carrying #113 and #115 (or 7.48.0)
    python build_corpus.py               # 176 real pairs, pinned to shas (below)
    python fuzz_corpus.py                # 3,000 synthetic pairs, seeded
    python py_side.py                    # refuses to run unless styxx/diffgate.py hashes to 5007bcae…
    node js_side.js
    python differential.py
    node check_pairs.js                  # the 153 pinned pairs against their expect blocks (+ path2a_moves.json)

The pin moved twice between the last two runs of this differential, and one of those moves is a
finding rather than a routine bump. COMPAT-2 (#124) changed the compat reading and the port had to
follow it — that is ordinary. But #124 merged with only its Python half, and separately the
`fetch_pr` door landed on `main` after the previous pin was written, so `py_side.py` had been
**refusing to run on `main`** ever since: the check that guards the two-implementation claim was
itself disabled, which is exactly how the port was able to fall a whole cycle behind without
anything failing. `fetch_pr` fetches a pull request over the network and is no part of the reading
the port transliterates; it moves this whole-file hash without changing a single verdict. Both
facts are recorded here rather than quietly corrected.

`path1_pairs.json` carries eight pairs for PATH-1: a basename claim satisfied by files in
subfolders, a nested single file, a basename claim that still accuses when something else changed,
a dotted identifier (`Assert.NotNull`), a CSS selector (`.k-step-link`), a slashed prefix that
still anchors, and two pinning the failure modes PATH-1 deliberately does **not** repair. Those
last two assert the instrument is still wrong; they exist so a later change cannot claim an
unrepaired mode without its own preregistration.

The corpus needed them. Before PATH-1 added these, `corpus_real.json` carried exactly **four**
`only_touches` claims in 3,212 pairs and PATH-1 changed none of them — so the differential's
"0 disagreements" was true and almost meaningless for that change. A check that does not exercise
what changed is not evidence, and the honest py/js comparison for PATH-1 was run separately over
the 604-row BENCH corpus (297 `only_touches` readings, 0 disagreements).

Result, 2026-09-18, this branch at the COMPAT-2 reading plus PATH-1:

    3220 pairs, 6933 claims (606 verified, 1616 contradicted, 4711 uncheckable) — 0 disagreement(s)

The 3,220 are the 3,176 below plus 44 pinned pairs committed as JSON (the `.gitignore` here
ignores generated JSON and names these five as exceptions): `bc1_pairs.json`, the four pairs BC-2
owes the differential (a TypeScript commit saying "Added 2 tests", "adds a method to reload",
"only modifies the footer", two prefixes), also checked on the Python side by
`tests/test_diffgate_bc1.py`; and `compat_pairs.json`, nineteen more — the compatibility
reading in every branch (Python, JS/TS, Go, Rust, Java and Kotlin, a moved definition, no
covered language, more than five names, all nine phrases, an empty diff), the V14 bare-name and
containment cases, "added 3 test cases", second prefixes that are and are not path-shaped, a
changed path with an apostrophe (Python's `repr()` switches to double quotes; so does the port),
and the demo diff with CRLF line endings; and `bin1_pairs.json`, six for the #118 repair — a
binary beside a text file, three binaries added / modified / deleted, a pure rename and a mode
change, a file count that is true only once the binaries are seen, an `only_touches` lie hidden
behind a binary, a quoted path — also checked on the Python side by `tests/test_diffgate_bin1.py`;
and `compat2_pairs.json`, seven for the COMPAT-2 reading — a surface drop beside a scaffolding drop
and a signature change, scaffolding-only removals, a move with the same parameters, a Go signature
re-flowed over two lines, an exported arrow function's parameters, Java's name-then-paren regex,
seven surface drops beside one under `internal/`. Their `expect` blocks are the Python's output,
written down so a reader can see the intended readings without running anything. The 3,199
pre-BIN-2 records were byte-identical before and after that repair (G-BIN-2); of the 3,205
pre-COMPAT-2 records, 3,196 are byte-identical after it and 9 differ only in `compat_claim`
reasons and details (17 claims), which is COMPAT-2's G-C2-4.

The first result, the 7.47.0 port against the 7.47.0 file, was 3176 pairs, 8904 claims
(1024 verified, 2172 contradicted, 5708 uncheckable), 0 disagreements; that port is in this
directory's history.

The corpus is pinned so it is the same bytes everywhere: the 160 commits reachable from
`fa8bcde725252dd6264d89045fa58fce7f3d9e51` (each message against its own diff), three
pull-request descriptions written that day (`bodies/pr94.md`, `pr95.md`, `pr98.md`, each against
its branch head at the time — shas in `build_corpus.py`), the README demo, twelve edge cases from
the test suite, and 3,000 fuzzed pairs from `random.seed(20260916)`. The fuzzer aims at the
template set's soft spots on purpose (quoted and bare paths, Node.js-style non-file nouns,
"the same way x.py was", bullets, counts of zero, empty diffs, text that is not a diff), because a
port that agrees on easy input proves little. The figure posted the same day, 3,177 pairs, had one
more pair: an external project's PR description, left out of the committed corpus because it is
not ours to republish. It carried zero diff-shaped claims, so the claim count is the same 8,904.

## Drift: the port is ahead of the release

Until 7.48.0 ships, `pip install styxx` gives the 7.47.0 file and the port does not match it.
`python py_side.py --installed` runs the installed package instead of the checkout and is
expected to disagree. Measured the same day, port against the 7.47.0 wheel:

    3199 pairs, 8932 claims (1031 verified, 2189 contradicted, 5712 uncheckable) — 5448 disagreement(s)

What moved, on the 3,176 corpus pairs (the pinned pairs were written for the new file and are
left out of these counts; the #118 repair changes none of the 3,176 records, and COMPAT-2 changes
only `compat_claim` reasons, a kind the wheel never reads, so the figures below are unchanged by
either): the wheel makes 573 accusations the port does not — `symbol_added`
219, `only_touches` 205, `tests_added` 149 — and the port makes none the wheel does not. 286
pairs flip from FAIL to PASS and none flip the other way. The port reads 2,044 fewer
`file_touched` claims (the V14 repairs declining bare and ambiguous path mentions) and one more
sentence kind, `compat_claim`, which the wheel never reads (one sentence in this corpus carries
it). Every one of the 573 now ends in a reason citing #110 — 368 "no Python file in the diff;
this template counts `def` lines", 205 "prefix '…' is not a path" — because every one was an
accusation the issue showed to be unsupported by construction. On the EXTERNAL-1 corpus the same repair withdrew 569 of
665 (`papers/closed-model-frontier/RESULT_bc2_by_construction_lands_2026_09_16.md`).

7.48.0 is `main`'s reader without the PATH-2a block, so `py_side.py --installed` against it does not
print 0 disagreements on this branch: the expected disagreements are exactly the pairs holding the
overlay's abstentions — 75 pairs holding 80 abstained claims on `main`'s committed corpora, which
`path2a_recall.py` counts claim by claim; `differential.py` counts pairs. Any other disagreement means
the release is not the file the port claims to be, and the header of `diffgate.js` says which one it is.

## PATH-2a: abstaining where #97, #121 or #101 can make a verdict wrong

`styxx/diffgate.py` and `diffgate.js` each carry one block between the markers
`=== PATH-2a abstain-only overlay: BEGIN ===` and `... END ===`
(`papers/closed-model-frontier/NOTE_path2a_abstain_overlay_2026_09_30.md`; for the second review pass
`NOTE_path2a_second_pass_2026_09_30.md` and `NOTE_path2a_second_pass_corrections_2026_09_30.md`; for the third
`NOTE_path2a_third_pass_2026_09_30.md`; for the fourth `NOTE_path2a_fourth_pass_2026_09_30.md`). `main`'s reader runs
unchanged — at the raw door, at the git door and in this port — and then the block reads each decided claim once more.
Where the #97 mechanism (the earliest entry in diff order matching by exact path, suffix or base name), the #121
mechanism (`_norm`'s `lstrip("./")`, which gives `.x` and `x` one key) or the #101 mechanism (a changed `def` counted
as added) can have made `main`'s verdict wrong, the claim becomes UNCHECKABLE with the reason

    {V} withheld by PATH-2a ({defect}): {phrase}. main's reading: {main's reason, verbatim}

and the gate verdict is recomputed with `main`'s own formula. Nothing else in the record moves: the
claim list, every other verdict and every other reason are `main`'s, byte for byte, and `--strict`
changes the gate verdict only. Cut the block out and revert the door hooks (two lines per file) and
you have `main`'s two files back, sha for sha; `tests/test_diffgate_path2a.py` does exactly that and
uses the result as its reference.

**The three defects are not repaired.** PATH-2a never gives VERIFIED where `main` was wrong; it only
stops `main`'s false verdicts on these shapes from standing, and says which verdict it withheld and
why. PREREG_path2's G-P1 (on #161's branch) expects VERIFIED on the reproductions, so PATH-2a does not
meet G-P1: whether it stands in for it is the operator's decision. `--strict` fails on every new
abstention, as on any UNCHECKABLE.

**Not released.** PyPI's 7.48.0 carries `main`'s reader without the block, so `python -m styxx.diffgate`
from `pip install styxx` disagrees with this checkout (and with the rebuilt bookmarklet) on every PATH-2a
abstention. The bookmarklet's panel text (`bookmarklet_ui.js`, not edited here) still calls it the 7.48.0
port and points to `pip install styxx` to reproduce; reproduce a PATH-2a reading from a checkout of this
branch instead. The GitHub Action's job summary shows a reason whole only when the overlay wrote it — an
UNCHECKABLE claim of a kind the overlay may move whose reason starts with the overlay's form — and cuts every
other reason at 100 characters, as `main` does; the Action installs styxx from PyPI, so it shows no PATH-2a
reason until a release carries the block.

The rules. A path claim (VERIFIED only; the path accusation is withheld on `main`) is kept only when
three readers without the mechanism verify it too: V97 (exact, then suffix, then base name — base
name only for a bare claim), V121 (keys keep their leading dots) and both. A file count abstains when
two changed paths differ only by a leading dot and the dot-kept count could read otherwise (a
CONTRADICTED count only when the dot-kept count range contains the claimed number; a VERIFIED count
unless that range is exactly it). `only_touches` abstains when V121 reads "every changed path lies under the
prefix set it uses" otherwise than `main` reads it under the set `main` uses, and, with a second prefix claimed,
when the leading prefix is a path for `main` only because its leading dots were dropped ("Only touches github and
.github/workflows/.": V121 says the prefix is not a path; `shape`). `tests_added` abstains when a
counted test is also defined in the removed lines and the claim lies in `[got − changed, got]`, read for each
port's `main` whose count gives the claim the verdict it has (each port's count is read exactly from the
bytes); `symbol_added` (VERIFIED) when a removed line defines the name where V101 reads a definition (at the line
start, after white space and an optional `async`). The claimed number is read from the claim's detail, not from
`main`'s reason, since the port's `main` prints a number of 10^21 or more in exponent form, and digits are read from a
fixed table of the ten ASCII digits, never through `int()`.

A decision reads the claim's kind, verdict and detail, `main`'s own counts in its reason, and the door's bytes — the
diff, the `--name-status` listing and the summary — never the claim's text, which the two ports cut (160 code points
in Python, 160 UTF-16 units here) and strip (`strip()` and `trim()` part on U+001C to U+001F, U+0085 and U+FEFF)
differently. Where the two ports' templates may read a claim's path, name, prefix or number apart, the claim abstains
in both (`extract`): an ASCII path, name or number that occurs in the summary inside a run of path, name or digit
characters holding a *wordish* code point; a name holding a code point from 0x80 up; a number the Python read in
digits outside ASCII; a path holding a code point from 0x80 up where some changed file with the claimed status has an
ASCII base name the path holds, so that the port's shorter reading could verify; and an `only_touches` prefix that
occurs in a sentence (cut where both ports cut one: a line feed, or `.`, `!` or `?` then a space, tab or carriage
return) in which a wordish or divergent code point follows that sentence's earliest `only`. *Wordish* is every code
point from 0x80 up but U+0085, U+2028, U+2029, U+FEFF and 1,828 *neutral* ones: the punctuation and symbols of the
Basic Multilingual Plane's punctuation, currency, arrow, operator, technical, box, shape, symbol, dingbat,
CJK-punctuation and full-width-punctuation blocks (connector punctuation, digits and letters left out), the white
space both ports share, and the emoji variation selectors. None of them is a word character or has a case in either
port, none folds to an ASCII letter, each is white space in both ports or in neither, and each one Unicode 3.2 knew
was punctuation, a symbol or a space there too — all pinned by enumeration on every runtime the tests run. Declared
(DECLARE-1) claims skip the summary test: their sentence is written from a value both ports read only when it is
ASCII.

Where the overlay cannot compute its readers exactly it abstains and says so:

| key | phrase |
|---|---|
| dir | the claim's path has a directory part, and only a changed file with the same base name in another directory matches it |
| tier | a changed path that matches the claim more closely than the one main resolved it to reads otherwise |
| dot | with leading dots kept, the changed path the claim resolves to reads otherwise |
| dot_earliest | with leading dots kept, the earliest changed path matching the claim reads otherwise, though the closest one does not |
| dot_tier | with leading dots kept and the closest match taken, the claim reads otherwise |
| count | two changed paths differ only by a leading dot, which the path key drops, and counted apart the claim reads otherwise |
| only | with leading dots kept, whether every changed path lies under the prefix reads otherwise |
| shape | with leading dots kept, no changed path has the prefix as a segment, so it is not read as a path |
| tests | a test the added lines count is also defined in the removed lines, and a changed test is not an added one |
| split | a test the added lines count is also defined in the removed lines, and the Python and JavaScript readers split or space these lines differently |
| symbol | the removed lines define this name too, and a changed definition is not an added one |
| extract | the summary holds a character the Python and JavaScript readers may read differently where the claim's path, name, prefix or number is read, so the two may extract the claim differently |
| divergent | a file header of this diff holds a character that the Python and JavaScript readers split or strip differently |
| odd | a path here has a drive-like prefix or a final '.' segment, where base names are read differently |
| case | a path here compares only where case outside ASCII is folded, which this overlay does not do |
| unreproduced | this overlay does not reproduce main's reading of the diff |
| unparsed | main's reason does not have the form this overlay reads |
| error | this overlay failed while reading the diff |

Running it, from the repository root:

    python -m pytest tests/test_diffgate_path2a.py tests/test_diffgate_path2a_truth.py tests/test_port_is_current.py
    cd web/gate/differential
    python path2a_recall.py --corpora DIR [EXTRA.json ...] [--truth] [--path-flavour posix|windows]
    node check_pairs.js                                   # the pinned pairs, path2a_pairs.json + path2a_moves.json

The port half of the PATH-2a tests, and `test_port_is_current`'s check of the port against the pinned pairs, need
`node`; without it they skip locally and fail under `CI` or `GITHUB_ACTIONS`. `path2a_recall.py` prints three totals:
`main`'s committed corpora (the figure below), the overlay's own pins, and `main`'s corpora with any extra files.
`path2a_pairs.json` pins 99 pairs, each for the decision it names (13 of them pin kind and verdict only, where
`main`'s two ports print different reasons); it is not one of `py_side.py`'s corpora (some of its pairs are inputs
`main`'s own two ports read differently), so the port differential over `main`'s corpora reads 0 disagreements.
`path2a_moves.json` records the one pinned claim of `main`'s own files the overlay moves —
`path1:unrepaired-typo` claim 0, ".githiub/workflows/dependabot.yml" resolved by base name to
`.github/workflows/dependabot.yml`, a false VERIFIED — so `path1_pairs.json` stays `main`'s record,
byte for byte.

**Measured**, on this file (`styxx/diffgate.py` `5007bcae…`, reader `9b620e00…`). Figures marked *pinned* are
asserted exactly by the committed tests, and hold on CPython 3.12.10 (Unicode 15.0) and 3.14.2 (16.0); the rest
were measured on CPython 3.12.10 and Node 24.13.0 (Unicode 16) and are not asserted.

Recall (D): of `main`'s decided claims, how many the overlay withholds (`path2a_recall.py`). `main`
reads `Path(p).name`, and a drive-like name such as `c:x.py` reads otherwise under the Windows and the
POSIX path flavour, so the figures are given under both where they differ.

| corpus | sha256 (LF) | decided | withheld |
|---|---|---|---|
| `corpus_real.json` | `1b21418a…` | 41 | 0 |
| `corpus_fuzz.json` | `2e80cd1d…` | 2,144 | 79 — 51 touched, 22 created and 5 deleted `dir`, 1 created `tier`; all #97, and all 79 are false by the statuses the fuzz generator wrote |
| `main`'s six pinned files | | 46 | 1 (the move above) |
| **`main`'s committed corpora** | | **2,231** | **80 (3.6%)**, both flavours |
| `path2a_pairs.json` (the overlay's own pins, built to abstain) | `ef7440ee…` | 112 (Windows), 111 (POSIX) | 56 (Windows), 55 (POSIX) |
| #161's `path2_pairs.json` (branch `fix/diffgate-path-resolution`) | `7ba272c8…` | 530 | 188 |
| `main`'s corpora + #161's pairs | | 2,761 | 268 (9.7%); #161's head withheld 493 of 2,749 |

One fewer is withheld on #161's pairs than at the previous head: #161's `w2` probe ("Added function col·leccio.").
Both ports' templates read the name `col`, the middle dot is now neutral, and `main`'s VERIFIED there is wrong for a
reason outside #97, #121 and #101, which the overlay does not answer for. The only real-world #97 record in `main`'s
corpora, `corpus_real` pr98 ("integrations/git/README.md — created."), is left as `main` has it: neither of its two
records is a false decided verdict (the created claim is UNCHECKABLE with a reason naming the wrong status, the
touched claim VERIFIED on the wrong file). Every abstention on `main`'s corpora is on synthetic input.

Coverage (B), judged by truth from base/head file models (`tests/test_diffgate_path2a_truth.py`,
*pinned*). A decided claim is attributable when `main`'s verdict is false and a variant of `main` without
the mechanism (V97, V121, V101, or all three) does not read it false.

| door | cases | main decides | false | attributable | withheld | right verdicts lost | undecided withheld |
|---|---|---|---|---|---|---|---|
| raw door (Python) | 1,706 | 7,614 | 1,535 | 1,206 | **1,206** | 245 of 5,897 | 35 of 182 |
| the port, in its own terms (variants built from `main`'s port) | 1,706 | 7,393 | 1,385 | 1,100 | **1,100** | 169 of 5,739 | 122 of 269 |
| git door (Python), the reproductions with their own `--name-status` | 96 | 63 | 17 | 14 | **14** | 0 of 46 | — |

The cases are #161's 101 reproductions, the 5 PREREG_path2 reproductions and 1,600 generated cases. In
the port, 942 of the 1,100 attributable claims carry the same `main` record as in Python; the other 158
are paths and names outside ASCII, which the two templates extract apart. #161's recorded `--name-status`
carries type changes but no rename or copy, so renames, a copy and a type change are read at a real git
door by `tests/test_diffgate_path2a.py`. The fourth review's three `shape` reproductions (a GNU deletion, a mnemonic
header, a rename into `.github/`, the last at both doors) are judged by truth in a test of their own. The truth test
prints any miss with its shape.

**Where "attributable" is narrower than the definition.** With a single `only_touches` prefix that is a path for
`main` only through its dropped dots, V121 abstains ("is not a path") and the overlay keeps `main`'s verdict: a
VERIFIED there is already withheld (`only`), and a kept CONTRADICTED is false only when every changed file lies under
the dotted directory while `main` lists a path outside it, which needs two independent faults in how the diff was
rendered. No truth world here holds one; a scan that needs no oracle (every variant must give a kept claim `main`'s
verdict) finds these kept claims and no other kind — 76 on the committed inputs, 37 on six fuzz sets. Withholding them
too (operator option O-10) would cost 141, 95 and 239 right verdicts in the three truth worlds (the
fourth review's figures). The other known gaps
(case-only merges with no dot, names `strip()` merges, multi-commit renderings, renderings over real `a/` or `b/`
directories, `async def` tests `main` does not count, `def` in non-Python files, and joint shapes where #121 is one of
two causes) are listed in the NOTEs.

Coverage was also measured on the third review's two independent truth worlds (2,000 and 6,000 cases built by
real git, each read at the git door, at the raw door on git's bytes and at the raw door on a second rendering;
two truth oracles; the builder's variants plus an ast-paired #101 variant; measured, not pinned): 0 attributable false verdicts kept at any door, in Python or in the port; right verdicts lost 297 and 785 at the raw doors, 75 and 210 at the git door, 297 and 780 in the port, as at `ea677740`.

Per phrase, on the raw door's truth world (measured): how often each fires, and whether `main`'s verdict
was false, right or undecided there.

| key | fired | false (attributable) | right | undecided |
|---|---|---|---|---|
| count | 407 | 370 (366) | 37 | 0 |
| dir | 423 | 404 (404) | 18 | 1 |
| dot | 183 | 177 (177) | 6 | 0 |
| dot_earliest | 6 | 1 (1) | 5 | 0 |
| dot_tier | 111 | 107 (107) | 4 | 0 |
| tier | 17 | 12 (12) | 1 | 4 |
| case | 2 | 2 (2) | 0 | 0 |
| only | 11 | 9 (9) | 2 | 0 |
| shape | 43 | 0 (0) | 43 | 0 |
| tests | 132 | 54 (51) | 53 | 25 |
| symbol | 35 | 30 (30) | 0 | 5 |
| extract | 167 | 91 (47) | 76 | 0 |

The costs, as the independent worlds read them (measured on the third review's world A, 2,000 cases, and world
B, 6,000 cases; "right lost" is a right verdict of `main` withheld):

- `tests` pairs names across the whole diff, so it also withdraws right verdicts where a test is moved into or
  out of a class, a test file is renamed, a test name is reused in another module beside a changed one, two
  non-ASCII test names share their ASCII run, or a removed line of prose or of another language holds `def test_`
  (V101's pairing reads it there too). Right verdicts lost: world A 162 of 2,331 at the raw door (7.0%)
  and 69 of 1,244 at the git door (5.5%); world B 472 of 6,939 (6.8%) and 195 of 3,669 (5.3%). The builder's own
  world above loses 53. Operator option O-8 (pair by ASCII run only when the name is all ASCII) is not taken.
- `dot_earliest` caught no false verdict in either world (A: 18 right, 4 mixed; B: 31 right, 16 mixed). At
  world B's git door it withholds 12 of 8,859 right path verdicts, the only path phrase there that loses any.
  Operator option O-6 would keep a claim both V97 and V97+V121 verify.
- `dir` withholds right verdicts on renderings whose prefixes `main` reads as directories or drops: at world B's
  raw door, mnemonic (`c/` `w/`) 158 of 1,189 right path verdicts, plain 48 of 990, index 42 of 890, no-prefix 36
  of 1,050. `main`'s right verdict there rests on the #97 base-name fallback. On git's own bytes `dir` withholds
  none.
- `shape` fired 43 times in the builder's world, every time on a right verdict ("Only touches cfg and .cfg/app/."
  beside files that all lie under the second prefix), and never in worlds A and B.
- `extract` reads every occurrence of the claim's path, name or number in the summary, since the claim's own position
  differs between the ports: a name that also occurs elsewhere next to a letter outside ASCII is withheld (#161's
  `x1` probes, "Added function fo²o. … Added function fo."). An `only_touches` claim is withheld wherever a
  wordish or divergent code point follows `only` in its sentence: an accented name ("reviewed by José"), or an emoji
  outside the Basic Multilingual Plane ("Only touches docs/ 🎉"; operator option O-11 would read those pictographs
  as neutral). Punctuation and symbols of the Basic Multilingual Plane — guillemets, low-9 quotes, CJK and
  full-width punctuation, ✔, ✅, ·, §, ⇒ — withhold nothing; each of those shapes is a pinned pair.

By construction (A): over the 6,635 committed inputs (the pinned pairs, the 3,000 fuzz pairs regenerated in memory,
#161's reproductions and pair inputs, a seeded 2,000-pair PATH-2a fuzz and a seeded 1,000-input text-seam set), both
strict modes, every branch record is `main`'s but for abstentions in reach with the overlay's reason, each claim
reads the same with `--strict` as without it, and where `main` raises the branch raises the same exception — in
Python and in the port (13,270 port runs, 0 broken; the relation compares `unparsed_claims` and each claim's keys
too); the per-key abstention counts are *pinned* for both path flavours and for Unicode 13.0 and 14.0 (emulated by
re-reading the inputs with their five Unicode 14 letters mapped to code points no version assigns), 15.0 and 16.0;
Unicode 15.1 (CPython 3.13) was not measured and fails the test by name. Measured beyond the committed inputs, against
`main` reconstructed (sha-equal to `git show 1cde8b82`'s two files) and, in the port, `main`'s own file from `git
show`: the reviewers' hostile (two seeds), calm, lone-surrogate, cross-port, text-seam, realistic-text and token-adjacency fuzz and the previous pass's committed set — 203,606 inputs, each run in both strict modes in Python and without `--strict` in the port: 0 outside the relation, every raise the same, the error phrase never (CPython 3.12.10; 72,606 of them again on 3.14.2, 0) — and the rebuilt bookmarklet, loaded in a stub page, equals this port on 37,288 runs (18,644 inputs, both strict modes).

The git door and the raw door can part, each still against its own `main`: where `main`'s two status maps differ
(a `diff.noprefix` header `main`'s raw reader cannot parse is still in `--name-status`) or a header holds a
divergent character (a path with U+2028 under `core.quotepath=false`), the raw door may withhold a claim the git
door keeps. `test_git_door` asserts the doors agree only where `main`'s claims, `main`'s status maps and the
divergent guard all allow it, and reads both repositories at both doors. At a real git door (574 repositories built with plumbing — symlinks, gitlinks, renames, copies, mode and type changes, hostile names and diff configs; 26 of 600 did not build): 9,184 runs, 0 outside the relation, the raw door read on the same `git diff` bytes each time; 63 claims read differently at the two doors while `main`'s two records agree (of the 40 the run lists, 39 under a divergent header and 1 where `main`'s two status maps differ).

Cross-port (C). What holds by construction: a decision reads nothing of a claim but its kind, verdict and detail (and
the door's bytes), so wherever `main`'s two ports give claims the same (kind, verdict, detail) — at the same position,
matched across the two lists, or anywhere in either — the overlay decides them alike (asserted). What is measured,
and asserted on the committed inputs only: claims at the same position with the same (kind, verdict, text); claims
left over in lists of equal length whose details differ but may be one match read two ways (they read the same
fields, the two values of each field nest, and each lies in the other claim's text — where the `extract` guards do
their work); and, on inputs whose claims all pair so, the gate verdict. Left-over claims that are not one match
(each port reading a different sentence, as in "Changed the parser in `docs/résumé/index.md`. Added
`docs/café.md`.") are not paired, and their decisions may differ, as the claims do. On the committed inputs (CPython 3.12.10) and on 14 further sets (223,084 inputs: the three reviews' hostile, calm, lone-surrogate, cross-port, text-seam and token-adjacency fuzz, two realistic-text sets and the previous pass's committed set): 480,520 claims at the same position with the same (kind, verdict, detail), 471,482 with the same (kind, verdict, text), 531,195 matched across the lists, 23,171 left over as one match — 0 splits under any key, and 0 gate splits on the 180,943 inputs whose claims all pair (measured).

The overlay does add gate disagreements where `main`'s two gate verdicts agree but some claim rows differ between
`main`'s ports (a claim one port reads and the other does not, or reads with another verdict): each port withholds a
claim its own `main` decides. On the committed inputs that is 5 inputs without `--strict` — #161's four f2 separator
cases (`path2:f2-a-vertical-tab-…`, `-form-feed-…`, `-line-separator-…`, `-paragraph-separator-…`) and
`path2:y2-a-changed-test-beside-a-created-bom-test-under-bare-hunks`, pinned by the test so a sixth fails. On the 223,084 inputs above, 1,523 inputs without `--strict` and 614 with it, most of them hostile or token-adjacency input; on the 40,000 realistic-text inputs, 0 without `--strict` and 48 with it, while the overlay makes 373 of `main`'s 373 non-strict gate splits there agree. Nearly all of the 48 (26 of the 27 in one of the two sets) are inputs where one port's `main` reads a claim the other's does not (a path holding letters outside ASCII) and the overlay withholds it on that side.
The bookmarklet calls the gate without `--strict`. Operator option O-9 (abstain in both ports wherever either port's
reading fires) is not taken.

The static checks. `selfcheck_p2a_only_abstains` re-derives from the Python block's source that it can only
abstain and reads no Unicode table at run time: no import; only names the block binds or a short list of builtins
and `main`'s names (`int` is refused by name: it reads CPython's table of decimal digits, so the block reads digits
from a fixed table); only a short list of attributes (so no dunder and no unlisted method); `re` only as
`re.<function>(<static pattern>)`; no strip or split without its characters; no `str()`, `repr`, `format`, `!r`,
`getattr` or `eval`. The port's token scan reads code with white space removed between tokens, refuses `normalize`
anywhere, `new String(`, `.match(`, `.search(` and `.matchAll(`, allows `RegExp` only as `new RegExp(<one static
string>)` whose decoded value is checked, and refuses computed member access outside a short list of index
expressions and any call on a computed member. Every plant the third and fourth reviews found is a committed refused
case. These checks are static and infer no types (an f-string of a list would read `repr` and pass); the block holds
none.

Cost per call. Python: +15% on the committed inputs over `main` (6,635 inputs, CPython 3.12.10); `corpus_real.json`
7.93 ms on `main` and 7.90 ms here per call, `corpus_fuzz.json` 0.204 → 0.227 ms. Node: +21% on the committed inputs.
The worst shapes the reviews found are bounded by committed timing tests that time the overlay alone on `main`'s
record, the least of three runs, in both ports: a `def` beside a 5,000- or 50,000-code-point run of letters outside
ASCII, one 220 KB added line of `def test_a`, 2,000 files with 200 path claims under 50 directories, 2,000 files
that all share the base name `__init__.py` with 200 path claims, 500 symbol claims over 50,000 removed `def` lines,
300 `class` claims beside NBSP runs, and four 64 KB summaries whose path, symbol, scope and count claims each meet
thousands of runs or zones — each under 0.5 s in Python and 0.3 s here. Measured the test's way, five times over
(CPython 3.12.10): the symbols case 0.088 to 0.101 s, the others 0.039 s or less; in Node 24.13.0 the slowest case
takes 37 to 41 ms. On `ea677740` the overlay alone took 3.76 s on the 64 KB path case, on `40bba05b` 38.6 s on the
symbols. A large diff costs more than the corpus figures suggest, because the overlay reads the diff once more: on a
35 MB diff of 4,000 files the overlay takes 2.2 to 3.1 times a `main` call in Python (0.54 → 1.22 s for one path claim, 0.59 → 1.86 s for five claims together, CPython 3.12.10 under load) and 1.9 to 3.5 times here, with peak memory 127 MB against `main`'s 122 MB (at `ea677740`: up to 5.3 times, and 182 MB against 128 MB in the review's run). The bookmarklet grew from 24,335 (`main`) to 40,419 characters.

## What this is not

It is not a second instrument. The Python is the instrument; the JavaScript exists only where the
Python cannot run, and the day the two disagree on a pair in this corpus, the JavaScript is wrong
by definition. It is not a check of anything but transliteration fidelity — the template set's
own precision was measured elsewhere (EXTERNAL-1: path accusations at 0.23 against a 0.95 floor
on 71,016 agent-authored PRs, and withheld since), and no number here bears on that.
