# web/gate — the diff gate where Python cannot run

The instrument is `styxx/diffgate.py`. Two browser surfaces cannot import it: the paste-in
preview page and the bookmarklet. They run `diffgate.js`, a JavaScript transliteration of one
specific file — `styxx/diffgate.py` on this branch, sha256
`2ffa83b8a5654b9a369d46efa119848c2c6d347ad86e42eaec5917a3ab09b5a5` (LF line endings; a wheel
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

    bookmarklet.min.js    sha256 bfe8c047244b6d2d373a965fa6197af1a96e805f880b29d81692a0870b0729f5   36,933 chars
    bookmarklet.href.txt  sha256 6991eb41eeee314dec37cd8a390af9a8a2cb38de02938f4778cc7e9948466393   36,944 chars

(Earlier builds: `b04d14dc…`, 11,437 chars, from the 7.47.0 file — accuses outside Python;
`9ea8f572…`, 17,686 chars, the BC-2 + COMPAT-1 re-cut — cannot see a binary file;
`1be19a65…`, 24,335 chars, `main` at `1cde8b82` before PATH-2a, whose README still named an older
`54dca73a…`, 21,632 chars; `c457cca3…`, 34,285 chars, PATH-2a's pass 1. A bookmark that hashes to any of
these is an old port; drag the new one. The panel text still names the 7.48.0 port; see *PATH-2a* below.)

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
    python py_side.py                    # refuses to run unless styxx/diffgate.py hashes to 2ffa83b8…
    node js_side.js
    python differential.py
    node check_pairs.js                  # the 110 pinned pairs against their expect blocks (+ path2a_moves.json)

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
print 0 disagreements on this branch: the expected disagreements are exactly the overlay's abstentions,
which `path2a_recall.py` counts (80 on `main`'s committed corpora). Any other disagreement means the
release is not the file the port claims to be, and the header of `diffgate.js` says which one it is.

## PATH-2a: abstaining where #97, #121 or #101 can make a verdict wrong

`styxx/diffgate.py` and `diffgate.js` each carry one block between the markers
`=== PATH-2a abstain-only overlay: BEGIN ===` and `... END ===`
(`papers/closed-model-frontier/NOTE_path2a_abstain_overlay_2026_09_30.md`, and for the second review pass
`NOTE_path2a_second_pass_2026_09_30.md` and `NOTE_path2a_second_pass_corrections_2026_09_30.md`). `main`'s reader
runs unchanged — at the raw door, at the git door and in this port — and then the block reads each decided claim once
more. Where the #97 mechanism (the earliest entry in diff order matching by exact path, suffix or base name), the #121
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
branch instead.

The rules. A path claim (VERIFIED only; the path accusation is withheld on `main`) is kept only when
three readers without the mechanism verify it too: V97 (exact, then suffix, then base name — base
name only for a bare claim), V121 (keys keep their leading dots) and both. A file count abstains when
two changed paths differ only by a leading dot and the dot-kept count could read otherwise (a
CONTRADICTED count only when the dot-kept count range contains the claimed number; a VERIFIED count
unless that range is exactly it). `only_touches` abstains when keeping the dots changes whether every
changed path lies under the prefix. `tests_added` abstains when a counted test is also defined in the
removed lines and the claim lies in `[got − changed, got]`, read for each port's `main` whose count
gives the claim the verdict it has (each port's count is read exactly from the bytes); `symbol_added`
(VERIFIED) when a removed line defines the name. Where the overlay cannot compute these readers
exactly, or the two ports' templates read the claim's path, name or prefix apart, it abstains and
says so:

| key | phrase |
|---|---|
| dir | the claim names a directory, and only a file of the same name elsewhere matches it |
| tier | a changed path that matches the claim more closely than the one main resolved it to reads otherwise |
| dot | with leading dots kept, the changed path the claim resolves to reads otherwise |
| dot_earliest | with leading dots kept, the earliest changed path matching the claim reads otherwise, though the closest one does not |
| dot_tier | with leading dots kept and the closest match taken, the claim reads otherwise |
| count | two changed paths differ only by a leading dot, which the path key drops, and counted apart the claim reads otherwise |
| only | with leading dots kept, whether every changed path lies under the prefix reads otherwise |
| tests | a test the added lines count is also defined in the removed lines, and a changed test is not an added one |
| split | a test the added lines count is also defined in the removed lines, and the Python and JavaScript readers split or space these lines differently |
| symbol | the removed lines define this name too, and a changed definition is not an added one |
| extract | the claim's path or name touches a character outside ASCII, where the Python and JavaScript readers extract it differently |
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

`path2a_recall.py` prints three totals: `main`'s committed corpora (the figure below), the overlay's own
pins, and `main`'s corpora with any extra files. `path2a_pairs.json` pins 56 pairs, each for the decision
it names; it is not one of `py_side.py`'s corpora (two of its pairs are inputs `main`'s own two ports
read differently), so the port differential over `main`'s corpora reads 0 disagreements.
`path2a_moves.json` records the one pinned claim of `main`'s own files the overlay moves —
`path1:unrepaired-typo` claim 0, ".githiub/workflows/dependabot.yml" resolved by base name to
`.github/workflows/dependabot.yml`, a false VERIFIED — so `path1_pairs.json` stays `main`'s record,
byte for byte.

**Measured**, on this file (`styxx/diffgate.py` `2ffa83b8…`, reader `9b620e00…`), CPython 3.12.10 and 3.14.2,
Node 24.13.0 (Unicode 16). Figures marked *pinned* are asserted exactly by the committed tests; the
rest were measured and are not asserted.

Recall (D): of `main`'s decided claims, how many the overlay withholds (`path2a_recall.py`). `main`
reads `Path(p).name`, and a drive-like name such as `c:x.py` reads otherwise under the Windows and the
POSIX path flavour, so the figures are given under both where they differ.

| corpus | sha256 (LF) | decided | withheld |
|---|---|---|---|
| `corpus_real.json` | `1b21418a…` | 41 | 0 |
| `corpus_fuzz.json` | `2e80cd1d…` | 2,144 | 79 — 51 touched, 22 created and 5 deleted `dir`, 1 created `tier`; all #97, and all 79 are false by the statuses the fuzz generator wrote |
| `main`'s six pinned files | | 46 | 1 (the move above) |
| **`main`'s committed corpora** | | **2,231** | **80 (3.6%)**, both flavours |
| `path2a_pairs.json` (the overlay's own pins, built to abstain) | `4997abf6…` | 68 (Windows), 67 (POSIX) | 40 (Windows), 39 (POSIX) |
| #161's `path2_pairs.json` (branch `fix/diffgate-path-resolution`) | `7ba272c8…` | 530 | 184 |
| `main`'s corpora + #161's pairs | | 2,761 | 264 (9.6%); #161's head withheld 493 of 2,749 |

The only real-world #97 record in `main`'s corpora, `corpus_real` pr98 ("integrations/git/README.md —
created."), is left as `main` has it: neither of its two records is a false decided verdict (the created
claim is UNCHECKABLE with a reason naming the wrong status, the touched claim VERIFIED on the wrong file).
Every abstention on `main`'s corpora is on synthetic input.

Coverage (B), judged by truth from base/head file models (`tests/test_diffgate_path2a_truth.py`,
*pinned*). A decided claim is attributable when `main`'s verdict is false and a variant of `main` without
the mechanism (V97, V121, V101, or all three) does not read it false.

| door | cases | main decides | false | attributable | withheld | right verdicts lost | undecided withheld |
|---|---|---|---|---|---|---|---|
| raw door (Python) | 1,706 | 7,614 | 1,535 | 1,206 | **1,206** | 202 of 5,897 | 35 of 182 |
| the port, in its own terms (variants built from `main`'s port) | 1,706 | 7,393 | 1,385 | 1,100 | **1,100** | 126 of 5,739 | 122 of 269 |
| git door (Python), the reproductions with their own `--name-status` | 96 | 63 | 17 | 14 | **14** | 0 of 46 | — |

The cases are #161's 101 reproductions, the 5 PREREG_path2 reproductions and 1,600 generated cases. In
the port, 942 of the 1,100 attributable claims carry the same `main` record as in Python; the other 158
are paths and names outside ASCII, which the two templates extract apart. #161's recorded `--name-status`
carries type changes but no rename or copy, so renames, a copy and a type change are read at a real git
door by `tests/test_diffgate_path2a.py`. The truth test prints any miss with its shape. The known gaps
(case-only merges with no dot, names `strip()` merges, multi-commit renderings, renderings over real `a/`
or `b/` directories, `async def` tests `main` does not count, `def` in non-Python files, and joint shapes
where #121 is one of two causes) are listed in the NOTEs.

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
| tests | 132 | 54 (51) | 53 | 25 |
| symbol | 35 | 30 (30) | 0 | 5 |
| extract | 167 | 91 (47) | 76 | 0 |

`tests` pairs names across the whole diff, so it also withdraws right verdicts where a test name is
reused in another class or module beside a changed one, or where two non-ASCII test names share their
ASCII run; `extract` withdraws right verdicts on paths outside ASCII, which the two ports read apart;
`dot_earliest` withdraws a right verdict wherever V121 alone, which still carries #97, reads otherwise.
Each is an operator option in the NOTEs.

By construction (A): over the 5,592 committed inputs (the pinned pairs, the 3,000 fuzz pairs regenerated
in memory, #161's reproductions and pair inputs, and a seeded 2,000-pair PATH-2a fuzz), both strict
modes, every branch record is `main`'s but for abstentions in reach with the overlay's reason, each
claim reads the same with `--strict` as without it, and where `main` raises the branch raises the same
exception — in Python and in the port (11,184 port runs, 0 broken); the per-key abstention counts are
*pinned* for both path flavours and for Unicode 13.0 to 16.0 (13.0 and 14.0 by re-reading the inputs with their
five Unicode 14 letters mapped to code points no version assigns). Measured beyond the committed inputs: the
reviewers' 26,000-case hostile, 22,000-case calm and 3,000-case lone-surrogate fuzz, and two 26,000- and
36,000-case cross-port fuzz sets, 0 broken in either port.

Cross-port (C): on the committed inputs, 18,331 claims get the same kind, verdict and text from `main`'s
two ports, and the overlay gives all 18,331 the same verdict and phrase in both (asserted: any split
fails the test; on `e12214e5` there were 17); 18,137 get the same whole record, and the overlay's records
agree on all of them. No input on which `main`'s ports agree on every claim's kind, verdict and text gets
two gate verdicts. On the 113,000 fuzz inputs above: 345,397 of 345,397 claims (measured).

Cost per call. Python, summed over both strict modes: the committed inputs +24%, the fuzz sets +24% to
+37% over `main`; `corpus_real.json` 7.75 ms on `main` and 7.52 ms on the branch per call,
`corpus_fuzz.json` 0.197 → 0.217 ms. Node: +20% to +35% on the same sets, the overlay's own worst call
7 ms. The worst cases the second review found (a `def` beside a 5,000- or 50,000-character run of letters
outside ASCII, one 280 KB line of `def test_a`, 2,000 files with 200 path claims) are bounded by committed
timing tests in both ports: under 1 s a call, and on the 2,000-file case the overlay's own cost under 0.5 s
(Python) and 0.3 s (Node) beyond `main`'s. On `e12214e5` the 5,000 run took 10.9 s in Python, the 280 KB
line 17.3 s in Node, and the 2,000-file case 10.9 s beyond `main` in Python (1.5 s in Node). The
bookmarklet grew from 24,335 (`main`) to 36,933 characters.

## What this is not

It is not a second instrument. The Python is the instrument; the JavaScript exists only where the
Python cannot run, and the day the two disagree on a pair in this corpus, the JavaScript is wrong
by definition. It is not a check of anything but transliteration fidelity — the template set's
own precision was measured elsewhere (EXTERNAL-1: path accusations at 0.23 against a 0.95 floor
on 71,016 agent-authored PRs, and withheld since), and no number here bears on that.
