# web/gate — the diff gate where Python cannot run

The instrument is `styxx/diffgate.py`. Two browser surfaces cannot import it: the paste-in
preview page and the bookmarklet. They run `diffgate.js`, a JavaScript transliteration of one
specific file — `styxx/diffgate.py` as it stands on `main` (BC-2 + COMPAT-1 + BIN-2 + COMPAT-2,
pull requests #113, #115, #120 and #124, plus the `fetch_pr` door; the file **7.48.0** ships),
sha256 `04ec58c3ac3c3e21fec08dab8e236899b7194fc65773b673678e2b491690c63e` (LF line endings; a wheel
built on Windows carries CRLF and hashes differently, so `py_side.py` normalises before it
compares) — and this directory is the receipt for that port: the differential test that holds
it to the Python's output, and the build that turns it into the bookmarklet people drag into
their bookmarks bar. That file is `main`'s `9b620e00a19464589308a987819894ae7cc3c111c66a5f8a457a84b8a6c604eb`
reader, unchanged, plus the PATH-2a block (an overlay that only abstains; see *PATH-2a* below), and a
committed test cuts the block out and gets `main`'s file back byte for byte, in both languages.

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

    bookmarklet.min.js    sha256 c457cca3d0672f2c321caad8e40f9e21f53971ccfd859778582c98ffc041ebc4   34,285 chars
    bookmarklet.href.txt  sha256 4f6261b0e074dfb257bee473c49f2a41a0cddfad62c7d95da16c6bcc066f3950   34,296 chars

(Earlier builds: `b04d14dc…`, 11,437 chars, from the 7.47.0 file — accuses outside Python;
`9ea8f572…`, 17,686 chars, the BC-2 + COMPAT-1 re-cut — cannot see a binary file;
`1be19a65…`, 24,335 chars, `main` at `1cde8b82` before PATH-2a, whose README still named an older
`54dca73a…`, 21,632 chars. A bookmark that hashes to any of these is an old port; drag the new one.)

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
    python py_side.py                    # refuses to run unless styxx/diffgate.py hashes to 186d5f2c…
    node js_side.js
    python differential.py
    node check_pairs.js                  # the 103 pinned pairs against their expect blocks (+ path2a_moves.json)

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

When 7.48.0 is on PyPI, `py_side.py --installed` should print 0 disagreements against it; if it
does not, the release is not the file the port claims to be, and the header of `diffgate.js`
says which one it is.

## PATH-2a: abstaining where #97, #121 or #101 can make a verdict wrong

`styxx/diffgate.py` and `diffgate.js` each carry one block between the markers
`=== PATH-2a abstain-only overlay: BEGIN ===` and `... END ===`
(`papers/closed-model-frontier/NOTE_path2a_abstain_overlay_2026_09_30.md`). `main`'s reader runs
unchanged — at the raw door, at the git door and in this port — and then the block reads each decided
claim once more. Where the #97 mechanism (the earliest entry in diff order matching by exact path,
suffix or base name), the #121 mechanism (`_norm`'s `lstrip("./")`, which gives `.x` and `x` one key)
or the #101 mechanism (a changed `def` counted as added) can have made `main`'s verdict wrong, the
claim becomes UNCHECKABLE with the reason

    {V} withheld by PATH-2a ({defect}): {phrase}. main's reading: {main's reason, verbatim}

and the gate verdict is recomputed with `main`'s own formula. Nothing else in the record moves: the
claim list, every other verdict and every other reason are `main`'s, byte for byte. Cut the block
out and revert the door hooks (two lines per file) and you have `main`'s two files back, sha for
sha; `tests/test_diffgate_path2a.py` does exactly that and uses the result as its reference.

**The three defects are not repaired.** PATH-2a never gives VERIFIED where `main` was wrong; it only
stops `main`'s false verdicts on these shapes from standing, and says which verdict it withheld and
why. PREREG_path2's G-P1 (on #161's branch) expects VERIFIED on the reproductions, so PATH-2a does not
meet G-P1: whether it stands in for it is the operator's decision. `--strict` fails on every new
abstention, as on any UNCHECKABLE.

The rules. A path claim (VERIFIED only; the path accusation is withheld on `main`) is kept only when
three readers without the mechanism verify it too: V97 (exact, then suffix, then base name — base
name only for a bare claim), V121 (keys keep their leading dots) and both. A file count abstains when
two changed paths differ only by a leading dot and the dot-kept count could read otherwise (a
CONTRADICTED count only when the dot-kept count range contains the claimed number; a VERIFIED count
unless that range is exactly it). `only_touches` abstains when keeping the dots changes whether every
changed path lies under the prefix. `tests_added` abstains when a counted test is also defined in the
removed lines and the claim lies in `[got − changed, got]`; `symbol_added` (VERIFIED) when a removed
line defines the name. Where the overlay cannot compute these readers exactly it abstains and says so:

| key | phrase |
|---|---|
| dir | the claim names a directory, and only a file of the same name elsewhere matches it |
| tier | a changed path that matches the claim more closely than the one main resolved it to reads otherwise |
| dot | with leading dots kept, the changed path the claim resolves to reads otherwise |
| dot_tier | with leading dots kept and the closest match taken, the claim reads otherwise |
| count | two changed paths differ only by a leading dot, which the path key drops, and counted apart the claim reads otherwise |
| only | with leading dots kept, whether every changed path lies under the prefix reads otherwise |
| tests | a test the added lines count is also defined in the removed lines, and a changed test is not an added one |
| symbol | the removed lines define this name too, and a changed definition is not an added one |
| divergent | a file header of this diff holds a character that the Python and JavaScript readers split or strip differently |
| odd | a path here has a drive-like prefix or a final '.' segment, where base names are read differently |
| case | a path here compares only where case outside ASCII is folded, which this overlay does not do |
| unreproduced | this overlay does not reproduce main's reading of the diff |
| unparsed | main's reason does not have the form this overlay reads |
| error | this overlay failed while reading the diff |

Running it:

    cd web/gate/differential
    python path2a_recall.py --corpora DIR [EXTRA.json ...] [--truth]   # recall; DIR holds the gitignored corpora
    node check_pairs.js                                                # path2a_pairs.json + path2a_moves.json
    python -m pytest tests/test_diffgate_path2a.py tests/test_diffgate_path2a_truth.py tests/test_port_is_current.py

`path2a_pairs.json` pins 49 pairs, each for the decision it names. `path2a_moves.json` records the one
pinned claim of `main`'s own files the overlay moves — `path1:unrepaired-typo` claim 0,
".githiub/workflows/dependabot.yml" resolved by base name to `.github/workflows/dependabot.yml`, a
false VERIFIED — so `path1_pairs.json` stays `main`'s record, byte for byte.

**Measured**, on this file (`styxx/diffgate.py` `186d5f2c…`, reader `9b620e00…`), CPython 3.12.10 and
Node 24.13.0 (Unicode 16); the committed tests re-derive every figure marked *pinned*.

Recall (D): of `main`'s decided claims, how many the overlay withholds (`path2a_recall.py`).

| corpus | sha256 (LF) | decided | withheld |
|---|---|---|---|
| `corpus_real.json` | `1b21418a…` | 41 | 0 |
| `corpus_fuzz.json` | `2e80cd1d…` | 2,144 | 79 — 51 touched, 23 created, 5 deleted; all #97, and all 79 are false by the statuses the fuzz generator wrote |
| `main`'s six pinned files | | 46 | 1 (the move above) |
| **`main`'s committed corpora** | | **2,231** | **80 (3.6%)** |
| `path2a_pairs.json` (the overlay's own pins) | `ff735091…` | 59 | 34 |
| #161's `path2_pairs.json` (branch `fix/diffgate-path-resolution`) | `7ba272c8…` | 530 | 172 |
| `main`'s corpora + #161's pairs | | 2,761 | 252 (9.1%); #161's head withheld 493 of 2,749 |

Coverage (B), judged by truth from base/head file models (`tests/test_diffgate_path2a_truth.py`,
*pinned*): on #161's 101 reproductions, the 5 PREREG_path2 reproductions and 1,600 generated cases,
main decides 7,614 claims and 1,535 of them are false; 1,206 of those are attributable to #97, #121
or #101 (a variant of `main` without the mechanism does not read them false), and **all 1,206 are
withheld**, in Python and, wherever the port's `main` reads the claim alike (942 of them), in the
port. The cost: 142 of 5,897 right verdicts withheld, 35 of 182 undecided ones. A wider scratch run
over 35,000 generated cases from the design stage (not committed) read the same way: 22,885 of 22,885
attributable false verdicts withheld, 2,633 of 123,031 right verdicts lost (2.1%). The truth test
prints any miss with its shape. The known gaps (case-only merges with no dot, names `strip()` merges,
multi-commit renderings, renderings over real `a/` or `b/` directories, `async def` tests `main` does
not count, `def` in non-Python files) are listed in the NOTE.

By construction (A), *pinned*: over the 5,585 committed inputs (the pinned pairs, the 3,000 fuzz pairs
regenerated in memory, #161's reproductions and pair inputs, and a seeded 2,000-pair PATH-2a fuzz),
both strict modes, every branch record is `main`'s but for abstentions in reach with the overlay's
reason, in Python and in the port (11,170 port runs, 0 broken); where `main` raises, the branch
raises the same exception. The overlay's error fallback never fires on them.

Cross-port (C), *pinned*: on the committed inputs, 18,133 claims get the same record from `main`'s
two ports, and the overlay decides all 18,133 alike, reason for reason; on the 380 claims `main`'s
ports already read differently, 123 get different overlay decisions (reported, not asserted). A wider
scratch run adding `corpus_real.json` and 17,000 generated cases: 124,229 of 124,229.

Cost per call (Python, mean): `corpus_real.json` 7.49 ms on `main`, 7.30 ms on the branch (noise);
`corpus_fuzz.json` 0.183 → 0.208 ms. The bookmarklet grew from 24,335 to 34,285 characters.

## What this is not

It is not a second instrument. The Python is the instrument; the JavaScript exists only where the
Python cannot run, and the day the two disagree on a pair in this corpus, the JavaScript is wrong
by definition. It is not a check of anything but transliteration fidelity — the template set's
own precision was measured elsewhere (EXTERNAL-1: path accusations at 0.23 against a 0.95 floor
on 71,016 agent-authored PRs, and withheld since), and no number here bears on that.
