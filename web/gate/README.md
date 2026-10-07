# web/gate — the diff gate where Python cannot run

The instrument is `styxx/diffgate.py`. Two browser surfaces cannot import it: the paste-in
preview page and the bookmarklet. They run `diffgate.js`, a JavaScript transliteration of one
specific file — `styxx/diffgate.py` on this branch, sha256
`09867056957082f129985591f5ed048021e85ed48627e73854494b4a1e6e2b3a` (LF line endings; a wheel
built on Windows carries CRLF and hashes differently, so `py_side.py` normalises before it
compares) — and this directory is the receipt for that port: the differential test that holds
it to the Python's output, and the build that turns it into the bookmarklet people drag into
their bookmarks bar. That file is `main`'s reader (BC-2 + COMPAT-1 + BIN-2 + COMPAT-2, pull requests
#113, #115, #120 and #124, plus the `fetch_pr` door) — the file **7.48.0** ships, sha256
`9b620e00a19464589308a987819894ae7cc3c111c66a5f8a457a84b8a6c604eb`, with one change outside any reading
path: its `--demo` prints a CONTRADICTED verdict as CONTRADICTED, not LIE (sha256
`68873068c0a665c2f5a2c40e2ca4d1e2cfccf28318f6f8d6fc03087d51fcdcc1`) — plus the PATH-2a block (an
overlay that only abstains, not yet released; see *PATH-2a* below), and a committed test cuts the
block out and gets that reader back byte for byte, in both languages.

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
nothing stored, nothing sent anywhere) and pins `[ok ]` / `[CONTRADICTED]` / `[ ? ]` lines, each with its reason, the verdict
and the never-read count to the page.

`build_bookmarklet.py` — assembles the two into `bookmarklet_src.js`, minifies with
`terser -c -m --format ascii_only`, writes `bookmarklet.min.js` and `bookmarklet.href.txt`.
`--check` rebuilds and compares against the committed files. The shipped bookmarklet is

    bookmarklet.min.js    sha256 5843f53d0a982216ecffa0919c044c8d49966c68e5310d319a3509306f5fbdf3   53,380 chars
    bookmarklet.href.txt  sha256 77d7a002a259019ee50a4e65bd7464628d3560511982bbb425187c40d5bf45fc   53,391 chars

(Earlier builds: `b04d14dc…`, 11,437 chars, from the 7.47.0 file — accuses outside Python;
`9ea8f572…`, 17,686 chars, the BC-2 + COMPAT-1 re-cut — cannot see a binary file;
`1be19a65…`, 24,335 chars, `main` at `1cde8b82` before PATH-2a, whose README still named an older
`54dca73a…`, 21,632 chars; `c457cca3…`, 34,285 chars, PATH-2a's pass 1; `bfe8c047…`, 36,933 chars, its pass 2;
`c726c904…`, 38,763 chars, its pass 3; `eeb0c298…`, 40,419 chars, its pass 4; `185352f8…`, 44,199 chars, its pass 5;
`7dd3628e…`, 48,227 chars, its pass 6; `98b1f5ad…`, 49,888 chars, its pass 7; `f873dc6a…`, 56,575 chars, its pass 8;
`c72ae2db…`, 51,838 chars, its pass 9; `e550ecbd…`, 52,937 chars, its pass 10; `3659b422…`, 53,246 chars, its pass 11;
`6bf6121a…`, 53,371 chars, PATH-2a as merged (#187), whose panel labelled a CONTRADICTED verdict LIE.
A bookmark that hashes to any of these is an old port; drag the new one. The panel text still names the 7.48.0 port; see *PATH-2a* below.)

Whatever a browser holds under that bookmark either hashes to the first line (drop the
`javascript:` prefix) or is not this build. terser 5.46.0 produced these bytes, and rebuilds
`main`'s 24,335-char build from `main`'s sources byte for byte. (`build_bookmarklet.py`'s docstring, which is `main`'s
text and is not edited here, names terser 5.51.2: the version that built `main`'s bookmarklet. Its `--check` writes
`bookmarklet_src.js` before it compares, so run it on a copy; the committed test does.)

`differential/` — the test. Read on.

## The differential test

A port is held to the original by output, not by trust. `differential/` builds a corpus of
(summary, diff) pairs, runs the released Python and the JavaScript over every pair, and compares
the records field by field: verdict, measured, why_unmeasured, sentences_total,
uncovered_sentences, the never-read list in order, and every claim as
(kind, verdict, why, text, detail) in order. A port that gets the verdict right for the wrong
reason, or reads one sentence more or less, is a disagreement.

    cd web/gate/differential             # on a checkout whose styxx/diffgate.py is the file named above
    python build_corpus.py               # 176 real pairs, pinned to shas (below)
    python fuzz_corpus.py                # 3,000 synthetic pairs, seeded
    python py_side.py                    # refuses to run unless styxx/diffgate.py hashes to 4cded2e3…
    node js_side.js
    python differential.py
    node check_pairs.js                  # the 190 pinned pairs against their expect blocks (+ path2a_moves.json)

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
ignores generated JSON and then named these five as exceptions; it names eight now, the seven pinned-pair files
`check_pairs.js` reads and `path2a_moves.json`): `bc1_pairs.json`, the four pairs BC-2
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
`=== PATH-2a abstain-only overlay: BEGIN ===` and `... END ===`. The record of how it got here is twelve notes under
`papers/closed-model-frontier/`, one per review pass, each committed before its code: `NOTE_path2a_abstain_overlay_2026_09_30.md`,
then `NOTE_path2a_second_pass_…` to `NOTE_path2a_seventh_pass_2026_09_30.md`, `NOTE_path2a_eighth_pass_2026_10_01.md`,
`NOTE_path2a_ninth_pass_2026_10_04.md`, and `NOTE_path2a_tenth_pass_2026_10_05.md` to
`NOTE_path2a_twelfth_pass_2026_10_05.md`, with the corrections and departures notes beside the second, fifth, sixth,
and eighth to twelfth. This section describes the block as it is now and gives figures measured at this
head; what earlier heads measured is in those notes.

`main`'s reader runs unchanged — at the raw door, at the git door and in this port — and then the block reads each
decided claim once more. Where the #97 mechanism (the earliest entry in diff order matching by exact path, suffix or
base name), the #121 mechanism (`_norm`'s `lstrip("./")`, which gives `.x` and `x` one key) or the #101 mechanism (a
changed `def` counted as added) can have made `main`'s verdict wrong, the claim becomes UNCHECKABLE with the reason

    {V} withheld by PATH-2a ({defect}): {phrase}. main's reading: {main's reason, verbatim}

and the gate verdict is recomputed with `main`'s own formula. Nothing else in the record moves: the
claim list, every other verdict and every other reason are `main`'s, byte for byte, and `--strict`
changes the gate verdict only. Cut the block out and revert the door hooks (two lines per file) and
you have `main`'s two files back, sha for sha; `tests/test_diffgate_path2a.py` does exactly that and
uses the result as its reference.

**Why nothing else can move: DECIDE and APPLY.** Since the tenth pass the block is two parts, in both ports. DECIDE
(`_p2a_decisions`; `_p2aDecisions` here) holds every rule and every reader. It is given a copy made for it — one
object per claim carrying the claim's kind, verdict and reason and, of its detail, the strings and booleans of the six
fields a rule reads (`path`, `n`, `name`, `prefix`, `prefix2`, `declared`); not the claim's text; in Python instances
of two classes of the block's own with `__slots__` and no method (since the eleventh pass: before, each copy was a
`main` `DiffClaim`), here plain objects — and it returns plain data, a list of (claim index, phrase key, defect tag).
APPLY (`_p2a_apply`, `_p2aApply`; fifty to sixty lines) is the only code of the block that touches the record. It takes
a decision only if the result is a list (in Python exactly a list, and the decision exactly a tuple of an `int` and two
`str`s, so that no method of a subclass runs in APPLY), the index is that of a claim in reach, the phrase is a key of
the fixed table below other than APPLY's own `error` and `malformed`, and the tag is one of that claim's kind's; for a
decision it takes it writes exactly two fields, the reason in the form above and the verdict; a second decision for
one claim is ignored, as is everything else DECIDE returned; then it recomputes the gate verdict (in Python that reads
`strict` as a truth value where `main` short-circuited on the only CONTRADICTED, now withheld, and never read it: a
`strict` whose truth value raises, outside the `bool` the signature names, then raises at the door where `main` returned
FAIL, on 59 of #161's 497 reproductions by the eleventh construction reviewer's count; the port's door passes
`!!strict`). If DECIDE raises an
`Exception` (here: throws anything), every claim in reach is withheld with the phrase `error`; if it returns something
that is not a list, with `malformed`; in Python a `BaseException` that is not an `Exception` (`KeyboardInterrupt`,
`SystemExit`, `GeneratorExit`) goes up from the door with nothing written, where this port would withhold. So the
record leaves APPLY as `main`'s but for abstentions in reach, whatever data DECIDE returns, **for a DECIDE that uses
what it is handed as data and calls the `facts` function it is given** — the block's own, or one edited by mistake.
Code that reaches around that by reflection is not covered; *What holds bar A, and the lints* below says what that
is.

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

**The GitHub Action runs the styxx beside it, not PyPI's.** `action.yml` installs styxx from PyPI and then runs
`python "${{ github.action_path }}/diffgate_action.py"`; Python puts the script's own directory at the head of `sys.path`,
so `from styxx.diffgate import gate_diff_text` imports the `styxx` package of the checkout the action runs from, at
the ref the workflow names. This repository's own `diffgate` job (`uses: ./`) therefore runs the overlay on this
branch, and a workflow using `fathom-lab/styxx@main` runs it once this merges, before any release. (`main`'s
`.github/workflows/diffgate.yml` says in a comment that the Action runs PyPI's package; that file is `main`'s and
this branch does not touch `.github/`. Whether the Action should import the released package instead is the
operator's call.) Its job summary shows the overlay's own words whole — only for an UNCHECKABLE claim of a kind the
overlay may move whose reason starts with the overlay's form — and cuts `main`'s reading after them at 100 characters,
as `main` cuts every reason, so a long path cannot carry the table past GitHub's step-summary limit.

### The rules

A path claim (VERIFIED only; the path accusation is withheld on `main`) is kept only when
three readers without the mechanism verify it too: V97 (exact, then suffix, then base name — base
name only for a bare claim), V121 (keys keep their leading dots) and both. A file count abstains when
two changed paths differ only by a leading dot and the dot-kept count could read otherwise (a
CONTRADICTED count only when the dot-kept count range contains the claimed number; a VERIFIED count
unless that range is exactly it). Where the summary holds a *count seam*, every count read from the summary over a
dot twin abstains in both ports (`seam`), since there one port can read a count, with a clean number, that the other
does not read at all: a white space only one port's `\s` reads (U+001C to U+001F and U+0085 in CPython, U+FEFF in the
port) inside a run of either port's white space between a character that can end a number or a word of the count (an
ASCII digit, `e` or `s` in either case, U+017F) and one that can begin one (`c`, `f` or `w`); or `file` spelled with a
letter only CPython's IGNORECASE folds to ASCII (U+0130, U+0131, U+017F; U+212A folds to `k`, which no word of the
count holds). `only_touches` abstains when V121 reads "every changed path lies under the prefix set it uses" otherwise
than `main` reads it under the set `main` uses, and, with a second prefix claimed, when the leading prefix is a path
for `main` only because its leading dots were dropped ("Only touches github and .github/workflows/.": V121 says the
prefix is not a path; `shape`). `tests_added` abstains when a counted test is also defined in the removed lines and the
claim lies in `[got − changed, got]`, read for each port's `main` whose count gives the claim the verdict it has (each
port's count is read exactly from the bytes; `tests`, or `split` where only one port's reading holds it), or, with the
tests an unchanged line of the diff defines paired too, the claim lies in that wider interval (`redefined`: a test
defined again beside its own unchanged definition is counted by `main` and is not an added test); `symbol_added`
(VERIFIED) when a removed line defines the name where V101 reads a definition (at the line start, after white space
and an optional `async`; `symbol`), or an unchanged line does (`again`). Unchanged lines are read as removed ones are,
in both line views and in the pieces no view reads, except that a definition there whose name reads through NFKC is
passed over, so that a context line of prose cannot withhold every tests and symbol claim. The removed side is read
three ways: the two line views (the diff split at CPython's line breaks, and at the port's), and the removed text
neither reads as a line of its own, that is, a piece after a lone CR inside a removed git line (CPython's tokenizer
ends a line there, as at LF and CRLF, and nowhere else) and base-side lines joined by a backslash continuation. A name
CPython reads through NFKC (a code point from 0x80 up in the gap after `def` or `class` or where the ASCII name run
ends) may be any name, so it pairs with every name: such a removed test with every counted one, such a counted test
with every removed one, and such a removed definition defines every claimed name. The claimed number is read from the
claim's detail, not from `main`'s reason, since the port's `main` prints a number of 10^21 or more in exponent form,
and digits are read from a fixed table of the ten ASCII digits, never through `int()`.

**A CONTRADICTED in reach is decided by its kind's rule, exactly as a VERIFIED is.** From the sixth pass to the
eighth it was not: a switch (C-1, `apart`) kept every CONTRADICTED wherever `main`'s two ports might read the claims
apart, so that the two ports' gate verdicts agreed wherever `main`'s did. Wherever it fired it kept an accusation the
overlay knew the mechanism could have made false — a truth world kept 139 attributable false CONTRADICTEDs in each port
once a byte-order mark stood before the summary, and all 214 once a sentence named styxx beside a line separator — and
the blockers that the reviews of passes six, seven and eight found came from it (the case tables of two Unicode
versions, a `def` that ends its line, a regex backtrack in the scope window). The ninth pass removed it by the lead's
decision of 2026-10-04, which the operator confirms at merge. That decision's text, and the ninth note after it, date
the switch from pass five and lay the blockers of four passes on it. The notes show otherwise, and the tenth note
records the correction: pass five's gate-agreement rule was the count seam, which caused the blocker the review of
pass five found (a tests claim standing where one of the seam's two counts stood), and the switch entered at the
sixth pass to clear that. The cross-port bar is restated under *Cross-port* below, and what the removal costs is
measured there.

**The count seam is still here, and it only withholds.** It is the phrase `seam` of the rules above: over a dot twin,
a count read from the summary is withheld in both ports where the summary holds a seam. It never keeps a verdict, so it
is not a switch of the kind that was removed. Under the restated bar it carries neither C(i) nor C(ii): those hold
with or without the summary-side guards (`seam`, and `extract` for an ASCII path, name, number or prefix in odd
company), as a ninth-pass reviewer showed by dropping them from both ports at once. (The ninth note says each "still
carries C(i)", from dropping a rule in one port only; that shows that the two ports must carry the same rules, and no
more.) They are kept for what they do to the gate differences of C(iii), in both directions, and `extract` also for
coverage on paths outside ASCII. Keeping or dropping the summary-side part is operator option O-16, not taken; the
reviewer's figures with it dropped from both ports, on the committed inputs: claims withheld 2,627 → 2,319 in Python
and 2,587 → 2,239 in the port, gates apart only under the overlay without `--strict` 6 → 11, bar B not measured.
`divergent` is not of that kind: each port builds its file list from its own line split.

**A count beside a case pair outside ASCII is withheld in both ports** (`case_count`). `main` keys each path
through its runtime's lowercase, and the runtimes' case tables differ by Unicode version (CPython 3.9 to 3.12 carry
Unicode 13 to 15, CPython 3.14 and Node 24 Unicode 16, and a newer Node may carry Unicode 17), so two paths that
differ only in case outside ASCII may be one key in one port and two in the other. The overlay reads which code points
some runtime may merge from two static tables. The 1,432 lowercase mappings of Unicode 16.0 (U+0130 and U+212A aside,
which the fold form already reads), written as 184 runs, give each cased code point its class (its lowercase). The
blocks of scripts and symbols with no case — Hebrew through Myanmar, Hangul Jamo and Ethiopic, the Canadian syllabics
through Ol Chiki, the CJK radicals, symbols, kana, Bopomofo, Hangul compatibility and enclosed blocks, the CJK
ideographs and their extensions, Yi, Lisu, Vai, Bamum, the Brahmic and Southeast Asian blocks from U+A800 to U+AB2F,
Meetei Mayek, the Hangul syllables, the Hebrew and Arabic presentation forms, the half-width kana and Hangul, and the
neutral set — hold code points no runtime's lowercase maps or maps to. Any other code point (one a later version may
give a case, as Unicode 16 gave U+A7DC the lowercase U+019B) may pair with any code point outside those blocks. Over
the forms `main` keys, at each position where forms of one ASCII shape differ, a code point with no case is read as
itself and the others as their class, or all as one placeholder where some form there holds a code point of neither
table: #CA counts the keys that leaves, a lower bound of every runtime's count, as #A (ASCII case folded only) is the
upper one. Where #CA < #A, a count claim whose number lies in [#CA, #A] is withheld in both ports, whatever its
verdict, since the two `main`s may decide it apart; a number outside the range is CONTRADICTED by every `main`, and the
overlay decides it by the dot-twin rule. Two premises, stated: Unicode's case-pair stability (a case pair, once
assigned, stays one; two assigned code points that are not one never become one), and that the blocks listed gain no
case; the tests pin both tables against the lowercase of every runtime they run (exactly on Unicode 16, a subset on
15).

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
was punctuation, a symbol or a space there too — all pinned by enumeration on every runtime the tests run. In the
summary, the emoji of five pictograph blocks (U+1F300 to U+1F64F, U+1F680 to U+1F6FF, U+1F900 to U+1F9FF, U+1FA70 to
U+1FAFF; operator option O-11, taken in the seventh pass) are read as U+2190, a neutral code point, whether the string holds one code point or
the two surrogates the port's string holds: none of them is a word character or white space for either port's
templates, folds to an ASCII letter or has a case, and the tests pin that and the two regexes that read them by
enumeration too. Declared (DECLARE-1) claims skip the summary test: their sentence is written from a value both ports
read only when it is ASCII. Where a file header of the diff holds a character the two ports split or strip apart, every
path, count and scope claim abstains in both (`divergent`).

Where the overlay cannot compute its readers exactly it abstains and says so:

| key | phrase |
|---|---|
| dir | the claim's path has a directory part, and only a changed file with the same base name in another directory matches it |
| tier | a changed path that matches the claim more closely than the one main resolved it to reads otherwise |
| dot | with leading dots kept, the changed path the claim resolves to reads otherwise |
| dot_earliest | with leading dots kept, the earliest changed path matching the claim reads otherwise, though the closest one does not |
| dot_tier | with leading dots kept and the closest match taken, the claim reads otherwise |
| count | two changed paths differ only by a leading dot, which the path key drops, and counted apart the claim reads otherwise |
| seam | the summary holds a character that only one of the Python and JavaScript readers reads as a space, or as a letter, where a count is read, so the two may read different counts |
| only | with leading dots kept, whether every changed path lies under the prefix reads otherwise |
| shape | with leading dots kept, no changed path has the prefix as a segment, so it is not read as a path |
| tests | a test the added lines count is also defined in the removed lines, and a changed test is not an added one |
| split | a test the added lines count is also defined in the removed lines, and the Python and JavaScript readers split or space these lines differently |
| redefined | a test the added lines count is also defined in an unchanged line of the diff, and a test defined again is not an added one |
| symbol | the removed lines define this name too, and a changed definition is not an added one |
| again | an unchanged line of the diff defines this name too, and a name defined again is not an added one |
| extract | the summary holds a character the Python and JavaScript readers may read differently where the claim's path, name, prefix or number is read, so the two may extract the claim differently |
| divergent | a file header of this diff holds a character that the Python and JavaScript readers split or strip differently |
| odd | a path here has a drive-like prefix or a final '.' segment, where base names are read differently |
| case | a path here compares only where case outside ASCII is folded, which this overlay does not do |
| case_count | two changed paths differ only in case outside ASCII, which the Python and JavaScript readers' case tables may merge or keep apart, so the two may count the files differently |
| unreproduced | this overlay does not reproduce main's reading of the diff |
| unparsed | main's reason does not have the form this overlay reads |
| error | this overlay failed while reading the diff |
| malformed | this overlay's decisions did not come back as a list, so none of them was applied |

Running it, from the repository root:

    python -m pytest tests/test_diffgate_path2a.py tests/test_diffgate_path2a_truth.py tests/test_port_is_current.py
    cd web/gate/differential
    python path2a_recall.py --corpora DIR [EXTRA.json ...] [--truth] [--path-flavour posix|windows]
    node check_pairs.js                                   # the pinned pairs, path2a_pairs.json + path2a_moves.json

The port half of the PATH-2a tests, and `test_port_is_current`'s check of the port against the pinned pairs, need
`node`; without it they skip locally and fail under `CI` or `GITHUB_ACTIONS`. `path2a_recall.py` prints three totals:
`main`'s committed corpora, the overlay's own pins, and `main`'s corpora with any extra files.
`path2a_pairs.json` pins 136 pairs, each for the decision it names (19 of them pin kind and verdict only, where
`main`'s two ports print different reasons); it is not one of `py_side.py`'s corpora (some of its pairs are inputs
`main`'s own two ports read differently), so the port differential over `main`'s corpora reads 0 disagreements.
`path2a_moves.json` records the one pinned claim of `main`'s own files the overlay moves —
`path1:unrepaired-typo` claim 0, ".githiub/workflows/dependabot.yml" resolved by base name to
`.github/workflows/dependabot.yml`, a false VERIFIED — so `path1_pairs.json` stays `main`'s record,
byte for byte. Both files and the three fixtures under `tests/fixtures` are written one record per line.

### Measured at this head

On the file PATH-2a merged (`styxx/diffgate.py` `4cded2e3…`, reader `9b620e00…`; the figures marked *pinned* are re-asserted
on every later file by the committed tests, and the measured ones were not re-run when the demo's label changed), on CPython 3.12.10 (Unicode 15.0) and Node 24.13.0
(Unicode 16) unless a line says otherwise. A figure marked *pinned* is asserted exactly by a committed test. The
rest were measured once, with scripts kept in the builder's scratch, and are not asserted: by the builder of the
twelfth pass at this head where a line says *measured at this head*, by the builder of the eleventh pass at its head,
`ee82d2f3`, where a line says *measured at `ee82d2f3`*, otherwise by the builder of the tenth pass at its head,
`16daa725`. Those carry over: the eleventh and twelfth passes changed what DECIDE is handed and how APPLY reads, and no
decision. The records at this head equal `ee82d2f3`'s on every input read again (the committed inputs in both ports
and both strict modes, and the adversarial sets named below), as `ee82d2f3`'s equalled `16daa725`'s there and on the
20,400-input decorated world of *Coverage*, in both ports. The tenth pass changed no decision against the ninth
(`c69b161b`). Figures credited to a
*probe* were measured before the ninth pass's code, on its parent with the switch's call site disabled, and not
again; figures credited to a *reviewer* were measured by the reviewer named, at the head named, and not again.

**(A) By construction.** Each record is `main`'s but for abstentions in reach with the overlay's reason, each claim
reads the same with `--strict` as without it, and where `main` raises the branch raises the same exception.
- The 6,672 committed inputs (the pinned pairs, `main`'s 3,000 fuzz pairs regenerated in memory, #161's reproductions
  and pair inputs, a seeded 2,000-pair PATH-2a fuzz and a seeded 1,000-input text-seam set), both strict modes, in
  Python and in the port: 0 records outside the relation (*pinned*; in the port 13,344 runs, 6 of which raise in
  `main` and raise alike). Of 12,718 claims `main`'s Python decides there the overlay withholds 2,633, and of 12,526
  its port decides, 2,590; the per-phrase counts are *pinned* for both path flavours and for Unicode 13.0 to 16.0
  (13.0 and 14.0 by re-reading the inputs with their five Unicode 14 letters mapped to code points no version
  assigns, 15.1 because the inputs hold no code point assigned in 15.0 or 15.1; 15.0 and 16.0 were run).
- The same with what only the Python doors take, a run leg that exits 0, a green test report, and a report under a
  commit it does not name, over #161's 482 reproductions with a sentence `main` reads as a `tests_pass` claim
  appended, at the raw door and (the 96 that carry their own `--name-status`) at the git door: 0 outside the relation
  (*pinned*). `tests_pass` is outside the overlay's reach, and APPLY takes no decision for a claim outside reach: the
  eighth integration review's plant, set where the decisions are computed, asks for an abstention on every
  `tests_pass` VERIFIED (1,866 times over that run) and the record does not move (*pinned*).
- APPLY against DECIDE functions written to break it, 32 in Python and 31 in the port beside the block's own (the
  tenth pass's text said 26 and 24, counting the block's own in each): ones that change, empty and grow their copy;
  return nothing, a number, a string, a mapping, their own argument or a list of junk; return indices out of range,
  negative, fractional, boolean or as strings; return one index twice with different phrases; return phrases and tags
  outside the fixed sets, another kind's tag, or only APPLY's own `error` and `malformed`; return decisions for every
  claim outside reach, or for every index; raise; in Python, a list, tuple, int or str subclass (one that raises or
  lies when read, a key whose hash raises, an index that claims to equal every index, the block's own decisions in
  subclassed containers), and four that patch the class of their copy, reached by `type()` (a property on `verdict`, a
  `__setattr__`, a property that swallows the reason, a `__setattr__` on the container's class); in the port a decision
  whose field throws when read, a Proxy that answers differently on each read, and five that patch, in the realm the
  record was built in, what every object or array inherits (a phrase on the one, `includes`, `some`, `push` and
  `Array.isArray` on the other), and, since the twelfth pass, two that reach APPLY's own work before DECIDE runs (a
  `push` left patched to drop a lone `null`, and a key a page put on what every object inherits before the port
  loaded). Each over #161's reproductions, every other one with a sentence `main` reads as a `tests_pass` claim, in
  both strict modes (958 runs each in Python, on records of the module's own classes, and 962 in the port; measured,
  the tests assert more than 900): 0 records outside the relation; the record is `main`'s, untouched, wherever APPLY
  should ignore everything; every claim in reach is withheld with `error` where DECIDE raises (an `Exception`, in
  Python) and with `malformed` where it returns no list (*pinned*). The four class patches again through both Python
  doors, with the module's DECIDE replaced, both strict modes (1,150 runs each; measured, the test asserts more than
  1,100): 0 outside the relation, and with the class left patched the next call gives `main`'s record where nothing is
  in reach (*pinned*). The stated limit, through the same doors: a DECIDE that takes `main`'s `DiffClaim` from the
  door's `facts.__globals__` and patches it puts 1,010 of the 1,150 runs outside the relation (measured), and a
  committed test asserts that more than 100 do (*What holds bar A* below). Against `16daa725`'s files, the three class
  patches of the claims put 945 or 946 of 958 Python runs outside the relation, and four of the port's five patches
  200 to 604 of 962 runs; the fifth, and the `error` and `malformed` cases, there fail what they assert of the record.
  Against `ee82d2f3`'s port, the twelfth pass's two cases each put 604 of 962 runs outside the relation (*measured at
  this head*).
- The ninth construction review's fourteen plants (five in Python, nine in the port), each of which dropped, added or
  rewrote claims of `main`'s record at `c69b161b` while every static check passed, set where the decisions are
  computed: 0 records outside the relation on the reproductions (*pinned*; the same test, run on `c69b161b`'s two
  files, fails on all fourteen with the claim count or a claim moved).
- APPLY takes every decision the block's own DECIDE returns: 2,633 of 2,633 on the committed inputs in Python, 482 of
  482 on the reproductions in the port (*pinned*). An ignored decision would leave `main`'s verdict standing in
  silence, so this is asserted and not assumed.
- The git door: the reproductions with their own `--name-status`, and thirteen small repositories the test builds
  with git (dot twins, a rename, a copy, a type change, a `diff.noprefix` diff, a path holding U+2028), each against
  its own `main` (*pinned*).
- 160,000 adversarial inputs from the earlier passes' and reviews' generators (two hostile sets of 20,000, a mixed
  seam set and a count-seam set of 20,000 each, three further hostile sets of 12,000, 20,000 and 9,000, a case-pair
  set of 10,000, a line-break set of 4,000, case-skew, emoji and DECLARE-1 sets of 9,000, and a truth world under ten
  decorations, 16,000): 319,986 Python runs and 320,000 port runs in both strict modes, 0 outside the relation, every
  raise `main`'s (7 inputs in Python, 18 in the port), no fallback phrase. 19,000 of them were read again with the
  port's engine patched to fold a pair no runtime folds yet, for bar C below. All of it was read again at `16daa725`,
  beside `c69b161b`'s two files: every count the same, and no record differing from the ninth pass's in either port.
  Six of those sets were read again at `ee82d2f3` beside `16daa725`'s two files, and at this head beside
  `ee82d2f3`'s (62,000 inputs: the 20,000 of one hostile set, 12,000 and 9,000 of two more, the 4,000 of the
  line-break set, 8,000 of the decorated truth world, and the 9,000 of the case-skew, emoji and DECLARE-1 sets under
  the patched engine): each time no record differs from the earlier head's in either port, and 0 of 124,000 Python
  runs and 0 of 106,000 port runs lie outside the relation (*measured at this head*).
- The EXTERNAL-1 shelf of agent pull requests (the AIDev dataset's `pr_commit_details`: one row per file per commit),
  one body per pull request id against the diff the committed harness's own `reconstruct` rebuilds from all of that
  id's rows (one group per file, with the net status of `_fold_statuses`): 69,054 read (of 71,104 ids with a body
  that is not blank, 2,003 have no file row with a name and 47 are skipped, their rebuilt diff over 3 MB), 138,108
  Python runs in both strict modes: 0 outside the relation, no raise. `main`'s parser reads 69,053 of the rebuilt
  diffs back as the folded listing, the harness's own check of its reconstruction.
- The minified bookmarklet, loaded in a stub page, equals this port on 14,308 runs (the 6,672 committed inputs and
  #161's 482 reproductions with the `tests_pass` sentence on every other one, both strict modes; 3,350 carry an
  overlay reason; *measured at this head*, on this head's build) and on every pinned pair (*pinned*). The tenth pass
  measured its own build the same way on 20,000 runs.

**(B) Coverage**, judged by truth from base/head file models (`tests/test_diffgate_path2a_truth.py`). A decided claim
is attributable when `main`'s verdict is false and a variant of `main` without the mechanism (V97, V121, V101, or all
three) does not read it false: the but-for reading. Every figure in this table is *pinned*.

| world | cases | main decides | false | attributable | withheld | right verdicts lost | undecided withheld |
|---|---|---|---|---|---|---|---|
| raw door (Python) | 1,706 | 7,614 | 1,535 | 1,206 | **1,206** | 253 of 5,897 | 39 of 182 |
| the port, in its own terms (variants built from `main`'s port) | 1,706 | 7,393 | 1,385 | 1,100 | **1,100** | 177 of 5,739 | 126 of 269 |
| git door (Python), the reproductions with their own `--name-status` | 96 | 63 | 17 | 14 | **14** | 0 of 46 | — |

The cases are #161's 101 reproductions that carry a file model, the 5 PREREG_path2 reproductions and 1,600 generated
cases (400 per family: #97, #121, #121 with case outside ASCII, #101). In the port, 942 of the 1,100 attributable
claims carry the same `main` record as in Python; the other 158 are paths and names outside ASCII, which the two
templates extract apart. Further truth tests, each *pinned*:
- the 1,600 generated cases under four transforms, in Python and in the port, every attributable claim withheld:
  two changed CJK docs and a right count sentence (1,073 and 969 attributable); "(naïve)" inside each tests or count
  claim's own sentence (1,193 and 1,087); a U+FEFF before the summary (1,175 and 1,087); a sentence naming styxx beside
  U+2028 (1,193 and 1,087). Right verdicts lost under the last three: 242 of 5,842 or of 5,827 in Python and 166 of
  5,684 in the port, the plain world's figures. The parent head kept 139 attributable verdicts in each port under the
  third and 214 under the fourth (a probe's count);
- the fourth review's three `shape` reproductions (a GNU deletion, a mnemonic header, a rename into `.github/`, the
  last at both doors), the fifth review's definition reproductions (a test or function CPython reads across a
  backslash continuation, after a lone CR, or through NFKC: 33 claims at the raw door, the git door and in the port)
  and the sixth review's (a test defined again beside its own unchanged definition, a name defined again, a moved
  file's old path: 15 claims at three doors). The definition reproductions are attributed by an **ast-paired V101**
  (`main`'s own count less the test functions the base and the head of the same changed file both define, read by
  CPython's parser under NFKC), since the committed V101 reads the removed lines one by one, as `main` does;
- three of #161's reproductions on which the port's `main` alone gives a false tests verdict through #101 (it counts
  the `def test_` after a U+FEFF, which CPython's `\s` is not, beside a changed test), with file models written by
  hand: 4 attributable verdicts in the port, 4 withheld, 1 of 3 right verdicts lost. Up to the eighth pass the switch
  kept these, and no test read the port on them; the port's decisions on seven such ids are *pinned* too.

Not misses under the but-for reading, and kept: #161's five joint #121 reproductions
(`m-121-a-submodule-line-licenses-nothing`, `k4-a-dotted-twin-beside-a-submodule-line`,
`k4-a-dotted-twin-beside-an-hg-binary-notice`, `l-r10-np1-…`, `l-r10-np2-…`) keep `main`'s false CONTRADICTED count:
V121's count is false there too (a submodule line, an hg binary notice or a no-prefix directory `main` registers no
file for). They are *pinned*, both ports. Operator option O-13 (a dot twin, `main`'s count below the claimed number,
and a change marker `main` registers no file for) would withhold them and is not taken; neither is O-7, its wider
form.

Per phrase and per `main`'s verdict, on the raw door's truth world (measured at `16daa725`): how often each fires, and
whether `main`'s verdict was false, right or undecided there. `case_count`'s false verdicts are `main`'s case merges,
outside the three defects.

| key | `main`'s verdict | fired | false (attributable) | right | undecided |
|---|---|---|---|---|---|
| count | CONTRADICTED | 243 | 208 (204) | 35 | 0 |
| count | VERIFIED | 162 | 162 (162) | 0 | 0 |
| dir | VERIFIED | 423 | 404 (404) | 18 | 1 |
| dot | VERIFIED | 183 | 177 (177) | 6 | 0 |
| dot_earliest | VERIFIED | 6 | 1 (1) | 5 | 0 |
| dot_tier | VERIFIED | 111 | 107 (107) | 4 | 0 |
| tier | VERIFIED | 17 | 12 (12) | 1 | 4 |
| case | VERIFIED | 2 | 2 (2) | 0 | 0 |
| case_count | CONTRADICTED | 17 | 13 (0) | 4 | 0 |
| case_count | VERIFIED | 13 | 13 (0) | 0 | 0 |
| only | VERIFIED | 11 | 9 (9) | 2 | 0 |
| shape | CONTRADICTED | 43 | 0 (0) | 43 | 0 |
| tests | CONTRADICTED | 66 | 20 (17) | 33 | 13 |
| tests | VERIFIED | 66 | 34 (34) | 20 | 12 |
| redefined | CONTRADICTED | 13 | 6 (0) | 6 | 1 |
| redefined | VERIFIED | 9 | 9 (0) | 0 | 0 |
| symbol | VERIFIED | 35 | 30 (30) | 0 | 5 |
| again | VERIFIED | 4 | 1 (0) | 0 | 3 |
| extract | VERIFIED | 167 | 91 (47) | 76 | 0 |

What the table shows, and what it cannot. `shape` fires 43 times there and every time on a right CONTRADICTED ("Only
touches cfg and .cfg/app/." beside files that all lie under the second prefix); `tests` withholds more right than
false verdicts on CONTRADICTED and the reverse on VERIFIED; `dot_earliest` catches one false verdict for five right
ones; `extract` loses 76 right verdicts (paths and names outside ASCII). The generated world's summaries are ASCII
and its diffs are well formed, so it says little about `seam`, `divergent`, `split`, `odd`, or `extract` on real
prose. Independent reviewers measured the phrases on worlds of their own at earlier heads (worlds built by real git,
a grammar fuzz, the EXTERNAL-1 shelf); those figures are in the notes of the passes that carry them and were not
measured again here, except the shelf's below. The known losses they found stand as described there: `tests` and
`redefined` pair names across the whole diff, blind to classes and files; `dir` withholds a right verdict that rests
on the #97 base-name fallback (mnemonic and no-prefix renderings, a moved file's old path under rename detection, a
dot-directory file in a monorepo); the NFKC rule reads removed prose in any file; a test or a name defined again is
not read where its earlier definition lies outside the hunk (next paragraph); in Chinese, Japanese or Korean prose
written without spaces one mention of a path joined to the text withholds every claim on that path. Operator options
O-6, O-8, O-10, O-12, O-14 and O-15 would each trade some of these, and none is taken.

**The limit of `redefined` and `again`.** A test defined again is invisible to the overlay whenever its earlier
definition lies outside the hunk's context lines, at any context width; `-U0`, which earlier text named, is only the
extreme. The fact is not in the diff. In the truth world above, at git's default three lines of context, the head
keeps 8 false tests verdicts of this kind, 4 of them CONTRADICTED (false accusations) and 4 VERIFIED, and 6 more at
context 0 and 1 (2 CONTRADICTED, 4 VERIFIED), against 12 it withholds as `redefined` (a ninth-pass reviewer's
count, by CPython's `ast` over the base and head files; measured again at `16daa725`). They are not misses under the
bar's attribution, since the committed V101 pairs by removed lines and reads them false too; under the ast-paired
V101 of the definition reproductions they would be. Withholding every tests claim on a modified file would close the
gap at the cost of the recall the rule exists to keep; that trade is not taken.

**Under decorated summaries** the losses are larger than the plain table shows. The same four families at another
seed (300 per family), each summary under seventeen decorations — a byte-order mark; a sentence naming styxx beside a
line separator; accents before and inside a claim's sentence; emoji in and outside O-11's ranges before a claim and
right after its last token; CJK text; a long lead-in; the whole summary as one sentence; markdown; U+0085, U+2028 or
a form feed as sentence breaks; zero-width and bidi characters; quotes; back slashes; a mix — 20,400 inputs (a
ninth-pass reviewer's world; measured again at `16daa725`, and at `ee82d2f3` its records equal that head's in both
ports). Every attributable false verdict is withheld in both
ports, 12,559 of 12,559 in Python and 11,521 of 11,521 in the port, at 4,358 of 61,777 right verdicts lost in Python
(7.1%) and 3,897 of 61,942 in the port (6.3%). By decoration the right verdicts lost in Python run from 3.0% (a long
lead-in) and 3.4% (plain) to 10.5% (an accent inside the sentence), 11.2% (an emoji right after the last token: flags,
the coloured circles, keycaps and ZWJ sequences lie outside O-11's four ranges), 11.8% (one joined sentence), 15.8%
(separators as sentence breaks) and 21.7% (zero-width or bidi characters); in the port from 2.2% to 21.2%. Per phrase
and per `main`'s verdict, what was withheld there (Python; the port in brackets):

| key | `main`'s verdict | false | right | undecided |
|---|---|---|---|---|
| count | CONTRADICTED | 2,048 (2,048) | 338 (338) | 0 (0) |
| count | VERIFIED | 1,579 (1,579) | 0 (0) | 0 (0) |
| dir | VERIFIED | 3,877 (3,989) | 149 (154) | 0 (0) |
| dot | VERIFIED | 2,233 (1,509) | 64 (48) | 0 (0) |
| dot_earliest | VERIFIED | 0 (0) | 45 (47) | 0 (0) |
| dot_tier | VERIFIED | 852 (873) | 0 (0) | 0 (0) |
| tier | VERIFIED | 149 (153) | 0 (0) | 70 (75) |
| case | VERIFIED | 16 (16) | 42 (45) | 0 (0) |
| case_count | CONTRADICTED | 110 (113) | 27 (28) | 0 (0) |
| case_count | VERIFIED | 152 (156) | 0 (0) | 0 (0) |
| only | VERIFIED | 86 (90) | 0 (0) | 0 (0) |
| shape | CONTRADICTED | 0 (0) | 288 (306) | 0 (0) |
| shape | VERIFIED | 9 (10) | 0 (0) | 0 (0) |
| tests | CONTRADICTED | 203 (207) | 264 (272) | 45 (45) |
| tests | VERIFIED | 252 (258) | 145 (152) | 138 (143) |
| redefined | CONTRADICTED | 43 (44) | 26 (28) | 14 (15) |
| redefined | VERIFIED | 77 (79) | 27 (28) | 14 (14) |
| symbol | VERIFIED | 420 (430) | 0 (0) | 30 (30) |
| again | VERIFIED | 0 (0) | 0 (0) | 28 (30) |
| extract | CONTRADICTED | 65 (150) | 2,101 (2,218) | 0 (0) |
| extract | VERIFIED | 1,300 (349) | 842 (233) | 60 (866) |

Five cells of this table withhold more right verdicts than false ones: `extract` on CONTRADICTED, by far the largest
once summaries are decorated; `shape` on CONTRADICTED; `tests` on CONTRADICTED; `dot_earliest`; and `case`. The
`extract` cell is mostly one loss, older than the ninth pass: a scope claim that shares its sentence with an accented letter, a CJK
character or an emoji outside O-11's ranges after the word `only` is withheld in both ports whatever its truth.
"Only touches docs/ — thanks José!" over changes to `src/app.py` and `docs/guide.md` is CONTRADICTED by `main`, rightly,
in both ports, and is withheld (`extract`) in both; "Only touches docs/. Thanks." stays CONTRADICTED. No rule changes
for these: `dot_earliest` and `shape` are what bar B requires as worded (some variant not reading the verdict false),
`extract` and `case` are where the two ports' templates or case tables may part, and the trades are the operator
options already listed.

`seam` has no row because it never fires on that world: 0 times in either port (*measured at `ee82d2f3`*), since its
decorations put no white space of one port alone between the end of one word of a count and the start of the next.
It does fire on worlds of the tenth coverage reviewer's own, and there two cells outside the five withhold more right
verdicts than false ones (the reviewer's figures, at `16daa725`): `seam` on CONTRADICTED, 89 right against 54 false in
Python and 50 against 32 in the port on the builder's four families under seventeen other decorations (seed 51005,
4,080 inputs), and 80 against 60, 40 against 20 and 36 against 27 at the git door on renderings by real git (seed
2027); and `redefined` on VERIFIED there, 39 right against 0 false. `seam` withholds a right CONTRADICTED count
whatever number is claimed, so it can turn a right FAIL into PASS: "7 files changed. Changes" then U+001F and "were
small." over `.env`, `env` and `src/a.py` is CONTRADICTED by `main`, rightly, and withheld (`seam`) in both ports. The `case_count` CONTRADICTED row of the table
(110 false against 27 right in Python, 113 against 28 in the port) is the other figure the operator weighs with these
under O-16, which would drop the summary-side guards from both ports; a narrower `seam`, only for a claimed number
between `main`'s count and the count with the twins apart (the range the count rule already reads), is listed beside
it. Neither is taken. Two more cells tip on the eleventh coverage reviewer's worlds (the reviewer's figures, at
`ee82d2f3`): `extract` on VERIFIED, 192 false against 204 right in Python on one decorated world (seed 11101), and in
the port 97 against 162 (plus 281 undecided) on another (seed 11102) and 85 against 109 (plus 247) on renderings by
real git (seed 11103); and `case_count` on CONTRADICTED, 27 against 32 in Python and 24 against 25 in the port (seed
11102).

**Where "attributable" is narrower than the definition: a kept gap** (operator option O-10). With a single
`only_touches` prefix that is a path for `main` only through its dropped dots, V121 abstains ("is not a path") and the
overlay keeps `main`'s verdict: a VERIFIED there is already withheld (`only`), and a kept CONTRADICTED is false when
every changed file lies under the prefix's directory, while `main`, reading a path whose leading directory the
rendering dropped, lists a path outside it. Any rendering that drops a leading directory `D` holding `.D/` opens the
gap, for any `D`: git's own `git diff --no-prefix` over a top directory `b/` (or `a/`), where `main` strips the real
`b/` as git's prefix; `git diff --relative=D` or `git format-patch --relative=D` at the raw door; and
`diff.relative=true` at the git door with `D` as the repository, where git prints the `--name-status` listing and the
diff relative to it. `main` keys `.D/conf.py` as `D/conf.py` (#121), reads `D` as a path and accuses "paths outside
'D'" where every changed file lies under `D/`; V121 does not read it false, so the accusation is attributable under the
committed definition, and it stands, gate FAIL in both strict modes. The truth tests pin the tenth coverage reviewer's
four `--no-prefix` repositories and the eleventh's four `--relative` ones (`docs`, `config`, `vscode`), built with git,
as kept attributable CONTRADICTEDs: 8 at the raw door, 8 in the port and 4 at the git door (*pinned*), so that a change
that closes or widens the gap moves the pin. By the lead's direction no rule changed in the eleventh or twelfth pass.
Neither rule offered in the eleventh pass closes `--relative` (a narrow `shape` when `main`'s key for the prefix is
`a` or `b`; a trigger on `diff --git` headers with no `a/` `b/` split): its prefix is any directory, and its headers
keep the split. The general rule does: withhold every single-prefix CONTRADICTED that V121 reads as not a path. Its
cost, the eleventh coverage reviewer's figures at `ee82d2f3`, each a right verdict withheld: 98 per port on the
committed truth world, and no false one (planted in both ports at this head: right verdicts lost 253 → 351 in Python
and 177 → 275 in the port, claims withheld +98 in each, no other count moved, and the single-prefix pins fail;
*measured at this head*); 155 in
Python and 145 in the port on one decorated world (seed 11101), 144
and 147 on another (seed 11102); 58, 58 and 19 at the git door on renderings by real git (seed 11103). Not taken; the
operator decides. The other known gaps (case-only merges with no dot, names `strip()` merges,
multi-commit renderings, `async def` tests `main` does not count, `def` in non-Python files, joint shapes where #121
is one of two causes, and #161's off-tree prefix family, where `main` reads "Only touches docs/.." as `docs` through
`rstrip("/.")`) are none of the three mechanisms or are not reachable by an abstain-only rule; the notes list them.

**(D) Recall**: of `main`'s decided claims, how many the overlay withholds (`path2a_recall.py`, measured). `main`
reads `Path(p).name`, and a drive-like name such as `c:x.py` reads otherwise under the Windows and the POSIX path
flavour, so the figures are given under both where they differ.

| corpus | sha256 (LF) | decided | withheld |
|---|---|---|---|
| `corpus_real.json` | `1b21418a…` | 41 | 0 |
| `corpus_fuzz.json` | `2e80cd1d…` | 2,144 | 79 — 51 touched, 22 created and 5 deleted `dir`, 1 created `tier`; all #97 |
| `main`'s six pinned files | | 46 | 1 (the move above) |
| **`main`'s committed corpora** | | **2,231** | **80 (3.6%)**, both flavours |
| `path2a_pairs.json` (the overlay's own pins, built to abstain) | `ad7a93db…` | 155 (Windows), 154 (POSIX) | 90 (Windows), 89 (POSIX) |
| #161's `path2_pairs.json` (branch `fix/diffgate-path-resolution`) | `7ba272c8…` | 530 | 200 |
| `main`'s corpora + #161's pairs | | 2,761 | 280 (10.1%) |

Every abstention on `main`'s corpora is on synthetic input: the only real-world #97 record there, `corpus_real` pr98
("integrations/git/README.md — created."), is left as `main` has it, since neither of its two records is a false
decided verdict. On the committed inputs as a whole (above) the overlay withholds 2,633 of 12,718 in Python; most of
those inputs are fuzz built from the characters the rules read.
On the EXTERNAL-1 shelf (69,054 agent pull requests, each read once by its id; measured at `16daa725`, the database
opened read-only): 7,836 hold a decided claim in reach, 12,785 such claims in Python, of which the overlay withholds
161 (1.3%): 149 path claims by `dir` or `tier`, 1 by `dot`, 4 by `extract`, and 7 tests claims (`redefined` 6, `tests`
1). `main`'s gate FAILs on 87 of those pull requests, and the overlay moves 3 gate verdicts. In the port, 161 of 12,768.
A figure that needs no truth: of the 12,785, the four committed variants of `main` (V97, V121, V101, all three) read
152 otherwise than `main` does, and the overlay withholds all 152; it keeps no claim that a variant reads otherwise, so
under bar B's but-for reading it keeps no attributable miss there, whatever the truth is.

Two things the ninth pass's text said about the shelf were wrong, and its figures are replaced by those above. It
counted 69,058 "bodies": its tally walked the file table in row order and took a change of id for a new pull request,
and nine ids lie in runs of rows that do not touch, so some pull requests were read twice, each time with part of
their files. And it said verdicts were "judged against GitHub's own file list": the table is the dataset's per-commit
rows (228,693 of 1,409,108 pull-request-and-file pairs have more than one row; 25,316 of the pull requests read here
have such a file and 11,702 one whose rows disagree in status), the tally took one row's status as the file's, and
`external1_harness.py` records (`_fold_statuses`, its correction of 2026-08-31) that this was a defect the lab had
already corrected once. The judged figures (146 false, 115 of them withheld; 6,793 right, 2 withheld) are withdrawn
and not replaced. The fold does not make such a judgement sound: it is order-free because the shelf carries no commit
order, so a file added and later removed nets to "modified" by rule, a merge commit's rows list files the pull request
does not change, and a commit the dataset lacks cannot be seen. The shelf holds the listing the diff is rebuilt from
and no pull-request file list to judge a verdict against, and a verdict that agrees with its own input is not thereby
right. The removed switch never fired on these bodies: a probe read the same abstentions with it and without it.

**(C) Cross-port.** The bar, as the ninth pass restates it, word for word:

> (i) wherever main's two ports give a claim the same kind, verdict and detail, the overlay gives it the same decision
> and the same phrase in both ports; (ii) on every input where main's two ports read the SAME claim list (same length,
> and the same kind, verdict and detail at each position), the two ports' final claim lists and gate verdicts are the
> same, in both strict modes; both hold because a decision reads only the claim's kind, verdict and detail, main's
> counts and the door bytes, through code that asks no runtime a Unicode question; (iii) where main's two lists differ,
> nothing is promised: the committed tests MEASURE how often that happens and how often the gates then differ under
> main and under the overlay, and the README states it.

Two final lists are "the same" when they have one length and, at each position, the same kind, verdict and detail
and, where the overlay wrote the reason, the same phrase and defect tag; a claim's text and `main`'s own reason are not
compared, since `main`'s two ports cut the text and print some reasons differently. (i) and (ii) hold by construction
up to four things: each port builds its file list from its own `main`'s line split, and where the two splits can part
on a header both ports withhold (`divergent`); the fallback phrases (`unreproduced` and `unparsed`, which read the
port's own `main`, and APPLY's `error` and `malformed`) are outside the claim, and the tests assert that neither port
writes one; the port's white space is a static table that must equal its engine's — on an engine whose `\s` or
`trim()` read one character more, the port would write `unreproduced` where the Python decides, which the committed
table test catches on any engine the tests run on and nothing catches in a browser running the bookmarklet, where a
fallback phrase in a record is the sign (the Python's `_P2A_PY_SPACE` and `_P2A_PY_BREAKS` bind it to its interpreter
the same way, pinned only where the tests run, so a deployed Action on a CPython whose `isspace` or `splitlines` read
otherwise is in the same case); and "the same code in both ports" is a transliteration held by tests — the
pinned pairs, the comparison itself, and plants that change one port alone — not by a proof. The ninth cross-port
review planted 33 one-edit slips in one port and 11 passed every committed behaviour test; each of those, and a
tag only one port names otherwise, is now told by a cross-port case named for it (*pinned*: planted, the port decides
that case otherwise). Asserted (*pinned*): (i) on the committed inputs, the 84 cross-port cases and three seeded
sets, 0 claims decided apart — in verdict, phrase or, since the tenth pass, defect tag — under the by-construction
keys (the same position; matched across the two lists; anywhere in either list); (ii) on the same sets, with a floor on how many inputs with equal lists hold a
withheld claim, so that it is not vacuous. Measured on the 160,000 adversarial inputs: 0 under (i), and 0 under (ii)
on the 81,917 whose lists are the same (19,478 of them with a withheld claim); with the port's engine patched, on the
19,000 read again, 0 and 0 as well. Two further keys are heuristics for one match
read two ways and are asserted on the committed inputs only: claims at the same position with the same (kind, verdict,
text), and left-over claims whose details nest and lie in each other's text; each also pairs two different matches
(L1 and OM1 are pinned cross-port cases whose split is expected; on the adversarial inputs the text key paired 4 such).

(iii), measured. The lists differ on two sides, which are counted apart. On the **description side** the two ports
read different claims from the description (another length, or a kind or a detail apart): that is issue #181, the
port's templates reading the description with JavaScript's regex classes where the Python's are Unicode-aware. On the
**diff side** both ports read the same claims and decide one apart on the diff: `str.splitlines()` against a split at
CR and LF only, each language's white space at `def` sites and headers, `^` after U+2028 and U+2029 in the port, and
the two runtimes' case tables (#173 and #184 for the last; the line-break and white-space part has no issue of its own
yet, which is for the lead or the operator to decide). "Only under the overlay" counts the inputs where `main`'s two
gate verdicts agree and the overlay's do not; `main`'s two gates differ on the rest without any overlay.

| set | inputs | lists the same (with a withheld claim) | lists differ: description + diff side | gates differ without `--strict`: `main` / overlay (only under the overlay) | with `--strict` |
|---|---|---|---|---|---|
| committed inputs, CPython 3.12, Windows path flavour (*pinned*) | 6,672 | 6,067 (1,127) | 480 + 120 | 19 / 23 (6) | 20 / 13 (2) |
| the same, POSIX path flavour (*pinned*) | 6,672 | 6,070 (1,127) | 480 + 117 | 19 / 23 (6) | 19 / 13 (2) |
| the same, CPython 3.14.2, Windows flavour (*pinned*) | 6,672 | 6,068 (1,128) | 482 + 117 | 20 / 24 (6) | 21 / 13 (2) |
| the 84 cross-port cases (each *pinned* by itself) | 84 | 49 (39) | 29 + 6 | 0 / 14 (14) | 0 / 4 (4) |
| a `def` that ends its line, 600 seeded inputs (*pinned*) | 600 | 451 (451) | 0 + 149 | 0 / 149 (149) | 0 / 0 |
| a case pair only the patched engine folds, 1,500 seeded inputs (*pinned*) | 1,500 | 669 (294) | 0 + 831 | 421 / 0 | 358 / 0 |
| a truth world under decorations, 1,000 seeded inputs (*pinned*) | 1,000 | 752 (355) | 248 + 0 | 0 / 0 | 0 / 0 |
| the 160,000 adversarial inputs | 160,000 | 81,917 (19,478) | 71,961 + 6,104 | 30,928 / 17,940 (2,628) | 27,302 / 25,550 (1,972) |
| the EXTERNAL-1 shelf, one body per pull request id against the diff rebuilt from its folded rows | 69,054 | 69,035 (108) | 19 + 0 | 0 / 0 | 4 / 4 (0) |

Five inputs of each of the committed rows raise in a `main` and are left out. The six committed inputs whose gates
part only under the overlay without `--strict` are #161's four `f2` separator inputs (a vertical tab, a form feed,
U+2028 or U+2029 before `def test_a`) and `y2`, where the port's `main` alone counts the test, its two false verdicts
are withheld (PASS) and the Python's right CONTRADICTED stands (FAIL); and `f2`'s context line, the other way round.
All six are on the diff side. Under `--strict` the two are text-seam inputs ("Modified lib/core.jsé": only the port's
`main` reads `lib/core.js`, which `extract` withholds there). On the adversarial inputs the overlay parts 2,628 pairs
of gates that `main` has together and joins 15,616 that `main` has apart (description side 71,961, diff side 6,104);
1,016 of the 2,628 are one generator, a `def` that ends its line, where every input on the diff side parts. Under
`--strict` a withheld claim fails the gate in the port that reads it, so a claim only one port's `main` reads can part
them whatever the overlay withholds; the bookmarklet calls the gate without `--strict`, and operator option O-9
(abstain in both ports wherever either port's reading fires) is not taken. What the removed switch bought
was the "only under the overlay" column at 0 without `--strict`; what it cost is in *Coverage* above and in the
pinned cases: of the 26 cross-port cases whose decisions moved when it went, the Python's 28 changed claims are 20 of
`main`'s false CONTRADICTEDs now withheld and 8 right ones lost (17 and 6 in the port; a probe's count), the plainest
of the lost being "Only touches docs/." then U+0085 and "Thanks." over a change to `src/app.py`.

On the decorated truth world of *Coverage* the exchange runs the other way, and the operator confirms the decision on
both numbers. Against the parent head `3bdc3bc4`, on those 20,400 inputs at the Python raw door (a ninth-pass
reviewer's measurement; measured again at `16daa725`): the removal withholds 903 more of `main`'s false CONTRADICTEDs
and loses 1,812 more right ones (right CONTRADICTEDs withheld 1,232 → 3,044 of 46,437; right verdicts of either kind
2,546 → 4,358 of 61,777, 4.1% → 7.1%), 1,397 of the 1,812 through `extract` on a scope claim. It also confirms the
decision's ground more widely than the two pinned decorations do: the parent head kept 796 of that world's 12,559
attributable false verdicts, under accents, emoji, CJK text, separators and zero-width characters as well as under
the byte-order mark and the styxx sentence, and this head keeps none. On real pull-request bodies the switch never
fired.

On a runtime without a pin, each figure of a *pinned* row must lie within a tenth, or within 10, of the pin nearest
to it: the pin for the running interpreter's Unicode version, and, for the patched-engine set, the row for whether the
two runtimes fold U+A7DC alike. At the ninth pass's head the comparison took whichever pin sorted ahead, and the
patched-engine set failed on an engine below Unicode 16 for no fault of the code. Exercised under emulation only:
an engine reporting Unicode 15.1 and one reporting 17, beside CPython 3.12.10 and 3.14.2. On all four pairs the three
seeded sets pass, and the patched-engine set's figures equal one of its two pinned rows. The eleventh cross-port
reviewer emulated a Unicode 13 CPython (CI's 3.9 and 3.10) against Node 24.13.0: every pinned bar C figure within the
committed tolerance of the nearest pin (on the committed inputs, Windows flavour, 6,065 equal lists, 1,125 of them
with a withheld claim, and 480 + 122 differing, against the pin's 6,067, 1,127 and 480 + 120), 0 C(i) splits and 0
C(ii) breaks (the reviewer's figures, at `ee82d2f3`).

### What holds bar A, and the lints

Bar A does not rest on a check of the block's source. It rests on APPLY, and on tests of what APPLY must withstand
(*By construction* above): DECIDE functions written to break it, handed to it in both ports and through both Python
doors, the ninth construction review's plants set where the decisions are computed, and a count of the decisions it
takes.

**What is claimed, and where it stops** (the lead's statement of the boundary, recorded in the eleventh note). Bar A
by construction is claimed for a DECIDE that uses what it is handed as data and calls the `facts` function it is
given, whatever data it returns. It is not claimed for code that reaches around that through reflection: a class
reached by `type()`, a function's `__globals__` or closure, a frame, `sys.modules` or `gc`, or a built-in patched
through what every object or array inherits. No guard inside one Python interpreter or one JavaScript realm can close
those, and DECIDE is the lab's own code, so the hostile tests guard against an honest mistake in it, not an attacker.
Within that boundary, what the eleventh pass changed:

- **In Python, nothing reachable from DECIDE's argument as data is `main`'s.** At `16daa725` each copy was a `main`
  `DiffClaim`, the class of every claim in the record, so a DECIDE that set a property or a `__setattr__` on
  `type(seen.claims[0])` ran its code in APPLY's own two stores on the record (CONTRADICTED became VERIFIED, FAIL became
  PASS, at both doors, and the patch stayed for later calls), and the container's `__init__` held the module in its
  `__globals__`. The copies are now of two classes of the block's own with `__slots__` and no method, built by
  attribute stores; a patch of them, through `type()`, reaches DECIDE's own copies and nothing else. Committed cases
  assert it at APPLY and at both doors.
- **In the port, after DECIDE returns APPLY calls no method an object or array inherits.** DECIDE reaches those from
  its argument, and at `16daa725`, in one realm (as in a page), a phrase put on what every object inherits, an
  `includes` or a `some` or a `push` put on what every array inherits rewrote the record or the gate verdict. APPLY now
  reads by index on arrays it or `main` built, compares with `===`, tests an array with the `Array.isArray` it took
  when the module loaded, finds a phrase key in a literal list of the keys DECIDE may name (since the twelfth pass;
  before, a list built at load by `for … in` over the phrase table, which also takes a key a page put on what every
  object inherits), and stores only into elements it filled, by index store and since the twelfth pass not by an
  inherited `push`, before DECIDE ran. Its cost was measured against `16daa725`'s APPLY (*Cost*): no slower.
- **Not covered, and pinned as such.** DECIDE's second argument is the door's `facts`; in Python its `__globals__` is
  the module. A DECIDE that takes `main`'s `DiffClaim` from there and patches it moves the record, and a committed test
  asserts that it does (`test_a_decide_that_reaches_the_module_through_facts_is_the_stated_limit`), so that a change
  that closes the route is noticed and this sentence moves with it. In the port, a patch of what objects or arrays
  inherit that is left in place can reach APPLY's own work before DECIDE runs (a `Set`'s `has`, an index setter that a
  store to an index not yet held looks up) as well as the next call's `main` reader, and a function built from a
  string reaches whatever a page's script can; neither is covered. (Up to the eleventh pass this said such a patch
  reaches `main`'s reader before APPLY; the eleventh construction reviewer's `push` left patched to drop a lone `null`
  left `main`'s reader unchanged and put 317 of 320 records outside the relation through APPLY's own `push`.)

Two things inside the boundary are held otherwise:

- **APPLY itself, and the door's call of it.** An edit of `_p2a_apply`, the copy's two classes, `_p2a_abstain` or the
  two hook lines of the Python, or of `_p2aApply`, the tools it takes at load, `_p2aAbstain` or this port's
  `gateDiffText`, is an edit of the code that writes `main`'s record. Those lines are pinned as text, one hash per port
  (`test_the_code_that_touches_the_record_is_the_pinned_text`, 80 lines at most, 75 before the eleventh pass; a named
  test also holds that the copy's two classes define no `__init__` and no method, since the twelfth pass), so that
  such an edit fails a test that names what moved; the byte pins of the whole file fail on a comment anywhere and do
  not single it out. A mutation check of the tenth pass planted eleven edits in each port's APPLY or DECIDE (a
  decision taken for any index, written twice, with any phrase or any tag; the record or a claim's detail handed to
  DECIDE; no `try`; a non-list ignored; the gate verdict not recomputed; a copy returned; a tag APPLY refuses): each
  failed a behaviour test as well as the pin; the eleventh pass's mutation check is in its corrections note.
- **A DECIDE whose decisions APPLY refuses.** An ignored decision keeps `main`'s verdict. The count of decisions taken
  (above) and the coverage tests hold that.

**The lints.** `selfcheck_p2a_asks_no_runtime` (Python, with `ast`) and the port's token scan (in the test module)
are lints against an honest future edit, not proofs against a hostile one. What they guard is the premise of bar C,
that the block asks no runtime a Unicode question; they say nothing about what the block may write. Up to the ninth
pass they also carried record-write checks (every store, an alias reader, the shape of the abstain loop, the port's
store scan). Each review found another way round those, the ninth among them through a default argument, `*args`, a
class attribute, a `for … of` target and an optional call; the tenth pass deleted them, with their plants, instead of
extending them once more.

The Python lint: no import; no flag group holding `i`, `u`, `L` or `a` in any position; only names the block binds
or a short list of builtins and `main`'s names (`int` is refused by name, since it reads CPython's table of decimal
digits, except as `T` in `type(x) is T`, the one form in which `type` is allowed); only a short list of attributes
(so no dunder and no unlisted method); `re` only as `re.<function>(<static pattern>)`, with no class escape, no
named-character escape and no unescaped `.`; no strip or split without its characters; no `str()`, `repr`, `format`,
`!r`, `getattr` or `eval`, and, since the eleventh pass, no `%`-formatting of a string literal (`'%s' % ([x],)` reads
`repr` as `%r` does; the block formats with `%` nowhere); ASCII source.

The port's scan reads the block's code outside strings and comments, with white space removed between tokens and,
since the tenth pass, a Unicode escape read as the character it spells: an escape in code can only be part of a name,
and at `c69b161b` `toLowerCase` written with `\u0061` passed every rule. A backslash left in code is refused. Since
the eleventh pass it reads each `${…}` of a template literal as code: up to then a template was read whole as a
string, so `${Number(x)}` or `${x == 3}` met no rule (C-3 and I-4 of the tenth reviews; eleven plants). The
scan refuses `normalize`, `toLowerCase`, `trim` and their kin anywhere and whatever the form of the call (an optional
call or chain included), `.sort(` and, since the twelfth pass, `.toSorted(` (both order by UTF-16 code unit, not by
CPython's code point), `new String(`, `.match(`, `.search(` and `.matchAll(`, regex literals and regex modifiers,
`==` and `!=`, a unary `+` or `-` before a name, a call, a bracket or a string, `~`, `|`, `&`, `^`, `<<` and, except
after `.length`, `>>`, and the words `Number`, `parseInt`, `parseFloat`, `isNaN`, `isFinite`, `Math`, `Date`, `BigInt`,
`DataView`, every typed-array constructor, `Object` and `Proxy`; it allows `RegExp` only as
`new RegExp(<one static string>)` whose decoded value is checked, and computed member access only with a short list of
index expressions, never called. Every plant a review found for these rules is a committed refused case.

**What the lints do not cover**, and are not meant to. A name built from strings at run time: `getattr` and `eval`
are refused by name, a table of functions indexed by a computed key is not. An alias of a refused function carried
through data. In Python, an f-string of a list, which reads `repr`. In the port, a binary `*`, `-`, `%` or relational
operator, or `++` and `--`, on a string, each of which converts with the engine's white space (committed tests pin
that the scan accepts `claimed * 1` and `t++`); the block gives such operators numbers only. The numeric parameters of
built-in methods. In the port, a call of one of `main`'s own helpers that asks the engine (`_hasRealExtension` and
`_norm` lowercase, `_prefixIsPathShaped` trims): the port's scan has no list of the names the block may read, as the
Python lint has, and the block's comments name such helpers as the ones its own functions mirror (C-4 of the tenth
cross-port review; the Python lint refuses the analogues). None of this is chased: the lints are there so that an honest edit that reaches for `lower()` or
`trim()` is stopped by name, and the transliteration itself is held by the cross-port tests.

### Cost

Measured the committed tests' way: the overlay alone on `main`'s record, against `main`'s own whole call on the same
input, the least of three runs (two on the large summaries), on a machine other jobs were using.

- **Per call.** On `main`'s corpora (one `path2a_recall.py` run): `corpus_fuzz.json` 0.204 → 0.234 ms a pair (+15%),
  `corpus_real.json` 7.92 → 7.89 ms. The copy APPLY hands DECIDE takes 5.5 to 5.8 ms for 10,000 claims in Python and 9
  to 10 ms for 30,000 in the port, beside calls of `main` that take 630 to 750 ms and 200 to 290 ms on those
  summaries; where no claim is in reach nothing is copied (the tenth pass's figures; the Python copies were then
  `DiffClaim`s). The eleventh pass's APPLY (slotted copies in Python; reads by index and no inherited method after
  DECIDE in the port), measured at `ee82d2f3` alone against `16daa725`'s on
  records of 1,000, 10,000 and 30,000 claims in reach, the least of seven runs: in the port 0.28, 2.6 and 9.6 ms against
  0.34, 4.2 and 11.1 with a DECIDE that returns nothing, and 0.35, 3.4 and 9.9 against 0.91, 7.4 and 25.1 with one that
  returns a decision for every claim; in Python 0.34, 4.1 and 17.9 ms against 0.73, 5.5 and 22.7, and 0.8, 14.3 and
  68.5 against 1.1, 14.0 and 55.0 (the last pair within the run-to-run spread of this machine). The twelfth pass's
  port APPLY (a literal key list, per-claim arrays filled by index), against `ee82d2f3`'s the same way, two runs in
  turn: 0.35 to 0.51, 2.4 to 3.3 and 8.6 to 11.4 ms against 0.25 to 0.40, 2.5 to 3.3 and 7.6 to 12.0, within the
  spread (*measured at this head*).
- **At import** (*measured at `ee82d2f3`*, the tenth pass's method; the twelfth pass changed a docstring and comments
  of the Python). The module body, executed in one process 25 times
  in turn with `main`'s, the regex cache purged before each run: 30.3 ms (least) and 34.6 ms (median) against
  `main`'s 5.1 and 6.2 on CPython 3.12.10, ×5.9 of the leasts and ×5.6 of the medians; 25.9 and 27.8 against 4.7 and
  5.4 on 3.14.2, ×5.5 and ×5.1. `import styxx.diffgate` in a fresh interpreter, with the standard-library modules it
  imports and `styxx.declare` already loaded and the bytecode cached, nine runs in turn: 62 ms (least) and 71 ms
  (median) against 18 and 19 on 3.12.10, ×3.4 and ×3.7; 70 and 75 against 20 and 20 on 3.14.2, ×3.5 and ×3.7. The
  tenth pass's head measured ×5.4 and ×3.8 on 3.12.10 the same way; the eleventh pass added two classes and a frozen
  set, so the spread is the machine's. So a start costs about 50 ms more than on `main`, at every CLI, hook and
  Action start. Most of it is the block's 26 patterns over wide character classes (19 to 20 ms when compiled one by
  one). The reviewer of the ninth pass measured ×5.7 to ×6.8 for the module body, and 106 to 158 ms
  against 31 to 38 in a fresh interpreter, on a busier machine; the earlier text's "27 to 30 ms against `main`'s 6" was
  the module body alone. Compiling the classes on demand would spare a start that meets no claim in reach; that is a
  change of the block and is not made in this pass.
- **The committed timing cases**, the worst shapes the reviews found (a `def` beside 5,000 or 50,000 letters outside
  ASCII, one 220 KB added line of `def test_a`, 2,000 files with 200 path claims under 50 directories or one base name,
  500 symbol claims over 50,000 removed `def` lines, 300 `class` claims beside NBSP runs, four 64 KB summaries whose
  claims each meet thousands of runs or zones, one path of 40 KB and 100 paths of 4 KB, 3,000 directories or base names
  outside ASCII under 3,000 to 4,369 claims, 4,000 distinct ASCII claims over 3,000 files of one base name, a run of
  40,000 `./`). *Pinned*: each within the larger of 0.5 s in Python (1.0 s below CPython 3.11, which has no
  specialising interpreter and was not available here; 0.3 s in the port) and five times `main`'s own call, and under
  64 MB of peak memory in Python. Measured: slowest overlay 0.161 s on CPython 3.12.10 (3,000 distinct claims over a
  base name outside ASCII; `main`'s call 0.087 s), then the symbols case, 0.138 s (0.047 s); 0.132 s on 3.14.2 (the
  symbols; 0.037 s); 50 ms in Node (the symbols; 12 ms). Largest ratios: ×30 in Python on the run of `./` (×26 on
  3.14.2) and ×6 to ×7 on the `def` beside 50,000 letters (×6.4 to ×10 in Node), where `main`'s call takes a
  millisecond or less and the absolute bound is the one that applies; ×3.6 at most elsewhere in Python and ×4.5 in
  Node. Peak memory 14.3 MB.
- **Large summaries** (1.2 MB with 10,000 distinct count, path or scope claims; three times that in the port).
  *Pinned*: the overlay alone within 1.5 times `main`'s call in Python and 3 times in the port. Measured: 0.10, 0.34
  and 0.47 of `main`'s call on 3.12.10, 0.12, 0.28 and 0.35 on 3.14.2, and 0.48, 1.34 and 1.42 in Node. Two more
  committed tests bound peak memory on thousands of distinct long tokens (32 MB in Python, a 64 MB heap in Node).
- **The input the eighth review found** for the window reader that went with the switch ("3 files changed. " then
  16,379 times a zero-width space and `add`, 65,534 characters). Over a diff of one changed file, where the count rule
  returns before it reads the summary, the overlay alone takes 0.9 ms in Python (`main`'s call 56 ms) and 0.10 ms in
  Node (1.3 ms), and 3.4 ms and 0.20 ms at four times the length; that is the diff the earlier figure was measured on.
  Over a diff with dot twins, where the summary is read: 7.7 ms in Python (54 ms) and 1.8 ms in Node (1.1 ms), and a
  ninth-pass reviewer found it linear to 1 MB there. The eighth review measured about 3 s in each port at the ninth
  pass's parent, growing with the square of the length.
- **Line-heavy diffs cost more**, because the overlay reads the diff again, in two line views, linearly (measured, not
  bounded by a test; `main`'s call → the overlay alone). The largest factor in Python is on unchanged lines that
  hold definitions, which the overlay reads and `main` does not (the last row, a ninth-pass reviewer's shape, who
  found no cost that grows faster than `main`'s on 38 scaling families):

| input | CPython 3.12.10 | Node 24.13.0 | peak memory of a whole call, Python: `main` → here |
|---|---|---|---|
| one added line of 40,000 pieces joined by U+0085 and VT, half of them `def test_` sites (0.5 MB) | 20 → 42 ms (×2.0) | 0.4 → 15.0 ms (×36) | 2 → 5 MB |
| one added line of 60,000 U+2028-separated `def test_x(): pass` segments (1.2 MB) | 27 → 154 ms (×5.7; ×9.8 on 3.14.2) | 3.1 → 56 ms (×18) | 4 → 15 MB |
| 200,000 added lines each holding U+2028, U+0085, VT or FF (2.7 MB) | 245 → 303 ms (×1.2) | 45 → 110 ms (×2.4) | 26 → 54 MB |
| 400,000 lines, `-a` and `+a` in turn (1.2 MB) | 240 → 340 ms (×1.4) | 46 → 112 ms (×2.5) | 23 → 43 MB |
| one 5.8 MB added line of `def ` | 61 → 96 ms (×1.6) | 6.7 → 4.2 ms (×0.6) | not measured |
| 80,000 unchanged `def test_` lines and one added test (1.4 MB) | 36 → 323 ms (×9.0; ×16 on 3.14.2) | 6.9 → 138 ms (×20) | 5 → 24 MB |
| one 1 MB added line of `def test_` repeated, beside a removed line of `def ` repeated (a tenth-pass reviewer's) | 17 → 182 ms (×10.5) | ×17 | not measured |

The tenth construction reviewer, over about 70 scaling families against `main`'s whole call, found the overlay linear
in both ports; one family grows a little faster than `main` without growing faster than linear: scope claims in a
summary of many zones, from 1,000 to 16,000 claims, where the overlay alone moves from 0.08 to 0.13 of `main`'s call in
Python (exponents 1.12 to 1.34 against `main`'s 0.99 to 1.05) and from 0.15 to 0.20 in Node. The reviewer's other
largest constant factors on line-heavy diffs were ×8.5 in Python and ×24 in Node on one shape of many context lines.
The eleventh construction reviewer (ten families at five doubling sizes, at `ee82d2f3`) found the overlay linear in
both ports, and three shapes above the ratios given here, each `main`'s call → the branch's whole call: dot twins, 4,000
pairs (8,000 files with count, only and path claims), 38 → 205 ms in Python (×5.4) and 14.7 → 71 ms in Node (×4.8);
one 1 MB added line of `def test_x(): pass; ` repeated, 8.9 → 106 ms (×12) and 1.05 → 9.2 ms (×8.8); 16,000 lines
joined by NEL or U+2028, 6.75 → 52 ms in Node (×7.7). Each lies within the pinned absolute bounds.

- **`main`'s own call is not bounded** on a description that holds one unbroken run of thousands of word characters:
  its path template backtracks cubically there, in both ports, with no overlay involved (a probe's measurement before
  this pass: `gate_diff_text` over a 2,000-letter word took 12.8 s on CPython 3.12.10, and a 3,200-letter one 6.8 s in
  Node). That is none of the three defects and this branch does not touch `main`'s reader; the bounds above are
  relative to that call, and no committed input or timing case holds such a run.

The bookmarklet grew from 24,335 characters (`main`) to 53,371 (53,246 before the keys DECIDE may name were written
out; 52,937 before APPLY stopped calling inherited methods; 51,838 before APPLY was set apart; 56,575 before the
switch went).

## Who did this earlier

The gate this directory ports checks an agent's account of a change against the change. Others
did that, or parts of it, before `styxx/diffgate.py` existed (7.29.0, 2026-08-01), and some did it
better:

- **PR-MCI** (Jingzhi Gong, Giovanni Pinna, Yixin Bian, Jie M. Zhang; arXiv 2601.04886, 2026-01-08)
  measured description-versus-code inconsistency on 23,247 AIDev agent pull requests and
  hand-labelled 974 of them, about seven months before this lab did.
- **AgentLiar** (Daksh Jain per its account name, 2026-05-20) shipped a GitHub Action that checks an agent's completion
  claim against its file changes, including a scope-narrowing check and assertion-free tests.
- **Swarm Orchestrator** (Brad Kinnard; v10.0.0, 2026-05-23, self-reported) gates AI-written pull
  requests on their diffs with ten cheat detectors and a hash-chained ledger, and measured its own
  detectors on 300 planted cheats (2026-06-02).
- **commitlint-scope** (thumbrise, 2026-05-25) checks changed paths against a declared scope, the
  comparison `only_touches` makes.
- **backcheck** (Vector Institute, 2026-08-04) checks an agent's closing claims against the tool
  records of its session, and binds a "tests pass" claim to the run it names, which this gate
  cannot.
- **DeerFlow** (RFC #4651; citation verification merged in #5076 2026-08-29) and **NabaOS**
  (2026-03-09) check agent claims against receipts the model never writes; **readback** (2026-09-13) checks a declared claim block.
- Older: **DOCER** (a GitHub Action for outdated code references in docs, 2023), **iComment**
  (comments against code, 2007), linux-next's **`Fixes:` tag checks** (since at least 2019), Qodo's
  LLM **ticket compliance** (2024-11-26), and **FEVER**'s Supported / Refuted / NotEnoughInfo
  verdict (2018).

The sentences of this lab's that these neighbours correct, and the dates on both sides, are in
`papers/closed-model-frontier/NOTE_prior_art_credit_2026_09_29.md`.

## What this is not

It is not a second instrument. The Python is the instrument; the JavaScript exists only where the
Python cannot run, and the day the two disagree on a pair in this corpus, the JavaScript is wrong
by definition. It is not a check of anything but transliteration fidelity — the template set's
own precision was measured elsewhere (EXTERNAL-1: path accusations at 0.23 against a 0.95 floor
on 71,016 agent-authored PRs, and withheld since), and no number here bears on that.
