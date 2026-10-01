# web/gate — the diff gate where Python cannot run

The instrument is `styxx/diffgate.py`. Two browser surfaces cannot import it: the paste-in
preview page and the bookmarklet. They run `diffgate.js`, a JavaScript transliteration of one
specific file — `styxx/diffgate.py` on this branch, sha256
`40bb973b3cfce08c48e25b202c311be9664f6eacaee55d7003021e808d5e1eb9` (LF line endings; a wheel
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

    bookmarklet.min.js    sha256 f873dc6a9cc65179a1355f2d1c0130c383e50d92c97497e593e8b2f9ebb5183a   56,575 chars
    bookmarklet.href.txt  sha256 d0133602b703ab181fff6c81617f8e77234de5fa7b2dd2dd2fee8261a9e5beb8   56,586 chars

(Earlier builds: `b04d14dc…`, 11,437 chars, from the 7.47.0 file — accuses outside Python;
`9ea8f572…`, 17,686 chars, the BC-2 + COMPAT-1 re-cut — cannot see a binary file;
`1be19a65…`, 24,335 chars, `main` at `1cde8b82` before PATH-2a, whose README still named an older
`54dca73a…`, 21,632 chars; `c457cca3…`, 34,285 chars, PATH-2a's pass 1; `bfe8c047…`, 36,933 chars, its pass 2;
`c726c904…`, 38,763 chars, its pass 3; `eeb0c298…`, 40,419 chars, its pass 4; `185352f8…`, 44,199 chars, its pass 5;
`7dd3628e…`, 48,227 chars, its pass 6; `98b1f5ad…`, 49,888 chars, its pass 7.
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

    cd web/gate/differential             # on a checkout carrying #113 and #115 (or 7.48.0)
    python build_corpus.py               # 176 real pairs, pinned to shas (below)
    python fuzz_corpus.py                # 3,000 synthetic pairs, seeded
    python py_side.py                    # refuses to run unless styxx/diffgate.py hashes to 40bb973b…
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
`NOTE_path2a_third_pass_2026_09_30.md`; for the fourth `NOTE_path2a_fourth_pass_2026_09_30.md`; for the fifth
`NOTE_path2a_fifth_pass_2026_09_30.md` and `NOTE_path2a_fifth_pass_corrections_2026_09_30.md`; for the sixth
`NOTE_path2a_sixth_pass_2026_09_30.md` and `NOTE_path2a_sixth_pass_departures_2026_09_30.md`; for the seventh
`NOTE_path2a_seventh_pass_2026_09_30.md`; for the eighth `NOTE_path2a_eighth_pass_2026_10_01.md` and
`NOTE_path2a_eighth_pass_corrections_2026_10_01.md`). `main`'s reader runs
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
branch instead.

**The GitHub Action runs the styxx beside it, not PyPI's.** `action.yml` installs styxx from PyPI and then runs
`python "${{ github.action_path }}/diffgate_action.py"`; Python puts the script's own directory at the head of `sys.path`,
so `from styxx.diffgate import gate_diff_text` imports the `styxx` package of the checkout the action runs from, at
the ref the workflow names. This repository's own `diffgate` job (`uses: ./`) therefore runs the overlay on this
branch, and a workflow using `fathom-lab/styxx@main` runs it once this merges, before any release. (The fifth pass's
note and this README said the Action ran PyPI's package; they were wrong, NOTE_path2a_sixth_pass_2026_09_30 I-1.
`main`'s `.github/workflows/diffgate.yml` says the same in a comment; that file is `main`'s and this branch does not
touch `.github/`. Whether the Action should import the released package instead is the operator's call.) Its job
summary shows the overlay's own words whole — only for an UNCHECKABLE claim of a kind the overlay may move whose reason
starts with the overlay's form — and cuts `main`'s reading after them at 100 characters, as `main` cuts every reason,
so a long path cannot carry the table past GitHub's step-summary limit.

The rules. A path claim (VERIFIED only; the path accusation is withheld on `main`) is kept only when
three readers without the mechanism verify it too: V97 (exact, then suffix, then base name — base
name only for a bare claim), V121 (keys keep their leading dots) and both. A file count abstains when
two changed paths differ only by a leading dot and the dot-kept count could read otherwise (a
CONTRADICTED count only when the dot-kept count range contains the claimed number; a VERIFIED count
unless that range is exactly it). Where the summary holds a *count seam*, every count read from the summary abstains
in both ports (`seam`) unless C-1 below keeps a CONTRADICTED, since there one port can read a count, with a clean
number, that the other does not read at all: a white space only one port's `\s` reads (U+001C to U+001F and U+0085 in CPython, U+FEFF in the port) inside a
run of either port's white space between a character that can end a number or a word of the count (an ASCII digit, `e`
or `s` in either case, U+017F) and one that can begin one (`c`, `f` or `w`); or `file` spelled with a letter only
CPython's IGNORECASE folds to ASCII (U+0130, U+0131, U+017F; U+212A folds to `k`, which no word of the count holds).
`only_touches` abstains when V121 reads "every changed path lies under the
prefix set it uses" otherwise than `main` reads it under the set `main` uses, and, with a second prefix claimed,
when the leading prefix is a path for `main` only because its leading dots were dropped ("Only touches github and
.github/workflows/.": V121 says the prefix is not a path; `shape`). `tests_added` abstains when a
counted test is also defined in the removed lines and the claim lies in `[got − changed, got]`, read for each
port's `main` whose count gives the claim the verdict it has (each port's count is read exactly from the
bytes; `tests`, or `split` where only one port's reading holds it), or, with the tests an unchanged line of the diff
defines paired too, the claim lies in that wider interval (`redefined`: a test defined again beside its own unchanged
definition is counted by `main` and is not an added test); `symbol_added` (VERIFIED) when a removed line defines the
name where V101 reads a definition (at the line start, after white space and an optional `async`; `symbol`), or an
unchanged line does (`again`). Unchanged lines are read as removed ones are, in both line views and in the pieces no
view reads, except that a definition there whose name reads through NFKC is passed over, so that a context line of
prose cannot withhold every tests and symbol claim. The removed side is read three ways: the two line views, and
the removed text neither reads as a line of its own, that is, a piece after a lone CR inside a removed git line
(CPython's tokenizer ends a line there, as at LF and CRLF, and nowhere else: a definition after any other break
`str.splitlines` knows does not parse, and one after a leading form feed, which the tokenizer reads as white space, is
read by the line views) and base-side lines joined by a backslash continuation. A name CPython reads through NFKC (a code point from
0x80 up in the gap after `def` or `class` or where the ASCII name run ends) may be any name, so it pairs with every
name: such a removed test with every counted one, such a counted test with every removed one, and such a removed
definition defines every claimed name. The claimed number is read from the claim's detail, not from
`main`'s reason, since the port's `main` prints a number of 10^21 or more in exponent form, and digits are read from a
fixed table of the ten ASCII digits, never through `int()`.

**Where the two ports may read the claims apart, a CONTRADICTED stands** (C-1 of the sixth pass). Without `--strict`
a gate verdict is FAIL exactly when a CONTRADICTED is left, so the overlay moves it only by withholding a
CONTRADICTED. Where `main`'s two ports may read apart which claims can be CONTRADICTED, or decide such a claim apart,
withholding one could move one port's gate verdict and not the other's; there the overlay withholds no CONTRADICTED,
in either port, and each port's gate verdict without `--strict` is its own `main`'s. That is the case (`apart`) when
(1) a sentence, as both ports end one, holds a wordish or divergent code point inside the *window* a match of a
template that can give CONTRADICTED can cover in either port: from the character before the match (its `\b`; for a
count, before the number run that precedes `file`) through its last word, and on past it where the template reads on
(the one character after `test(s)` and its optional noun, a symbol's name run and the character after it, a scope's
prefix runs and its `, and …` tail), each part read as a run of the union of both ports' classes (either port's white
space; ASCII or wordish for `\w` and `\d`; ASCII path characters or wordish for a path) and each literal word in ASCII
case with U+0130 and U+0131 as `i`, U+017F as `s` and U+212A as `k`; or one of the DECLARE-1 keys `files_changed`,
`only_touches`, `tests_added` and `adds_symbol` (a key's line writes the sentence) and such a code point anywhere in its
sentence (the eighth pass's B-2: the seventh read such a code point anywhere in the sentence that held the template's
words in order, so "Added 1 test for naïve inputs." and "3 files changed (café config)." kept a false CONTRADICTED;
the sixth read the words in any order, so a decorated sentence elsewhere kept every one); (2) the summary holds `styxx`
(a DECLARE-1 fence, found by `^`, which the port's `m` flag also matches after CR, U+2028 and U+2029) and a divergent
code point or a lone CR; or (3) among the kinds of claims `main` read, the diff may make the two `main`s decide one
apart: a file header they split or strip apart; for a tests claim, the two line views' own counts of `def test_` sites
differ; for a symbol claim, an added `def` or `class` site on a line only one view reads, or with a white space of one
port alone before the name, or whose ASCII name run is a claimed name followed by a code point from 0x80 up, which
CPython's `\b` may read as a word character (the seventh pass's B-3; the sixth read any name through NFKC here, which
neither `main` does), or an added line that is, from its start, white space of either port, `def` or `class`, and white
space to its end (the eighth pass's C-1: `main`'s regex `^\s*(?:def|class)\s+NAME\b` runs on the joined added lines, so
its `\s+` spans the line break and reads the name on a later line, where the two ports' `\s` and `\b` part; at
`8eead84f` the review's generator split 538 of 4,000 non-strict gate verdicts where `main`'s agree); and added lines
empty to one `main` only where no path is registered. Everywhere else both `main`s read the same such claims, and give
them the same verdicts but for a count claim the two runtimes' case tables may count apart, which the overlay withholds
in both ports whatever its verdict (below); the overlay decides every other claim alike, so the two gate verdicts agree
wherever `main`'s do: a proof, not a measurement, up to the fold lemma below (each runtime's lowercase maps only U+0130
and U+212A to text holding ASCII) and the case tables' premises, whatever Unicode version each runtime carries. A test
patches the port's engine to fold one pair no runtime folds yet and asserts it. It costs coverage there: a false CONTRADICTED of #121 or #101 that the fifth
pass withheld stands, in both ports (the pinned cross-port cases X2, X2-13, X3, X5, X5b, G1, G2 and G3 each keep one
port's false count), and `main`'s two ports may already disagree on such inputs. Under `--strict` a withheld VERIFIED
moves a gate verdict too, and a claim one port's `main` reads alone can be withheld on that port only; `apart` does not
reach it (see *Cross-port* below).

**A count beside a case pair outside ASCII is withheld in both ports** (the eighth pass's B-1). `main` keys each path
through its runtime's lowercase, and the runtimes' case tables differ by Unicode version (CPython 3.9 to 3.12 carry
Unicode 13 to 15, CPython 3.14 and Node 24 Unicode 16, and CI's newer Node 17), so two paths that differ only in case
outside ASCII may be one key in one port and two in the other. The overlay reads which code points some runtime may
merge from two static tables. The 1,432 lowercase mappings of Unicode 16.0 (U+0130 and U+212A aside, which the fold
form already reads), written as 184 runs, give each cased code point its class (its lowercase). The blocks of scripts
and symbols with no case — Hebrew through Myanmar, Hangul Jamo and Ethiopic, the Canadian syllabics through Ol Chiki,
the CJK radicals, symbols, kana, Bopomofo, Hangul compatibility and enclosed blocks, the CJK ideographs and their
extensions, Yi, Lisu, Vai, Bamum, the Brahmic and Southeast Asian blocks from U+A800 to U+AB2F, Meetei Mayek, the Hangul
syllables, the Hebrew and Arabic presentation forms, the half-width kana and Hangul, and the neutral set — hold code
points no runtime's lowercase maps or maps to. Any other code point (one a later version may give a case, as Unicode 16
gave U+A7DC the lowercase U+019B, and CI's Node 17 more in U+A7CE to U+A7D5) may pair with any code point outside those
blocks. Over the forms `main` keys, at each position where forms of one ASCII shape differ, a code point with no case is
read as itself and the others as their class, or all as one placeholder where some form there holds a code point of
neither table: #CA counts the keys that leaves, a lower bound of every runtime's count, as #A (ASCII case folded only)
is the upper one. Where #CA < #A, a count claim whose number lies in [#CA, #A] is withheld in both ports, whatever its
verdict (`case_count`), since the two `main`s may decide it apart; a number outside the range is CONTRADICTED by every
`main`, and the overlay decides it as before. Two premises, stated: Unicode's case-pair stability (a case pair, once
assigned, stays one; two assigned code points that are not one never become one), and that the blocks listed gain no
case; the tests pin both tables against the lowercase of every runtime they run (exactly on Unicode 16, a subset on
15). The seventh pass instead kept every CONTRADICTED wherever two changed paths differed only outside ASCII (#WA <
#A), so two CJK or Cyrillic names that are not case pairs kept the false tests verdict of a changed test beside a count
claim (the seventh review's CJK world: 19 attributable claims kept; on the EXTERNAL-1 shelf 6 of 69,654 PRs hold such
a pair, all CJK or Cyrillic names that are not case pairs).

A decision reads the claim's kind, verdict and detail, `main`'s own counts in its reason, and the door's bytes — the
diff, the `--name-status` listing and the summary — never the claim's text, which the two ports cut (160 code points
in Python, 160 UTF-16 units here) and strip (`strip()` and `trim()` part on U+001C to U+001F, U+0085 and U+FEFF)
differently. `main`'s counts in its reason agree between the ports within the overlay's bracket on any engine whose
lowercase keeps the fold lemma the tests pin on the engines they run (only U+0130 and U+212A lower to text holding
ASCII); the bookmarklet also runs on browser engines where that is not checked. Where the two ports' templates may read a claim's path, name, prefix or number apart, the claim abstains
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
U+1FAFF; operator option O-11, taken in the seventh pass) are read as U+2190, a neutral code point, whether the string
holds one code point or the two surrogates the port's string holds: none of them is a word character or white space
for either port's templates, folds to an ASCII letter or has a case, and the tests pin that and the two regexes that
read them by enumeration too. Declared
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

Running it, from the repository root:

    python -m pytest tests/test_diffgate_path2a.py tests/test_diffgate_path2a_truth.py tests/test_port_is_current.py
    cd web/gate/differential
    python path2a_recall.py --corpora DIR [EXTRA.json ...] [--truth] [--path-flavour posix|windows]
    node check_pairs.js                                   # the pinned pairs, path2a_pairs.json + path2a_moves.json

The port half of the PATH-2a tests, and `test_port_is_current`'s check of the port against the pinned pairs, need
`node`; without it they skip locally and fail under `CI` or `GITHUB_ACTIONS`. `path2a_recall.py` prints three totals:
`main`'s committed corpora (the figure below), the overlay's own pins, and `main`'s corpora with any extra files.
`path2a_pairs.json` pins 136 pairs, each for the decision it names (19 of them pin kind and verdict only, where
`main`'s two ports print different reasons); it is not one of `py_side.py`'s corpora (some of its pairs are inputs
`main`'s own two ports read differently), so the port differential over `main`'s corpora reads 0 disagreements.
`path2a_moves.json` records the one pinned claim of `main`'s own files the overlay moves —
`path1:unrepaired-typo` claim 0, ".githiub/workflows/dependabot.yml" resolved by base name to
`.github/workflows/dependabot.yml`, a false VERIFIED — so `path1_pairs.json` stays `main`'s record,
byte for byte.

**Measured**, on this file (`styxx/diffgate.py` `40bb973b…`, reader `9b620e00…`). Figures marked *pinned* are
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
| `path2a_pairs.json` (the overlay's own pins, built to abstain) | `5e0de5bb…` | 155 (Windows), 154 (POSIX) | 76 (Windows), 75 (POSIX) |
| #161's `path2_pairs.json` (branch `fix/diffgate-path-resolution`) | `7ba272c8…` | 530 | 190 |
| `main`'s corpora + #161's pairs | | 2,761 | 270 (9.8%); #161's head withheld 493 of 2,749 |

At the eighth pass `main`'s corpora read as before, and #161's pairs move 184 → 190: `case_count` withholds six k2
claims there (Unicode 14 and 16 case pairs; 2 CONTRADICTED and 4 VERIFIED, which the seventh pass kept). The
seventh review's run on the EXTERNAL-1 shelf (69,109 real agent PRs, at `8eead84f`, not re-read here): 152 of 11,859
decided claims in reach withheld (1.3%); judged against GitHub's file list, 115 of 115 attributable false verdicts and
0 of 31 others withheld, at 2 of 6,793 right verdicts (both `extract`). No figure on `main`'s corpora or #161's pairs
moved at the seventh pass. At the sixth, five fewer were withheld on
#161's pairs than at the fifth: C-1 keeps eight CONTRADICTEDs there (three counts and
four scopes under a divergent header, one `split`), and `redefined` withholds three VERIFIED tests claims. The count
seam and C-1 move nothing on `main`'s corpora. The only real-world #97 record in `main`'s
corpora, `corpus_real` pr98 ("integrations/git/README.md — created."), is left as `main` has it: neither of its two
records is a false decided verdict (the created claim is UNCHECKABLE with a reason naming the wrong status, the
touched claim VERIFIED on the wrong file). Every abstention on `main`'s corpora is on synthetic input.

Coverage (B), judged by truth from base/head file models (`tests/test_diffgate_path2a_truth.py`,
*pinned*). A decided claim is attributable when `main`'s verdict is false and a variant of `main` without
the mechanism (V97, V121, V101, or all three) does not read it false.

| door | cases | main decides | false | attributable | withheld | right verdicts lost | undecided withheld |
|---|---|---|---|---|---|---|---|
| raw door (Python) | 1,706 | 7,614 | 1,535 | 1,206 | **1,206** | 253 of 5,897 | 39 of 182 |
| the port, in its own terms (variants built from `main`'s port) | 1,706 | 7,393 | 1,385 | 1,100 | **1,100** | 177 of 5,739 | 126 of 269 |
| git door (Python), the reproductions with their own `--name-status` | 96 | 63 | 17 | 14 | **14** | 0 of 46 | — |

The cases are #161's 101 reproductions, the 5 PREREG_path2 reproductions and 1,600 generated cases. In
the port, 942 of the 1,100 attributable claims carry the same `main` record as in Python; the other 158
are paths and names outside ASCII, which the two templates extract apart. #161's recorded `--name-status`
carries type changes but no rename or copy, so renames, a copy and a type change are read at a real git
door by `tests/test_diffgate_path2a.py`. The fourth review's three `shape` reproductions (a GNU deletion, a mnemonic
header, a rename into `.github/`, the last at both doors) are judged by truth in a test of their own. So are the fifth
review's definition reproductions (a test or function CPython reads across a backslash continuation, after a lone CR, or
through NFKC; ten of the review's and one reverse NFKC case of the builder's), at the raw door, the git door and in
the port, 33 claims, all withheld. Their attribution uses an **ast-paired V101**: `main`'s own count less the test
functions the base and the head of the same changed file both define, and `main`'s VERIFIED symbol only where no
changed file's base defines the name, definitions read by CPython's parser under NFKC. The committed V101 reads the
removed lines one by one, as `main` does, and cannot see these; at `5ebe0b6b` every one of them was kept at every door.
The sixth review's are judged the same way (`tests/fixtures/path2a_pass6_repros.json`, 15 claims at three doors): a
test defined again beside its own unchanged definition (both summaries false by truth and not false under the
ast-paired V101, withheld as `redefined`; kept at every door at `495d2204`), a name defined again (undecided by
truth, withheld as `again`), and a claim naming a moved file's old path (right by truth, withheld: a known loss, below).
Beyond the committed V101's attribution the unchanged lines withhold 16 false verdicts in the builder's world ("false
other" 51 → 67 withheld in Python, 7 → 23 in the port), at 6 right and 4 undecided verdicts in each port. On the sixth
review's git-built world (900 cases; default, `--no-renames`, `-U0` and `--no-prefix` renderings at the raw door, and
the git door): 0 attributable false verdicts kept, and all 12 claims per rendering that its attribution reading
context lines names are withheld (none is visible under `-U0`), at 8 right tests verdicts per rendering. #161's five
joint #121 reproductions (`m-121-a-submodule-line-licenses-nothing`, `k4-a-dotted-twin-beside-a-submodule-line`,
`k4-a-dotted-twin-beside-an-hg-binary-notice`, `l-r10-np1-…`, `l-r10-np2-…`) keep `main`'s false CONTRADICTED count:
V121's count is false there too (a submodule line, an hg binary notice or a no-prefix directory `main` registers no
file for), so under the but-for attribution they are not misses; they are pinned, both ports. Operator option O-7
(withhold a CONTRADICTED count above `main`'s count wherever a dot twin exists) would withhold them, at 289 right
CONTRADICTEDs for 41 false ones in the builder's world and 215 for 38 in the review's. The seventh review's narrower
rule (a dot twin, `main`'s count below the claimed number, and a change marker `main` registers no file for: a
`Submodule` line, a binary notice it does not key, or a header with no `---`/`+++` pair it keys) is operator option
O-13, not measured.
The truth test prints any miss with its shape.

The truth worlds' summaries are ASCII, so the seventh pass also read them with a decorated sentence appended — "## 🧪
Tests added", "Thanks to José for the review; only a typo fix otherwise." and "Behaviour is unchanged for naïve
callers." (measured): in the builder's 1,600 generated cases 1,193 of 1,193 attributable claims are withheld under
each, at 236 of 5,842 right verdicts lost, as without one (the sixth pass's head withheld 979, keeping every
attributable CONTRADICTED); in the sixth review's git-built world every attributable claim at all five renderings under
each (raw 619, `--no-renames` 748, `-U0` 619, `--no-prefix` 620, git door 610), 0 count misses judged by name-status
lines; in the port, the count and tests CONTRADICTEDs withheld are 227 and 79 with a decoration or without (0 and 0 with
one at the sixth pass's head). C-1's case-pair clause kept 6 CONTRADICTEDs in each port's truth world at the seventh
pass (4 counts, 2 of them false through neither defect and 2 right, and 2 right scopes), no attributable one.

At the eighth pass (measured, Python and the port alike unless stated): the seventh review's `cjk2` world (the builder's
1,600 cases with two changed CJK docs and a right count sentence where `main` reads no count) withholds 1,073 of 1,073
attributable claims (1,054 at `8eead84f`, whose case-pair clause read any two names outside ASCII as a pair), 1,067 of
1,067 without the count sentence and 1,073 of 1,073 with ASCII names; its `insent` world ("(naïve)" inside each tests
or count claim's own sentence, outside the window) 1,193 of 1,193 (980 at `8eead84f`); the builder's world under the
three appended decorations 1,193 of 1,193 each, right verdicts lost 242 of 5,842 under each (236 at `8eead84f`: the 6
are `case_count`'s 4 right CONTRADICTEDs and 2 right scopes the case-pair clause kept); the sixth review's git-built
world, plain, decorated, or with the in-sentence decoration, every attributable claim at all five renderings (raw 619,
`--no-renames` 748, `-U0` 619, `--no-prefix` 620, git door 610; with the in-sentence decoration `8eead84f` withheld
496, 564, 496, 497 and 487), 0 count misses judged by name-status lines (181 of 181); in the port, 246 count and 79
tests CONTRADICTEDs withheld with a decoration or without (227 and 79 at `8eead84f`). The pinned truth figures moved by
B-1 alone: +34 withheld in each port (26 false verdicts `main` gives by merging two paths' case, outside the three
defects, 13 CONTRADICTED and 13 VERIFIED; 4 right CONTRADICTEDs; and the CONTRADICTEDs the clause kept, now decided: 2
false counts as `count`, 2 right scopes as `shape`); no attributable verdict moved. Two committed truth tests run the
review's `cjk2` and `insent` worlds (`8eead84f` keeps 19 and 213 there).

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
two causes) are listed in the NOTEs. One more is #161's off-tree prefix family (`v4-*` and `f4`): `main` reads a scope
prefix through `rstrip("/.")`, so "Only touches docs/.." is read as `docs` though `docs/..` is the repository root, and
`main`'s CONTRADICTED there is false; #161 withholds these as #121, but the mechanism is none of the three defects
(V121 models `lstrip("./")`), and the overlay keeps them.

Coverage was also measured on the third review's two independent truth worlds (2,000 and 6,000 cases built by
real git, each read at the git door, at the raw door on git's bytes and at the raw door on a second rendering;
two truth oracles; the builder's variants plus an ast-paired #101 variant; measured, not pinned): 0 attributable false verdicts kept at any door, in Python or in the port; right verdicts lost 297 and 785 at the raw doors, 75 and 210 at the git door, 297 and 780 in the port, as at `ea677740`. That ast-paired variant paired definitions line by line, so those worlds held none of the fifth review's shapes; the fourth review's own world T (600 cases built by real git, at `5ebe0b6b`) found them, and they are covered as above.

Per phrase and per `main`'s verdict, on the raw door's truth world at this head (measured; the seventh review's B-4
asked for the split; `case_count`'s false verdicts are `main`'s case merges, outside the three defects): how often each fires, and whether `main`'s verdict was false, right or undecided there.

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

The costs, as the independent worlds read them (measured on the third review's world A, 2,000 cases, and world
B, 6,000 cases; "right lost" is a right verdict of `main` withheld):

- `tests` pairs names across the whole diff, so it also withdraws right verdicts where a test is moved into or
  out of a class, a test file is renamed, a test name is reused in another module beside a changed one, two
  non-ASCII test names share their ASCII run, or a removed line of prose or of another language holds `def test_`
  (V101's pairing reads it there too). Right verdicts lost: world A 162 of 2,331 at the raw door (7.0%)
  and 69 of 1,244 at the git door (5.5%); world B 472 of 6,939 (6.8%) and 195 of 3,669 (5.3%). The builder's own
  world above loses 53. Operator option O-8 (pair by ASCII run only when the name is all ASCII) is not taken. Split by
  `main`'s verdict, `tests` withholds more right verdicts than false ones on CONTRADICTED in both truth worlds (the
  builder's 33 right, 20 false; the sixth review's git-built world, raw door, 49 right, 35 false) and the reverse on
  VERIFIED (20 right, 34 false; 18 right, 44 false); much of the CONTRADICTED cost falls where the committed V101 reads
  `main`'s CONTRADICTED too. Operator option O-14 (keep a CONTRADICTED tests claim where every paired site is an
  anchored `def` on both sides, so the net count is exact) is not taken.
- `redefined` pairs a counted test with an unchanged definition of the same name anywhere in the diff, blind to classes
  and files, as `tests` is: in the builder's world 15 false, 6 right (a test moved into a class, which the removed
  pairing already counts as changed, beside a redefinition) and 1 undecided; in the sixth review's git world, 8 right
  per rendering. `again` fired 4 times there, never on a right verdict. Under `-U0` no unchanged line is visible, and a
  definition outside the hunk is never read: disclosed.
- `dot_earliest` caught no false verdict in either world (A: 18 right, 4 mixed; B: 31 right, 16 mixed). At
  world B's git door it withholds 12 of 8,859 right path verdicts, the only path phrase there that loses any.
  Operator option O-6 would keep a claim both V97 and V97+V121 verify. The sixth review's grammar fuzz (6,000 cases,
  every rendering) found the same at scale: 44 right, 0 false; its git-built world 22 right, 1 false.
- `dir` withholds right verdicts on renderings whose prefixes `main` reads as directories or drops: at world B's
  raw door, mnemonic (`c/` `w/`) 158 of 1,189 right path verdicts, plain 48 of 990, index 42 of 890, no-prefix 36
  of 1,050. `main`'s right verdict there rests on the #97 base-name fallback. On git's own bytes with rename
  detection it withholds a claim naming a moved file's old path, which git writes under its new path only, though
  `main`'s VERIFIED there is right: the sixth review counted 43 of 3,277 right verdicts (1.3%) in its 900-case world
  (`dir` 22, `dot` 9, `dot_tier` 12), at the raw door and at the git door alike, and none under `--no-renames`. The
  review's pair is pinned as a known loss. It also withholds a claim naming a dot-directory file in a monorepo
  (`.vscode/settings.json` over `apps/api/.vscode/settings.json`): V121 and V97+V121 verify it, and V97 alone fails
  because it still drops the leading dot (#121). The seventh review found that shape in all 4 right verdicts among
  `dir`'s 33 firings on the Claude Code, Devin and Cursor PRs of the EXTERNAL-1 shelf (29 false caught), and 6 in the
  builder's world; operator option O-15 (keep a `dir` claim that V121 and V97+V121 both verify) is not taken.
- `shape` fired 43 times in the builder's world, every time on a right CONTRADICTED ("Only touches cfg and .cfg/app/."
  beside files that all lie under the second prefix), and never in worlds A and B. In the fourth review's world T
  (600 cases built by real git, at `5ebe0b6b`) it fired 35 times: 33 withheld right CONTRADICTEDs, and the 2 false
  verdicts it caught were VERIFIEDs. On CONTRADICTED it has caught false verdicts only on the committed reproductions
  (a GNU deletion label, a mnemonic header). Operator option O-12 (`shape` on VERIFIED only) would keep those 76 right
  verdicts and those reproductions' false CONTRADICTEDs alike; it is not taken. The sixth review's grammar fuzz: 66
  right, 1 false, 11 undecided.
- `seam` withholds a count only on a diff with a dot twin (where the count rule reads past its no-twin return) whose
  summary holds a seam and, for a CONTRADICTED, no C-1 sentence: 1 claim on the committed inputs (a pinned pair),
  none on `main`'s corpora and none in the truth worlds.
- C-1 keeps, on the committed inputs, 509 CONTRADICTED decisions in Python and 538 in the port that the fifth pass
  withheld (`divergent` counts 156 and 154, scopes 105 and 101; `extract` scopes 154 and 155, counts 3 and 3; `tests` 83
  and 117; `seam` 5; `split` 3), nearly all on the seeded PATH-2a fuzz and the text-seam set, whose summaries and diffs
  are built from those characters. At the seventh pass its ordered triggers withhold 11 `tests` CONTRADICTEDs it kept
  there, and its case-pair clause keeps 11 more `count` CONTRADICTEDs (seeded fuzz whose paths differ in one letter
  outside ASCII; it would keep 12 without its range) and 6 in each truth world, none attributable. At the eighth pass
  the case-pair clause is gone (B-1: those counts are withheld as `case_count`, and the rest decided), and the sentence
  part reads windows (B-2): 5 scope CONTRADICTEDs of the text-seam set whose sentence holds such a character only outside
  the window are decided (and withheld, `extract`). None on `main`'s
  corpora; none of the truth worlds' attributable claims, decorated or not (above). On #161's reproductions it keeps
  10, none of them false through #97, #121 or #101: 7 under a header or separator the two `main`s split apart
  (`f2-a-separator-before-a-header-shape-adds-no-file`, a count and a scope; `v2-a-reason-prints-a-path-as-python-repr-does`;
  `k-canary-z3-a-path-holding-u0085` and `-u2028`, a count and a scope each), the tests claims of
  `f2-a-context-line-holding-a-separator-adds-nothing` and `x1-the-name-table-is-unicode-15-for-both-ports-a-katakana-middle-dot`
  (false by their names, which `redefined` would withhold, outside the committed V101's attribution), and `y2`'s, which
  is right. (The sixth pass's README said 8; it was these 10.)
- The NFKC rule reads every removed line of every file, not only Python definitions: a `def` followed by a run of
  white space and code points from 0x80 up that holds one, or a name whose ASCII run ends at one, pairs every counted
  test, and a `def` or `class` reached from the line start through such code points defines every claimed name. So
  removed prose withholds every tests and symbol claim of its diff (the sixth review's probes: `使用 def 定义函数`,
  `# def — see below`, `Write def “test_x” style names.`, `Use def x.`), not only definitions whose name or gap holds
  such a code point: 25 claims on the committed inputs (24 on the seeded PATH-2a fuzz, whose removed lines write
  U+00A0, U+3000 or U+FEFF between `def` and the name, and #161's y5), 0 on `corpus_real`; the fourth review found no
  such definition among the 112,086 Python patches of the EXTERNAL-1 shelf. Unchanged lines are not read so.
- `extract` reads every occurrence of the claim's path, name or number in the summary, since the claim's own position
  differs between the ports: a name that also occurs elsewhere next to a letter outside ASCII is withheld (#161's
  `x1` probes, "Added function fo²o. … Added function fo."). An `only_touches` claim is withheld wherever a
  wordish or divergent code point follows `only` in its sentence: an accented name ("reviewed by José"), or an emoji
  outside the Basic Multilingual Plane and the five pictograph blocks O-11 reads as neutral (at the seventh pass "Only
  touches docs/ 🎉" no longer withholds; U+1F7E0, a coloured circle, still does). Punctuation and symbols of the Basic Multilingual Plane — guillemets, low-9 quotes, CJK and
  full-width punctuation, ✔, ✅, ·, §, ⇒ — withhold nothing; each of those shapes is a pinned pair. In Chinese,
  Japanese or Korean prose written without spaces, a path joined to the text ("这个更新让README.md变得…",
  "snap/snapcraft.yaml를") lies in a wordish run, and one such mention withholds every claim on that path in the
  summary: on the EXTERNAL-1 shelf's real agent PRs (13,089 decided claims, the fourth review's run at `5ebe0b6b`),
  `extract` fired 7 times, caught no false verdict, and withheld 4 right and 3 undecided ones, all so.

By construction (A): over the 6,672 committed inputs (the pinned pairs, the 3,000 fuzz pairs regenerated in memory,
#161's reproductions and pair inputs, a seeded 2,000-pair PATH-2a fuzz and a seeded 1,000-input text-seam set), both
strict modes, every branch record is `main`'s but for abstentions in reach with the overlay's reason, each claim
reads the same with `--strict` as without it, and where `main` raises the branch raises the same exception — in
Python and in the port (13,344 port runs, 0 broken; the relation compares `unparsed_claims` and each claim's keys
too); the per-key abstention counts are *pinned* for both path flavours and for Unicode 13.0 and 14.0 (emulated by
re-reading the inputs with their five Unicode 14 letters mapped to code points no version assigns), 15.0, 15.1 (the
committed inputs hold no code point assigned in 15.0 or 15.1: the sixth integration review's inventory) and 16.0.
The PATH-2a test modules pass on CPython 3.12.10 at this head, and under real pytest 9.0.3 on 3.14.2 (409 passed, 1
skipped where terser is not on the path; 3.14.2 here has no pytest or numpy of its own, so 3.12's pure-Python packages
are put on `PYTHONPATH` and `styxx` is registered as a bare package by a `-p` plugin; up to the seventh pass a
stand-in runner was used there, which skipped every parametrized test). At this head (measured, CPython 3.12.10 and
Node 24.13.0): the sets of the seventh pass's head below again, and the seventh review's own — its line-break generator
`genxl` (4,000), its case-skew, emoji-window and DECLARE-1 sets (9,000, twice: as the runtimes stand, and with the port's
engine patched) and its hostile generator `fuzz7` at three seeds (9,000) — 171,000 inputs, 341,964 Python runs in both
strict modes and as many in the port (the other 36 raise in `main` and raise alike), 0 outside the relation, the error
phrase never; the minified bookmarklet in a stub page equal to this port on the hostile set's 18,000 runs (940 carrying
an overlay reason). At the seventh pass's head (measured, CPython 3.12.10 and Node 24.13.0): the sixth review's hostile generator at two seeds (40,000 inputs), its `hx` and
`gen7` generators (12,000 and 20,000), a mixed count, tests, symbol and scope seam set (20,000), the fifth pass's
count-seam set (20,000), the sixth review's case-pair generator `gen8` (10,000, twice: as the runtimes stand, and with
the port's engine patched to fold a later version's pair) and the builder's truth world under five decorations (8,000):
140,000 inputs, 279,986 Python runs in both strict modes and 279,986 in the port (the other 14 raise in `main` and
raise alike), 0 outside the relation, the error phrase never; the minified bookmarklet in a stub page equal to this port on 60,000 runs (10,230
of them carrying an overlay reason). At the sixth pass's head the review's own differential found 0 outside the
relation on 214,272 Python 3.12 runs, 155,678 Python 3.14 runs, 210,866 port runs, 181,626 bookmarklet runs and 45,200
real-git runs. At the fifth pass's head: the reviews' own generators (`gen4`
at two seeds, `gen`, `gen_tok`, `gen_real`; 34,000 inputs) and a count-seam generator (20,000 inputs over dot-twin
diffs), 108,000 Python runs in both strict modes and 108,000 in the port, 0 outside the relation, every raise the same
(the 238 runs where `main` raises), the error phrase never; the minified bookmarklet in a stub page equal to this port
on 85,310 runs. Under a Windows job-object memory cap the
fifth review's deep paths read as `main` reads them: at 256 MB a diff of 1,000 paths of 4 KB (8 MB) in 0.40 s, at 160
MB one path of 40 KB and 200 paths of 4 KB (at `5ebe0b6b`, under 1,024 MB, the 200-path diff withheld a right VERIFIED
with the error phrase). At the previous heads, against `main` reconstructed (sha-equal to `git show 1cde8b82`'s two
files) and, in the port, `main`'s own file from `git show`: the reviewers' hostile (two seeds), calm, lone-surrogate, cross-port, text-seam, realistic-text and token-adjacency fuzz and the previous pass's committed set — 203,606 inputs, each run in both strict modes in Python and without `--strict` in the port: 0 outside the relation, every raise the same, the error phrase never (CPython 3.12.10; 72,606 of them again on 3.14.2, 0) — and the rebuilt bookmarklet, loaded in a stub page, equals this port on 37,288 runs (18,644 inputs, both strict modes).

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
their work; a heuristic, which also pairs two different matches when one's value lies inside the other's: L1, where
`main`'s Python reads `..c.py` and `.c.py` and its port `..c.py` twice, is pinned as the one such pairing on the
committed inputs, its split expected; the sixth review's OM1, "Modıfied b/c.py." then U+FEFF and "`a/b/c.py` —
updated", where `main`'s Python reads `b/c.py` and its port `a/b/c.py`, is pinned too); and, on inputs whose claims all
pair so, the gate verdict without `--strict` (under `--strict` OM1 splits it: C-4's case). The position key (kind,
verdict, text) is a heuristic as well: in 4 of 180,000 fuzz inputs of the sixth review it paired two different matches
of one sentence ("Added 1 tests; Added 5 new tests." read as one claim each, different ones). Left-over claims that are not one match
(each port reading a different sentence, as in "Changed the parser in `docs/résumé/index.md`. Added
`docs/café.md`.") are not paired, and their decisions may differ, as the claims do. At this head, on the committed
inputs and on the 140,000 fuzz inputs above: 0 splits under the by-construction keys and under the measured keys (L1
and OM1 are cross-port cases, not committed inputs), and without `--strict` 0 gate splits where `main`'s gates agree,
whether or not every claim pairs (C-1 above; the fifth review's count reproductions X2-13, G1, G2 and G3 and the sixth
review's C1 inputs, which split the gates at `5ebe0b6b` and `495d2204`, are pinned cross-port cases with one gate
verdict each). At the fifth pass's head, on the committed inputs and 54,000 fuzz inputs: 0 splits under the
by-construction keys, 0 under the measured keys but L1's, and 0 gate splits where every claim pairs. At the heads
before it, on the committed inputs (CPython 3.12.10) and on 14 further sets (223,084 inputs: the three reviews' hostile, calm, lone-surrogate, cross-port, text-seam and token-adjacency fuzz, two realistic-text sets and the previous pass's committed set): 480,520 claims at the same position with the same (kind, verdict, detail), 471,482 with the same (kind, verdict, text), 531,195 matched across the lists, 23,171 left over as one match — 0 splits under any key, and 0 gate splits on the 180,943 inputs whose claims all pair (measured on those sets; the fourth review's count shapes, which none of them held, split the gates there, and its L1 split the one-match key).

The overlay does add gate disagreements where `main`'s two gate verdicts agree but some claim rows differ between
`main`'s ports (a claim one port reads and the other does not, or reads with another verdict): each port withholds a
claim its own `main` decides. On the committed inputs that is 5 inputs without `--strict` — #161's four f2 separator
cases (`path2:f2-a-vertical-tab-…`, `-form-feed-…`, `-line-separator-…`, `-paragraph-separator-…`) and
`path2:y2-a-changed-test-beside-a-created-bom-test-under-bare-hunks`, pinned by the test so a sixth fails. At the heads before it, on the 223,084 inputs above, 1,523 inputs without `--strict` and 614 with it, most of them hostile or token-adjacency input; on the 40,000 realistic-text inputs, 0 without `--strict` and 48 with it, while the overlay makes 373 of `main`'s 373 non-strict gate splits there agree. Nearly all of the 48 (26 of the 27 in one of the two sets) are inputs where one port's `main` reads a claim the other's does not (a path holding letters outside ASCII) and the overlay withholds it on that side.
That paragraph and the one before it describe the fifth pass's head. At this head, without `--strict`, C-1 makes the
overlay add no gate disagreement where `main`'s two gate verdicts agree: 0 on the committed inputs (the five above
included), 0 on the sixth review's hostile generator at two seeds (40,000 inputs; 179 and 195 at `495d2204`), 0 on a
mixed count, tests, symbol and scope seam set (20,000; 954 at `495d2204`), 0 on the fifth pass's count-seam set
(20,000); and, at the seventh pass, 0 on the sixth review's `hx` and `gen7` (32,000), 0 on the builder's world under
five decorations (8,000), and 0 on its case-pair generator `gen8` (10,000), where `fcd3ce6a` split 250 on CPython 3.12
and Node 24 and 309 with the port's engine patched to fold a later pair (C-1 above; a committed test asserts it on 1,500 such inputs with the
patched engine). At the eighth pass's head: 0 on all of those again; 0 on the seventh review's line-break generator
`genxl` (4,000), where `8eead84f` split 572 (the eighth pass's C-1; a committed test asserts it on 600 such inputs); 0
on its case-skew, emoji-window and DECLARE-1 sets (9,000, with the runtimes as they stand and with the port's engine
patched); and 0 on its hostile `fuzz7` (9,000). Under `--strict`, where a claim one
port's `main` reads alone is withheld on that port, the gates still split: the two committed inputs `p2a-seam:4242:335` and `:487` ("Modified lib/core.jsé", "Removed lib/core.jsé": only
the port's `main` reads `lib/core.js`, VERIFIED, and `extract` withholds it), pinned; 171 and 195 of the hostile
inputs (as at `495d2204`), 741 of the mixed seam set (690 at `495d2204`: `redefined` withholds one-port VERIFIED tests
claims there), 791 of the count-seam set (as at `495d2204`; at the seventh pass with a receipt kept, A-3), 2 of the 12,000
`hx` inputs and 34 of the 20,000 `gen7` inputs. Per set at the eighth pass's head (`8eead84f` in brackets where it
differs): `gen5` 174 and 199 of 20,000 each (171 and 195: the windows decide claims the whole-sentence rule kept, and
where one port's `main` reads a claim alone its VERIFIED may be withheld on that port), the mixed seam set 741 of 20,000,
the count-seam set 791 of 20,000, `hx` 2 of 12,000, `gen7` 34 of 20,000, `gen8` 0 of 10,000 on either engine, `genxl` 0
of 4,000, the case-skew, emoji-window and DECLARE-1 sets 4 of 9,000 on either engine, the decorated world 0 of 8,000,
and `fuzz7` 27 of 9,000; on the committed inputs the two pinned. An abstain-only rule cannot balance these: a decision
must read the same under `--strict` as without it, and withholding nothing in such a sentence would keep every false
VERIFIED of the path kinds in any sentence holding an accented path. The bookmarklet calls the gate without
`--strict`. Operator option O-9 (abstain in both ports wherever either port's reading fires) is not taken.

The static checks. `selfcheck_p2a_only_abstains` reads the Python block's source with `ast` and checks its stores and
its reads; it infers no type, so it does not prove that the block can only abstain — the relation tests over the
committed inputs and the large summaries are what pin the records — but it refuses every way to write the record that
it reads. The stores: every attribute store is `c.why = ...`, `c.verdict = "UNCHECKABLE"` or `g.verdict = (...) FAIL /
PASS` in `_p2a_abstain`, or `self.*` in `_P2aFacts.__init__`; no item store, mutating call or augmented store on what
may hold one of the record's containers (`g.claims`, a claim's `detail`), read to a fixed point through assignments,
`:=`, `for` and comprehension targets, `with ... as`, call results, the parameters at every call of the block's
functions and the facts' methods, and the facts' memo (the eighth pass's A-1: the seventh read only `name = x.detail`,
so `for cl in (g.claims,): cl.pop()` passed it, and, behind `len(todo) > 400`, every committed test; the large
summaries now run the relation in both ports too). The reads: no import; no flag group (`(?si)`, `(?mi:...)`, `(?-i:...)`) holding
`i`, `u`, `L` or `a` in any position; only names the block binds or a short list of builtins
and `main`'s names (`int` is refused by name: it reads CPython's table of decimal digits, so the block reads digits
from a fixed table); only a short list of attributes (so no dunder and no unlisted method); `re` only as
`re.<function>(<static pattern>)`; no strip or split without its characters; no `str()`, `repr`, `format`, `!r`,
`getattr` or `eval`. The port's token scan reads code with white space removed between tokens, refuses `normalize`
anywhere, `new String(`, `.match(`, `.search(` and `.matchAll(`, regex modifiers (`(?i:...)`, live in V8 13.6), and
`Number`, `parseInt`, `parseFloat`, `isNaN` and `isFinite` as words in code (each reads the engine's white space;
the port reads digits from a fixed table, as the Python does), allows `RegExp` only as `new RegExp(<one static
string>)` whose decoded value is checked, and refuses computed member access outside a short list of index
expressions and any call on a computed member. Since the eighth pass (A-2) it also scans the port's stores: every store
into a member by name (`x.y =`, `x.y += 1`, `x.y++`) is `c.why = ...`, `c.verdict = "UNCHECKABLE"` or `g.verdict =
(...) ? "FAIL" : "PASS"` inside `_p2aAbstain`; there is no `delete`; every store into a computed member and every call
of a mutating method (`push`, `pop`, `splice`, `shift`, `unshift`, `set`, `add`, `fill`, `delete`, `clear`, `reverse`,
`copyWithin`) is on a chain from a name the block made itself (an array literal, `new Map`, `new Set`, `_p2aFlags`, or
such a chain; no parameter; no binding mentioning `.claims` or `.detail`) that then reads only `[...]` and `.get(...)`,
and stores no `.claims` or detail; `Object` and `Proxy` are refused as words. It too types nothing and follows no
value through a parameter (the seventh review's plant, which appended a space to every claim's text behind
`todo.length > 400`, is refused by it and by the port's relation on the large summaries). Every plant the third, fourth and fifth reviews found is a committed
refused case. The sixth review's are too: the Python check refuses a named-character escape (`\N{...}`, which reads
the Unicode name table when the pattern compiles); the port's scan refuses `==` and `!=`, a unary `+` or `-` before a
name, a call, a bracket or a string, and the words `Math`, `Date`, `BigInt`, `DataView` and every typed-array
constructor, each of which converts a string with the engine's white space (`+"\u30003"` is 3); the block's
`Math.max`, `Math.min` and `Uint8Array` became conditionals and plain arrays. These checks are static and infer no
types (an f-string of a list would read `repr` and pass; a binary `*`, `-`, `%` or relational operator on a string, and
the numeric parameters of built-in methods, convert with the engine's white space too, and pass — a committed test
pins that the scan accepts `claimed * 1`, `claimed - 0` and `claimed < 3`); the block gives them numbers only.

Cost per call (CPython 3.12.10; at the fifth pass's head, least and median of seven alternating rounds): the
committed inputs 0.281 → 0.346 ms (+23%), `corpus_fuzz.json` 0.177 → 0.210 ms (+19%), `corpus_real.json` 6.99 → 6.98
ms, inside the rounds' spread (6.98 to 7.38 ms on `main`); Node 24.13.0: +24% on the committed inputs. At this head, one
`path2a_recall.py` run (not alternated, so noisier): `corpus_fuzz.json` 0.197 → 0.228 ms (+16%), `corpus_real.json`
8.76 → 8.33 ms. Import: `import styxx.diffgate` takes about 22 to 31 ms more self time than `main`'s with bytecode
cached (6 to 11 → 28 to 42 ms, five runs each at this head; about 37 ms more without at the fifth pass's), on every
CLI, hook and Action start, most of it the block's regex classes compiled at import.

The worst shapes the reviews found are bounded by committed timing tests that time the overlay alone on `main`'s
record, the least of three runs, in both ports: a `def` beside a 5,000- or 50,000-code-point run of letters outside
ASCII, one 220 KB added line of `def test_a`, 2,000 files with 200 path claims under 50 directories, 2,000 files
that all share the base name `__init__.py` with 200 path claims, 500 symbol claims over 50,000 removed `def` lines,
300 `class` claims beside NBSP runs, four 64 KB summaries whose path, symbol, scope and count claims each meet
thousands of runs or zones, one path of 40 KB and 100 paths of 4 KB, 3,000 directories or base names outside ASCII
under 3,000 to 4,369 claims, 4,000 distinct ASCII claims over 3,000 files of one base name, and a run of 40,000 `./` —
each within the larger of 0.5 s in Python (0.3 s here) and five times `main`'s own call on the same input, the least of
three each (the eighth pass's I-1: an absolute bound alone moves with the runner, and CI's CPython 3.9 and 3.10 lack the
specialising interpreter), and under 64 MB of peak memory in Python (`tracemalloc`). Measured the
test's way at this head (CPython 3.12.10): 3,000 distinct claims over a base name outside ASCII 0.157 s (`main`'s call
0.097 s), the symbols case 0.151 s (0.052 s), 4,000 distinct ASCII claims over one base name 0.119 s, the rest 0.1 s or
less; on 3.14.2 the slowest 0.137 s; peak memory 14.2 MB at most. Against `main`'s own call the overlay reaches ×32 on
the run of 40,000 `./` and ×6 to ×7 on a `def` beside 50,000 letters outside ASCII, where `main`'s call takes a
millisecond and the absolute floor is the bound that applies, and at most ×4 elsewhere. In Node the slowest case (the
symbols) takes 59 ms, and the largest ratio is ×8.8 on the same `def` case. At `5ebe0b6b` the overlay took 0.50 s
and 766 MB on the one 40 KB path, 6.1 s on 4,369 claims over 3,000 directories and 19.3 s on 3,000 distinct claims over
a base name outside ASCII.

The overlay reads the summary's runs and zones once for all the claims' tokens (the sixth review's A-1: at `495d2204`
each distinct count, path, name or scope prefix scanned the summary again, so the cost grew as claims times summary
length). Three more timing tests bound it on summaries of 1.2 MB with 10,000 distinct count, path and scope claims
(3.6 MB and 30,000 in the port), relative to `main`'s own call on the same input: the overlay alone within 1.5 times
`main`'s call in Python (within it at the sixth pass; the seventh review's I-1: the overlay's loops are Python where
`main`'s are mostly C regex, so a slower interpreter moves the overlay more, and CI's 3.9 to 3.11 were not available
here) and within three times it in the port. Whole calls, least of two, `main` → `495d2204` → the sixth pass's head:

| input | Python | Node |
|---|---|---|
| 10,000 counts, 1.2 MB | 0.68 → 1.72 → 0.83 s (×1.23) | 97 → 367 → 176 ms (×1.8) |
| 10,000 paths, 1.2 MB | 0.79 → 3.83 → 1.08 s (×1.37) | 121 → 541 → 292 ms (×2.4) |
| 10,000 scopes, 1.3 MB | 0.70 → 5.23 → 1.05 s (×1.50) | 124 → 550 → 286 ms (×2.3) |
| 30,000 counts, 3.6 MB | 1.99 → 10.58 → 2.55 s (×1.28) | 302 → 2,906 → 600 ms (×2.0) |
| 30,000 paths, 3.6 MB | 2.32 → 26.51 → 3.51 s (×1.51) | 452 → 4,134 → 1,127 ms (×2.5) |
| 30,000 scopes, 3.9 MB | 2.12 → 38.93 → 3.19 s (×1.51) | 426 → 3,565 → 902 ms (×2.1) |

The overlay alone there: 0.18 to 0.34 s and 0.55 to 1.13 s in Python, 65 to 142 ms and 235 to 605 ms in Node; it grows
with the input, not with claims times the summary. At this head (CPython 3.12.10, least of two): 10,000 counts 0.67 →
0.84 s (×1.25), paths 0.81 → 1.04 s (×1.29), scopes 0.70 → 1.04 s (×1.48); the overlay alone 0.23 to 0.49 of `main`'s
call in one run (Linux CPython 3.12.3 read 0.35 to 0.65 at the sixth pass's head). In Node, at three times the size,
0.92 to 1.34 of `main`'s call. At the eighth pass's head, the overlay alone: 0.21 to 0.47 of `main`'s call on CPython
3.12.10 and 0.19 to 0.32 on 3.14.2, and 1.01 to 1.43 in Node at three times the size (one run each, the test's way).
Such ratios move by about a tenth from run to run: the seventh pass's head read 0.23 to 0.49 in one run, its test's
docstring 0.27 to 0.56 on 3.12.10 and 0.22 to 0.40 on 3.14.2, and the seventh review 0.25 to 0.56 and 0.17 to 0.36
(the eighth pass's I-4).

Peak memory on summaries of many distinct long tokens (the seventh review's A-1). At `fcd3ce6a` one matching automaton
held every named token whenever more than 32 were named, even over an empty text: the review's 10,000 path claims of
60-character directories (0.82 MB) took 157 MB at the overlay's peak in Python, where `main`'s call takes 8.5 MB, and
the port aborted under a 64 MB heap (the review's cap was 100 MB) where `main`'s port needs less than 16 MB. Now the
text is read as its distinct pieces between the characters a token never holds, an empty text builds nothing, and one
automaton holds at most 65,536 characters of tokens (or a sixteenth of the text): 12.1 MB in Python, and under a 24 MB
heap in Node. Two committed tests bound it: 3,000 distinct 110-digit counts over a dot twin, with an empty text and with
each number in a wordish run, within 32 MB in Python (0.4 and 16.5 MB; `main`'s call 2.6 and 2.8 MB; `fcd3ce6a` 68.1
and 68.5 MB), and the counts at 6,000 with the review's input under a 64 MB heap in Node, for `main`'s port and this
one. The port also builds its key forms as one joined string: appended code point by code point, a 70-character path
was a chain of pieces V8 keeps until it is read whole, and 10,000 such claims retained 38 MB.

A whole call on a large diff costs more than the corpus figures suggest, because the overlay reads the diff again (the
line views, the lone-CR pieces, the removed definitions), linearly. On the fourth review's large inputs, the least of
three whole calls, `main` → here (CPython 3.12.10 / Node 24.13.0):

| input | Python | Node | peak memory (Python) |
|---|---|---|---|
| 20,000 files, CRLF, 600 claims (3.1 MB) | 18.3 → 18.7 s (×1.02) | 875 → 1,136 ms (×1.3) | 22 → 53 MB |
| 200,000 content lines with U+2028, NEL, VT and FF (3.3 MB) | 0.19 → 0.97 s (×5.1) | 63 → 326 ms (×5.2) | 22 → 49 MB |
| a 1.2 MB summary over 5,700 files | 10.6 → 14.2 s (×1.3) | 1,263 → 2,531 ms (×2.0) | 25 → 43 MB |
| one 5.8 MB line of `def` tokens | 0.10 → 0.44 s (×4.4) | 11 → 80 ms (×7.1) | 19 → 69 MB |
| 5,000 duplicated claims over a DECLARE-1 block | 0.30 → 0.32 s (×1.1) | 160 → 176 ms (×1.1) | 6 → 6 MB |

The timing tests bound the shapes above, not these whole-call factors; the 35 MB diff of earlier heads was not
measured again. The sixth review measured more line-heavy shapes at `495d2204` (least of two, whole calls): a
1,000,000-line `+a`/`-a` diff ×2.35 in Python; `-a\rb` lines ×2.82; 150,000 lines of removed definitions with CR
pieces and backslash continuations (3.3 MB) ×5.58 in Python and ×6.15 in Node; one 2.8 MB line of U+2028-separated
`def test_a():` sites ×3.46 in Python and ×8.2 in Node (×9.3 at a third of the size); peak memory on the 1,000,000-line
diff 61 → 112 MB, and 69 → 175 MB with VT in the added lines, since both line views are built. The seventh review
measured one more at `fcd3ce6a`: a single added line of U+2028-separated ` def test_x(): pass` segments (a space after
each U+2028) under "Added 1 tests.", ×13 to ×17 in Node and ×7 to ×8 in Python, linear in the size (from 3.2 → 55.6 ms
at 50,000 segments to 20.2 → 314.8 ms at 400,000 in Node). The eighth review measured more at `8eead84f`: one added
line of 200,000 to 480,000 pieces joined by NEL and VT (or FF and U+001C) before `def test_` sites, up to 9.6 MB, ×19.8
to ×27 in Node and ×4.2 in Python, linear in the size. The ranges to expect on such diffs are up to about ×27 in Node
and ×8 in Python, and about 2.5 times the peak memory in Python; the second line view is still built eagerly. The
bookmarklet grew from 24,335 (`main`) to 56,575 characters (49,888 at the seventh pass; the eighth adds the case
tables, 184 runs, and the windows).

## What this is not

It is not a second instrument. The Python is the instrument; the JavaScript exists only where the
Python cannot run, and the day the two disagree on a pair in this corpus, the JavaScript is wrong
by definition. It is not a check of anything but transliteration fidelity — the template set's
own precision was measured elsewhere (EXTERNAL-1: path accusations at 0.23 against a 0.95 floor
on 71,016 agent-authored PRs, and withheld since), and no number here bears on that.
