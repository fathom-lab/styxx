# web/gate — the diff gate where Python cannot run

The instrument is `styxx/diffgate.py`. Two browser surfaces cannot import it: the paste-in
preview page and the bookmarklet. They run `diffgate.js`, a JavaScript transliteration of one
specific file — `styxx/diffgate.py` as it stands on `main` (BC-2 + COMPAT-1 + BIN-2 + COMPAT-2 +
PATH-1 + DECLARE-1, pull requests #113, #115, #120, #124, #127 and #129, plus the `fetch_pr` door),
re-cut for the PATH-2 repairs
(`papers/closed-model-frontier/PREREG_path2_resolution_2026_09_17.md`: a path claim resolves exact,
then suffix, then basename, #97; a dotfile keeps its dots in the path key, #121; a `def` the same
file's removed lines also define is changed, not added, #101; as amended by
`AMENDMENT_path2_resolution_2026_09_17.md`: definitions pair one to one per name, `only_touches`
lists only the paths outside by more than a dot, COMPAT's scaffold reading keeps `.storybook/` as
scaffolding; and by `NOTE_path2_third_pass_2026_09_25.md`, `NOTE_path2_fourth_pass_2026_09_25.md` and
`NOTE_path2_fifth_pass_2026_09_25.md`) on the file that carries them,
sha256 `0a5522ebdec4c16890070f79920f04f48f4b6a439c0086a01bbf85c1b9c2ec2b` (LF line endings; a wheel
built on Windows carries CRLF and hashes differently, so `py_side.py` normalises before it
compares) — and this directory is the receipt for that port: the differential test that holds
it to the Python's output, and the build that turns it into the bookmarklet people drag into
their bookmarks bar.

That gap is closed. Until #126 the port lacked COMPAT-2's sharpened compatibility reading (surface
vs scaffolding, signature changes, the candidate flag) and the differential counted 10 `compat_claim`
records as disagreements. The port carries that reading now, and the run below reads 0.

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
`--check` rebuilds all three in memory and compares them with the files on disk, byte for byte, and
writes nothing; every output is written with LF. `.gitattributes` marks `bookmarklet_src.js` `-text`,
so a checkout keeps those LF bytes. Before that line existed, `--check` failed on a stock Windows
checkout (`core.autocrlf=true`) with the committed blob correct: git rewrote the source to CRLF and the
byte comparison reported `bookmarklet_src.js … DIFFERS`, exit 1 (NOTE_path2_fourth_pass_2026_09_25,
B-1). Measured after the line, on this Windows checkout, with terser 5.46.0: all three `matches`,
exit 0. The shipped bookmarklet is

    bookmarklet.min.js    sha256 d8ce5111990c4e4d4a17c89fd964621f339ac8b7782ee2c62643300fccab9dca   27,738 chars
    bookmarklet.href.txt  sha256 3886387a7f8c1c559afe6f21006e1a17d43ae4c14c1b9a7b6b3e1ca00b94c88e   27,749 chars

(Earlier builds: `b04d14dc…`, 11,437 chars, from the 7.47.0 file — accuses outside Python;
`9ea8f572…`, 17,686 chars, the BC-2 + COMPAT-1 re-cut — cannot see a binary file; `4b2d34e1…`,
19,002 chars, the BIN-2 re-cut — resolves a path claim to an earlier basename match, keys a
dotfile without its dot, counts a changed `def` as added; `22c31746…`, 20,255 chars, an unmerged
PATH-2 cut — one changed test cancels every same-named new one, and `only_touches` abstains when the
dot is on the prefix; `af252434…`, 26,236 chars, the unmerged third-pass cut — reads "only touches
.gitignore" as not a path, counts an NBSP re-indent of a test as an added test, and reads a line
holding U+000B, U+000C, U+2028 or U+2029 differently from the Python; `5b6f3167…`, 26,691 chars, the
unmerged fourth-pass cut — reads a form-feed re-indent of an existing function as an added one, does not
count a new test led by a form feed, and reads a COMPAT line holding U+0085, a header path followed by
U+0085 and a printed path differently from the Python. A bookmark that hashes to any of them is an old
port; drag the new one.)

Whatever a browser holds under that bookmark either hashes to the line above (drop the
`javascript:` prefix) or is not this build. terser 5.46.0 produced these bytes; the same terser
rebuilds the `4b2d34e1…` bookmarklet byte for byte from its sources, which terser 5.51.2 produced.
`build_bookmarklet.py`'s docstring names the same version.

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
    python py_side.py                    # refuses to run unless styxx/diffgate.py hashes to 0a5522eb…
    node js_side.js
    python differential.py
    node check_pairs.js                  # the 161 pinned pairs against their expect blocks

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

Result, 2026-09-25, this branch after the fifth pass (`NOTE_path2_fifth_pass_2026_09_25`), on `main`
`98a5c368`:

    3337 pairs, 7094 claims (671 verified, 1571 contradicted, 4852 uncheckable) — 0 disagreement(s)
    161 pinned pairs, 0 disagreement(s)

**Zero is a count over this corpus, not a property of the two implementations.** The third pass reported
the same zero while the two returned opposite `tests_added` verdicts on any re-indent by U+000B, U+000C,
U+2028 or U+2029, and the fourth pass reported it while they disagreed on `compat_claim` lines holding
U+0085 and on a symbol name followed by a non-ASCII letter; no record in the corpus carried such a line.
The same 3,337 pairs with the fourth-pass head on BOTH sides read **17** disagreements, and with `origin/main`
on both sides **27**. The fifth pass enumerates the definition-line class instead of sampling it: every
character of Python's `\s` and U+FEFF leading an added or removed test or symbol definition, between `def`
and the name, and after the name, plus six name-end characters that are not spaces — 312 inputs, through
`gate_diff_text` and the port on one text, and through `gate_diff` on a real repository and the port on the
bytes git printed. `origin/main` disagrees on 40 of them on each door, the fourth-pass head on 4 (the `\b`
case), this branch on **0** on both; the 300 raw-door inputs are a committed test
(`tests/test_diffgate_path2.py::test_v1_the_port_reads_the_grid_as_the_python_does`), so that zero is a
receipt rather than a number typed once.

**What still disagrees, counted and not repaired.** The claim templates read the DESCRIPTION with
JavaScript's `\s`, `\w` and `\b`, not Python's. One whitespace character between the words of twelve claim
shapes (Python's 29 and U+FEFF, 360 inputs): **72** disagreements, exactly the six characters whose
membership differs (U+001C–U+001F and U+0085 are whitespace to Python, U+FEFF to JavaScript) in all twelve
positions. A non-ASCII letter or mark inside a claimed name or path (20 inputs): **12**. Both counts are the
same on `origin/main` and at the fourth-pass head. On the diff side the fifth pass closed two classes: a
`---`/`+++` path followed by one of 28 such characters (6 disagreements on
`main`, 0 now) and a reason printing a path that holds one (25 on `main`, 20 at the fourth-pass head, 0
now). Below any test sits the engines' Unicode version: Node 24's `\p{L}\p{N}` reads 5,004 code points
assigned after Unicode 15.0 as word characters where Python 3.12 does not, and 27 code points lower-case
differently.

Against `origin/main`, PATH-2 moves **185** of the 3,337 records in the Python and **194** in the port (the
difference is pinned pairs on which the two disagreed on `main`); 111 of them lie outside the pinned-pair
files. The fifth pass itself moves, of the 3,301 records that predate it, the four pinned pairs it re-pins
(V-1: a form-feed re-indent now pairs as a changed test, a U+2028-led `def` defines nothing, and two
changed generic definitions pair and abstain) and **90 fuzzed records**, 92 claims CONTRADICTED →
UNCHECKABLE: the fuzzer writes `docs/.` followed by a sentence period, and V-4 reads a prefix written to
end in `..` as the parent it spells. No real-corpus record moves.

`path2_gates.py differential` now attributes every moved claim by counterfactual: its own copy of the
repaired instrument, with one rule reverted, must give the baseline claim back (NOTE_path2_fifth_pass,
V-3). From a clean tree at `7f6d9303` (the commit
before this README's; it touches no file the scorer's provenance reads): exit 0, every gate passes, no
violation; 185 records moved; claims attributed to #97 27, #121 35, #101 26, F-2 18, F-3 4, V-1 3, V-4 93,
jointly #101+R-1 4, #101+V-1 4, F-2+V-1 4, #121+V-4 2; new accusations 4 `only_touches` (the amendment's
dotted-prefix exception), 2 `symbol_added` (V-1: definitions CPython refuses) and 4 `tests_added` (F-2,
F-3, V-1), each counted in the payload; 6 `compat2_candidate` flips, all False → True and all admitted
because F-2 alone explains them; 4 F-4 withdrawals. Scorer `522b1e87…`, harness `75bfbc39…`, repaired
`diffgate.py` `0a5522eb…`, baseline `98a5c368`. The gate can fail: on a synthetic three-PR shelf, with a
scratch copy of this `diffgate.py` carrying the round-3 blocker swapped in, one record touching only
`.github/` with a stray U+2028 in an unrelated context line, `path2_gates.py corpus` exits 1 with
`G-C4_direction:only_touches`; the fourth-pass scorer (`60d678a5`) admitted that move under F-2 and
exited 0. With the instrument unmutated the same shelf exits 0.

The earlier result, kept as it was measured: at the fourth pass, 3301 pairs, 7056 claims (668 verified,
1652 contradicted, 4736 uncheckable), 0 disagreements; 125 pinned pairs, 0; with the third-pass head on both
sides 9, with `origin/main` 11; a 242-input whitespace grid read 1 disagreement (the `\b` case, closed by
the fifth pass). Its `path2_gates.py` run (exit 0, 69 records moved) used the record-wide F-2 attribution
the fifth pass replaced. At the third pass: 3282 pairs, 7028 claims (653 verified, 1640 contradicted, 4735
uncheckable), 0 disagreements.

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

`path2_pairs.json` carries the PATH-2 pairs — two README files in either order, a suffix match
beating an earlier basename, an exact match beating an earlier suffix, the basename fallback, a
dotfile and its undotted twin, "only touches github/" over `.github/` (abstains on the dot), two
prefixes and leading slashes, a dotfile beside a nested undotted name, a dotfile outside the
prefix, the #101 issue diff, a new test beside a changed one at three claimed counts, a changed
`def` beside a fresh one elsewhere, a rename, a test moved between files (counts as added, the
disclosed limit) and counted cases over a changed test; and, for the amendment, the same test name
in two classes, a changed test beside a same-named new one, the shelf's fold under an `A` and an `M`
header, a BOM strip on a test and on a function, non-ASCII test names, a non-ASCII suffix, a
same-named method added in another class and one changed in place, a changed `def` under an `A`
header, a generic `def`, a dot miss beside a real outside path (one and two prefixes), a dotted
prefix over an undotted file and directory, a `..` path, binary dotfile twins with no hunks, a pure
rename to a dotted name, a file named `.py`, and the `.storybook/` COMPAT-2 reading; and, for the fourth
pass, a slashless dotted prefix whose suffix the extension list does not hold in the three positions it
can take (VERIFIED, a real outside path, C-3's accusation), a test re-indented by U+000B, U+000C,
U+2028 and U+2029, a line separator inside an added line, inside a context line and before a
header-shaped fragment, a form-feed indent that does define a function and a line-separator indent the
symbol test still read as one (a disclosed limit, pinned so it could not move unseen; the fifth pass
re-pinned it, and the line now defines nothing), an NBSP and an
ideographic-space re-indent, and an off-tree prefix beside an on-tree one read three ways; and, for the
fifth pass, the definition-line class (a form-feed re-indent of a function, a form-feed-led new test, a
vertical-tab re-indent, NBSP-, U+0085-, U+001C- and U+FEFF-led definitions, a definition after a
mid-line U+2028, a form-feed separator, a changed generic class, a name followed by a non-ASCII letter
or a middle dot, an added `async def` alone and beside a changed `def`), the port's COMPAT reading in
every language with U+0085 in a `\s` position and the U+001C, U+001F, U+FEFF, `\w` and `\b` cases, a
header path's strip, binary headers holding U+2028, `repr()` of a printed path, a prefix written to end
in `..` or holding `..` after a named segment, and F-4's two boundaries — also checked on
the Python side by `tests/test_diffgate_path2.py`. Their `expect` blocks are the Python's output, written down so a
reader can see the intended readings without running anything. The 3,199 pre-BIN-1 records are
byte-identical before and after that repair (no binary in the corpus), which is its G-BIN-2.

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

## What this is not

It is not a second instrument. The Python is the instrument; the JavaScript exists only where the
Python cannot run, and the day the two disagree on a pair in this corpus, the JavaScript is wrong
by definition. It is not a check of anything but transliteration fidelity — the template set's
own precision was measured elsewhere (EXTERNAL-1: path accusations at 0.23 against a 0.95 floor
on 71,016 agent-authored PRs, and withheld since), and no number here bears on that.
