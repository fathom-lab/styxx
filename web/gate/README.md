# web/gate — the diff gate where Python cannot run

The instrument is `styxx/diffgate.py`. Two browser surfaces cannot import it: the paste-in
preview page and the bookmarklet. They run `diffgate.js`, a JavaScript transliteration of one
specific file — `styxx/diffgate.py` at the BC-2 + COMPAT-1 + BIN-1 checkout (pull requests #113,
#115 and the #118 repair, the file **7.48.0** ships once they merge), sha256
`397624d583edc3a147c74bf8791e5356f26a946c7f905d851e453b5297dc40a1`, re-cut for the PATH-2
repairs (`papers/closed-model-frontier/PREREG_path2_resolution_2026_09_17.md`: a path claim
resolves exact, then suffix, then basename, #97; a dotfile keeps its dots in the path key, #121; a
`def` the same file's removed lines also define is changed, not added, #101; as amended by
`AMENDMENT_path2_resolution_2026_09_17.md`: definitions pair one to one per name, `only_touches` lists
only the paths outside by more than a dot, COMPAT's scaffold reading keeps `.storybook/` as
scaffolding) on the file that carries them, sha256
`d9f8ddd58841d875287dd636f722965cc66424885bdcd0e07f171a6fb6e16dbf` (LF line
endings; a wheel built on Windows carries CRLF and hashes differently, so `py_side.py` normalises
before it compares) — and this directory is the receipt for that port: the differential test that
holds it to the Python's output, and the build that turns it into the bookmarklet people drag into
their bookmarks bar.

One known gap, measured below: that file also carries COMPAT-2's sharpened compatibility reading
(surface vs scaffolding, signature changes, the candidate flag), and the port does not. Its
`compat_claim` reasons and detail are COMPAT-1's, the verdict (always UNCHECKABLE) is the same, and
the differential counts every such record as a disagreement: 10 on the corpus below, and nothing else.

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
writes nothing; every output is written with LF. The shipped bookmarklet is

    bookmarklet.min.js    sha256 b7123d36f815b58c2993ba40f235ca9f73be53b9c8f26c23a44ba12cd04095d7   20,646 chars
    bookmarklet.href.txt  sha256 ecbe5a1a9ab899c324b58d67f075d7f37344fb3b5207ea6acc0501547be0923f   20,657 chars

(Earlier builds: `b04d14dc…`, 11,437 chars, from the 7.47.0 file — accuses outside Python;
`9ea8f572…`, 17,686 chars, the BC-2 + COMPAT-1 re-cut — cannot see a binary file; `4b2d34e1…`,
19,002 chars, the BIN-2 re-cut — resolves a path claim to an earlier basename match, keys a
dotfile without its dot, counts a changed `def` as added; `22c31746…`, 20,255 chars, an unmerged
PATH-2 cut — one changed test cancels every same-named new one, and `only_touches` abstains when the
dot is on the prefix. A bookmark that hashes to any of them is an old port; drag the new one.)

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
    python py_side.py                    # refuses to run unless styxx/diffgate.py hashes to d9f8ddd5…
    node js_side.js
    python differential.py
    node check_pairs.js                  # the 67 pinned pairs against their expect blocks

Result, 2026-09-17, the checkout at the amended PATH-2 repairs:

    3243 pairs, 6980 claims (636 verified, 1628 contradicted, 4716 uncheckable) — 10 disagreement(s)
    67 pinned pairs, 0 disagreement(s)

The ten are `compat_claim` records, every one COMPAT-2's reason and detail against the port's
COMPAT-1 reading: eight pinned compatibility pairs and one commit message, which disagree byte for
byte the same way before PATH-2, and the PATH-2 pair pinned for COMPAT-2's `.storybook/` reading,
which cannot agree while the port lacks COMPAT-2 (its `expect` block pins kind and verdict for the
port; the Python side pins the reason and the candidate flag). Nothing else disagrees.

Before PATH-2, `py_side.py` cannot be run at `87dded26`: its pin `397624d5…` is not that checkout's
file `473a7dd7…`. The before-figure is measured with the same loops over `git show
87dded26:styxx/diffgate.py` and `git show 87dded26:web/gate/diffgate.js`, on the same 3,205 pairs:
3205 pairs, 6914 claims (600 verified, 1612 contradicted, 4702 uncheckable) — 9 disagreement(s), the
nine above. PATH-2 moved 21 of those 3,205 Python records, and the port moved the same 21, all path
claims: 20 on diffs that carry two files with one basename (#97, among them `edge:6`, the two-README
diff the issue was filed with, now VERIFIED) and one PR description whose
`.github/workflows/diffgate.yml` now prints with its dot (#121). No `tests_added`, `symbol_added`,
`only_touches` or `compat_claim` record among them moved: the fuzzer never writes a removed `def`
line or a dotfile, which is why `path2_pairs.json` exists.

Before PATH-2, the checkout at the #118 repair: 3205 pairs, 6914 claims (600 verified, 1612
contradicted, 4702 uncheckable) — 0 disagreements, against the file before COMPAT-2 merged.

The 3,205 are the 3,176 below plus 29 pinned pairs committed as JSON (the `.gitignore` here
ignores generated JSON and names these three as exceptions): `bc1_pairs.json`, the four pairs BC-2
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
and `path2_pairs.json`, thirty-eight for PATH-2 — two README files in either order, a suffix match
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
rename to a dotted name, a file named `.py`, and the `.storybook/` COMPAT-2 reading — also checked on
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
left out of these counts; the #118 repair changes none of the 3,176 records, so the figures
below are unchanged by it): the wheel makes 573 accusations the port does not — `symbol_added`
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
