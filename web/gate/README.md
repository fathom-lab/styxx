# web/gate — the diff gate where Python cannot run

The instrument is `styxx/diffgate.py`. Two browser surfaces cannot import it: the paste-in
preview page and the bookmarklet. They run `diffgate.js`, a JavaScript transliteration of one
specific file — `styxx/diffgate.py` as shipped in the **styxx 7.47.0** wheel, sha256
`fb2d9b3e8426650bc20fc8613c9bcad16c1dcd86b85b0ca8c532fdd2a23e7304` — and this directory is the
receipt for that port: the differential test that holds it to the Python's output, and the build
that turns it into the bookmarklet people drag into their bookmarks bar.

(The `/gate` page on the site is different: it runs the real 7.47.0 Python under CPython 3.13 via
Pyodide, with the two module files hashed in the browser against the wheel. No port there.)

## What is here

`diffgate.js` — `gateDiffText(summary, diff, {strict})` and `parseUnifiedDiff(text)`, the same
closed template set, the same verdict strings, the same `why` text. Two deliberate gaps, both
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

    bookmarklet.min.js    sha256 b04d14dc6ecca983add8053d46c548bb07479a452c8ba58613a82d5da1912aa6   11,437 chars
    bookmarklet.href.txt  sha256 e2c74b1571e097edf74bd05f505314fb649049289adbf20ea3b6236dc24f22c2   11,448 chars

Whatever a browser holds under that bookmark either hashes to the first line (drop the
`javascript:` prefix) or is not this build. terser 5.51.2 produced these bytes.

`differential/` — the test. Read on.

## The differential test

A port is held to the original by output, not by trust. `differential/` builds a corpus of
(summary, diff) pairs, runs the released Python and the JavaScript over every pair, and compares
the records field by field: verdict, measured, why_unmeasured, sentences_total,
uncovered_sentences, the never-read list in order, and every claim as
(kind, verdict, why, text, detail) in order. A port that gets the verdict right for the wrong
reason, or reads one sentence more or less, is a disagreement.

    pip install styxx==7.47.0            # the file the port was made from; py_side.py checks the hash
    cd web/gate/differential
    python build_corpus.py               # 176 real pairs, pinned to shas (below)
    python fuzz_corpus.py                # 3,000 synthetic pairs, seeded
    python py_side.py                    # refuses to run unless styxx/diffgate.py hashes to fb2d9b3e…
    node js_side.js
    python differential.py

Result, 2026-09-16, from a fresh clone of `fa8bcde7`:

    3176 pairs, 8904 claims (1024 verified, 2172 contradicted, 5708 uncheckable) — 0 disagreement(s)

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

## Drift: main is ahead of the port

`main` at `fa8bcde7` is 283 commits past v7.47.0 and its `styxx/diffgate.py` (sha256
`4e6f6550…`) is not the file the port was made from. `python py_side.py --working-tree` runs
the checkout's module instead and is expected to disagree. Measured the same day:

    3176 pairs, 6860 claims (633 verified, 2172 contradicted, 4055 uncheckable) — 4945 disagreement(s)

What moved, and what did not: every pair gets the same PASS/FAIL under both; the set of
CONTRADICTED claims is identical (2,172, same kind, same text, same pair); the movement is
entirely in `file_touched` — main reads 2,044 fewer of them (2,770 → 1,117 UNCHECKABLE,
671 → 280 VERIFIED), the V13/V14 repairs declining bare and ambiguous path mentions the 7.47.0
templates still took as claims. The port will be re-cut from the next release and re-run through
this harness; until then `diffgate.js` is the release, not main, and says so in its header.

## What this is not

It is not a second instrument. The Python is the instrument; the JavaScript exists only where the
Python cannot run, and the day the two disagree on a pair in this corpus, the JavaScript is wrong
by definition. It is not a check of anything but transliteration fidelity — the template set's
own precision was measured elsewhere (EXTERNAL-1: path accusations at 0.23 against a 0.95 floor
on 71,016 agent-authored PRs, and withheld since), and no number here bears on that.
