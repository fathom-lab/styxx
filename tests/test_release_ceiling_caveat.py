# -*- coding: utf-8 -*-
"""The headline AUC triple may not appear in shipped copy without the ceiling.

`0.998` / `0.976` / `0.943` are register-detection figures at a construct ceiling.
Quoted bare -- in a launch thread, a funding note, a submission to a standards body
-- they read as a claim about honesty detection that this lab's own published work
bounds: the same construct scores 0.498 on text-only deception
(`papers/THESIS_the_honesty_standard_2026_05_31.md`, and the scope erratum at the top
of `papers/every-mind-leaves-vitals.md`).

This guard is mechanical and narrow. It does not read meaning; it checks that the
word "ceiling" is nearby, or that the file carries the dated erratum block. It is a
tripwire against a *new* bare figure, not a proof that existing copy is well caveated.

Two clauses, either one satisfies a figure:

  1. the word "ceiling" appears within +/-3 lines of the figure, or
  2. the dated ceiling erratum (`ERRATUM_SENTINEL`) appears earlier in the file.

Clause 2 exists because the errata were *prepended* to already-written documents
whose original text is preserved verbatim and must not be edited to satisfy a test.
Its cost is stated rather than hidden: once a file carries the erratum, this guard
stops policing figures added to it later. Clause 1 is what covers new prose.

BOUNDARY MATCHING -- read before touching `_FIGURE`. The right-hand guard is
`(?!\\d)` and nothing more. `styxx/certify.py`'s `_NUM` uses `(?![\\w.])`, which
means a decimal followed by a sentence-ending period never matches at all: not
abstained, not accused, absent from the ledger (89 of 208 certified documents,
168 tokens). `test_figure_regex_matches_a_figure_that_ends_a_sentence` pins the
distinction so this guard cannot acquire the same blind spot.
"""
import pathlib
import re
import subprocess
import urllib.parse

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
SCANNED_DIRS = ("release", "docs")

FIGURES = ("0.998", "0.976", "0.943")

# Right guard is (?!\d) ONLY -- see BOUNDARY MATCHING in the module docstring.
# Left guard rejects a longer decimal ("10.998", "0.9765") but nothing else.
_FIGURE = re.compile(
    r"(?<![\d.])(?:" + "|".join(re.escape(f) for f in FIGURES) + r")(?!\d)"
)

CAVEAT_WORD = "ceiling"
CONTEXT_LINES = 3

ERRATUM_SENTINEL = "Ceiling erratum (2026-09-01)"

# Receipts are records of a measurement that was taken. They are never rewritten,
# so they are never asked to carry a caveat. Excluded by name, not by guesswork.
RECEIPT_SUFFIXES = (
    "_result.json",
    ".certificate.json",
    ".seal.json",
    ".capsule.html",
)

TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".html", ".htm", ".rst", ".json", ".yaml", ".yml"}


# ── scanning ────────────────────────────────────────────────────────────────


def _tracked_paths():
    """Every path git tracks, or None outside a checkout.

    THE POPULATION IS WHAT THE REPOSITORY IS. A walk of the filesystem also finds untracked and
    git-ignored artifacts — a stale release bundle, a scratch copy — and polices the lab for prose
    it never published. `styxx/corpus_audit.py` names this defect in its own comments, after a
    stale worktree made 49% of a corpus census phantom: the population was defined by what a glob
    matched rather than by what the thing is. It bit here too: an ignored release bundle in one
    checkout failed this guard while CI, which clones, passed.
    """
    try:
        r = subprocess.run(["git", "-C", str(REPO), "ls-files", "-z"],
                           capture_output=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    return {(REPO / p.decode("utf-8", "replace")).resolve()
            for p in r.stdout.split(b"\0") if p}


def _scanned_files():
    tracked = _tracked_paths()
    for d in SCANNED_DIRS:
        root = REPO / d
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            if p.suffix.lower() not in TEXT_SUFFIXES:
                continue
            if any(p.name.endswith(s) for s in RECEIPT_SUFFIXES):
                continue
            if tracked is not None and p.resolve() not in tracked:
                continue          # untracked or ignored: the lab did not publish it
            yield p


def _lines(path):
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except (UnicodeDecodeError, OSError):
        return None


def _hits(line):
    """Figures on this line, counting percent-encoded copy in share links.

    `release/launch-go.html` builds tweet-intent URLs where `AUC 0.998` is encoded
    as `%200.998%20` -- lexically "200.998", which no boundary-respecting regex
    matches. Public copy is still public copy when it is URL-encoded.
    """
    found = {m.group(0) for m in _FIGURE.finditer(line)}
    decoded = urllib.parse.unquote(line)
    if decoded != line:
        found |= {m.group(0) for m in _FIGURE.finditer(decoded)}
    return found


def _erratum_line(lines):
    """Index of the erratum sentinel, or None.

    Position is load-bearing: the erratum exempts only figures that come *after*
    it, so a reader meets the caveat before the number. An erratum pasted at the
    bottom of a file exempts nothing. In `release/launch-go.html` the block sits
    at line 106 -- after 104 lines of <head> and CSS, but still ahead of every
    figure on the page -- which is why this is an ordering test and not a
    fixed-size window on the top of the file.
    """
    for i, ln in enumerate(lines):
        if ERRATUM_SENTINEL in ln:
            return i
    return None


def _covered(lines, idx):
    lo = max(0, idx - CONTEXT_LINES)
    hi = min(len(lines), idx + CONTEXT_LINES + 1)
    return any(CAVEAT_WORD in ln.lower() for ln in lines[lo:hi])


def _violations():
    out = []
    for path in _scanned_files():
        lines = _lines(path)
        if lines is None:
            continue
        erratum_at = _erratum_line(lines)
        for idx, line in enumerate(lines):
            if erratum_at is not None and idx > erratum_at:
                continue
            figs = _hits(line)
            if figs and not _covered(lines, idx):
                out.append((path, idx + 1, sorted(figs), line.strip()))
    return out


# ── the guard ───────────────────────────────────────────────────────────────


def test_no_bare_auc_triple_under_release_or_docs():
    violations = _violations()
    if not violations:
        return

    report = ["", f"{len(violations)} bare AUC figure(s) in shipped copy:", ""]
    for path, lineno, figs, text in violations:
        rel = path.relative_to(REPO).as_posix()
        snippet = text if len(text) <= 100 else text[:97] + "..."
        report += [
            f"  {rel}:{lineno}",
            f"      figure(s): {', '.join(figs)}",
            f"      line:      {snippet}",
            "",
        ]
    report += [
        "These are register-detection figures at a construct ceiling. Quoted bare they",
        "overclaim; the same construct scores 0.498 on text-only deception.",
        "",
        "Fix, either way:",
        f"  1. say so inline -- the word '{CAVEAT_WORD}' within {CONTEXT_LINES} lines of the figure; or",
        "  2. if the document is already written and must be preserved as written,",
        f"     prepend the dated erratum block containing '{ERRATUM_SENTINEL}'",
        "     (see release/formal/nist-airmf-submission.md for the house form).",
        "",
        "Bound published in papers/THESIS_the_honesty_standard_2026_05_31.md and the",
        "scope erratum at the top of papers/every-mind-leaves-vitals.md.",
        "Do not resolve this by changing a number.",
    ]
    pytest.fail("\n".join(report), pytrace=False)


# ── the guard's own blind spot, pinned ──────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        "the detector reaches AUC 0.998.",            # sentence-ending period
        "the detector reaches AUC 0.998",             # end of line, no period
        "AUC 0.998, and 0.976 out-of-family",         # comma
        "| HaluEval-QA | 0.998 |",                    # table cell
        "AUC **0.943**",                              # bold markdown
        "(0.976)",                                    # parenthesised
        "0.998± 0.001",                          # followed by a sign
    ],
)
def test_figure_regex_matches_a_figure_that_ends_a_sentence(text):
    """The `(?![\\w.])` defect, pinned as a negative.

    A right guard of `(?![\\w.])` makes every one of the period cases silently
    unmatchable -- absent from the scan rather than reported. That is the exact
    failure found in `styxx/certify.py`'s `_NUM`. If someone "tightens" `_FIGURE`
    into the same shape, this test is what says so.
    """
    assert _FIGURE.search(text), f"figure went unmatched -- the guard is blind to: {text!r}"


@pytest.mark.parametrize(
    "text",
    [
        "10.998 seconds",       # longer decimal, left side
        "0.9765 precision",     # longer decimal, right side
        "0.9986",               # the dogfood receipt's deception score, not the triple
        "0.94",                 # not one of the three
        "version 6.2.0",
    ],
)
def test_figure_regex_does_not_match_neighbouring_numbers(text):
    assert not _FIGURE.search(text), f"false positive on: {text!r}"


def test_percent_encoded_figures_are_seen():
    """`%200.998%20` is `AUC 0.998` in a tweet-intent link, and reads as 200.998."""
    encoded = "href=\"...text=AUC%200.998%20HaluEval-QA\""
    assert not _FIGURE.search(encoded), "precondition: raw text should not match here"
    assert _hits(encoded) == {"0.998"}


def test_scanner_actually_reaches_the_release_tree():
    """A guard that silently scans nothing passes forever."""
    scanned = list(_scanned_files())
    assert scanned, "scanned no files -- check SCANNED_DIRS and TEXT_SUFFIXES"
    carriers = [p for p in scanned if _lines(p) and any(_hits(ln) for ln in _lines(p))]
    assert carriers, "found no file quoting the figures -- the scan is not reaching the corpus"
