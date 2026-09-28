# -*- coding: utf-8 -*-
"""styxx.diffgate — the zero-receipt gate: an agent's summary cannot lie about its diff.

The wedge the trust stack points at UNMODIFIED agent work. No receipts, no preregs, no
cooperation from the agent that wrote the summary: given the text an agent shipped (a PR
body, a commit message, a session report) and the git range it describes, extract every
diff-shaped claim and verify it against what `git diff` actually says. One command, exit
0/1 — a CI gate that catches the quiet lie in "updated the tests" when the diff touched no
test, "only touches docs/" when it edited source, "adds function retry" when it doesn't.

Construct ceiling, stated like every styxx instrument states it: the template set is
CLOSED (touched/created/deleted paths · files-changed counts · added tests · added
functions/classes · insertion/deletion counts · only-touches prefixes · tests-pass with
--evidence or --run). Prose outside the templates is NOT judged — uncovered sentences are
listed as uncovered, never scored. Verdicts per claim: VERIFIED / CONTRADICTED /
UNCHECKABLE(named). The gate fails on any CONTRADICTED; UNCHECKABLE fails only under
--strict.

THE ``tests_pass`` VOCABULARY IS TWO WORDS
------------------------------------------

::

    _TESTS_PASS_VERDICTS = ("VERIFIED", "UNCHECKABLE")

There is no accusing verdict for this claim kind. Not disabled, not behind a flag —
**absent**. The branch that read a nonzero ``--run`` exit as CONTRADICTED has been
deleted; ``selfcheck_tests_pass_never_accuses()`` re-derives its absence from this
module's own source with ``ast``, and every surviving occurrence of the word inside the
tests-pass leg is prose explaining why there is no such branch.

Why deletion rather than a flag, in one line each:

  * **Never measured.** The standing commitment out of
    ``RESULT_v14_naming_the_defects_did_not_save_it_2026_09_01.md`` is DO NOT SHIP AN
    ACCUSING VERDICT WHOSE PRECISION HAS NOT BEEN MEASURED BY A BLIND PANEL. This one has
    been measured on neither leg — not the extraction, not the adjudication.
  * **The extractor under it reads open prose with a closed regex.** The template at the
    bottom of ``_TEMPLATES`` fires on ``- [ ] all tests pass`` (an unchecked PR-template
    box, i.e. an author explicitly *declining* to assert), on ``all tests pass locally but
    CI is red``, on ``once all tests pass I will mark this ready``, and inside blockquotes
    and code fences. It reaches none of the negation, referential or containment guards —
    those live in ``_REFERENTIAL`` / ``_CONTAINMENT`` and are applied only to
    ``_PATH_KINDS``. A hard adjudicator behind a soft extractor inherits the extractor's
    precision, not its own; that is the architecture RESULT_v14 measured at 0.16.
  * **A nonzero exit is not a lie.** It is also what pytest rc=5 (no tests collected), a
    misspelled command, a missing dependency and a flaky test produce.
  * **A flag is what a maintainer who did not read the paper flips.**
    ``WITHHOLD_PATH_ACCUSATION`` below is exactly such a flag. Absence of a branch cannot
    be toggled by someone in a hurry; re-enabling must require writing the branch, in a
    diff a reviewer can see.

``VERIFIED`` survives, on the asymmetry ``styxx.evidence`` states for itself: a wrong
VERIFIED repeats a claim the author already made in prose, a wrong CONTRADICTED attacks a
stranger inside their own pull request. **It must never be printed as "the tests
passed."** It reads: *the supplied evidence, or the supplied command, said so* — and
because the extractor is unmeasured, a VERIFIED can attach to the sentence "Not all tests
pass." That is a false RECORD, not an accusation. It is disclosed here and left in place
rather than patched, because a fourth undisclosed repair cycle on this class is exactly
what RESULT_v14 forbids.

MONOTONICITY: TWO PROPERTIES, ONE HOLDS AND ONE DELIBERATELY DOES NOT
---------------------------------------------------------------------

These are different claims about ``--evidence`` and only one of them is a guarantee.
Conflating them is how this file previously came to promise something it does not do.

**Monotone against the empty baseline — HOLDS. This is the guarantee callers get.**
Supplying evidence never leaves the gate worse off than supplying none. A
``tests_pass`` claim is UNCHECKABLE with no report, so evidence can only move it to
VERIFIED or leave it exactly where it was: no other claim kind reads ``evidence``,
supplying it adds and removes no claims, and there is no route to CONTRADICTED —
``styxx.evidence``'s vocabulary is two words, ``_evidence_leg`` clamps anything else
to UNCHECKABLE, and ``selfcheck_tests_pass_never_accuses`` re-derives that from this
file's own source. Handing the gate a report therefore cannot fail a build that would
have passed with no report at all.

**Monotone under set extension — DOES NOT HOLD, and that is DELIBERATE.**
Going from evidence set E to E union {x} CAN demote VERIFIED to UNCHECKABLE. Measured
directly, under ``--strict``, on one ``tests_pass`` claim:

===========================  ==========  =====================
evidence supplied            overall     ``tests_pass``
===========================  ==========  =====================
(none)                       FAIL        UNCHECKABLE
``green.xml``                PASS        VERIFIED
``green.xml`` ``empty.xml``  FAIL        UNCHECKABLE
===========================  ==========  =====================

Row three is not a regression and must not be "repaired" — not by letting a partial
read affirm, and not by special-casing empty files. It is the contract's own rule,
stated in ``styxx.evidence``: ANY unparsed source blocks VERIFIED, because a partial
read may honestly DECLINE but may not honestly AFFIRM. You cannot certify "all tests
pass" from nine shards out of ten. Adding a file adds a question, and a question the
gate could not read is not an answer. Affirming from an incomplete set is the exact
failure mode this whole module exists to refuse, so the demotion is the module
working, not the module breaking.

THE CASE THAT WILL BITE SOMEONE, in plain words: a sharded CI matrix where nine JUnit
reports parse and the tenth is truncated — the runner died, the artifact upload raced
the job, the shard timed out. The gate DECLINES. It does not affirm from the nine,
and under ``--strict`` that is a red build. That is the intended behaviour.

What an operator should do about it: **supply complete evidence, or supply none.**
Those are the two honest positions and there is no third. Wire the job so a missing
shard report fails collection outright, rather than passing whichever files happened
to land in the directory; or, when a shard is known-lost and you would rather not
block, pass no evidence at all and let the claim stand UNCHECKABLE on its own terms.
Do not pass the partial set hoping the good files carry it — they will not, by
design. The ``why`` on the declining claim names how many of how many sources could
not be read and names one of them by path and reason, so the red build already says
which shard to go and fix.

``--run`` IS CODE EXECUTION
---------------------------

``--run CMD`` executes CMD **through a shell, with cwd set to --repo**. On an untrusted
pull request that is remote code execution with extra steps: ``pytest`` imports the PR's
``conftest.py`` at collection, ``pytest.ini``/``pyproject.toml`` ``addopts`` can load a
plugin, ``npm test`` runs the string in the PR's ``package.json``, ``make test`` runs the
PR's Makefile. ``os.environ`` is inherited unscrubbed. And the PR author controls the exit
code in **both** directions, so the check is trivially green-lit by the adversary it
exists to catch while remaining fully dangerous to the defender. Correct in first-party CI
on a repository you own; never on a stranger's branch. Prefer ``--evidence``, which reads
bytes and executes nothing.

CLI::

    python -m styxx.diffgate SUMMARY.md --repo . --base main --head HEAD \
        [--evidence junit.xml attestation.json --commit <40-hex>] \
        [--run "pytest -q"] [--strict] [--out GATE.json]
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["gate_diff", "gate_diff_text", "DiffGate", "DiffClaim",
           "selfcheck_tests_pass_never_accuses"]

# A claimed path must end in a KNOWN file extension (closed set — the same honesty as the
# template set itself). The naive any-dotted-token form false-accused decimals (0.5349),
# versions (v0.6.2), DOIs, and module names (styxx.framelocality) in the 80-commit
# history sweep that validated this module — six of six contradictions were false
# accusations until this whitelist existed. A gate that can accuse a number of not being
# a file does not ship.
_EXT = (r"py|md|json|jsonl|txt|yml|yaml|toml|cfg|ini|js|ts|tsx|jsx|css|html|tex|sh|ps1|"
        r"bat|ipynb|csv|tsv|npz|npy|pdf|png|jpg|svg|gz|zip|lock|xml|rst|c|h|cpp|rs|go|java")
_PATH = rf"[\w./\\-]*[A-Za-z_][\w-]*\.(?:{_EXT})\b"
# a window between the verb and the path: real summaries write "updated the parser in
# styxx/certify.py", not "updated styxx/certify.py". Bounded (no sentence crossing — the
# splitter already scoped us to one sentence) and path-shaped on the right.
_W = r"[^.!?\n]{0,60}?"
_TEMPLATES = [
    # Verb forms pinned, not `creat\w+`. The stem also matches the NOUN
    # "creation", which in real code prose means a place in the code where a
    # struct is constructed -- "at both TestComparison creation sites in
    # component_report.go" -- and that produced a file-created accusation
    # against a real public PR during the 7.44.2 market sweep. Exactly the
    # `fix\w+` / "fixture" catch from 7.29.2, one stem over.
    ("file_created", re.compile(
        rf"\b(?:create|creates|created|creating|new)\s+(?:file|module|script|test file)?\s*{_W}[`\"']?(?P<path>{_PATH})[`\"']?",
        re.I)),
    ("file_created", re.compile(
        rf"[`\"']?(?P<path>{_PATH})[`\"']?\s*(?::|—|--)\s*(?:new|created)\b", re.I)),
    ("file_deleted", re.compile(
        rf"\b(?:delet\w+|remov\w+)\s+(?:the\s+file\s+)?{_W}[`\"']?(?P<path>{_PATH})[`\"']?", re.I)),
    ("file_touched", re.compile(
        rf"\b(?:modif\w+|updat\w+|edit\w+|chang\w+|refactor\w+|fix(?:es|ed|ing)?\b|add\w+|extend\w+|"
        rf"hard\w+|wir\w+|patch\w+)\s+{_W}[`\"']?(?P<path>{_PATH})[`\"']?", re.I)),
    ("file_touched", re.compile(
        rf"^[\s*-]*[`\"']?(?P<path>{_PATH})[`\"']?\s*(?::|—|--)\s+", re.M)),
    ("files_changed_count", re.compile(r"\b(?P<n>\d+)\s+files?\s+(?:were\s+)?changed", re.I)),
    # BC-1/BC-2 (PREREG_bc2_by_construction_2026_09_16): the counted noun is captured so
    # "added 3 test cases" is not read as a count of `def test_` functions, and
    # "added a function named foo" reads foo, not `named`.
    ("tests_added", re.compile(
        r"\b(?:add\w+|creat\w+)\s+(?P<n>\d+)\s+(?:new\s+)?tests?\b"
        r"(?:\s+(?P<noun>cases?|files?|scenarios?|suites?|class(?:es)?|functions?|methods?)\b)?", re.I)),
    ("symbol_added", re.compile(
        r"\b(?:add\w+|introduc\w+)\s+(?:(?:a|an|the|new)\s+){0,2}(?P<kind>function|class|method)\s+"
        r"(?:(?:named|called)\s+)?[`\"']?(?P<name>[A-Za-z_]\w*)", re.I)),
    ("only_touches", re.compile(
        r"\bonly\s+(?:touch\w+|modif\w+|chang\w+)\s+(?:files?\s+(?:in|under)\s+)?"
        r"[`\"']?(?P<prefix>[\w./\\-]+)[`\"']?"
        r"(?:,?\s+and\s+(?:files?\s+(?:in|under)\s+)?[`\"']?(?P<prefix2>[\w-]*[./\\][\w./\\-]*)[`\"']?)?", re.I)),
    ("tests_pass", re.compile(r"\b(?:all\s+)?tests\s+(?:pass|are\s+passing|green)\b", re.I)),
    # COMPAT-1 (PREREG_compat1_2026_09_16): the compatibility claim. Read, never judged: the
    # verdict is UNCHECKABLE with the public definitions the diff removed named in the reason.
    ("compat_claim", re.compile(
        r"\b(?:no\s+breaking\s+changes?|non[- ]breaking|backwards?[- ]compatib(?:le|ility)|"
        r"(?:zero|no)\s+(?:behaviou?r(?:al)?|functional)\s+changes?|fully\s+compatible|"
        r"does\s+not\s+(?:break|change)\s+(?:any\s+|the\s+)?(?:existing\s+)?(?:behaviou?r|api|public\s+api))\b", re.I)),
]


# EXTERNAL-1 consequence, preregistered and paid: the path-claim accusation is
# WITHHELD until a held-out blind panel licenses its return (PREREG_v13_repair).
# Exposed as a flag, not a deletion, so the counterfactual stays measurable — the
# question "how much of the failure was the instrument and how much was the
# harness feeding it" is answerable only if this can be toggled in a measurement.
WITHHOLD_PATH_ACCUSATION = True

# V14 repairs (PREREG_v14_repair_2026_08_31), flags rather than deletions so the
# counterfactual stays measurable: containment extended to touch claims, and a
# bare basename absent from the diff abstaining instead of accusing.
V14_CONTAINMENT_TOUCH = True
V14_BARE_NAME_ABSTAIN = True

# V13 repair 2 (PREREG_v13_repair_2026_08_31): FROZEN NON-FILE NOUNS. The
# extension whitelist cannot tell the runtime `Node.js` from a file named
# node.js, and EXTERNAL-1 caught the gate accusing agent prose of not
# containing a file called "Next.js". Closed list, quoted in full in the
# RESULT so the closure is auditable, and applied only to bare tokens with no
# directory part -- a real `lib/node.js` still claims normally.
# BC-2 (PREREG_bc2_by_construction_2026_09_16, after BC-1's INVALID; issue #110). On the EXTERNAL-1
# corpus 549 of the 665 accusations the gate still made were unsupported by
# construction: `tests_added` and `symbol_added` count `def` lines, and 227 of
# their 249 accusations were in diffs with no Python file; `only_touches`
# takes the token after the verb as a path prefix, and 322 of its 341
# accusations captured an English word ("only modifies THE footer"). The
# four rules below remove those accusations. They add none: a claim they
# touch becomes UNCHECKABLE or stops being a claim, never CONTRADICTED.
BC1_BY_CONSTRUCTION = True
_PY_SUFFIXES = (".py", ".pyi")
_TEST_NOUNS_NOT_FUNCTIONS = frozenset({"case", "cases", "file", "files", "scenario",
                                       "scenarios", "suite", "suites", "class", "classes"})
_SYMBOL_WORDS = frozenset({
    "to", "with", "that", "for", "in", "on", "of", "by", "and", "or", "as", "the", "a",
    "an", "this", "which", "it", "its", "is", "declaration", "implementation",
    "definition", "signature", "body", "stub", "call", "wrapper", "override",
    "overload", "level", "support", "named", "called",
})


def _diff_touches_python(status: dict) -> bool:
    # AMENDMENT_path2 C-2: read on the undotted key, as before #121, so a file named `.py` is not Python.
    return any(_undotted(p).lower().endswith(_PY_SUFFIXES) for p in status)


# PATH-1 (PREREG_path1_only_touches_repair_2026_09_17, sha256 618d800f...). Two of the six
# `only_touches` failure modes published in RESULT_bench2_INVALID_2026_09_17 are repaired here and
# four are not; the prereg says which and why. This set is the committed data for mode 2 and
# mirrors papers/closed-model-frontier/path1_extensions.txt byte for byte (test_path1 pins it).
# It was frozen with the preregistration, before measurement, from general language conventions --
# never tuned against the eleven pull requests the repair is scored on.
PATH1_EXTENSIONS = frozenset("""
c cc cpp cxx h hh hpp hxx m mm
py pyi pyx rb rs go java kt kts scala clj cljs swift dart
js jsx mjs cjs ts tsx vue svelte
cs fs vb fsx pas pp
php pl pm t r rmd jl lua tcl groovy gradle
sh bash zsh fish ps1 psm1 psd1 bat cmd
html htm xml xsl xslt svg css scss sass less styl
json json5 yaml yml toml ini cfg conf properties env plist
md markdown mdx rst adoc txt text tex bib
sql graphql gql proto thrift avsc
lock sum mod work
dockerfile makefile mk cmake gemspec podspec csproj vbproj fsproj sln props targets
tf tfvars hcl bicep nix
at ac am in out golden snap
png jpg jpeg gif webp ico bmp tiff pdf
zip tar gz tgz bz2 xz 7z jar war whl
""".split())


def _has_real_extension(token: str) -> bool:
    """`package.json` yes, `Assert.NotNull` no. PATH-1 mode 2."""
    if "." not in token:
        return False
    return token.rsplit(".", 1)[-1].strip().lower() in PATH1_EXTENSIONS


def _is_bare_filename(pref: str) -> bool:
    """A prefix naming a file rather than a location: no slash, and a real extension.

    PATH-1 mode 1. "Only modify package.json ... in each package folder" means a file with that
    name anywhere in the tree, not a file at the repository root, so containment for such a
    prefix matches on the basename.
    """
    return "/" not in pref and _has_real_extension(pref)


def _path_inside(path: str, pref: str) -> bool:
    if _is_bare_filename(pref):
        return path == pref or path.endswith("/" + pref)
    return path == pref or path.startswith(pref + "/")


def _prefix_is_path_shaped(prefix: str, status: dict) -> bool:
    """A scope prefix is a path when it looks like one or names a segment of a changed path.

    Judged on the prefix as written, minus a sentence-final period: "docs/" is a path
    because of its slash, "package.json" because of its dot, "src" because a changed path
    has that segment; "the", "files" and "markdown" are words.
    """
    raw = prefix.strip("`\"'").rstrip(".")
    if not raw:
        return False
    if any(ch in raw for ch in "/\\"):
        return True
    # PATH-1 mode 2: a dot alone used to be enough, which admitted `Assert.NotNull` as a path.
    # The suffix must be a real file extension from the committed list; otherwise fall through
    # to the segment test below, exactly as a dotless word does.
    if "." in raw and _has_real_extension(raw.rstrip("/")):
        return True
    # NOTE_path2_fourth_pass_2026_09_25 (F-1): the prefix is undotted too. Undotting the changed
    # paths alone compared ".gitignore" with "gitignore" and ".github" with "github", so every
    # slashless dotted prefix whose last dot-segment is not a listed extension stopped being a path:
    # correct VERIFIEDs and correct CONTRADICTEDs were withdrawn as "not a path", and C-3's own
    # accusation class never fired for them. Both sides undotted is the reading main had.
    low = _undotted(_norm(raw)).rstrip("/").lower()
    for changed in status:
        # PATH-2 (#121): the segments are read with the key's leading dots dropped, as they were
        # before the key kept them, so "github" still names the ".github" directory here; whether
        # the changed paths lie under it is decided with the dots, in `_gate`.
        if low in (seg.lower() for seg in _undotted(changed).split("/")):
            return True
    return False


# PATH-2 (PREREG_path2_resolution_2026_09_17, issue #101). `tests_added` and `symbol_added` read
# only the added lines, so a `def` line that merely changed -- a signature edit, a trailing
# comment, a re-indent -- counted as added, and "Added 2 tests" over two edited tests was
# VERIFIED. A definition that the removed lines of the SAME FILE also define is changed, not
# added. Both doors hand `_gate` the per-file sides, so both read it the same way. When nothing
# changed, every verdict and every reason is what it was.
#
# AMENDMENT_path2_resolution_2026_09_17 (C-1): removed and added definitions pair ONE TO ONE, per
# file and per name, and a file whose status is `A` pairs nothing (it has no base; removed lines
# under an `A` header are the shelf's fold). The definition-line patterns are written without
# `\s`, `\w` or `\b`, with one optional leading U+FEFF, so this file and web/gate/diffgate.js read
# a BOM strip and a non-ASCII name the same way.
#
# NOTE_path2_third_pass_2026_09_25 (R-1, R-2): the ADDED-side pattern and the added-blob count
# `got` must read the same set of lines, or a line the pairing subtracts is one `got` never added
# and `net` falls below the tests really added -- a false VERIFIED of the #101 kind, which is what
# C-1 was frozen to remove. They are made to agree by counting a leading U+FEFF in `got` too, in
# this file and in the port, rather than by dropping it from the pairing: that is the reading the
# port already had, and it leaves Python and JavaScript on the same verdict AND the same reason.
# The REMOVED-side pattern alone accepts `async`, so an `async def test_x` rewritten as `def
# test_x` is a changed test, not an added one; the added side must not, because `got` does not
# count `async def test_` and the two sets would part again.
#
# NOTE_path2_fourth_pass_2026_09_25 (F-3): R-1 made the pairing a SUBSET of `got`, not the same
# set. `got` read its indent as `\s*`, which also takes U+00A0, U+3000 and every other Unicode
# space, while the pairing reads `[ \t]*`; an NBSP re-indent of a test was counted as added and
# paired with nothing, and "Added 1 test." over it VERIFIED on both ports. `got` now reads the indent
# the pairing reads, so the two count exactly the same lines. (Of the characters this drops, only
# U+000C is legal Python indentation, and the Python never counted a line it led: `splitlines()`
# cut the line there until F-2.)
#
# NOTE_path2_fifth_pass_2026_09_25 (V-1): ONE reading of a Python definition line. Rounds 2, 3 and 4
# each repaired one whitespace character in one of these patterns and opened the next, because
# patterns that must read the same lines were written one by one. CPython's tokenizer takes space,
# tab and form feed as indentation and as the space between two tokens, and nothing else; a U+FEFF
# may open a file. So every pattern that reads a definition line opens with `_DEF_INDENT` and
# separates its keywords with `_DEF_SEP`. `got` counts the lines the added-side test pairing reads,
# and `hit` asks whether an added line is read by the symbol pairing's added side. (V-1 said neither
# count could then read a line its pairing cannot; that held per line and failed per diff, because
# `got` and `hit` read the added blob of one parser and the pairings the sides of another -- W-1
# below makes them one reading.) The REMOVED side alone also takes `async`, for tests and symbols
# alike (R-2): `got` and `hit` do not read `async def`, so the added side must not. V-1's name end,
# an ASCII stop set, is replaced by W-2's identifier.
#
# NOTE_path2_sixth_pass_2026_09_25 (W-2): ONE reading of a name. V-1 ended a definition's name at an
# ASCII stop set while the claimed name came from the claim template's `\w`, so the two ended in
# different places: "Added function caf<U+00E9>." read CONTRADICTED on the port ('caf') and "Added
# function col<U+00B7>leccio." on the Python ('col'), both true claims. The test side still read a
# literal single space after `def` and ran a test name to `[^ \t(:]*`, so a form feed or a `[` after it
# became part of the name. Now a name, claimed or defined, is the Python identifier that starts there, read by
# `_identifier_at` with the language's own rule (`str.isidentifier`: XID_Start or `_`, then
# XID_Continue); every definition line is `_DEF_INDENT`, `def`/`class` (the removed side alone also
# `async`), `_DEF_SEP`, then that identifier, and the character after it is ASCII or the line ends -- for
# tests, symbols, `got`, `hit` and both pairings. (A non-ASCII character right after a name is either
# XID_Continue, so the identifier already holds it and the name is another name, or one CPython refuses
# there -- U+00A0, U+3000, U+FEFF, U+2028 -- so the line defines nothing, as V-1 read it.) The indent
# no longer takes a U+FEFF: W-1's parse drops one where CPython reads it, at line 1 of a file, and a
# U+FEFF anywhere else is a character CPython refuses.
#
# NOTE_path2_seventh_pass_2026_09_25. (1) ONE table. W-2 asked the runtime which characters an identifier
# holds -- `str.isidentifier` here (Unicode 13.0 to 15.0 on the Pythons this package supports), the
# runtime's `\p{XID_Continue}` in web/gate/diffgate.js (16.0 on Node 24) -- so "Added function
# foo<U+30FB>bar." over `def foo<U+30FB>bar():` read CONTRADICTED here and VERIFIED in the port. Both
# ports now read styxx/_xid.py's table, the same string in both files: Unicode 15.0.0, CPython 3.12's.
# (2) The claim side is read as the definition side is. W-2 took the identifier that starts where the
# template's `name` group starts, and when the summary's name ran on past it -- "Added function
# fo<U+00B2>o." (U+00B2 is `\w` but not XID_Continue) -- it truncated the name to `fo` and verified the
# claim on any `def fo`, where main read the whole name and matched nothing. `_defined_name` refuses a
# line whose name runs on into a character no identifier holds; `_claimed_name` now does the same: a
# claimed name followed by a word character the identifier cannot hold names no identifier, and the
# claim reads UNCHECKABLE with that reason, never a truncated name. A claimed identifier that ENDS in a
# middle dot (U+00B7, U+0387: the two punctuation characters XID_Continue holds in 15.0; U+0387 is the
# Greek semicolon) is read the same way -- prose writes one after a word, so where the name ends is not
# certain. A name printed in a reason is quoted as repr() quotes an identifier, without asking the
# runtime which characters it can print.
_DEF_INDENT = r"^[ \t\f]*"
_DEF_SEP = r"[ \t\f]+"
_DEF_HEAD = re.compile(_DEF_INDENT + r"(def|class)" + _DEF_SEP)
_DEF_HEAD_REMOVED = re.compile(_DEF_INDENT + r"(?:async" + _DEF_SEP + r")?(def|class)" + _DEF_SEP)
_PROSE_DOTS = "··"

from ._xid import continues_identifier as _xid_continues  # noqa: E402
from ._xid import in_skew as _xid_skew  # noqa: E402
from ._xid import is_word as _xid_word  # noqa: E402
from ._xid import opens_identifier as _xid_opens  # noqa: E402
from ._fold import fold as _case_fold  # noqa: E402  (NOTE_path2_tenth_pass, K-2)

# NOTE_path2_eighth_pass_2026_09_27. The convergence principle: wherever this branch's new reading cannot be
# sure it reads a claim at least as well as main, it ABSTAINS -- in both ports, with a reason that names the
# uncertainty. The seventh pass asserted in five places where it could not be sure; each now abstains:
#   Y-1  the file list. A `---`/`+++` line read as a header after lines no hunk count placed may be content (a
#        removed `-- users` prints `--- users`, an added `++ x` prints `+++ x`) or a header; and two header
#        paths that differ only in case are one key, on main too. Where the list may hold a phantom or a merge
#        main's other errors balanced, a count, a scope and a path claim abstain. A pair with a header's shape
#        -- `--- X`, `+++ X` (or /dev/null), then `@@` -- is read as a header, as main read it.
#   Y-2  U+FEFF. Outside the counts, one is dropped from a line the diff shows is line 1 -- the opening added
#        line of a created file, the opening line of a side after a hunk header starting at 1 -- as CPython reads it
#        and main's port read it. A definition a U+FEFF opens anywhere else makes a claim that could read it
#        abstain. And any added test definition a U+FEFF opens, line 1 included, abstains the test count: main's
#        Python counted it as no test and its port as one, so a count elsewhere may have balanced either.
#   Y-3  the Pythons this package supports read identifiers by Unicode 13.0 to 16.0, the table by 15.0; a
#        claimed name, or a test definition's name, that meets a code point those versions read differently
#        (styxx/_xid.py's SKEW) is not one name on every supported Python, and the claim abstains.
#   Y-5  the #101 pairing withdraws; it does not verify. The reading is by line and by text, on main as here:
#        a `def test_` line inside a string (opened where the diff does not show it), in a file that is not
#        Python, or split by a backslash is read as Python does not read it, and main's count was sometimes
#        right only because it also counted a changed test. So where a changed test is paired away, a claimed
#        count equal to what is left reads UNCHECKABLE, never VERIFIED; a count outside [net, got] is one main
#        contradicts too.
# And Y-4, a repair of a defect main has too: GNU diff writes `/dev/null` followed by a TAB and a timestamp,
# and neither parser recognised it, so two deletions shared the key "dev/null\t<time>" and a created file
# read as modified; the check now cuts the timestamp. Every other header path keeps it, exactly as main keys
# it, so no claim main abstained on by that key is read now.


def _skew(ch: str) -> bool:
    """Y-3: the character is one some supported Python reads differently from the table (styxx/_xid.py)."""
    return _xid_skew(ch)


def _wide_identifier_at(text: str, i: int) -> str:
    """Y-3: the longest name any supported Python could read at `text[i]`: the table's identifier, widened by
    every code point of the skew set."""
    if i >= len(text) or not (_xid_opens(text[i]) or _skew(text[i])):
        return ""
    j = i + 1
    while j < len(text) and (_xid_continues(text[j]) or _skew(text[j])):
        j += 1
    return text[i:j]


_LEADING_BOM_RUN = re.compile("^[ \t\f\ufeff]*")


def _bom_hidden(line: str):
    """Y-2: the line with the U+FEFF dropped from its indent, when a U+FEFF opens it (after any indent), else
    None. main's port read such a line through JavaScript's \\s; CPython reads one only at line 1 of a file."""
    lead = _LEADING_BOM_RUN.match(line).group(0)
    if "\ufeff" not in lead:
        return None
    return lead.replace("\ufeff", "") + line[len(lead):]


def _skew_test(line: str):
    """Y-3: the earliest skew code point in the name of a test definition (`def test_`, `async` too) on this line,
    read as widely as any supported Python reads it; else None."""
    text = _bom_hidden(line) or line
    m = _DEF_HEAD_REMOVED.match(text)
    if m is None or m.group(1) != "def":
        return None
    wide = _wide_identifier_at(text, m.end())
    if not wide.startswith("test_"):
        return None
    return next((ch for ch in wide if _skew(ch)), None)


_Y2_WHY = "opens with U+FEFF where the diff does not show it is line 1 of its file, the one place CPython reads one"
_Y2_TEST = "an added test definition opens with U+FEFF, which main's Python counted as no test and its port as one"
_Y3_VERSIONS = "the Pythons this package supports (Unicode 13.0 to 16.0)"


def _bom_test_note(raw: str, text: str):
    """Y-2: `text` is an added line with the U+FEFF opening line 1 of its file dropped (`raw` as the diff wrote
    it); when what is left defines a test `got` counts, the reason the count abstains, else None."""
    return _Y2_TEST if text != raw and _test_name(text) else None


def _test_doubt(added_blob: str, sides: dict | None, notes: dict | None = None):
    """Y-2 and Y-3 for `tests_added`: why a test definition on some added or removed line cannot be read the
    same way by every reading that matters, or None. A U+FEFF the parse dropped at line 1 (`notes`), then the
    added lines in the blob's order and each file's removed lines in the sides' order; the earliest found."""
    if (notes or {}).get("bom"):
        return notes["bom"]
    removed = [line for _a, r in (sides or {}).values() for line in r]
    for line in added_blob.split("\n") + removed:
        hidden = _bom_hidden(line)
        if hidden is not None and _test_name(hidden, removed=True):
            return f"a test definition {_Y2_WHY}"
        ch = _skew_test(line)
        if ch is not None:
            return f"a test definition's name holds U+{ord(ch):04X}, which {_Y3_VERSIONS} read differently"
    return None


def _symbol_doubt(name: str, added_blob: str, sides: dict | None):
    """Y-2 for `symbol_added`: a line that defines `name` once the U+FEFF opening it is dropped."""
    removed = [line for _a, r in (sides or {}).values() for line in r]
    for line in added_blob.split("\n") + removed:
        hidden = _bom_hidden(line)
        if hidden is not None and _defines(hidden, name, removed=True):
            return f"a definition of {_qname(name)} {_Y2_WHY}"
    return None


def _pairing_withdraws(chg: int) -> bool:
    """Y-5: the #101 pairing paired `chg` added test definitions with removed ones. It may then withdraw a
    verdict main gave (a count inside [net, got] abstains) but not give one main did not: `net` equal to the
    claim is not verified, since `got` itself may read a line Python does not define (a string opened where the
    diff does not show it, a file that is not Python) and main's contradiction was then right."""
    return chg > 0


def _identifier_at(text: str, i: int) -> str:
    """The Python identifier that starts at `text[i]`, or "": the opening character XID_Start or `_`,
    each later one XID_Continue, by styxx/_xid.py's table (Unicode 15.0.0), not the runtime's."""
    if i >= len(text) or not _xid_opens(text[i]):
        return ""
    j = i + 1
    while j < len(text) and _xid_continues(text[j]):
        j += 1
    return text[i:j]


def _qname(name: str) -> str:
    """repr() of an identifier, which holds no quote, no backslash and no character repr() escapes."""
    return "'" + name + "'"


def _claimed_name(sentence: str, m) -> tuple:
    """(name, why) for a `symbol_added` claim: the identifier starting where the template's `name` group
    starts (W-2), read by the rule a definition's name is read by. `why` is None when the claim names that
    identifier, and otherwise the reason the claim is UNCHECKABLE (NOTE_path2_seventh_pass): the name runs
    on into a word character no identifier holds, or it ends in a middle dot."""
    start = m.start("name")
    name = _identifier_at(sentence, start)
    end = start + len(name)
    # NOTE_path2_eighth_pass (Y-3): a name, or the character right after it, that some supported Python reads
    # differently from the table is not one name on every Python; checked before the table's own rules.
    met = next((ch for ch in sentence[start:end + 1] if _skew(ch)), None)
    if met is not None:
        return name, (f"the claimed name {_qname(name)} meets U+{ord(met):04X}, which {_Y3_VERSIONS} read "
                      "differently; no definition is read for it")
    if end < len(sentence) and _xid_word(sentence[end]):
        ch = sentence[end]
        return name, (f"the claimed name runs past {_qname(name)} into {_qname(ch)} (U+{ord(ch):04X}), which "
                      "no Python identifier holds; no definition is read for it")
    if name and name[-1] in _PROSE_DOTS:
        return name, (f"the claimed name {_qname(name)} ends in {_qname(name[-1])} (U+{ord(name[-1]):04X}), "
                      "which prose also writes after a word; no definition is read for it")
    return name, None


def _defined_name(line: str, removed: bool = False):
    """(`def` or `class`, name) that a diff line defines, or None."""
    m = (_DEF_HEAD_REMOVED if removed else _DEF_HEAD).match(line)
    if m is None:
        return None
    name = _identifier_at(line, m.end())
    end = m.end() + len(name)
    if not name or (end < len(line) and ord(line[end]) > 0x7F):
        return None
    return m.group(1), name


def _test_name(line: str, removed: bool = False):
    """The test a diff line defines (`def test_...`), or None."""
    d = _defined_name(line, removed)
    return d[1] if d is not None and d[0] == "def" and d[1].startswith("test_") else None


def _defines(line: str, name: str, removed: bool = False) -> bool:
    """Whether a diff line defines `name` (a function or a class)."""
    d = _defined_name(line, removed)
    return d is not None and d[1] == name


def _symbol_hit(name: str, added_blob: str) -> bool:
    """`symbol_added`'s question: does some added line define `name`? One line at a time, with the
    added-side pairing's own reading (V-1, W-2)."""
    return any(_defines(line, name) for line in added_blob.split("\n"))


def _added_tests(added_blob: str) -> int:
    """`tests_added`'s `got`: the added lines that define a test, by the added-side pairing's own
    reading (`_test_name`), one line at a time (W-2)."""
    return sum(1 for line in added_blob.split("\n") if _test_name(line))


def _async_tests_added(sides: dict | None, status: dict | None = None) -> int:
    """NOTE_path2_seventh_pass (A-1): the `async def test_` definitions a file adds beyond those its removed
    lines define (a file whose status is `A` removes none), summed. `got` counts no `async def test_` line
    -- on main either -- so where the diff adds one, the count is not the diff's: main's reading was right
    there only by a second miscount, and the pairing (#101) removed that miscount. Such a claim abstains."""
    n = 0
    for path, (added, removed) in (sides or {}).items():
        fresh: dict = {}
        for line in added:
            t = _test_name(line, removed=True)
            if t and not _test_name(line):
                fresh[t] = fresh.get(t, 0) + 1
        if not fresh:
            continue
        gone: dict = {}
        if (status or {}).get(path) != "A":
            for line in removed:
                t = _test_name(line, removed=True)
                if t:
                    gone[t] = gone.get(t, 0) + 1
        n += sum(max(0, k - gone.get(t, 0)) for t, k in fresh.items())
    return n


def _changed_test_defs(sides: dict | None, status: dict | None = None) -> int:
    """Test definitions paired one to one: per file whose status is not `A`, per test name,
    min(added lines defining it, removed lines defining it), summed. The caller clamps to `got`.

    The added side reads `_test_name(line)`, every line of which `got` also counts; the removed side
    reads `_test_name(line, removed=True)`, which also accepts `async` (NOTE_path2_third_pass_2026_09_25).
    """
    n = 0
    for path, (added, removed) in (sides or {}).items():
        if (status or {}).get(path) == "A":
            continue
        gone: dict = {}
        for line in removed:
            t = _test_name(line, removed=True)
            if t:
                gone[t] = gone.get(t, 0) + 1
        if not gone:
            continue
        new: dict = {}
        for line in added:
            t = _test_name(line)
            if t:
                new[t] = new.get(t, 0) + 1
        n += sum(min(k, gone.get(name, 0)) for name, k in new.items())
    return n


def _definition_only_changed(name: str, sides: dict | None, status: dict | None = None) -> bool:
    """Some file both adds and removes a definition of `name`, and no file adds more definitions of
    it than it removes (a file whose status is `A` removes none). Counted per file, one to one.
    The added side reads `_defines(line, name)`, the reading `hit` uses (V-1, W-2)."""
    paired = False
    for path, (added, removed) in (sides or {}).items():
        a = sum(1 for line in added if _defines(line, name))
        r = 0 if (status or {}).get(path) == "A" else sum(1 for line in removed if _defines(line, name, True))
        if a > r:
            return False
        if a and r:
            paired = True
    return paired


# COMPAT-1 (PREREG_compat1_2026_09_16). 8,467 of the 71,016 EXTERNAL-1 descriptions claim
# compatibility and the gate read none of them. It reads them now and says one thing: which
# public top-level definitions the diff removed without re-defining, per language, with files.
# There is no VERIFIED and no CONTRADICTED for this kind -- a removed name is not proof of a
# break and an intact surface is not proof of compatibility -- and selfcheck below pins it.
# COMPAT-2 (PREREG_compat2_surface_and_panel_2026_09_16). The reading is sharpened -- a removed
# definition under a test / example / docs / scripts / internal / vendor path is scaffolding, and a
# definition re-defined with a different parameter list is a signature change, reported, never a
# drop -- and a CANDIDATE is computed: a covered language and at least one removed public
# definition on the surface. The verdict stays UNCHECKABLE until a blind panel licenses it; the
# licence is this flag, flipped only by that panel's RESULT, and a test pins it false.
COMPAT2_LICENSED = False
_COMPAT_VERDICTS = ("UNCHECKABLE",) if not COMPAT2_LICENSED else ("UNCHECKABLE", "CONTRADICTED")
_COMPAT_SCAFFOLD = re.compile(
    r"(?:^|/)(?:tests?|testing|specs?|__tests__|examples?|samples?|demos?|docs?|scripts?|tools?|bench|"
    r"benchmarks?|fixtures?|internal|_internal|private|vendor|third_party|migrations?|cmd|e2e|integration|"
    r"mocks?|stories|storybook|playground|sandbox|experiments?|dev|build)/"
    r"|(?:^|/)(?:test_[^/]*|[^/]*_test\.(?:go|py)|[^/]*\.(?:test|spec)\.[^/]+|conftest\.py|setup\.py)$")
_COMPAT_LANGS: dict = {
    # language: (suffixes, regex over ONE removed line with a `name` group)
    "python": ((".py",), re.compile(r"^(?:async\s+)?(?:def|class)\s+(?P<name>[A-Za-z]\w*)")),
    "js/ts": ((".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".mts", ".cts"),
              re.compile(r"^export\s+(?:default\s+)?(?:async\s+)?(?:function\*?|class|const|let|var|"
                         r"interface|type|enum)\s+(?P<name>[A-Za-z_$]\w*)")),
    "go": ((".go",), re.compile(r"^(?:func\s+(?:\([^)]*\)\s*)?|type\s+)(?P<name>[A-Z]\w*)\b")),
    "rust": ((".rs",), re.compile(r"^\s*pub\s+(?:async\s+)?(?:fn|struct|enum|trait|type|const|static)\s+(?P<name>[A-Za-z_]\w*)")),
    "java": ((".java", ".kt"), re.compile(r"^\s*public\s+(?:static\s+|final\s+|abstract\s+)*[\w<>\[\],\s]+?\s+(?P<name>[a-zA-Z_]\w*)\s*\(")),
}
_COMPAT_MAX_NAMED = 5


# BIN-1 (PREREG_bin1_binary_files_2026_09_16, issue #118). A unified diff carries a binary
# change, a mode-only change or a pure rename as a `diff --git` header with NO `---`/`+++`
# pair, and both parsers below registered a file only from that pair. EXTERNAL-5's cross-check
# caught it: eight PNGs and one .scss read as one file, and a truthful count was one click
# from being called a lie. A header that reaches the next header without a pair now registers
# its file: A on `new file mode` / `Binary files /dev/null and …`, D on `deleted file mode` /
# `… and /dev/null differ`, else M. Files with hunks are read exactly as before.
_DIFF_GIT = re.compile(r'^diff --git (?:"a/(?P<qa>(?:[^"\\]|\\.)*)"|a/(?P<a>.*?)) (?:"b/(?P<qb>(?:[^"\\]|\\.)*)"|b/(?P<b>.*))$')
_BINARY_LINE = re.compile(r"^Binary files (?P<a>.+?) and (?P<b>.+?) differ$")


def _header_paths(line: str) -> tuple[str, str]:
    """`diff --git a/X b/Y` -> (X, Y). Same-name headers split at the middle; the rest by regex."""
    body = line[len("diff --git "):]
    if len(body) % 2 == 1:
        mid = len(body) // 2
        if body[mid] == " " and body[:mid].startswith("a/") and body[mid + 1:].startswith("b/") \
                and body[2:mid] == body[mid + 3:]:
            return body[2:mid], body[mid + 3:]
    m = _DIFF_GIT.match(line)
    if not m:
        return "", ""
    a = m.group("qa") if m.group("qa") is not None else (m.group("a") or "")
    b = m.group("qb") if m.group("qb") is not None else (m.group("b") or "")
    return a, b


class _Pending:
    """One `diff --git` header waiting to learn whether a `---`/`+++` pair follows."""
    __slots__ = ("a", "b", "status")

    def __init__(self, line: str):
        self.a, self.b = _header_paths(line)
        self.status = "M"

    def note(self, line: str) -> None:
        if line.startswith("new file mode"):
            self.status = "A"
        elif line.startswith("deleted file mode"):
            self.status = "D"
        elif line.startswith("rename from "):
            self.a = line[len("rename from "):]
        elif line.startswith("rename to "):
            self.b = line[len("rename to "):]
        else:
            m = _BINARY_LINE.match(line)
            if m:
                if m.group("a") == "/dev/null":
                    self.status = "A"
                elif m.group("b") == "/dev/null":
                    self.status = "D"

    def path(self) -> str:
        raw = self.a if self.status == "D" else self.b
        return _norm(raw) if raw else ""


# NOTE_path2_fourth_pass_2026_09_25 (F-2). Git separates the lines of a diff with "\n"; a CRLF file
# shows its "\r" before it. `str.splitlines()` also breaks on U+000B, U+000C, U+001C-U+001E, U+0085,
# U+2028 and U+2029, so a line holding one of them was cut in two, the second half lost its "+" and
# was dropped, and Python and web/gate/diffgate.js (which splits on \r\n, \r and \n) returned opposite
# `tests_added` verdicts on the same diff. Every place a diff is split into lines -- both doors,
# `gate_diff_text` and `gate_diff` -- splits exactly as the port does.
_DIFF_LINE_BREAK = re.compile(r"\r\n|\r|\n")


def _diff_lines(text: str) -> list:
    """`text` split on \\r\\n, \\r and \\n only, with no trailing empty line: `str.splitlines()` for the
    three line endings a diff can carry, and nothing else."""
    lines = _DIFF_LINE_BREAK.split(text)
    if lines and lines[-1] == "":
        lines.pop()
    return lines


# NOTE_path2_sixth_pass_2026_09_25 (W-1): ONE hunk-aware reading of a diff. `parse_unified_diff`
# (the status map and the added blob `got` and `hit` read) and `parse_unified_diff_sides` (the
# per-file sides both pairings read) were two parsers that kept different lines, and neither counted a
# hunk: a removed line whose text opens with "-- " (a SQL, Lua or Haskell comment, an email signature)
# prints as `--- x`, and an added line opening with "++ " prints as `+++ x`. The sides parser read the
# removed one as a file header and dropped every later line of the file while the blob kept them, so a changed
# `def` read VERIFIED as added (#101's kind); both read the second as a header, a phantom path the git
# door never saw. Now both come from `_read_diff`: a `---`/`+++` line is a header only OUTSIDE a hunk,
# and inside one the `@@ -a,b +c,d @@` counts say how many removed and added lines are still owed. A
# line the counts do not allow for (a `diff --git` header, a hunk that ends early) closes the hunk and
# is read as before; outside any counted hunk every line is read exactly as it was, so a hand-written
# diff with no `@@` header reads as it did on main. A hand-written hunk often declares more lines than
# it carries, and the next file's `---`/`+++` pair then falls inside its counts; the sixth pass kept such
# a pair a file header through an exception (`_header_pair`), which the seventh pass replaces (below).
#
# The same numbers say which line is line 1 of a file, the one place CPython reads a U+FEFF (a byte-order
# mark opens a file; anywhere else it is a character CPython refuses). The parse drops a U+FEFF that
# opens line 1 of either side, and nowhere else, and the definition patterns read none: R-1 had them take
# one on any line, so a definition led by U+FEFF in the middle of a file counted, which `main` did not.
#
# NOTE_path2_seventh_pass_2026_09_25: the counts are trusted only for a hunk that CARRIES what it declares.
# The sixth pass trusted every hunk's counts and kept main's reading through a header-pair exception; a
# hand-written hunk that declares more lines than it carries, followed by a next file written without
# `a/`/`b/` and without a numeric `@@` right after its header, read that header as a removed and an added
# line: the next file never entered the status map and its lines were filed under the previous file (new
# false VERIFIEDs and CONTRADICTEDs against main on the raw door and the port). Now each hunk is scanned
# before it is read (`_hunk_is_exact`): its counts are walked over the lines that follow, a `---`/`+++`
# pair followed by a hunk header ending the walk, and so does a `--- ` line right after an added line --
# git, `diff -u` and difflib write each change's removed lines before its added ones, so a `---` line
# there is a file header, not a removed line. The hunk is EXACT when the walk closes and what follows can
# end a hunk -- the end of the diff, a `diff --git` line, a file header, a hunk header that moves forward
# in the same file, or a line no hunk carries. An exact hunk is read by its counts, with no exception: a
# `---`/`+++` line inside it is content, `-- a/x` beside `++ b/x` included. Any other hunk -- one that
# carries fewer lines than it declares, or whose counts close on a line that goes on as a hunk would, or
# that orders its lines as no generator does -- is read exactly as main read it, line by line. git
# writes only exact hunks.
_HUNK_HEADER = re.compile(r"^@@ -([0-9]+)(?:,([0-9]+))? \+([0-9]+)(?:,([0-9]+))? @@")   # [0-9]: `\d` is Unicode
_FILE_BOM = "\uFEFF"


def _hunk_counts(m) -> tuple:
    """(old start, old count, new start, new count) of a hunk header match; an omitted count is 1."""
    return (int(m.group(1)), 1 if m.group(2) is None else int(m.group(2)),
            int(m.group(3)), 1 if m.group(4) is None else int(m.group(4)))


def _hunk_is_exact(lines: list, k: int) -> bool:
    """Whether the hunk whose header is `lines[k]` carries exactly the lines its counts declare, so that
    reading it by its counts cannot read the next file's header as content (NOTE_path2_seventh_pass)."""
    a, b, c, d = _hunk_counts(_HUNK_HEADER.match(lines[k]))
    old_left, new_left = b, d
    j = k + 1
    after_added = False                          # the last counted line was an added one
    while old_left or new_left:
        if j >= len(lines):
            return False                         # the diff ends before the counts close: too many declared
        line = lines[j]
        head = line[:1]
        if (line.startswith("--- ") and j + 2 < len(lines) and lines[j + 1].startswith("+++ ")
                and lines[j + 2].startswith("@@")):
            return False                         # a file header and its hunk end the walk
        if after_added and line.startswith("--- "):
            return False                         # git and diff -u write removed lines before added ones
        if head == "+" and new_left:
            new_left -= 1
            after_added = True
        elif head == "-" and old_left:
            old_left -= 1
        elif (head == " " or line == "") and old_left and new_left:
            old_left -= 1
            new_left -= 1
            after_added = False
        elif head != "\\":                       # "\ No newline at end of file" is not counted
            return False                         # the counts do not allow this line
        j += 1
    while j < len(lines) and lines[j].startswith("\\"):
        j += 1
    blank = j
    while j < len(lines) and lines[j] == "":
        j += 1
    if j >= len(lines):
        return True                              # the end of the diff
    line = lines[j]
    if line.startswith("diff --git "):
        return True
    if line.startswith("--- ") and j + 1 < len(lines) and lines[j + 1].startswith("+++ "):
        return True                              # the next file's header
    if line.startswith("@@"):
        m = _HUNK_HEADER.match(line)
        if m is None or j != blank:
            return False
        a2, _b2, c2, _d2 = _hunk_counts(m)
        return a2 >= a + b and c2 >= c + d       # the same file, further on, as git writes it
    if line == "-- " and (j + 1 >= len(lines) or lines[j + 1][:1] not in ("+", "-", " ", "@", "\\")):
        return True                              # a `git format-patch` signature
    return line[:1] not in ("+", "-", " ", "\\")   # a line no hunk carries


def _dev_null(path) -> bool:
    """NOTE_path2_eighth_pass (Y-4): a header path naming /dev/null -- as git writes it, or as GNU diff does,
    followed by a TAB and a timestamp. main compared the whole string, so `+++ /dev/null<TAB>2024-...` keyed a
    deletion as "dev/null<TAB>2024-..." (two deletions shared that key) and `--- /dev/null<TAB>...` read a
    created file as modified."""
    return path == "/dev/null" or (path or "").startswith("/dev/null\t")


def _header_shape(path: str) -> str:
    """Y-1: a header path as a header's shape is compared: cut at a TAB (GNU's timestamp), then `a/` or `b/`
    dropped."""
    p = path.split("\t", 1)[0].strip()
    return p[2:] if p.startswith(("a/", "b/")) else p


def _clean_header(lines: list, k: int) -> bool:
    """Y-1: whether `lines[k]` (a `--- ` line) opens a pair with a file header's shape: a `+++ ` line, then a
    line opening `@@`, the two naming the same path or one of them /dev/null. A pair of a removed `-- X` and an
    added `++ Y` read outside the counts has that shape only when X and Y are the same text."""
    if not (k + 2 < len(lines) and lines[k + 1].startswith("+++ ") and lines[k + 2].startswith("@@")):
        return False
    x, y = _header_shape(lines[k][4:]), _header_shape(lines[k + 1][4:])
    return x == y or "/dev/null" in (x, y)


def _line_one_bom(text: str, at_one: bool) -> str:
    """Y-2: outside the counts, a U+FEFF is dropped from a line the diff shows is line 1 of its side (`at_one`),
    where CPython reads one; W-1 drops it inside the counts."""
    return text[len(_FILE_BOM):] if at_one and text.startswith(_FILE_BOM) else text


_Y1_LOOSE = ("a `---` or `+++` line after lines no hunk count holds may be content (a SQL or Lua comment, "
             "a `++` line) or a file header")
_Y1_COLLIDE = "two header paths that differ only in case are one key"
_Y1_UNCOUNTED = ("a line names a changed file no header pair counts (GNU's `Binary files ... differ`, "
                 "`Only in ...` and the like)")
_UNCOUNTED = re.compile(r"^(?:(?:Binary files|Files|Symbolic links) .+ and .+ differ|Only in .+: .+|File .+ is a .+ while file .+ is a .+)$")


def _read_diff(diff_text: str, notes: dict | None = None) -> tuple[dict, list, dict]:
    """Unified diff text -> (status map, added lines in order, per-file sides): the one reading W-1
    gives both parsers. An added line outside any file (no `+++` header before it) is in the blob and in
    no file's sides, exactly as before.

    NOTE_path2_eighth_pass: `notes`, when given, is filled with what this reading cannot be sure of: "files"
    when a header was read that may be content, two paths are one key, or a line names a changed file no header
    pair counts (Y-1); "bom" when a U+FEFF was dropped from an added line 1 that defines a test (Y-2).
    NOTE_path2_ninth_pass: and "differs" when this file list is not licensed against main's (Z-3)."""
    status: dict[str, str] = {}
    added: list[str] = []
    sides: dict = {}
    old_path = None
    cur = None
    pending: _Pending | None = None          # BIN-1: a header still waiting for its pair
    old_left = new_left = 0                  # removed and added lines the open hunk still owes
    old_no = new_no = 0                      # the line numbers its next removed and added lines carry
    loose = False                            # Y-1: a line no count placed was read since the last `diff --git`
    clean_plus = -1                          # Y-1: the `+++ ` line of a header pair read with a header's shape
    lead_old = lead_new = False            # Y-2: the next removed / added line read outside the counts is line 1
    found: dict = {}
    forms: dict = {}                         # Y-1: each header path as written, case kept, by its fold (K-2)
    inside: set = set()                      # Z-3: the lines an exact hunk's counts read (W-1)
    soft: list = []                          # Z-3: doubts main's reading of the file list also held
    binary = False                           # K-4: inside a `GIT binary patch` block (until a blank line)

    def register(raw_path: str) -> None:
        # Y-1: the key lower-cases, so two files whose paths differ only in case are one key -- one count, one
        # side -- on main too; a count main's other errors balanced is not sure. NOTE_path2_tenth_pass (K-2): two
        # paths are compared by the one fold both ports carry, not by this runtime's lower-casing (the Pythons and
        # the port read 67 code points differently); every pair any supported runtime keys alike folds alike
        form = _LEADING_SLASH_SEGMENTS.sub("", raw_path.replace("\\", "/"))
        if forms.setdefault(_case_fold(form), form) != form:
            found.setdefault("files", _Y1_COLLIDE)

    def flush() -> None:
        if pending is not None and pending.path():
            register(pending.a if pending.status == "D" else pending.b)
            if pending.path() not in status:
                status[pending.path()] = pending.status
            sides.setdefault(pending.path(), ([], []))
        elif pending is not None:
            soft.append(_Z3_UNREAD)                                   # Z-3: dropped, as main dropped it

    lines = _diff_lines(diff_text)
    for k, line in enumerate(lines):
        if old_left or new_left:
            head = line[:1]
            if head == "+" and new_left:
                text = line[1:]
                if new_no == 1 and text.startswith(_FILE_BOM):
                    text = text[len(_FILE_BOM):]
                    why = _bom_test_note(line[1:], text)           # Y-2
                    if why:
                        found.setdefault("bom", why)
                new_left -= 1
                new_no += 1
                added.append(text)
                if cur is not None:
                    sides[cur][0].append(text)
                inside.add(k)
                continue
            if head == "-" and old_left:
                text = line[1:]
                if old_no == 1 and text.startswith(_FILE_BOM):
                    text = text[len(_FILE_BOM):]
                old_left -= 1
                old_no += 1
                if cur is not None:
                    sides[cur][1].append(text)
                inside.add(k)
                continue
            if (head == " " or line == "") and old_left and new_left:
                old_left -= 1
                new_left -= 1
                old_no += 1
                new_no += 1
                inside.add(k)
                continue
            if head == "\\":                     # "\ No newline at end of file"
                inside.add(k)
                continue
            old_left = new_left = 0              # the counts do not allow this line: the hunk is over
        if line.startswith("diff --git "):
            flush()
            pending = _Pending(line)
            cur = None
            loose = binary = False
            lead_old = lead_new = False
        elif line.startswith("--- "):
            if loose:                            # Y-1: after lines no count placed, a header is not certain
                if _clean_header(lines, k):
                    clean_plus, loose = k + 1, False
                    soft.append(_Z3_SHAPED)      # Z-3: read as a header, as main read it, for its shape
                else:
                    found.setdefault("files", _Y1_LOOSE)
            old_path = line[4:].strip()
            cur = None
            lead_old = lead_new = False
        elif line.startswith("+++ "):
            if loose and k != clean_plus:
                found.setdefault("files", _Y1_LOOSE)
            new = line[4:].strip()
            if pending is not None and pending.path() and \
                    _pair_names((old_path or "") if _dev_null(new) else new) not in (_norm(pending.a), _norm(pending.b)):
                soft.append(_Z3_REPLACED.format(_shown(pending.path())))   # Z-3: dropped, as main dropped it
            if _dev_null(new) and old_path is None:
                # NOTE_path2_tenth_pass (K-3): a deletion with no `---` line before it names no file, and is not read
                # as one. main raised on it where its own reading held no `---` line either (every claim then abstains:
                # Z-1 to Z-3 read main as raising); where main read one this reading does not -- inside an exact hunk,
                # or before a `+++ /dev/null<TAB>` it did not read as /dev/null -- Z-3 abstains the file-list claims.
                # This reading raised there, where main did not.
                cur = None
            else:
                if _dev_null(new):               # Y-4: a GNU timestamp after /dev/null
                    status[_norm(old_path[2:] if old_path.startswith("a/") else old_path)] = "D"
                    raw = old_path[2:] if old_path.startswith("a/") else old_path
                else:
                    raw = new[2:] if new.startswith("b/") else new
                    status[_norm(raw)] = "A" if (old_path is None or _dev_null(old_path)) else "M"
                cur = _norm(raw)
                register(raw)
                sides.setdefault(cur, ([], []))
            pending = None
            # Y-2: a created file's opening added line, a deleted file's opening removed line, is line 1
            lead_new = old_path is not None and _header_shape(old_path) == "/dev/null"
            lead_old = _header_shape(new) == "/dev/null"
        elif line.startswith("@@") and _HUNK_HEADER.match(line):
            if _hunk_is_exact(lines, k):         # NOTE_path2_seventh_pass: else read as main read it
                old_no, old_left, new_no, new_left = _hunk_counts(_HUNK_HEADER.match(line))
                lead_old = lead_new = False
            else:
                loose = True
                a, _b, c, _d = _hunk_counts(_HUNK_HEADER.match(line))
                lead_old, lead_new = lead_old or a == 1, lead_new or c == 1
        elif line.startswith("+") and not line.startswith("+++"):
            loose = True
            text = _line_one_bom(line[1:], lead_new)
            why = _bom_test_note(line[1:], text)                   # Y-2
            if why:
                found.setdefault("bom", why)
            lead_new = False
            added.append(text)
            if cur is not None:
                sides[cur][0].append(text)
        elif line.startswith("-") and not line.startswith("---"):
            loose = True
            text = _line_one_bom(line[1:], lead_old)
            lead_old = False
            if cur is not None:
                sides[cur][1].append(text)
            elif pending is not None:
                pending.note(line)
        else:
            if line.startswith("@@") or line.startswith(" ") or line == "":
                loose = True                     # a hunk header with no counts, a context line, a blank
                if not line.startswith("@@"):
                    lead_old = lead_new = False
                if line == "":
                    binary = False
            elif _UNCOUNTED.match(line) and not (pending is not None and _BINARY_LINE.match(line)):
                found.setdefault("files", _Y1_UNCOUNTED)   # Y-1: a changed file no header pair counts
            elif line == "GIT binary patch" or _BINARY_PATCH.match(line):
                binary = True                    # K-4: its `literal N`/`delta N` blocks, each ended by a blank line
            elif not (binary or line.startswith("\\") or _GIT_META.match(line)
                      or (pending is not None and _BINARY_LINE.match(line))):
                soft.append(_Z3_UNPLACED)        # K-4: a line no reading places may name a file neither counts
            if pending is not None:
                pending.note(line)
    flush()
    if notes is not None:
        why = _file_list_differs(diff_text, lines, status, inside, soft)   # Z-3
        if why:
            found["differs"] = why
        notes.update(found)
    return status, added, sides


_Z3_SHAPED = ("main's reading also took a `---`/`+++` pair after lines no hunk count holds for a header because it "
              "has a header's shape, and it may be content (a SQL `-- ` comment beside a `++` line)")
_Z3_REPLACED = ("main's reading also dropped the `diff --git` file {} for the next `---`/`+++` pair, which names "
                "another")
_Z3_UNREAD = "main's reading also dropped a `diff --git` file whose header paths neither reading can read"
# NOTE_path2_tenth_pass (K-4): a line outside every header, hunk and git extended header that neither reading places --
# git's `Submodule p a..b` under diff.submodule=log, svn's `Index:` and `Cannot display` blocks, hg's `diff -r` and
# `Binary file p has changed`, and formats not listed -- may name a changed file neither reading counts; beside a
# licensed difference (#121's dotted key) the repair may have removed the error that balanced it.
_Z3_UNPLACED = ("main's reading also passed over a line no reading places, which may name a changed file neither "
                "reading counts (git's `Submodule` line, svn's and hg's binary notices, and the like)")
_BINARY_PATCH = re.compile(r"^(?:literal|delta) [0-9]+$")      # a block of git's binary patch (git-diff(1) --binary)
# git's extended header lines (git-diff(1), "generating patch text with -p"), which both readings read as a header's
_GIT_META = re.compile(r"^(?:index |old mode |new mode |deleted file mode |new file mode |copy from |copy to |"
                       r"rename from |rename to |similarity index |dissimilarity index )")


def _pair_names(raw: str) -> str:
    """Z-3: the file a `---`/`+++` header path names: cut at a TAB (GNU's timestamp), quotes and `a/`/`b/` dropped,
    keyed."""
    p = raw.split("\t", 1)[0].strip()
    if len(p) >= 2 and p.startswith('"') and p.endswith('"'):
        p = p[1:-1]
    return _norm(p[2:] if p.startswith(("a/", "b/")) else p)


def _diff_notes(diff_text: str) -> dict:
    """NOTE_path2_eighth_pass: what the one reading of `diff_text` cannot be sure of -- {"files": why} (Y-1),
    {"bom": why} (Y-2), each only when it holds."""
    notes: dict = {}
    _read_diff(diff_text, notes)
    return notes


def _status_notes(paths: list) -> dict:
    """NOTE_path2_eighth_pass (Y-1), the git door: git's `--name-status` is a sure file list, except where two of
    its paths differ only in case, which the key reads as one file. NOTE_path2_tenth_pass (K-2): compared by the one
    fold both ports carry, as the raw door compares them."""
    forms: dict = {}
    for p in paths:
        form = _LEADING_SLASH_SEGMENTS.sub("", p.replace("\\", "/"))
        forms.setdefault(_case_fold(form), set()).add(form)
    return {"files": _Y1_COLLIDE} if any(len(v) > 1 for v in forms.values()) else {}


def _files_unsure(notes: dict | None):
    """NOTE_path2_eighth_pass (Y-1): why the file list a gate reads is not sure, or None -- a count, a scope and a
    path claim then abstain."""
    return (notes or {}).get("files")


# ══════════════════════════════════════════════════════════════════════════════
# NOTE_path2_ninth_pass_2026_09_27: THE LICENSED-DIFFERENCE RULE
# ══════════════════════════════════════════════════════════════════════════════
#
# The endpoint of the eighth pass's convergence principle. Eight rounds each found a place where a repair read a
# line rightly and a claim wrongly, because main's answer there had been right by two errors that cancelled and the
# repair removed one of them. A list of the shapes where that happens has never been complete. So main's own reading
# of the diff is now computed beside this one -- main's patterns over main's line split and main's status map -- in
# each of main's two spellings: its Python's (str.splitlines(), Python's `\s`, `\w` and `\b`, a line start only after
# \n) and its port's (\r\n, \r and \n; JavaScript's `\s`, ASCII `\w` and `\b`, a line start after U+2028 and U+2029
# too). Both ports compute both spellings, from the same code point lists, so the condition is the same in both.
# Where this reading differs from either and no named repair licenses the difference -- #97's exact or suffix tier,
# #121's dotted key, #101's one-to-one pairing, W-1's exact hunk, each with its own precondition -- the claim
# ABSTAINS, with a reason that names the difference:
#   Z-1  tests_added: `got` is not main's `^\s*def test_` count, or BC-1's "a Python file" is not main's.
#   Z-2  symbol_added: `hit` is not main's `^\s*(?:def|class)\s+NAME\b`, or BC-1's answer is not main's.
#   Z-3  the file list: this status map is not main's up to #121's dotted keys and W-1's exact hunks; main's two
#        ports read it apart; or it differs from main's by one of those repairs where main's reading also held a
#        doubt of its own (a header read after lines no count placed because it had a header's shape, a `diff --git`
#        file its next `---`/`+++` pair replaced, a `diff --git` header whose paths cannot be read), which the repair
#        may have been balancing. A count, a scope and a path claim then abstain, as for Y-1.
#   Z-4  a path claim with a directory component that only the basename tier matches (a file with the same name in
#        another directory): main's single loop matched it too, and #97 licenses only the exact and suffix tiers.
#   Z-5  the whole-file reading: this reading reads a file line by line, CPython whole. An added definition line in a
#        Python file (or outside any file) that this reading refuses -- CPython may refuse it too, and a file CPython
#        refuses defines nothing -- makes tests_added abstain, and symbol_added where the claimed name's definition
#        is in that file.
# None of them gives a verdict main could not give: each only turns this reading's verdict into an abstention.
#
# NOTE_path2_tenth_pass_2026_09_28: the licences are #97's exact and suffix tiers, #121's dotted key and #101's
# pairing, and no other. W-1's exact hunk licensed a file main read from content being removed, and that file had
# balanced a changed file neither reading counts (git's `Submodule` line, svn's and hg's binary notices): K-1 abstains
# the file list wherever main's reading with an exact hunk's lines is not its reading without them. Beside #121's
# dotted key, a line no reading places is such a doubt too (K-4). Y-1's collision compares two header paths by one
# fixed case fold (styxx/_fold.py), not by the runtime's lower-casing (K-2). A `+++ /dev/null` with no `---` line
# before it names no file instead of raising where main did not (K-3). A path claim the two ports' templates may
# read apart reads as main read it (K-5). A key a Z-3 or Z-4 reason prints is folded and ascii()-escaped, so the
# reason is the same on every runtime.


def _chars(cps) -> str:
    return "".join(map(chr, cps))


_ZWS = tuple(range(0x2000, 0x200B))                  # U+2000 to U+200A
# Python's `\s` for a str pattern, which is str.isspace() and what str.strip() strips (the same on 3.9 to 3.14).
_PY_SPACE = _chars((0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x1C, 0x1D, 0x1E, 0x1F, 0x20, 0x85, 0xA0, 0x1680) + _ZWS
                   + (0x2028, 0x2029, 0x202F, 0x205F, 0x3000))
# JavaScript's `\s`, which is what String.prototype.trim() strips.
_JS_SPACE = _chars((0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20, 0xA0, 0x1680) + _ZWS
                   + (0x2028, 0x2029, 0x202F, 0x205F, 0x3000, 0xFEFF))
_PY_SPACE_CLS = "[" + re.escape(_PY_SPACE) + "]"
_JS_SPACE_CLS = "[" + re.escape(_JS_SPACE) + "]"
# str.splitlines()'s line boundaries, and the characters after which JavaScript's multiline `^` matches.
_PY_LINE_BREAK = re.compile("\r\n|[" + re.escape(_chars((0x0A, 0x0B, 0x0C, 0x0D, 0x1C, 0x1D, 0x1E, 0x85,
                                                           0x2028, 0x2029))) + "]")
_JS_TERMINATORS = _chars((0x0A, 0x0D, 0x2028, 0x2029))
_JS_DOT = "[^" + re.escape(_JS_TERMINATORS) + "]"                # JavaScript's `.`
_JS_SEGMENT = re.compile("[" + re.escape(_chars((0x2028, 0x2029))) + "]")
_ASCII_WORD = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
# main's port's BIN-1 patterns, with JavaScript's `.` (the Python's are _DIFF_GIT and _BINARY_LINE above).
_DIFF_GIT_JS = re.compile(r'^diff --git (?:"a/(?P<qa>(?:[^"\\]|\\' + _JS_DOT + r')*)"|a/(?P<a>' + _JS_DOT
                          + r'*?)) (?:"b/(?P<qb>(?:[^"\\]|\\' + _JS_DOT + r')*)"|b/(?P<b>' + _JS_DOT + r'*))$')
_BINARY_LINE_JS = re.compile(r"^Binary files (?P<a>" + _JS_DOT + r"+?) and (?P<b>" + _JS_DOT + r"+?) differ$")


def _py_lines(text: str) -> list:
    """main's Python's split: str.splitlines(), spelled out (the port carries the same list)."""
    lines = _PY_LINE_BREAK.split(text)
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _main_key(path: str) -> str:
    """main's `_norm`, before #121: the backslashes turned, every leading dot and slash stripped, lower-cased."""
    return path.replace("\\", "/").lstrip("./").lower()


class _MainPending:
    """main's BIN-1 pending header, in one of main's spellings (`js`: JavaScript's `.` in its two patterns)."""
    __slots__ = ("a", "b", "status", "js")

    def __init__(self, line: str, js: bool):
        self.js = js
        self.a, self.b = self._paths(line)
        self.status = "M"

    def _paths(self, line: str) -> tuple:
        body = line[len("diff --git "):]
        if len(body) % 2 == 1:
            mid = len(body) // 2
            if body[mid] == " " and body[:mid].startswith("a/") and body[mid + 1:].startswith("b/") \
                    and body[2:mid] == body[mid + 3:]:
                return body[2:mid], body[mid + 3:]
        m = (_DIFF_GIT_JS if self.js else _DIFF_GIT).match(line)
        if not m:
            return "", ""
        return (m.group("qa") if m.group("qa") is not None else (m.group("a") or ""),
                m.group("qb") if m.group("qb") is not None else (m.group("b") or ""))

    def note(self, line: str) -> None:
        if line.startswith("new file mode"):
            self.status = "A"
        elif line.startswith("deleted file mode"):
            self.status = "D"
        elif line.startswith("rename from "):
            self.a = line[len("rename from "):]
        elif line.startswith("rename to "):
            self.b = line[len("rename to "):]
        else:
            m = (_BINARY_LINE_JS if self.js else _BINARY_LINE).match(line)
            if m:
                if m.group("a") == "/dev/null":
                    self.status = "A"
                elif m.group("b") == "/dev/null":
                    self.status = "D"

    def key(self) -> str:
        raw = self.a if self.status == "D" else self.b
        return _main_key(raw) if raw else ""


def _main_status(lines: list, js: bool, skip=frozenset()):
    """main's `parse_unified_diff` status map over `lines`, keyed by main's `_norm`, in main's Python's spelling or
    (`js`) its port's; a line whose index is in `skip` is not read. None where main raises (a `+++ /dev/null` with no
    `---` line before it)."""
    space = _JS_SPACE if js else _PY_SPACE
    status: dict = {}
    old_path = None
    pending = None
    for i, line in enumerate(lines):
        if i in skip:
            continue
        if line.startswith("diff --git "):
            if pending is not None and pending.key() and pending.key() not in status:
                status[pending.key()] = pending.status
            pending = _MainPending(line, js)
        elif line.startswith("--- "):
            old_path = line[4:].strip(space)
        elif line.startswith("+++ "):
            new = line[4:].strip(space)
            if new == "/dev/null":
                if old_path is None:
                    return None
                status[_main_key(old_path[2:] if old_path.startswith("a/") else old_path)] = "D"
            else:
                status[_main_key(new[2:] if new.startswith("b/") else new)] = \
                    "A" if old_path in ("/dev/null", None) else "M"
            pending = None
        elif line.startswith("+") and not line.startswith("+++"):
            continue
        elif pending is not None:
            pending.note(line)
    if pending is not None and pending.key() and pending.key() not in status:
        status[pending.key()] = pending.status
    return status


def _main_added(lines: list) -> list:
    """main's added lines: every line opening `+` and not `+++`, in both doors and both ports."""
    return [line[1:] for line in lines if line.startswith("+") and not line.startswith("+++")]


_PY_TEST_LINE = re.compile(_PY_SPACE_CLS + "*def test_")
_JS_TEST_LINE = re.compile(_JS_SPACE_CLS + "*def test_")


def _main_touches_python(status) -> bool:
    """main's BC-1 test on main's own keys."""
    return status is not None and any(p.lower().endswith(_PY_SUFFIXES) for p in status)


class _MainReading:
    """NOTE_path2_ninth_pass: main's reading of one diff's added lines, in its two spellings, and BC-1's answer on
    main's file list(s). `raises` when main's reading of the file list raises."""
    __slots__ = ("added_py", "added_js", "tests", "python", "raises", "maps")

    def __init__(self, diff_text: str, maps: tuple):
        self.maps = maps                     # NOTE_path2_tenth_pass (K-5): main's file list(s), its Python's leading
        self.added_py = _main_added(_py_lines(diff_text))
        self.added_js = _main_added(_diff_lines(diff_text))
        # `^\s*def test_` over the added lines joined by \n: a line start only after \n in the Python (re.M), also
        # after U+2028 and U+2029 in the port (/m); the leading `\s*` may run over empty lines, never past `def`.
        self.tests = (sum(1 for line in self.added_py if _PY_TEST_LINE.match(line)),
                      sum(1 for line in self.added_js for seg in _JS_SEGMENT.split(line) if _JS_TEST_LINE.match(seg)))
        self.raises = any(m is None for m in maps)
        self.python = tuple(_main_touches_python(m) for m in maps)


def _main_reading(diff_text: str) -> "_MainReading":
    """The raw door's: main's two readings of the diff text, its file list in each spelling."""
    return _MainReading(diff_text, (_main_status(_py_lines(diff_text), False),
                                    _main_status(_diff_lines(diff_text), True)))


def _main_names(sentence: str, start: int) -> tuple:
    """(main's Python's name, main's port's name) for a symbol claim whose template `name` group starts at `start`:
    `[A-Za-z_]\\w*`, with Python's `\\w` (the table's) and with JavaScript's (ASCII, "" where it cannot open)."""
    j = start + 1
    while j < len(sentence) and _xid_word(sentence[j]):
        j += 1
    k = start
    while k < len(sentence) and sentence[k] in _ASCII_WORD:
        k += 1
    opens = start < len(sentence) and sentence[start] in _ASCII_WORD and not sentence[start].isdigit()
    return sentence[start:j], (sentence[start:k] if opens else "")


def _main_symbol_hit(name_py: str, name_js: str, main: "_MainReading") -> tuple:
    """main's `hit` in each spelling, and a code point its Python's `\\b` turned on that the supported Pythons read
    differently (None when there is none): `^\\s*(?:def|class)\\s+NAME\\b` over the added lines joined by \\n."""
    py, skew_cp = False, None
    if name_py:
        blob = "\n".join(main.added_py)
        rx = re.compile("^" + _PY_SPACE_CLS + "*(?:def|class)" + _PY_SPACE_CLS + "+" + re.escape(name_py), re.M)
        for m in rx.finditer(blob):
            nxt = blob[m.end():m.end() + 1]
            if nxt and _skew(nxt):
                skew_cp = skew_cp if skew_cp is not None else ord(nxt)
            elif not nxt or not _xid_word(nxt):
                py = True
    js = False
    if name_js:
        blob = "\n".join(main.added_js)
        rx = re.compile("(?:(?<![\\s\\S])|(?<=[" + re.escape(_JS_TERMINATORS) + "]))" + _JS_SPACE_CLS
                        + "*(?:def|class)" + _JS_SPACE_CLS + "+" + re.escape(name_js) + "(?![A-Za-z0-9_])")
        js = rx.search(blob) is not None
    return py, js, (None if py else skew_cp)


def _python_differs(status: dict, main: "_MainReading | None"):
    """Z-1, Z-2: why BC-1's "the diff holds a Python file" is not main's (main raises, or answers otherwise), else
    None. Only asked where this reading holds a Python file; where it holds none it abstains already."""
    if main is None:
        return None
    if main.raises:
        return "main raises on this diff (`+++ /dev/null` with no `---` line before it)"
    if not all(main.python):
        return ("this reading finds a Python file in the diff's file list where main's reading of it found none "
                "(BC-1 read on main's keys); no repair licenses the difference")
    return None


def _tests_differ(got: int, status: dict, main: "_MainReading | None"):
    """Z-1: why `got`, or BC-1's answer, is not main's in both of main's spellings, else None. #101's pairing is
    licensed only where it pairs lines this count and main's read alike, so it is asked after this."""
    if main is None:
        return None
    why = _python_differs(status, main)
    if why:
        return why
    py, js = main.tests
    if got == py == js:
        return None
    return (f"this reading counts {got} added test definitions where main's Python counted {py} and its port {js} "
            "(`^\\s*def test_` over main's line split); no repair licenses the difference")


def _symbol_differs(hit: bool, name: str, name_py: str, name_js: str, status: dict, main: "_MainReading | None"):
    """Z-2: why `hit`, or BC-1's answer, is not main's in both of main's spellings, else None."""
    if main is None:
        return None
    why = _python_differs(status, main)
    if why:
        return why
    py, js, skew_cp = _main_symbol_hit(name_py, name_js, main)
    if skew_cp is not None:
        return (f"main's Python read a definition of {_qname(name_py)} through `\\b` before U+{skew_cp:04X}, which "
                f"{_Y3_VERSIONS} read differently")
    if py == hit and js == hit:
        return None
    return (f"this reading finds {'an' if hit else 'no'} added definition of {_qname(name)} where main's "
            f"Python {'did' if py else 'did not'} and its port {'did' if js else 'did not'} "
            "(`^\\s*(?:def|class)\\s+NAME\\b` over main's line split); no repair licenses the difference")


_ANY_SPACE_CLS = "[" + re.escape(_PY_SPACE + _chars((0xFEFF,))) + "]"     # either spelling's `\s`, and U+FEFF
_LOOSE_DEF = re.compile("^" + _ANY_SPACE_CLS + "*(?:async" + _ANY_SPACE_CLS + "+)?(?:def|class)" + _ANY_SPACE_CLS + "+")


def _refused_definition(line: str) -> bool:
    """Z-5: the line opens a definition of a name when read loosely -- any whitespace either of main's spellings
    reads, or U+FEFF, around `def`/`class` (`async` too) -- and this reading refuses it: a character CPython does not
    take for indentation or a separator, or a name running into one no identifier holds. CPython may refuse such a
    line (a supported Python may accept a skew code point), and a file it refuses defines nothing."""
    m = _LOOSE_DEF.match(line)
    if m is None:
        return False
    i = m.end()
    if i >= len(line) or not (_xid_opens(line[i]) or _skew(line[i])):
        return False
    return _defined_name(line, removed=True) is None


def _stray_lines(added_blob: str, sides: dict | None) -> list:
    """The added lines outside any file (no `+++` header before them), in order: the blob less every file's side."""
    left = Counter(line for added, _removed in (sides or {}).values() for line in added)
    out = []
    for line in added_blob.split("\n"):
        if left[line] > 0:
            left[line] -= 1
        else:
            out.append(line)
    return out


def _refused_files(added_blob: str, sides: dict | None) -> dict:
    """Z-5: {file: its first added line `_refused_definition` reads} over the Python files (BC-1's suffix on the
    undotted key) and, as the file None, the added lines outside any file."""
    out: dict = {}
    for path, (added, _removed) in (sides or {}).items():
        if _undotted(path).lower().endswith(_PY_SUFFIXES):
            line = next((x for x in added if _refused_definition(x)), None)
            if line is not None:
                out[path] = line
    line = next((x for x in _stray_lines(added_blob, sides) if _refused_definition(x)), None)
    if line is not None:
        out[None] = line
    return out


def _refused_why(path) -> str:
    """Z-5's reason. The line is not printed: repr() would ask the runtime which characters it can print."""
    where = "outside any file" if path is None else f"in {path!r}"
    return (f"an added definition line {where} is one this reading refuses and CPython may refuse too, and a file "
            "CPython refuses defines nothing; this reading reads it line by line")


def _whole_file_tests(added_blob: str, sides: dict | None):
    """Z-5 for tests_added, which counts over every file: why, else None."""
    refused = _refused_files(added_blob, sides)
    return _refused_why(next(iter(refused))) if refused else None


def _whole_file_symbol(name: str, added_blob: str, sides: dict | None):
    """Z-5 for symbol_added: why, where a file whose added lines define `name` holds a refused definition line."""
    refused = _refused_files(added_blob, sides)
    if not refused:
        return None
    for path, (added, _removed) in (sides or {}).items():
        if path in refused and any(_defines(x, name) for x in added):
            return _refused_why(path)
    if None in refused and any(_defines(x, name) for x in _stray_lines(added_blob, sides)):
        return _refused_why(None)
    return None


def _read_apart(claimed: str, before: str) -> bool:
    """NOTE_path2_tenth_pass (K-5): whether the two ports' path templates may read this claim's path apart. The path
    template's `\\w` is Python's (Unicode) here and ASCII in the port, and the `[^.!?\\n]{0,60}?` before it lets the
    port start the path past a character it cannot hold, so a path holding a non-ASCII character, or one the sentence
    runs into from a non-ASCII character, is another path in the other port (`Docs/<U+A7D0>/a.md` here, `/a.md`
    there). main read each in its two ports as that port's template reads it; this reading licenses no difference
    the two ports do not share, so such a claim reads as main read it (`_main_find`)."""
    return not claimed.isascii() or not before.isascii()


def _main_find(main_map: dict, claimed: str):
    """NOTE_path2_tenth_pass (K-5): main's own resolution of a path claim over main's file list -- one loop in diff
    order, the earliest entry the claim matches exactly, by suffix or by basename."""
    c = _main_key(claimed)
    for p, st in main_map.items():
        if p == c or p.endswith("/" + c) or Path(p).name == Path(c).name:
            return p, st
    return None, None


def _basename_only(status: dict, claimed: str):
    """Z-4: why a path claim with a directory component that only the basename tier matches abstains, else None."""
    c = _norm(claimed)
    if "/" not in c:
        return None
    p, st = _find_path(status, claimed)
    if p is None or p == c or p.endswith("/" + c):
        return None
    return (f"{claimed!r}: only a file with the same name in another directory is in the diff "
            f"({_shown(p)}, status {st!r}); #97 licenses the exact and suffix tiers only")


def _shown(key: str) -> str:
    """NOTE_path2_tenth_pass: a key as a Z-3 reason prints it, the same on every runtime -- folded by the one table (a
    key is the runtime's lower case, which the supported runtimes read differently for 67 code points) and escaped as
    ascii() escapes it (repr() asks the runtime which characters it can print)."""
    return ascii(_case_fold(key))


def _licensed_against(status: dict, main: dict):
    """Z-3: why this status map is not `main` (main's map, with W-1's exact hunks read as content) up to #121's
    dotted keys -- each key of this one, undotted, is one of main's, and main's status for it is one of theirs --
    else None."""
    groups: dict = {}
    for k, st in status.items():
        groups.setdefault(_undotted(k), []).append((k, st))
    for k, st in main.items():
        if k not in groups:
            return f"main reads {_shown(k)} ({st!r}), which this reading does not"
    for u, ks in groups.items():
        if u not in main:
            return f"this reading reads {_shown(ks[0][0])} ({ks[0][1]!r}), which main does not"
        if main[u] not in [st for _k, st in ks]:
            return f"main reads {_shown(u)} as {main[u]!r}, this reading {_shown(ks[0][0])} as {ks[0][1]!r}"
    return None


_Z3_PREFIX = "this reading's file list differs from main's"


def _apart(py: dict, js: dict) -> str:
    """Z-3: the first place main's Python's file list and its port's differ, in words."""
    for k, st in py.items():
        if k not in js:
            return f"main's Python reads {_shown(k)} ({st!r}), which its port does not"
        if js[k] != st:
            return f"main's Python reads {_shown(k)} as {st!r}, its port as {js[k]!r}"
    k, st = next((k, st) for k, st in js.items() if k not in py)
    return f"main's port reads {_shown(k)} ({st!r}), which its Python does not"


def _w1_moved(full, skipped) -> str:
    """K-1: the earliest place main's file list read with an exact hunk's lines (`full`) and without them (`skipped`)
    differ, in words."""
    if skipped is None:
        return "without those lines main raises (`+++ /dev/null` with no `---` line before it)"
    for k, st in full.items():
        if k not in skipped:
            return f"main reads {_shown(k)} ({st!r}) from them"
        if skipped[k] != st:
            return f"main reads {_shown(k)} as {st!r} with them, as {skipped[k]!r} without"
    k, st = next((k, st) for k, st in skipped.items() if k not in full)
    return f"main reads {_shown(k)} ({st!r}) only without them"


_K1_WHY = ("main read the file list from lines an exact hunk's counts hold, which W-1 reads as content ({}); a "
           "changed file neither reading counts may have balanced it, so W-1 licenses no file-list difference")


def _file_list_differs(diff_text: str, lines: list, status: dict, inside: set, soft: list):
    """Z-3 at the raw door: why this reading's file list is not licensed against main's, else None.

    NOTE_path2_tenth_pass (K-1): W-1's exact hunk no longer licenses a file-list difference. Where main's reading of the
    file list with the lines an exact hunk's counts hold is not its reading without them, main read a file (or a
    status) from content; that error may have balanced a changed file neither reading counts (git's `Submodule`
    line under diff.submodule=log, svn's `Cannot display` block, hg's `Binary file ... has changed`, and formats not
    listed), so the file-list claims abstain."""
    py, js = _main_status(_py_lines(diff_text), False), _main_status(lines, True)
    if py is None or js is None:
        return f"{_Z3_PREFIX}: main raises on it (`+++ /dev/null` with no `---` line before it)"
    if py != js:
        return (f"main's Python and its port read the file list apart (str.splitlines() breaks lines JavaScript "
                f"does not): {_apart(py, js)}")
    full, skipped = _main_status(lines, False), _main_status(lines, False, skip=inside)
    if full is None:
        return f"{_Z3_PREFIX}: main raises on it (`+++ /dev/null` with no `---` line before it)"
    if skipped != full:
        return f"{_Z3_PREFIX}: " + _K1_WHY.format(_w1_moved(full, skipped))
    why = _licensed_against(status, full)
    if why:
        return f"{_Z3_PREFIX} where no repair accounts for it: {why}"
    if soft and set(status) != set(js):
        repair = ("#121 keeps a dotfile's dot" if {_undotted(k) for k in status} == set(js)
                  else "W-1 reads an exact hunk's `---`/`+++` line as content")
        return f"{_Z3_PREFIX} by a repair ({repair}), and {soft[0]}; the repair may have balanced that error"
    return None


def _main_name_status(name_status: str) -> dict:
    """main's git-door file list: git's `--name-status` split by str.splitlines(), keyed by main's `_norm`."""
    main: dict = {}
    for line in _py_lines(name_status):
        parts = line.split("\t")
        if len(parts) >= 2:
            main[_main_key(parts[-1])] = parts[0][:1]
    return main


def _status_differs(main: dict, status: dict):
    """Z-3 at the git door: why this file list is not main's (`_main_name_status`) up to #121's dotted keys."""
    why = _licensed_against(status, main)
    return f"{_Z3_PREFIX} where no repair accounts for it: {why}" if why else None


def _files_differ(notes: dict | None):
    """Z-3: why the file list a gate reads is not licensed against main's, or None."""
    return (notes or {}).get("differs")


def parse_unified_diff_sides(diff_text: str) -> dict:
    """Unified diff text -> {normalized new-or-old path: (added_lines, removed_lines)}.

    The per-file companion of `parse_unified_diff`, added for COMPAT-1; the original's
    return shape is untouched because callers unpack it. A header without a `---`/`+++`
    pair (BIN-1) registers its path with empty sides. Read by `_read_diff`, the one
    hunk-aware reading `parse_unified_diff` also returns (NOTE_path2_sixth_pass, W-1).
    """
    return _read_diff(diff_text)[2]


def _compat_params(line: str, at: int) -> str | None:
    """The parameter list of a definition line: the text inside the first `(` at or after `at`,
    whitespace-collapsed, or `None` when the line has no `(` there. A list that does not close
    on the line is taken as far as the line goes, with `…` appended, so a re-flowed multi-line
    signature compares as changed only when its first line changed."""
    k = line.find("(", at)
    if k < 0:
        return None
    depth = 0
    for e in range(k, len(line)):
        if line[e] == "(":
            depth += 1
        elif line[e] == ")":
            depth -= 1
            if depth == 0:
                return re.sub(r"\s+", " ", line[k + 1:e]).strip()
    return re.sub(r"\s+", " ", line[k + 1:]).strip() + "…"


def _compat_removed_public_names(sides: dict) -> tuple[list, list, list]:
    """(removed public definitions not re-defined or referenced in the added lines of the same
    language, as (path, language, name, on_surface)), (signature changes, as (path, language,
    name, before, after)), (languages of the diff this reading covers)."""
    # AMENDMENT_path2 C-2: the suffix and scaffold tests read the undotted key -- the key before #121
    # -- so `.storybook/` stays scaffolding and a file named `.py` stays unread; paths print dotted.
    by_lang_added: dict = {}
    langs_present: list = []
    for path, (added, _removed) in sides.items():
        for lang, (sufs, _rx) in _COMPAT_LANGS.items():
            if _undotted(path).endswith(sufs):
                by_lang_added.setdefault(lang, []).extend(added)
                if lang not in langs_present:
                    langs_present.append(lang)
    dropped: list = []
    changed: list = []
    for path, (_added, removed) in sides.items():
        for lang, (sufs, rx) in _COMPAT_LANGS.items():
            if not _undotted(path).endswith(sufs):
                continue
            added_lines = by_lang_added.get(lang, [])
            ablob = "\n".join(added_lines)
            for line in removed:
                m = rx.match(line)
                if not m:
                    continue
                name = m.group("name")
                if name.startswith("_"):
                    continue                        # private by convention
                if re.search(r"\b" + re.escape(name) + r"\b", ablob):
                    # re-defined or still referenced: a change or a move. COMPAT-2 reads the
                    # re-definition's parameter list beside the removed one, and reports a
                    # difference; it is never a drop.
                    before = _compat_params(line, m.end("name"))
                    if before is not None:
                        afters = []
                        for al in added_lines:
                            am = rx.match(al)
                            if am and am.group("name") == name:
                                ap = _compat_params(al, am.end("name"))
                                if ap is not None:
                                    afters.append(ap)
                        if afters and before not in afters and (path, lang, name) not in [c[:3] for c in changed]:
                            changed.append((path, lang, name, before, afters[0]))
                    continue
                if (path, lang, name) not in [d[:3] for d in dropped]:
                    dropped.append((path, lang, name, not _COMPAT_SCAFFOLD.search(_undotted(path))))
    return dropped, changed, langs_present


def _compat_reading(sides: dict | None) -> tuple[str, str, dict]:
    """The one verdict this kind has (until COMPAT-2's panel licenses a second), with what the
    diff shows about the claim in the reason and the candidate flag in the detail."""
    empty = {"removed": [], "languages": [], "surface_removed": 0, "signature_changed": [],
             "compat2_candidate": False}
    if not sides:
        return ("UNCHECKABLE", "compatibility claimed; no per-file diff available to read "
                               "(behaviour beyond names not checked)", dict(empty))
    dropped, changed, langs = _compat_removed_public_names(sides)
    if not langs:
        return ("UNCHECKABLE", "compatibility claimed; no language this reading covers in the diff "
                               "(python, js/ts, go, rust, java)", dict(empty))
    sig = f"; {len(changed)} signature(s) changed" if changed else ""
    detail = {"removed": [{"path": p, "language": l, "name": n, "surface": sf} for p, l, n, sf in dropped],
              "languages": langs,
              "surface_removed": sum(1 for d in dropped if d[3]),
              "signature_changed": [{"path": p, "language": l, "name": n, "before": b, "after": a}
                                    for p, l, n, b, a in changed],
              "compat2_candidate": any(d[3] for d in dropped)}
    if not dropped:
        return ("UNCHECKABLE", "compatibility claimed; no public top-level definition removed "
                               f"({', '.join(langs)} read; behaviour beyond names not checked){sig}", detail)
    surface = [d for d in dropped if d[3]]
    scaffold = [d for d in dropped if not d[3]]
    named = surface or scaffold
    shown = ", ".join(f"{p}: {n}" for p, _l, n, _s in named[:_COMPAT_MAX_NAMED])
    more = f" (+{len(named) - _COMPAT_MAX_NAMED} more)" if len(named) > _COMPAT_MAX_NAMED else ""
    if surface:
        rest = f"; {len(scaffold)} more in test/example/internal code" if scaffold else ""
        why = (f"compatibility claimed; the diff removes {len(surface)} public definition(s) from the "
               f"surface, not re-defined in the added lines: {shown}{more}{rest}{sig}")
        verdict = "CONTRADICTED" if COMPAT2_LICENSED else "UNCHECKABLE"
    else:
        why = (f"compatibility claimed; {len(scaffold)} public definition(s) removed, all in "
               f"test/example/internal code: {shown}{more}{sig}")
        verdict = "UNCHECKABLE"
    return (verdict, why, detail)


_NON_FILE_NOUNS = frozenset({
    "node.js", "next.js", "express.js", "vue.js", "nuxt.js", "react.js",
    "angular.js", "ember.js", "backbone.js", "three.js", "d3.js", "chart.js",
    "moment.js", "jquery.js", "socket.io", "nest.js", "svelte.js", "alpine.js",
})


def _is_non_file_noun(claimed: str) -> bool:
    return ("/" not in claimed and "\\" not in claimed
            and claimed.lower() in _NON_FILE_NOUNS)


# V13 repair 1 (PREREG_v13_repair_2026_08_31): VERB-OBJECT BINDING. "Removed
# the helper FROM mantineTheme.ts" claims something about content INSIDE a file
# the diff shows as modified -- the verb binds to the helper, not to the file.
# EXTERNAL-1's largest surviving defect: such sentences read as deletions and
# were accused of not deleting the file. Containment prepositions demote
# creation/deletion claims to "touched"; "at" and bare direct objects are left
# alone, because "created the docs at path/x.md" really does claim that file.
_CONTAINMENT = re.compile(
    r"\b(?:from|in|inside|within|out\s+of|of)\s+"
    r"(?:the\s+|its\s+|this\s+)?[`\"']?$", re.I)


def _demoted_by_containment(sentence: str, m) -> bool:
    try:
        start = m.start("path")
    except (IndexError, re.error):
        return False
    return bool(_CONTAINMENT.search(sentence[max(0, start - 40):start]))


_PATH_KINDS = ("file_created", "file_deleted", "file_touched")

# A path mentioned after one of these is being REFERRED to, not claimed. Closed
# set, in the same spirit as the extension whitelist: the gate would rather miss
# a real lie than accuse a summary that told the truth.
_REFERENTIAL = (
    # comparative -- the path belongs to some OTHER change
    "same way", "same as", "same fix", "just like", "as in ", "similar to",
    "mirrors", "analogous", "cf.", "compare", "unlike", "whereas", "matching the",
    # deferred or explicitly excluded -- the path is NOT in this diff
    "staged", "unstaged", "uncommitted", "will be", "would be", "to be ",
    "follow-up", "followup", "next commit", "separate commit", "separately",
    "not in this", "left for", "deferred", "pending", "in a later", "later commit",
    "still needs", "yet to be", "planned", "TODO", "todo",
    # V13 repair 3 (PREREG_v13_repair_2026_08_31): NEGATION. A sentence saying a
    # file was NOT changed makes its absence from the diff the sentence coming
    # TRUE. EXTERNAL-1 caught the gate accusing "avoids the need to modify
    # tsconfig.json" because tsconfig.json was absent -- which is what the
    # sentence promised. Same pathway as every other referential cue: the path
    # is named, not claimed.
    "avoid", "avoids", "without modif", "without chang", "without touch",
    "without altering", "no need to", "does not modify", "does not change",
    "does not touch", "doesn't modify", "doesn't change", "doesn't touch",
    "not modified", "not changed", "not touched", "no changes to",
    "unchanged", "untouched", "preserves",
)
_REF_BEFORE = 110          # run-up inspected before the matched path
_REF_AFTER = 70            # and after it: "test.yml) is staged" puts the
                           # disclaimer on the far side of the filename


def _names_without_claiming(sentence: str, m) -> bool:
    """Is this path being referred to rather than claimed as changed?

    Both windows are inspected. The first version of this check only looked
    backwards and still accused *"(fetch-depth: 0 in test.yml) is staged"* —
    a sentence that says in words the file is not in this diff.

    A false negative here is a missed lie; a false positive is an accusation
    against someone who told the truth. Those are not symmetric, and this
    function is deliberately biased toward the first.
    """
    try:
        start, end = m.start("path"), m.end("path")
    except (IndexError, re.error):
        return False
    window = (sentence[max(0, start - _REF_BEFORE):start] +
              " " + sentence[end:end + _REF_AFTER]).lower()
    return any(k.lower() in window for k in _REFERENTIAL)


# ══════════════════════════════════════════════════════════════════════════════
# the tests_pass leg — two words, and the accusing branch is gone
# ══════════════════════════════════════════════════════════════════════════════
#
# Anything that consumes this tuple gets the whole vocabulary. There is no hidden
# member and no flag that adds one. See the module docstring for why the third
# word was deleted rather than gated.
_TESTS_PASS_VERDICTS = ("VERIFIED", "UNCHECKABLE")

# The functions that decide a tests_pass verdict, named here so that
# selfcheck_tests_pass_never_accuses() scans exactly them. The path-claim
# branches inside _gate are a DIFFERENT class with a different (published,
# withheld) history and are deliberately out of scope for that check.
_TESTS_PASS_FUNCTIONS = ("_evidence_leg", "_run_leg", "_tests_pass_verdict")

# PREREG_evidence_leg_2026_09_01 committed to replacing the old string, "no --run
# command supplied; the gate does not take the agent's word for test results",
# because it implied the remedy was to supply a shell string — the one thing this
# gate should not push a reader toward against an untrusted branch, and the thing
# capsule R4 refuses to seal. This note is APPENDED to the adjudicator's own
# words rather than replacing them, and it names the reading channel instead.
# Strictly more informative at an identical verdict.
_NO_EVIDENCE_NOTE = (
    "No test REPORT was handed to the gate. It does not take the agent's word "
    "for test results, so with nothing to read it declines — absence of evidence "
    "is not a contradiction. The channel that makes this readable is a report "
    "passed with --evidence: a JUnit XML, or better a test-result attestation "
    "whose subject names the head commit. Even then no signature is checked, and "
    "the answer is VERIFIED or UNCHECKABLE — there is no accusing verdict for "
    "this claim kind.")


def _evidence_leg(paths, commit: str | None) -> tuple[str, str]:
    """Read a `tests_pass` claim through styxx.evidence. VERIFIED or UNCHECKABLE.

    THE IMPORT IS LOCAL AND EVERY FAILURE IS CONTAINED, the same way `_gate`
    imports `styxx.claimdetect` and `styxx.undeclared` imports
    `parse_unified_diff`: the dependency direction stays one-way (diffgate ->
    evidence, never back), and a missing, unreadable or malformed evidence file
    can never break the rest of the gate. A claim that could not be read is
    UNCHECKABLE — not an exception, and never an accusation.

    NO PARSING AND NO VERDICT TABLE IS REIMPLEMENTED HERE. This function calls
    `load_evidence` and `adjudicate_tests_pass` and does nothing else with the
    bytes. A second copy of a JUnit reader would drift from the first, and a
    drifting second parser is what produced the correction this lab published on
    2026-08-31.

    NOTHING HERE READS ``ev["observed"]``. That is styxx.evidence's REPORT-ONLY
    band, and its own note says a caller that turns ``failing_tests > 0`` into a
    failing check "has reintroduced, without measurement, exactly the verdict
    this module declines to ship". `selfcheck_no_accusation` states its boundary
    as not being able to prove a caller has not done so. This is that caller, and
    `selfcheck_tests_pass_never_accuses()` below is the counterpart it asked for.

    THE TWO MONOTONICITY PROPERTIES, because this is the leg they are about:

      * **Monotone against the empty baseline — HOLDS.** Evidence never leaves a
        `tests_pass` claim worse than supplying none. That is the guarantee.
      * **Monotone under set extension — DOES NOT HOLD, deliberately.** Adding a
        source to an already-affirming set can demote VERIFIED to UNCHECKABLE,
        because `adjudicate_tests_pass` blocks VERIFIED on ANY unparsed source. A
        partial read DECLINES rather than AFFIRMS: nine parsed shards out of ten
        cannot certify "all tests pass", and affirming from an incomplete set is
        the failure mode this module exists to refuse. Do not "fix" this by
        letting a partial read affirm, and do not special-case empty files.

    So `--evidence green.xml` can be VERIFIED while `--evidence green.xml
    empty.xml` is UNCHECKABLE. Operators: supply COMPLETE evidence or supply
    NONE. The `why` returned here carries the adjudicator's own count of how many
    of how many sources were unreadable, and names one of them, so a sharded CI
    matrix that lost one report can see which one from the failing build.
    """
    paths = [str(p) for p in (paths or [])]
    try:
        from styxx.evidence import SPEC, adjudicate_tests_pass, load_evidence
    except Exception as exc:                      # pragma: no cover - env-dependent
        return "UNCHECKABLE", (
            f"{len(paths)} evidence file(s) were named but styxx.evidence could "
            f"not be imported ({exc.__class__.__name__}: {exc}). The gate does "
            "not take the agent's word for test results, and it could not read "
            "the evidence either, so it declines.")
    try:
        ev = load_evidence(paths)
        verdict, why = adjudicate_tests_pass(ev, commit)
    except Exception as exc:                      # pragma: no cover - defensive
        return "UNCHECKABLE", (
            f"styxx.evidence raised {exc.__class__.__name__}: {exc} while reading "
            f"{len(paths)} supplied file(s). A crash is not a verdict.")

    tag = (f"styxx.evidence ({SPEC}) read {len(paths)} supplied file(s)"
           + (f", required to assert commit {commit}" if commit else
              ", with no commit supplied, so nothing ties a report to this change"))
    # The clamp. styxx.evidence's VERDICTS tuple is two words and its
    # selfcheck re-derives that from source, but this line does not depend on
    # that promise holding: anything that is not the affirming word becomes
    # UNCHECKABLE here, on this side of the boundary.
    if verdict != "VERIFIED":
        return "UNCHECKABLE", f"{tag} — {why}"
    return "VERIFIED", f"{tag} — {why}"


def _run_leg(run: str, repo) -> tuple[str, str]:
    """Execute the operator-supplied command. VERIFIED on exit 0, else UNCHECKABLE.

    THE ACCUSING HALF OF THIS BRANCH IS DELETED. It used to read
    ``r.returncode != 0`` as the author having lied. The exit code is still
    recorded — in the `why`, where a reader can act on it — and it decides
    nothing.
    """
    if repo is None:
        # The quiet defect on the zero-receipt path: `gate_diff_text` takes
        # repo=None by default, and `subprocess.run(cwd=None)` executes in the
        # VERIFIER'S OWN working directory. The one entry point built for having
        # no checkout was the one where cwd silently became the operator's tree.
        # Refusing beats defaulting.
        return "UNCHECKABLE", (
            f"--run {run!r} was supplied with no repository to run it in. "
            "REFUSED rather than executed: with cwd unset the command would run "
            "in the verifier's own working directory, not in any tree under "
            "review. Pass a repo, or use --evidence, which executes nothing.")
    try:
        r = subprocess.run(run, shell=True, cwd=repo,
                           capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=1800)
    except subprocess.TimeoutExpired:
        # Previously this propagated out of _gate as a traceback. A crash is not
        # a verdict, and a gate that dies mid-claim has not measured anything.
        return "UNCHECKABLE", (
            f"--run {run!r} did not finish within 1800s and was killed. A timeout "
            "is absence of evidence about the claim, not evidence against it.")
    except OSError as exc:
        return "UNCHECKABLE", (
            f"--run {run!r} could not be started ({exc.__class__.__name__}: "
            f"{exc}). That is a fact about this machine, not about the author.")
    if r.returncode == 0:
        return "VERIFIED", (
            f"--run {run!r} exited 0. Read that as exactly what it says: the "
            "supplied command exited 0 — IT DOES NOT MEAN THE TESTS PASSED. It "
            "restates the author's own claim and checks NOTHING about the "
            "sentence that was extracted, which may have been an unchecked "
            "checkbox, a negation or a quotation. Nothing about the command's "
            "provenance was checked either: on an untrusted branch the author "
            "controls this exit code in both directions.")
    return "UNCHECKABLE", (
        f"--run {run!r} exited {r.returncode}. A nonzero exit is NOT evidence "
        "that the author lied — it is also what pytest rc=5 (no tests "
        "collected), a misspelled command, a missing dependency and a flaky test "
        "produce, and on an untrusted branch the author controls this number in "
        "both directions. The accusing verdict for this claim kind is deleted, "
        "not disabled: its precision has never been measured by a blind panel on "
        "either leg. The exit code is recorded here and gates nothing.")


def _tests_pass_verdict(*, evidence, commit: str | None,
                        run: str | None, repo) -> tuple[str, str]:
    """Resolve one `tests_pass` claim. Called ONCE PER GATE, never per match.

    Before this repair the command ran once per REGEX MATCH: a body carrying N
    lines that say "all tests pass" launched N subprocesses, each with its own
    1800-second budget — unbounded, PR-author-controlled runner-hour
    amplification, and N chances to hang. Every `tests_pass` match in one summary
    asks the same question about the same suite, so it gets one answer.

    The two channels are a DISJUNCTION, deliberately: either may affirm, neither
    may accuse, and `--run` is not even executed if `--evidence` already
    affirmed. Requiring both would let *adding* --evidence turn a --strict PASS
    into a --strict FAIL relative to supplying no evidence at all, and that
    baseline is the one this gate guarantees.

    STATE THE GUARANTEE PRECISELY, because two properties get conflated here:
    supplying evidence is monotone AGAINST THE EMPTY BASELINE (it never does
    worse than supplying none), and it is NOT monotone UNDER SET EXTENSION
    (E -> E union {x} can demote VERIFIED to UNCHECKABLE when x cannot be
    parsed). The second is deliberate and is `styxx.evidence`'s rule, not this
    function's: a partial read declines rather than affirms. See the module
    docstring and `_evidence_leg` for what an operator should do about it.
    """
    verdict = "UNCHECKABLE"
    whys: list[str] = []
    # The adjudicator is consulted even when NO paths were supplied. It is a pure
    # function of bytes and it has its own words for that case — "no evidence was
    # supplied. Absence of a report is not a failing report; an unattested commit
    # is unattested." Paraphrasing them here would put a second adjudicator in
    # the file, which is the drift this wiring exists to avoid.
    v, why = _evidence_leg(evidence, commit)
    whys.append(why)
    if v == "VERIFIED":
        verdict = "VERIFIED"
    if verdict != "VERIFIED" and run:
        v, why = _run_leg(run, repo)
        whys.append(why)
        if v == "VERIFIED":
            verdict = "VERIFIED"
    if verdict != "VERIFIED" and not evidence:
        whys.append(_NO_EVIDENCE_NOTE)
    if verdict not in _TESTS_PASS_VERDICTS:       # unreachable clamp, kept anyway
        verdict = "UNCHECKABLE"
    return verdict, "  ||  ".join(whys)


def selfcheck_tests_pass_never_accuses(source: str | None = None) -> dict:
    """Re-derive from this module's own source that the tests_pass leg cannot accuse.

    The caller-side counterpart of ``styxx.evidence.selfcheck_no_accusation``,
    which states its own boundary as: it "does not prove a caller has not
    invented an accusation of its own out of the report-only `observed` band."
    diffgate IS that caller, so this check belongs here — in the file where the
    branch actually lived.

    Three questions, answered with ``ast`` rather than a grep, because the word
    necessarily appears in the prose explaining its absence and in the path-claim
    branches, which are a different class and out of scope:

      * does the accusing string appear as a NON-DOCSTRING constant anywhere
        inside the functions that decide a ``tests_pass`` verdict,
      * does ``_TESTS_PASS_VERDICTS`` still hold exactly the two surviving words,
      * does this module touch styxx.evidence's report-only ``observed`` band
        anywhere at all — the band whose only possible misuse is being turned
        into a verdict.

    Boundary, stated the way that module states its own: this proves the string
    is not produced by these functions. It does not prove the extraction that
    reaches them is sound — that is Panel A, and it has never been run.
    """
    if source is None:
        try:
            source = Path(__file__).read_text(encoding="utf-8")
        except OSError as exc:                    # pragma: no cover
            return {"ok": None, "reason": f"could not read own source: {exc}"}
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:                    # pragma: no cover
        return {"ok": None, "reason": f"own source does not parse: {exc}"}

    word = "CONTRA" + "DICTED"     # split so this line is not itself an occurrence
    band = "obse" + "rved"         # ditto for the report-only band

    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            body = getattr(node, "body", None) or []
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                docstrings.add(id(body[0].value))

    funcs = {n.name: n for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
             and n.name in _TESTS_PASS_FUNCTIONS}
    missing = sorted(set(_TESTS_PASS_FUNCTIONS) - set(funcs))

    occurrences = []
    for name in sorted(funcs):
        for node in ast.walk(funcs[name]):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and word in node.value and id(node) not in docstrings):
                occurrences.append({"function": name, "lineno": node.lineno,
                                    "excerpt": node.value[:80]})

    band_reads = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and node.value == band and id(node) not in docstrings):
            band_reads.append({"lineno": node.lineno, "form": "string key"})
        elif isinstance(node, ast.Attribute) and node.attr == band:
            band_reads.append({"lineno": node.lineno, "form": "attribute"})

    vocab_ok = tuple(_TESTS_PASS_VERDICTS) == ("VERIFIED", "UNCHECKABLE")
    ok = not occurrences and not band_reads and not missing and vocab_ok
    return {
        "ok": ok,
        "reason": ("the accusing verdict is absent from the tests_pass leg: no "
                   "code path in these functions produces it, the vocabulary "
                   "holds only VERIFIED and UNCHECKABLE, and this module never "
                   "reads styxx.evidence's report-only band"
                   if ok else "see code_occurrences / band_reads / missing"),
        "verdicts": list(_TESTS_PASS_VERDICTS),
        "functions_checked": sorted(funcs),
        "missing": missing,
        "code_occurrences": occurrences,
        "band_reads": band_reads,
        "boundary": ("proves the accusing string is not produced by these "
                     "functions and that the report-only band is never read. It "
                     "does NOT prove the extraction feeding them is sound: the "
                     "template fires on unchecked checkboxes, negations, "
                     "conditionals and quoted prose, and its precision has never "
                     "been measured by a blind panel."),
    }


from .declare import declaration_pass as _declaration_pass


@dataclass
class DiffClaim:
    kind: str
    text: str
    detail: dict
    verdict: str = "UNCHECKABLE"       # VERIFIED | CONTRADICTED | UNCHECKABLE
    why: str = ""


@dataclass
class DiffGate:
    verdict: str                        # PASS | FAIL
    base: str
    head: str
    claims: list = field(default_factory=list)
    uncovered_sentences: int = 0
    # The never-read band, first-class. The agent-branch attestation
    # (RESULT_agent_gate_boundary_2026_08_30) measured this gate reading 6
    # sentences of an agent's own 2,738 — a coverage boundary that a bare count
    # buried in a dict could not confess. So the gate now carries the
    # denominator and the sentences themselves: what was never read is
    # auditable, not just countable. Same promotion epistemics_summary gave
    # OATH's abstained band; observation only, verdict logic untouched.
    sentences_total: int = 0
    uncovered_texts: list = field(default_factory=list)
    # Of the never-read band, which sentences a STRUCTURAL reader thinks are claims the
    # templates simply failed to parse. This is the boundary's boundary: not "prose we did
    # not judge" but "claims we should have judged and could not". OBSERVATION ONLY —
    # STRUCT-1 never touches a verdict, exactly as the epistemics annotation never touched
    # OATH's ladder. See PREREG_claim_detector_2026_08_30.md.
    unparsed_claims: list = field(default_factory=list)
    # A gate that had NO EVIDENCE still has to answer PASS or FAIL, and PASS is
    # the flattering half. `measured` is the third answer the two-valued verdict
    # cannot carry: this gate did not run. A leg that cannot fail must not gate.
    measured: bool = True
    why_unmeasured: str = ""

    def to_dict(self):
        return {"diffgate": "v0", "verdict": self.verdict, "base": self.base,
                "head": self.head,
                "claims": [c.__dict__ for c in self.claims],
                "uncovered_sentences": self.uncovered_sentences,
                "sentences_total": self.sentences_total,
                "uncovered_texts": self.uncovered_texts,
                "unparsed_claims": self.unparsed_claims,
                "measured": self.measured,
                "why_unmeasured": self.why_unmeasured}


def _git(repo, *args) -> str:
    r = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()[:200]}")
    return r.stdout


# PATH-2 (PREREG_path2_resolution_2026_09_17, issue #121). The key used to be
# `p.replace("\\", "/").lstrip("./").lower()`, and `lstrip` removes ANY run of dots and slashes:
# `.pr_agent.toml` and `pr_agent.toml` shared one key, `.github/x` printed as `github/x`, and a
# status map keyed by path could hold only one of a dotfile and its undotted twin. Only a leading
# run of `/` and `./` segments is removed now; a dotfile, `..env` and `../x` keep their dots.
_LEADING_SLASH_SEGMENTS = re.compile(r"^(?:\.?/)+")


def _norm(p: str) -> str:
    return _LEADING_SLASH_SEGMENTS.sub("", p.replace("\\", "/")).lower()


def _undotted(key: str) -> str:
    """A key with its leading dots and slashes dropped: exactly the key `_norm` made before #121.

    Read where a reading of a path's shape must not move with the dot: BC-2's path-shape test for
    a bare prefix, BC-1's "no Python file" test, and COMPAT's language suffix and scaffold tests
    (AMENDMENT_path2_resolution_2026_09_17, C-2). Keys, matches and printed paths keep the dots.
    """
    return key.lstrip("./")


# NOTE_path2_third_pass_2026_09_25 (R-3). C-3's new accusation class is a prefix WRITTEN with a
# leading dot over a path without one ("Only touches .env." over `env`), and the key now keeps that
# dot. A key opening with two dots is a different thing: `../docs`, `./../docs` and `.../src/x.py`
# are relative or elided notations, and git never emits a changed path that starts with `../`, so
# EVERY changed path is outside such a prefix and the gate accused whatever the PR did. There is no
# repo path to check the claim against, so it abstains -- the lab's habit where the claim is not a
# repo path -- rather than accusing or verifying.
_DOTFILE_PREFIX = re.compile(r"^\.[^./\\]")


def _prefix_off_tree(pref: str) -> bool:
    """A prefix KEY that opens with a dot and is not a dotfile name.

    What reaches this is a key, `_norm(prefix).rstrip("/.")`: `../docs`, `./../docs` (normalised to
    `../docs`), `.../src/x.py`, and a literal name opening with two dots such as `..docs`, which the
    reason string then calls relative although it may be a name (NOTE_path2_fourth_pass, section E).
    A bare `..` or `...` never gets here: its key is empty, and BC-1 answers "is not a path" before it.
    """
    return pref.startswith(".") and not _DOTFILE_PREFIX.match(pref)


def _parent_prefix(raw: str) -> str:
    """NOTE_path2_fifth_pass_2026_09_25 (V-4): the prefix as written, BEFORE `rstrip("/.")`, when it
    ends in a `..` segment -- trailing `/` and `.` segments dropped, so `src/..`, `docs/../` and
    `.github/../.` (a sentence period after `../`) all do -- else "". `rstrip("/.")` deleted that
    segment, so `../docs/..` was read as `../docs` and `src/..` as `src`; the parent of a directory
    can hold any path, so such a prefix is off-tree and could hold anything."""
    segs = raw.split("/")
    while segs and segs[-1] in ("", "."):
        segs.pop()
    return "/".join(segs) if segs and segs[-1] == ".." else ""


def _could_lie_under(path: str, pref: str, raw: str = "") -> bool:
    """NOTE_path2_fourth_pass F-4: whether SOME reading of an off-tree prefix could hold `path`.

    `../docs` from an unknown directory X is `X/docs`, so a changed path lies under it on some
    reading exactly when the prefix's named segments occur, in order and contiguously, among the
    path's segments. Both sides are compared undotted and the dots-only segments of the prefix
    (`..`, `.`, `...`) are dropped, so the test errs towards "could": `../.github` could hold
    `.github/x.yml`, and a bare `..` could hold anything.

    NOTE_path2_fifth_pass V-4: a prefix whose written form ends in `..` (`raw`, before
    `rstrip("/.")`), and a `..` or `...` segment AFTER a named one (`../src/../docs` is `X/docs`,
    not `X/src/docs`), could hold anything; dropping those segments had turned "could" into "no"."""
    if _parent_prefix(raw):
        return True
    named = False
    for seg in pref.split("/"):
        if seg.strip("."):
            named = True
        elif named and seg not in ("", "."):
            return True
    want = [seg.lstrip(".") for seg in pref.split("/") if seg.strip(".")]
    have = [seg.lstrip(".") for seg in path.split("/")]
    if not want:
        return True
    return any(have[i:i + len(want)] == want for i in range(len(have) - len(want) + 1))


def _dot_miss(path: str, prefs: list) -> bool:
    """AMENDMENT_path2 C-3: `path` lies outside every prefix only by a dot the prose left off --
    some prefix key has no leading dot, the path's leading segment starts with exactly one dot
    (not `..`), and the path without that dot lies INSIDE the prefix by PATH-1's `_path_inside`
    (PREREG_path1_only_touches_repair_2026_09_17), the same containment test that decided the path
    was outside to begin with. Using anything weaker here would make a bare-filename prefix
    mean one thing for `outside` and another for the dot reading."""
    if not path.startswith(".") or path.startswith(".."):
        return False
    rest = path[1:]
    return any(not x.startswith(".") and _path_inside(rest, x) for x in prefs)


def parse_unified_diff(diff_text: str) -> tuple[dict[str, str], str]:
    """Unified diff text -> ({normalized_path: A|M|D}, added-lines blob).

    Lets the gate run on a raw ``.diff`` (webhook payloads, GitHub's ``.diff`` URL) with
    no checkout at all — the zero-receipt promise taken literally. Read by `_read_diff`, the one
    hunk-aware reading the per-file sides also come from (NOTE_path2_sixth_pass, W-1).
    """
    status, added, _sides = _read_diff(diff_text)
    return status, "\n".join(added)


def gate_diff_text(summary_text: str, diff_text: str,
                   run: str | None = None, strict: bool = False,
                   repo: str | Path | None = None,
                   evidence=None, commit: str | None = None) -> DiffGate:
    """Gate a summary against a RAW unified diff — no git checkout required.

    `evidence` is a sequence of test-report paths handed to `styxx.evidence`.
    `commit` is the revision the evidence must assert; with none, styxx.evidence
    says in its own `why` that the report is not tied to any particular change.
    Neither can produce an accusation — see `_tests_pass_verdict`.
    """
    status, added_blob = parse_unified_diff(diff_text)
    return _gate(summary_text, status, added_blob, run=run, strict=strict,
                 repo=repo, base="(diff-text)", head="(diff-text)",
                 evidence=evidence, commit=commit,
                 raw_input_len=len(diff_text or ""), main=_main_reading(diff_text or ""),
                 sides=parse_unified_diff_sides(diff_text or ""),
                 notes=_diff_notes(diff_text or ""))


def gate_diff(summary_text: str, repo: str | Path, base: str, head: str,
              run: str | None = None, strict: bool = False,
              evidence=None, commit: str | None = None) -> DiffGate:
    """Extract diff-shaped claims from *summary_text* and verify against base..head.

    `evidence` is a sequence of test-report paths (a JUnit XML, a test-result
    attestation) routed through `styxx.evidence`, whose vocabulary is VERIFIED
    and UNCHECKABLE. MEASURED AGAINST SUPPLYING NOTHING, it can only turn an
    UNCHECKABLE `tests_pass` claim into a VERIFIED one: there is no route by
    which passing a report fails a build that would have passed with no report.
    That comparison is the guarantee, and it is the whole of it — this is NOT
    monotone under set extension, and adding an unreadable path to a set that
    was already affirming demotes it to UNCHECKABLE by design. See the module
    docstring, "MONOTONICITY: TWO PROPERTIES".

    `commit` is NOT defaulted to the resolved `head`. That was tried and backed
    out: it made this function silently stricter than `gate_diff_text` over the
    same evidence bytes, and an adjudication that changes with which entry point
    called it is not a pure function of the bytes. The caller says which revision
    the report has to name. Supplying one can only WITHHOLD a VERIFIED.
    """
    repo = Path(repo)
    name_status = _git(repo, "diff", "--name-status", f"{base}..{head}")
    status: dict[str, str] = {}
    paths: list = []
    for line in _diff_lines(name_status):                  # NOTE_path2_fourth_pass F-2
        parts = line.split("\t")
        if len(parts) >= 2:
            st, path = parts[0][:1], parts[-1]
            status[_norm(path)] = st            # A / M / D / R
            paths.append(path)
    diff_text = _git(repo, "diff", f"{base}..{head}")
    # NOTE_path2_sixth_pass W-1: the added blob and the sides are the one hunk-aware reading of git's
    # bytes. The status stays git's own `--name-status`, which is not a reading of the diff text.
    added_blob = parse_unified_diff(diff_text)[1]
    # NOTE_path2_eighth_pass: the file list here is git's own, so the only thing it can be unsure of is two paths
    # the key reads as one (they differ only in case, Y-1); a U+FEFF dropped from a test at line 1 is the parse's (Y-2).
    parsed, listed = _diff_notes(diff_text), _status_notes(paths)
    # NOTE_path2_ninth_pass (Z-3): main keyed the same `--name-status` lines with str.splitlines() and its `_norm`.
    main_map = _main_name_status(name_status)
    notes = {k: v for k, v in (("files", listed.get("files")), ("bom", parsed.get("bom")),
                               ("differs", _status_differs(main_map, status))) if v}
    return _gate(summary_text, status, added_blob, run=run, strict=strict,
                 repo=repo, base=base, head=head, main=_MainReading(diff_text, (main_map,)),
                 evidence=evidence, commit=commit,
                 sides=parse_unified_diff_sides(diff_text), notes=notes)


def _find_path(status: dict, claimed: str):
    """The status entry a path claim names: (path, status), or (None, None).

    PATH-2 (PREREG_path2_resolution_2026_09_17, issue #97): resolved in TIERS over every entry --
    an entry equal to the claim, else one ending in "/" + claim, else one with the claim's
    basename; diff order decides only within a tier. The loop this replaces returned the entry
    that cleared ANY of the three in diff order, so a diff that modified README.md and then
    created integrations/git/README.md resolved "Created integrations/git/README.md" to the root
    README by basename. A claim no tier matches is (None, None) exactly when it was before.
    """
    c = _norm(claimed)
    for tier in (lambda p: p == c,
                 lambda p: p.endswith("/" + c),
                 lambda p: Path(p).name == Path(c).name):
        for p, st in status.items():
            if tier(p):
                return p, st
    return None, None


def _path_claim_verdict(kind: str, claimed: str, find_path) -> tuple[str, str]:
    """Resolve a file_created / file_deleted / file_touched claim.

    Lifted out of `_gate` UNCHANGED — same branches, same order, same strings.
    It lives in its own function so that the path-claim accusation, which is a
    DIFFERENT class with its own published history and its own withholding flag,
    is not lexically nested inside the loop that also handles `tests_pass`. A
    structural check asking "is an accusation produced under a `tests_pass`
    guard" cannot answer that about a chain of elifs sharing one `for`
    statement. The honest fix is to separate the code, not to word the check
    more loosely.
    """
    p, st = find_path(claimed)
    want = {"file_created": "A", "file_deleted": "D"}.get(kind)
    accuse = not WITHHOLD_PATH_ACCUSATION
    bare = (V14_BARE_NAME_ABSTAIN and "/" not in claimed and "\\" not in claimed)
    if p is None and bare:
        # V14 repair 2 (PREREG_v14_repair_2026_08_31): a bare name ending in a
        # code-like extension is not reliably a file. The corpus accuses
        # `asmcrypto.js` and `ethers.js` — npm packages named in prose — and no
        # frozen list can close that set, because library names are open and a
        # list is always one package behind. A claimed path with no directory
        # component, absent from the diff entirely, is ambiguous between a file
        # and a library and the instrument cannot tell which. It says so.
        # DELIBERATE RECALL SACRIFICE, preregistered as one: this stops the gate
        # catching some genuine lies about bare-named files. Made knowingly —
        # measured at 0.23 precision, a false accusation costs more than a missed
        # catch. Paths carrying a directory component are unaffected and still
        # accuse.
        return "UNCHECKABLE", (
            f"{claimed!r} is a bare name absent from the diff — ambiguous "
            "between a file and a library, so no accusation is made (V14 "
            "repair 2, a deliberate recall sacrifice)")
    if p is None:
        # EXTERNAL-1 (RESULT_external1_the_gate_fails_in_the_wild_2026_08_31):
        # over 100 blind-adjudicated accusations on an external corpus of
        # agent-authored PRs this branch reached precision 0.23 against a
        # preregistered floor of 0.95. The preregistration's committed
        # consequence was to disable the accusing verdict for this class until
        # repaired, and this is that consequence being paid. Four mechanical
        # defects account for it: bare basenames never met full diff paths
        # ('glob.ts' vs 'src/node/glob.ts'); "removed X from FILE" bound the verb
        # to the file instead of X; prose nouns passed the extension whitelist
        # ('Node.js', 'Express.js'); and negation ("avoids modifying
        # tsconfig.json") was read as assertion. An instrument that cannot accuse
        # precisely must abstain. Repair is preregistered separately and must
        # clear its gate on the HELD-OUT split before this line accuses again.
        return (("CONTRADICTED",
                 f"{claimed!r} does not appear in the diff at all")
                if accuse else
                ("UNCHECKABLE",
                 f"{claimed!r} does not appear in the diff — accusation "
                 "WITHHELD: this class failed EXTERNAL-1 precision "
                 "(0.23 vs 0.95 floor), disabled pending repair"))
    if want and st != want:
        return (("CONTRADICTED",
                 f"{claimed!r} is status {st!r} in the diff, "
                 f"claim wants {want!r}")
                if accuse else
                ("UNCHECKABLE",
                 f"{claimed!r} is status {st!r}, claim wants {want!r} — "
                 "accusation WITHHELD pending the EXTERNAL-1 repair"))
    return "VERIFIED", f"diff status {st!r} for {p!r}"


def _gate(summary_text: str, status: dict[str, str], added_blob: str, *,
          run: str | None, strict: bool, repo, base: str, head: str,
          evidence=None, commit: str | None = None,
          raw_input_len: int | None = None, sides: dict | None = None,
          notes: dict | None = None, main: "_MainReading | None" = None,
          _declared: bool = False) -> DiffGate:

    # Some claim kinds are VACUOUSLY TRUE against an empty diff. `only_touches`
    # asks "is anything outside the prefix?" and an empty status answers "no" —
    # so until 2026-08-21 this gate returned VERIFIED for the input
    # "Sorry, I could not produce a diff."  The module whose entire purpose is
    # refusing to take the agent's word took the agent's word.
    #
    # An empty status is not agreement. It is the absence of evidence, and this
    # file already has the right word for that: UNCHECKABLE.
    no_evidence: str | None = None
    if not status and not added_blob:
        no_evidence = "the diff carries no file statuses and no added lines"
        if raw_input_len:
            no_evidence += (f"; {raw_input_len} characters of input parsed to "
                            f"nothing, which is a parse failure, not an empty change")
    no_paths = "the diff carries no file paths, so scope cannot be checked" \
        if not status else None
    # NOTE_path2_eighth_pass (Y-1): what the one reading of the diff is not sure of; NOTE_path2_ninth_pass (Z-3):
    # where its file list is not main's and no repair licenses the difference.
    unsure_files = _files_unsure(notes) or _files_differ(notes)
    not_sure = f"the diff's file list is not certain: {unsure_files}" if unsure_files else None

    def find_path(claimed: str):
        return _find_path(status, claimed)

    # ONE resolution of the tests_pass question per gate invocation, memoised
    # here and shared by every match. See `_tests_pass_verdict` for what this
    # repairs: the command used to run once per REGEX MATCH.
    _tp: list[tuple[str, str]] = []

    def tests_pass_leg() -> tuple[str, str]:
        if not _tp:
            _tp.append(_tests_pass_verdict(
                evidence=evidence, commit=commit, run=run, repo=repo))
        return _tp[0]

    claims: list[DiffClaim] = []
    sentences = re.split(r"(?<=[.!?])\s+|\n+", summary_text)
    covered = set()
    for si, sent in enumerate(sentences):
        for kind, rx in _TEMPLATES:
            for m in rx.finditer(sent):
                # A path can be NAMED without being CLAIMED. Two forms, both found
                # by re-sweeping 150 real commits on 7.44.1 and both false
                # accusations:
                #   "Fixed the same way sla.py was"       -- comparative reference
                #   "(fetch-depth: 0 in test.yml) is staged"  -- explicitly NOT here
                # The second one says in words that the file is not in this diff,
                # and the gate accused it anyway. A gate that cannot read "staged"
                # does not get to call a summary a liar.
                if kind in _PATH_KINDS and _names_without_claiming(sent, m):
                    continue
                if kind in _PATH_KINDS and _is_non_file_noun(m.group("path")):
                    continue                        # V13 repair 2
                if (kind in ("file_created", "file_deleted")
                        and _demoted_by_containment(sent, m)):
                    kind = "file_touched"           # V13 repair 1
                # V14 repair 1 (PREREG_v14_repair_2026_08_31): containment was
                # repaired for the wrong verbs. "added tests for the hash
                # functions IN file" is the same shape as "removed the helper
                # FROM file" — a claim about content within, not about the file
                # changing — and V13 left the touch form accusing. Same closed
                # preposition set; the path is named, not claimed.
                if (V14_CONTAINMENT_TOUCH and kind == "file_touched"
                        and _demoted_by_containment(sent, m)):
                    continue
                if (BC1_BY_CONSTRUCTION and kind == "symbol_added"
                        and m.group("name").lower() in _SYMBOL_WORDS):
                    continue                        # BC-1 repair 3: a word, not a symbol
                covered.add(si)
                d = {k: v for k, v in m.groupdict().items() if v is not None}
                c = DiffClaim(kind=kind, text=sent.strip()[:160], detail=d)
                # The `and kind != "tests_pass"` exemption that used to live on
                # this line is REMOVED. With input that parsed to nothing the
                # gate returns measured=False and the CLI prints "UNMEASURED
                # this gate did not run" — and the exempted branch went on to
                # execute a shell command and pronounce on the claim anyway. A
                # gate that says it did not run must not reach a verdict, and it
                # must not launch a subprocess to get there.
                if no_evidence:
                    c.verdict, c.why = "UNCHECKABLE", no_evidence
                    claims.append(c)
                    continue
                if kind in _PATH_KINDS:
                    at = m.start("path")
                    apart = main is not None and _read_apart(d["path"], sent[at - 1:at])
                    only_name = None if (not_sure or apart) else _basename_only(status, d["path"])
                    if not_sure:                        # NOTE_path2_eighth_pass (Y-1)
                        c.verdict, c.why = "UNCHECKABLE", not_sure
                    elif apart:                         # NOTE_path2_tenth_pass (K-5): as main read it
                        c.verdict, c.why = _path_claim_verdict(kind, d["path"],
                                                               lambda claimed: _main_find(main.maps[0], claimed))
                    elif only_name:                     # NOTE_path2_ninth_pass (Z-4)
                        c.verdict, c.why = "UNCHECKABLE", only_name
                    else:
                        c.verdict, c.why = _path_claim_verdict(kind, d["path"],
                                                               find_path)
                elif kind == "files_changed_count":
                    n = int(d["n"])
                    if no_paths:
                        c.verdict, c.why = "UNCHECKABLE", no_paths
                    elif not_sure:                      # NOTE_path2_eighth_pass (Y-1)
                        c.verdict, c.why = "UNCHECKABLE", f"{not_sure}; claim says {n}"
                    else:
                        c.verdict = "VERIFIED" if n == len(status) else "CONTRADICTED"
                        c.why = f"diff changes {len(status)} files, claim says {n}"
                elif kind == "tests_added":
                    n = int(d["n"])
                    noun = d.get("noun", "").lower()
                    if BC1_BY_CONSTRUCTION and not _diff_touches_python(status):
                        c.verdict = "UNCHECKABLE"           # BC-1 repair 1
                        c.why = ("no Python file in the diff; this template counts "
                                 "`def` lines (#110)")
                    else:
                        # NOTE_path2_third_pass R-1: one optional leading U+FEFF;
                        # NOTE_path2_fourth_pass F-3 and NOTE_path2_fifth_pass V-1: the indent is
                        # `_DEF_INDENT`, the pairing's; NOTE_path2_sixth_pass W-2: the count IS the
                        # pairing's reading (`_test_name`), and W-1: the blob is the sides' own added
                        # lines. So `chg <= got` holds line by line.
                        got = _added_tests(added_blob)
                        # PATH-2 (#101): `chg` added `def test_` lines re-define a test the same
                        # file's removed lines define, paired one to one (AMENDMENT C-1). The true
                        # number added lies in [net, got]: verify `net`, abstain inside the
                        # interval, accuse only outside it.
                        chg = min(_changed_test_defs(sides, status), got)
                        net = got - chg
                        note = f" ({chg} changed, not added: #101)" if chg else ""
                        # NOTE_path2_seventh_pass (A-1): an added `async def test_` is a test this
                        # count does not read; beside one, the count is not the diff's.
                        unread = _async_tests_added(sides, status)
                        # NOTE_path2_eighth_pass: a test definition some reading that matters cannot read
                        # alike (Y-2, Y-3).
                        doubt = _test_doubt(added_blob, sides, notes)
                        # NOTE_path2_ninth_pass: `got`, or BC-1's answer, not main's (Z-1); a definition line the
                        # whole file may not survive (Z-5).
                        unlicensed = _tests_differ(got, status, main)
                        whole = _whole_file_tests(added_blob, sides)
                        if unread:
                            c.verdict = "UNCHECKABLE"
                            c.why = (f"diff adds {unread} async test functions, which this template does not "
                                     f"count; claim says {n}")
                        elif doubt:
                            c.verdict, c.why = "UNCHECKABLE", f"{doubt}; claim says {n}"
                        elif unlicensed:
                            c.verdict, c.why = "UNCHECKABLE", f"{unlicensed}; claim says {n}"
                        elif whole:
                            c.verdict, c.why = "UNCHECKABLE", f"{whole}; claim says {n}"
                        elif net == n and _pairing_withdraws(chg):
                            # NOTE_path2_eighth_pass (Y-5): the pairing withdraws, it does not verify
                            c.verdict = "UNCHECKABLE"
                            c.why = (f"diff adds {net} test functions and changes {chg}, claim says {n}; a count "
                                     "left after pairing changed tests away is not verified, since a line this "
                                     "template reads may be one Python does not define (#101)")
                        elif net == n:
                            c.verdict, c.why = "VERIFIED", f"diff adds {net} test functions, claim says {n}{note}"
                        elif BC1_BY_CONSTRUCTION and noun in _TEST_NOUNS_NOT_FUNCTIONS:
                            # BC-2 repair 2: a case, file, scenario, suite or class is not a
                            # function; a matching count verifies, a differing one abstains.
                            c.verdict = "UNCHECKABLE"
                            one = {"classes": "class", "cases": "case", "files": "file",
                                   "scenarios": "scenario", "suites": "suite"}.get(noun, noun)
                            c.why = (f"counts test {noun}, diff adds {net} test functions; "
                                     f"a {one} is not a function (#110){note}")
                        elif chg and net < n <= got:
                            c.verdict = "UNCHECKABLE"
                            c.why = (f"diff adds {net} test functions and changes {chg}, claim says {n}; "
                                     "a changed test is not an added one (#101)")
                        else:
                            c.verdict = "CONTRADICTED"
                            c.why = f"diff adds {net} test functions, claim says {n}{note}"
                elif kind == "symbol_added":
                    if BC1_BY_CONSTRUCTION and not _diff_touches_python(status):
                        c.verdict = "UNCHECKABLE"           # BC-1 repair 1
                        c.why = ("no Python file in the diff; this template counts "
                                 "`def` lines (#110)")
                    else:
                        # NOTE_path2_fifth_pass V-1: the added-side pairing pattern, one line at a
                        # time. `^\s*(?:def|class)\s+NAME\b` read lines the pairing did not (a form
                        # feed re-indent, a changed generic), and each was a false VERIFIED.
                        # NOTE_path2_sixth_pass W-2: the claimed name is the identifier the summary
                        # writes, read by the rule a definition's name is read by, so the two end in
                        # the same place; the template's `name` group only says where it starts.
                        # NOTE_path2_seventh_pass: a claimed name that runs on past the identifier
                        # (or ends in a middle dot) names no identifier, and is not truncated to one.
                        name, why_name = _claimed_name(sent, m)
                        # NOTE_path2_eighth_pass (Y-2): a line defining the name behind a U+FEFF.
                        doubt = _symbol_doubt(name, added_blob, sides) if why_name is None else None
                        hit = why_name is None and _symbol_hit(name, added_blob)
                        # NOTE_path2_ninth_pass: `hit`, or BC-1's answer, not main's (Z-2); a file holding the
                        # definition that the whole-file reading may refuse (Z-5).
                        unlicensed = whole = None
                        if why_name is None:
                            unlicensed = _symbol_differs(hit, name, *_main_names(sent, m.start("name")), status, main)
                            whole = _whole_file_symbol(name, added_blob, sides) if hit else None
                        if why_name is not None:
                            c.verdict, c.why = "UNCHECKABLE", why_name
                        elif doubt:
                            c.verdict, c.why = "UNCHECKABLE", doubt
                        elif unlicensed:
                            c.verdict, c.why = "UNCHECKABLE", unlicensed
                        elif whole:
                            c.verdict, c.why = "UNCHECKABLE", whole
                        elif hit and _definition_only_changed(name, sides, status):
                            c.verdict = "UNCHECKABLE"               # PATH-2 (#101)
                            c.why = (f"added lines define {d['kind']} {_qname(name)} only where the "
                                     "removed lines of the same file define it too; a changed "
                                     "definition is not an added one (#101)")
                        else:
                            c.verdict = "VERIFIED" if hit else "CONTRADICTED"
                            c.why = (f"added lines {'do' if hit else 'do NOT'} define "
                                     f"{d['kind']} {_qname(name)}")
                elif kind == "only_touches":
                    prefs = [_norm(d["prefix"]).rstrip("/.")]   # sentence-final periods are not path
                    if d.get("prefix2"):
                        prefs.append(_norm(d["prefix2"]).rstrip("/."))
                    # BC-2 repair 4: a second prefix is read only after "and" and only when it
                    # is path-shaped by the same test; otherwise the first prefix decides alone.
                    # NOTE_path2_sixth_pass (V-4, completed): a second prefix WRITTEN as a parent (a
                    # bare `..`) is read before that test drops it -- it is off-tree, as `../` is --
                    # instead of leaving the leading prefix to accuse alone.
                    parent2 = bool(d.get("prefix2")) and bool(_parent_prefix(d["prefix2"].replace("\\", "/")))
                    if d.get("prefix2") and not parent2 and not _prefix_is_path_shaped(d["prefix2"], status):
                        prefs = prefs[:1]
                    raw_prefs = [d["prefix"]] + ([d["prefix2"]] if len(prefs) == 2 else [])
                    not_paths = [_norm(x).rstrip("/.") for i, x in enumerate(raw_prefs)
                                 if not (i == 1 and parent2)
                                 and not _prefix_is_path_shaped(x, status)] if BC1_BY_CONSTRUCTION else []
                    # PATH-1 mode 1: _path_inside matches a bare filename on its basename.
                    outside = [p for p in status
                               if not any(_path_inside(p, x) for x in prefs)]
                    # PATH-2 (#121), AMENDMENT C-3: an outside path is a DOT MISS when a prefix
                    # written without a leading dot holds it once its own single leading dot is
                    # dropped ("Only touches github/" over `.github/...`); every other outside path
                    # is REAL. Dot misses alone abstain; any real path accuses, and only real paths
                    # are listed. A dotted prefix over an undotted path, and a `..` path, are real.
                    # Containment inside `_dot_miss` is PATH-1's `_path_inside`, the same test that
                    # decided `outside` above, so a bare filename keeps its basename reading.
                    dot_miss = [p for p in outside if _dot_miss(p, prefs)]
                    real = [p for p in outside if p not in dot_miss]
                    # NOTE_path2_fifth_pass V-4: off-tree-ness is also read on the prefix as written,
                    # before rstrip("/."), so a prefix ending in `..` is off-tree (`_parent_prefix`).
                    written = [_norm(x) for x in raw_prefs]
                    off_pairs = [(x, r) for x, r in zip(prefs, written)
                                 if _prefix_off_tree(x) or _parent_prefix(r)]
                    off_tree = [_parent_prefix(r) or x for x, r in off_pairs]
                    # NOTE_path2_fourth_pass F-4: beside an ON-tree prefix, an off-tree one no longer
                    # withdraws a correct accusation. A real path outside every on-tree prefix that
                    # no reading of the off-tree prefix could hold is outside the claim on every
                    # reading, so it still accuses, and only such paths are listed. Anything less
                    # certain -- and an off-tree prefix standing alone -- abstains as R-3 did.
                    beside_on_tree = bool(off_tree) and len(off_tree) < len(prefs)
                    if beside_on_tree:
                        real = [p for p in real
                                if not any(_could_lie_under(p, x, r) for x, r in off_pairs)]
                    if no_paths:
                        c.verdict, c.why = "UNCHECKABLE", no_paths
                    elif not_sure:                              # NOTE_path2_eighth_pass (Y-1)
                        c.verdict, c.why = "UNCHECKABLE", not_sure
                    elif not_paths:                             # BC-1 repair 4
                        c.verdict = "UNCHECKABLE"
                        c.why = f"prefix {not_paths[0]!r} is not a path (#110)"
                    elif off_tree and not (beside_on_tree and real):   # R-3, narrowed by F-4
                        c.verdict = "UNCHECKABLE"
                        c.why = (f"prefix {off_tree[0]!r} is relative to a directory the diff does "
                                 "not name (#121)")
                    elif dot_miss and not real:
                        c.verdict = "UNCHECKABLE"
                        c.why = ((f"paths outside {prefs[0]!r} differ from it only by a leading dot: "
                                  f"{dot_miss[:3]} (#121)") if len(prefs) == 1 else
                                 (f"paths outside {' and '.join(repr(x) for x in prefs)} differ from them "
                                  f"only by a leading dot: {dot_miss[:3]} (#121)"))
                    else:
                        c.verdict = "VERIFIED" if not real else "CONTRADICTED"
                        shown = prefs[0] if len(prefs) == 1 else " and ".join(repr(x) for x in prefs)
                        c.why = ("all changed paths under prefix" if not real else
                                 (f"paths outside {shown!r}: {real[:3]}" if len(prefs) == 1
                                  else f"paths outside {shown}: {real[:3]}"))
                elif kind == "compat_claim":
                    # COMPAT-1: one verdict, the evidence in the reason. See _compat_reading.
                    c.verdict, c.why, extra = _compat_reading(sides)
                    c.detail.update(extra)
                    if c.verdict not in _COMPAT_VERDICTS:      # unreachable clamp, kept anyway
                        c.verdict = "UNCHECKABLE"
                elif kind == "tests_pass":
                    # VERIFIED or UNCHECKABLE. There is no third answer here and
                    # no flag that adds one — see the module docstring and
                    # selfcheck_tests_pass_never_accuses(). The extraction that
                    # reached this line is UNMEASURED: this template fires on
                    # unchecked PR-template checkboxes, on negations, on
                    # conditionals and inside code fences, so a verdict of either
                    # word may be attached to a sentence that asserts nothing.
                    # That is disclosed rather than patched.
                    c.verdict, c.why = tests_pass_leg()
                claims.append(c)

    # DECLARE-1 (PREREG_declare1_the_toll_2026_09_18, sha256 7ffd0ba1...). The prose pass above
    # is finished and is not changed by any of this. A body may ALSO declare its claims in one
    # fenced `styxx` block; each declaration is normalised into the canonical sentence this same
    # reader already understands and read by this same function one level down. Nothing here
    # re-implements a verdict, so a declared claim and a prose claim of the same content cannot
    # drift apart, and the differential sees one reading rather than two.
    #
    # The recursion terminates in one step: synthesized text never contains a styxx fence.
    if not _declared:
        _dtext, _drep = _declaration_pass(summary_text)
        if _drep["declared"]:
            if _dtext:
                _sub = _gate(_dtext, status, added_blob, run=run, strict=strict, repo=repo,
                             base=base, head=head, evidence=evidence, commit=commit,
                             raw_input_len=raw_input_len, sides=sides, notes=notes, main=main,
                             _declared=True)
                for _c in _sub.claims:
                    _c.detail = dict(_c.detail or {})
                    _c.detail["declared"] = True
                    claims.append(_c)
            # Declared but deliberately unverifiable (`tests_pass`), and unreadable lines. Both
            # are reported as UNCHECKABLE: a declaration that cannot be read is not a lie, and a
            # declaration that tests passed is not evidence that they did.
            for _u in _drep["unverifiable"]:
                claims.append(DiffClaim(kind=_u["key"], text=f"{_u['key']}: {_u['value']}",
                                        detail={"declared": True},
                                        verdict="UNCHECKABLE", why=_u["why"]))
            for _p in _drep["problems"]:
                claims.append(DiffClaim(kind="declaration_problem", text=_p,
                                        detail={"declared": True},
                                        verdict="UNCHECKABLE", why=_p))

    contradicted = any(c.verdict == "CONTRADICTED" for c in claims)
    uncheckable = any(c.verdict == "UNCHECKABLE" for c in claims)
    # STRICT MODE AND THE EVIDENCE LEG. --strict turns any UNCHECKABLE into FAIL,
    # so the guarantee has to be stated against a named baseline. Say which:
    #
    # MONOTONE AGAINST THE EMPTY BASELINE — HOLDS. "Can supplying --evidence fail
    # a build that would have passed with NO --evidence?" It cannot. The evidence
    # leg is consulted only for `tests_pass`; that kind is UNCHECKABLE without it;
    # no other claim's verdict reads `evidence`; supplying it adds and removes no
    # claims. So against the empty baseline the verdict set is pointwise
    # at-least-as-good. There is also no route to CONTRADICTED: styxx.evidence's
    # vocabulary is two words, `_evidence_leg` clamps anything else to
    # UNCHECKABLE, and `selfcheck_tests_pass_never_accuses` re-derives that from
    # this file's own source.
    #
    # NOT MONOTONE UNDER SET EXTENSION — DELIBERATE. The set of UNCHECKABLE claims
    # does NOT simply shrink as evidence is added. Going from E to E union {x}
    # can put `tests_pass` back to UNCHECKABLE, because ANY unparsed source blocks
    # VERIFIED in styxx.evidence: a partial read may honestly decline but may not
    # honestly affirm. `--evidence green.xml` PASSes here where `--evidence
    # green.xml empty.xml` FAILs, and that is correct — nine readable shards out
    # of ten cannot certify "all tests pass". The earlier version of this comment
    # claimed the shrink-only property for both baselines at once; only the empty
    # one holds, and the difference is a contract, not a bug. Operators: supply
    # complete evidence or supply none.
    verdict = "FAIL" if (contradicted or (strict and uncheckable)) else "PASS"
    uncovered_texts = [s.strip() for i, s in enumerate(sentences)
                       if s.strip() and i not in covered]
    total = sum(1 for s in sentences if s.strip())
    # The never-read band, read structurally. Import is local and failure is silent: the
    # gate must run identically whether or not the observer is available, because a
    # verdict that depends on an observer is not an observation.
    unparsed = []
    try:
        from styxx.claimdetect import detect as _detect
        unparsed = [s for s in uncovered_texts if _detect(s).is_claim]
    except Exception:
        unparsed = []
    return DiffGate(verdict=verdict, base=base, head=head, claims=claims,
                    measured=not no_evidence, why_unmeasured=no_evidence or "",
                    uncovered_sentences=len(uncovered_texts),
                    sentences_total=total, uncovered_texts=uncovered_texts,
                    unparsed_claims=unparsed)


_DEMO_SUMMARY = ("Refactored src/retry.py for resilience. Adds function backoff with "
                 "jitter. Added 3 tests covering the retry path. Only touches files "
                 "under src/. All tests pass.")
_DEMO_DIFF = """\
--- a/src/retry.py
+++ b/src/retry.py
@@ -1,3 +1,6 @@
 def retry(n):
     return n
+
+def retry_once(n):
+    return retry(1)
--- a/config/settings.yml
+++ b/config/settings.yml
@@ -1,2 +1,2 @@
-timeout: 30
+timeout: 5
--- /dev/null
+++ b/tests/test_retry.py
@@ -0,0 +1,2 @@
+def test_retry_once():
+    assert True
"""

_WHAT_IT_CHECKS = """\
  the gate checks a CLOSED template set; prose outside it is never judged:
    "modified/created/deleted <path>"     vs the diff's file statuses
    "adds function/class <name>"          vs added definitions
    "added N tests"                       vs added test functions
    "N files changed"                     vs the diff
    "only touches <prefix>"               vs every changed path
    "tests pass"                          only with --evidence (a test report, read
                                          as bytes) or --run (which EXECUTES a shell
                                          command) — we don't take its word, and we
                                          never call it a lie: VERIFIED or UNCHECKABLE
  example of a checkable sentence:  'Modified src/app.py and added 2 tests.'
"""


_PR_URL = re.compile(
    r"^(?:https?://)?(?:www\.)?github\.com/(?P<owner>[\w.-]+)/(?P<repo>[\w.-]+)/pull/"
    r"(?P<number>\d+)(?:[/?#].*)?$")


def fetch_pr(url: str, token: str | None = None, timeout: int = 60, _open=None) -> dict:
    """The description and the diff of a public GitHub pull request, from the API.

    No checkout, no clone: the two things the gate needs are the body the agent wrote
    and the unified diff GitHub serves for the request, and `gate_diff_text` takes
    exactly those. A token (``GITHUB_TOKEN`` / ``GH_TOKEN``) is used only for the rate
    limit; without one GitHub allows 60 requests an hour per address. Nothing from the
    response is executed and nothing but these two documents is read. `_open` exists so
    tests can hand in a fake opener; it is `urllib.request.urlopen` otherwise.
    """
    m = _PR_URL.match(url.strip())
    if not m:
        raise ValueError(f"not a GitHub pull request URL: {url!r} "
                         "(expected github.com/OWNER/REPO/pull/N)")
    owner, repo, number = m.group("owner"), m.group("repo"), int(m.group("number"))
    api = f"https://api.github.com/repos/{owner}/{repo}/pulls/{number}"
    opener = _open or urllib.request.urlopen

    def get(accept: str) -> bytes:
        headers = {"Accept": accept, "User-Agent": "styxx-diffgate",
                   "X-GitHub-Api-Version": "2022-11-28"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            with opener(urllib.request.Request(api, headers=headers), timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise SystemExit(f"{owner}/{repo}#{number}: GitHub answers 404 — the pull "
                                 "request does not exist or is private. For a private one, "
                                 "check it out and use SUMMARY --repo --base --head.") from e
            if e.code in (403, 429):
                raise SystemExit(f"{owner}/{repo}#{number}: GitHub answers {e.code} — the "
                                 "unauthenticated API limit (60/hour per address) is used "
                                 "up, or the token is refused. Set GITHUB_TOKEN, or wait.") from e
            if e.code == 406:
                raise SystemExit(f"{owner}/{repo}#{number}: GitHub will not serve this diff "
                                 "over the API (too large). Check it out and use "
                                 "SUMMARY --repo --base --head.") from e
            raise SystemExit(f"{owner}/{repo}#{number}: GitHub answers HTTP {e.code}") from e

    meta = json.loads(get("application/vnd.github+json").decode("utf-8", errors="replace"))
    diff = get("application/vnd.github.diff").decode("utf-8", errors="replace")
    return {
        "repo": f"{owner}/{repo}", "number": number, "title": meta.get("title") or "",
        "base": ((meta.get("base") or {}).get("ref")) or "",
        "head": ((meta.get("head") or {}).get("sha")) or "",
        "html_url": meta.get("html_url") or f"https://github.com/{owner}/{repo}/pull/{number}",
        "body": meta.get("body") or "", "diff": diff,
    }


def _demo() -> int:
    print("styxx diffgate --demo : an agent PR summary vs the diff it shipped with\n")
    print("the summary the agent wrote:")
    print(f"  {_DEMO_SUMMARY}\n")
    print("what the diff actually shows: retry.py +retry_once, settings.yml timeout "
          "30->5, one new test\n")
    g = gate_diff_text(_DEMO_SUMMARY, _DEMO_DIFF)
    for c in g.claims:
        mark = {"VERIFIED": "ok ", "CONTRADICTED": "LIE", "UNCHECKABLE": " ? "}[c.verdict]
        print(f"  [{mark}] {c.kind:20s} {c.why}")
    print(f"\nverdict: {g.verdict} — this summary would fail your CI with each lie "
          "named.\n(demo always exits 0; point it at real work: "
          "python -m styxx.diffgate SUMMARY.md --repo . --base main)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="styxx.diffgate",
                                 description="An agent's summary cannot lie about its diff.")
    ap.add_argument("summary", nargs="?")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--base")
    ap.add_argument("--head", default="HEAD")
    ap.add_argument(
        "--evidence", nargs="+", action="extend", default=None, metavar="REPORT",
        help="one or more test reports (a JUnit XML, or a test-result "
             "attestation naming the commit). Read as BYTES by styxx.evidence, "
             "which EXECUTES NOTHING and whose whole vocabulary is VERIFIED and "
             "UNCHECKABLE. Repeatable, and accepts several paths at once. An "
             "absent, unreadable or red report is UNCHECKABLE, never an "
             "accusation.")
    ap.add_argument(
        "--commit", default=None, metavar="SHA",
        help="the commit the evidence must assert. Not defaulted to --head: the "
             "caller says which revision the report has to name, so the same "
             "bytes get the same answer from every entry point. Evidence that "
             "does not assert it withholds VERIFIED; it never accuses. No "
             "signature is checked anywhere in this path — a digest assertion "
             "is bytes the producer chose.")
    ap.add_argument(
        "--run", default=None, metavar="CMD",
        help="DANGER — EXECUTES CMD THROUGH A SHELL with cwd=--repo. On an "
             "untrusted pull request this is remote code execution: pytest "
             "imports the PR's conftest.py at collection, `npm test` runs the "
             "PR's package.json, addopts loads plugins, and os.environ is "
             "inherited unscrubbed. The PR author also controls the exit code in "
             "both directions. Correct in first-party CI on a repo you own; "
             "never on a stranger's branch. Prefer --evidence. Exit 0 gives "
             "VERIFIED; any other exit gives UNCHECKABLE, never an accusation.")
    ap.add_argument(
        "--pr", default=None, metavar="URL",
        help="a public GitHub pull request URL. The description and the diff are "
             "read from api.github.com and gated with no checkout, exactly as the "
             "GitHub Action does; SUMMARY, if also given, replaces the description. "
             "Refuses --run: there is no repository to run anything in, and it "
             "would be someone else's branch.")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.demo:
        return _demo()
    if a.pr:
        if a.run:
            ap.error("--run needs a checkout and --pr has none; it would also execute a "
                     "stranger's branch. Use --evidence, or check the PR out.")
        try:
            pr = fetch_pr(a.pr, token=os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN"))
        except ValueError as e:
            ap.error(str(e))
        text = Path(a.summary).read_text(encoding="utf-8") if a.summary else pr["body"]
        print(f"# {pr['repo']}#{pr['number']} — description and diff read from api.github.com, "
              f"no checkout; base {pr['base'] or '?'}, head {pr['head'][:12] or '?'}"
              + ("" if pr["body"].strip() or a.summary else " — the description is EMPTY"))
        g = gate_diff_text(text, pr["diff"], strict=a.strict,
                           evidence=a.evidence, commit=a.commit)
        g.base, g.head = f"{pr['repo']}#{pr['number']}:{pr['base']}", pr["head"]
        return _report(g, a.out)
    if not a.summary or not a.base:
        ap.error("summary and --base are required (or try --demo, or --pr URL)")
    if a.run:
        # Said out loud on every run, not left in --help for a reader to find.
        print(f"--run WILL EXECUTE {a.run!r} through a shell in {a.repo!r}. That "
              "is code execution; do not point it at a pull request you did not "
              "write. --evidence reads bytes and executes nothing.")
    text = Path(a.summary).read_text(encoding="utf-8")
    g = gate_diff(text, a.repo, a.base, a.head, run=a.run, strict=a.strict,
                  evidence=a.evidence, commit=a.commit)
    return _report(g, a.out)


def _report(g: "DiffGate", out: str | None) -> int:
    """Print a gate the way this CLI always has, and return its exit code."""
    if out:
        Path(out).write_text(json.dumps(g.to_dict(), indent=2) + "\n", encoding="utf-8")
    if not g.measured:
        print(f"UNMEASURED  this gate did not run: {g.why_unmeasured}")
        print("            a PASS here would mean 'nothing contradicted the summary',")
        print("            which is true of any summary when there is no diff to read.")
    print(f"{g.verdict}  claims={len(g.claims)} "
          f"contradicted={sum(1 for c in g.claims if c.verdict == 'CONTRADICTED')} "
          f"uncheckable={sum(1 for c in g.claims if c.verdict == 'UNCHECKABLE')} "
          f"uncovered_sentences={g.uncovered_sentences}")
    if g.sentences_total:
        # The boundary, confessed on every run: a PASS over N sentences the gate
        # never read is a PASS over the templates, not over the summary.
        print(f"never read: {g.uncovered_sentences} of {g.sentences_total} "
              f"sentences — prose outside the closed template set is listed "
              f"in --out, not judged")
        if g.unparsed_claims:
            print(f"            of those, {len(g.unparsed_claims)} look like claims a "
                  f"structural reader would check but these templates cannot parse")
    for c in g.claims:
        if c.verdict != "VERIFIED":
            print(f"  [{c.verdict}:{c.kind}] {c.why}")
    if not g.claims:
        print("\nno diff-shaped claims found — silence is scope, not weakness:")
        print(_WHAT_IT_CHECKS)
    return 0 if g.verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
