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
    return any(p.lower().endswith(_PY_SUFFIXES) for p in status)


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
    low = _norm(raw).rstrip("/").lower()
    for changed in status:
        if low in (seg.lower() for seg in changed.split("/")):
            return True
    return False


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


def parse_unified_diff_sides(diff_text: str) -> dict:
    """Unified diff text -> {normalized new-or-old path: (added_lines, removed_lines)}.

    The per-file companion of `parse_unified_diff`, added for COMPAT-1; the original's
    return shape is untouched because callers unpack it. A header without a `---`/`+++`
    pair (BIN-1) registers its path with empty sides.
    """
    sides: dict = {}
    old_path = None
    cur = None
    pending: _Pending | None = None

    def flush() -> None:
        if pending is not None and pending.path():
            sides.setdefault(pending.path(), ([], []))

    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            flush()
            pending = _Pending(line)
            cur = None
        elif line.startswith("--- "):
            old_path = line[4:].strip()
            cur = None
        elif line.startswith("+++ "):
            new = line[4:].strip()
            if new == "/dev/null":
                raw = old_path[2:] if old_path and old_path.startswith("a/") else (old_path or "")
            else:
                raw = new[2:] if new.startswith("b/") else new
            cur = _norm(raw)
            sides.setdefault(cur, ([], []))
            pending = None
        elif cur is not None and line.startswith("+") and not line.startswith("+++"):
            sides[cur][0].append(line[1:])
        elif cur is not None and line.startswith("-") and not line.startswith("---"):
            sides[cur][1].append(line[1:])
        elif pending is not None:
            pending.note(line)
    flush()
    return sides


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
    by_lang_added: dict = {}
    langs_present: list = []
    for path, (added, _removed) in sides.items():
        for lang, (sufs, _rx) in _COMPAT_LANGS.items():
            if path.endswith(sufs):
                by_lang_added.setdefault(lang, []).extend(added)
                if lang not in langs_present:
                    langs_present.append(lang)
    dropped: list = []
    changed: list = []
    for path, (_added, removed) in sides.items():
        for lang, (sufs, rx) in _COMPAT_LANGS.items():
            if not path.endswith(sufs):
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
                    dropped.append((path, lang, name, not _COMPAT_SCAFFOLD.search(path)))
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


def _norm(p: str) -> str:
    return p.replace("\\", "/").lstrip("./").lower()


def parse_unified_diff(diff_text: str) -> tuple[dict[str, str], str]:
    """Unified diff text -> ({normalized_path: A|M|D}, added-lines blob).

    Lets the gate run on a raw ``.diff`` (webhook payloads, GitHub's ``.diff`` URL) with
    no checkout at all — the zero-receipt promise taken literally.
    """
    status: dict[str, str] = {}
    added: list[str] = []
    old_path = None
    pending: _Pending | None = None          # BIN-1: a header still waiting for its pair

    def flush() -> None:
        if pending is not None and pending.path() and pending.path() not in status:
            status[pending.path()] = pending.status

    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            flush()
            pending = _Pending(line)
        elif line.startswith("--- "):
            old_path = line[4:].strip()
        elif line.startswith("+++ "):
            new = line[4:].strip()
            if new == "/dev/null":
                status[_norm(old_path[2:] if old_path.startswith("a/") else old_path)] = "D"
            elif old_path in ("/dev/null", None):
                status[_norm(new[2:] if new.startswith("b/") else new)] = "A"
            else:
                status[_norm(new[2:] if new.startswith("b/") else new)] = "M"
            pending = None
        elif line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:])
        elif pending is not None:
            pending.note(line)
    flush()
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
    g = _gate(summary_text, status, added_blob, run=run, strict=strict,
                 repo=repo, base="(diff-text)", head="(diff-text)",
                 evidence=evidence, commit=commit,
                 raw_input_len=len(diff_text or ""),
                 sides=parse_unified_diff_sides(diff_text or ""))
    return _p2a_abstain(g, strict, lambda: _P2aFacts(diff_text or "", None, summary_text))  # PATH-2a


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
    for line in name_status.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            st, path = parts[0][:1], parts[-1]
            status[_norm(path)] = st            # A / M / D / R
    diff_text = _git(repo, "diff", f"{base}..{head}")
    added_lines = [l[1:] for l in diff_text.splitlines()
                   if l.startswith("+") and not l.startswith("+++")]
    added_blob = "\n".join(added_lines)
    g = _gate(summary_text, status, added_blob, run=run, strict=strict,
                 repo=repo, base=base, head=head,
                 evidence=evidence, commit=commit,
                 sides=parse_unified_diff_sides(diff_text))
    return _p2a_abstain(g, strict, lambda: _P2aFacts(diff_text, name_status, summary_text))  # PATH-2a


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

    def find_path(claimed: str):
        c = _norm(claimed)
        for p, st in status.items():
            if p == c or p.endswith("/" + c) or Path(p).name == Path(c).name:
                return p, st
        return None, None

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
                    c.verdict, c.why = _path_claim_verdict(kind, d["path"],
                                                           find_path)
                elif kind == "files_changed_count":
                    n = int(d["n"])
                    if no_paths:
                        c.verdict, c.why = "UNCHECKABLE", no_paths
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
                        got = len(re.findall(r"^\s*def test_", added_blob, re.M))
                        if got == n:
                            c.verdict, c.why = "VERIFIED", f"diff adds {got} test functions, claim says {n}"
                        elif BC1_BY_CONSTRUCTION and noun in _TEST_NOUNS_NOT_FUNCTIONS:
                            # BC-2 repair 2: a case, file, scenario, suite or class is not a
                            # function; a matching count verifies, a differing one abstains.
                            c.verdict = "UNCHECKABLE"
                            one = {"classes": "class", "cases": "case", "files": "file",
                                   "scenarios": "scenario", "suites": "suite"}.get(noun, noun)
                            c.why = (f"counts test {noun}, diff adds {got} test functions; "
                                     f"a {one} is not a function (#110)")
                        else:
                            c.verdict = "CONTRADICTED"
                            c.why = f"diff adds {got} test functions, claim says {n}"
                elif kind == "symbol_added":
                    if BC1_BY_CONSTRUCTION and not _diff_touches_python(status):
                        c.verdict = "UNCHECKABLE"           # BC-1 repair 1
                        c.why = ("no Python file in the diff; this template counts "
                                 "`def` lines (#110)")
                    else:
                        pat = (r"^\s*(?:def|class)\s+" + re.escape(d["name"]) + r"\b")
                        hit = bool(re.search(pat, added_blob, re.M))
                        c.verdict = "VERIFIED" if hit else "CONTRADICTED"
                        c.why = (f"added lines {'do' if hit else 'do NOT'} define "
                                 f"{d['kind']} {d['name']!r}")
                elif kind == "only_touches":
                    prefs = [_norm(d["prefix"]).rstrip("/.")]   # sentence-final periods are not path
                    if d.get("prefix2"):
                        prefs.append(_norm(d["prefix2"]).rstrip("/."))
                    # BC-2 repair 4: a second prefix is read only after "and" and only when it
                    # is path-shaped by the same test; otherwise the first prefix decides alone.
                    if d.get("prefix2") and not _prefix_is_path_shaped(d["prefix2"], status):
                        prefs = prefs[:1]
                    raw_prefs = [d["prefix"]] + ([d["prefix2"]] if len(prefs) == 2 else [])
                    not_paths = [_norm(x).rstrip("/.") for x in raw_prefs
                                 if not _prefix_is_path_shaped(x, status)] if BC1_BY_CONSTRUCTION else []
                    # PATH-1 mode 1: _path_inside matches a bare filename on its basename.
                    outside = [p for p in status
                               if not any(_path_inside(p, x) for x in prefs)]
                    if no_paths:
                        c.verdict, c.why = "UNCHECKABLE", no_paths
                    elif not_paths:                             # BC-1 repair 4
                        c.verdict = "UNCHECKABLE"
                        c.why = f"prefix {not_paths[0]!r} is not a path (#110)"
                    else:
                        c.verdict = "VERIFIED" if not outside else "CONTRADICTED"
                        shown = prefs[0] if len(prefs) == 1 else " and ".join(repr(x) for x in prefs)
                        c.why = ("all changed paths under prefix" if not outside else
                                 (f"paths outside {shown!r}: {outside[:3]}" if len(prefs) == 1
                                  else f"paths outside {shown}: {outside[:3]}"))
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
                             raw_input_len=raw_input_len, sides=sides, _declared=True)
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


# === PATH-2a abstain-only overlay: BEGIN ===
#
# NOTE_path2a_abstain_overlay_2026_09_30, NOTE_path2a_second_pass_2026_09_30, NOTE_path2a_third_pass_2026_09_30,
# NOTE_path2a_fourth_pass_2026_09_30, NOTE_path2a_fifth_pass_2026_09_30 and NOTE_path2a_sixth_pass_2026_09_30.
# Everything outside this block is main's reader at 1cde8b82 (sha256 9b620e00..., LF), unchanged; the two doors call
# `_p2a_abstain` on the gate main's `_gate` returns. The overlay reads each DECIDED claim once more and turns it
# UNCHECKABLE, with a reason that names the verdict it withholds, the defect and main's own reason verbatim, only
# where #97, #121 or #101 can have made it wrong:
#
#   #97   find_path takes the earliest entry in diff order matching by exact path, suffix OR base name;
#   #121  _norm's lstrip("./") drops leading dots, so `.env` and `env` share one key;
#   #101  the `def` counts read added lines only, so a changed test or function reads as added.
#
# A path or count claim is kept only when the readers without the defect -- V97 (exact, then suffix, then base
# name for a bare claim), V121 (keys keep their leading dots) and both together -- decide it the same way,
# computed from the paths main registered, as written. A test or symbol claim is kept only when no count that
# treats a changed `def` as not added could decide it otherwise. Every comparison is structural: ASCII case,
# bytes, '/', '.', and fixed character sets. No lower(), no \w \s \b \d, no unicodedata, no pathlib.
# Where the overlay cannot read exactly (a claim the two ports' templates may extract apart, a header or line the
# two ports split apart, a drive-like path, case outside ASCII, a reason it does not parse, or its own failure) it
# abstains, and says so. A decision reads the claim's kind, verdict and detail, main's own counts in its reason,
# and the door's bytes (the diff, the name-status listing, the summary); never the claim's text, which the two ports
# cut and strip differently. Everything that does not depend on the claim is computed once per diff.

P2A_DIRECTORY_BASENAME_ABSTAINS = True   # V97 resolves a claim whose path has a directory part by exact path or suffix

_P2A_PY_BREAKS = "\n\r\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029"            # str.splitlines(); tests pin it
_P2A_PY_SPACE = ("\t\n\x0b\x0c\r\x1c\x1d\x1e\x1f \x85\xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006"
                 "\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000")   # str.isspace(); tests pin it
_P2A_JS_SPACE = ("\t\n\x0b\x0c\r \xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a"
                 "\u2028\u2029\u202f\u205f\u3000\ufeff")         # the port's \s and trim(); tests pin it
_P2A_DIVERGENT = "\x0b\x0c\x1c\x1d\x1e\x1f\x85\u2028\u2029\ufeff"       # where the two ports' readers part
# Where the two ports' count templates can read a count apart that neither port reads beside a digit
# (NOTE_path2a_fifth_pass_2026_09_30, C-1; NOTE_path2a_fifth_pass_corrections_2026_09_30): the white space only one of
# them reads (\s, here CPython's U+001C to U+001F and U+0085, there U+FEFF), in a run of either port's white space
# between a character that can end a number or a word of the count (an ASCII digit, e, s, or U+017F) and one that can
# begin one (c, f or w); and 'file' spelled with a code point CPython's IGNORECASE folds to i (U+0130, U+0131) or s
# (U+017F), where the port's folds none. U+212A folds to k, which no word of the count holds. Tests pin each set.
_P2A_ONE_SPACE = "\x1c\x1d\x1e\x1f\x85\ufeff"
_P2A_ANY_SPACE = _P2A_PY_SPACE + "\ufeff"
# Code points from 0x80 up that neither port's templates read as a word character or fold to an ASCII letter, that
# have no case, and that both read alike as white space or not (NOTE_path2a_fourth_pass_2026_09_30, B-2): the
# punctuation and symbols of the Basic Multilingual Plane's punctuation, arrow, operator, technical, box, shape,
# symbol, dingbat, CJK-punctuation and full-width-punctuation blocks (connector punctuation, digits and letters left
# out), the white space both ports share, and the two emoji variation selectors, as a regex class. The summary's
# checks below do not count them. Tests pin every property named here by enumeration, on every runtime they run.
_P2A_NEUTRAL = ("\xa0-\xa9\xab\xac\xae-\xb1\xb4\xb6-\xb8\xbb\xbf\xd7\xf7\u1680\u2000-\u200a\u2010-\u2027"
                "\u202f-\u203e\u2041-\u2053\u2055-\u205f\u20a0-\u20bf\u2190-\u23ff\u2500-\u2775\u2794-\u27bf"
                "\u2b00-\u2b73\u2b76-\u2b95\u2b97-\u2bff\u3000-\u3004\u3008-\u3020\u3030\u3036\u3037\u303d-\u303f"
                "\ufe0e-\ufe19\ufe30-\ufe32\ufe35-\ufe4c\ufe50-\ufe52\ufe54-\ufe66\ufe68-\ufe6b\uff01-\uff0f"
                "\uff1a-\uff20\uff3b-\uff3e\uff40\uff5b-\uff65")
# Every code point from 0x80 up except the neutral ones and U+0085, U+2028, U+2029 and U+FEFF (never word characters
# either): the characters CPython's templates may read as a word character, or fold to an ASCII letter, where the
# port's templates read neither. Tests pin these ranges against the sets above.
_P2A_WORDISH = ("\x80-\x84\x86-\x9f\xaa\xad\xb2\xb3\xb5\xb9\xba\xbc-\xbe\xc0-\xd6\xd8-\xf6\xf8-\u167f\u1681-\u1fff"
                "\u200b-\u200f\u202a-\u202e\u203f\u2040\u2054\u2060-\u209f\u20c0-\u218f\u2400-\u24ff\u2776-\u2793"
                "\u27c0-\u2aff\u2b74\u2b75\u2b96\u2c00-\u2fff\u3005-\u3007\u3021-\u302f\u3031-\u3035\u3038-\u303c"
                "\u3040-\ufe0d\ufe1a-\ufe2f\ufe33\ufe34\ufe4d-\ufe4f\ufe53\ufe67\ufe6c-\ufefe\uff00\uff10-\uff19"
                "\uff21-\uff3a\uff3f\uff41-\uff5a\uff66-\U0010ffff")
_P2A_OWN = 0                    # this port's main reads line view 0 (CPython's breaks) with CPython's \s
_P2A_HEADERS = ("diff --git ", "--- ", "+++ ", "rename from ", "rename to ", "new file mode",
                "deleted file mode", "Binary files ")
_P2A_FINE = re.compile("\r\n|[" + _P2A_PY_BREAKS + "]")
_P2A_COARSE = re.compile("\r\n|\r|\n")
_P2A_CR = re.compile("\r")
_P2A_FINE_ONLY = re.compile("[\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029]")     # where the two splits above can part
_P2A_COUNT_HEAD = re.compile("diff changes ([0-9]+) files, claim says ")
_P2A_TESTS_HEAD = re.compile("diff adds ([0-9]+) test functions, claim says ")
_P2A_DIGITS = re.compile("[0-9]+")
_P2A_DIV_RX = re.compile("[" + _P2A_DIVERGENT + "]")
_P2A_ONE_RX = re.compile("[" + _P2A_ONE_SPACE + "]")
_P2A_ANY_SPACE_RX = re.compile("[" + _P2A_ANY_SPACE + "]+")
_P2A_FILE_FOLD_RX = re.compile("[Ff][\u0130\u0131][Ll][Ee]|[Ff][Ii\u0130\u0131][Ll][Ee]\u017f")
_P2A_WIDE = re.compile("[\x80-\U0010ffff]")                              # a code point outside ASCII
_P2A_WORDISH_RX = re.compile("[" + _P2A_WORDISH + "]")
_P2A_BAD_RX = re.compile("[" + _P2A_DIVERGENT + _P2A_WORDISH + "]")      # a character the two ports may read apart
_P2A_PATH_RUN = re.compile("[A-Za-z0-9_./\\\\" + _P2A_WORDISH + "-]+")   # characters a path template may read
_P2A_NAME_RUN_ANY = re.compile("[A-Za-z0-9_" + _P2A_WORDISH + "]+")      # characters a name template may read
_P2A_COUNT_RUN = re.compile("[0-9" + _P2A_WORDISH + "]+")                 # characters a count template may read
_P2A_RUN_RX = {"path": _P2A_PATH_RUN, "name": _P2A_NAME_RUN_ANY, "count": _P2A_COUNT_RUN}
_P2A_DIGIT = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9}
_P2A_SEP = "\x00"                  # joins the summary's runs and zones; never in a claimed path, name, prefix or number
_P2A_CUT = re.compile("\n|[.!?][ \t\r]")          # where both ports end a sentence: a line break, or '.!?' + a space
_P2A_COARSE_RUN = re.compile("[\x00-\x20\x7f-\U0010ffff]*")              # controls, space, DEL, from 0x80 up
_P2A_WORD_RUN = re.compile("[A-Za-z0-9_]*")                              # an ASCII name run
_P2A_NAME_RUN = re.compile("[A-Za-z0-9_\x80-\U0010ffff]*")               # a name run: ASCII word, or from 0x80
_P2A_SEG = re.compile("[\u2028\u2029]")
_P2A_LEAD_APART = re.compile("[\x1c\x1d\x1e\x1f\x85\ufeff]")    # white space of one main only
_P2A_LEADS = (re.compile("[" + _P2A_PY_SPACE + "]*"), re.compile("[" + _P2A_JS_SPACE + "]*"))  # each main's \s run
_P2A_FOLD = {**{cp: cp + 32 for cp in range(65, 91)}, 0x212A: "k", 0x130: "i\u0307"}
_P2A_ASCII_LOWER = {cp: cp + 32 for cp in range(65, 91)}
# C-1 (NOTE_path2a_sixth_pass_2026_09_30): words every match of a template that can give CONTRADICTED holds together
# (the count template's `changed`, the scope template's `only`, the tests template's verb and `test`, the symbol
# template's verb and kind), and every DECLARE-1 line that writes such a sentence (files_changed, only_touches,
# tests_added, adds_symbol), read in ASCII case with the four code points CPython's IGNORECASE folds to an ASCII letter
# read as that letter, one for one.
_P2A_TRIGGERS = (("changed",), ("only",), ("add", "test"), ("creat", "test"),
                 ("add", "function"), ("add", "class"), ("add", "method"), ("add", "symbol"),
                 ("introduc", "function"), ("introduc", "class"), ("introduc", "method"), ("introduc", "symbol"))
_P2A_TRIGGER_LOW = {**_P2A_ASCII_LOWER, 0x130: "i", 0x131: "i", 0x17F: "s", 0x212A: "k"}
_P2A_ACCUSE = frozenset({"files_changed_count", "only_touches", "tests_added", "symbol_added"})   # may be CONTRADICTED
_P2A_MANY = 32                     # more distinct words than this are read through one automaton, not one scan each
_P2A_REACH = frozenset({
    ("file_created", "VERIFIED"), ("file_deleted", "VERIFIED"), ("file_touched", "VERIFIED"),
    ("files_changed_count", "VERIFIED"), ("files_changed_count", "CONTRADICTED"),
    ("only_touches", "VERIFIED"), ("only_touches", "CONTRADICTED"),
    ("tests_added", "VERIFIED"), ("tests_added", "CONTRADICTED"), ("symbol_added", "VERIFIED")})
_P2A_KIND_DEFECT = {"file_created": "#97, #121", "file_deleted": "#97, #121", "file_touched": "#97, #121",
                    "files_changed_count": "#121", "only_touches": "#121", "tests_added": "#101",
                    "symbol_added": "#101"}
_P2A_PHRASES = {
    "dir": ("the claim's path has a directory part, and only a changed file with the same base name in another "
            "directory matches it"),
    "tier": "a changed path that matches the claim more closely than the one main resolved it to reads otherwise",
    "dot": "with leading dots kept, the changed path the claim resolves to reads otherwise",
    "dot_earliest": ("with leading dots kept, the earliest changed path matching the claim reads otherwise, though "
                     "the closest one does not"),
    "dot_tier": "with leading dots kept and the closest match taken, the claim reads otherwise",
    "count": ("two changed paths differ only by a leading dot, which the path key drops, and counted apart "
              "the claim reads otherwise"),
    "only": "with leading dots kept, whether every changed path lies under the prefix reads otherwise",
    "shape": "with leading dots kept, no changed path has the prefix as a segment, so it is not read as a path",
    "tests": ("a test the added lines count is also defined in the removed lines, and a changed test is not "
              "an added one"),
    "split": ("a test the added lines count is also defined in the removed lines, and the Python and JavaScript "
              "readers split or space these lines differently"),
    "redefined": ("a test the added lines count is also defined in an unchanged line of the diff, and a test defined "
                  "again is not an added one"),
    "symbol": "the removed lines define this name too, and a changed definition is not an added one",
    "again": "an unchanged line of the diff defines this name too, and a name defined again is not an added one",
    "extract": ("the summary holds a character the Python and JavaScript readers may read differently where the "
                "claim's path, name, prefix or number is read, so the two may extract the claim differently"),
    "seam": ("the summary holds a character that only one of the Python and JavaScript readers reads as a space, or "
             "as a letter, where a count is read, so the two may read different counts"),
    "divergent": ("a file header of this diff holds a character that the Python and JavaScript readers split "
                  "or strip differently"),
    "odd": "a path here has a drive-like prefix or a final '.' segment, where base names are read differently",
    "case": "a path here compares only where case outside ASCII is folded, which this overlay does not do",
    "unreproduced": "this overlay does not reproduce main's reading of the diff",
    "unparsed": "main's reason does not have the form this overlay reads",
    "error": "this overlay failed while reading the diff",
}


def _p2a_lines(text: str, rx) -> list:
    out = rx.split(text)
    if out and out[-1] == "":
        out.pop()
    return out


def _p2a_regs_raw(lines: list) -> list:
    """(path as written, status, via) wherever parse_unified_diff hands a path to _norm, in its order: its loop
    line for line (`lines`, the diff split at CPython's line breaks), `status[_norm(x)] = st` read as an append; via
    '+' assigns, 'p' (a BIN-1 flush) registers only a key not yet held. A flush is kept even where its key is already
    registered."""
    regs: list = []
    old_path = None
    pending = None

    def flush() -> None:
        if pending is not None:
            raw = pending.a if pending.status == "D" else pending.b
            if raw:
                regs.append((raw, pending.status, "p"))

    for line in lines:
        if line.startswith("diff --git "):
            flush()
            pending = _Pending(line)
        elif line.startswith("--- "):
            old_path = line[4:].strip(_P2A_PY_SPACE)
        elif line.startswith("+++ "):
            new = line[4:].strip(_P2A_PY_SPACE)
            if new == "/dev/null":
                regs.append((old_path[2:] if old_path.startswith("a/") else old_path, "D", "+"))
            elif old_path in ("/dev/null", None):
                regs.append((new[2:] if new.startswith("b/") else new, "A", "+"))
            else:
                regs.append((new[2:] if new.startswith("b/") else new, "M", "+"))
            pending = None
        elif line.startswith("+") and not line.startswith("+++"):
            pass
        elif pending is not None:
            pending.note(line)
    flush()
    return regs


def _p2a_regs_git(name_status: str) -> list:
    regs = []
    for line in _p2a_lines(name_status, _P2A_FINE):
        parts = line.split("\t")
        if len(parts) >= 2:
            regs.append((parts[-1], parts[0][:1], "+"))
    return regs


def _p2a_view(lines: list) -> tuple:
    return ([x[1:] for x in lines if x.startswith("+") and not x.startswith("+++")],
            [x[1:] for x in lines if x.startswith("-") and not x.startswith("---")])


def _p2a_views(diff_text: str, fine: list) -> list:
    """[(added, removed)] under CPython's line breaks (`fine`, the diff split at them) and under the port's; each
    port's main reads one. Where the text holds no break only CPython's split reads, the two are one object."""
    v0 = _p2a_view(fine)
    if not _P2A_FINE_ONLY.search(diff_text):
        return [v0, v0]
    return [v0, _p2a_view(_p2a_lines(diff_text, _P2A_COARSE))]


def _p2a_divergent(diff_text: str) -> bool:
    if not _P2A_DIV_RX.search(diff_text):
        return False
    for line in _p2a_lines(diff_text, _P2A_COARSE):
        if _P2A_DIV_RX.search(line):
            for piece in [line] + _p2a_lines(line, _P2A_FINE):
                if piece.startswith(_P2A_HEADERS):
                    return True
    return False


def _p2a_bs(s: str) -> str:
    return s.replace("\\", "/")


def _p2a_strip(s: str) -> str:            # main's key before lower()
    return _p2a_bs(s).lstrip("./")


def _p2a_dotted(s: str) -> str:           # PREREG_path2 R-121's key before lower(): ^(?:\.?/)+ removed
    s = _p2a_bs(s)
    i = 0                                 # one scan, one slice (NOTE_path2a_fifth_pass_2026_09_30, A-3)
    while True:
        if s.startswith("./", i):
            i = i + 2
        elif s.startswith("/", i):
            i = i + 1
        else:
            return s[i:]


def _p2a_run(s: str) -> str:              # the leading dots and slashes main drops and R-121 keeps
    d = _p2a_dotted(s)
    return d[:len(d) - len(_p2a_strip(s))]


def _p2a_fold(s: str) -> str:             # lower() on ASCII, and the two code points whose lower() holds ASCII
    return s.translate(_P2A_FOLD)


def _p2a_wild(s: str) -> str:             # every code point outside ASCII read as one placeholder
    return _P2A_WIDE.sub("\ufffd", _p2a_fold(s))


def _p2a_A(s: str) -> str:
    return _p2a_fold(_p2a_strip(s))


def _p2a_K(s: str) -> str:
    return _p2a_fold(_p2a_dotted(s))


def _p2a_WA(s: str) -> str:
    return _p2a_wild(_p2a_strip(s))


def _p2a_WK(s: str) -> str:
    return _p2a_wild(_p2a_dotted(s))


_P2A_FORMS = {"A": (_p2a_A, _p2a_WA), "K": (_p2a_K, _p2a_WK)}


def _p2a_base(p: str) -> str:             # the port's _basename; Path(p).name agrees unless _p2a_odd(p)
    q = p.rstrip("/")
    return q[q.rfind("/") + 1:]


def _p2a_odd(p: str) -> bool:             # a final "." segment, or a drive-like second code point not before '/'
    q = p.rstrip("/")
    return q == "." or q.endswith("/.") or (len(q) >= 2 and q[1] == ":" and (len(q) == 2 or q[2] != "/"))


def _p2a_tier(p: str, c: str):
    if p == c:
        return 0
    if p.endswith("/" + c):
        return 1
    if _p2a_base(p) == _p2a_base(c):
        return 2
    return None


def _p2a_build(regs: list, key) -> dict:
    """main's status map over (raw, st, via), keyed by `key`: '+' assigns; 'p' registers a non-empty key once."""
    m: dict = {}
    for raw, st, via in regs:
        k = key(raw)
        if via == "+":
            m[k] = st
        elif k and k not in m:
            m[k] = st
    return m


def _p2a_scan(m: dict, c: str, tiered: bool):
    """_p2a_resolve read key by key, for a claim whose base name is empty."""
    for t in ((0, 1, 2) if tiered else (None,)):
        if t == 2 and "/" in c and P2A_DIRECTORY_BASENAME_ABSTAINS:
            return None
        for p in m:
            u = _p2a_tier(p, c)
            if u is not None and (t is None or u == t):
                return p, m[p]
    return None


def _p2a_resolve(f: "_P2aFacts", space: str, c: str, tiered: bool):
    """(key, status) main's find_path returns (tiered=False), or V97's: exact, then suffix, then base name, over the
    status map of `space`. With a non-empty base name, the keys that match the claim at some tier are exactly the keys
    of its base name, so main takes the earliest of those; V97 takes the claim itself, else the earliest key ending
    in '/' + the claim (read through the claims' tree, `_p2a_ends`), else (only for a bare claim) the earliest key of
    its base name."""
    m = f.status(space)
    b = _p2a_base(c)
    if not b:
        return _p2a_scan(m, c, tiered)
    cand = f.groups(space).get(b)
    if not cand:
        return None
    if not tiered:
        return cand[0], m[cand[0]]
    if c in m:
        return c, m[c]
    i = f.ends(space, "key", c)[0]
    if i >= 0:
        p = f.order(space)[i]
        return p, m[p]
    if "/" in c and P2A_DIRECTORY_BASENAME_ABSTAINS:
        return None
    return cand[0], m[cand[0]]


def _p2a_tree(claims) -> dict:
    """The claims' '/'-segments, read from their ends, as nested dicts; under the key None, the claim ending there."""
    root: dict = {}
    for c in claims:
        node = root
        segs = c.split("/")
        k = len(segs)
        while k > 0:
            k = k - 1
            node = node.setdefault(segs[k], {})
        node[None] = c
    return root


def _p2a_ends(texts: list, claims) -> dict:
    """{c: [the index of the earliest text that ends in '/' + c, or -1; how many texts do]} for each c of `claims`. A
    text ends in '/' + c exactly when its '/'-segments end in c's and it has at least one more, so each text is read
    from its end one segment at a time through the claims' tree, and only while the segments read so far end some
    claim's: time linear in the texts and the claims, memory linear in the claims. No text's suffixes are stored
    (NOTE_path2a_fifth_pass_2026_09_30, A-1)."""
    root = _p2a_tree(claims)
    out = {c: [-1, 0] for c in claims}
    for i, p in enumerate(texts):
        node = root
        j = len(p)
        k = p.rfind("/")
        while k >= 0:
            node = node.get(p[k + 1:j])
            if node is None:
                break
            c = node.get(None)
            if c is not None:
                got = out[c]
                if got[0] < 0:
                    got[0] = i
                got[1] = got[1] + 1
            j = k
            k = p.rfind("/", 0, j)
    return out


def _p2a_automaton(words) -> tuple:
    """An Aho-Corasick automaton over `words` (non-empty strings): per state, its moves, its fallback, and whether a
    word ends there or at a fallback of it. Built in time linear in the words (NOTE_path2a_fifth_pass_2026_09_30, A-2)."""
    moves: list = [{}]
    hit = [False]
    for w in words:
        s = 0
        for ch in w:
            t = moves[s].get(ch)
            if t is None:
                t = len(moves)
                moves.append({})
                hit.append(False)
                moves[s][ch] = t
            s = t
        hit[s] = True
    back = [0] * len(moves)
    queue = [t for _ch, t in moves[0].items()]
    k = 0
    while k < len(queue):
        s = queue[k]
        k = k + 1
        for ch, t in moves[s].items():
            u = back[s]
            while u and ch not in moves[u]:
                u = back[u]
            v = moves[u].get(ch, 0)
            back[t] = v
            hit[t] = hit[t] or hit[v]
            queue.append(t)
    return moves, back, hit


def _p2a_holds(auto: tuple, text: str) -> bool:
    """Whether some word of the automaton occurs in `text`, read once, code point by code point."""
    moves, back, hit = auto
    s = 0
    for ch in text:
        while s and ch not in moves[s]:
            s = back[s]
        s = moves[s].get(ch, 0)
        if hit[s]:
            return True
    return False


def _p2a_sites(line: str, word: str) -> list:
    """(j, r) for each `word` in the line followed by one or more coarse characters, which run from j to r (the
    character that ends them, or the end of the line). The runs of distinct sites never overlap."""
    out = []
    i = line.find(word)
    while i >= 0:
        j = i + len(word)
        r = _P2A_COARSE_RUN.match(line, j).end()
        if r > j:
            out.append((j, r))
        i = line.find(word, max(i + 1, r))
    return out


def _p2a_counted(line: str, lead) -> list:
    """The ASCII name runs at the `def test_` sites of one added line that a main's count of `def test_` after
    white space reads, where `lead` matches a run of that main's white space: a site counts when, from the line
    start or the last U+2028 / U+2029 before it (where the port's `^` also matches), only that run precedes it.
    Line view 0 holds no U+2028 or U+2029, so for CPython's count only the line start is read. A site counts exactly
    where a segment's leading run ends (`d` is no white space), and a start inside a run already read reaches the same
    end, so each segment is read once, never each `def test_` of the line (NOTE_path2a_fourth_pass_2026_09_30, A-2)."""
    if "def test_" not in line:
        return []
    starts = [0]
    if "\u2028" in line or "\u2029" in line:
        starts += [m.end() for m in _P2A_SEG.finditer(line)]
    out = []
    last = -1
    for b in starts:
        if b <= last:
            continue
        r = lead.match(line, b).end()
        last = r
        if line.startswith("def test_", r):
            e = _P2A_WORD_RUN.match(line, r + 4).end()
            out.append(None if _P2A_WIDE.match(line, e) else line[r + 4:e])
    return out


def _p2a_wide_name(line: str, j, r, e) -> bool:
    """Whether the coarse run [j, r) after a `def` or `class`, or the character that ends the ASCII name run [r, e),
    is a code point from 0x80 up: CPython reads such a name through NFKC, so it may be any name, ASCII or not
    (NOTE_path2a_fifth_pass_2026_09_30, B-2). The runs of distinct sites never overlap, so reading them is linear."""
    return _P2A_WIDE.search(line, j, r) is not None or _P2A_WIDE.match(line, e) is not None


def _p2a_removed(views: list, extra: list) -> list:
    """The removed lines of each distinct line view, and the removed lines no view reads (`_p2a_joined`)."""
    return [removed for _a, removed in _p2a_distinct(views)] + [extra]


def _p2a_test_defs(groups: list, nfkc: bool = True) -> tuple:
    """(the ASCII name runs of the tests the lines may define after `def`, whether some `def` there has a name read
    through NFKC and so may define any test). A definition is read as the ASCII run of its name: pairing names by
    their sets {full name run, ASCII run} is pairing them by their ASCII runs, since a full run fixes its ASCII run and
    a full run equal to an ASCII run is all ASCII (NOTE_path2a_fifth_pass_2026_09_30, B-2). With `nfkc` False, a `def`
    whose name reads through NFKC is passed over (the unchanged lines, NOTE_path2a_sixth_pass_2026_09_30, B-1)."""
    names = set()
    for lines in groups:
        for line in lines:
            for j, r in _p2a_sites(line, "def"):
                e = _P2A_WORD_RUN.match(line, r).end()
                if _p2a_wide_name(line, j, r, e):
                    if nfkc:
                        return names, True        # every counted site pairs now: nothing more to read
                    continue
                if line.startswith("test_", r):
                    names.add(line[r:e])
    return names, False


def _p2a_pairing(views: list, alike: bool = False, extra: list = (), unchanged: list = ()) -> list:
    """Per line view v (0: CPython's line breaks and white space, as main reads the added lines here; 1: the port's),
    (the number of `def test_` sites that view's main counts, how many of them name a test a removed line may define
    after `def`, how many name a test a removed or an unchanged line may define). A name read through NFKC (a code point
    from 0x80 up in it or before it) pairs with every name: a defined one with every counted site, a counted one with
    every defined test (NOTE_path2a_fifth_pass_2026_09_30, B-2). `extra`: the removed lines no view reads (B-1).
    `unchanged`: the unchanged lines of each view and those no view reads (NOTE_path2a_sixth_pass_2026_09_30, B-1: a
    test defined again beside its own unchanged definition is counted by main and is not an added test). `alike`: the
    two views are one object and the text holds no character where the two mains' white space parts (U+001F, U+FEFF;
    the others are line breaks of view 0), so both views count alike."""
    rem, wild = _p2a_test_defs(_p2a_removed(views, extra))
    ctx, _nfkc = _p2a_test_defs([unchanged], False)
    out = []
    for (added, _r), lead in zip(_p2a_distinct(views) if alike else views, _P2A_LEADS):
        got = paired = based = 0
        for line in added:
            for x in _p2a_counted(line, lead):
                got += 1
                p = wild or (bool(rem) if x is None else x in rem)
                paired += p
                based += p or (bool(ctx) if x is None else x in ctx)
        out.append((got, paired, based))
    return out * 2 if len(out) == 1 else out


def _p2a_distinct(views: list) -> list:
    return views[:1] if views[0] is views[1] else views


def _p2a_context(diff_text: str, fine: list) -> list:
    """The unchanged lines (git lines starting with ' ') of each distinct line view, without the ' '
    (NOTE_path2a_sixth_pass_2026_09_30, B-1)."""
    out = [x[1:] for x in fine if x.startswith(" ")]
    if not _P2A_FINE_ONLY.search(diff_text):
        return out
    return out + [x[1:] for x in _p2a_lines(diff_text, _P2A_COARSE) if x.startswith(" ")]


def _p2a_joined(diff_text: str, side: str = "-") -> list:
    """The removed text no line view reads as a line (NOTE_path2a_fifth_pass_2026_09_30, B-1), as more removed lines:
    each piece after a lone CR inside a git line (the diff split at '\\n') that starts with '-', which no view reads as
    removed and CPython's tokenizer reads as a line of its own; and each run of the base side's pieces (the pieces of
    git lines starting with '-' or ' ', in order, '+' and '\\' lines passed over, any other line ending the run) joined
    where a piece ends in a backslash, which CPython reads as one line, each such backslash read as a space, when a
    piece of the run is removed. The tokenizer ends a line at LF, CRLF and a lone CR only; at any other break
    `str.splitlines` knows, the line does not parse, and the line views read those already. Linear in the diff.
    With `side` ' ', the unchanged text no view reads the same way (NOTE_path2a_sixth_pass_2026_09_30, B-1): the pieces
    after a lone CR in lines starting with ' ', and the joined runs no piece of which is removed."""
    out = []
    acc: list = []
    hit = False

    def flush() -> None:
        if len(acc) > 1 and hit == (side == "-"):
            out.append("".join(x[:-1] + " " for x in acc[:-1]) + acc[-1])

    for line in diff_text.split("\n"):
        head = line[:1]
        if head == "+" or head == "\\":
            continue
        if head != "-" and head != " ":
            flush()
            acc = []
            hit = False
            continue
        pieces = _p2a_lines(line[1:], _P2A_CR) if "\r" in line else [line[1:]] if len(line) > 1 else []
        if head == side:
            for piece in pieces[1:]:
                out.append(piece)
        for piece in pieces:
            if not acc:
                hit = False
            acc.append(piece)
            hit = hit or head == "-"
            if not piece.endswith("\\"):
                flush()
                acc = []
    flush()
    return out


def _p2a_anchored(line: str) -> list:
    """(j, r) for each `def` or `class` of one removed line that V101's symbol check can read a definition after, in
    either port: from the line start (or, a superset, a U+2028 or U+2029), a run of coarse characters, optionally
    `async` and one or more coarse characters, then the word, then one or more coarse characters, which run from j to r
    (NOTE_path2a_fourth_pass_2026_09_30, B-3). A start inside a coarse run already read reaches the same word."""
    starts = [0]
    if "\u2028" in line or "\u2029" in line:
        starts += [m.end() for m in _P2A_SEG.finditer(line)]
    out = []
    last = -1
    for b in starts:
        if b <= last:
            continue
        p = _P2A_COARSE_RUN.match(line, b).end()
        last = p
        at = [p]
        if line.startswith("async", p):
            q = _P2A_COARSE_RUN.match(line, p + 5).end()
            if q > p + 5:
                at.append(q)
        for p in at:
            for w in ("def", "class"):
                if line.startswith(w, p):
                    j = p + len(w)
                    r = _P2A_COARSE_RUN.match(line, j).end()
                    if r > j:
                        out.append((j, r))
    return out


def _p2a_def_runs(views: list, extra: list = ()) -> tuple:
    """(the ASCII name run at every anchored `def` or `class` site of a removed line, in either view or in `extra`;
    whether some such site's name is read through NFKC, `_p2a_wide_name`, and so may be any name). A name of ASCII
    word characters can start only where a site's coarse run ends (the positions inside the run are coarse), and ends
    where its ASCII run does, so it is defined by `_p2a_defines` exactly when it is in this set."""
    out: set = set()
    seen: set = set()
    for removed in _p2a_removed(views, extra):
        for line in removed:
            if line in seen:
                continue
            seen.add(line)
            for j, r in _p2a_anchored(line):
                e = _P2A_WORD_RUN.match(line, r).end()
                if _p2a_wide_name(line, j, r, e):
                    return out, True              # every claimed name is defined now: nothing more to read
                out.add(line[r:e])
    return out, False


def _p2a_name_defs(groups: list) -> set:
    """The ASCII name run at every anchored `def` or `class` site of the lines whose name does not read through NFKC:
    the names the unchanged lines define (NOTE_path2a_sixth_pass_2026_09_30, B-1)."""
    out: set = set()
    seen: set = set()
    for lines in groups:
        for line in lines:
            if line in seen:
                continue
            seen.add(line)
            for j, r in _p2a_anchored(line):
                e = _P2A_WORD_RUN.match(line, r).end()
                if not _p2a_wide_name(line, j, r, e):
                    out.add(line[r:e])
    return out


def _p2a_defines(views: list, name: str, extra: list = ()) -> bool:
    """Whether a removed line, in either view or in `extra`, may define `name` after an anchored `def` or `class`: the
    name starts where the coarse run after the word does or anywhere in it, and ends where its full name run or its
    ASCII run ends. Read this way only for a name outside ASCII word characters; the others are looked up in
    `_p2a_def_runs`."""
    if not name:
        return False
    full = _P2A_NAME_RUN.fullmatch(name) is not None
    word = _P2A_WORD_RUN.fullmatch(name) is not None
    for removed in _p2a_removed(views, extra):
        for line in removed:
            for j, r in _p2a_anchored(line):
                k = line.find(name, j + 1, r + len(name))
                while k >= 0:
                    e = k + len(name)
                    if (full and _P2A_NAME_RUN.match(line, e).end() == e) or \
                            (word and _P2A_WORD_RUN.match(line, e).end() == e):
                        return True
                    k = line.find(name, k + 1, r + len(name))
    return False


def _p2a_runs(summary: str, run_rx) -> str:
    """The summary's maximal runs of path (name, number) characters that hold a character CPython's template may read
    as a word character and the port's does not, joined by _P2A_SEP, which no run holds: a string that holds s, where
    s does not hold _P2A_SEP, exactly when some run does (NOTE_path2a_fourth_pass_2026_09_30, A-1)."""
    return _P2A_SEP.join(m.group() for m in run_rx.finditer(summary) if _P2A_WORDISH_RX.search(m.group()))


def _p2a_seam(summary: str) -> bool:
    """Whether the summary holds a count seam (_P2A_ONE_SPACE, _P2A_FILE_FOLD_RX): read in one pass over the maximal runs
    of either port's white space, and only where a white space of one port alone occurs at all."""
    if _P2A_FILE_FOLD_RX.search(summary):
        return True
    if not _P2A_ONE_RX.search(summary):
        return False
    for m in _P2A_ANY_SPACE_RX.finditer(summary):
        a, e = m.start(), m.end()
        if a > 0 and e < len(summary) and summary[a - 1] in "0123456789EeSs\u017f" and summary[e] in "CcFfWw" \
                and _P2A_ONE_RX.search(summary, a, e):
            return True
    return False


def _p2a_found(words, text: str) -> set:
    """The words (non-empty, none holding _P2A_SEP) that occur in `text`. Up to _P2A_MANY words, one scan each; more,
    one read of the text through one automaton, marking each state's words once: time linear in the words and the
    text however many words the claims name (NOTE_path2a_sixth_pass_2026_09_30, A-1). Both give the same set."""
    if len(words) <= _P2A_MANY:
        return {w for w in words if w in text}
    moves, back, hit = _p2a_automaton(words)
    ends: dict = {}
    for w in words:
        s = 0
        for ch in w:
            s = moves[s][ch]
        ends[s] = w
    seen: set = set()
    got: set = set()
    s = 0
    for ch in text:
        while s and ch not in moves[s]:
            s = back[s]
        s = moves[s].get(ch, 0)
        t = s
        while t and hit[t] and t not in seen:
            seen.add(t)
            if t in ends:
                got.add(ends[t])
            t = back[t]
    return got


def _p2a_apart_in(summary: str, low: str, a, e) -> bool:
    return _P2A_BAD_RX.search(summary, a, e) is not None and any(
        all(low.find(w, a, e) >= 0 for w in words) for words in _P2A_TRIGGERS)


def _p2a_apart_diff(f: "_P2aFacts", kinds: frozenset) -> bool:
    """Whether the two ports' mains may decide apart, on this diff, a claim of `kinds` (the kinds that can be
    CONTRADICTED among the claims main read, which both ports read alike wherever the summary holds no C-1 seam): a
    file header they split or strip apart (the statuses: counts, scopes, and BC-1 for tests and symbols); for a tests
    claim, a different count of `def test_` sites (each view's own count, as main makes it); for a symbol claim, an
    added `def` or `class` site whose line only one view reads, or whose leading white space or name the two ports'
    `\\s` and `\\b` may read apart; and added lines that are empty to one main only where no path is registered (main's
    no-evidence reading)."""
    if not kinds:
        return False
    if _p2a_divergent(f.diff_text):
        return True
    views = f.views()
    if "tests_added" in kinds and f.pairing()[0][0] != f.pairing()[1][0]:
        return True
    if "symbol_added" in kinds:
        one = set(views[0][0]) ^ set(views[1][0]) if views[0] is not views[1] else set()
        for added, _r in _p2a_distinct(views):
            for line in added:
                for j, r in _p2a_anchored(line):
                    if line in one or _P2A_ONE_RX.search(line, 0, r) is not None or \
                            _p2a_wide_name(line, j, r, _P2A_WORD_RUN.match(line, r).end()):
                        return True
    return not f.status("A") and (not "\n".join(views[0][0])) != (not "\n".join(views[1][0]))


def _p2a_apart(f: "_P2aFacts", kinds: frozenset) -> bool:
    """Whether the two ports' mains may read apart which claims can be CONTRADICTED, or decide one such claim apart
    (NOTE_path2a_sixth_pass_2026_09_30, C-1). Where they may, the overlay withholds no CONTRADICTED verdict, so each
    port's gate verdict without --strict is its main's; where they may not, both read the same such claims with the
    same verdicts, and the overlay decides each alike. The summary: a sentence, as both ports end one, that holds a
    character the two ports' templates may read apart and the words every match of such a template holds; or a
    DECLARE-1 fence word beside a line break only one port reads, or a lone CR, after which only the port's `^` matches.
    The diff: `_p2a_apart_diff`, read for the kinds of the claims main read; where the summary holds no seam, both
    ports read the same claims of those kinds, so both read the same kinds."""
    summary = f.summary
    if _p2a_apart_diff(f, kinds):
        return True
    if "styxx" in summary and (_P2A_DIV_RX.search(summary) or "\r" in summary.replace("\r\n", "\n")):
        return True
    if not _P2A_BAD_RX.search(summary):
        return False
    low = summary.translate(_P2A_TRIGGER_LOW)
    a = 0
    for m in _P2A_CUT.finditer(summary):
        e = m.start() if m.group() == "\n" else m.start() + 1
        if _p2a_apart_in(summary, low, a, e):
            return True
        a = e + 1 if m.group() == "\n" else e
    return _p2a_apart_in(summary, low, a, len(summary))


def _p2a_zones(summary: str) -> list:
    """(where its earliest 'only' ends, where it ends) for each sentence, as both ports end one, in which a character the
    two ports may read apart follows the earliest 'only' (ASCII case). A scope claim is read within one sentence of
    either port, and each port's sentence lies inside one of these."""
    low = summary.translate(_P2A_ASCII_LOWER)
    out = []
    a = 0
    for m in _P2A_CUT.finditer(summary):
        e = m.start() if m.group() == "\n" else m.start() + 1
        o = low.find("only", a, e)
        if o >= 0 and _P2A_BAD_RX.search(summary, o, e):
            out.append((o + 4, e))
        a = e + 1 if m.group() == "\n" else e
    o = low.find("only", a)
    if o >= 0 and _P2A_BAD_RX.search(summary, o):
        out.append((o + 4, len(summary)))
    return out


class _P2aFacts:
    """What the overlay reads, from the door's own bytes, computed once per diff when a claim needs it."""

    def __init__(self, diff_text: str, name_status: str | None = None, summary: str = ""):
        self.diff_text, self.name_status, self.summary, self._m = diff_text, name_status, summary, {}

    def _get(self, k, make):
        if k not in self._m:
            self._m[k] = make()
        return self._m[k]

    def fine(self) -> list:
        return self._get("fine", lambda: _p2a_lines(self.diff_text, _P2A_FINE))

    def regs(self) -> list:
        if self.name_status is not None:
            return self._get("regs", lambda: _p2a_regs_git(self.name_status))
        return self._get("regs", lambda: _p2a_regs_raw(self.fine()))

    def views(self) -> list:
        return self._get("views", lambda: _p2a_views(self.diff_text, self.fine()))

    def divergent(self) -> bool:
        return self.name_status is None and self._get("div", lambda: _p2a_divergent(self.diff_text))

    def status(self, space: str) -> dict:
        return self._get(space, lambda: _p2a_build(self.regs(), _P2A_FORMS[space][0]))

    def order(self, space: str) -> list:
        """The status map's keys, in main's order."""
        return self._get("order" + space, lambda: [p for p in self.status(space)])

    def groups(self, space: str) -> dict:
        """The status map's keys by base name, in main's order."""
        def make():
            groups: dict = {}
            for p in self.status(space):
                groups.setdefault(_p2a_base(p), []).append(p)
            return groups
        return self._get("groups" + space, make)

    def prime(self, paths: tuple) -> tuple:
        """Names the claimed paths of the claims in reach, before any claim is read (NOTE_path2a_fifth_pass_2026_09_30,
        A-1): the claims' tree (`_p2a_ends`) holds their forms. A claim not named here is read on its own."""
        return self._get("claimed", lambda: paths)

    def ends(self, space: str, which: str, c: str) -> list:
        """`_p2a_ends` for the claim form c of `space` over the status map's keys ("key", c a fold form) or over every
        registration's fold ("fold") or wild ("wild") form: [the earliest index, how many end in '/' + c]."""
        form = _P2A_FORMS[space][1 if which == "wild" else 0]
        if which == "key":
            texts = self.order(space)
        else:
            texts = self.space(space)[1 if which == "wild" else 0]
        got = self._get(("ends", space, which), lambda: _p2a_ends(
            texts, {form(x) for x in self._get("claimed", tuple)}))
        if c in got:
            return got[c]
        return self._get(("ends", space, which, c), lambda: _p2a_ends(texts, {c})[c])

    def automaton(self, want) -> tuple:
        """`_p2a_automaton` over the ASCII base names (a wild form without a placeholder) of main's registrations,
        those with status `want` when it is given."""
        def make():
            regs = self.regs()
            words = set()
            for i, w in enumerate(self.space("A")[1]):
                b = _p2a_base(w)
                if b and not _P2A_WIDE.search(b) and (want is None or regs[i][1] == want):
                    words.add(b)
            return _p2a_automaton(words)
        return self._get(("automaton", want), make)

    def joined(self) -> list:
        return self._get("joined", lambda: _p2a_joined(self.diff_text))

    def seam(self) -> bool:
        return self._get("seam", lambda: _p2a_seam(self.summary))

    def space(self, space: str) -> tuple:
        """Per registration, its fold and wild forms; per wild form, the fold forms and the statuses registered under
        it; and the registrations whose fold form holds a code point from 0x80 up, by the base name of their wild form
        and all together. Only those can compare otherwise once case outside ASCII is a placeholder."""
        def make():
            form, wform = _P2A_FORMS[space]
            regs = self.regs()
            fs, ws = [form(r[0]) for r in regs], [wform(r[0]) for r in regs]
            keys: dict = {}
            sts: dict = {}
            wide_by_base: dict = {}
            wide = []
            for i, r in enumerate(regs):
                keys.setdefault(ws[i], set()).add(fs[i])
                sts.setdefault(ws[i], set()).add(r[1])
                if _P2A_WIDE.search(fs[i]):
                    wide_by_base.setdefault(_p2a_base(ws[i]), []).append(i)
                    wide.append(i)
            return fs, ws, keys, sts, wide_by_base, wide
        return self._get("space" + space, make)

    def odd(self) -> bool:
        return self._get("odd", lambda: any(_p2a_odd(x) for sp in ("A", "K") for x in self.space(sp)[0]))

    def count(self) -> tuple:
        """(a possible twin, #WA and #A over main's registrations, #WK and #K over V121's)."""
        def make():
            regs = self.regs()
            ra = [r[0] for r in regs if r[2] == "+" or _p2a_strip(r[0])]
            rk = [r[0] for r in regs if r[2] == "+" or _p2a_dotted(r[0])]
            twin = len(rk) != len(ra)             # a dotted-only name ("." , "..") V121 would register
            runs: dict = {}
            for raw in ra:
                if runs.setdefault(_p2a_WA(raw), _p2a_run(raw)) != _p2a_run(raw):
                    twin = True
            return (twin, len({_p2a_WA(x) for x in ra}), len({_p2a_A(x) for x in ra}),
                    len({_p2a_WK(x) for x in rk}), len({_p2a_K(x) for x in rk}))
        return self._get("count", make)

    def unchanged(self) -> list:
        """The unchanged lines of each view, and those no view reads (NOTE_path2a_sixth_pass_2026_09_30, B-1)."""
        return self._get("unchanged", lambda: _p2a_context(self.diff_text, self.fine())
                         + _p2a_joined(self.diff_text, " "))

    def pairing(self) -> tuple:
        return self._get("pairing", lambda: _p2a_pairing(self.views(), self.views()[0] is self.views()[1]
                                                         and not _P2A_LEAD_APART.search(self.diff_text),
                                                         self.joined(), self.unchanged()))

    def kinds(self, found=()) -> frozenset:
        """The kinds that can be CONTRADICTED among the claims main read, named before any claim is read (C-1)."""
        return self._get("kinds", lambda: frozenset(found) & _P2A_ACCUSE)

    def apart(self, kinds: frozenset) -> bool:
        return self._get("apart", lambda: _p2a_apart(self, kinds))

    def tokens(self, kind: str, words) -> frozenset:
        """Names the words the claims in reach may look up in the summary's runs of `kind` ('path', 'name', 'count')
        or in its zones ('zone'), before any claim is read (NOTE_path2a_sixth_pass_2026_09_30, A-1): those with no
        character the two ports read apart, the only ones a claim looks up there, each text of the Basic Multilingual
        Plane outside the surrogates, which the port reads unit by unit as this reads code points."""
        return self._get(("tokens", kind), lambda: frozenset(
            w for w in words if w and _P2A_SEP not in w and _P2A_BAD_RX.search(w) is None))

    def occurs(self, kind: str, s: str) -> bool:
        """Whether s occurs in the summary's runs of `kind`, or in its zones: for a word named by `tokens`, read from
        one pass over that text for all of them; for any other, by a scan of its own."""
        text = self.zone_text() if kind == "zone" else self.runs(kind)
        words = self._get(("tokens", kind), frozenset)
        if s in words:
            return s in self._get(("found", kind), lambda: _p2a_found(words, text))
        return self._get(("in", kind, s), lambda: s in text)

    def defines(self, name: str) -> bool:
        runs, wild = self._get("defs", lambda: _p2a_def_runs(self.views(), self.joined()))
        if wild:
            return True
        if name and _P2A_WORD_RUN.fullmatch(name):
            return name in runs
        return self._get(("defines", name), lambda: _p2a_defines(self.views(), name, self.joined()))

    def redefines(self, name: str) -> bool:
        """Whether an unchanged line may define `name`, an ASCII name, after an anchored `def` or `class`."""
        return bool(name) and _P2A_WORD_RUN.fullmatch(name) is not None and \
            name in self._get("udefs", lambda: _p2a_name_defs([self.unchanged()]))

    def runs(self, kind: str) -> str:
        return self._get("runs" + kind, lambda: _p2a_runs(self.summary, _P2A_RUN_RX[kind]))

    def zones(self) -> list:
        return self._get("zones", lambda: _p2a_zones(self.summary))

    def zone_text(self) -> str:
        """The zones' text joined by _P2A_SEP: it holds s, where s does not hold _P2A_SEP, exactly when a zone does."""
        return self._get("zone_text", lambda: _P2A_SEP.join(self.summary[lo:hi] for lo, hi in self.zones()))

    def scope(self, prefixes: tuple) -> dict:
        return self._get(("scope", prefixes), lambda: _p2a_scope(self, prefixes))


def _p2a_in_runs(f: "_P2aFacts", s: str, kind: str) -> bool:
    """Whether some occurrence of s, which is ASCII, lies in the summary inside a run of path (name, number) characters
    holding a character CPython's template may read as a word character and the port's does not: where the two ports'
    templates may extract a path, a name or a number apart. An occurrence lies inside exactly one maximal run, and no
    run holds _P2A_SEP, so a string holding it lies in none."""
    if not s or _P2A_SEP in s:
        return False
    return f.occurs(kind, s)


def _p2a_port_may_verify(f: "_P2aFacts", claimed: str, want) -> bool:
    """For a claimed path holding a code point from 0x80 up: whether the port's reading of it could be VERIFIED. The
    port's template reads only ASCII, so its path is a part of this one, and a path main verifies ends in the base name
    of some registration with the claimed status. So it is enough that some such registration has an ASCII base name
    (its wild form holds no placeholder) that the claimed path, folded, holds: one read of it through the automaton of
    those base names, memoised per claimed path (NOTE_path2a_fifth_pass_2026_09_30, A-2)."""
    held = _p2a_fold(_p2a_bs(claimed))
    return f._get(("port", held, want), lambda: _p2a_holds(f.automaton(want), held))


def _p2a_extract(f: "_P2aFacts", c, claimed: str, want) -> bool:
    """Where the two ports' templates may extract a claimed path apart, and both mains could verify it. A declared
    claim's sentence is written by DECLARE-1 from a value both ports read only when it is ASCII."""
    if _P2A_WIDE.search(claimed):
        return _p2a_port_may_verify(f, claimed, want)
    return not c.detail.get("declared") and _p2a_in_runs(f, claimed, "path")


def _p2a_case_doubt(f: "_P2aFacts", space: str, claimed: str, want) -> bool:
    """U1: a path matches the claim at another tier once case outside ASCII is a placeholder. U2 (a status claim):
    a path the claim may match shares that placeholder key with another key and a status other than the one
    claimed, so a runtime's lower() could merge them into a key that reads otherwise. A tier holds only within one
    base name, and the wild form keeps every '/', so only the claim's base-name group is read; and in it, only the
    registrations whose fold form holds a code point from 0x80 up: for the others the fold form is the wild form and
    no other registration shares it, so neither U1 nor U2 can hold. Memoised per fold form (the wild form is a function
    of it) and status (NOTE_path2a_fifth_pass_2026_09_30, A-2)."""
    cf = _P2A_FORMS[space][0](claimed)
    return f._get(("case", space, cf, want), lambda: _p2a_case(f, space, claimed, want))


def _p2a_case(f: "_P2aFacts", space: str, claimed: str, want) -> bool:
    """`_p2a_case_doubt` without a scan of the group per claim (NOTE_path2a_fifth_pass_2026_09_30, A-2). In the group G
    of the claim's wild base name b, a fold match implies a wild match at the same tier, so the wild tier of a
    registration is never above its fold tier, and U1 holds exactly when (1) some fold base name in G is not the
    claim's, or (2) a registration with the claim's wild form has another fold form, or (3) more registrations end in
    '/' + the wild form than in '/' + the fold form. (1) is one set per group, (2) the per-wild-form fold sets, (3)
    two reads through the claims' tree; U2 is one flag per group and status. An empty base name keeps the scan."""
    form, wform = _P2A_FORMS[space]
    fs, ws, keys, sts, wide_by_base, wide = f.space(space)
    cw, cf = wform(claimed), form(claimed)
    b = _p2a_base(cw)
    if not b:
        for i in wide:
            tw = _p2a_tier(ws[i], cw)
            if tw != _p2a_tier(fs[i], cf):
                return True
            if want is not None and tw is not None and len(keys[ws[i]]) > 1 and sts[ws[i]] != {want}:
                return True
        return False
    group = wide_by_base.get(b, ())
    if cw != cf:                          # a claim in ASCII compares alike under both forms with every registration
        bases = f._get(("bases", space, b), lambda: {_p2a_base(fs[i]) for i in group})
        if len(bases) > 1 or (bases and _p2a_base(cf) not in bases):
            return True
        if cw in keys and keys[cw] != {cf}:
            return True
        if f.ends(space, "wild", cw)[1] != f.ends(space, "fold", cf)[1]:
            return True
    return want is not None and f._get(("merged", space, b, want), lambda: any(
        len(keys[ws[i]]) > 1 and sts[ws[i]] != {want} for i in group))


def _p2a_path(c, f: "_P2aFacts"):
    claimed = c.detail["path"]
    want = {"file_created": "A", "file_deleted": "D"}.get(c.kind)
    if f.divergent():
        return "divergent", "#97, #121"
    if _p2a_extract(f, c, claimed, want):
        return "extract", "#97, #121"
    if f.odd() or _p2a_odd(_p2a_A(claimed)) or _p2a_odd(_p2a_K(claimed)):
        return "odd", "#97"
    if _p2a_case_doubt(f, "A", claimed, want) or _p2a_case_doubt(f, "K", claimed, want):
        return "case", "#97, #121"
    ca, ck = _p2a_A(claimed), _p2a_K(claimed)

    def ok(r) -> bool:
        return r is not None and (want is None or r[1] == want)

    if not ok(_p2a_resolve(f, "A", ca, False)):
        return "unreproduced", "#97"
    r97 = _p2a_resolve(f, "A", ca, True)
    v97 = ok(r97)
    v121 = ok(_p2a_resolve(f, "K", ck, False))
    vboth = ok(_p2a_resolve(f, "K", ck, True))
    if v97 and v121 and vboth:
        return None
    if v121 and not v97:
        return ("dir" if r97 is None else "tier"), "#97"
    if v97 and not v121:
        return ("dot_earliest" if vboth else "dot"), "#121"
    return "dot_tier", "#97, #121"


def _p2a_int(digits: str):
    """The value of a string of ASCII digits, read digit by digit from a fixed table: no int(), which reads CPython's
    table of decimal digits (NOTE_path2a_fourth_pass_2026_09_30, C-3). Any other character raises, and the overlay's
    error fallback withholds."""
    n = 0
    for ch in digits:
        n = n * 10 + _P2A_DIGIT[ch]
    return n


def _p2a_numbers(head, why: str, detail: dict):
    """(main's own count, the claimed number), or None. The count is read from main's reason; the number from the
    claim's detail where that is ASCII digits (the port's main prints a number of 10**21 or more in exponent form),
    else from main's reason."""
    m = head.match(why)
    if m is None:
        return None
    k = _P2A_DIGITS.fullmatch(detail.get("n") or "")
    if k is None:
        k = _P2A_DIGITS.fullmatch(why, m.end())
    if k is None:
        return None
    return _p2a_int(m.group(1)), _p2a_int(k.group(0))


def _p2a_count(c, f: "_P2aFacts"):
    if f.divergent():
        return "divergent", "#121"
    twin, wa, a, lo, hi = f.count()
    if not twin:
        return None
    # C-1 (NOTE_path2a_fifth_pass_2026_09_30): a count one port reads and the other does not, from a white space only
    # one of them reads or a letter only CPython folds, can be read cleanly by that port alone; so where the summary
    # holds such a character anywhere, every count read from the summary is withheld, in both ports, ahead of the
    # guard below, so that both ports name the same phrase whichever count each reads.
    if not c.detail.get("declared") and f.seam():
        return "seam", "#121"
    # C-1 (NOTE_path2a_fourth_pass_2026_09_30): the count template reads its number after `\b`, so where a digit or
    # letter outside ASCII sits beside it CPython's `\d` and `\b` read one number and the port's another (a full-width 3
    # then "3": 33 in CPython, 3 in the port). Withheld in both ports, as for a path or a name.
    claimed = c.detail.get("n") or ""
    if _P2A_DIGITS.fullmatch(claimed) is None or (not c.detail.get("declared") and _p2a_in_runs(f, claimed, "count")):
        return "extract", "#121"
    nums = _p2a_numbers(_P2A_COUNT_HEAD, c.why, c.detail)
    if nums is None:
        return "unparsed", "#121"
    g, n = nums
    if not wa <= g <= a:
        return "unreproduced", "#121"
    if c.verdict == "VERIFIED":
        return None if lo == hi == n else ("count", "#121")
    return ("count", "#121") if lo <= n <= hi else None


def _p2a_ext(token: str) -> bool:         # _has_real_extension, ASCII case
    return "." in token and _p2a_fold(token.rsplit(".", 1)[-1]) in PATH1_EXTENSIONS


def _p2a_inside(p: str, pref: str) -> bool:   # _path_inside
    if "/" not in pref and _p2a_ext(pref):
        return p == pref or p.endswith("/" + pref)
    return p == pref or p.startswith(pref + "/")


def _p2a_shaped(prefix: str, keys: list, form) -> bool:   # _prefix_is_path_shaped
    raw = prefix.strip("`\"'").rstrip(".")
    if not raw:
        return False
    if "/" in raw or "\\" in raw:
        return True
    if "." in raw and _p2a_ext(raw.rstrip("/")):
        return True
    low = form(raw).rstrip("/")
    return any(low in k.split("/") for k in keys)


_P2A_SPACES = {"A": (_p2a_A, _p2a_WA, _p2a_strip), "K": (_p2a_K, _p2a_WK, _p2a_dotted)}


def _p2a_scope(f: "_P2aFacts", prefixes: tuple) -> dict:
    """{(space, wild): (whether `prefix` reads as a path, whether prefix2 does, whether every changed path lies
    under the prefixes that reading uses)}, in main's key space A and V121's K, with case folded (wild False) and
    outside ASCII a placeholder (True)."""
    two = len(prefixes) == 2
    regs = f.regs()
    out = {}
    for sp, (form, wform, pre_key) in _P2A_SPACES.items():
        fs, ws = f.space(sp)[:2]
        keep = [i for i, r in enumerate(regs) if r[2] == "+" or pre_key(r[0])]
        for wild, fm, forms in ((False, form, fs), (True, wform, ws)):
            paths = [forms[i] for i in keep]
            lead = _p2a_shaped(prefixes[0], paths, fm)
            shaped = two and _p2a_shaped(prefixes[1], paths, fm)
            ps = [fm(x).rstrip("/.") for x in (prefixes if shaped else prefixes[:1])]
            out[sp, wild] = (lead, shaped, all(any(_p2a_inside(p, x) for x in ps) for p in paths))
    return out


def _p2a_scope_doubt(f: "_P2aFacts", d: dict) -> bool:
    """Whether the two ports' templates may read this scope claim's prefixes apart: a prefix holds a character they
    read apart, or an occurrence of the prefix after the earliest 'only' of a sentence (as both ports end one) shares that
    sentence with such a character after that 'only'. A declared claim's sentence is written by DECLARE-1 from a value
    both ports read only when it is ASCII."""
    prefix = d["prefix"]
    if _P2A_BAD_RX.search(prefix) or _P2A_BAD_RX.search(d.get("prefix2") or ""):
        return True
    if d.get("declared"):
        return False
    if _P2A_SEP in prefix:
        return any(f.summary.find(prefix, lo, hi) >= 0 for lo, hi in f.zones())
    return f.occurs("zone", prefix)


def _p2a_only(c, f: "_P2aFacts"):
    """Keep the claim when V121 (keys with their leading dots) reads the claim as main does: `prefix` as a path
    where main does, when a second prefix is claimed; and every changed path under the prefix set it uses exactly
    as main does under the set main uses."""
    if f.divergent():
        return "divergent", "#121"
    d = c.detail
    if _p2a_scope_doubt(f, d):
        return "extract", "#121"
    got = f.scope((d["prefix"],) + ((d["prefix2"],) if d.get("prefix2") else ()))
    if got["A", False] != got["A", True] or got["K", False] != got["K", True]:
        return "case", "#121"
    lead, _shaped, under = got["A", False]
    if not lead or ("VERIFIED" if under else "CONTRADICTED") != c.verdict:
        return "unreproduced", "#121"
    # B-1 (NOTE_path2a_fourth_pass_2026_09_30): V121 does not read a `prefix` that only the dropped dots make a
    # path ("github" beside `.github/`), and says "is not a path (#110)". With a second prefix claimed, main's verdict
    # there can be false while the two under-readings agree; with one prefix, a false verdict there needs two
    # independent rendering faults besides #121, and the recall it would cost is measured in the NOTE.
    if d.get("prefix2") and not got["K", False][0]:
        return "shape", "#121"
    if got["K", False][2] != under:
        return "only", "#121"
    return None


def _p2a_tests(c, f: "_P2aFacts"):
    """PREREG R-101's interval [got - changed, got], read for each port's main whose count gives this claim the
    verdict it has here (so both ports decide alike wherever their mains agree): 'tests' when every such reading
    holds the claim, 'split' when only one does."""
    counts = f.pairing()
    if not any(b for _g, _p, b in counts):
        return None
    nums = _p2a_numbers(_P2A_TESTS_HEAD, c.why, c.detail)
    if nums is None:
        return "unparsed", "#101"
    got, n = nums
    if counts[_P2A_OWN][0] != got:
        return "unreproduced", "#101"
    fires = [bool(min(g, p)) and g - min(g, p) <= n <= g for g, p, _b in counts
             if (g == n) == (c.verdict == "VERIFIED")]
    if fires and all(fires):
        return "tests", "#101"
    if any(fires):
        return "split", "#101"
    # B-1 (NOTE_path2a_sixth_pass_2026_09_30): the same interval with the tests the unchanged lines define paired too
    fires = [bool(min(g, b)) and g - min(g, b) <= n <= g for g, _p, b in counts
             if (g == n) == (c.verdict == "VERIFIED")]
    return ("redefined", "#101") if any(fires) else None


def _p2a_symbol(c, f: "_P2aFacts"):
    name = c.detail.get("name")
    if _P2A_WIDE.search(name) or (not c.detail.get("declared") and _p2a_in_runs(f, name, "name")):
        return "extract", "#101"
    if f.defines(name):
        return "symbol", "#101"
    if f.redefines(name):                 # B-1 (NOTE_path2a_sixth_pass_2026_09_30)
        return "again", "#101"
    return None


def _p2a_decide(c, f: "_P2aFacts"):
    # C-1 (NOTE_path2a_sixth_pass_2026_09_30): withholding a CONTRADICTED can move a gate verdict without --strict, and
    # where the two ports' mains may read apart which claims are CONTRADICTED, it could move one port's and not the
    # other's. There main's CONTRADICTED stands, in both ports.
    if c.verdict == "CONTRADICTED" and f.apart(f.kinds()):
        return None
    if c.kind in _PATH_KINDS:
        return _p2a_path(c, f)
    if c.kind == "files_changed_count":
        return _p2a_count(c, f)
    if c.kind == "only_touches":
        return _p2a_only(c, f)
    if c.kind == "tests_added":
        return _p2a_tests(c, f)
    return _p2a_symbol(c, f)


def _p2a_reason(verdict: str, defect: str, key: str, why: str) -> str:
    return f"{verdict} withheld by PATH-2a ({defect}): {_P2A_PHRASES[key]}. main's reading: {why}"


def _p2a_abstain(g: DiffGate, strict: bool, facts) -> DiffGate:
    """Turn a decided verdict UNCHECKABLE where #97, #121 or #101 can have made it wrong, then recompute the gate
    verdict with main's own formula. Nothing else in the record moves."""
    todo = [c for c in g.claims if (c.kind, c.verdict) in _P2A_REACH]
    if not todo:
        return g
    try:
        f = facts()
        f.prime(tuple(c.detail.get("path") for c in todo if c.kind in _PATH_KINDS
                      and isinstance(c.detail.get("path"), str)))
        for kind, field in (("path", "path"), ("name", "name"), ("count", "n"), ("zone", "prefix")):
            f.tokens(kind, [c.detail.get(field) for c in todo if isinstance(c.detail.get(field), str)])
        f.kinds([c.kind for c in g.claims])
        hits = [(c, _p2a_decide(c, f)) for c in todo]
    except Exception:                     # an abstain-only overlay that cannot read withholds, and says so
        hits = [(c, ("error", _P2A_KIND_DEFECT[c.kind])) for c in todo]
    moved = False
    for c, hit in hits:
        if hit is not None:
            c.why = _p2a_reason(c.verdict, hit[1], hit[0], c.why)
            c.verdict = "UNCHECKABLE"
            moved = True
    # A-2 (NOTE_path2a_sixth_pass_2026_09_30): where no claim moved, the record is main's object, untouched
    if moved:
        contradicted = any(c.verdict == "CONTRADICTED" for c in g.claims)
        uncheckable = any(c.verdict == "UNCHECKABLE" for c in g.claims)
        g.verdict = "FAIL" if (contradicted or (strict and uncheckable)) else "PASS"
    return g


_P2A_MARK = "# === PATH-2a abstain-only overlay: "          # + "BEGIN ===" / "END ===", never written whole here
_P2A_BANNED_NAMES = frozenset({"Path", "os", "unicodedata", "pathlib", "locale", "setattr", "delattr", "vars",
                               "repr", "ascii", "format", "getattr", "eval", "exec", "compile", "globals", "locals",
                               "__import__", "float", "complex", "builtins", "operator", "sys", "type", "map",
                               "int"})
_P2A_BANNED_ATTRS = frozenset({"lower", "upper", "casefold", "swapcase", "title", "capitalize", "splitlines",
                               "normalize", "I", "IGNORECASE", "U", "UNICODE", "L", "LOCALE", "__dict__", "format",
                               "format_map", "encode", "decode"})
_P2A_ARGLESS = frozenset({"strip", "lstrip", "rstrip", "split", "rsplit"})
_P2A_MUTATORS = frozenset({"append", "extend", "insert", "pop", "popitem", "remove", "clear", "update",
                           "setdefault", "sort", "reverse", "__setitem__", "__setattr__", "__delitem__"})
_P2A_RECORD = frozenset({"verdict", "why", "kind", "text", "detail", "claims"})
_P2A_RE_CALLS = frozenset({"compile", "match", "fullmatch", "search", "findall", "finditer", "split", "sub", "subn"})
_P2A_CHECKERS = frozenset({"selfcheck_p2a_only_abstains", "_p2a_regex_problems"})   # not run on a claim
# Besides the names the block binds, a block function may read only these: builtins that read no Unicode table, and
# main's names the block uses. A module, `getattr`, `operator`, `builtins` or `unicodedata` is refused by name.
_P2A_NAMES_OK = frozenset({"len", "set", "list", "dict", "tuple", "frozenset", "any", "all", "min", "max",
                           "enumerate", "zip", "range", "bool", "isinstance", "str", "Exception",
                           "re", "_Pending", "PATH1_EXTENSIONS", "_PATH_KINDS", "DiffGate"})
# ... and read only these attributes: the str, list, dict and set methods the block calls (none reads a Unicode
# table: every strip and split carries its characters), the regex methods, and the fields of the record, of main's
# _Pending and of the facts object. A dunder, `__getattribute__` or any method not listed is refused.
_P2A_ATTRS_OK = frozenset({
    "startswith", "endswith", "find", "rfind", "replace", "translate", "strip", "lstrip", "rstrip", "split", "rsplit",
    "add", "append", "pop", "get", "items", "setdefault",
    "compile", "match", "fullmatch", "search", "finditer", "sub", "group", "start", "end", "join",
    "verdict", "why", "kind", "detail", "claims", "a", "b", "status", "note",
    "diff_text", "name_status", "summary", "_m", "_get", "fine", "regs", "views", "divergent", "space", "odd",
    "count", "pairing", "defines", "runs", "zones", "zone_text", "scope",
    "order", "groups", "prime", "ends", "automaton", "joined", "seam", "unchanged", "apart", "tokens", "occurs",
    "kinds", "redefines"})
_P2A_TYPES = frozenset({"str", "list", "dict", "set", "tuple", "frozenset", "int", "bool"})


def _p2a_regex_problems(pattern: str) -> list:
    """Class escapes, a named-character escape, an unescaped '.', and inline case or Unicode flags in one static
    pattern."""
    out = []
    i, in_class = 0, False
    while i < len(pattern):
        ch = pattern[i]
        if ch == "\\":
            if pattern[i + 1:i + 2] in ("w", "W", "s", "S", "b", "B", "d", "D"):
                out.append("class escape " + pattern[i:i + 2])
            elif pattern[i + 1:i + 2] == "N":
                # C-2 (NOTE_path2a_sixth_pass_2026_09_30): \N{NAME} reads the Unicode name table when it compiles
                out.append("named-character escape " + pattern[i:i + 2])
            i += 2
            continue
        if ch == "[" and not in_class:
            in_class = True
        elif ch == "]" and in_class:
            in_class = False
        elif ch == "." and not in_class:
            out.append("unescaped '.'")
        elif ch == "(" and not in_class and pattern[i + 1:i + 2] == "?":
            # NOTE_path2a_fifth_pass_2026_09_30, C-2: a flag group (`(?si)`, `(?mi:...)`, `(?-i:...)`) with a case or
            # Unicode flag in any position, not only the leading one
            j = i + 2
            while j < len(pattern) and pattern[j] in "aiLmsux-":
                j += 1
            if j > i + 2 and pattern[j:j + 1] in (":", ")") and any(x in pattern[i + 2:j] for x in "iuLa"):
                out.append("inline flag " + pattern[i:j + 1])
        i += 1
    return out


def selfcheck_p2a_only_abstains(source: str | None = None) -> dict:
    """Re-derive from the PATH-2a block's own source, with `ast`, that the overlay can only abstain and reads no
    Unicode table at run time.

    Checked: every attribute store is `c.verdict = "UNCHECKABLE"`, `c.why = ...` or `g.verdict = FAIL/PASS` inside
    `_p2a_abstain`, or `self.*` inside `_P2aFacts.__init__`; no subscript store into an attribute except `self._m`;
    no mutating call on a record field; no augmented, annotated or deleted attribute; no import; every name read is one
    the block binds or on a short list (builtins that read no table, main's names the block uses), and every attribute
    read is on a short list of methods and fields, so a module, a dunder or an unlisted method is refused; `re` is only
    ever called as `re.<function>(<static pattern>)`; no attribute read off a builtin type (`str.split`); no case,
    Unicode-table, path or locale call, and no call that reads one indirectly (repr, format, !r, %r, getattr, eval,
    encode, int(), whose digits are CPython's table); every strip and split carries its characters (none, `None` or a
    keyword is refused); every regex static, with no class escape, no unescaped '.', and no case or Unicode flag. The
    self-check's own two functions are exempt from the name, attribute and call rules, not from the store rules.
    `source` is the module text (default: this file)."""
    if source is None:
        with open(__file__, encoding="utf-8") as fh:
            source = fh.read()
    begin, end = _P2A_MARK + "BEGIN ===", _P2A_MARK + "END ==="
    if source.count(begin) != 1 or source.count(end) != 1:
        return {"ok": False, "problems": ["the block's markers are not each present exactly once"]}
    block = source[source.index(begin):source.index(end)]
    tree = ast.parse(block)
    parent: dict = {}
    for node in ast.walk(tree):
        for ch in ast.iter_child_nodes(node):
            parent[ch] = node

    def owner(node):
        names = []
        while node in parent:
            node = parent[node]
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.append(node.name)
        return names

    consts: dict = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            consts[node.targets[0].id] = node.value

    def static(expr):
        if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
            return expr.value
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
            a, b = static(expr.left), static(expr.right)
            return None if a is None or b is None else a + b
        if isinstance(expr, ast.Name) and expr.id in consts:
            return static(consts[expr.id])
        return None

    bound: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            bound.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)

    problems = []
    judged: set = set()                       # Attribute / Subscript store targets an Assign accounted for
    aliases = set()                           # local names bound to a record field (d = c.detail)

    def bad(node, what):
        problems.append(f"line {node.lineno}: {what}")

    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Attribute) and node.value.attr in _P2A_RECORD):
            aliases.add(node.targets[0].id)
    for node in ast.walk(tree):
        where = owner(node)
        checker = bool(_P2A_CHECKERS & set(where))
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            bad(node, "an import")
        if isinstance(node, (ast.AugAssign, ast.AnnAssign)) and not isinstance(node.target, ast.Name):
            bad(node, "augmented or annotated store into an attribute or item")
            judged.add(id(node.target))
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                for t in (tgt.elts if isinstance(tgt, ast.Tuple) else [tgt]):
                    judged.add(id(t))
                    if isinstance(t, ast.Attribute):
                        base = t.value.id if isinstance(t.value, ast.Name) else None
                        if base == "self" and where[:2] == ["__init__", "_P2aFacts"]:
                            continue
                        if where[:1] != ["_p2a_abstain"]:
                            bad(t, f"store into .{t.attr} outside _p2a_abstain")
                        elif t.attr == "why":
                            continue
                        elif t.attr == "verdict" and isinstance(node.value, ast.Constant):
                            if node.value.value != "UNCHECKABLE":
                                bad(t, f"verdict literal {node.value.value!r}")
                        elif (t.attr == "verdict" and base == "g" and isinstance(node.value, ast.IfExp)
                              and {getattr(node.value.body, "value", None),
                                   getattr(node.value.orelse, "value", None)} == {"FAIL", "PASS"}):
                            continue
                        else:
                            bad(t, f"store into .{t.attr}")
                    elif isinstance(t, ast.Subscript) and isinstance(t.value, ast.Attribute):
                        v = t.value
                        if not (isinstance(v.value, ast.Name) and v.value.id == "self" and v.attr == "_m"):
                            bad(t, f"item store into .{v.attr}")
                    elif isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and t.value.id in aliases:
                        bad(t, f"item store into {t.value.id}, a record field")
        if isinstance(node, (ast.Attribute, ast.Subscript)) and not isinstance(node.ctx, ast.Load) \
                and id(node) not in judged:
            bad(node, "attribute or item stored or deleted outside a plain assignment")
        if checker:
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id in _P2A_BANNED_NAMES:
                bad(node, f"name {node.id}")
            elif node.id not in bound and node.id not in _P2A_NAMES_OK:
                bad(node, f"name {node.id}, not one the block binds or may read")
            if node.id == "re":
                up = parent.get(node)
                if not (isinstance(up, ast.Attribute) and up.attr in _P2A_RE_CALLS
                        and isinstance(parent.get(up), ast.Call) and parent[up].func is up):
                    bad(node, "re used other than as re.<function>(...)")
        if isinstance(node, ast.Attribute):
            if node.attr in _P2A_BANNED_ATTRS or node.attr.startswith("is"):
                bad(node, f"attribute .{node.attr}")
            elif node.attr not in _P2A_ATTRS_OK:
                bad(node, f"attribute .{node.attr}, not one the block may read")
            if isinstance(node.value, ast.Name) and node.value.id in _P2A_TYPES:
                bad(node, f"attribute read off the type {node.value.id}")
        if isinstance(node, ast.FormattedValue) and node.conversion in (ord("r"), ord("a")):
            bad(node, "an f-string !r or !a conversion")
        if (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod) and isinstance(node.left, ast.Constant)
                and isinstance(node.left.value, str) and ("%r" in node.left.value or "%a" in node.left.value)):
            bad(node, "%r or %a formatting")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "str":
            bad(node, "str() of a value, which can read repr")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            f = node.func
            if f.attr in _P2A_ARGLESS and (not node.args or node.keywords or any(
                    isinstance(a, ast.Constant) and a.value is None for a in node.args)):
                bad(node, f".{f.attr}() without an explicit argument")
            if f.attr in _P2A_MUTATORS and isinstance(f.value, ast.Attribute) and f.value.attr in _P2A_RECORD:
                bad(node, f".{f.value.attr}.{f.attr}()")
            if f.attr in _P2A_MUTATORS and isinstance(f.value, ast.Name) and f.value.id in aliases:
                bad(node, f"{f.value.id}.{f.attr}(), a record field")
            if isinstance(f.value, ast.Name) and f.value.id == "re" and f.attr in _P2A_RE_CALLS:
                pat = static(node.args[0]) if node.args else None
                if pat is None:
                    bad(node, f"re.{f.attr} with a pattern that is not static")
                else:
                    problems.extend(f"line {node.lineno}: {p}" for p in _p2a_regex_problems(pat))
                if len(node.args) > 1 or node.keywords:
                    bad(node, f"re.{f.attr} with flags or extra arguments")
    if not all(ord(ch) < 128 for ch in block):
        problems.append("the block holds a character outside ASCII")
    return {"ok": not problems, "problems": problems,
            "checked": ["attribute and item stores", "record-field mutators", "imports", "names read",
                        "attributes read", "re used only as re.<function>", "banned names and attributes",
                        "indirect table reads", "argument-less strip and split", "static regexes", "ASCII source"]}

# === PATH-2a abstain-only overlay: END ===


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
