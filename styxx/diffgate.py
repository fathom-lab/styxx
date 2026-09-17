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
    # AMENDMENT_path2 C-2: read on the undotted key, as before #121, so a file named `.py` is not Python.
    return any(_undotted(p).lower().endswith(_PY_SUFFIXES) for p in status)


def _prefix_is_path_shaped(prefix: str, status: dict) -> bool:
    """A scope prefix is a path when it looks like one or names a segment of a changed path.

    Judged on the prefix as written, minus a sentence-final period: "docs/" is a path
    because of its slash, "package.json" because of its dot, "src" because a changed path
    has that segment; "the", "files" and "markdown" are words.
    """
    raw = prefix.strip("`\"'").rstrip(".")
    if not raw:
        return False
    if any(ch in raw for ch in "/\\."):
        return True
    low = _norm(raw).rstrip("/").lower()
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
# a BOM strip and a non-ASCII name the same way. The added-blob counts (`got`, `hit`) are unchanged.
_DEF_TEST_LINE = re.compile(r"^\uFEFF?[ \t]*def (test_[^ \t(:]*)")


def _symbol_def_line(name: str) -> re.Pattern:
    return re.compile(r"^\uFEFF?[ \t]*(?:async[ \t]+)?(?:def|class)[ \t]+" + re.escape(name) + r"(?=[ \t(:]|$)")


def _changed_test_defs(sides: dict | None, status: dict | None = None) -> int:
    """Test definitions paired one to one: per file whose status is not `A`, per test name,
    min(added lines defining it, removed lines defining it), summed. The caller clamps to `got`."""
    n = 0
    for path, (added, removed) in (sides or {}).items():
        if (status or {}).get(path) == "A":
            continue
        gone: dict = {}
        for line in removed:
            m = _DEF_TEST_LINE.match(line)
            if m:
                gone[m.group(1)] = gone.get(m.group(1), 0) + 1
        if not gone:
            continue
        new: dict = {}
        for line in added:
            m = _DEF_TEST_LINE.match(line)
            if m:
                new[m.group(1)] = new.get(m.group(1), 0) + 1
        n += sum(min(k, gone.get(name, 0)) for name, k in new.items())
    return n


def _definition_only_changed(name: str, sides: dict | None, status: dict | None = None) -> bool:
    """Some file both adds and removes a definition of `name`, and no file adds more definitions of
    it than it removes (a file whose status is `A` removes none). Counted per file, one to one."""
    rx = _symbol_def_line(name)
    paired = False
    for path, (added, removed) in (sides or {}).items():
        a = sum(1 for line in added if rx.match(line))
        r = 0 if (status or {}).get(path) == "A" else sum(1 for line in removed if rx.match(line))
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


def _dot_miss(path: str, prefs: list) -> bool:
    """AMENDMENT_path2 C-3: `path` lies outside every prefix only by a dot the prose left off --
    some prefix key has no leading dot, the path's leading segment starts with exactly one dot
    (not `..`), and the path without that dot is the prefix or lies under it."""
    if not path.startswith(".") or path.startswith(".."):
        return False
    rest = path[1:]
    return any(not x.startswith(".") and (rest == x or rest.startswith(x + "/")) for x in prefs)


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
    return _gate(summary_text, status, added_blob, run=run, strict=strict,
                 repo=repo, base="(diff-text)", head="(diff-text)",
                 evidence=evidence, commit=commit,
                 raw_input_len=len(diff_text or ""),
                 sides=parse_unified_diff_sides(diff_text or ""))


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
    return _gate(summary_text, status, added_blob, run=run, strict=strict,
                 repo=repo, base=base, head=head,
                 evidence=evidence, commit=commit,
                 sides=parse_unified_diff_sides(diff_text))


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
          raw_input_len: int | None = None, sides: dict | None = None) -> DiffGate:

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
                        # PATH-2 (#101): `chg` added `def test_` lines re-define a test the same
                        # file's removed lines define, paired one to one (AMENDMENT C-1). The true
                        # number added lies in [net, got]: verify `net`, abstain inside the
                        # interval, accuse only outside it.
                        chg = min(_changed_test_defs(sides, status), got)
                        net = got - chg
                        note = f" ({chg} changed, not added: #101)" if chg else ""
                        if net == n:
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
                        pat = (r"^\s*(?:def|class)\s+" + re.escape(d["name"]) + r"\b")
                        hit = bool(re.search(pat, added_blob, re.M))
                        if hit and _definition_only_changed(d["name"], sides, status):
                            c.verdict = "UNCHECKABLE"               # PATH-2 (#101)
                            c.why = (f"added lines define {d['kind']} {d['name']!r} only where the "
                                     "removed lines of the same file define it too; a changed "
                                     "definition is not an added one (#101)")
                        else:
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
                    outside = [p for p in status
                               if not any(p.startswith(x + "/") or p == x for x in prefs)]
                    # PATH-2 (#121), AMENDMENT C-3: an outside path is a DOT MISS when a prefix
                    # written without a leading dot holds it once its own single leading dot is
                    # dropped ("Only touches github/" over `.github/...`); every other outside path
                    # is REAL. Dot misses alone abstain; any real path accuses, and only real paths
                    # are listed. A dotted prefix over an undotted path, and a `..` path, are real.
                    dot_miss = [p for p in outside if _dot_miss(p, prefs)]
                    real = [p for p in outside if p not in dot_miss]
                    if no_paths:
                        c.verdict, c.why = "UNCHECKABLE", no_paths
                    elif not_paths:                             # BC-1 repair 4
                        c.verdict = "UNCHECKABLE"
                        c.why = f"prefix {not_paths[0]!r} is not a path (#110)"
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
