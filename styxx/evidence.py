# -*- coding: utf-8 -*-
"""styxx.evidence — read a test report as BYTES and refuse, loudly, in specific terms.

Spec: ``styxx-evidence/v0.2``.

THE VOCABULARY IS TWO WORDS, AND THAT IS THE POINT
--------------------------------------------------

::

    VERDICTS = ("VERIFIED", "UNCHECKABLE")

There is no accusing verdict in this module. Not disabled, not behind a flag,
not "pending measurement" — **absent**. The branch that once returned
CONTRADICTED has been deleted along with the constant, the non-zero exit and the
helpers that existed only to reach it. Grep this file for the word and every
survivor is prose explaining why there is no such branch.

Why deletion rather than a flag:

  * **The accusation's precision is structurally unmeasurable here.** The bound
    branch — the only one that was ever permitted to accuse — fires **11 times
    across 1,775,765 changed files** in this lab's corpus (303 attestation-shaped
    paths, 0.017%). A 100-item blind panel cannot be assembled from eleven
    events. The number is not merely unmeasured; there is no sample to measure.
    The standing commitment out of
    ``RESULT_v14_naming_the_defects_did_not_save_it_2026_09_01.md`` is DO NOT
    SHIP AN ACCUSING VERDICT WHOSE PRECISION HAS NOT BEEN MEASURED BY A BLIND
    PANEL. An unmeasurable one is not an exception to that rule; it is the
    hardest case of it.
  * **The gate that was supposed to license the accusation is a string compare
    on attacker-writable bytes.** See "WHAT `binding` IS NOT" below. There is no
    honest accusing verdict available here at any price this module can
    currently pay.
  * **A flag is what a maintainer who did not read the paper flips.**
    ``WITHHOLD_PATH_ACCUSATION`` already exists in this codebase as exactly such
    a flag. The absence of a branch is a stronger guarantee than a flag set to
    off, because absence cannot be toggled by someone in a hurry.

Design this module as though report-only is **PERMANENT**. It is written that
way deliberately. Nothing below says "pending", because a future maintainer
reads "pending" as "nearly ready".

**The asymmetry between the two surviving answers is deliberate and is stated
here rather than left to be inferred.** A forged VERIFIED merely repeats a claim
the author already made in prose — the cost of believing it is that you believe
something the author already told you. A forged accusation is an attack on
someone else's pull request. The two errors are not the same size, so they do
not get the same treatment.

**VERIFIED MUST NEVER BE READ AS "THE TESTS PASSED."** It means exactly: *an
attestation or report naming this commit reports PASSED, at least one test
executed and passed, every supplied file parsed, and no signature was checked.*

THE THREE LAWS
--------------

1. **PURE FUNCTION OF BYTES.** No subprocess, no socket, no clock, no
   ``os.environ``, no randomness, anywhere in this file. Same input bytes, same
   verdict, forever. ``sys.argv`` is touched in ``main()`` only, above the byte
   boundary; nothing below the CLI reads ambient state. ``selfcheck_purity()``
   re-derives this from the module's own source with ``ast`` and is printed on
   every CLI run, because a law nobody re-checks is a comment.

2. **THERE IS NO ACCUSING VERDICT.** Evidence of failure is read, counted and
   returned in a REPORT-ONLY band (``observed``). It is never a verdict, nothing
   downstream may gate on it, and ``main()`` never exits non-zero because of it.
   This replaces the older Law 2 ("binding gates the accusation"), which was a
   rule about how to accuse safely. The rule now is that this module does not
   accuse.

3. **ABSENCE IS NOT FAILURE.** Zero tests, no file, unparsable file, a harness
   that could not load a module, a shard whose report never arrived — all
   UNCHECKABLE. UNCHECKABLE is a first-class, respectable verdict here and is
   printed loudly, never hidden.

WHAT ``binding`` IS NOT
-----------------------

It is **not cryptographic** and this module no longer uses the word "bound" as
though it were. What the code computes is a **DIGEST ASSERTION CARRIED IN THE
BYTES**: some in-toto subject in a file we were handed contains a hex string
equal to the hex string the caller passed on the command line. That is a string
compare over bytes that whoever wrote the file chose.

Concretely, and each of these was a live repro against the previous version:

  * A DSSE envelope whose ``signatures`` array is empty asserts the digest just
    as well as a real one.
  * So does an envelope whose single signature is the base64 of the ASCII text
    ``not-a-signature``.
  * Fabricating a digest assertion against any commit in the world costs one
    hand-written JSON file and no key material.

**Real DSSE verification is not done here.** No key material is consulted, no
trust root is pinned, no Rekor, no Fulcio. What that costs us is precisely the
accusing verdict: a digest assertion cannot carry the weight of an accusation
against a stranger's pull request, which is one of the two reasons the accusing
branch is gone rather than gated. It costs the affirming verdict less, because
VERIFIED restates the author's own claim rather than contradicting anyone.

Accordingly the concept is split into two independent booleans, per source,
because one flag cannot express "report identity asserted, commit assertion
absent":

``report_identity_asserted``
    a subject digest equals the sha256 of a report we actually loaded. This
    says "this is the report that was attested" and is COMPLETELY SILENT about
    which commit the tests ran against.

``commit_assertion_matches``
    some subject's ``gitCommit`` digest equals the commit the caller named. Not
    "verified" — *matches*. Nothing was verified.

The frozen contract names these ``report_identity_verified`` and
``commit_binding_verified``; both spellings are present in every binding dict so
that a contract-conformant reader and an honest reader see the same value. The
``*_verified`` spellings are aliases retained for the contract's sake and the
word "verified" in them is inaccurate — prefer the other two.

Binding is recorded **PER SOURCE**. There is no single aggregate binding dict
spanning a list of files, because one attested file's digest assertion must
never appear to cover an unattested sibling globbed out of the same directory.

ENTITY DEFENCE IS OURS, NOT INHERITED
-------------------------------------

XML is parsed through an ``expat`` parser configured to **refuse entity
declarations, unparsed-entity declarations, notation declarations and external
entity references outright**, at the declaration. A billion-laughs document is
rejected because *this module* rejects it, not because libexpat's amplification
limit happened to catch it. The previous substring sweep for ``<!doctype`` in
the first 8192 characters did the opposite of what it promised in both
directions: it poisoned ordinary green reports whose ``<system-out>`` captured
HTML, and it walked an entity bomb straight past the window behind a 9 KB
comment. If the handlers cannot be installed on this interpreter, a document is
refused into ``unparsed`` rather than parsed with a promise this file cannot
keep.

WHAT THIS MODULE IS
-------------------

A defensive READER. Three separate facts establish that a JUnit XML file cannot
on its own sustain a claim about "tests pass": an empty run and an all-disabled
run are shape-identical to green; jest-junit under its default config DELETES a
suite that failed to compile, leaving a fully green report; and NO JUnit dialect
carries a commit, so nothing in the file ties it to the code it supposedly
tested. The one channel that can carry a commit at all is an in-toto test-result
attestation whose subject names a ``gitCommit`` digest — and even there we read
unauthenticated bytes and say so on every line of output.

CLI::

    python -m styxx.evidence junit.xml attestation.intoto.jsonl
    python -m styxx.evidence junit.xml --commit <40-hex> --json out.json

The CLI exit code is 0 for every possible content of the evidence. Non-zero is
reserved for usage errors (argparse's own exit 2, and an unwritable ``--json``
path). An exit code that moved with the evidence would be an accusing verdict
wearing a different hat.
"""
from __future__ import annotations

import argparse
import ast
import base64
import binascii
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
import xml.parsers.expat as _expat
from pathlib import Path

__all__ = [
    "SPEC",
    "VERDICTS",
    "INTOTO_TEST_RESULT_PREDICATE",
    "EntityDeclarationRefused",
    "load_evidence",
    "binding_against_commit",
    "adjudicate_tests_pass",
    "selfcheck_purity",
    "selfcheck_no_accusation",
    "main",
]

SPEC = "styxx-evidence/v0.2"

# Matched as an EXACT literal and kept in one place. Predicates are versioned
# independently of the in-toto spec, so a future v1 of this predicate is a
# DIFFERENT string and must stop matching rather than be assumed compatible.
INTOTO_TEST_RESULT_PREDICATE = "https://in-toto.io/attestation/test-result/v0.1"
INTOTO_STATEMENT_TYPE = "https://in-toto.io/Statement/v1"

# Two words. See the module docstring for why there is no third. Anything that
# consumes this tuple gets the whole vocabulary; there is no hidden member and
# no flag that adds one.
VERDICTS = ("VERIFIED", "UNCHECKABLE")

# ── Law 1, checked at import ──────────────────────────────────────────────────
# This catches the cheapest way the purity law dies: somebody adds `import os`
# at the top of the file during a hurried fix. It does NOT prove purity — a
# function-local import would slip past it, which is why selfcheck_purity()
# below re-derives the same fact from the source with `ast` and the CLI prints
# the answer on every run.
_FORBIDDEN_MODULES = (
    "subprocess", "socket", "time", "random", "os", "urllib", "requests",
    "datetime", "http", "ssl", "shutil", "secrets", "platform", "getpass",
)
# Impurities no banned-IMPORT list can catch, because the module they live on is
# one this file legitimately needs. `pathlib` reads the evidence, so `Path`
# cannot be forbidden — but `Path.cwd()` and `Path.home()` are ambient state
# wearing a permitted name, and a record carrying either is a function of where
# the process happened to be started. G-E1 names `Path.cwd` explicitly and
# nothing here watched for it. Matched as dotted EXPRESSIONS below, never as
# substrings, so this tuple does not accuse itself.
_FORBIDDEN_EXPRESSIONS = ("Path.cwd", "Path.home", "os.getcwd")

_leaked = sorted(n for n in _FORBIDDEN_MODULES if n in globals())
if _leaked:  # pragma: no cover - structural guard
    raise RuntimeError(
        "styxx.evidence violates its own purity law: " + ", ".join(_leaked)
        + " is bound at module scope. The verdict must be a function of the "
          "input bytes and nothing else.")


# ══════════════════════════════════════════════════════════════════════════════
# purity self-check
# ══════════════════════════════════════════════════════════════════════════════

def selfcheck_purity(source: str | None = None) -> dict:
    """Re-derive Law 1 from this module's own source text.

    Uses ``ast`` rather than a substring sweep on purpose: the forbidden names
    appear in this file's prose and in ``_FORBIDDEN_MODULES`` itself, so a grep
    would accuse the law of breaking the law. Walking the tree asks the only
    question that matters — is any of them actually imported or referenced.
    """
    if source is None:
        try:
            source = Path(__file__).read_text(encoding="utf-8")
        except OSError as exc:
            return {"ok": None, "reason": f"could not read own source: {exc}",
                    "imported": [], "referenced": []}
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:  # pragma: no cover - would fail py_compile first
        return {"ok": None, "reason": f"own source does not parse: {exc}",
                "imported": [], "referenced": []}

    forbidden = set(_FORBIDDEN_MODULES)
    forbidden_exprs = set(_FORBIDDEN_EXPRESSIONS)
    imported: set[str] = set()
    referenced: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.add(node.module.split(".")[0])
        elif isinstance(node, ast.Attribute):
            base = node.value
            if isinstance(base, ast.Name):
                dotted = f"{base.id}.{node.attr}"
                # Two independent questions: is the BASE a forbidden module, and
                # is the whole EXPRESSION forbidden on a permitted module. The
                # second is the one Path.cwd needs — pathlib is allowed here, so
                # no module-level rule can ever reach it.
                if base.id in forbidden or dotted in forbidden_exprs:
                    referenced.add(dotted)

    bad_imports = sorted(imported & forbidden)
    return {
        "ok": not bad_imports and not referenced,
        "reason": ("no forbidden module is imported or referenced in this file"
                   if not bad_imports and not referenced
                   else "forbidden module usage found"),
        "imported": bad_imports,
        "referenced": sorted(referenced),
        "checked": sorted(forbidden),
        "checked_expressions": sorted(forbidden_exprs),
        "boundary": ("proves this FILE imports none of them; it does not prove "
                     "the callers are pure, and it does not prove the bytes on "
                     "disk are honest"),
    }


def selfcheck_no_accusation(source: str | None = None) -> dict:
    """Re-derive from this module's own source that no accusing verdict exists.

    Two independent questions, both answered with ``ast`` rather than a grep,
    because the word necessarily appears in the prose that explains its absence:

      * is the accusing string ever the value of a ``return``/constant in code
        (as opposed to appearing inside a docstring or a comment), and
      * does ``VERDICTS`` contain anything but the two surviving words.

    A string literal that is a bare expression statement — a docstring — is not
    code producing a verdict, so those are excluded. Everything else is
    reported, including the location, so that a reader can look rather than
    trust this paragraph.
    """
    if source is None:
        try:
            source = Path(__file__).read_text(encoding="utf-8")
        except OSError as exc:
            return {"ok": None, "reason": f"could not read own source: {exc}",
                    "code_occurrences": []}
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:  # pragma: no cover
        return {"ok": None, "reason": f"own source does not parse: {exc}",
                "code_occurrences": []}

    word = "CONTRA" + "DICTED"  # split so this line is not itself an occurrence
    docstring_nodes: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            body = getattr(node, "body", None) or []
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                docstring_nodes.add(id(body[0].value))

    occurrences = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and word in node.value and id(node) not in docstring_nodes):
            occurrences.append({"lineno": node.lineno,
                                "excerpt": node.value[:80]})

    return {
        "ok": not occurrences and set(VERDICTS) == {"VERIFIED", "UNCHECKABLE"},
        "reason": ("the accusing verdict is absent from this module: it is not "
                   "produced by any code path and VERDICTS holds only "
                   "VERIFIED and UNCHECKABLE"
                   if not occurrences else
                   "the accusing string appears in non-docstring code"),
        "verdicts": list(VERDICTS),
        "code_occurrences": occurrences,
        "boundary": ("proves the string is not returned by THIS file; it does "
                     "not prove a caller has not invented an accusation of its "
                     "own out of the report-only `observed` band"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# small guarded primitives
# ══════════════════════════════════════════════════════════════════════════════

def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _localname(tag) -> str:
    """`{ns}testsuite` -> `testsuite`. Surefire 3.x puts xmlns:xsi on the root.

    Exact match after this, never substring: Surefire admits `flakyFailure` and
    `rerunFailure` as siblings of `failure`, and a `tag.endswith("failure")`
    calls a PASSING flaky test a failure. That is the exact false-accusation
    shape this lab retired.
    """
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1]


def _num(value) -> int | None:
    """Counts are declared xs:string in Surefire's own XSD, so a non-integer
    count is schema-valid. A failed parse yields unknown, never 0 and never an
    exception — because in go-junit-report a MISSING count means zero and in
    pytest it means unknown, and the bytes cannot tell you which."""
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return int(float(text))
    except ValueError:
        return None


def _is_full_hex_digest(value) -> bool:
    """Full-length lowercase-comparable hex only.

    No prefix matching. The community reference's own example commit property
    uses a 7-character abbreviated SHA, and an implementer who follows it writes
    `startswith()` — at which point an empty string prefix-matches every commit
    and the whole comparison becomes a no-op.
    """
    if not isinstance(value, str):
        return False
    v = value.strip().lower()
    if len(v) not in (40, 64):
        return False
    return all(c in "0123456789abcdef" for c in v)


def _norm_commit(value) -> str | None:
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    return v if _is_full_hex_digest(v) else None


# ══════════════════════════════════════════════════════════════════════════════
# hardened XML parsing — the entity defence is OURS
# ══════════════════════════════════════════════════════════════════════════════

class EntityDeclarationRefused(ValueError):
    """Raised when a document declares an entity or a notation.

    A ValueError subclass on purpose: it lands in the same ``unparsed`` band as
    every other refusal, which is a report about what was NOT read rather than a
    silent drop.
    """


def _parse_xml_hardened(data: bytes) -> tuple[ET.Element, dict]:
    """Parse XML with entity declarations refused AT THE DECLARATION.

    Returns ``(root, notes)``. The defence is this module's own: expat is driven
    directly and every declaration handler that could introduce an expandable or
    external entity raises. It does not depend on libexpat's amplification
    limit, on the ``DEFAULT`` value of any build flag, or on a substring sweep
    over a prefix of the text.

    What is refused, at the point of declaration and before any expansion:

      * ``<!ENTITY …>`` — general or parameter, internal or external
      * ``<!ENTITY … NDATA …>`` — unparsed entities
      * ``<!NOTATION …>``
      * any external entity reference expat asks us to resolve

    What is NOT refused, and this is the E-4 repair: a ``<!DOCTYPE>`` that
    declares none of the above, and — crucially — the literal text
    ``<!DOCTYPE html>`` appearing INSIDE ``<system-out>`` because a snapshot
    test captured some HTML. That is character data. It is not a declaration,
    and a reader that cannot tell the difference hands any pull-request author a
    one-line denial of adjudication for the whole evidence set.

    Bytes are handed to expat rather than a pre-decoded ``str`` so that the
    document's own XML declaration governs its encoding, rather than this
    module's guess about it.
    """
    notes: dict = {"entity_defence": "ours", "doctype_seen": None}
    try:
        parser = _expat.ParserCreate(namespace_separator="}")
    except Exception as exc:  # pragma: no cover - exotic interpreter
        raise EntityDeclarationRefused(
            "could not construct an expat parser with the entity-declaration "
            f"handlers this module requires ({exc.__class__.__name__}: {exc}); "
            "the document is refused rather than parsed under a protection "
            "this module cannot demonstrate it has") from exc

    builder = ET.TreeBuilder()

    def _qname(name: str) -> str:
        # expat with namespace_separator="}" yields "uri}local"; restore the
        # ElementTree spelling so _localname and any recorded tag agree with
        # what the rest of this file has always seen.
        return "{" + name if "}" in name else name

    def _doctype(name, sysid, pubid, has_internal_subset):
        notes["doctype_seen"] = {
            "name": name, "system_id": sysid, "public_id": pubid,
            "has_internal_subset": bool(has_internal_subset),
            "note": ("a DOCTYPE is recorded, not refused. Only DECLARATIONS "
                     "inside it are refused, and none were seen at this point."),
        }

    def _refuse_entity(name, *_rest):
        raise EntityDeclarationRefused(
            f"document declares entity {name!r}; refused at the declaration, "
            "unexpanded. Entity expansion is a memory-amplification surface and "
            "this reader closes it itself rather than relying on the expat "
            "build's amplification limit.")

    def _refuse_notation(name, *_rest):
        raise EntityDeclarationRefused(
            f"document declares notation {name!r}; refused at the declaration.")

    def _refuse_external(*_a):
        raise EntityDeclarationRefused(
            "document references an external entity; refused. Resolving it "
            "would make the verdict a function of the network, which Law 1 "
            "forbids outright.")

    try:
        parser.buffer_text = True
        parser.StartDoctypeDeclHandler = _doctype
        parser.EntityDeclHandler = _refuse_entity
        parser.UnparsedEntityDeclHandler = _refuse_entity
        parser.NotationDeclHandler = _refuse_notation
        parser.ExternalEntityRefHandler = _refuse_external
        parser.StartElementHandler = lambda tag, attrib: builder.start(
            _qname(tag), attrib)
        parser.EndElementHandler = lambda tag: builder.end(_qname(tag))
        parser.CharacterDataHandler = builder.data
    except (AttributeError, TypeError) as exc:
        raise EntityDeclarationRefused(
            "this interpreter's expat parser does not accept the "
            f"entity-declaration handlers this module installs ({exc}); the "
            "document is refused rather than parsed under an entity protection "
            "that would be inherited rather than ours") from exc

    try:
        parser.Parse(data, True)
    except _expat.ExpatError as exc:
        raise ET.ParseError(str(exc)) from exc

    return builder.close(), notes


# ══════════════════════════════════════════════════════════════════════════════
# JUnit XML
# ══════════════════════════════════════════════════════════════════════════════

# Outcome-bearing children of <testcase>, matched EXACTLY.
_CASE_ERROR = "error"
_CASE_FAILURE = "failure"
_CASE_SKIPPED = "skipped"
# Surefire: a test that failed then PASSED on rerun. Its own documentation says
# existing consumers still consider it a passing test, and Jenkins reads it into
# a separate collection that never makes a case fail. Ignored for verdicts.
_CASE_FLAKY = ("flakyFailure", "flakyError")
# These accompany a genuine failure, alongside the ordinary <failure>/<error>
# for the first run. Redundant, so never independently decisive.
_CASE_RERUN = ("rerunFailure", "rerunError")

_KNOWN_ELEMENTS = {
    "testsuites", "testsuite", "testcase", "properties", "property",
    "system-out", "system-err", "failure", "error", "skipped",
    "flakyFailure", "flakyError", "rerunFailure", "rerunError",
}

# Property keys that HAVE been used in the wild to smuggle a commit into JUnit
# XML. Enumerated, never guessed at by pattern. Recording one of these is worth
# doing; TRUSTING one is not, and the code below never treats it as a match —
# the <properties> block is written by the same job that produced the report,
# which makes it a restatement of the claim rather than a check on it.
_COMMIT_PROPERTY_KEYS = (
    "commit", "git_commit", "gitcommit", "git.commit", "git.commit.id",
    "GIT_COMMIT", "revision", "sha", "vcs.revision",
)


def _resolve_case(case) -> tuple[str, list[str]]:
    """Decide one <testcase>'s outcome from CHILD ELEMENTS ONLY.

    Never from an attribute. `status` means a different thing in every dialect
    that emits it: trx2junit writes 1 for success and 0 for failure and lets an
    operator redefine both through env vars; googletest writes
    status="run" result="completed" on FAILING testcases, where "completed"
    means "ran to completion", not "passed". Jenkins, the twenty-year reference
    consumer, ignores the attribute entirely.

    Precedence is explicit — error > failure > skipped > pass — because a single
    testcase can carry BOTH <failure> and <skipped>: googletest's own golden
    fixture has `SkippedAfterFailure` with both, counted by its suite as a
    failure. Any rule of the form "has a <skipped> child therefore skipped"
    silently converts a real failure into a skip.
    """
    names = [_localname(child.tag) for child in list(case)]
    unknown = [n for n in names if n not in _KNOWN_ELEMENTS]

    if _CASE_ERROR in names:
        return "errored", unknown
    if _CASE_FAILURE in names:
        return "failed", unknown
    if _CASE_SKIPPED in names:
        return "skipped", unknown
    if any(n in _CASE_RERUN for n in names):
        # A rerun element with no plain <failure>/<error> beside it should not
        # occur. Rather than guess which way it leans, refuse: `indeterminate`
        # blocks VERIFIED and is reported.
        return "indeterminate", unknown
    if any(n in _CASE_FLAKY for n in names):
        return "passed", unknown

    # DEMOTE-ONLY attribute read, and the only one in this file. googletest
    # marks a never-executed DISABLED_ test status="notrun" result="suppressed"
    # with NO <skipped> child, so element-only rules resolve it to a PASS and an
    # all-disabled binary renders green. Reading these two attributes can only
    # take a test OUT of the passed column; it can never put one into a failure
    # column.
    status = (case.get("status") or "").strip().lower()
    result = (case.get("result") or "").strip().lower()
    if status == "notrun" or result in ("suppressed", "skipped"):
        return "notrun", unknown
    return "passed", unknown


# Bound on how deep a <testsuite>/<testsuites> nest this module will descend.
# The walk below is ITERATIVE, so depth costs no Python stack frames and this
# limit is not there to protect the interpreter — it is there so that a
# pathological document is REPORTED as refused rather than walked in silence.
_MAX_SUITE_DEPTH = 512


def _walk_suites(node, out: list) -> None:
    """Collect every <testsuite>, at any depth. ITERATIVE, by defect report.

    Both roots are accepted and dispatch is on the tag name: <testsuites> for
    pytest, jest-junit, go-junit-report, trx2junit and googletest; a bare
    <testsuite> for Maven Surefire, whose own XSD declares testsuite as the root
    element. Depth is not assumed — <testsuite> nesting occurs in Ant-aggregated
    output.

    This used to recurse once per nesting level, so a 25 KB file of a thousand
    nested <testsuites> raised RecursionError OUT of load_evidence instead of
    landing in `unparsed`. A crash is not a verdict and it is not a refusal
    either; it is the caller's problem, arriving as a traceback. An explicit
    stack costs one list and removes the failure mode at its source.
    """
    stack = [(node, 0)]
    while stack:
        current, depth = stack.pop()
        if depth > _MAX_SUITE_DEPTH:
            raise ValueError(
                f"<testsuite>/<testsuites> nesting exceeds {_MAX_SUITE_DEPTH} "
                "levels. No real producer emits this; the document is refused "
                "and recorded rather than walked.")
        if _localname(current.tag) == "testsuite":
            out.append(current)
        for child in list(current):
            if _localname(child.tag) in ("testsuite", "testsuites"):
                stack.append((child, depth + 1))


def _guess_producer(root, suites) -> tuple[str, str]:
    """Producer identification is heuristic, weak, and user-overridable.

    It is recorded so that a dialect-specific rule can be prevented from firing
    on an unidentified producer — never so that one can fire more eagerly.
    """
    root_name = (root.get("name") or "").strip()
    prop_names = set()
    for suite in suites:
        for props in suite:
            if _localname(props.tag) != "properties":
                continue
            for prop in props:
                if _localname(prop.tag) == "property":
                    prop_names.add((prop.get("name") or "").strip())
    if "go.version" in prop_names:
        return "go-junit-report", "weak"
    if any(p.startswith("surefire.") for p in prop_names):
        return "maven-surefire", "weak"
    if root_name == "pytest tests":
        return "pytest", "weak"
    if root_name == "AllTests":
        return "googletest", "weak"
    if root_name == "jest tests":
        return "jest-junit", "weak"
    if any((s.get("package") or "") == "not available" for s in suites):
        return "trx2junit", "weak"
    if _localname(root.tag) == "testsuite":
        return "maven-surefire?", "very weak"
    return "unknown", "none"


def _empty_binding(kind: str, commit, why: str, **extra) -> dict:
    """One shape for every per-source binding record, aliases included.

    The two honest names are ``report_identity_asserted`` and
    ``commit_assertion_matches``. The frozen contract's ``*_verified``
    spellings are carried alongside them with the same values so that a
    contract-conformant reader and an honest reader never see different
    answers — but nothing in this module verifies anything, and the alias names
    are inaccurate. See "WHAT `binding` IS NOT" in the module docstring.
    """
    identity = bool(extra.pop("report_identity_asserted", False))
    matches = bool(extra.pop("commit_assertion_matches", False))
    out = {
        "kind": kind,
        "commit": commit,
        "report_identity_asserted": identity,
        "commit_assertion_matches": matches,
        # Contract-name aliases. Same values. The word "verified" in them is
        # wrong and is kept only because the frozen contract names these keys.
        "report_identity_verified": identity,
        "commit_binding_verified": matches,
        "signature_verified": False,
        "assertion_only": True,
        "why": why,
    }
    out.update(extra)
    return out


def _parse_junit(data: bytes, path: str) -> dict:
    """Resolve a JUnit XML document to per-testcase verdicts.

    Root attributes are recorded as-found and NEVER trusted. googletest's own
    golden EXPECTED_NO_TEST_XML — the file its test suite asserts against, not a
    bug report — has a root claiming tests="0" failures="0" wrapping a suite
    with a real <failure>. A parser trusting that root reports GREEN on a red
    run. Jenkins recomputes every figure it displays from the <testcase>
    children and reads none of the counts. Twenty years of production exposure
    produced the rule: ignore the counts.
    """
    root, xml_notes = _parse_xml_hardened(data)

    suites: list = []
    _walk_suites(root, suites)
    if not suites and _localname(root.tag) == "testsuite":
        suites = [root]

    counts = {"passed": 0, "failed": 0, "errored": 0, "skipped": 0,
              "notrun": 0, "indeterminate": 0}
    unknown_elements: set[str] = set()
    suite_level_errors = 0
    seen_root = _localname(root.tag)
    if seen_root not in ("testsuites", "testsuite"):
        raise ValueError(f"root element is <{seen_root}>, which is neither "
                         "<testsuites> nor <testsuite>")

    for suite in suites:
        cases = [c for c in suite if _localname(c.tag) == "testcase"]
        for child in suite:
            n = _localname(child.tag)
            if n not in _KNOWN_ELEMENTS:
                unknown_elements.add(n)
        for case in cases:
            verdict, unknown = _resolve_case(case)
            counts[verdict] += 1
            unknown_elements.update(unknown)
        if not cases:
            # Jenkins SuiteResult synthesises a case named <init> for a direct
            # <error> child of <testsuite>, with the comment that this happens
            # "when the test class failed to load". jest-junit encodes the same
            # condition as suite-level errors meaning "this test FILE failed to
            # run". Either way a harness broke; that is ABSENCE of evidence and
            # is kept in its own column, never merged into failures.
            direct_errors = [c for c in suite if _localname(c.tag) == "error"]
            declared = _num(suite.get("errors")) or 0
            if direct_errors or declared > 0:
                suite_level_errors += max(len(direct_errors), declared)
                counts["errored"] += max(len(direct_errors), declared)

    root_as_found = {k: _num(root.get(k)) for k in
                     ("tests", "failures", "errors", "skipped", "disabled")}
    summed_children = {}
    for k in ("tests", "failures", "errors", "skipped", "disabled"):
        vals = [_num(s.get(k)) for s in suites]
        present = [v for v in vals if v is not None]
        summed_children[k] = sum(present) if present else None

    resolved_tests = (counts["passed"] + counts["failed"] + counts["errored"]
                      + counts["skipped"] + counts["notrun"]
                      + counts["indeterminate"])
    executed = counts["passed"] + counts["failed"] + counts["errored"]

    producer, confidence = _guess_producer(root, suites)

    # An asserted commit smuggled through <properties>. Recorded so the census
    # can count the channel; it never counts as a match.
    asserted_commit = None
    for suite in suites:
        for props in suite:
            if _localname(props.tag) != "properties":
                continue
            for prop in props:
                if _localname(prop.tag) != "property":
                    continue
                key = (prop.get("name") or "").strip()
                if key in _COMMIT_PROPERTY_KEYS:
                    val = (prop.get("value") or "").strip()
                    if val:
                        asserted_commit = val

    # EMPTY is defined over EXECUTED tests, not over emitted records. An
    # all-skipped run and an all-DISABLED_ binary emit testcases, execute
    # nothing, and must not render as a satisfied "tests pass".
    if executed == 0:
        outcome = "EMPTY"
    elif counts["failed"] > 0 or counts["errored"] > 0:
        outcome = "FAILED"
    else:
        outcome = "PASSED"

    return {
        "path": path,
        "format": "junit",
        "sha256": _sha256(data),
        # The contract's four count keys, filled from RESOLVED TESTCASES and not
        # from the root attributes the schema's shape invites.
        "tests": resolved_tests,
        "failures": counts["failed"],
        "errors": counts["errored"],
        "skipped": counts["skipped"],
        # Everything below is report-only and exists because the four keys above
        # cannot express what the format actually does.
        "passed": counts["passed"],
        "notrun": counts["notrun"],
        "indeterminate": counts["indeterminate"],
        "executed": executed,
        "outcome": outcome,
        "errors_are_absence": True,
        "suite_level_errors": suite_level_errors,
        "counts_source": "resolved from <testcase> child elements",
        "root_as_found": root_as_found,
        "summed_children_as_found": summed_children,
        "root_children_disagree": (
            root_as_found["tests"] is not None
            and summed_children["tests"] is not None
            and root_as_found["tests"] != summed_children["tests"]),
        "unrecognized_elements": sorted(unknown_elements),
        "producer_guess": producer,
        "producer_confidence": confidence,
        "suites": len(suites),
        "xml_notes": xml_notes,
        "binding": _empty_binding(
            "asserted" if asserted_commit else "none",
            asserted_commit,
            ("no JUnit dialect defines any schema-level field for a commit, "
             "revision, branch or repository — verified against Surefire's "
             "XSD, the Ant/Jenkins reference, pytest, go-junit-report, "
             "jest-junit and trx2junit. The only carrier is the free-form "
             "<properties> channel, which is written by the same job that "
             "produced the report and is therefore ASSERTED-NOT-CHECKED at "
             "best."),
            git_commits=[],
            subjects_seen=0,
            git_digest_keys_seen=[],
            degraded_key=None,
            notes=[]),
        "known_blind_spots": [
            "jest-junit under its default reportTestSuiteErrors:'false' DROPS a "
            "suite that failed to compile from the XML entirely — a broken test "
            "file produces a fully green report with no trace. Unrecoverable "
            "from these bytes.",
            "a shard whose report was never uploaded leaves no mark anywhere in "
            "this document; absence of a file is invisible to a file reader.",
            "`tests=` counts testcase records emitted, not tests executed: "
            "googletest counts never-run DISABLED_ tests, pytest counts "
            "collection failures where zero test bodies ran.",
        ],
    }


# ══════════════════════════════════════════════════════════════════════════════
# in-toto test-result
# ══════════════════════════════════════════════════════════════════════════════

def _b64_tolerant(text: str) -> bytes:
    """DSSE allows standard OR url-safe base64. Try both, tolerate padding."""
    raw = "".join(str(text).split())
    pad = "=" * (-len(raw) % 4)
    for decoder in (base64.b64decode, base64.urlsafe_b64decode):
        try:
            return decoder(raw + pad)
        except (binascii.Error, ValueError):
            continue
    raise ValueError("payload is not decodable as standard or url-safe base64")


def _statement_from(obj) -> tuple[dict, dict]:
    """Return (statement, envelope_info). Accepts a DSSE envelope or a bare
    Statement.

    We decode and read. We do NOT verify. in-toto's own words, quoted rather
    than paraphrased away: "To obtain predicate information that is
    authenticated, consumers MUST parse the Envelope's `payload`, and verify it
    against its `signatures`." We parse. We do not verify. Everything derived
    from this payload is unauthenticated, and every line of output says so.

    Under the previous version this was the entry point to an accusation, which
    made an empty `signatures` array a weapon. It is no longer an entry point to
    anything but a report, which is the only reason reading unauthenticated
    bytes here is defensible at all.
    """
    if not isinstance(obj, dict):
        raise ValueError("top-level JSON value is not an object")

    if "payload" in obj and "payloadType" in obj:
        ptype = obj.get("payloadType")
        # Both forms are conforming: the generic type and the
        # per-predicate application/vnd.in-toto.<predicate>+json.
        ok_type = isinstance(ptype, str) and (
            ptype == "application/vnd.in-toto+json"
            or (ptype.startswith("application/vnd.in-toto.")
                and ptype.endswith("+json")))
        if not ok_type:
            raise ValueError(f"payloadType {ptype!r} is not an in-toto type")
        stmt = json.loads(_b64_tolerant(obj["payload"]).decode("utf-8"))
        sigs = obj.get("signatures")
        n_sigs = len(sigs) if isinstance(sigs, list) else 0
        env = {
            "dsse": True,
            "payloadType": ptype,
            "signature_count": n_sigs,
            "signature_checked": False,
            "signature_note": (
                "NOT CHECKED. No key material was consulted and no trust root "
                "was pinned. This envelope carries "
                + (f"{n_sigs} signature(s) that were not examined"
                   if n_sigs else
                   "an EMPTY signatures array, which this reader treats "
                   "exactly the same as a full one — because it checks "
                   "neither")
                + ". A passing check would have proven only that some key "
                  "signed these bytes; it would say nothing about whether the "
                  "tests ran or whether the signer is honest."),
        }
        return stmt, env

    if obj.get("_type") == INTOTO_STATEMENT_TYPE:
        return obj, {"dsse": False, "signature_count": 0,
                     "signature_checked": False,
                     "signature_note": ("bare statement, unsigned; there is not "
                                        "even a signature here to decline to "
                                        "check")}
    raise ValueError("neither a DSSE envelope (payload/payloadType) nor a bare "
                     f"in-toto Statement (_type == {INTOTO_STATEMENT_TYPE})")


def _read_subjects(subjects, source_sha256: str, known_sha256: set[str]) -> dict:
    """Collect EVERY subject digest. No comparison target, no latching.

    The spec is explicit: "Subject artifacts are matched purely by digest,
    regardless of content type", and `name` is optional and may be the literal
    "_". So every element is scanned and EVERY ``gitCommit`` value is kept in a
    list.

    The previous version latched the FIRST ``gitCommit`` it saw into a scalar,
    and the function that later decided the match only ever saw that scalar —
    which made the multi-subject scan dead code and made the answer depend on
    the ORDER of the subject array. Identical digest sets, permuted, gave
    different answers. in-toto says order carries no meaning, so any function of
    it is a function of noise.

    `gitCommit` is the conventional key and is unambiguous. `sha1` is accepted
    as a DEGRADED key and deliberately never counts as a match: a 40-hex value
    under `sha1` is equally consistent with a commit id and with the SHA-1 of a
    file's bytes, and nothing in the document distinguishes them. `gitTree` is
    refused outright — it names CONTENT, not this commit.

    Report identity and commit assertion are kept in two independent booleans. A
    subject carrying sha256 of junit.xml proves "this is the report that was
    attested" and is completely SILENT on "which commit did it run against".
    Letting one word cover both is the most dangerous available mistake here,
    precisely because it looks like the strongest channel.
    """
    git_commits: list[str] = []
    sha1s: list[str] = []
    keys_seen: list[str] = []
    notes: list[str] = []
    subjects_seen = 0
    identity = False
    gittree_seen = False

    if not isinstance(subjects, list) or not subjects:
        notes.append("subject array is absent or empty; a Statement REQUIRES "
                     "it, so this attestation is malformed")
    else:
        for sub in subjects:
            if not isinstance(sub, dict):
                continue
            subjects_seen += 1
            digest = sub.get("digest")
            if not isinstance(digest, dict):
                continue
            for key in ("gitCommit", "gitTree", "gitBlob", "gitTag", "sha1"):
                if key in digest:
                    keys_seen.append(key)
            gc = _norm_commit(digest.get("gitCommit"))
            if gc and gc not in git_commits:
                git_commits.append(gc)
            s1 = _norm_commit(digest.get("sha1"))
            if s1 and s1 not in sha1s:
                sha1s.append(s1)
            if "gitTree" in digest:
                gittree_seen = True
            for value in digest.values():
                if isinstance(value, str) and value.strip().lower() in known_sha256:
                    identity = True

    if gittree_seen:
        notes.append("a subject carries gitTree, which names the CONTENT under "
                     "a possibly different commit. Recorded; never a commit "
                     "assertion.")
    if not keys_seen and subjects_seen:
        notes.append("no subject carries a git-shaped digest key. A sha256 of a "
                     "tarball says nothing about a commit, and reading one as "
                     "though it did would be the string-shaped inference this "
                     "lab retired.")
    if identity:
        notes.append("a subject digest matches a report loaded in this run: "
                     "REPORT IDENTITY IS ASSERTED. That is a different "
                     "proposition from a commit assertion, and neither was "
                     "cryptographically checked.")
    if len(git_commits) > 1:
        notes.append(f"{len(git_commits)} distinct gitCommit digests appear "
                     "among the subjects. All are recorded; a match against any "
                     "of them is order-independent, because in-toto attaches no "
                     "meaning to subject order.")

    return _empty_binding(
        "asserted" if (git_commits or sha1s) else "none",
        git_commits[0] if git_commits else None,
        ("what the bytes ASSERT about a commit. Nothing here was verified: no "
         "signature was checked, so this is a string equal to another string, "
         "in a file whoever wrote it chose the contents of."),
        report_identity_asserted=identity,
        commit_assertion_matches=False,
        git_commits=list(git_commits),
        sha1_digests=list(sha1s),
        subjects_seen=subjects_seen,
        git_digest_keys_seen=keys_seen,
        degraded_key=None,
        source_sha256=source_sha256,
        notes=notes)


def _binding_against(binding: dict, want: str | None) -> dict:
    """Re-derive one source's binding against the commit named by the caller.

    Order-independent by construction: it asks whether ``want`` is IN the set of
    gitCommit digests the document carries, never whether it equals the first
    one seen.
    """
    out = dict(binding)
    out["compared_against"] = want
    git_commits = list(out.get("git_commits") or [])
    sha1s = list(out.get("sha1_digests") or [])
    notes = list(out.get("notes") or [])
    matched = bool(want) and want in git_commits

    if matched:
        out["kind"] = "intoto-subject-asserted"
        out["commit"] = want
        out["degraded_key"] = None
    elif want and want in sha1s:
        out["kind"] = "asserted"
        out["commit"] = want
        out["degraded_key"] = "sha1"
        notes = notes + [
            "the commit under review appears under the ambiguous `sha1` key, "
            "which equally denotes the SHA-1 of a file's bytes. DEGRADED-KEY: "
            "recorded as a sighting, refused as an assertion about a commit."]
    elif git_commits and want:
        notes = notes + [
            "this statement names commit(s) and none of them is the commit "
            "under review. A statement about other bytes is not evidence about "
            "these."]
    elif git_commits:
        notes = notes + [
            "this statement names a commit, and no commit was supplied to "
            "compare it against."]

    out["commit_assertion_matches"] = matched
    out["commit_binding_verified"] = matched  # contract-name alias, same value
    out["notes"] = notes
    return out


def binding_against_commit(ev: dict, commit: str | None = None) -> list[dict]:
    """Re-derive EVERY source's binding against `commit`. One dict PER SOURCE.

    ``load_evidence`` cannot do this itself — a commit is not an input to
    reading bytes off disk — so the comparison lives here and is the same one
    ``adjudicate_tests_pass`` and the CLI use. A library caller that wants to
    know "does this evidence assert commit X" calls this; there is deliberately
    no single aggregate answer, because collapsing a list of files into one
    boolean is how an attested file's assertion ends up appearing to cover an
    unattested sibling.

    Order-independent: it asks whether `commit` is IN the set of gitCommit
    digests each document carries. Nothing here is verified; see "WHAT
    `binding` IS NOT" in the module docstring.
    """
    want = _norm_commit(commit)
    out = []
    for s in ev.get("sources") or []:
        b = _binding_against(s.get("binding") or {}, want)
        out.append({"path": s.get("path"), "format": s.get("format"), **b})
    return out


def _parse_intoto(data: bytes, path: str, known_sha256: set[str]) -> dict:
    text = data.decode("utf-8", errors="replace")
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # .intoto.jsonl carries one envelope per line.
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if len(lines) != 1:
            raise ValueError(
                "not a single JSON document; JSON Lines with more than one "
                "envelope is refused rather than guessed at, because merging "
                "envelopes from different runs is how a superseded red result "
                "gets read alongside the green one that replaced it")
        obj = json.loads(lines[0])

    stmt, env = _statement_from(obj)
    if stmt.get("_type") != INTOTO_STATEMENT_TYPE:
        raise ValueError(f"_type is {stmt.get('_type')!r}, not "
                         f"{INTOTO_STATEMENT_TYPE}")
    ptype = stmt.get("predicateType")
    if ptype != INTOTO_TEST_RESULT_PREDICATE:
        raise ValueError(
            f"predicateType {ptype!r} is not the exact literal "
            f"{INTOTO_TEST_RESULT_PREDICATE}. A test-result-shaped predicate "
            "under an unrecognised URI is refused rather than assumed "
            "schema-compatible: predicates are versioned independently and a "
            "future major version is a different string.")

    pred = stmt.get("predicate")
    if not isinstance(pred, dict):
        raise ValueError("predicate is absent or is not an object")
    result = pred.get("result")
    if result not in ("PASSED", "WARNED", "FAILED"):
        raise ValueError(f"predicate.result is {result!r}, which is outside the "
                         "enum PASSED|WARNED|FAILED")

    # ALL THREE LISTS ARE OPTIONAL. An absent failedTests is legal silence, not
    # a statement that nothing failed, so .get() with an explicit None rather
    # than a default [] that would read as corroboration.
    def _names(key):
        v = pred.get(key)
        return v if isinstance(v, list) else None

    passed = _names("passedTests")
    warned = _names("warnedTests")
    failed = _names("failedTests")
    lists_present = any(v is not None for v in (passed, warned, failed))

    n_pass = len(passed) if passed is not None else 0
    n_warn = len(warned) if warned is not None else 0
    n_fail = len(failed) if failed is not None else 0
    executed = n_pass + n_warn + n_fail

    # Both directions of internal inconsistency, and neither is resolved.
    inconsistent_reasons = []
    if result == "PASSED" and n_fail > 0:
        inconsistent_reasons.append(
            "result=PASSED while failedTests names "
            f"{n_fail} test(s)")
    if result == "FAILED" and lists_present and n_fail == 0 and executed > 0:
        inconsistent_reasons.append(
            "result=FAILED while failedTests is empty and "
            f"{executed} named test(s) are all non-failing")

    binding = _read_subjects(stmt.get("subject"), _sha256(data), known_sha256)

    # E-1. The bare `result` string is an ASSERTION WITH NO TEST-LEVEL EVIDENCE
    # BEHIND IT whenever the three optional name lists are absent. The previous
    # version read `result == "FAILED"` straight into a FAILED outcome, so a
    # document naming zero tests produced totals.tests == 0 AND outcome ==
    # FAILED simultaneously — which contradicts this module's own definition of
    # EMPTY and, under the old vocabulary, walked to an accusation on an
    # attestation reporting no failing test at all.
    #
    # EMPTY IS DEFINED OVER EXECUTED TESTS. Zero executed tests is EMPTY no
    # matter what the producer's summary string says, and the summary string is
    # preserved verbatim in `result` for any reader who wants it.
    if executed == 0:
        outcome = "EMPTY"
    elif n_fail > 0 or result == "FAILED":
        outcome = "FAILED"
    elif result == "WARNED":
        # WARNED is neither pass nor fail and is not folded into either.
        outcome = "WARNED"
    else:
        outcome = "PASSED"

    return {
        "path": path,
        "format": "intoto-testresult",
        "sha256": _sha256(data),
        "tests": executed,
        "failures": n_fail,
        # The predicate HAS NO ERROR CATEGORY. This zero means "not expressible
        # in this format", not "observed to be zero" — a distinction that
        # mattered when an errors-only reading was a refusal path, and is kept
        # explicit so no future reader mistakes the one for the other.
        "errors": 0,
        "errors_expressible": False,
        "errors_are_absence": True,
        "skipped": 0,
        "passed": n_pass,
        "warned": n_warn,
        "notrun": 0,
        "indeterminate": 0,
        "executed": executed,
        "outcome": outcome,
        "result": result,
        "result_note": ("`result` is the producer's own summary string. It is "
                        "reported as found and it never sets the outcome by "
                        "itself: with no test-name lists there are zero "
                        "executed tests behind it, and zero executed tests is "
                        "EMPTY."),
        "test_name_lists_present": lists_present,
        "internally_inconsistent": bool(inconsistent_reasons),
        "inconsistent_reasons": inconsistent_reasons,
        "predicate_type": ptype,
        "envelope": env,
        "binding": binding,
        "counts_source": ("the predicate's optional test-name lists; the "
                          "predicate carries no count of tests run, no skipped "
                          "category, no exit code and no framework identity"),
        "known_blind_spots": [
            "result=\"PASSED\" is an ASSERTION BY THE PRODUCER, not an "
            "observation by us; a subject digest names a commit in the "
            "STATEMENT, and does not tie the test EXECUTION to that commit.",
            "the three test-name lists are OPTIONAL, so a conforming "
            "attestation may say PASSED and name nothing. Their absence carries "
            "zero information.",
            "nothing here reports whether the run was complete: one matrix leg "
            "of twelve, a -k subset, a --maxfail short-circuit and a full green "
            "suite all serialise identically.",
            "the signatures array is not examined, so an empty one, a forged "
            "one and a valid one are indistinguishable to this reader.",
        ],
    }


# ══════════════════════════════════════════════════════════════════════════════
# load
# ══════════════════════════════════════════════════════════════════════════════

def _sniff(data: bytes) -> str | None:
    """Format from CONTENT, never from the extension.

    A file named .json holding XML is a mislabelled file, not a different
    format, and dispatching on the name would put it in `unparsed` for the wrong
    reason.
    """
    head = data.lstrip(b"\xef\xbb\xbf \t\r\n")[:1]
    if head == b"<":
        return "xml"
    if head in (b"{", b"["):
        return "json"
    return None


_BOUNDARY = (
    "This module reads bytes. It does NOT verify any signature and consults no "
    "key material or trust root, so in in-toto's own words what it read is not "
    "authenticated predicate information. It does NOT know whether the report "
    "is honest: whoever could write the file could write any verdict into it, "
    "including one naming the right commit. It does NOT know whether the run "
    "was complete — one matrix leg, a -k subset, a --maxfail short-circuit and "
    "a whole green suite serialise identically. It does NOT know what was never "
    "uploaded. It inherits the CI's trust entirely and adds none of its own. "
    "It cannot accuse: the accusing verdict is absent from its vocabulary.")

_NOT_CHECKED = [
    "signatures — no key material, no pinned trust root, no Rekor, no Fulcio; a "
    "half-verified chain reporting 'verified' would be the most dangerous line "
    "in this repository, so none is attempted. This is why `binding` is called "
    "an ASSERTION and never a cryptographic binding.",
    "whether the tests named are the tests that matter, or that any test exists",
    "whether the report is complete: shards, matrix legs, continue-on-error "
    "legs, -k selection and --maxfail short-circuits are all invisible here",
    "whether a jest suite that failed to compile was silently dropped from the "
    "XML by the reporter's default configuration",
    "the SCOPE of the prose claim being read. This function is handed evidence, "
    "never a sentence. On this lab's own corpus only 40.9% of 'tests pass' "
    "matches were unqualified assertions; 45.5% named a subset, 14.0% were "
    "PR-template checkboxes and 3.7% were explicitly about the author's local "
    "machine. A caller that feeds those here as bare claims has an extraction "
    "problem this module cannot see and does not fix.",
]

_CONTRACT_DEVIATIONS = [
    "THE ACCUSING VERDICT IS DELETED, NOT DISABLED. VERDICTS is "
    "('VERIFIED', 'UNCHECKABLE') and no code path in this module returns a "
    "third word. The frozen table's accusing row has no implementation: its "
    "precision is structurally unmeasurable (11 firings across 1,775,765 "
    "changed files cannot furnish a 100-item blind panel) and the check that "
    "was to license it is a string compare on attacker-writable bytes. This is "
    "a permanent design decision, not a suspension.",
    "evidence of failure is returned in the report-only `observed` band. It is "
    "not a verdict, nothing downstream may gate on it, and main() never exits "
    "non-zero because of it.",
    "row 1's quantifier ('EVERY source unparsed') would permit concluding on a "
    "tenth of the evidence. Here ANY unparsed source blocks VERIFIED: a partial "
    "read may honestly decline, it may not honestly affirm.",
    "EMPTY is computed over EXECUTED tests rather than emitted testcase "
    "records, so an all-skipped run and an all-DISABLED_ binary are both EMPTY "
    "and therefore UNCHECKABLE. VERIFIED additionally requires passed > 0.",
    "an in-toto predicate's bare `result` string never sets the outcome by "
    "itself. With no test-name lists there are zero executed tests behind it, "
    "and zero executed tests is EMPTY whatever the summary string says.",
    "errors (a harness that could not load a module, a jest file that would not "
    "run) are ABSENCE of evidence and are kept in their own column, never "
    "merged into failures. They cannot produce any verdict; they can only "
    "block VERIFIED.",
    "two sources resolving to different outcomes yield UNCHECKABLE rather than "
    "a merge. A stale artifact beside a fresh one, and a re-run of failed jobs, "
    "both produce exactly this shape and both are green in reality.",
    "binding is recorded PER SOURCE. There is no aggregate binding dict "
    "spanning the list, because one attested file's digest assertion must never "
    "appear to cover an unattested sibling globbed out of the same directory.",
    "`commit_binding_verified` is an alias whose name the frozen contract "
    "fixes and whose word 'verified' is inaccurate. It is set only by a "
    "subject gitCommit digest EQUAL to the commit the caller named, and no "
    "signature is checked, so it means 'the bytes assert this', never 'this is "
    "proven'. A `sha1` match is DEGRADED-KEY and a <properties> commit is "
    "ASSERTED; neither sets it.",
    "the entity defence is this module's own: expat is driven directly with "
    "entity, unparsed-entity and notation declarations refused at the "
    "declaration, replacing a substring sweep that both false-positived on "
    "green reports containing HTML and false-negatived on a bomb hidden behind "
    "a long comment.",
    "every deviation above moves verdicts toward UNCHECKABLE. None of them can "
    "produce a conclusion the frozen table would not, which is the property to "
    "check rather than trust.",
]


def load_evidence(paths) -> dict:
    """Read test evidence from disk. A pure function of the bytes at `paths`.

    Reads files and nothing else — no environment, no clock, no network. Which
    paths get globbed is the caller's business and is ambient state this module
    cannot see; the resolved path list is recorded so that a capsule re-gate
    diverges on a different source set rather than silently agreeing with it.

    The returned dict carries ``contract_deviations`` ITSELF. The previous
    version documented that key in this docstring and then attached it only in
    ``main()``, so every library caller read a promise the library did not keep.

    It also carries ``observed``: the REPORT-ONLY band holding what the evidence
    says about failure. That band is information, not a verdict. Nothing
    downstream may gate on it, and this module never turns it into one.
    """
    if isinstance(paths, str):
        paths = [paths]
    paths = list(paths or [])

    # Two passes: read all bytes first so that an attestation's subject digest
    # can be compared against the sha256 of the reports actually in hand. A
    # digest naming a DIFFERENT file is not report identity, and the only way to
    # notice is to have both digests at once.
    blobs: list[tuple[str, bytes | None, str | None]] = []
    for p in paths:
        try:
            blobs.append((str(p), Path(p).read_bytes(), None))
        except OSError as exc:
            blobs.append((str(p), None, f"unreadable: {exc.__class__.__name__}: {exc}"))
    known_sha256 = {_sha256(b) for _, b, _ in blobs if b is not None}

    # A commit is not an input to loading. The comparison is re-derived at
    # adjudication against the commit the caller names there; what is recorded
    # per source here is the "what does this file carry" half, with no
    # comparison target.
    sources: list[dict] = []
    unparsed: list[dict] = []
    for path, data, read_error in blobs:
        if read_error is not None:
            unparsed.append({"path": path, "reason": read_error})
            continue
        if not data.strip():
            unparsed.append({"path": path, "reason": "file is empty (0 bytes of "
                                                     "content); absence, not a "
                                                     "green run"})
            continue
        kind = _sniff(data)
        try:
            if kind == "xml":
                sources.append(_parse_junit(data, path))
            elif kind == "json":
                sources.append(_parse_intoto(data, path, known_sha256))
            else:
                head = data.lstrip()[:16]
                raise ValueError(
                    "content is neither XML nor JSON (first bytes: "
                    f"{head!r}); format is detected from content, not the "
                    "extension")
        # RecursionError is listed EXPLICITLY and is not covered by any other
        # member: it descends from RuntimeError, not ValueError. A thousand
        # nested <testsuites> in 25 KB, or a comparably nested JSON document,
        # used to raise it straight out of this function. A traceback is not a
        # verdict and it is not a refusal; it is the caller's problem wearing
        # this module's name. It belongs in `unparsed` with everything else this
        # reader declined to read.
        except (ET.ParseError, ValueError, json.JSONDecodeError,
                UnicodeDecodeError, KeyError, TypeError, RecursionError) as exc:
            detail = str(exc) or exc.__class__.__name__
            unparsed.append({"path": path,
                             "reason": f"{exc.__class__.__name__}: {detail}"})

    totals = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    extra = {"passed": 0, "notrun": 0, "indeterminate": 0, "executed": 0}
    for s in sources:
        for k in totals:
            totals[k] += int(s.get(k) or 0)
        for k in extra:
            extra[k] += int(s.get(k) or 0)

    # MERGE BY UNIONING RESOLVED TESTCASES, NEVER BY SUMMING ROOTS. Sharded runs
    # emit one root per shard each claiming the same name; Surefire emits one
    # file per class. Summing root attributes compounds every defect the roots
    # already have.
    # The per-source outcome already applies the rule that EMPTY is defined over
    # EXECUTED tests rather than over emitted testcase records. The aggregate
    # only combines those judgements; it does not re-derive them from summed
    # counts, because that is the arithmetic the roots already get wrong.
    per_source_outcomes = sorted({s.get("outcome") for s in sources})
    if not sources:
        outcome = "EMPTY"
    elif "FAILED" in per_source_outcomes:
        outcome = "FAILED"
    elif "WARNED" in per_source_outcomes:
        outcome = "WARNED"
    elif "PASSED" in per_source_outcomes:
        outcome = "PASSED"
    else:
        outcome = "EMPTY"

    return {
        "spec": SPEC,
        "verdict_vocabulary": list(VERDICTS),
        "sources": sources,
        "totals": totals,
        "resolved": extra,
        "outcome": outcome,
        # ── REPORT-ONLY BAND ─────────────────────────────────────────────────
        # What the evidence says about failure, kept rather than discarded, and
        # kept OUT of the verdict. Deleting the accusing branch must not delete
        # the information; a maintainer reading a red CI still needs to see the
        # red. What must not happen is a gate growing back on top of this band,
        # which is why it says so about itself.
        "observed": {
            "failing_tests": int(totals["failures"]),
            "harness_errors": int(totals["errors"]),
            "skipped_tests": int(totals["skipped"]),
            "not_run_tests": int(extra["notrun"]),
            "indeterminate_tests": int(extra["indeterminate"]),
            "sources_reporting_failure": [
                s["path"] for s in sources if s.get("outcome") == "FAILED"],
            "note": ("REPORT ONLY. These figures are what the supplied bytes "
                     "SAY; they are not a verdict, no verdict is derived from "
                     "them, and nothing downstream may gate on them. This "
                     "module's vocabulary is VERIFIED and UNCHECKABLE — see the "
                     "module docstring for why there is no third word and why "
                     "its absence is permanent rather than suspended. A caller "
                     "that turns `failing_tests > 0` into a failing check has "
                     "reintroduced, without measurement, exactly the verdict "
                     "this module declines to ship."),
            "errors_note": ("`harness_errors` is ABSENCE of evidence — a module "
                            "that would not import, a jest file that would not "
                            "run — and is deliberately not merged into "
                            "`failing_tests`."),
        },
        "unparsed": unparsed,
        "paths_requested": [str(p) for p in paths],
        "per_source_outcomes": per_source_outcomes,
        "sources_disagree": len([o for o in per_source_outcomes
                                 if o in ("PASSED", "FAILED")]) > 1,
        # Binding is PER SOURCE and lives on each entry of `sources`. There is
        # deliberately no top-level `binding` key: the previous version returned
        # a hardcoded {"kind": "none", ...} constant here for EVERY input, so a
        # library caller reading it saw "none" even when a subject carried a
        # real gitCommit digest. A wrong constant is worse than an absent key,
        # because an absent key raises.
        "binding_per_source": [
            {"path": s["path"], "format": s["format"], **(s.get("binding") or {})}
            for s in sources],
        "boundary": _BOUNDARY,
        "not_checked": _NOT_CHECKED,
        "contract_deviations": _CONTRACT_DEVIATIONS,
    }


# ══════════════════════════════════════════════════════════════════════════════
# adjudication
# ══════════════════════════════════════════════════════════════════════════════

def adjudicate_tests_pass(ev: dict, commit: str | None = None) -> tuple[str, str]:
    """Read a "tests pass" claim against evidence bytes. Returns (verdict, why).

    The verdict is one of exactly two words::

        VERIFIED      an attestation or report naming this commit reports
                      PASSED, at least one test executed and passed, every
                      supplied file parsed, and NO SIGNATURE WAS CHECKED.
        UNCHECKABLE   everything else, including every reading in which the
                      evidence looks red.

    **VERIFIED IS NOT "THE TESTS PASSED."** It is "the bytes in hand say so, and
    a forged VERIFIED would only repeat a claim the author already made in
    prose". That last clause is the entire reason this half survives while the
    accusing half was deleted: the two errors are not the same size. A forged
    accusation is an attack on someone else's pull request.

    A total function of `ev` and `commit`. Reads no files, no environment and no
    clock; `ev` is whatever `load_evidence` returned, and re-running this on the
    same dict in ten years returns the same pair. Every branch that can decline
    comes before the one branch that can conclude.
    """
    sources = ev.get("sources") or []
    unparsed = ev.get("unparsed") or []
    totals = ev.get("totals") or {}
    resolved = ev.get("resolved") or {}
    outcome = ev.get("outcome")

    def _u(why: str) -> tuple[str, str]:
        return "UNCHECKABLE", why

    # ── declining branches, all of them before the one conclusion ────────────
    if not sources and not unparsed:
        return _u("no evidence was supplied. Absence of a report is not a "
                  "failing report; an unattested commit is unattested.")
    if not sources:
        return _u(f"every one of the {len(unparsed)} supplied files failed to "
                  f"parse: {unparsed[0]['reason']}"
                  + (f" (and {len(unparsed) - 1} more)" if len(unparsed) > 1 else ""))
    if unparsed:
        # Strengthened quantifier. Reading one file in ten and concluding on it
        # is not a partial read, it is a guess with a citation.
        return _u(f"{len(unparsed)} of {len(sources) + len(unparsed)} supplied "
                  "files could not be read, so this is a partial view of the "
                  f"evidence. First: {unparsed[0]['path']} — "
                  f"{unparsed[0]['reason']}")
    warned = [s["path"] for s in sources if s.get("result") == "WARNED"]
    if warned:
        return _u(f"{len(warned)} attestation(s) report result=WARNED, which is "
                  "neither pass nor fail. Folding it into either would be "
                  "inventing a judgement the producer declined to make: "
                  + ", ".join(warned[:3]))
    inconsistent = [s["path"] for s in sources if s.get("internally_inconsistent")]
    if inconsistent:
        detail = []
        for s in sources:
            detail.extend(s.get("inconsistent_reasons") or [])
        return _u("an attestation contradicts itself "
                  f"({'; '.join(detail[:2])}). Reported loudly and not "
                  "resolved, because silently preferring either field is a "
                  f"guess: {', '.join(inconsistent[:3])}")
    # LAW 3. A test that ran and failed is evidence about the claim; a harness
    # that could not run a test is ABSENCE of evidence. The two share a key in
    # the frozen totals dict and must not share a reading. Deciding this before
    # anything else changes no verdict — every red reading declines now — and
    # yields the reason a reader can act on: the runner broke, everything that
    # ran passed.
    if (outcome == "FAILED" and int(totals.get("failures") or 0) == 0
            and int(totals.get("errors") or 0) > 0):
        return _u(f"{totals['errors']} harness error(s) and zero test failures. "
                  "A collection failure from a missing optional dependency, a "
                  "module that would not import, or a jest file that would not "
                  "run is a broken runner, not a false claim. Everything that "
                  "ran, passed.")
    if outcome == "EMPTY":
        executed = int(resolved.get("executed") or 0)
        return _u("no tests were executed. "
                  + (f"{totals.get('tests', 0)} testcase records are present and "
                     f"{executed} of them executed — an all-skipped suite, an "
                     "all-DISABLED_ binary and a collection that found nothing "
                     "are shape-identical to a green run, and none of them is "
                     "evidence that anything passed."
                     if totals.get("tests") else
                     "no testcase or named test appears anywhere in the "
                     "evidence. An attestation whose summary string says "
                     "PASSED or FAILED while naming no tests at all carries no "
                     "test-level evidence in either direction."))
    if ev.get("sources_disagree"):
        return _u("two sources disagree about the same claim "
                  f"({', '.join(str(o) for o in ev.get('per_source_outcomes') or [])})"
                  ". A stale artifact left beside a fresh one, and GitHub's "
                  "'Re-run failed jobs' where both runs genuinely name the "
                  "same commit, both produce exactly this. Refusing beats "
                  "picking, and refusing beats a union.")
    if int(resolved.get("indeterminate") or 0) > 0:
        return _u(f"{resolved['indeterminate']} testcase(s) carry a rerun "
                  "element with no plain <failure>/<error> beside it, which "
                  "should not occur. Their outcome is not resolvable from these "
                  "bytes.")

    # ── the red reading, which is a REFUSAL and not an accusation ────────────
    # This is where the deleted branch used to be. What replaces it is a
    # decline. The counts that used to feed the accusation are still read, still
    # returned in ev["observed"], and still printed — they are simply not a
    # verdict. Nothing here inspects the binding first, because there is no
    # longer any conclusion for a binding to license.
    if outcome == "FAILED":
        obs = ev.get("observed") or {}
        return _u(
            f"the evidence in hand reads red — {obs.get('failing_tests', 0)} "
            f"failing test(s), {obs.get('harness_errors', 0)} harness error(s) "
            f"across {len(sources)} source(s) — and this module has no verdict "
            "for that. It reports what it read (see `observed`) and stops. "
            "There is no accusing verdict here: the branch is deleted rather "
            "than disabled, because its precision cannot be measured by a "
            "blind panel from the eleven events its gate fires on across "
            "1,775,765 changed files, and because that gate is a string "
            "compare on bytes the writer of the file chose. A reading is not a "
            "finding.")

    # ── the one conclusion ───────────────────────────────────────────────────
    if outcome == "PASSED":
        passed = int(resolved.get("passed") or 0)
        # VERIFIED requires that something actually ran AND passed. The EMPTY
        # branch above already covers zero-executed, and this is the same guard
        # stated where the affirmation is made, so a future reordering cannot
        # delete it by accident.
        if passed <= 0:
            return _u("the evidence resolves to no passing test. Zero passes is "
                      "not a green run under any reading; VERIFIED requires at "
                      "least one test that executed and passed.")

        # A commit was named and nothing in the evidence asserts it. Declining,
        # which is the safe direction: this can only withhold VERIFIED.
        want = _norm_commit(commit)
        bindings = binding_against_commit(ev, commit)
        if commit is not None and not all(
                b.get("commit_assertion_matches") for b in bindings):
            return _u(_unasserted_why(bindings, sources, commit))

        note = ("the evidence in hand says the tests passed. That is what the "
                "bytes assert, not something this module observed: no "
                "signature was checked, and nothing here shows the run was "
                "complete. VERIFIED means an attestation naming this commit "
                "reports PASSED — it does not mean the tests passed.")
        if commit is None:
            note += (" No commit was supplied, so this report is not tied to "
                     "any particular change.")
        else:
            note += (f" Every source carries a subject gitCommit digest equal "
                     f"to {want} — an ASSERTION in bytes the producer chose, "
                     "not a checked signature.")
        return "VERIFIED", (
            f"{passed} passed, 0 failed, 0 errored across "
            f"{len(sources)} source(s); {note}")

    return _u(f"outcome {outcome!r} is outside the decision table; refusing "
              "rather than guessing which way it leans.")


def _unasserted_why(bindings: list, sources: list, commit: str | None) -> str:
    """Name the missing channel specifically. A refusal that says what would fix
    it is an adoption ramp; a refusal that says 'unchecked' is a complaint."""
    kinds = {b.get("kind", "none") for b in bindings}
    formats = sorted({s.get("format") for s in sources})
    lead = ("the evidence does not assert the commit under review "
            f"({commit})")
    detail = []
    attestations = [s for s in sources if s.get("format") == "intoto-testresult"]
    if attestations and not any((s.get("binding") or {}).get("git_digest_keys_seen")
                                for s in attestations):
        detail.append("no attestation subject carries a git-shaped digest key "
                      "at all — a sha256 of a tarball says nothing about a "
                      "commit, and reading one as though it did would be the "
                      "string-shaped inference this lab retired")
    if "junit" in formats:
        detail.append("no JUnit dialect defines any field for a commit — "
                      "checked against Surefire's XSD, the Ant/Jenkins "
                      "reference, pytest, go-junit-report, jest-junit and "
                      "trx2junit — so a JUnit file can never carry one on its "
                      "own")
    if "asserted" in kinds:
        detail.append("a commit is asserted in the evidence but is written by "
                      "the same job that produced the report, which makes it a "
                      "restatement of the claim rather than a check on it "
                      "(ASSERTED-NOT-CHECKED)")
    if any(b.get("degraded_key") == "sha1" for b in bindings):
        detail.append("the commit appears under the ambiguous `sha1` key, which "
                      "equally denotes the SHA-1 of a file's bytes "
                      "(DEGRADED-KEY)")
    if any(b.get("report_identity_asserted") for b in bindings):
        detail.append("an attestation subject digest DOES match the report's "
                      "own bytes, which asserts report identity and is silent "
                      "about which commit the tests ran against")
    fix = ("the channel that would make this readable is an in-toto test-result "
           "attestation whose subject carries digest.gitCommit equal to the "
           "head commit — e.g. actions/attest over the test report. Even then "
           "no signature is checked here, so the answer would be an assertion "
           "restated, never a proof.")
    return lead + ". " + ("; ".join(detail) + ". " if detail else "") + fix


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_source(s: dict) -> list[str]:
    out = [f"  {s['format']:<18} {s['path']}"]
    out.append(f"     sha256 {s['sha256'][:16]}…")
    if s["format"] == "junit":
        out.append(
            f"     resolved from testcases: {s['passed']} passed · "
            f"{s['failures']} failed · {s['errors']} errored · "
            f"{s['skipped']} skipped · {s['notrun']} not-run"
            + (f" · {s['indeterminate']} indeterminate" if s["indeterminate"] else ""))
        # Printed side by side because that is what the report-only signal IS:
        # the figure the root claims, the figure the suites claim, and the
        # figure the testcases actually resolve to. `?` means the attribute was
        # absent, which is UNKNOWN and not zero — go-junit-report omits a count
        # that happens to be zero, pytest never writes one at all, and the bytes
        # cannot tell you which of those you are looking at.
        out.append("     root attributes as found (NOT trusted): "
                   + ", ".join(f"{k}={'?' if v is None else v}"
                               for k, v in s["root_as_found"].items()))
        out.append("     summed from child suites (NOT trusted): "
                   + ", ".join(f"{k}={'?' if v is None else v}"
                               for k, v in s["summed_children_as_found"].items()))
        if s["root_children_disagree"]:
            out.append("     root and children disagree on `tests` — REPORT "
                       "ONLY. At least three dialects emit mismatched counts "
                       "by design; this is not evidence of tampering.")
        out.append(f"     producer guess: {s['producer_guess']} "
                   f"({s['producer_confidence']})")
        doctype = (s.get("xml_notes") or {}).get("doctype_seen")
        if doctype:
            out.append("     DOCTYPE present and recorded; entity, "
                       "unparsed-entity and notation declarations are refused "
                       "at the declaration by this reader")
        if s["unrecognized_elements"]:
            out.append("     unrecognized elements (leading indicator of a new "
                       "dialect): " + ", ".join(s["unrecognized_elements"]))
    else:
        out.append(f"     predicate result as found: {s['result']}"
                   + (f" · named {s['passed']} passed / {s['warned']} warned / "
                      f"{s['failures']} failed"
                      if s["test_name_lists_present"]
                      else " · NO test-name lists (legal silence, zero "
                           "information — the summary string alone carries no "
                           "test-level evidence)"))
        out.append(f"     outcome from EXECUTED tests: {s['outcome']}")
        if s["internally_inconsistent"]:
            out.append("     INTERNALLY INCONSISTENT: "
                       + "; ".join(s.get("inconsistent_reasons") or [])
                       + ". Reported, not resolved.")
        out.append("     signature: NOT CHECKED "
                   f"({s['envelope'].get('signature_count', 0)} present) — an "
                   "empty, a forged and a valid signatures array are "
                   "indistinguishable to this reader")
    b = s.get("binding") or {}
    out.append(f"     binding (ASSERTION IN BYTES, not cryptographic): "
               f"{b.get('kind', 'none')}"
               + (f" commit={b['commit']}" if b.get("commit") else "")
               + (f" [DEGRADED-KEY {b['degraded_key']}]" if b.get("degraded_key") else ""))
    out.append(f"       commit assertion matches: "
               f"{bool(b.get('commit_assertion_matches'))} · report identity "
               f"asserted: {bool(b.get('report_identity_asserted'))} · "
               f"signature verified: False")
    return out


def main(argv=None) -> int:
    """CLI. Returns 0 for EVERY possible content of the evidence.

    Non-zero is reserved for USAGE errors: argparse's own exit 2 for bad
    arguments, and 2 here for a ``--json`` path that cannot be written. An exit
    code that moved with the evidence would be an accusing verdict wearing a
    different hat — a caller would gate on it, and the gate would carry a
    precision nobody has measured.
    """
    ap = argparse.ArgumentParser(
        prog="styxx.evidence",
        description="Read a 'tests pass' claim from test-report BYTES. A pure "
                    "function: same bytes, same verdict, forever. The "
                    "vocabulary is VERIFIED and UNCHECKABLE; there is no "
                    "accusing verdict and the exit code is 0 whatever the "
                    "evidence says. UNCHECKABLE is a first-class answer here "
                    "and is printed loudly.")
    # nargs="*" and not "+": being handed no evidence at all is the single most
    # common real condition this module will meet, and it has a verdict —
    # UNCHECKABLE, because an unattested commit is unattested. Letting argparse
    # exit 2 with a usage error would convert the module's most important
    # answer into a reason to think the tool is broken.
    ap.add_argument("paths", nargs="*",
                    help="JUnit XML and/or in-toto test-result attestations. "
                         "Format is detected from content, not extension. With "
                         "none supplied the answer is UNCHECKABLE, printed like "
                         "any other.")
    ap.add_argument("--commit", default=None,
                    help="the commit under review, full 40- or 64-hex. "
                         "Recorded as ASSERTED-NOT-CHECKED unless an in-toto "
                         "subject gitCommit digest in the evidence equals it — "
                         "and even then no signature is checked.")
    ap.add_argument("--json", dest="out", default=None,
                    help="write the full evidence record to this path")
    a = ap.parse_args(argv)

    ev = load_evidence(a.paths)
    verdict, why = adjudicate_tests_pass(ev, a.commit)

    # Re-derive each source's binding against the commit named here, so that
    # what the printout shows is the same comparison the verdict used.
    want = _norm_commit(a.commit)
    ev["binding_per_source"] = binding_against_commit(ev, a.commit)
    for s, b in zip(ev["sources"], ev["binding_per_source"]):
        s["binding"] = {k: v for k, v in b.items()
                        if k not in ("path", "format")}
    ev["commit_under_review"] = a.commit
    ev["commit_under_review_normalized"] = want
    ev["verdict"] = verdict
    ev["why"] = why
    ev["purity"] = selfcheck_purity()
    ev["no_accusation"] = selfcheck_no_accusation()

    if a.out:
        try:
            Path(a.out).write_text(json.dumps(ev, indent=1) + "\n",
                                   encoding="utf-8")
        except OSError as exc:
            # A USAGE error — the operator named an unwritable path — and the
            # only kind of thing this CLI is allowed to exit non-zero for.
            print(f"usage error: could not write --json {a.out}: {exc}",
                  file=sys.stderr)
            return 2

    t, r = ev["totals"], ev["resolved"]
    print(f"{SPEC} — a pure function of the bytes at {len(ev['paths_requested'])} path(s)")
    pur = ev["purity"]
    print("purity self-check: "
          + ("PASS — " + pur["reason"] if pur["ok"]
             else f"ATTENTION — {pur['reason']} {pur['imported']} {pur['referenced']}"))
    acc = ev["no_accusation"]
    print("no-accusation self-check: "
          + ("PASS — " + acc["reason"] if acc["ok"]
             else f"ATTENTION — {acc['reason']} {acc['code_occurrences']}"))
    print(f"vocabulary: {' | '.join(VERDICTS)}")
    print()
    print(f"sources read: {len(ev['sources'])}")
    for s in ev["sources"]:
        for line in _fmt_source(s):
            print(line)

    if ev["unparsed"]:
        print(f"\nunparsed ({len(ev['unparsed'])}) — recorded, never silently dropped:")
        for u in ev["unparsed"]:
            print(f"  {u['path']}\n     {u['reason']}")

    print(f"\ntotals (unioned resolved testcases, never summed roots): "
          f"{r.get('passed', 0)} passed · {t.get('failures', 0)} failed · "
          f"{t.get('errors', 0)} errored · {t.get('skipped', 0)} skipped · "
          f"{r.get('notrun', 0)} not-run")
    print(f"outcome: {ev['outcome']}")

    obs = ev["observed"]
    print(f"\nobserved (REPORT ONLY — not a verdict, nothing may gate on it): "
          f"{obs['failing_tests']} failing test(s) · "
          f"{obs['harness_errors']} harness error(s) · "
          f"{obs['skipped_tests']} skipped · {obs['not_run_tests']} not-run")
    if obs["sources_reporting_failure"]:
        for p in obs["sources_reporting_failure"]:
            print(f"  reads red: {p}")
    print(f"  {obs['note']}")

    print(f"\nVERDICT: {verdict}")
    print(f"  {why}")

    print("\nwhat this module did NOT check:")
    for item in ev["not_checked"]:
        print(f"  - {item}")
    print(f"\nboundary: {ev['boundary']}")

    # Law 2. The exit code is a constant with respect to the evidence. There is
    # no content of any input file that makes this return non-zero, because a
    # caller would gate CI on it and that gate would be the accusing verdict
    # this module does not ship, reintroduced through the back door.
    return 0


if __name__ == "__main__":
    sys.exit(main())
