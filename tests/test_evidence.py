"""styxx.evidence — the RE-FROZEN contract v0.2, where the accusing branch is gone.

    VERDICTS = ("VERIFIED", "UNCHECKABLE")

Written in parallel with the implementation, against the re-frozen contract only,
and RED ON ARRIVAL by construction: every load-bearing test below was watched
failing against the shipped module of 2026-09-01, which still carried all seven
defects E-1..E-7. That is deliberate. The suite this one replaces was GREEN ON
ARRIVAL, survived four of ten mutants, and one of its load-bearing tests asserted
a guarantee that was simply false — its docstring claimed ``main()`` has no
commit argument while ``main()`` defines ``--commit``, and it passed only because
it never tried. A test that has never been seen to fail is a decoration.

WHAT MAKES A TEST HERE LOAD-BEARING

  * ``test_the_accusing_branch_is_absent_from_the_source_not_merely_unreached``
    — THE structural one, and the exact inverse of the retired
    ``test_a_bound_failing_report_does_contradict``. That test's docstring read:
    "the module must still work; if the only reachable verdict is UNCHECKABLE
    the badge never changes colour and every other badge on the page is worth
    less." The inverse, now the standing position of this lab: a badge that can
    change colour by accusing someone is worth LESS than a badge that cannot,
    because the accusing branch fires 11 times in 1,775,765 changed files and a
    100-item blind panel cannot be assembled from eleven events. Its precision
    is not unmeasured, it is STRUCTURALLY UNMEASURABLE, and a verdict whose
    precision can never be measured is not shippable at any price. This test
    asserts the ABSENCE OF THE BRANCH. A test that only checked outputs would
    pass on a module carrying the branch behind a condition nobody reached, and
    a branch nobody reached today is a branch someone reaches after a refactor.
  * ``test_no_configuration_flag_governs_the_verdict_vocabulary``. A flag is
    what a maintainer who did not read the paper flips. ``WITHHOLD_PATH_ACCUSATION``
    already exists in this repository as exactly such a flag. Absence of a branch
    is a stronger guarantee than a flag set to off.
  * The ``E-1``..``E-7`` family. Seven confirmed defects with live repros, each
    one an accusation the shipped module would have made against somebody's pull
    request for the cost of a hand-written JSON file.
  * ``test_main_exits_zero_for_every_evidence_content``. Evidence CONTENT may
    never move an exit code. Non-zero is reserved for usage errors.
  * The purity and determinism families. Same bytes, same verdict, forever — the
    only reason a capsule can re-derive anything this module says.

THE ASYMMETRY, because it is the whole design and a reader will ask

VERIFIED survives and CONTRADICTED does not, and that is not squeamishness. A
forged VERIFIED merely repeats a claim the author already made in prose. A forged
CONTRADICTED is an attack on someone else's pull request. VERIFIED here never
means "the tests passed"; it means "an attestation naming this commit reports
PASSED, and no signature was checked."
"""
from __future__ import annotations

import ast
import base64
import hashlib
import inspect
import io
import json
import re
import tokenize

import pytest

import styxx.evidence as E
from styxx.evidence import (adjudicate_tests_pass, binding_against_commit,
                            load_evidence, main)

VERDICTS = {"VERIFIED", "UNCHECKABLE"}

HEAD_COMMIT = "9a04d1ee393b5be2773b1ce204f61fe0fd02366a"
# The refs/pull/N/merge sha, not the head. Handing this instead of head.sha is
# the single most likely integration bug around this module.
MERGE_COMMIT = "04f61fe0fd02366a9a04d1ee393b5be2773b1ce2"
# A commit that is not the one under review. Used as the decoy subject in the
# E-2 permutation, where it sits in front of the real one.
DECOY_COMMIT = "1111111111111111111111111111111111111111"


# --------------------------------------------------------------------- helpers

def w(tmp_path, name, text):
    """Write a fixture with LF endings and return its path as a str."""
    p = tmp_path / name
    p.write_bytes(text.encode("utf-8"))
    return str(p)


def adj(paths, commit=None):
    """load + adjudicate, the way a caller actually uses the module."""
    return adjudicate_tests_pass(load_evidence(list(paths)), commit)


def verdict(paths, commit=None):
    return adj(paths, commit)[0]


def statement(result, subjects=None, commit=HEAD_COMMIT, digest_key="gitCommit",
              passed=(), warned=(), failed=(), predicate_type=None):
    """A bare in-toto Statement carrying the test-result predicate.

    `subjects` overrides the single-subject default, which is what the E-2
    permutation needs: in-toto matches subjects PURELY BY DIGEST, so an array of
    two is not an ordered preference list.
    """
    if subjects is None:
        if commit is None:
            # A build-artifact attestation: a real digest, no git-shaped key.
            subjects = [{"name": "junit.xml", "digest": {"sha256": "a" * 64}}]
        else:
            subjects = [{"name": "_", "digest": {digest_key: commit}}]
    predicate = {"result": result, "configuration": []}
    if passed:
        predicate["passedTests"] = list(passed)
    if warned:
        predicate["warnedTests"] = list(warned)
    if failed:
        predicate["failedTests"] = list(failed)
    return json.dumps({
        "_type": "https://in-toto.io/Statement/v1",
        "subject": subjects,
        "predicateType": (predicate_type
                          or "https://in-toto.io/attestation/test-result/v0.1"),
        "predicate": predicate,
    }, indent=2) + "\n"


def dsse(statement_text, signatures=None,
         payload_type="application/vnd.in-toto+json"):
    """Wrap a Statement in a DSSE envelope.

    Nothing here is signed. `signatures` is handed in verbatim so E-3 can post
    the three forgeries that cost one text editor: a signature that is base64 of
    the ASCII text "not-a-signature", an empty array, and a payloadType nobody
    ever registered.
    """
    if signatures is None:
        signatures = [{"keyid": "unverified",
                       "sig": base64.b64encode(b"nope").decode("ascii")}]
    return json.dumps({
        "payloadType": payload_type,
        "payload": base64.b64encode(statement_text.encode("utf-8")).decode("ascii"),
        "signatures": signatures,
    }, indent=2) + "\n"


# -------------------------------------------------------------- JUnit fixtures

PYTEST_GREEN = """<?xml version="1.0" encoding="utf-8"?>
<testsuites name="pytest tests"><testsuite name="pytest" errors="0" failures="0" \
skipped="0" tests="2" time="0.012" timestamp="2026-09-01T00:00:00.000000" hostname="runner">
<testcase classname="tests.test_app" name="test_one" time="0.001" />
<testcase classname="tests.test_app" name="test_two" time="0.001" />
</testsuite></testsuites>
"""

PYTEST_RED = """<?xml version="1.0" encoding="utf-8"?>
<testsuites name="pytest tests"><testsuite name="pytest" errors="0" failures="1" \
skipped="0" tests="2" time="0.031" timestamp="2026-09-01T00:00:00.000000" hostname="runner">
<testcase classname="tests.test_app" name="test_one" time="0.001" />
<testcase classname="tests.test_app" name="test_two" time="0.002">
<failure message="assert 1 == 2">E       assert 1 == 2</failure>
</testcase>
</testsuite></testsuites>
"""

# The same red report with the commit written into <properties>. This key is a
# community convention, not a schema field, and it is self-authored by the very
# job under adjudication.
PYTEST_RED_SELF_ASSERTED_COMMIT = """<?xml version="1.0" encoding="utf-8"?>
<testsuites name="pytest tests"><testsuite name="pytest" errors="0" failures="1" \
skipped="0" tests="2" time="0.031" timestamp="2026-09-01T00:00:00.000000" hostname="runner">
<properties>
<property name="commit" value="%s" />
<property name="GIT_COMMIT" value="%s" />
</properties>
<testcase classname="tests.test_app" name="test_one" time="0.001" />
<testcase classname="tests.test_app" name="test_two" time="0.002">
<failure message="assert 1 == 2">E       assert 1 == 2</failure>
</testcase>
</testsuite></testsuites>
""" % (HEAD_COMMIT, HEAD_COMMIT)

# Collected nothing. Byte-identical in shape to a healthy green run.
PYTEST_EMPTY = """<?xml version="1.0" encoding="utf-8"?>
<testsuites name="pytest tests"><testsuite name="pytest" errors="0" failures="0" \
skipped="0" tests="0" time="0.001" timestamp="2026-09-01T00:00:00.000000" hostname="runner" />
</testsuites>
"""

# An unimportable module. tests="1" errors="1", and zero test BODIES ran.
PYTEST_COLLECTION_ERROR = """<?xml version="1.0" encoding="utf-8"?>
<testsuites name="pytest tests"><testsuite name="pytest" errors="1" failures="0" \
skipped="0" tests="1" time="0.004" timestamp="2026-09-01T00:00:00.000000" hostname="runner">
<testcase classname="" name="tests/test_integration.py" time="0.0">
<error message="collection failure">ImportError: No module named 'optional_dep'</error>
</testcase>
</testsuite></testsuites>
"""

# Integration tests guarded behind an env var that CI does not set.
PYTEST_ALL_SKIPPED = """<?xml version="1.0" encoding="utf-8"?>
<testsuites name="pytest tests"><testsuite name="pytest" errors="0" failures="0" \
skipped="3" tests="3" time="0.003" timestamp="2026-09-01T00:00:00.000000" hostname="runner">
<testcase classname="tests.test_db" name="test_a" time="0.0">
<skipped type="pytest.skip" message="needs DATABASE_URL">needs DATABASE_URL</skipped></testcase>
<testcase classname="tests.test_db" name="test_b" time="0.0">
<skipped type="pytest.skip" message="needs DATABASE_URL">needs DATABASE_URL</skipped></testcase>
<testcase classname="tests.test_db" name="test_c" time="0.0">
<skipped type="pytest.skip" message="needs DATABASE_URL">needs DATABASE_URL</skipped></testcase>
</testsuite></testsuites>
"""

# googletest marks a never-executed DISABLED_ test status="notrun"
# result="suppressed" with NO <skipped> child. Element-only rules render an
# all-disabled binary as a full green.
GTEST_ALL_DISABLED = """<?xml version="1.0" encoding="UTF-8"?>
<testsuites tests="2" failures="0" disabled="2" errors="0" time="0.001" name="AllTests">
<testsuite name="Disabled" tests="2" failures="0" disabled="2" errors="0" time="0.001">
<testcase name="DISABLED_a" status="notrun" result="suppressed" classname="Disabled" time="0" />
<testcase name="DISABLED_b" status="notrun" result="suppressed" classname="Disabled" time="0" />
</testsuite>
</testsuites>
"""

# Dialect two: Surefire's own XSD declares <testsuite> as the ROOT element.
SUREFIRE_RED = """<?xml version="1.0" encoding="UTF-8"?>
<testsuite xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" \
name="org.example.FailingTest" time="0.418" tests="2" errors="0" skipped="0" failures="1">
<properties>
<property name="surefire.version" value="3.2.5" />
<property name="java.version" value="17.0.9" />
</properties>
<testcase name="passes" classname="org.example.FailingTest" time="0.01" />
<testcase name="breaks" classname="org.example.FailingTest" time="0.02">
<failure message="expected:&lt;1&gt; but was:&lt;2&gt;" type="java.lang.AssertionError">\
org.junit.ComparisonFailure</failure>
</testcase>
</testsuite>
"""

# A test that failed, was rerun, and PASSED. Surefire's own documentation:
# "existing consumers will still consider it as a passing test."
SUREFIRE_FLAKY_GREEN = """<?xml version="1.0" encoding="UTF-8"?>
<testsuite xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" \
name="org.example.FlakyTest" time="0.902" tests="1" errors="0" skipped="0" failures="0">
<properties><property name="surefire.version" value="3.2.5" /></properties>
<testcase name="eventuallyPasses" classname="org.example.FlakyTest" time="0.9">
<flakyFailure message="expected:&lt;1&gt; but was:&lt;2&gt;" type="java.lang.AssertionError">\
first attempt</flakyFailure>
</testcase>
</testsuite>
"""

# googletest EXPECTED_NO_TEST_XML in shape: the root asserts zero tests and zero
# failures over a suite containing a real <failure>. Its own golden file.
GTEST_ROOT_CLAIMS_GREEN = """<?xml version="1.0" encoding="UTF-8"?>
<testsuites tests="0" failures="0" disabled="0" errors="0" time="0.007" \
timestamp="2026-09-01T00:00:00Z" name="AllTests">
<testsuite name="NonTestSuiteFailure" tests="1" failures="1" skipped="0" errors="0" time="0.001">
<testcase name="" status="run" result="completed" classname="" time="0.001">
<failure message="Expected equality of these values" type="">gtest_xml_output.cc:0</failure>
</testcase>
</testsuite>
</testsuites>
"""

# The inverse: a count with no test behind it.
ROOT_OVERCOUNTS_FAILURES = """<?xml version="1.0" encoding="utf-8"?>
<testsuites name="pytest tests"><testsuite name="pytest" errors="0" failures="2" \
skipped="0" tests="2" time="0.012">
<testcase classname="tests.test_app" name="test_one" time="0.001" />
<testcase classname="tests.test_app" name="test_two" time="0.001" />
</testsuite></testsuites>
"""

# E-4. A perfectly ordinary GREEN run from a project with snapshot tests. The
# captured HTML lands in <system-out> and carries the four characters the
# shipped module substring-scanned for in the first 8192 bytes. The report
# poisons itself, and its author is the person the verdict is about.
PYTEST_GREEN_HTML_IN_SYSTEM_OUT = """<?xml version="1.0" encoding="utf-8"?>
<testsuites name="pytest tests"><testsuite name="pytest" errors="0" failures="0" \
skipped="0" tests="2" time="0.030">
<testcase classname="tests.test_render" name="test_page_snapshot" time="0.02">
<system-out><![CDATA[<!DOCTYPE html>
<html><head><title>ok</title></head><body><p>rendered</p></body></html>]]></system-out>
</testcase>
<testcase classname="tests.test_render" name="test_other" time="0.01" />
</testsuite></testsuites>
"""

# E-5. The same class of document the guard PROMISES to refuse, hidden behind a
# 9 KB XML comment so that the DOCTYPE falls outside an 8192-character window.
# The billion-laughs case was stopped by libexpat's own amplification limit; the
# guard contributed nothing, and its docstring said otherwise.
ENTITY_BEHIND_A_LONG_COMMENT = (
    '<?xml version="1.0"?>'
    + "<!--" + ("padding " * 1200) + "-->"
    + '<!DOCTYPE testsuites [<!ENTITY smuggled "expanded-by-the-parser">]>'
    + '<testsuites name="pytest tests"><testsuite name="pytest" tests="1">'
    + '<testcase classname="t" name="&smuggled;" time="0.0" />'
    + "</testsuite></testsuites>\n"
)

# E-6. 1000 nested <testsuites> in 25 KB. Per-level recursion in the suite walker
# raises RecursionError, and RecursionError does not inherit from Exception's
# usual parse-error family, so it escapes the reader entirely.
_NEST = 1000
DEEPLY_NESTED_SUITES = (
    '<?xml version="1.0"?>'
    + "<testsuites>" * _NEST
    + '<testsuite name="deep" tests="1"><testcase classname="t" name="one" /></testsuite>'
    + "</testsuites>" * _NEST
    + "\n"
)

NOT_XML = "this file is a build log, not a test report\ntraceback follows\n"
XML_BUT_NOT_JUNIT = '<?xml version="1.0"?>\n<html><body><p>404</p></body></html>\n'


ALL_FIXTURES = [
    ("pytest-green", PYTEST_GREEN),
    ("pytest-red", PYTEST_RED),
    ("pytest-red-self-asserted-commit", PYTEST_RED_SELF_ASSERTED_COMMIT),
    ("pytest-empty", PYTEST_EMPTY),
    ("pytest-collect-error", PYTEST_COLLECTION_ERROR),
    ("pytest-all-skipped", PYTEST_ALL_SKIPPED),
    ("gtest-all-disabled", GTEST_ALL_DISABLED),
    ("surefire-red", SUREFIRE_RED),
    ("surefire-flaky", SUREFIRE_FLAKY_GREEN),
    ("gtest-root-lies", GTEST_ROOT_CLAIMS_GREEN),
    ("root-overcounts", ROOT_OVERCOUNTS_FAILURES),
    ("green-html-in-system-out", PYTEST_GREEN_HTML_IN_SYSTEM_OUT),
    ("entity-behind-a-comment", ENTITY_BEHIND_A_LONG_COMMENT),
    ("deeply-nested", DEEPLY_NESTED_SUITES),
    ("not-xml", NOT_XML),
    ("not-junit", XML_BUT_NOT_JUNIT),
]
_FIXTURE_IDS = [n for n, _ in ALL_FIXTURES]


# ═════════════════════════════════════════════════════════════════════════════
# THE STRUCTURAL ONE — the branch is gone, not merely unreached
# ═════════════════════════════════════════════════════════════════════════════

SRC = inspect.getsource(E)
SRC_LINES = SRC.splitlines()
TREE = ast.parse(SRC)
ACCUSER = "CONTRA" + "DICTED"   # spelled in halves so this file's own scan of
                                # itself, should anyone ever write one, is honest


def _docstring_nodes():
    """Every string Constant that is a module/class/function docstring."""
    out = {}
    for node in ast.walk(TREE):
        body = getattr(node, "body", None)
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)) or not body:
            continue
        first = body[0]
        if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            out[id(first.value)] = first.value
    return out


_DOCSTRINGS = _docstring_nodes()


def _docstring_lines():
    lines = set()
    for node in _DOCSTRINGS.values():
        lines.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    return lines


def _comment_lines():
    lines = set()
    for tok in tokenize.generate_tokens(io.StringIO(SRC).readline):
        if tok.type == tokenize.COMMENT:
            lines.add(tok.start[0])
    return lines


PROSE_LINES = _docstring_lines() | _comment_lines()


def test_the_accusing_branch_is_absent_from_the_source_not_merely_unreached():
    """THE LOAD-BEARING ONE, and the exact inverse of the retired
    ``test_a_bound_failing_report_does_contradict``.

    That test said: the module must still work, and if the only reachable
    verdict is UNCHECKABLE the badge never changes colour and every other badge
    on the page is worth less. The re-frozen position is the opposite. A badge
    that changes colour by accusing a stranger's pull request is worth less than
    one that cannot, because the bound branch — the only branch ever permitted
    to accuse — fires 11 times across 1,775,765 changed files in this lab's
    corpus. A 100-item blind panel cannot be built from eleven events, so the
    accusing verdict's precision is not merely unmeasured; it is STRUCTURALLY
    UNMEASURABLE, and E-3 showed the binding meant to gate it was a string
    compare on attacker-writable bytes.

    This asserts the ABSENCE OF THE BRANCH rather than the absence of the
    output. A test that only checked verdicts would pass on a module carrying
    the branch behind a guard nobody currently reaches — and a branch nobody
    reaches today is a branch a refactor reaches tomorrow.
    """
    offenders = [(i + 1, line.strip())
                 for i, line in enumerate(SRC_LINES)
                 if ACCUSER in line and (i + 1) not in PROSE_LINES]
    assert not offenders, (
        "styxx/evidence.py carries the accusing verdict in EXECUTABLE code at "
        + "; ".join(f"line {n}: {t}" for n, t in offenders)
        + ". The re-frozen vocabulary is two words. Absence of the branch is "
          "the guarantee; an unreached branch is not.")


def test_the_accusing_word_appears_only_inside_comments_and_docstrings():
    """The same fact stated over the AST rather than over lines, because a
    string constant is not a comment however it is indented."""
    for node in ast.walk(TREE):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        if id(node) in _DOCSTRINGS:
            continue
        assert ACCUSER not in node.value, (
            f"line {node.lineno}: a live string constant carries the accusing "
            "verdict. Only prose may name it.")
    names = {n.id for n in ast.walk(TREE) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(TREE) if isinstance(n, ast.Attribute)}
    assert not [n for n in names if ACCUSER in n.upper()], \
        "an identifier is named after the retired verdict"


def test_the_verdict_vocabulary_is_two_words():
    assert tuple(E.VERDICTS) == ("VERIFIED", "UNCHECKABLE"), E.VERDICTS
    assert ACCUSER not in set(E.VERDICTS)


def test_the_adjudicator_returns_nothing_outside_the_two_word_vocabulary():
    """Structural, over ``adjudicate_tests_pass`` only: every ALL-CAPS string it
    can return must be one of the two. This catches a third verdict smuggled in
    under a different spelling, which output-only tests would never see unless
    someone happened to write the fixture that reaches it."""
    fn = next((n for n in ast.walk(TREE)
               if isinstance(n, ast.FunctionDef) and n.name == "adjudicate_tests_pass"),
              None)
    assert fn is not None, "adjudicate_tests_pass is gone; this suite is blind"

    returned = set()
    for node in ast.walk(fn):
        if not isinstance(node, ast.Return) or node.value is None:
            continue
        candidates = (node.value.elts if isinstance(node.value, ast.Tuple)
                      else [node.value])
        for c in candidates:
            if isinstance(c, ast.Constant) and isinstance(c.value, str) \
                    and re.fullmatch(r"[A-Z][A-Z_]+", c.value):
                returned.add(c.value)
    assert returned <= VERDICTS, f"returns verdict-shaped strings {sorted(returned - VERDICTS)}"


_FLAG_SHAPED = re.compile(r"WITHHOLD|ACCUS|SUPPRESS|ENABLE|DISABLE|REPORT_ONLY_MODE",
                          re.IGNORECASE)


def _module_level_assigned_names():
    names = []
    for node in TREE.body:
        targets = ([node.target] if isinstance(node, ast.AnnAssign)
                   else node.targets if isinstance(node, ast.Assign) else [])
        for t in targets:
            if isinstance(t, ast.Name):
                names.append((t.id, node))
    return names


def test_no_configuration_flag_governs_the_verdict_vocabulary():
    """LOAD-BEARING. A flag is what a maintainer who did not read the paper
    flips. ``WITHHOLD_PATH_ACCUSATION`` exists in this repository as exactly such
    a flag, which is the argument against having one here: absence of a branch is
    a stronger guarantee than a flag set to off.

    Asserted over module-level BINDINGS rather than over the source text on
    purpose. A line scan for the word "accuse" reads the module's own denial —
    "it cannot accuse: the accusing verdict is absent from its vocabulary" — as
    the violation, which is precisely the false-positive shape this lab retired
    an instrument for."""
    offenders = [name for name, _ in _module_level_assigned_names()
                 if _FLAG_SHAPED.search(name)]
    assert not offenders, f"a switch is bound at module scope: {offenders}"

    # And no module-level boolean constant is consulted as a condition inside
    # the adjudicator, whatever it happens to be called.
    bools = {name for name, node in _module_level_assigned_names()
             if isinstance(getattr(node, "value", None), ast.Constant)
             and isinstance(node.value.value, bool)}
    fn = next((n for n in ast.walk(TREE)
               if isinstance(n, ast.FunctionDef) and n.name == "adjudicate_tests_pass"),
              None)
    assert fn is not None
    consulted = {n.id for node in ast.walk(fn) if isinstance(node, ast.If)
                 for n in ast.walk(node.test) if isinstance(n, ast.Name)}
    assert not (bools & consulted), \
        f"the verdict is conditioned on the module-level switch {sorted(bools & consulted)}"


def test_the_docstring_states_the_reasoning_and_promises_no_future_measurement():
    """The re-frozen contract requires the WHY in the module docstring, and
    forbids the phrasing a future maintainer reads as an invitation: report-only
    is designed as PERMANENT, not as "pending measurement".

    The promise is matched as a PHRASE and only outside a negation. The module is
    expected to state the prohibition in its own prose — it says report-only is
    not "pending measurement" — and a bare substring test would read that denial
    as the breach and fail a correct module."""
    doc = E.__doc__ or ""
    lowered = doc.lower()
    assert lowered.strip(), "the module has no docstring"
    assert "unmeasurable" in lowered, \
        "the docstring must say the accusation's precision is structurally unmeasurable"
    assert "blind panel" in lowered, \
        "the docstring must name the instrument that cannot be assembled"
    assert "permanent" in lowered, \
        "the docstring must say report-only is permanent, not a waypoint"

    for phrase in ("pending measurement", "pending a panel", "pending precision",
                   "pending repair", "until precision"):
        for m in re.finditer(re.escape(phrase), lowered):
            window = lowered[max(0, m.start() - 48):m.start()]
            assert any(neg in window for neg in ("not ", "never", "no ", "nothing")), \
                ("the docstring promises a future measurement: ..."
                 + doc[max(0, m.start() - 48):m.end() + 40])


def test_the_docstring_states_the_asymmetry_and_what_verified_does_not_mean():
    """VERIFIED must never be readable as "the tests passed". The asymmetry is
    deliberate and the contract requires it stated: a forged VERIFIED repeats a
    claim the author already made in prose, a forged accusation attacks someone
    else's pull request."""
    doc = (E.__doc__ or "").lower()
    assert "no signature was checked" in doc or "signature is not checked" in doc, \
        "the docstring must say VERIFIED comes with no signature check"


# ═════════════════════════════════════════════════════════════════════════════
# E-1 .. E-7 — one live repro each, all seven confirmed against the 2026-09-01
#              module before this file was written
# ═════════════════════════════════════════════════════════════════════════════

def test_e1_a_bare_failed_result_naming_zero_tests_is_empty_not_a_reading(tmp_path):
    """E-1. The shipped in-toto leg set its outcome from the bare `result`
    string with no test-level evidence, so an attestation with no lists at all
    produced totals.tests == 0 AND outcome == FAILED simultaneously — a
    self-contradiction against the contract's own definition of EMPTY. The
    errors-only refusal could not catch it because that leg hardcoded errors: 0.

    Zero executed tests is EMPTY. It was EMPTY when it was green and it is EMPTY
    now."""
    p = w(tmp_path, "attestation.intoto.json", statement("FAILED"))
    ev = load_evidence([p])
    src = ev["sources"][0]
    assert src["outcome"] == "EMPTY", (
        "an attestation naming no test at all reports outcome "
        f"{src['outcome']!r} while totals say {ev['totals']}")
    assert ev["totals"]["tests"] == 0
    v, why = adj([p], HEAD_COMMIT)
    assert v == "UNCHECKABLE", why


def test_e1_the_intoto_leg_does_not_hardcode_zero_errors(tmp_path):
    """The other half of E-1: `errors: 0` was a constant on that leg, which is
    why the "harness broke, that is absence" refusal could never fire there."""
    p = w(tmp_path, "attestation.intoto.json", statement("FAILED"))
    src = load_evidence([p])["sources"][0]
    assert src.get("executed", 0) == 0, \
        "an attestation naming no test executed nothing; the record should say so"


def test_e2_permuting_the_subject_array_changes_nothing(tmp_path):
    """E-2. LOAD-BEARING. The shipped module latched the FIRST gitCommit it saw
    and the function deciding the binding only ever read that latched value, so
    the multi-subject scan was dead code. in-toto is explicit: subjects match
    PURELY BY DIGEST. Identical digest sets in a different order gave different
    verdicts, which means the verdict was a function of JSON array order — and
    array order is chosen by whoever wrote the file."""
    subs = [{"name": "decoy", "digest": {"gitCommit": DECOY_COMMIT}},
            {"name": "head", "digest": {"gitCommit": HEAD_COMMIT}}]
    a = w(tmp_path, "a.intoto.json",
          statement("PASSED", subjects=list(subs), passed=["t::one"]))
    b = w(tmp_path, "b.intoto.json",
          statement("PASSED", subjects=list(reversed(subs)), passed=["t::one"]))

    va, wa = adj([a], HEAD_COMMIT)
    vb, wb = adj([b], HEAD_COMMIT)
    assert va == vb, (
        f"subject order flipped the verdict: {va} vs {vb}. in-toto matches "
        "subjects purely by digest; array order is the file author's choice.")
    assert va == "VERIFIED", wa
    assert wa == wb


def test_e2_the_per_source_binding_is_order_invariant(tmp_path):
    """The same fact one level down, so a future implementation cannot restore
    order-dependence in the binding while papering over it in the verdict.

    Read from ``binding_against_commit`` rather than from the load-time record.
    The load-time binding has no comparison target, so its
    ``commit_binding_verified`` is a constant ``False`` and comparing it across
    two orderings compares nothing. The comparison the verdict uses is the one
    that has to be order-invariant."""
    subs = [{"name": "decoy", "digest": {"gitCommit": DECOY_COMMIT}},
            {"name": "head", "digest": {"gitCommit": HEAD_COMMIT}}]

    def binding_of(name, ordering):
        p = w(tmp_path, name, statement("PASSED", subjects=ordering,
                                        passed=["t::one"]))
        ev = load_evidence([p])
        b = binding_against_commit(ev, HEAD_COMMIT)[0]
        return {"kind": b.get("kind"),
                "commit": b.get("commit"),
                "commit_binding_verified": b.get("commit_binding_verified"),
                "commit_assertion_matches": b.get("commit_assertion_matches"),
                "report_identity_verified": b.get("report_identity_verified"),
                "keys": sorted(b.get("git_digest_keys_seen") or [])}

    forward = binding_of("f.intoto.json", list(subs))
    reverse = binding_of("r.intoto.json", list(reversed(subs)))
    assert forward == reverse, (
        "subject order changed the binding: in-toto matches subjects purely by "
        f"digest, and array order is the file author's choice. {forward} vs "
        f"{reverse}")
    assert forward["commit_assertion_matches"] is True, \
        "the head commit is among the subjects and the comparison missed it"


@pytest.mark.parametrize("label,signatures,payload_type", [
    ("sig-is-base64-of-ascii-text",
     [{"keyid": "k", "sig": base64.b64encode(b"not-a-signature").decode("ascii")}],
     "application/vnd.in-toto+json"),
    ("no-signatures-at-all", [], "application/vnd.in-toto+json"),
    ("payload-type-nobody-registered",
     [{"keyid": "k", "sig": "AAAA"}],
     "application/vnd.in-toto.totally-made-up+json"),
])
def test_e3_a_hand_written_forgery_cannot_produce_a_reading_against_a_commit(
        tmp_path, label, signatures, payload_type):
    """E-3. LOAD-BEARING. "Cryptographically bound" was never implemented.
    ``binding.verified = True`` meant only "this file contains the string I was
    handed", so all three of these forgeries — each one a text editor and thirty
    seconds — reached the accusing verdict against any commit their author cared
    to name. That is the entire reason the branch is gone rather than gated: no
    gate built on attacker-writable bytes can hold.

    What must hold now is that a red forgery bound to the head commit produces
    nothing an automated caller can act on, and that the record never claims a
    signature was checked."""
    env = dsse(statement("FAILED", failed=["tests.test_app::test_two"]),
               signatures=signatures, payload_type=payload_type)
    p = w(tmp_path, "forged.intoto.json", env)
    ev = load_evidence([p])
    v, why = adjudicate_tests_pass(ev, HEAD_COMMIT)
    assert v in VERDICTS, v
    assert v != "VERIFIED", f"a FAILED forgery certified green: {why}"
    assert ACCUSER not in json.dumps(ev) + why, \
        f"{label}: the forgery reached an accusing reading"


@pytest.mark.parametrize("signatures", [
    [{"keyid": "k", "sig": base64.b64encode(b"not-a-signature").decode("ascii")}],
    [],
])
def test_e3_the_record_says_the_signature_was_not_checked(tmp_path, signatures):
    """No key material is consulted and no trust root is pinned. in-toto's own
    words: to obtain AUTHENTICATED predicate information a consumer must verify
    the payload against its signatures. This module parses. It does not verify,
    and it must say so rather than let a reader assume."""
    p = w(tmp_path, "a.intoto.json",
          dsse(statement("PASSED", passed=["t::one"]), signatures=signatures))
    src = load_evidence([p])["sources"][0]
    env = src.get("envelope") or {}
    assert env.get("signature_checked") is False, \
        "the record does not state that the signature went unchecked"
    assert env.get("signature_count") == len(signatures)


def test_e3_no_code_path_ever_claims_a_signature_was_checked():
    """Structural companion to E-3. A half-verified chain reporting "verified"
    would be the most dangerous line in this repository, so the absence is
    asserted over the source rather than over one fixture's output."""
    seen = False
    for node in ast.walk(TREE):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if (isinstance(key, ast.Constant) and key.value == "signature_checked"
                    and isinstance(value, ast.Constant)):
                seen = True
                assert value.value is False, \
                    f"line {node.lineno}: signature_checked is set to {value.value!r}"
    assert seen, ("no literal `signature_checked` key is set anywhere; this test "
                  "has stopped watching the field it was written for")


def test_e4_ordinary_green_output_that_captured_html_is_not_refused(tmp_path):
    """E-4. LOAD-BEARING. The DTD guard substring-scanned the first 8192
    characters for "<!doctype", so any snapshot test capturing an HTML page into
    <system-out> poisoned its own report. Because ANY unparsed source blocks
    every verdict, one such file denied adjudication for the whole set — and the
    person who controls that file is the pull request author. A denial-of-service
    on the instrument, delivered by writing a normal test."""
    p = w(tmp_path, "junit.xml", PYTEST_GREEN_HTML_IN_SYSTEM_OUT)
    ev = load_evidence([p])
    assert not ev["unparsed"], (
        "a green report was refused unread because a snapshot test captured "
        f"HTML: {ev['unparsed']}")
    assert [s["path"] for s in ev["sources"]] == [p]
    v, why = adjudicate_tests_pass(ev, None)
    assert v == "VERIFIED", why


def test_e4_one_poisoned_file_cannot_deny_adjudication_for_the_set(tmp_path):
    """The consequence, stated separately: the blast radius was the whole
    evidence set, not the one file."""
    a = w(tmp_path, "one.xml", PYTEST_GREEN)
    b = w(tmp_path, "two.xml", PYTEST_GREEN_HTML_IN_SYSTEM_OUT)
    ev = load_evidence([a, b])
    assert not ev["unparsed"], ev["unparsed"]
    assert verdict([a, b]) == "VERIFIED"


def test_e5_an_entity_hidden_behind_a_long_comment_is_still_refused(tmp_path):
    """E-5. LOAD-BEARING, and the exact mirror of E-4. The same 8192-character
    window that false-positived on green output false-negatives on an entity
    declaration parked behind a 9 KB comment: the document is opened, and the
    entity is expanded by the parser the guard's docstring promised would never
    see it. The billion-laughs case was stopped by libexpat's own amplification
    limit, NOT by this module. The protection was inherited and the promise was
    false.

    A guard positioned by byte offset is not a guard. Whatever replaces it must
    refuse this document wherever the declaration sits."""
    p = w(tmp_path, "bomb.xml", ENTITY_BEHIND_A_LONG_COMMENT)
    ev = load_evidence([p])
    assert not ev["sources"], (
        "the document was opened and its entity expanded; the refusal is "
        "positional, not structural")
    assert len(ev["unparsed"]) == 1
    reason = ev["unparsed"][0]["reason"].lower()
    assert "doctype" in reason or "entity" in reason or "dtd" in reason, reason
    assert verdict([p]) == "UNCHECKABLE"


def test_e6_a_deeply_nested_document_lands_in_unparsed_and_does_not_escape(tmp_path):
    """E-6. LOAD-BEARING. The suite walker recursed once per nesting level, so
    1000 nested <testsuites> in 25 KB raised RecursionError straight out of
    ``load_evidence``. RecursionError is not in the parse-error family the reader
    catches, so the caller saw a traceback instead of a refusal — and a reader
    that crashes on hostile input is a reader that can be removed from a pipeline
    by anyone who can add a file to it.

    Catching RecursionError would also stop the traceback, and it is NOT enough.
    A recursive walker that survives only because the interpreter's limit fires
    is protected by the same inherited accident that E-5 exposed: whether it
    raises depends on the ambient recursion limit and on how deep the CALLER's
    stack already was, which makes the record a function of something other than
    the bytes. The refusal has to be this module's own decision, stated in its
    own terms."""
    p = w(tmp_path, "deep.xml", DEEPLY_NESTED_SUITES)
    ev = load_evidence([p])          # must not raise
    assert isinstance(ev, dict)
    assert len(ev["sources"]) + len(ev["unparsed"]) == 1
    for u in ev["unparsed"]:
        reason = u["reason"]
        assert reason.strip()
        assert "RecursionError" not in reason and "recursion depth" not in reason, (
            "the document was refused by the interpreter's stack limit rather "
            f"than by this module: {reason}. An inherited protection is not a "
            "guarantee — it moves with the ambient recursion limit.")
    assert adjudicate_tests_pass(ev, HEAD_COMMIT)[0] in VERDICTS


def test_the_record_does_not_move_with_the_callers_stack_depth(tmp_path):
    """Law 1, stated where the recursion defect made it reachable. Reading the
    same bytes from 400 frames down must give the same record as reading them
    from the top. Anything recursive over attacker-chosen nesting fails this by
    construction, whether or not it remembers to catch the error."""
    p = w(tmp_path, "deep.xml", DEEPLY_NESTED_SUITES)
    shallow = load_evidence([p])

    def descend(n):
        if n:
            return descend(n - 1)
        return load_evidence([p])

    assert descend(400) == shallow


def test_e6_no_recursion_error_reaches_the_caller_from_any_fixture(tmp_path):
    """Stated as a class. ``load_evidence`` is a reader; a reader returns a
    report about what it could not read, and never raises."""
    paths = [w(tmp_path, f"f{i}.xml", text)
             for i, (_, text) in enumerate(ALL_FIXTURES)]
    ev = load_evidence(paths)         # must not raise for ANY of them
    assert len(ev["sources"]) + len(ev["unparsed"]) == len(paths)


def test_e7_the_returned_record_carries_a_real_per_source_binding(tmp_path):
    """E-7, first half. The shipped ``load_evidence`` hardcoded the record's
    binding to a constant for EVERY input, so a library caller reading it got
    ``{"kind": "none", ...}`` no matter what the evidence carried. The CLI
    patched the truth in afterwards, which means the library and the command line
    disagreed about the same bytes."""
    p = w(tmp_path, "a.intoto.json", statement("PASSED", passed=["t::one"]))
    ev = load_evidence([p])
    b = ev["sources"][0]["binding"]
    assert b.get("commit") == HEAD_COMMIT, \
        f"the source carries a gitCommit subject and the record says {b!r}"
    assert b.get("kind") != "none"
    top = ev.get("binding")
    assert top is None or not (top.get("kind") == "none" and top.get("commit") is None), \
        ("the record carries a top-level binding constant that contradicts its "
         "own sources; binding is PER-SOURCE")


def test_e7_contract_deviations_are_in_the_dict_load_evidence_returns(tmp_path):
    """E-7, second half. The docstring promised deviations under
    ``contract_deviations`` in the returned dict; they were attached only in
    ``main()``, so no library caller ever saw one. A documented guarantee that
    only the CLI honours is a guarantee for nobody who imports the module."""
    p = w(tmp_path, "junit.xml", PYTEST_GREEN)
    ev = load_evidence([p])
    assert "contract_deviations" in ev, \
        "contract_deviations is promised by the docstring and absent from the record"
    assert isinstance(ev["contract_deviations"], list) and ev["contract_deviations"]
    assert all(isinstance(d, str) and d.strip() for d in ev["contract_deviations"])


def test_e7_the_binding_is_per_source_and_two_sources_do_not_share_one(tmp_path):
    """Binding is PER-SOURCE, not one dict spanning a list. One attested file's
    binding must never license a reading sourced from an unattested sibling
    globbed out of the same directory."""
    j = w(tmp_path, "junit.xml", PYTEST_GREEN)
    a = w(tmp_path, "a.intoto.json", statement("PASSED", passed=["t::one"]))
    ev = load_evidence([j, a])
    by_path = {s["path"]: s["binding"] for s in ev["sources"]}
    assert len(by_path) == 2
    assert by_path[j].get("commit") is None
    assert by_path[j].get("commit_binding_verified") is False
    assert by_path[a].get("commit") == HEAD_COMMIT


# ═════════════════════════════════════════════════════════════════════════════
# the two independent booleans
# ═════════════════════════════════════════════════════════════════════════════

def test_binding_splits_into_two_independent_booleans(tmp_path):
    """One flag cannot express "report identity proven, commit binding
    unproven", and that is the exact state a sha256 subject leaves you in."""
    p = w(tmp_path, "a.intoto.json", statement("PASSED", passed=["t::one"]))
    b = load_evidence([p])["sources"][0]["binding"]
    for key in ("report_identity_verified", "commit_binding_verified"):
        assert key in b, f"binding is missing {key}"
        assert isinstance(b[key], bool), f"{key} is {type(b[key]).__name__}"


def test_report_identity_is_established_without_any_commit_binding(tmp_path):
    """A subject carrying the sha256 of junit.xml proves "this is the report that
    was attested" and is COMPLETELY SILENT on which commit the tests ran against.
    Letting one word cover both is the most dangerous mistake available here,
    precisely because it looks like the strongest channel."""
    junit = w(tmp_path, "junit.xml", PYTEST_GREEN)
    digest = hashlib.sha256(PYTEST_GREEN.encode("utf-8")).hexdigest()
    att = w(tmp_path, "a.intoto.json", statement(
        "PASSED", passed=["t::one"],
        subjects=[{"name": "junit.xml", "digest": {"sha256": digest}}]))
    ev = load_evidence([junit, att])
    b = next(s["binding"] for s in ev["sources"] if s["path"] == att)
    assert b["report_identity_verified"] is True
    assert b["commit_binding_verified"] is False, \
        "a report-identity digest was read as a commit binding"


def test_a_junit_source_never_claims_a_commit_binding(tmp_path):
    """No JUnit dialect surveyed — Surefire's XSD, the Ant/Jenkins reference,
    pytest, go-junit-report, jest-junit, trx2junit — defines any field for a
    commit. The only carrier is <properties>, written by the job under
    adjudication, which makes it a restatement of the claim."""
    p = w(tmp_path, "junit.xml", PYTEST_RED_SELF_ASSERTED_COMMIT)
    b = load_evidence([p])["sources"][0]["binding"]
    assert b["commit_binding_verified"] is False
    assert b.get("kind") != "intoto-subject"


@pytest.mark.parametrize("key", ["gitTree", "gitBlob", "gitTag"])
def test_a_git_digest_about_the_wrong_object_never_binds_a_commit(tmp_path, key):
    """gitTree binds CONTENT under a possibly different commit. gitBlob and
    gitTag are different objects entirely.

    Asserted against ``binding_against_commit`` — the function that actually
    performs the comparison and the one the adjudicator calls. This test used to
    call ``adjudicate_tests_pass`` for a side effect it does not have and then
    read ``ev["sources"][0]["binding"]``, where ``commit_assertion_matches`` is
    hardcoded ``False`` at load time because loading has no comparison target.
    It therefore asserted a constant and could not fail. Mutants M5 and M11 —
    a degraded `sha1` counting as a match, and abbreviated-SHA prefix matching —
    both survived the suite for exactly this reason."""
    p = w(tmp_path, "a.intoto.json",
          statement("PASSED", digest_key=key, passed=["t::one"]))
    ev = load_evidence([p])
    b = binding_against_commit(ev, HEAD_COMMIT)[0]
    assert b["commit_binding_verified"] is False
    assert b["commit_assertion_matches"] is False
    assert adjudicate_tests_pass(ev, HEAD_COMMIT)[0] == "UNCHECKABLE"


def test_a_sha1_digest_is_degraded_and_never_a_commit_binding(tmp_path):
    """A 40-hex value under `sha1` is equally consistent with a commit id and
    with the SHA-1 of a file's bytes, and nothing in the document distinguishes
    them.

    Asserted against the real comparison, for the reason recorded above."""
    p = w(tmp_path, "a.intoto.json",
          statement("PASSED", digest_key="sha1", passed=["t::one"]))
    ev = load_evidence([p])
    b = binding_against_commit(ev, HEAD_COMMIT)[0]
    assert b["commit_binding_verified"] is False
    assert b["commit_assertion_matches"] is False
    assert b.get("degraded_key") == "sha1", \
        "the sighting under the ambiguous key is not even recorded as degraded"
    assert adjudicate_tests_pass(ev, HEAD_COMMIT)[0] == "UNCHECKABLE", \
        "a DEGRADED-KEY sighting licensed an affirmation"


@pytest.mark.parametrize("abbrev", ["9a04d1e", "", "9A04D1EE393B5BE2773B1CE204F61FE0FD02366"])
def test_an_abbreviated_or_empty_sha_is_not_a_digest(tmp_path, abbrev):
    """The community reference's own example commit property uses a 7-character
    abbreviated SHA. An implementer who follows it writes ``startswith()``, at
    which point the empty string prefix-matches every commit in existence.

    Asserted against the real comparison, for the reason recorded above."""
    p = w(tmp_path, "a.intoto.json",
          statement("PASSED", commit=abbrev, passed=["t::one"]))
    ev = load_evidence([p])
    b = binding_against_commit(ev, HEAD_COMMIT)[0]
    assert b["commit_binding_verified"] is False
    assert b["commit_assertion_matches"] is False
    assert adjudicate_tests_pass(ev, HEAD_COMMIT)[0] == "UNCHECKABLE", \
        f"an abbreviated SHA ({abbrev!r}) prefix-matched the commit under review"


# ═════════════════════════════════════════════════════════════════════════════
# the report-only band: the information survives, the verdict does not move
# ═════════════════════════════════════════════════════════════════════════════

def test_observed_failures_are_reported_rather_than_discarded(tmp_path):
    """The re-frozen contract does not throw the failures away. It declines to
    call them a verdict. A reader who wants to look still can; a caller that
    wants to gate still cannot."""
    p = w(tmp_path, "junit.xml", PYTEST_RED)
    ev = load_evidence([p])
    obs = ev.get("observed")
    assert isinstance(obs, dict), "the record has no report-only observed band"
    assert obs.get("failing_tests") == 1, obs
    assert isinstance(obs.get("note"), str) and obs["note"].strip(), \
        "the observed band carries a count with no statement of what it is not"


def test_the_observed_band_carries_no_verdict_language(tmp_path):
    """If this fails, someone gave a report-only record an opinion. Same shape
    as test_undeclared::test_no_verdict_language_anywhere_in_the_report."""
    p = w(tmp_path, "junit.xml", PYTEST_RED)
    obs = load_evidence([p])["observed"]
    blob = json.dumps({k: v for k, v in obs.items() if k != "note"})
    assert ACCUSER not in blob
    assert "VERIFIED" not in blob


def test_nothing_downstream_gates_on_the_observed_band(tmp_path):
    """LOAD-BEARING. The band is inert by construction: rewriting it to say ten
    thousand failures must not move the verdict by one character. If it does, it
    is not a report-only band, it is the accusing branch with a new name."""
    p = w(tmp_path, "junit.xml", PYTEST_GREEN)
    ev = load_evidence([p])
    before = adjudicate_tests_pass(ev, None)
    ev["observed"] = {"failing_tests": 9999, "note": "tampered by this test"}
    after = adjudicate_tests_pass(ev, None)
    assert after == before, "the verdict reads the report-only band"


def test_errors_are_kept_distinct_from_failures_and_are_absence(tmp_path):
    """A test that ran and failed is evidence. A harness that could not run one
    is ABSENCE of evidence, and absence never contributes to a failing reading.
    They share a key in the frozen totals dict and must not share a meaning."""
    p = w(tmp_path, "junit.xml", PYTEST_COLLECTION_ERROR)
    ev = load_evidence([p])
    assert ev["totals"]["errors"] == 1
    assert ev["totals"]["failures"] == 0
    assert ev["observed"]["failing_tests"] == 0, \
        "a collection error was counted as a failing test"
    v, why = adjudicate_tests_pass(ev, HEAD_COMMIT)
    assert v == "UNCHECKABLE", why


# ═════════════════════════════════════════════════════════════════════════════
# EMPTY is over EXECUTED tests, and VERIFIED requires passed > 0
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("name,text", [
    ("all-skipped", PYTEST_ALL_SKIPPED),
    ("all-disabled", GTEST_ALL_DISABLED),
    ("collected-nothing", PYTEST_EMPTY),
], ids=["all-skipped", "all-disabled", "collected-nothing"])
def test_a_run_that_executed_nothing_never_certifies_green(tmp_path, name, text):
    """EMPTY is defined over EXECUTED tests, not over emitted records. An
    all-skipped suite, an all-DISABLED_ binary and a collection that found
    nothing are shape-identical to a healthy green run, and none of them is
    evidence that anything passed."""
    p = w(tmp_path, "junit.xml", text)
    v, why = adj([p])
    assert v == "UNCHECKABLE", f"{name}: {why}"


@pytest.mark.parametrize("name,text", [
    ("all-skipped", PYTEST_ALL_SKIPPED),
    ("all-disabled", GTEST_ALL_DISABLED),
    ("collected-nothing", PYTEST_EMPTY),
], ids=["all-skipped", "all-disabled", "collected-nothing"])
def test_a_junit_source_that_executed_nothing_resolves_to_empty(tmp_path, name, text):
    """The JUnit half of the same rule, asserted AT THE SOURCE.

    ``test_a_run_that_executed_nothing_never_certifies_green`` checks only the
    final verdict, and the adjudicator's ``passed > 0`` guard alone is enough to
    make that assertion hold. So redefining EMPTY over EMITTED RECORDS instead of
    EXECUTED tests in ``_parse_junit`` — mutant M1b — left the whole suite green.
    Two guards defend this and only one of them was watched. The in-toto leg has
    had this assertion since E-1; the JUnit leg had none."""
    p = w(tmp_path, "junit.xml", text)
    src = load_evidence([p])["sources"][0]
    assert src["executed"] == 0, f"{name}: {src['executed']} tests executed"
    assert src["outcome"] == "EMPTY", (
        f"{name}: outcome is {src['outcome']!r}. EMPTY is defined over EXECUTED "
        "tests, not over emitted testcase records — an all-skipped suite and an "
        "all-DISABLED_ binary are shape-identical to a green run.")


def test_an_attestation_naming_no_test_is_empty_even_when_it_says_passed(tmp_path):
    """The three test-name lists are OPTIONAL. A conforming attestation may say
    PASSED and name nothing, and the absence of the lists carries zero
    information. VERIFIED requires passed > 0."""
    p = w(tmp_path, "a.intoto.json", statement("PASSED"))
    v, why = adj([p], HEAD_COMMIT)
    assert v == "UNCHECKABLE", why


def test_verified_requires_at_least_one_passed_test(tmp_path):
    p = w(tmp_path, "a.intoto.json", statement("PASSED", passed=["t::one"]))
    assert verdict([p], HEAD_COMMIT) == "VERIFIED"
    q = w(tmp_path, "b.intoto.json", statement("PASSED", warned=["t::one"]))
    assert verdict([q], HEAD_COMMIT) == "UNCHECKABLE", \
        "WARNED is neither pass nor fail and may not be folded into either"


# ═════════════════════════════════════════════════════════════════════════════
# any unparsed source blocks VERIFIED
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("junk", [NOT_XML, XML_BUT_NOT_JUNIT],
                         ids=["build-log", "html"])
def test_any_unparsed_source_blocks_verified(tmp_path, junk):
    """LOAD-BEARING. A partial read may honestly DECLINE; it may not honestly
    AFFIRM. Reading one file in ten and certifying on it is not a partial read,
    it is a guess with a citation attached."""
    good = w(tmp_path, "junit.xml", PYTEST_GREEN)
    bad = w(tmp_path, "build.log", junk)
    ev = load_evidence([good, bad])
    assert len(ev["unparsed"]) == 1
    v, why = adjudicate_tests_pass(ev, None)
    assert v == "UNCHECKABLE", why


def test_a_missing_file_is_absence_and_is_reported_not_dropped(tmp_path):
    good = w(tmp_path, "junit.xml", PYTEST_GREEN)
    missing = str(tmp_path / "never-uploaded.xml")
    ev = load_evidence([good, missing])
    assert [u["path"] for u in ev["unparsed"]] == [missing]
    assert adjudicate_tests_pass(ev, None)[0] == "UNCHECKABLE"


def test_an_empty_file_is_absence_not_a_green_run(tmp_path):
    p = w(tmp_path, "junit.xml", "")
    ev = load_evidence([p])
    assert not ev["sources"] and len(ev["unparsed"]) == 1
    assert adjudicate_tests_pass(ev, None)[0] == "UNCHECKABLE"


def test_no_evidence_at_all_is_uncheckable():
    """An unattested commit is unattested. Absence of a report is not a report."""
    v, why = adj([])
    assert v == "UNCHECKABLE"
    assert isinstance(why, str) and why.strip()


# ═════════════════════════════════════════════════════════════════════════════
# roots lie, in both directions
# ═════════════════════════════════════════════════════════════════════════════

def test_a_root_claiming_green_over_a_failing_child_is_resolved_from_the_child(tmp_path):
    """googletest's own golden EXPECTED_NO_TEST_XML ships a root reading
    tests="0" failures="0" wrapping a suite with a real <failure>. Jenkins,
    twenty years the reference consumer, recomputes every figure and reads none
    of the counts."""
    p = w(tmp_path, "junit.xml", GTEST_ROOT_CLAIMS_GREEN)
    ev = load_evidence([p])
    assert ev["observed"]["failing_tests"] == 1, \
        "the root's zero was believed over a real <failure> child"
    assert adjudicate_tests_pass(ev, None)[0] != "VERIFIED"


def test_a_root_that_overcounts_failures_is_resolved_from_the_children(tmp_path):
    """The inverse. Surefire's counts are xs:string in its own XSD, so a
    non-integer count is schema-valid and a wrong one is routine."""
    p = w(tmp_path, "junit.xml", ROOT_OVERCOUNTS_FAILURES)
    ev = load_evidence([p])
    assert ev["observed"]["failing_tests"] == 0, \
        "a root attribute manufactured two failures that no testcase carries"
    assert adjudicate_tests_pass(ev, None)[0] == "VERIFIED"


def test_a_flaky_surefire_rerun_that_passed_is_not_a_failure(tmp_path):
    """Surefire's own documentation: existing consumers still consider it a
    passing test. A parser matching ``tag.endswith("failure")`` calls a green
    build red — the retired false-accusation shape, exactly."""
    p = w(tmp_path, "junit.xml", SUREFIRE_FLAKY_GREEN)
    ev = load_evidence([p])
    assert ev["observed"]["failing_tests"] == 0
    assert adjudicate_tests_pass(ev, None)[0] == "VERIFIED"


# ═════════════════════════════════════════════════════════════════════════════
# Law 1 — purity
# ═════════════════════════════════════════════════════════════════════════════

_BANNED_IMPORTS = {"subprocess", "socket", "random", "time", "requests",
                   "urllib", "http", "asyncio", "shutil", "datetime", "secrets"}
# Read as EXPRESSIONS, never as substrings. The module is expected to quote its
# own purity law in prose, and a text scan would read the promise as the breach.
_BANNED_EXPRESSIONS = ("os.environ", "os.getenv", "os.popen", "os.system",
                       "datetime.now", "datetime.utcnow", "time.time",
                       "random.random", "environ", "getenv",
                       # G-E1 names Path.cwd by name and nothing watched it.
                       # `pathlib` is a legitimate import here — the module reads
                       # files — so the banned-IMPORT list can never catch this;
                       # only the expression can. A record carrying Path.cwd() is
                       # a function of the working directory, and every
                       # determinism test in this file runs in one directory, so
                       # the impurity is invisible to all of them. Mutant M4c
                       # folded Path.cwd() into the record and the suite stayed
                       # green.
                       "Path.cwd", "Path.home", "os.getcwd")


def _dotted_names(tree):
    def full(node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = full(node.value)
            return f"{base}.{node.attr}" if base else None
        return None

    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Name, ast.Attribute)):
            got = full(node)
            if got:
                names.add(got)
    return names


def _imported_top_level_modules(tree):
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mods.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            mods.add(node.module.split(".")[0])
    return mods


def _from_imported_names(tree):
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.name)
    return names


def test_the_module_imports_nothing_that_could_make_it_impure():
    """LOAD-BEARING. Same source-inspection trick as
    test_undeclared::test_it_uses_the_gates_own_parser_not_a_copy.

    This module exists because diffgate's --run path cannot be sealed in a
    capsule: its verdict is a function of an exit code, i.e. of the dependency
    set, the interpreter, the network and the clock. capsule.py refuses a
    --run-resolved tests_pass verdict by name. If any of these imports appear
    here, this module inherits the same disqualification and nothing it says can
    ever be re-derived."""
    found = _imported_top_level_modules(TREE) & _BANNED_IMPORTS
    assert not found, \
        f"styxx.evidence imports {sorted(found)}; it is no longer a function of bytes"
    smuggled = _from_imported_names(TREE) & {"environ", "getenv", "run", "Popen",
                                             "check_output", "urlopen"}
    assert not smuggled, f"styxx.evidence from-imports {sorted(smuggled)}"


@pytest.mark.parametrize("banned", _BANNED_EXPRESSIONS)
def test_the_module_never_reads_ambient_state(banned):
    """An environ read makes the record a function of ambient state, and capsule
    refusal R5 — divergence from the live re-gate — then fires on every capsule
    minted in CI and verified anywhere else.

    There is a clean alternative and the spec should say so: GitHub writes
    head.sha into a FILE at GITHUB_EVENT_PATH, so a capture step may read the
    environment and hand this module bytes. That restores purity and buys zero
    trust — whoever ran capture could have written anything."""
    assert banned not in _dotted_names(TREE), f"styxx.evidence evaluates {banned}"


def test_the_module_namespace_holds_no_impure_module_object():
    """Belt and braces: an alias or a function-local import slips past the ast
    walk but still binds a name in the module namespace."""
    for name in sorted(_BANNED_IMPORTS):
        assert not isinstance(getattr(E, name, None), type(json)), \
            f"styxx.evidence has a live {name} module object"


def test_the_modules_own_purity_receipt_passes_and_is_watched():
    """G-E1 says the static purity check "ships as a receipt". It is printed on
    every CLI run and, until now, asserted by nothing — so the receipt could have
    read ATTENTION on every run in production and this suite would have stayed
    green. A check nobody asserts is a decoration, which is this file's own
    stated standard applied to the module's self-check.

    Both halves: the answer must be ``ok``, and it must be watching the
    expression that no import rule can reach (``Path.cwd`` — ``pathlib`` is a
    legitimate import here, so only an expression-level rule can catch it)."""
    pur = E.selfcheck_purity()
    assert pur["ok"] is True, (
        f"the module's own purity receipt does not pass: {pur['reason']} "
        f"imported={pur['imported']} referenced={pur['referenced']}")
    assert "Path.cwd" in (pur.get("checked_expressions") or []), \
        "the purity receipt does not watch Path.cwd, which G-E1 names by name"

    acc = E.selfcheck_no_accusation()
    assert acc["ok"] is True, \
        f"the no-accusation receipt does not pass: {acc['reason']} {acc['code_occurrences']}"


def test_no_argv_or_stdin_read_below_the_cli():
    """``sys.argv`` may be touched in ``main()`` and nowhere else. Everything
    below the CLI boundary is a function of its arguments."""
    for fn in ast.walk(TREE):
        if not isinstance(fn, ast.FunctionDef) or fn.name == "main":
            continue
        used = _dotted_names(fn)
        assert "sys.argv" not in used, f"{fn.name} reads sys.argv"
        assert "sys.stdin" not in used, f"{fn.name} reads sys.stdin"


# ═════════════════════════════════════════════════════════════════════════════
# Law 1 — determinism
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("name,text", ALL_FIXTURES, ids=_FIXTURE_IDS)
def test_the_same_bytes_yield_the_same_verdict_twice(tmp_path, name, text):
    """LOAD-BEARING. This is what lets a capsule re-derive the verdict in ten
    years. If it fails, the module has acquired a hidden input."""
    p = w(tmp_path, "report.xml", text)
    assert adj([p], HEAD_COMMIT) == adj([p], HEAD_COMMIT)


def test_loading_the_same_bytes_twice_yields_an_identical_record(tmp_path):
    a = w(tmp_path, "pytest.xml", PYTEST_RED)
    b = w(tmp_path, "TEST-org.example.FailingTest.xml", SUREFIRE_RED)
    c = w(tmp_path, "a.intoto.json", statement("PASSED", passed=["t::one"]))
    assert load_evidence([a, b, c]) == load_evidence([a, b, c])


def test_the_record_is_json_serialisable(tmp_path):
    """A verdict that cannot be written into a capsule is a verdict nobody can
    re-derive."""
    a = w(tmp_path, "junit.xml", PYTEST_RED)
    b = w(tmp_path, "a.intoto.json", statement("FAILED", failed=["t::two"]))
    json.dumps(load_evidence([a, b]))


# ═════════════════════════════════════════════════════════════════════════════
# shape invariants over every fixture
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("name,text", ALL_FIXTURES, ids=_FIXTURE_IDS)
@pytest.mark.parametrize("commit", [None, HEAD_COMMIT, MERGE_COMMIT],
                         ids=["no-commit", "head", "merge-sha"])
def test_every_input_produces_exactly_one_of_the_two_verdicts(tmp_path, name, text, commit):
    p = w(tmp_path, "report.xml", text)
    v, why = adj([p], commit)
    assert v in VERDICTS, f"{name}: {v}"
    assert isinstance(why, str) and why.strip(), "a verdict with no stated reason"


@pytest.mark.parametrize("name,text", ALL_FIXTURES, ids=_FIXTURE_IDS)
def test_no_input_produces_an_accusing_reading_anywhere_in_the_record(tmp_path, name, text):
    """Stated as a class over every fixture and every commit, so that a new
    accusing path added later has to survive sixteen inputs rather than the one
    somebody remembered to write a fixture for."""
    p = w(tmp_path, "report.xml", text)
    for commit in (None, HEAD_COMMIT, MERGE_COMMIT):
        ev = load_evidence([p])
        v, why = adjudicate_tests_pass(ev, commit)
        blob = json.dumps(ev, default=str) + why + v
        assert ACCUSER not in blob, f"{name} with commit={commit}"


def test_the_record_reports_what_it_could_not_read(tmp_path):
    good = w(tmp_path, "junit.xml", PYTEST_GREEN)
    bad = w(tmp_path, "build.log", NOT_XML)
    ev = load_evidence([good, bad])
    assert [s["path"] for s in ev["sources"]] == [good]
    assert [u["path"] for u in ev["unparsed"]] == [bad]
    assert ev["paths_requested"] == [good, bad]


def test_the_record_names_the_spec_and_digests_the_bytes_it_read(tmp_path):
    p = w(tmp_path, "junit.xml", PYTEST_GREEN)
    ev = load_evidence([p])
    assert isinstance(ev["spec"], str) and ev["spec"].startswith("styxx-evidence/")
    assert ev["sources"][0]["sha256"] == \
        hashlib.sha256(PYTEST_GREEN.encode("utf-8")).hexdigest()


def test_the_record_says_what_it_did_not_check(tmp_path):
    p = w(tmp_path, "junit.xml", PYTEST_GREEN)
    ev = load_evidence([p])
    assert ev["not_checked"], "the record claims to have checked everything"
    blob = " ".join(ev["not_checked"]).lower() + (ev.get("boundary") or "").lower()
    assert "signature" in blob, \
        "the record does not name the signature it declined to verify"


# ═════════════════════════════════════════════════════════════════════════════
# the CLI — content never moves the exit code
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("name,text", ALL_FIXTURES, ids=_FIXTURE_IDS)
@pytest.mark.parametrize("commit", [None, HEAD_COMMIT], ids=["no-commit", "head"])
def test_main_exits_zero_for_every_evidence_content(tmp_path, name, text, commit, capsys):
    """LOAD-BEARING. ``main()`` may NEVER exit non-zero because of evidence
    CONTENT. An exit code is a gate, and a gate on an unmeasurable reading is the
    thing this contract deleted. Non-zero is reserved for usage errors."""
    p = w(tmp_path, "report.xml", text)
    argv = [p] + (["--commit", commit] if commit else [])
    rc = main(argv)
    capsys.readouterr()
    assert rc == 0, f"{name} with commit={commit} exited {rc}"


def test_main_exits_zero_on_a_bound_red_attestation(tmp_path, capsys):
    """The single case the old exit code existed for. A bound, red, forged
    attestation is exactly the input that used to return 1, and returning 1 is
    what let a CI job fail somebody's pull request on a reading whose precision
    is structurally unmeasurable."""
    p = w(tmp_path, "a.intoto.json",
          dsse(statement("FAILED", failed=["tests.test_app::test_two"])))
    assert main([p, "--commit", HEAD_COMMIT]) == 0
    out = capsys.readouterr()
    assert ACCUSER not in out.out + out.err


def test_main_exits_zero_on_no_evidence_at_all(capsys):
    """Being handed nothing is the most common real condition this module meets
    and it has an answer. Letting argparse exit 2 would convert the module's most
    important answer into a reason to think the tool is broken."""
    assert main([]) == 0
    out = capsys.readouterr()
    assert "UNCHECKABLE" in out.out + out.err, \
        "the refusal must be printed loudly, never swallowed"


def test_main_exits_non_zero_only_for_a_usage_error(capsys):
    """The reserved case, asserted so the "always zero" rule above cannot be
    satisfied by a function that has forgotten how to fail at all."""
    with pytest.raises(SystemExit) as exc:
        main(["--no-such-flag"])
    capsys.readouterr()
    code = exc.value.code
    assert code not in (0, None), f"a usage error exited {code!r}"


def test_main_does_expose_a_commit_flag(tmp_path, capsys):
    """Recorded as a fact about the SHIPPED surface, because the suite this one
    replaces carried a load-bearing test whose docstring asserted the opposite —
    it claimed main() has no commit argument while main() defines --commit, and
    it passed only because it never tried. If --commit is ever removed this
    raises SystemExit(2) and the CLI tests above stop testing anything."""
    p = w(tmp_path, "junit.xml", PYTEST_RED)
    assert main([p, "--commit", HEAD_COMMIT]) == 0
    capsys.readouterr()


def test_main_is_deterministic_in_its_exit_code(tmp_path, capsys):
    p = w(tmp_path, "junit.xml", PYTEST_RED)
    first = main([p, "--commit", HEAD_COMMIT])
    second = main([p, "--commit", HEAD_COMMIT])
    capsys.readouterr()
    assert isinstance(first, int) and first == second == 0


def test_the_cli_prints_the_refusal_rather_than_hiding_it(tmp_path, capsys):
    p = w(tmp_path, "junit.xml", PYTEST_RED)
    main([p, "--commit", HEAD_COMMIT])
    printed = capsys.readouterr()
    blob = printed.out + printed.err
    assert "UNCHECKABLE" in blob
    assert ACCUSER not in blob


def test_the_cli_writes_a_record_that_round_trips(tmp_path, capsys):
    p = w(tmp_path, "junit.xml", PYTEST_RED)
    out = tmp_path / "evidence.json"
    assert main([p, "--json", str(out)]) == 0
    capsys.readouterr()
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["verdict"] in VERDICTS
    assert rec["contract_deviations"]
    assert rec["observed"]["failing_tests"] == 1
