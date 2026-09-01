# -*- coding: utf-8 -*-
"""diffgate x styxx.evidence -- the wiring contract, and the tripwires on it.

WRITTEN AGAINST THE 581-LINE MODULE, WHICH HAD NO WIRING AT ALL

Every test in this file was written and watched failing against the shipped
`styxx.diffgate` of 2026-09-01 -- the 581-line version in which `gate_diff_text`
took no `evidence` parameter, nothing was imported from `styxx.evidence`, and
`diffgate.py:452` resolved a `tests_pass` claim to the accusing verdict whenever
the `--run` command exited nonzero. The wiring landed while this file was being
written. That is why the markers below are written as SELF-ARMING conditions on
probes of the CURRENTLY LOADED module rather than as bare xfails: a marker whose
condition is a probe disarms itself the moment the defect it names is repaired,
and `strict=True` turns an unexplained pass into a FAILURE rather than a quiet
XPASS. There is no configuration under which a test in this file is silently
green.

The suite this lab replaced two days ago was GREEN ON ARRIVAL and survived four
of ten mutants. `tests/test_evidence.py` carries the corrective in its own
docstring -- "a test that has never been seen to fail is a decoration."

WHAT IS ASSERTED, AND WHY EACH ONE IS LOAD-BEARING

  * TWO MONOTONICITY PROPERTIES, ASSERTED SEPARATELY, BECAUSE ONLY ONE HOLDS.
    An earlier draft of this file asserted one of them under one name and read
    the green result as though it had established both. It had not. The two are
    the difference between a guarantee and a wish, and conflating them is how a
    safety rule gets "repaired" by someone who read a receipt as saying more than
    it says.

    ``test_adding_evidence_never_worsens_the_gate_against_no_evidence`` -- the
    guarantee, and it HOLDS. Over nine claim combinations x six evidence contents
    plus a nonexistent path x two strictness settings, a gate handed evidence
    ranks no worse than the same gate handed NONE, at the overall verdict AND at
    every individual claim. The baseline is the EMPTY evidence set and nothing
    else, and the test name says so, because the comparison cannot be recovered
    from a report line that only says "never worsens". This is the invariant that
    makes the leg safe to ship before its precision is measured: an adjudicator
    that, relative to supplying nothing, can only resolve UNCHECKABLE upward
    cannot manufacture an accusation however wrong the extraction feeding it is.
    `_gate` argues this in a comment; this test measures it.

  * ``test_extending_an_evidence_set_can_demote_verified_and_must`` -- the
    property that DOES NOT hold, PINNED so that nobody repairs it. E -> E union
    {x} can move a `tests_pass` claim from VERIFIED back to UNCHECKABLE and a
    --strict PASS back to FAIL. That is the contract's own rule rather than a
    defect in it: a partial read may honestly DECLINE but may not honestly
    AFFIRM, so a source that could not be read blocks VERIFIED. You cannot
    certify "all tests pass" from nine shards out of ten. Measured rather than
    argued -- 12 of the 14 (addition x strictness) cells in that test's grid
    worsen the verdict, and the 2 that do not are the well-formed zero-test
    report, which parses and contributes nothing. That row is kept in the grid as
    the negative control, because a rule that demoted on ANY addition would be a
    file counter rather than a reader.

    PREREG_evidence_leg's G-E6 is neither of these exactly: it bars a pull
    request from GAINING AN ACCUSATION relative to the shipped gate run without
    `--run`, which is the accusation half of the empty-baseline property and says
    nothing at all about set extension. This file no longer describes itself as
    its caller-side analogue -- it asserts more than G-E6 in one direction (all
    three verdicts, not only the accusing one) and nothing in the other.

  * ``test_no_evidence_content_can_produce_an_accusation`` -- by any route, for
    any content, at any commit, in strict mode and out. `styxx.evidence` cannot
    accuse by construction, but its own ``selfcheck_no_accusation`` states the
    hole it cannot close: it "does not prove a caller has not invented an
    accusation of its own out of the report-only ``observed`` band". diffgate is
    that caller. Two tests close it from the caller's side -- one behavioural
    over six contents including a forged FAILED attestation, one structural over
    the source.

  * ``test_the_run_branch_cannot_reach_an_accusation_in_the_source`` -- asserted
    STRUCTURALLY, never by output alone. A test that only checks verdicts passes
    on a module carrying the branch behind ``if False`` or behind a flag set to
    off, and a branch nobody reaches today is a branch someone reaches after a
    refactor. The behavioural companion is kept and is labelled insufficient on
    its own. The AST helpers backing the structural tests are themselves watched
    firing against five planted accusations, three of which no output-level test
    can see.

  * ``test_it_uses_styxx_evidence_not_a_private_copy`` -- by source inspection,
    the way ``tests/test_undeclared.py`` asserts reconcile parses the diff with
    the gate's own parser rather than a second one that can drift. Here the
    drift would be worse than a disagreement about diffs: a private reader of
    JUnit or in-toto bytes inside diffgate would be a reader with no ``VERDICTS``
    tuple, no self-check, and no docstring saying what VERIFIED does not mean.

THE MUTANT LOG

Twenty-two deliberately defective copies of `diffgate` (and one of `evidence`)
were injected in place of the real module and this file was run against each with
``--runxfail``, so a self-arming marker could not absorb the failure. All 22 were
caught, and every test in this file except the three that validate its own AST
helpers was reddened by at least one of them. The four that no mutant reached on
the first pass -- the vocabulary check, the VERIFIED-disclaims check, the
determinism check and the CLI check -- had mutants written for them rather than
being left unexercised.

Two mutants SURVIVED on the first attempt and both were defects in THIS FILE, not
in the module:

  * M23 renamed the flag to ``--evidenceX`` and passed, because ``"--evidence" in
    src`` is a substring test and ``--evidence`` is a prefix of ``--evidenceX``.
    The check now reads exact option strings out of the parser's own AST.
  * M23b stripped the code-execution warning off ``--run`` and passed, because a
    whole-source search for "executes" was satisfied by ``--evidence``'s help
    saying it "executes nothing". The check is now scoped to ``--run``'s own help
    string.

Both are recorded here rather than quietly fixed, because a suite that finds its
own weak assertions is the only evidence that the mutants were real.

THE MONOTONICITY SWEEP, 2026-09-01 -- MP1..MP7, MB1..MB2

Eleven further mutants were injected into `styxx.evidence` and `styxx.diffgate`
and the two monotonicity tests were run against each with ``--runxfail``. The
module bytes were snapshotted before each injection and restored after it, and
the restore was asserted by SHA-256 and by CRLF count -- both modules are CRLF,
this test file is LF, and a mutant harness that silently renormalises line
endings has edited the module it was supposed to be measuring.

Reddening the PIN (``test_extending_an_evidence_set_can_demote_verified_and_must``):

  * MP1 skips unreadable sources instead of blocking on them -- ``if unparsed:``
    made unreachable. 6 of 14 cells red: this is the exact "repair" the pin's
    docstring names as the one someone reaches for.
  * MP2 special-cases the empty file, filtering `file is empty` out of
    ``unparsed``. 2 of 14 red -- only the `zero_bytes` row, which is why that
    row is in the grid rather than being folded into `unparsable`.
  * MP3 takes the best available reading on disagreement, returning VERIFIED
    where the module refuses to pick. 4 of 14 red (`red`, `forged_failed`).
  * MP4 folds a harness error into a pass. 2 of 14 red.
  * MP5 demotes on ANY second source. 2 of 14 red -- caught by the `empty`
    NEGATIVE CONTROL alone, which is the only thing in this file that can tell a
    reader from a file counter.
  * MP6 makes strictness decorative (``strict and uncheckable`` dropped from the
    verdict). 6 of 14 red, all in the strict half.
  * MP7 makes nothing ever VERIFIED. 14 of 14 red, at the pin's own baseline
    guard -- which is why that guard is there: a module that had stopped
    affirming altogether would otherwise satisfy a demotion test trivially.
  * MB1 and MB2 also red the pin, 14 of 14 each.

Reddening the BASELINE test
(``test_adding_evidence_never_worsens_the_gate_against_no_evidence``):

  * MB1 drops the `tests_pass` claim whenever evidence is supplied. 112 of 126
    cells red.
  * MB2 makes supplying evidence force a FAIL. 49 of 126 red.
  * MP1..MP7 leave it fully green, all 126 cells. That is the separation the
    rename was for: seven distinct weakenings of the partial-read rule are
    INVISIBLE to the empty-baseline property, and before the split there was no
    test in this file that any of them reddened.

ONE SURVIVOR, AND IT IS A FACT ABOUT THE MODULE, NOT A HOLE IN THE PIN

  * MP3b disables BOTH the ``sources_disagree`` branch and the red reading and
    still changes no output: the union of a green and a red report then falls
    through to ``outcome {outcome!r} is outside the decision table``, which
    declines anyway. Three independent layers defend that demotion, the same
    shape M1..M16 found on the --run leg. Only MP3c, which disables all three by
    forcing the PASSED arm, gets the module to affirm -- and the pin reds on it,
    4 of 14. Recorded rather than quietly dropped: a mutant that survives because
    the module is defended in depth is evidence about the module, and a mutant
    that survives because the test is weak is evidence about the test. These two
    are not the same result and a log that prints only "caught" cannot tell them
    apart.

WHAT M1 THROUGH M16 ESTABLISHED, WHICH IS A FACT ABOUT THE MODULE

Restoring the accusing verdict inside ``_run_leg`` -- reached, unreached, or
behind a flag -- changes NO OUTPUT. Three independent layers each swallow it:
``_tests_pass_verdict`` folds the leg's answer with a promote-to-VERIFIED-only
rule, the ``_TESTS_PASS_VERDICTS`` clamp rewrites anything else to UNCHECKABLE,
and the branch itself is gone. Only M17, which removes all three, makes the
behavioural test ``test_a_failing_run_command_does_not_accuse`` go red. So that
test is not merely weaker than its structural counterpart: against this module it
is unfalsifiable by any single-edit mutant, and it is kept as a regression guard
for the refactor that removes the other two layers, not as evidence of anything
today. The structural test is the one that bites, on all five accusation-restoring
mutants, which is the whole argument for asserting absence over the SOURCE.

WHAT THIS FILE DOES NOT ASSERT, SAID SO IT IS NOT MISTAKEN FOR COVERAGE

Nothing here measures PRECISION. A green run says the wiring cannot accuse and
cannot make a verdict worse than supplying no evidence at all. It says nothing
about whether the `tests_pass` extraction feeding it is correct, and that
extraction is the unmeasured leg: `extraction_census.json` reports that 6.27% of
matches CARRY A MECHANICAL NON-ASSERTION INDICATOR -- an unchecked PR-template
checkbox, a code fence, a blockquote, a negation -- and the status of the
remaining 83% is UNMEASURED, its only indicator an unvalidated regex of the exact
class this lab measured at 0.16. Panel A has never been run.

That 6.27% is a CONTAINMENT figure and is not a wrongness figure, and this file
used to state it as one. The census determines where a match SITS, which is a
function of the bytes and involves no judgment; whether sitting inside a
blockquote or an unchecked box makes an extraction WRONG is exactly the question
Panel A is preregistered to answer and has never been asked. Neither is there any
bar in this repository that 6.27% can be read against: the only preregistered
extraction threshold is G-J6 in PREREG_third_party_precision_2026_09_01.md,
extraction validity >= 0.90 -- a 10% budget on a quantity we know of no
measurement of, ours or anyone's, and not a 5% budget on this one. G-J6 is also
a different quantity: validity is what Panel A would adjudicate, and containment
is what the census counts. An earlier draft of this docstring compared 6.27%
against a "5% error budget" that does not exist anywhere in this repository. It
was invented here.

These tests are the reason the leg is safe to ship UNMEASURED; they are not a
substitute for measuring it.
"""
from __future__ import annotations

import ast
import inspect
import json
import subprocess
import types

import pytest

import styxx.diffgate as D
import styxx.evidence as E
from styxx.diffgate import gate_diff_text
from styxx.evidence import adjudicate_tests_pass, load_evidence

# Assembled at runtime so this file contains no occurrence of the accusing word.
# A test file that accuses itself is the same mention/use defect it is testing
# for -- the one ANALYSIS_base_rate_ceiling records against this lab's own
# certificate.
ACCUSE = "CONTRA" + "DICTED"

CLAIM_RANK = {"VERIFIED": 0, "UNCHECKABLE": 1, ACCUSE: 2}
GATE_RANK = {"PASS": 0, "FAIL": 1}

# The commit-binding parameter was renamed while this file was being written
# (`evidence_commit` -> `commit`). Both spellings are accepted here on purpose:
# the contract this file holds the wiring to is "the commit under review reaches
# the adjudicator, so a green report from another branch does not answer for
# this one". That is a behaviour, not a spelling, and pinning the spelling would
# make this suite fail for a reason that has nothing to do with what it guards.
# What IS pinned is that SOME such parameter exists -- see the CLI test.
_COMMIT_KW = next((k for k in ("commit", "evidence_commit")
                   if k in inspect.signature(gate_diff_text).parameters), None)


def gate(summary, diff, *, commit=None, **kw):
    """gate_diff_text, with the commit passed under whichever name it has."""
    if commit is not None:
        assert _COMMIT_KW, "gate_diff_text has no commit-binding parameter"
        kw[_COMMIT_KW] = commit
    return gate_diff_text(summary, diff, **kw)


# ============================================================== source helpers
# Each answers a question about the SOURCE, not about the output, and each is
# watched firing against a planted accusation below. A helper that has never
# been seen to fire is as much a decoration as a test that has never been seen
# to fail.

def _docstring_ids(tree) -> set:
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            body = getattr(node, "body", None) or []
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                ids.add(id(body[0].value))
    return ids


def _parents(tree) -> dict:
    out = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            out[id(child)] = node
    return out


def _stmt_chain(node, parents) -> list:
    """Every enclosing statement, innermost first. An accusation one level below
    its guard -- ``if False and r.returncode: c.verdict = ...`` -- is findable
    only by walking up, and that is precisely the shape an output-only test
    cannot see."""
    chain, cur = [], parents.get(id(node))
    while cur is not None:
        if isinstance(cur, ast.stmt):
            chain.append(cur)
        cur = parents.get(id(cur))
    return chain


def _guard_parts(stmt) -> list:
    """The parts of a statement that GUARD its body, never the body itself.

    Scoping this correctly is load-bearing and the first draft got it wrong.
    `_gate` builds one long ``elif kind == ...`` chain, which in the AST nests
    each arm inside the previous arm's ``orelse`` -- so asking "does any
    enclosing statement mention `tests_pass`?" made every path-claim accusation
    in the chain look like a `tests_pass` accusation, because the `tests_pass`
    arm is inside the subtree of the arm above it. Six false positives against
    the shipped module. The guard is the test, not the whole `if`."""
    if isinstance(stmt, (ast.If, ast.While)):
        return [stmt.test]
    if isinstance(stmt, (ast.For, ast.AsyncFor)):
        return [stmt.iter]
    if isinstance(stmt, ast.With):
        return list(stmt.items)
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                         ast.Try, ast.Module)):
        return []
    return [stmt]                      # a simple statement guards itself


def _references(node, word: str) -> bool:
    """Does this node name *word* -- as an attribute, a bare name, or a string
    literal? Matched as a whole token, never as a substring, so the words this
    file looks for do not accuse the prose that explains them."""
    for n in ast.walk(node):
        if isinstance(n, ast.Attribute) and n.attr == word:
            return True
        if isinstance(n, ast.Name) and n.id == word:
            return True
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value == word:
            return True
    return False


def accusation_sites(source: str, word: str) -> list:
    """Every place the accusing verdict is produced as CODE inside a region
    guarded by *word*. Docstrings are excluded: the word necessarily appears in
    the prose explaining its own absence, and a grep cannot tell mention from
    use."""
    tree = ast.parse(source)
    doc, parents = _docstring_ids(tree), _parents(tree)
    sites = []
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Constant) and isinstance(n.value, str)):
            continue
        if ACCUSE not in n.value or id(n) in doc:
            continue
        for stmt in _stmt_chain(n, parents):
            if any(_references(part, word) for part in _guard_parts(stmt)):
                sites.append({"lineno": n.lineno, "excerpt": n.value[:70],
                              "guard": word})
                break
    return sites


def run_branch_accusation_sites(source: str) -> list:
    return accusation_sites(source, "returncode")


def accusation_sites_under_tests_pass(source: str) -> list:
    return accusation_sites(source, "tests_pass")


def observed_band_accusation_sites(source: str) -> list:
    return accusation_sites(source, "observed")


# ================================================================== fixtures
# Byte-for-byte the shapes tests/test_evidence.py already pins, so a
# disagreement between the two files is a real disagreement about the wiring and
# not about what a green report looks like.

HEAD_COMMIT = "9a04d1ee393b5be2773b1ce204f61fe0fd02366a"
OTHER_COMMIT = "1111111111111111111111111111111111111111"

GREEN = """<?xml version="1.0" encoding="utf-8"?>
<testsuites name="pytest tests"><testsuite name="pytest" errors="0" failures="0" \
skipped="0" tests="2" time="0.012">
<testcase classname="tests.test_app" name="test_one" time="0.001" />
<testcase classname="tests.test_app" name="test_two" time="0.001" />
</testsuite></testsuites>
"""

RED = """<?xml version="1.0" encoding="utf-8"?>
<testsuites name="pytest tests"><testsuite name="pytest" errors="0" failures="1" \
skipped="0" tests="2" time="0.031">
<testcase classname="tests.test_app" name="test_one" time="0.001" />
<testcase classname="tests.test_app" name="test_two" time="0.002">
<failure message="assert 1 == 2">E       assert 1 == 2</failure>
</testcase>
</testsuite></testsuites>
"""

EMPTY = """<?xml version="1.0" encoding="utf-8"?>
<testsuites name="pytest tests"><testsuite name="pytest" errors="0" failures="0" \
skipped="0" tests="0" time="0.001" />
</testsuites>
"""

COLLECTION_ERROR = """<?xml version="1.0" encoding="utf-8"?>
<testsuites name="pytest tests"><testsuite name="pytest" errors="1" failures="0" \
skipped="0" tests="1" time="0.004">
<testcase classname="" name="tests/test_integration.py" time="0.0">
<error message="collection failure">ImportError: No module named 'optional_dep'</error>
</testcase>
</testsuite></testsuites>
"""

UNPARSABLE = "Sorry, the test job did not produce a report.\n"

# A forgery that costs one text editor and no key material: an in-toto statement
# asserting FAILED against the commit under review. styxx.evidence reads it and
# declines. The point of pinning it here is that DIFFGATE must decline too, on
# the caller side of the hole selfcheck_no_accusation names.
FORGED_FAILED = json.dumps({
    "_type": "https://in-toto.io/Statement/v1",
    "subject": [{"name": "_", "digest": {"gitCommit": HEAD_COMMIT}}],
    "predicateType": "https://in-toto.io/attestation/test-result/v0.1",
    "predicate": {"result": "FAILED", "configuration": [],
                  "failedTests": ["tests/test_app.py::test_two"]},
}, indent=2) + "\n"

EVIDENCE_CONTENTS = [
    ("green", GREEN),
    ("red", RED),
    ("empty", EMPTY),
    ("collection_error", COLLECTION_ERROR),
    ("unparsable", UNPARSABLE),
    ("forged_failed", FORGED_FAILED),
]
CONTENT_NAMES = [n for n, _ in EVIDENCE_CONTENTS]


@pytest.fixture(scope="module")
def ev(tmp_path_factory):
    """name -> list of paths. Plus three entries that are NOT in
    ``CONTENT_NAMES``, so adding them does not silently re-parametrize every
    other test in this file:

      "none"        nothing supplied at all
      "missing"     a path that does not exist
      "zero_bytes"  a file that exists and is empty

    All three are absences and none of them is the same absence. "zero_bytes" is
    the CI shape the set-extension pin below is written around: a shard whose
    upload produced a file but no report. It is a distinct case from
    ``unparsable`` (bytes that are not a report) and from ``missing`` (no file),
    and the contract treats all three identically on purpose -- a source that
    could not be read is a source that could not be read, and the module does not
    special-case the empty one."""
    d = tmp_path_factory.mktemp("evidence")
    (d / "zero_bytes.xml").write_bytes(b"")
    out = {"none": [], "missing": [str(d / "never_written.xml")],
           "zero_bytes": [str(d / "zero_bytes.xml")]}
    for name, text in EVIDENCE_CONTENTS:
        p = d / f"{name}.xml"
        p.write_bytes(text.encode("utf-8"))
        out[name] = [str(p)]
    return out


TOUCHED = ("diff --git a/styxx/app.py b/styxx/app.py\n--- a/styxx/app.py\n"
           "+++ b/styxx/app.py\n@@\n+def test_one(): pass\n")
OUTSIDE = ("diff --git a/other/y.py b/other/y.py\n--- a/other/y.py\n"
           "+++ b/other/y.py\n@@\n+y = 1\n")

# Deliberately including rows whose baseline is already FAIL (a path lie, a
# symbol lie, a count lie) and one whose baseline is UNMEASURED. Monotonicity
# has to hold on the unhappy rows or it is a statement about the happy path
# wearing the word "property".
CASES = [
    ("bare", "All tests pass.", TOUCHED),
    ("with_a_true_path_claim", "Modified styxx/app.py. All tests pass.", TOUCHED),
    ("with_a_path_lie", "This change only touches styxx/. All tests pass.", OUTSIDE),
    ("with_a_symbol_lie", "Adds function nonexistent_helper. Tests pass.", TOUCHED),
    ("with_a_count_lie", "3 files changed. All tests are passing.", TOUCHED),
    ("many_claims", "Modified styxx/app.py, added 1 tests, adds function test_one. "
                    "All tests pass.", TOUCHED),
    ("repeated_claims", "All tests pass. All tests pass. All tests pass.", TOUCHED),
    ("unreadable_diff", "All tests pass.", "Sorry, I could not produce a diff."),
    ("no_tests_pass_claim", "Modified styxx/app.py.", TOUCHED),
]


def tp(g):
    return [c for c in g.claims if c.kind == "tests_pass"]


# ==================================================================== probes
# Every marker's condition is a probe of the CURRENTLY LOADED module, so the
# marker disarms itself when the defect it names is repaired, and strict=True
# makes an unexplained pass a failure. All four probes are behavioural; the
# run-branch probe falls back to a coarse source read only if the call raised,
# and that overlap with the assertion it guards is disclosed rather than hidden.


class _Stub:
    """Stands in for subprocess.run so that no shell is ever spawned by this
    file. `--run` passes shell=True with cwd set to the tree under test, and
    executing that from a test suite is the code-execution path the module's own
    CLI help now warns about at diffgate.py:936. This file reads it, never runs
    it."""

    def __init__(self, returncode=7, raises=None):
        self.returncode, self.raises, self.calls = returncode, raises, 0

    def __call__(self, *a, **kw):
        self.calls += 1
        if self.raises is not None:
            raise self.raises
        return types.SimpleNamespace(returncode=self.returncode, stdout="", stderr="")


def _with_stub(stub, fn):
    """Synchronous, always restored. Probes run at import where monkeypatch does
    not exist yet, so the same helper serves both."""
    real = D.subprocess.run
    D.subprocess.run = stub
    try:
        return fn()
    finally:
        D.subprocess.run = real


def _run_gate(rc=0, raises=None, summary="All tests pass.", **kw):
    stub = _Stub(returncode=rc, raises=raises)
    g = _with_stub(stub, lambda: gate_diff_text(summary, TOUCHED, run="pytest -q",
                                                repo=".", **kw))
    return g, stub


def _probe_run_branch_accuses() -> bool:
    try:
        g, _ = _run_gate(rc=7)
        return any(c.verdict == ACCUSE for c in tp(g))
    except Exception:
        return bool(run_branch_accusation_sites(inspect.getsource(D)))


def _probe_run_executes_more_than_once() -> bool:
    try:
        _, stub = _run_gate(rc=0, summary="All tests pass.\nAll tests pass.\n"
                                          "All tests pass.")
    except Exception:
        return True
    return stub.calls > 1


def _probe_timeout_escapes() -> bool:
    try:
        _run_gate(raises=subprocess.TimeoutExpired(cmd="pytest -q", timeout=1))
    except Exception:
        return True
    return False


def _probe_tests_pass_exempt_from_no_evidence() -> bool:
    """The exemption at the old diffgate.py:357 (`no_evidence and kind !=
    "tests_pass"`) let a claim escape the branch that says the gate had nothing
    to read."""
    try:
        g = gate_diff_text("All tests pass.", "Sorry, I could not produce a diff.")
        return any("carries no file statuses" not in c.why for c in tp(g))
    except Exception:
        return True


EVIDENCE_WIRED = "evidence" in inspect.signature(gate_diff_text).parameters
RUN_BRANCH_ACCUSES = _probe_run_branch_accuses()
RUN_EXECUTES_PER_MATCH = _probe_run_executes_more_than_once()
TIMEOUT_ESCAPES = _probe_timeout_escapes()
TESTS_PASS_EXEMPT = _probe_tests_pass_exempt_from_no_evidence()

owed_wiring = pytest.mark.xfail(not EVIDENCE_WIRED, strict=True, reason=(
    "OWED: gate_diff_text takes no `evidence` parameter. Self-arming strict "
    "xfail -- it disarms when the parameter appears, and an unexplained pass "
    "before then is a FAILURE, not a quiet XPASS."))
owed_run = pytest.mark.xfail(RUN_BRANCH_ACCUSES, strict=True, reason=(
    "OWED: the DECISION of 2026-09-01 deletes the accusing half of the --run "
    "branch. A nonzero exit still resolves to an accusation."))


# ==============================================================================
# THE TRIPWIRE -- the one thing the self-arming markers cannot do for themselves
# ==============================================================================
#
# Every marker above is a LATCH: it arms on the defect it guards and disarms once
# that defect is repaired. A latch cannot tell "not fixed yet" from "fixed, then
# REGRESSED" -- both present as the defect being live, and in both cases the
# marker absorbs its own tests into xfail and pytest exits 0.
#
# Demonstrated on 2026-09-01 by adversarial review: reintroducing the
# TimeoutExpired defect this file was written to guard produced
# "277 passed, 2 xfailed", exit 0, three runs running -- the regression rendered
# as an OWED note. The four latches also explain the flip seen the same morning,
# 230 xfailed -> 279 passed (owed_wiring 220 + owed_run 8 + per-match 1 +
# timeout 1), when a concurrent rewrite disarmed all four at once.
#
# The module docstring's promise -- "there is no configuration under which a test
# in this file is silently green" -- is only TRUE with this test present. Each
# probe's repaired value is pinned below as a constant, so re-arming any latch
# now reds the suite instead of quietly widening it.
#
# When a new owed defect is added, add its probe here with the value it is
# expected to hold ONCE REPAIRED. That is what carries the repair forward.

_PINNED_PROBES = {
    "EVIDENCE_WIRED": True,
    "RUN_BRANCH_ACCUSES": False,
    "RUN_EXECUTES_PER_MATCH": False,
    "TIMEOUT_ESCAPES": False,
    "TESTS_PASS_EXEMPT": False,
}


def test_no_self_arming_marker_has_re_armed():
    """A latch that re-arms is a REGRESSION, not an outstanding debt.

    This is the only test in the file that fails when a REPAIRED defect comes
    back. Every other guard on those four defects is behind a marker that arms
    on exactly the condition it is meant to report.
    """
    observed = {name: globals()[name] for name in _PINNED_PROBES}
    regressed = {n: v for n, v in observed.items() if v is not _PINNED_PROBES[n]}
    assert not regressed, (
        "a self-arming xfail marker has RE-ARMED, which means a repair that had "
        "already landed has regressed. The markers cannot report this themselves "
        "-- they absorb it into xfail and exit 0. Expected "
        f"{ {n: _PINNED_PROBES[n] for n in regressed} }, observed {regressed}.")


# ====================================== LIVE: the helpers are watched biting

def _snippet(text: str) -> str:
    return text.replace("@A@", ACCUSE)


_MUTANT_REACHED = _snippet('''
def _gate(run=None):
    if run:
        r = subprocess.run(run, shell=True)
        c.verdict = "VERIFIED" if r.returncode == 0 else "@A@"
''')

_MUTANT_UNREACHED = _snippet('''
def _gate(run=None):
    """A branch nobody reaches today is a branch someone reaches after a
    refactor. No output-level test can see this one."""
    if run:
        r = subprocess.run(run, shell=True)
        if False and r.returncode != 0:
            c.verdict = "@A@"
        else:
            c.verdict = "UNCHECKABLE"
''')

_MUTANT_FLAGGED = _snippet('''
WITHHOLD_TESTS_PASS_ACCUSATION = True

def _gate(run=None):
    if run:
        r = subprocess.run(run, shell=True)
        if r.returncode != 0 and not WITHHOLD_TESTS_PASS_ACCUSATION:
            c.verdict = "@A@"
''')

_MUTANT_OBSERVED = _snippet('''
def _evidence_leg(ev):
    observed = ev.get("observed") or {}
    if observed.get("failing"):
        return "@A@", "the attested report reads red"
    return "UNCHECKABLE", "no"
''')

_MUTANT_KIND_SCOPED = _snippet('''
def _gate():
    for kind, rx in TEMPLATES:
        if kind == "tests_pass":
            if evidence_says_red:
                c.verdict = "@A@"
''')

# Here the word appears only as prose about its own absence. A helper that fires
# on this is a grep with extra steps, and would refuse the very paragraphs this
# codebase writes to explain a deletion.
_INNOCENT = _snippet('''
def _run_leg(run, repo):
    """The accusing verdict @A@ is DELETED here, not gated.

    A nonzero returncode is not evidence that the author lied: it is also what
    "no tests collected", a missing dependency and a flaky test produce.
    """
    r = subprocess.run(run, shell=True, cwd=repo)          # @A@ deleted
    return ("VERIFIED", "exited 0") if r.returncode == 0 else ("UNCHECKABLE", "x")
''')

# The elif-chain shape that produced six false positives in the first draft of
# `_guard_parts`. A path-claim accusation lives in an arm ABOVE the tests_pass
# arm, and in the AST the tests_pass arm sits inside its `orelse` -- so a helper
# that inspects whole enclosing statements rather than their guards reports it
# as a tests_pass accusation. It is not one.
_ELIF_CHAIN = _snippet('''
def _gate():
    for kind in kinds:
        if kind == "only_touches":
            c.verdict = "VERIFIED" if not outside else "@A@"
        elif kind == "tests_pass":
            c.verdict = tests_pass_leg()
''')


@pytest.mark.parametrize("label,src,helper", [
    ("reached", _MUTANT_REACHED, run_branch_accusation_sites),
    ("unreached-guard", _MUTANT_UNREACHED, run_branch_accusation_sites),
    ("withheld-behind-a-flag", _MUTANT_FLAGGED, run_branch_accusation_sites),
    ("report-only-band", _MUTANT_OBSERVED, observed_band_accusation_sites),
    ("kind-scoped", _MUTANT_KIND_SCOPED, accusation_sites_under_tests_pass),
])
def test_the_source_helpers_detect_what_they_claim_to_detect(label, src, helper):
    """Every structural assertion in this file is worth exactly what these
    helpers are worth, so each is watched firing against a planted accusation
    before it is trusted to report an absence.

    Three of the five are shapes no output-level test can see: a branch behind
    ``if False``, a branch behind a flag set to off, and an accusation invented
    by a caller out of the report-only band. WITHHOLD_PATH_ACCUSATION is the
    flag shape already in this repository, and PREREG_evidence_leg indicts it by
    name -- flags get flipped by people who did not read the paper."""
    assert helper(src), f"the {label} mutant went undetected"


def test_the_helpers_do_not_fire_on_prose_explaining_the_absence():
    """Mention is not use. ANALYSIS_base_rate_ceiling records this lab's own
    certificate failing on exactly this distinction, and a checker that repeats
    it would forbid the paragraphs that explain a deletion."""
    assert run_branch_accusation_sites(_INNOCENT) == []
    assert accusation_sites_under_tests_pass(_INNOCENT) == []
    assert observed_band_accusation_sites(_INNOCENT) == []


def test_the_kind_scoped_helper_does_not_blame_a_neighbouring_elif_arm():
    """The false positive that the first draft of this file shipped, pinned so
    it cannot come back. In the AST an ``elif`` arm nests inside the previous
    arm's ``orelse``, so `tests_pass` is inside the SUBTREE of the
    `only_touches` arm -- and a helper that reads whole enclosing statements
    instead of their guards reports the path-claim accusation as a `tests_pass`
    accusation. Six such reports against the shipped module."""
    assert accusation_sites_under_tests_pass(_ELIF_CHAIN) == []
    assert run_branch_accusation_sites(_ELIF_CHAIN) == []


# ============================ LIVE: the caller-side hole styxx.evidence names

def test_no_diffgate_path_maps_the_report_only_band_to_an_accusation():
    """``styxx.evidence.selfcheck_no_accusation`` states its own boundary: it
    "does not prove a caller has not invented an accusation of its own out of
    the report-only `observed` band". diffgate is that caller, and this is the
    caller-side counterpart -- the tripwire PREREG_evidence_leg's G-E6 was meant
    to be, relocated to the file where such a branch would actually live."""
    sites = observed_band_accusation_sites(inspect.getsource(D))
    assert sites == [], f"diffgate turns the report-only band into a verdict: {sites}"


def test_the_evidence_module_still_refuses_to_accuse():
    """The cheapest way to wire this leg wrongly is to give styxx.evidence its
    accusing verdict back and leave diffgate innocent. Its self-check is
    re-derived here so this file fails if that happens."""
    assert tuple(E.VERDICTS) == ("VERIFIED", "UNCHECKABLE"), E.VERDICTS
    sc = E.selfcheck_no_accusation()
    assert sc["ok"] is True, sc
    assert sc["code_occurrences"] == []


def test_diffgates_own_selfcheck_agrees_with_this_files_independent_read():
    """SECONDARY, and deliberately not the primary. A module grading itself is
    not evidence; the AST helpers above are this file's own instrument and the
    assertions that matter run through them. What this adds is a disagreement
    alarm: if the two readers ever diverge, one of them is wrong and both are
    worth looking at."""
    sc = D.selfcheck_tests_pass_never_accuses()
    assert sc["ok"] is True, sc
    assert sc["code_occurrences"] == [] and sc["band_reads"] == []
    assert sc["missing"] == [], (
        "the self-check names functions that no longer exist, so it is scanning "
        f"nothing: {sc['missing']}")
    assert tuple(D._TESTS_PASS_VERDICTS) == ("VERIFIED", "UNCHECKABLE")


# ============================ LIVE: the baseline the property is measured against

@pytest.mark.parametrize("label,summary,diff", CASES)
@pytest.mark.parametrize("strict", [False, True])
def test_without_evidence_tests_pass_is_uncheckable_and_says_why(label, summary,
                                                                 diff, strict):
    """With no report and no --run there is no execution evidence, and
    UNCHECKABLE is the honest answer -- ANALYSIS_base_rate_ceiling section 10
    counts 5,514 claims parked exactly there corpus-wide with an accusation
    count of zero. This is the EMPTY-SET BASELINE that the guarantee below is
    measured against -- the only baseline it is measured against; if it drifts,
    that test is comparing against something else."""
    g = gate_diff_text(summary, diff, strict=strict)
    for c in tp(g):
        assert c.verdict == "UNCHECKABLE", f"{label}: {c.verdict} -- {c.why}"
        assert c.why, "an UNCHECKABLE verdict must say why"


@pytest.mark.parametrize("label,summary,diff", CASES)
def test_the_gate_never_takes_the_agents_word_for_a_test_result(label, summary, diff):
    """The inverse failure of the accusation, and the one this module exists
    against: believing "all tests pass" because it was written down."""
    g = gate_diff_text(summary, diff)
    for c in tp(g):
        assert c.verdict != "VERIFIED", (
            f"{label}: the gate certified a test result from prose alone")


@pytest.mark.parametrize("label,summary,diff", CASES)
def test_the_tests_pass_vocabulary_is_a_subset_of_evidence_VERDICTS(label, summary,
                                                                    diff):
    """Whatever diffgate says about a `tests_pass` claim must be a word
    styxx.evidence would say. Two words. A wiring that invents a third has
    invented an accusation under another name."""
    g = gate_diff_text(summary, diff)
    for c in tp(g):
        assert c.verdict in set(E.VERDICTS), c.verdict


def test_the_retired_why_string_is_gone():
    """PREREG_evidence_leg committed to replacing "no --run command supplied",
    which implied the remedy was to supply a shell string -- the one thing this
    gate must not encourage against a branch you did not write. Strictly more
    informative at an identical verdict, which is why it shipped ahead of any
    measurement."""
    g = gate_diff_text("All tests pass.", TOUCHED)
    for c in tp(g):
        assert "no --run command supplied" not in c.why
        assert "--evidence" in c.why, (
            "the reader has to be told which channel can answer this")


# ================================================ LIVE/OWED: the --run branch

@owed_run
def test_the_run_branch_cannot_reach_an_accusation_in_the_source():
    """THE structural one. DECISION of 2026-09-01: the accusing half of the
    --run branch is DELETED, not withheld behind a flag, because
    WITHHOLD_PATH_ACCUSATION is the counter-example this codebase already
    carries and PREREG_evidence_leg indicts by name. Absence of a branch cannot
    be toggled by someone in a hurry.

    Asserted over the SOURCE. The behavioural companion below would pass on a
    module carrying the branch behind ``if False`` or behind a flag set to off,
    and both of those mutants are watched being caught above rather than here."""
    sites = run_branch_accusation_sites(inspect.getsource(D))
    assert sites == [], (
        "the accusing verdict is reachable from a returncode comparison: "
        f"{sites}. Its precision has never been measured by a blind panel on "
        "either leg -- not extraction, not adjudication.")


@owed_run
def test_no_accusation_is_produced_anywhere_under_a_tests_pass_guard():
    """Wider than the returncode form: any accusation produced under a
    `tests_pass` guard, whatever supplies it."""
    sites = accusation_sites_under_tests_pass(inspect.getsource(D))
    assert sites == [], sites


@owed_run
@pytest.mark.parametrize("rc", [1, 4, 5, 127])
def test_a_failing_run_command_does_not_accuse(rc):
    """The behavioural companion, kept and explicitly INSUFFICIENT ALONE, and
    the mutant sweep measured exactly how insufficient.

    Restoring the accusing verdict inside `_run_leg` -- reached (M1), unreached
    (M2), or behind a flag set to off (M3) -- changes no output at all, because
    `_tests_pass_verdict` folds the leg's answer with a promote-to-VERIFIED-only
    rule and the `_TESTS_PASS_VERDICTS` clamp rewrites anything else. Removing
    the accusing branch AND the fold still changes nothing (M16). Only M17,
    which removes all three layers, makes this test go red. So against today's
    module it is unfalsifiable by any single-edit mutant: it is a regression
    guard for the refactor that flattens those layers, not evidence of anything
    now. The structural test above is the one that bites, on all five.

    The exit codes are chosen, not arbitrary: pytest rc=5 is "no tests
    collected", rc=4 a usage error, 127 a misspelled command. Each was read as a
    lie by the shipped code, which is the whole argument -- a nonzero exit is a
    fact about a process, and on an untrusted branch it is a process the author
    controls in both directions from their own conftest.py."""
    g, _ = _run_gate(rc=rc)
    for c in tp(g):
        assert c.verdict != ACCUSE, f"rc={rc} produced an accusation: {c.why}"
    assert g.verdict == "PASS", f"rc={rc} failed the build on an unmeasured verdict"


@owed_run
def test_the_why_for_a_nonzero_exit_names_what_else_produces_one():
    """A verdict that declines has to say why in terms a reader can act on. The
    DECISION pins the sentence: a nonzero exit is also what "no tests
    collected", a misspelled command, a missing dependency and a flaky test
    produce."""
    g, _ = _run_gate(rc=5)
    whys = [c.why.lower() for c in tp(g)]
    assert whys
    for why in whys:
        assert "exited 5" in why, "the exit code belongs in the record"
        assert any(k in why for k in ("no tests collected", "misspelled",
                                      "missing dependency", "flaky")), why


@owed_run
def test_a_run_that_exits_zero_verifies_but_may_not_claim_the_tests_passed():
    """VERIFIED survives on evidence.py's stated asymmetry -- a wrong VERIFIED
    repeats a claim the author already made in prose, a wrong accusation attacks
    a stranger inside their own pull request. The disclosed defect rides with
    it: the same extraction that fires on "Not all tests pass." would stamp
    VERIFIED on a sentence asserting the opposite, so the why-string is required
    to disclaim rather than to be trusted."""
    g, _ = _run_gate(rc=0)
    claims = tp(g)
    assert claims
    for c in claims:
        assert c.verdict == "VERIFIED"
        assert "exited 0" in c.why
        low = c.why.lower()
        # A disjunction over phrasings rather than one pinned sentence: what is
        # load-bearing is that the record disclaims, not which words it uses.
        assert any(k in low for k in ("does not mean the tests passed",
                                      "not a statement about the suite",
                                      "checks nothing about the sentence")), c.why
        assert any(k in low for k in ("checkbox", "negation", "quotation")), (
            "the disclosed extraction defect has to ride with the verdict: this "
            "template fires on unchecked PR-template boxes and on negations, so "
            "a VERIFIED can attach to a sentence asserting the opposite")


@pytest.mark.xfail(RUN_EXECUTES_PER_MATCH, strict=True, reason=(
    "OWED: the command runs once per REGEX MATCH rather than once per gate. A "
    "body repeating the sentence is unbounded, author-controlled runner-hour "
    "amplification -- each launch carrying its own 1800s budget."))
def test_the_run_command_executes_at_most_once_per_gate():
    _, stub = _run_gate(rc=0, summary="All tests pass.\nAll tests pass.\n"
                                      "All tests pass.")
    assert stub.calls <= 1, f"{stub.calls} executions for one gate invocation"


@pytest.mark.xfail(TIMEOUT_ESCAPES, strict=True, reason=(
    "OWED: subprocess.TimeoutExpired propagates uncaught out of _gate. A "
    "traceback is not a verdict."))
def test_a_run_timeout_is_uncheckable_not_a_traceback():
    g, _ = _run_gate(raises=subprocess.TimeoutExpired(cmd="pytest -q", timeout=1800))
    claims = tp(g)
    assert claims
    for c in claims:
        assert c.verdict == "UNCHECKABLE"
        assert c.why


def test_run_with_no_repo_refuses_rather_than_running_in_the_verifiers_own_tree():
    """The quiet defect on the zero-receipt path: gate_diff_text takes repo=None
    by default and subprocess.run(cwd=None) executes wherever the verifier
    happens to be standing. The one entry point built for having no checkout was
    the one where cwd silently became the operator's own tree."""
    stub = _Stub(returncode=0)
    g = _with_stub(stub, lambda: gate_diff_text("All tests pass.", TOUCHED,
                                                run="pytest -q"))
    assert stub.calls == 0, "the command was executed with no repository to run it in"
    for c in tp(g):
        assert c.verdict == "UNCHECKABLE"
        assert "verifier" in c.why.lower() or "cwd" in c.why.lower()


@pytest.mark.xfail(TESTS_PASS_EXEMPT, strict=True, reason=(
    "OWED: the `and kind != \"tests_pass\"` exemption lets a tests_pass claim "
    "escape the branch that says the gate had nothing to read."))
def test_a_diff_that_parsed_to_nothing_leaves_every_claim_unread():
    """DECISION repair 1. With input that parsed to nothing the gate reports
    measured=False, and no claim -- `tests_pass` included -- may be resolved
    against evidence the gate does not have.

    A tension worth recording rather than hiding: a test report is arguably
    independent of whether the DIFF was readable, so it is defensible for
    --evidence to answer here. The shipped module says no, the DECISION's
    repair 1 says no, and this pins the shipped reading. Whoever changes it
    should change this test deliberately."""
    g = gate_diff_text("All tests pass.", "Sorry, I could not produce a diff.")
    assert g.measured is False
    for c in g.claims:
        assert c.verdict == "UNCHECKABLE"
        assert g.why_unmeasured in c.why or "carries no file" in c.why


# ============================================= LIVE/OWED: the evidence wiring

@owed_wiring
def test_a_supporting_report_moves_tests_pass_off_uncheckable(ev):
    g = gate_diff_text("Modified styxx/app.py. All tests pass.", TOUCHED,
                       evidence=ev["green"])
    claims = tp(g)
    assert claims, "the claim must still be extracted"
    for c in claims:
        assert c.verdict == "VERIFIED", f"{c.verdict} -- {c.why}"
    assert g.verdict == "PASS"


@owed_wiring
@pytest.mark.parametrize("name", CONTENT_NAMES + ["missing"])
def test_the_verdict_is_the_adjudicators_verdict_not_a_second_opinion(ev, name):
    """The gate may not re-derive an answer styxx.evidence already gave. Same
    bytes, same verdict -- otherwise there are two adjudicators and only one of
    them has a docstring saying what VERIFIED does not mean."""
    want, _ = adjudicate_tests_pass(load_evidence(ev[name]), None)
    g = gate_diff_text("All tests pass.", TOUCHED, evidence=ev[name])
    claims = tp(g)
    assert claims
    for c in claims:
        assert c.verdict == want, f"{name}: gate {c.verdict}, evidence {want}"


@owed_wiring
@pytest.mark.parametrize("name", CONTENT_NAMES + ["missing"])
def test_the_adjudicators_reason_is_preserved_verbatim(ev, name):
    """A paraphrase is a second opinion wearing the first one's clothes. The
    reason the reader acts on has to be the one styxx.evidence wrote -- the
    string that says an all-skipped suite is not a green run, or that a partial
    read is a guess with a citation. diffgate may add context around it; it may
    not restate it."""
    _, want_why = adjudicate_tests_pass(load_evidence(ev[name]), None)
    g = gate_diff_text("All tests pass.", TOUCHED, evidence=ev[name])
    for c in tp(g):
        assert want_why in c.why, (
            f"{name}: the adjudicator's reason was paraphrased away:\n"
            f"  gate:     {c.why}\n  evidence: {want_why}")


@owed_wiring
def test_a_supporting_report_is_not_read_as_the_tests_passed(ev):
    """VERIFIED means an attestation reports PASSED and no signature was
    checked. It does not mean the tests passed, and it says nothing about
    whether the sentence extracted was a claim at all -- 179 corpus-wide
    `tests_pass` extractions sit inside an UNCHECKED PR-template box where the
    author is explicitly declining to assert."""
    g = gate_diff_text("All tests pass.", TOUCHED, evidence=ev["green"])
    for c in tp(g):
        low = c.why.lower()
        assert "no signature was checked" in low or "not mean the tests passed" in low, \
            c.why


@owed_wiring
def test_no_report_supplied_leaves_it_uncheckable_and_names_the_channels(ev):
    """"--evidence with no report" has two readings and they are different
    absences. Nothing supplied at all is answered by the gate's own named
    reason; a path that does not exist is answered by the adjudicator, and that
    case is covered by the reason-preserved test above."""
    g = gate_diff_text("All tests pass.", TOUCHED, evidence=ev["none"])
    claims = tp(g)
    assert claims
    for c in claims:
        assert c.verdict == "UNCHECKABLE"
        assert c.why, "an UNCHECKABLE verdict must say why"
        assert "no --run command supplied" not in c.why
        assert "--evidence" in c.why
        assert "no accusing verdict" in c.why.lower(), (
            "the absence of the third word is part of the answer, not trivia")


@owed_wiring
@pytest.mark.parametrize("name", CONTENT_NAMES + ["none", "missing"])
@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("commit", [None, HEAD_COMMIT, OTHER_COMMIT])
def test_no_evidence_content_can_produce_an_accusation(ev, name, strict, commit):
    """By any route, for any content, at any commit, strict and not.

    `red`, `collection_error` and `forged_failed` are the readings where the
    evidence looks unambiguously bad, and `forged_failed` is an accusation
    against a named commit that costs one hand-written JSON file and no key
    material. None may become a verdict against the author. There is no
    sub-population of evidence contents on which this relaxes."""
    g = gate("Modified styxx/app.py. All tests pass.", TOUCHED,
             evidence=ev[name], commit=commit, strict=strict)
    for c in tp(g):
        assert c.verdict != ACCUSE, f"{name}/{commit}: {c.why}"
        assert c.verdict in set(E.VERDICTS), c.verdict


@owed_wiring
@pytest.mark.parametrize("label,summary,diff", CASES)
@pytest.mark.parametrize("name", CONTENT_NAMES + ["missing"])
@pytest.mark.parametrize("strict", [False, True])
def test_adding_evidence_never_worsens_the_gate_against_no_evidence(
        ev, label, summary, diff, name, strict):
    """THE LOAD-BEARING ONE, AND IT IS THE WEAKER OF THE TWO READINGS.

    RENAMED AND RESCOPED on 2026-09-01. This test shipped as
    ``test_adding_evidence_never_worsens_the_gate_verdict``, and under that name
    it was read as establishing that evidence never worsens a verdict FULL STOP.
    It never measured that and it never could: the only baseline it compares
    against is the EMPTY evidence set. The body is unchanged and correct. The
    name and the docstring were the defect, and they are corrected here rather
    than deleted, because the record of what a green test was taken to mean is
    part of what the test is for.

    WHAT IS ASSERTED. Over every claim combination, every evidence content and
    both strictness settings: the gate handed evidence ranks no worse than the
    same gate handed NONE -- at the overall verdict, and at every single claim.
    Evidence may move a `tests_pass` claim UPWARD, from UNCHECKABLE to VERIFIED.
    Relative to supplying nothing it may never move a claim downward, and it may
    never turn a would-have-been PASS into a FAIL.

    WHAT IS NOT ASSERTED, and is FALSE:
    ``E -> E union {x}`` never worsens. It can and it does, deliberately, and
    ``test_extending_an_evidence_set_can_demote_verified_and_must`` pins that so
    the two are never again read off one report line.

    This is what makes the leg safe to ship before Panel A exists: an adjudicator
    that can only resolve upward FROM THE NO-EVIDENCE BASELINE cannot manufacture
    an accusation however wrong the extraction feeding it is -- and that
    extraction carries a mechanical non-assertion indicator on 6.27% of matches,
    with the other 83% unmeasured. Whether such a match is a WRONG extraction is
    Panel A's question, not the census's, and Panel A has not been run.

    `_gate` argues this invariant in a comment. A comment is not a receipt."""
    base = gate_diff_text(summary, diff, strict=strict)
    with_ev = gate_diff_text(summary, diff, evidence=ev[name], strict=strict)

    assert GATE_RANK[with_ev.verdict] <= GATE_RANK[base.verdict], (
        f"{label}+{name}+strict={strict}: {base.verdict} -> {with_ev.verdict}")
    assert [c.kind for c in with_ev.claims] == [c.kind for c in base.claims], (
        "evidence added or removed claims; it is an adjudicator, not an extractor")
    for a, b in zip(base.claims, with_ev.claims):
        assert CLAIM_RANK[b.verdict] <= CLAIM_RANK[a.verdict], (
            f"{label}+{name}: {a.kind} {a.verdict} -> {b.verdict} ({b.why})")
    assert with_ev.measured == base.measured, (
        "evidence about tests cannot make a diff readable or unreadable")


# The grid for the pin below. Each row is an addition to an already-VERIFIED
# evidence set, paired with the mechanism by which the contract refuses to keep
# affirming once it is present. `demotes=False` rows are NEGATIVE CONTROLS and
# are as load-bearing as the rest: a rule that demoted on any addition at all
# would be counting files rather than reading them, and would be indistinguishable
# from the real rule on a grid made only of demoting rows.
SET_EXTENSIONS = [
    # name,             demotes, mechanism
    ("zero_bytes",      True,  "a file that exists and holds no report"),
    ("missing",         True,  "a shard whose upload never arrived"),
    ("unparsable",      True,  "bytes that are neither XML nor JSON"),
    ("red",             True,  "a second source that disagrees (FAILED, PASSED)"),
    ("forged_failed",   True,  "a hand-written FAILED attestation, no key material"),
    ("collection_error", True, "a harness error: the runner broke, nothing lied"),
    ("empty",           False, "a well-formed zero-test report: parses, adds nothing"),
]


@owed_wiring
@pytest.mark.parametrize("name,demotes,mechanism", SET_EXTENSIONS)
@pytest.mark.parametrize("strict", [False, True])
def test_extending_an_evidence_set_can_demote_verified_and_must(
        ev, name, demotes, mechanism, strict):
    """THE PIN. THIS TEST GOING RED MEANS THE SAFETY RULE WAS WEAKENED.

    Read this before you "fix" anything it reports.

    Adding evidence to an evidence set that already resolved VERIFIED can move
    that claim back to UNCHECKABLE, and under --strict can move the gate from
    PASS to FAIL. That is NOT a monotonicity bug. It is the contract's central
    rule, and this test exists so that the rule cannot be quietly removed by
    someone who read the sibling test's name -- "never worsens" -- as a promise
    that this could not happen.

    THE RULE: a partial read may honestly DECLINE, but it may not honestly
    AFFIRM. VERIFIED on a `tests_pass` claim asserts that the evidence in hand
    says every test passed. If one of the supplied sources could not be read,
    the evidence in hand is a subset of the evidence that was offered, and no
    subset licenses a statement about the whole. You cannot certify "all tests
    pass" from nine shards out of ten, and the tenth shard is exactly where a
    real CI failure hides -- an upload that timed out, a runner that OOMed, a
    matrix leg that never started. Affirming from an incomplete evidence set is
    the single failure this module was built to refuse. The same reasoning
    covers the sources that DID parse and disagree: refusing beats picking, and
    refusing beats taking the union.

    HOW SOMEONE BREAKS THIS. Not maliciously -- by seeing a green run demoted to
    UNCHECKABLE by one junk file, calling it a regression, and making the
    adjudicator skip sources it cannot read, or treat an empty file as "no
    evidence supplied", or take the best available reading. Each of those turns
    "I read everything and it all passed" into "I read what I could and what I
    could read passed", which is a different sentence with the same word on it.
    If this test is red, that is what changed. The repair is to revert it. If
    you believe the rule itself is wrong, that is a preregistration and a panel,
    not a patch.

    WHAT IS MEASURED, because a pin that asserts nothing pins nothing: the
    baseline `[green]` is confirmed VERIFIED and PASS first -- otherwise a module
    that had stopped affirming altogether would pass this test trivially -- and
    then each addition is checked against the direction recorded in
    ``SET_EXTENSIONS``. 12 of the 14 (addition x strictness) cells demote; the 2
    that do not are the well-formed zero-test report, which parses and adds
    nothing, and it is in the grid to show the rule reads sources rather than
    counting them."""
    base = gate_diff_text("All tests pass.", TOUCHED, evidence=ev["green"],
                          strict=strict)
    base_claims = tp(base)
    assert base_claims, "no tests_pass claim was extracted; the grid measures nothing"
    assert all(c.verdict == "VERIFIED" for c in base_claims), (
        "the baseline of this test is a green report resolving VERIFIED. It did "
        f"not: {[(c.verdict, c.why) for c in base_claims]}. Until that is true "
        "again this test cannot say anything about set extension.")
    assert base.verdict == "PASS", base.verdict

    ext = gate_diff_text("All tests pass.", TOUCHED,
                         evidence=ev["green"] + ev[name], strict=strict)
    ext_claims = tp(ext)
    assert [c.kind for c in ext.claims] == [c.kind for c in base.claims], (
        "extending the evidence set added or removed claims; the adjudicator "
        "reached the extractor")

    for c in ext_claims:
        assert c.verdict != ACCUSE, (
            f"{name}: extending the evidence set produced an accusation. The "
            "demotion this test pins is VERIFIED -> UNCHECKABLE and stops "
            f"there. {c.why}")

    if demotes:
        assert all(c.verdict == "UNCHECKABLE" for c in ext_claims), (
            f"SAFETY RULE WEAKENED. Adding {name} ({mechanism}) to a green "
            "evidence set left the `tests_pass` claim at "
            f"{[c.verdict for c in ext_claims]}, so the gate is now certifying "
            "'all tests pass' from an evidence set it did not fully read. A "
            "partial read may DECLINE; it may not AFFIRM. This is not a "
            "monotonicity bug to be fixed -- see this test's docstring. If the "
            "adjudicator was changed to skip unreadable sources, or to special-"
            "case empty files, or to take the best available reading, revert it.")
        if strict:
            assert ext.verdict == "FAIL", (
                f"SAFETY RULE WEAKENED. Under --strict, {name} ({mechanism}) "
                "demoted the claim but the gate still reported "
                f"{ext.verdict}. A --strict gate that passes on an UNCHECKABLE "
                "`tests_pass` claim has made strictness decorative.")
        assert any(c.why != b.why for c, b in zip(ext_claims, base_claims)), (
            "the verdict moved and the reason did not; the reader is not being "
            "told which source could not be read")
    else:
        assert all(c.verdict == "VERIFIED" for c in ext_claims), (
            f"NEGATIVE CONTROL FAILED. Adding {name} ({mechanism}) demoted the "
            f"claim to {[c.verdict for c in ext_claims]}. This source parses "
            "cleanly and contributes no failure, so nothing about it makes the "
            "read partial. A rule that demotes here is counting files rather "
            "than reading them, and the demotions this test pins would then be "
            "evidence of nothing.")
        assert ext.verdict == base.verdict, (ext.verdict, base.verdict)


@owed_wiring
@pytest.mark.parametrize("name", CONTENT_NAMES)
@pytest.mark.parametrize("commit", [None, HEAD_COMMIT, OTHER_COMMIT])
def test_evidence_never_worsens_the_run_leg_either(ev, name, commit):
    """The two channels are a disjunction, and the dangerous direction is the
    one where requiring BOTH would let adding a report turn a --strict PASS into
    a --strict FAIL. Measured against the NO-EVIDENCE baseline -- a green --run
    with nothing supplied on the evidence channel -- and so this is the run-leg
    instance of the guarantee that holds, not of the set-extension property that
    does not. Adding a SECOND report to a report already in hand is the pin
    above; adding a first one to a green run is this."""
    base, _ = _run_gate(rc=0, strict=True)
    with_ev = _with_stub(_Stub(returncode=0), lambda: gate(
        "All tests pass.", TOUCHED, run="pytest -q", repo=".",
        evidence=ev[name], commit=commit, strict=True))
    assert GATE_RANK[with_ev.verdict] <= GATE_RANK[base.verdict], (
        f"{name}/{commit}: adding a report broke a green run")
    for a, b in zip(base.claims, with_ev.claims):
        assert CLAIM_RANK[b.verdict] <= CLAIM_RANK[a.verdict], (a.verdict, b.verdict)


@owed_wiring
@pytest.mark.parametrize("name", CONTENT_NAMES)
def test_evidence_does_not_touch_any_other_claim_kind(ev, name):
    """A test report says nothing about whether the diff touched the file the
    summary named. If any non-`tests_pass` verdict or reason moves, the wiring
    has reached somewhere it was not invited."""
    summary = ("Modified styxx/app.py, added 1 tests, adds function test_one. "
               "This change only touches styxx/. All tests pass.")
    base = gate_diff_text(summary, TOUCHED)
    with_ev = gate_diff_text(summary, TOUCHED, evidence=ev[name])
    a = [(c.kind, c.verdict, c.why) for c in base.claims if c.kind != "tests_pass"]
    b = [(c.kind, c.verdict, c.why) for c in with_ev.claims if c.kind != "tests_pass"]
    assert a == b


@owed_wiring
def test_a_commit_the_evidence_does_not_name_stays_uncheckable(ev):
    """The commit has to reach the adjudicator or `--evidence-commit` is
    decoration. A green report that never names the change under review is a
    green report about something else -- and a report from another branch, or
    from last week, is the exact hazard styxx.evidence's binding section exists
    for."""
    want_verdict, want_why = adjudicate_tests_pass(
        load_evidence(ev["green"]), OTHER_COMMIT)
    assert want_verdict == "UNCHECKABLE"
    g = gate("All tests pass.", TOUCHED, evidence=ev["green"],
             commit=OTHER_COMMIT)
    claims = tp(g)
    assert claims
    for c in claims:
        assert c.verdict == "UNCHECKABLE", c.why
        assert want_why in c.why


@owed_wiring
def test_evidence_is_deterministic_over_the_same_bytes(ev):
    """styxx.evidence is a pure function of bytes and the wiring may not smuggle
    ambient state in around it. Same inputs, same record, twice."""
    a = gate_diff_text("All tests pass.", TOUCHED, evidence=ev["green"]).to_dict()
    b = gate_diff_text("All tests pass.", TOUCHED, evidence=ev["green"]).to_dict()
    assert a == b


@owed_wiring
def test_the_evidence_leg_runs_once_per_gate_not_once_per_match(ev):
    """Same repair as the --run leg, and the same reason: every `tests_pass`
    match in one summary asks the same question about the same suite. Counted
    by intercepting the loader rather than by reading the code."""
    calls = {"n": 0}
    real = E.load_evidence

    def counting(paths):
        calls["n"] += 1
        return real(paths)

    D_ev = D.styxx.evidence if hasattr(D, "styxx") else E
    E.load_evidence = counting
    try:
        gate_diff_text("All tests pass.\nAll tests pass.\nAll tests pass.",
                       TOUCHED, evidence=ev["green"])
    finally:
        E.load_evidence = real
    assert D_ev is E
    assert calls["n"] <= 1, f"{calls['n']} evidence reads for one gate invocation"


@owed_wiring
def test_it_uses_styxx_evidence_not_a_private_copy():
    """Asserted by source inspection, the way ``tests/test_undeclared.py``
    asserts reconcile parses the diff with the gate's own parser rather than a
    second one that can drift.

    Here the drift would be worse than a disagreement about diffs: a private
    reader of JUnit or in-toto bytes inside diffgate would be a reader with no
    ``VERDICTS`` tuple, no self-check, and no docstring saying what VERIFIED
    does not mean."""
    src = inspect.getsource(D)
    assert "from styxx.evidence import" in src, (
        "diffgate does not import styxx.evidence")
    assert "adjudicate_tests_pass" in src and "load_evidence" in src, (
        "diffgate does not call the adjudicator; it has an opinion of its own")

    tree = ast.parse(src)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "xml" not in imported, (
        "diffgate parses XML itself; that is a second evidence reader")

    # Identifiers only -- prose constants legitimately mention in-toto and JUnit,
    # and a substring grep over the whole file would refuse the paragraphs that
    # explain the delegation.
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    for smell in ("fromstring", "ElementTree", "predicateType", "failedTests",
                  "passedTests", "testsuite", "observed"):
        assert smell not in names, (
            f"diffgate appears to read evidence bytes itself ({smell!r}); that "
            "is a second adjudicator, and only one of them has a self-check")


@owed_wiring
def test_the_cli_exposes_the_evidence_channel_and_warns_about_the_other_one():
    """A library-only wiring is a wiring nobody uses. And the CLI is where the
    module advertises `--run`, which on an untrusted pull request is arbitrary
    code execution against the tree under test -- pytest imports the PR's
    conftest.py at collection, `npm test` runs the PR's package.json, and
    os.environ is inherited unscrubbed."""
    sig = inspect.signature(gate_diff_text).parameters
    assert "evidence" in sig
    assert _COMMIT_KW is not None, (
        "no commit-binding parameter: a green report from another branch, or "
        "from last week, would answer for this change")
    # Exact option strings, not substrings. A substring check passed a mutant
    # that renamed the flag to `--evidenceX`, because "--evidence" is a prefix of
    # it -- the same mention/use sloppiness this file exists to avoid, committed
    # by this file. Found by mutant M23.
    src = inspect.getsource(D.main)
    tree = ast.parse(src)
    helps = {}
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "add_argument" and n.args
                and isinstance(n.args[0], ast.Constant)):
            kw = {k.arg: k.value for k in n.keywords}
            h = kw.get("help")
            helps[n.args[0].value] = (h.value if isinstance(h, ast.Constant)
                                      and isinstance(h.value, str) else "")
    assert "--evidence" in helps, sorted(helps)
    assert set(helps) & {"--evidence-commit", "--commit"}, sorted(helps)
    assert "--run" in helps, sorted(helps)
    # Scoped to --run's OWN help. A whole-source search for "executes" was
    # satisfied by --evidence's help saying it "executes nothing", so a mutant
    # that stripped the warning off --run passed. Found by mutant M23b.
    run_help = helps["--run"].lower()
    assert "execut" in run_help, (
        "--run's own help must say out loud that it runs a shell command: on an "
        "untrusted pull request that is code execution against the tree under "
        f"test. Got: {helps['--run']!r}")
    assert "never an accusation" in run_help or "uncheckable" in run_help, (
        "--run's help must say which verdicts it can produce, because the one "
        "it cannot produce is the whole point")
