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
import re
import subprocess
import sys
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
    ("tests_added", re.compile(r"\b(?:add\w+|creat\w+)\s+(?P<n>\d+)\s+(?:new\s+)?tests?\b", re.I)),
    ("symbol_added", re.compile(
        r"\b(?:add\w+|introduc\w+)\s+(?:a\s+|the\s+)?(?P<kind>function|class|method)\s+"
        r"[`\"']?(?P<name>[A-Za-z_]\w*)", re.I)),
    ("only_touches", re.compile(
        r"\bonly\s+(?:touch\w+|modif\w+|chang\w+)\s+(?:files?\s+(?:in|under)\s+)?"
        r"[`\"']?(?P<prefix>[\w./\\-]+)[`\"']?", re.I)),
    ("tests_pass", re.compile(r"\b(?:all\s+)?tests\s+(?:pass|are\s+passing|green)\b", re.I)),
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
    for line in diff_text.splitlines():
        if line.startswith("--- "):
            old_path = line[4:].strip()
        elif line.startswith("+++ "):
            new = line[4:].strip()
            if new == "/dev/null":
                status[_norm(old_path[2:] if old_path.startswith("a/") else old_path)] = "D"
            elif old_path in ("/dev/null", None):
                status[_norm(new[2:] if new.startswith("b/") else new)] = "A"
            else:
                status[_norm(new[2:] if new.startswith("b/") else new)] = "M"
        elif line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:])
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
                 raw_input_len=len(diff_text or ""))


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
    added_lines = [l[1:] for l in _git(repo, "diff", f"{base}..{head}").splitlines()
                   if l.startswith("+") and not l.startswith("+++")]
    added_blob = "\n".join(added_lines)
    return _gate(summary_text, status, added_blob, run=run, strict=strict,
                 repo=repo, base=base, head=head,
                 evidence=evidence, commit=commit)


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
          raw_input_len: int | None = None) -> DiffGate:

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
                    got = len(re.findall(r"^\s*def test_", added_blob, re.M))
                    c.verdict = "VERIFIED" if got == n else "CONTRADICTED"
                    c.why = f"diff adds {got} test functions, claim says {n}"
                elif kind == "symbol_added":
                    pat = (r"^\s*(?:def|class)\s+" + re.escape(d["name"]) + r"\b")
                    hit = bool(re.search(pat, added_blob, re.M))
                    c.verdict = "VERIFIED" if hit else "CONTRADICTED"
                    c.why = (f"added lines {'do' if hit else 'do NOT'} define "
                             f"{d['kind']} {d['name']!r}")
                elif kind == "only_touches":
                    pref = _norm(d["prefix"]).rstrip("/.")   # sentence-final periods are not path
                    outside = [p for p in status if not p.startswith(pref + "/")
                               and p != pref]
                    if no_paths:
                        c.verdict, c.why = "UNCHECKABLE", no_paths
                    else:
                        c.verdict = "VERIFIED" if not outside else "CONTRADICTED"
                        c.why = ("all changed paths under prefix" if not outside else
                                 f"paths outside {pref!r}: {outside[:3]}")
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
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.demo:
        return _demo()
    if not a.summary or not a.base:
        ap.error("summary and --base are required (or try --demo)")
    if a.run:
        # Said out loud on every run, not left in --help for a reader to find.
        print(f"--run WILL EXECUTE {a.run!r} through a shell in {a.repo!r}. That "
              "is code execution; do not point it at a pull request you did not "
              "write. --evidence reads bytes and executes nothing.")
    text = Path(a.summary).read_text(encoding="utf-8")
    g = gate_diff(text, a.repo, a.base, a.head, run=a.run, strict=a.strict,
                  evidence=a.evidence, commit=a.commit)
    if a.out:
        Path(a.out).write_text(json.dumps(g.to_dict(), indent=2) + "\n", encoding="utf-8")
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
