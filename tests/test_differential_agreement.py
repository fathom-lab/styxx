"""The differential is a standing guard, not a number somebody once ran.

`papers/sworn/SPEC_differential_agreement_v01_2026_09_05.md` promises this file in D3 — "the
receipt carries the inputs, and a test re-derives a recorded case from its seed and asserts the
same bytes" — and the promise was unkept until now: the 150000-case run landed with no test at all
behind it. A spec that commits to a test and ships without one is the failure this corpus exists to
refuse, so it is refused here.

Two things are checked, and only one of them is about the recorded run.

**The recorded run.** Its receipt must be internally consistent, must actually pass the gates the
frozen spec set, and must still be the file the RESULT swears to. That is cheap and it is worth
little: it re-reads a number rather than re-earning it, and a receipt cannot be its own evidence.

**The standing guard, which is the point.** A differential test's value is not the run that was
published — it is that every future edit to either implementation gets differentially tested before
it lands. At roughly a thousand cases a second the guard can afford to be real, so it runs 5000
live cases through both shipped verifiers on a seed the recorded run never touched, and it demands
zero disagreements. If somebody changes a decimal comparison in `styxx/sworn.py` and not in
`styxx/_data/sworn_verify.js`, this fails, in this repository, before it is merged.

The guard carries its own G-C, for the reason the frozen spec made G-C a gate rather than a hope:
the first balanced grammar reached 1184 MALFORMED spans against a single HELD and no FAILED, which
fuzzes the lexer hard and the adjudicator barely, and the adjudicator is the half where a
disagreement would live. A guard that stopped reaching HELD would keep passing while measuring
nothing. So the guard asserts what it reached, not only that it agreed.

No digest computed today is pinned here. Pinning a number this file just produced would test that
the number is copied correctly and nothing else; every assertion below is a property — purity
across processes, agreement between two implementations, a vocabulary actually exercised — that a
future change can genuinely violate.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conformance.sworn.differential import case, js_digests, python_digest  # noqa: E402

RECEIPT = ROOT / "conformance" / "sworn" / "differential_agreement.json"
SPEC = ROOT / "papers" / "sworn" / "SPEC_differential_agreement_v01_2026_09_05.md"
RESULT = ROOT / "papers" / "sworn" / "RESULT_differential_agreement_2026_09_05.md"

R = json.loads(RECEIPT.read_text(encoding="utf-8"))

# A seed the recorded run never touched. Fixed, so a failure here is reproducible by anybody who
# reads this line; different from the recorded seed, so the guard is not re-running measured cases.
GUARD_SEED = 20260906
GUARD_CASES = 5000
GUARD_BATCH = 2500

needs_node = pytest.mark.skipif(shutil.which("node") is None,
                                reason="the JavaScript side needs node on PATH")


# ============================================================ D3: a case is a pure function

def _fingerprint(c):
    """The bytes of a case, in a form two processes can compare."""
    return {"index": c["index"], "document": c["document"].hex(), "name": c["name"],
            "commit": c["commit"],
            "manifest": None if c["manifest"] is None else json.dumps(c["manifest"], sort_keys=True,
                                                                     default=str)}


def test_a_case_is_the_same_bytes_every_time_it_is_generated():
    for i in (0, 1, 7, 999, 149999):
        assert _fingerprint(case(R["seed"], i)) == _fingerprint(case(R["seed"], i)), i


def test_two_indices_are_two_different_cases():
    """A generator that returns one document 150000 times would agree with itself perfectly."""
    seen = {json.dumps(_fingerprint(case(R["seed"], i)), sort_keys=True) for i in range(200)}
    assert len(seen) > 190, "the grammar is emitting the same case over and over: %d/200" % len(seen)


def test_a_case_is_a_pure_function_of_seed_and_index_across_processes():
    """D3's real content: reproducible from the seed ALONE, by somebody who does not have our
    process. A generator seeded off the clock, the PID or dict iteration order would pass every
    in-process check above and be worthless to anybody re-deriving a finding."""
    prog = (
        "import json,sys;sys.path.insert(0,%r);"
        "from conformance.sworn.differential import case;"
        "print(json.dumps([{'i':c['index'],'d':c['document'].hex(),'n':c['name'],"
        "'c':c['commit'],'m':json.dumps(c['manifest'],sort_keys=True,default=str)}"
        " for c in (case(%d,i) for i in (0,1,7,999,149999))]))" % (str(ROOT), R["seed"])
    )
    r = subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True,
                       encoding="utf-8", cwd=str(ROOT), timeout=300)
    assert r.returncode == 0, r.stderr[-800:]
    child = json.loads(r.stdout)
    mine = [{"i": c["index"], "d": c["document"].hex(), "n": c["name"], "c": c["commit"],
             "m": json.dumps(c["manifest"], sort_keys=True, default=str)}
            for c in (case(R["seed"], i) for i in (0, 1, 7, 999, 149999))]
    assert child == mine, "the generator is not a pure function of (seed, index)"


@needs_node
def test_every_recorded_disagreement_re_derives_from_its_seed_and_still_disagrees():
    """D3 literally, for the case the recorded run has none of.

    The run recorded zero disagreements, so there is nothing to re-derive and this test says so
    rather than passing silently on an empty list — an empty loop that reports success is how a
    guard rots. If a future run records one, this re-derives that exact case from the seed and the
    index alone and asserts the two implementations still part company there.
    """
    recorded = R["disagreements"]
    assert R["disagreements_total"] == len(recorded) == R["disagree"], R["disagreements_total"]
    if not recorded:
        assert R["disagree"] == 0
        pytest.skip("the recorded run has no disagreements to re-derive (agree=%d of %d)"
                    % (R["agree"], R["compared"]))
    with tempfile.TemporaryDirectory() as td:
        batch = [case(R["seed"], d["index"]) for d in recorded]
        js = js_digests(batch, Path(td))
        for d, c in zip(recorded, batch):
            pd, _pe, _cen = python_digest(c)
            row = js[c["index"]]
            assert pd != row.get("digest"), "case %d no longer disagrees" % d["index"]


# ============================================================ the standing guard

@needs_node
def test_the_two_implementations_agree_on_cases_the_recorded_run_never_saw():
    """The guard. 5000 live cases through both shipped verifiers, zero disagreements demanded.

    This is the test that earns its keep: it fails when somebody edits one implementation and not
    the other, which is the only way this format quietly acquires two meanings.
    """
    verdicts, doc_malformed, disagreements = Counter(), Counter(), []
    compared = 0
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        for off in range(0, GUARD_CASES, GUARD_BATCH):
            batch = [case(GUARD_SEED, i) for i in range(off, min(off + GUARD_BATCH, GUARD_CASES))]
            js = js_digests(batch, work)
            for c in batch:
                pd, pe, cen = python_digest(c)
                row = js[c["index"]]
                compared += 1
                if cen:
                    verdicts.update(cen["counts"])
                    if cen.get("document_malformed"):
                        doc_malformed[cen["document_malformed"]] += 1
                if (pd, bool(pe)) != (row.get("digest"), bool(row.get("error"))):
                    disagreements.append({"index": c["index"], "python": pd,
                                          "javascript": row.get("digest"),
                                          "python_error": pe, "javascript_error": row.get("error")})
    assert compared == GUARD_CASES, compared
    assert not disagreements, (
        "the two implementations disagree on %d of %d cases at seed %d — reproduce with "
        "conformance/sworn/differential.py --seed %d: %r"
        % (len(disagreements), compared, GUARD_SEED, GUARD_SEED, disagreements[:3]))

    # G-C in miniature. Without this the guard could pass while fuzzing only the lexer, which is
    # the exact degradation the frozen spec's G-C caught during the build.
    for v in ("HELD", "FAILED", "UNRESOLVED", "MALFORMED"):
        assert verdicts[v] > 0, (
            "the guard reached no %s span in %d cases, so it is measuring the generator and not "
            "the adjudicator: %r" % (v, GUARD_CASES, dict(verdicts)))
    assert doc_malformed, "the guard reached no document-level refusal in %d cases" % GUARD_CASES


# ============================================================ the recorded run's own receipt

def test_the_receipt_says_what_the_spec_named_it():
    assert R["schema"] == "styxx.sworn.differential-agreement/v1"
    assert R["spec"] == "papers/sworn/SPEC_differential_agreement_v01_2026_09_05.md"
    assert (ROOT / R["spec"]).exists(), "the receipt names a spec that is not in the tree"
    assert set(R["implementations"]) >= {"python", "javascript"}
    for name in ("python", "javascript"):
        side = R["implementations"][name]
        assert (ROOT / side["module"]).exists(), side["module"]
        assert len(side["sha256"]) == 64 and side["sha256"].islower()
    # The digests are content identity modulo newlines, which the receipt says out loud because a
    # reader hashing the file on a Windows checkout would otherwise get a different number and
    # think the implementation had moved.
    assert "modulo newlines" in R["implementations"]["note"]


def test_the_counts_add_up():
    assert R["compared"] == R["agree"] + R["disagree"]
    assert R["compared"] <= R["cases_requested"]
    assert R["gates"]["G-N"]["value"] == R["compared"]
    assert R["gates"]["G-A"]["value"] == R["agree"]
    assert R["gates"]["G-A"]["disagreements"] == R["disagree"]
    assert R["gates"]["G-C"]["value"] == R["census"]["span_verdicts"]
    assert R["gates"]["G-R"]["value"] == len(R["census"]["malformed_reasons"])
    assert R["void"] is False


def test_the_frozen_gates_are_the_ones_the_run_was_scored_against():
    """The bars are the spec's, not numbers chosen once the counts were in."""
    assert R["gates"]["G-N"]["bar"] == ">= 100000" and R["compared"] >= 100000
    assert R["gates"]["G-R"]["bar"] == ">= 12" and R["gates"]["G-R"]["value"] >= 12
    for v in ("HELD", "FAILED", "UNRESOLVED", "MALFORMED"):
        assert R["census"]["span_verdicts"][v] > 0, v
    assert sum(R["census"]["document_malformed"].values()) > 0
    text = SPEC.read_text(encoding="utf-8")
    assert "| G-N | cases compared | ≥ 100000 |" in text
    assert "≥ 12 of the closed set" in text


def test_the_exception_gate_is_reported_even_when_it_is_empty():
    """G-E: the two implementations must at least fail on the same inputs. Zero is a reading, not
    an absence, so the keys must be present rather than dropped when nothing raised."""
    g = R["gates"]["G-E"]
    assert set(g) >= {"python_only", "javascript_only", "both", "pairs"}
    assert g["python_only"] == g["javascript_only"] == 0, g


def test_the_result_swears_to_the_receipt_that_is_actually_in_the_tree():
    """The rename guard. Every `path:` receipt the RESULT cites must resolve, or the document is
    swearing to a file nobody can open — which is how the old name would have failed, silently,
    for every reader who was not this repository."""
    import re
    cited = set(re.findall(r'r="path:([^"#]+)', RESULT.read_text(encoding="utf-8")))
    assert cited, "the RESULT cites no path: receipt at all"
    for rel in sorted(cited):
        assert (ROOT / rel).exists(), "%s swears to a receipt that is not in the tree: %s" % (
            RESULT.name, rel)
    assert "conformance/sworn/differential_agreement.json" in cited


def test_the_receipt_does_not_wear_a_name_another_sweep_claims():
    """`*_result*.json` means "a prereg-scored experiment receipt" in this corpus and
    test_protocol_v2v3 sweeps for exactly that. This file has no prereg and nothing scores it."""
    assert "result" not in RECEIPT.name.lower(), RECEIPT.name
    assert "prereg" not in R, "if this ever gains a prereg it belongs under papers/, scored"


def test_the_harness_refuses_to_rewrite_a_tracked_receipt():
    """D6: a run is history. The harness must refuse to overwrite a receipt git already holds,
    rather than improving a published number in place."""
    r = subprocess.run([sys.executable, "conformance/sworn/differential.py",
                        "--cases", "1", "--out", str(RECEIPT)],
                       cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=300)
    assert r.returncode == 2, (r.returncode, r.stdout[-500:], r.stderr[-500:])
    assert "a run is history" in (r.stdout + r.stderr)
