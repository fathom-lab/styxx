# -*- coding: utf-8 -*-
"""prereg-ci B0: conformance vectors for the two gates scorers and for prereg-ci's re-score.

``conformance/prereg_ci/vectors_score.json`` holds one vector per census record of styxx main at
ca347730 (the design's R1: every committed record naming a ``gates_sha256``, re-scored from the
PREREG blob at the commit that added it), synthetic cases, and replays of the #224 pair (R2). A
vector names its PREREG and receipt bytes by sha256 (``conformance/prereg_ci/blobs.json``) and
the outcome of each leg:

* ``frozen``: ``research_capsule.score_frozen`` over the selected gates block, no smoke branch;
* ``experiment``: ``protocol.Experiment.score`` over the PREREG committed to a repository built
  in ``tmp_path`` (it reads a working copy and asks git), smoke read as ``styxx.seal`` reads it;
* ``rescore``: ``styxx.prereg_ci.core.rescore``;
* ``compare``: ``styxx.prereg_ci.core.compare``, for vectors that carry a recorded verdict.

Every leg of every vector is replayed here. A vector on which ``frozen`` and ``experiment`` part
ways is a named divergence (``agree`` false); the list of them is pinned below, so a new one is
never silent. Only the history checks skip, when ca347730 is not in the object store, and they
print why.
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SET = ROOT / "conformance" / "prereg_ci"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conformance.prereg_ci import REV  # noqa: E402
from conformance.prereg_ci.replay import (  # noqa: E402
    agree, blob_bytes, build_experiment_repo, content_id, experiment_leg, frozen_leg, normal,
    parse_receipt, sha256)
from styxx.prereg_ci import core  # noqa: E402

DOC = json.loads((SET / "vectors_score.json").read_text(encoding="utf-8"))
BLOBS = json.loads((SET / "blobs.json").read_text(encoding="utf-8"))["blobs"]
VECTORS = DOC["vectors"]
BY_NAME = {v["name"]: v for v in VECTORS}

# R1 at ca347730: 33 records, 32 distinct (PREREG, receipt) pairs, TALLY {"RESCORES": 33}
CENSUS_RECORDS, CENSUS_PAIRS = 33, 32
SMOKE_RECORD = "census/papers/frequency-resonance/realmodel_bridge_v3_smoke.json"
# R2: the #224 receipt under the blob sealed at 3598e59d, and under the one-bar edit
R2_SEALED = ("BUDGET_CONFOUND__gap_does_not_survive_3x_steps",
             "578924f8e776573cc82db8fbbd8bb9cf84106b81171a5ba6d5783e422c96e854")
R2_EDITED = ("ROBUST__ROTATION_LOAD_BEARING__beyond_diversity",
             "e513a1da1f8c068ea6ba60138392a031bd13edd2b2539a0f4f0fbe4ab9c61c82")

# every vector on which score_frozen and Experiment.score part ways, by name; DESIGN K1 and the
# newline reading. A vector that joins or leaves this list is a finding, never an edit to make
# the test pass.
DIVERGENT = sorted([
    SMOKE_RECORD,
    "k1/bar-nan", "k1/bar-infinite", "k1/bar-string-eq", "k1/bar-bool", "k1/bar-huge-int",
    "k1/row-names-undeclared-gate", "k1/row-names-undeclared-gate-null",
    "k1/malformed-row-after-the-landed-row", "k1/verdict-not-a-string",
    "k1/empty-gates-wildcard-row", "k1/smoke-receipt-with-numbers",
    "k1/smoke-receipt-without-metrics",
    "newline/crlf-blob", "newline/lone-cr-blob",
    "prereg-ci/smoke-both-conventions-true", "prereg-ci/smoke-truthy-string",
    "prereg-ci/smoke-table-invalid", "prereg-ci/smoke-verdict-not-a-string",
])


def _ids(vectors):
    return [v["name"] for v in vectors]


def _inputs(v):
    prereg = blob_bytes(BLOBS[v["inputs"]["prereg"]])
    receipt = parse_receipt(blob_bytes(BLOBS[v["inputs"]["receipt"]]))
    return prereg, receipt


def _history():
    r = subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", REV + "^{commit}"], capture_output=True)
    return r.returncode == 0


def _show(spec: str) -> bytes:
    return subprocess.run(["git", "--no-replace-objects", "-C", str(ROOT), "show", spec],
                          capture_output=True, check=True).stdout


# ---- the set itself ---------------------------------------------------------------------------------------

def test_the_set_names_its_schema_rules_and_revision():
    assert DOC["schema"] == "styxx.prereg-ci.conformance.score/v0"
    assert DOC["rev"] == REV
    assert set(DOC["rules"]) >= {"id", "receipt", "frozen", "experiment", "rescore", "compare"}
    assert len(BY_NAME) == len(VECTORS), "two vectors share a name"


def test_every_blob_hashes_to_its_key():
    for key, entry in BLOBS.items():
        assert sha256(blob_bytes(entry)) == key, key


def test_every_vector_id_is_its_content_address_and_every_input_is_in_the_store():
    for v in VECTORS:
        assert v["id"] == content_id(v["name"], v["inputs"]), v["name"]
        assert v["inputs"]["prereg"] in BLOBS and v["inputs"]["receipt"] in BLOBS, v["name"]


def test_the_agree_flag_is_what_the_two_scorer_legs_say():
    for v in VECTORS:
        assert v["agree"] == agree(v["expect"]["frozen"], v["expect"]["experiment"]), v["name"]


def test_the_divergences_are_named_and_pinned():
    assert sorted(v["name"] for v in VECTORS if not v["agree"]) == DIVERGENT
    for v in VECTORS:
        if not v["agree"]:
            assert v.get("why", "").strip(), f"{v['name']} parts the scorers and does not say why"


# ---- the legs, replayed -----------------------------------------------------------------------------------

@pytest.mark.parametrize("v", VECTORS, ids=_ids(VECTORS))
def test_frozen_leg(v):
    prereg, receipt = _inputs(v)
    assert frozen_leg(prereg, receipt) == v["expect"]["frozen"]


@pytest.fixture(scope="module")
def experiment_paths(tmp_path_factory):
    files = [(v["file"], blob_bytes(BLOBS[v["inputs"]["prereg"]])) for v in VECTORS]
    paths = build_experiment_repo(tmp_path_factory.mktemp("experiment"), files)
    return {v["name"]: p for v, p in zip(VECTORS, paths)}


@pytest.mark.parametrize("v", VECTORS, ids=_ids(VECTORS))
def test_experiment_leg(v, experiment_paths):
    _, receipt = _inputs(v)
    assert experiment_leg(experiment_paths[v["name"]], receipt) == v["expect"]["experiment"]


# a vector pins the keys it holds, so a later key in rescore's output moves no committed vector
RESCORE_KEYS = {"form", "code", "verdict", "gates", "row", "gates_sha256", "smoke", "smoke_verdict", "total",
                "holes", "unreachable_rows"}


@pytest.mark.parametrize("v", VECTORS, ids=_ids(VECTORS))
def test_rescore_leg(v):
    prereg, receipt = _inputs(v)
    want = v["expect"]["rescore"]
    assert set(want) == RESCORE_KEYS
    got = normal(core.rescore(prereg, receipt))
    assert {k: got[k] for k in want} == want


WITH_RECORD = [v for v in VECTORS if "recorded" in v["inputs"] or "declared" in v["inputs"]]


@pytest.mark.parametrize("v", WITH_RECORD, ids=_ids(WITH_RECORD))
def test_compare_leg(v):
    prereg, receipt = _inputs(v)
    got = core.compare(core.rescore(prereg, receipt), v["inputs"].get("recorded", {}),
                       sealed=v["inputs"].get("sealed"), declared=v["inputs"].get("declared"))
    assert [{"code": f["code"], "kind": f["kind"]} for f in got] == v["expect"]["compare"]
    assert all(isinstance(f["detail"], str) and f["detail"] for f in got)


# ---- what the vectors say --------------------------------------------------------------------------------

def test_the_census_re_scores_to_every_recorded_verdict_and_digest():
    """R1: 33 records, 32 pairs; rescore gives the recorded verdict and digest on 33 of 33; the two
    scorers agree on 32, and the 33rd is the smoke record, where only Experiment.score has a smoke
    branch."""
    census = [v for v in VECTORS if v["family"] == "census"]
    assert len(census) == CENSUS_RECORDS
    pairs = {(v["source"]["prereg_path"], v["source"]["receipt_path"]) for v in census}
    assert len(pairs) == CENSUS_PAIRS
    tally = Counter()
    for v in census:
        rec, got = v["inputs"]["recorded"], v["expect"]["rescore"]
        ok = got["code"] is None and got["verdict"] == rec["verdict"] and got["gates_sha256"] == rec["gates_sha256"]
        tally["RESCORES" if ok else "DIFFERS"] += 1
        assert [f["kind"] for f in v["expect"]["compare"] if f["kind"] == "FAIL"] == [], v["name"]
    assert dict(tally) == {"RESCORES": CENSUS_RECORDS}
    assert [v["name"] for v in census if not v["agree"]] == [SMOKE_RECORD]
    smoke = BY_NAME[SMOKE_RECORD]["expect"]
    assert smoke["experiment"]["verdict"] == smoke["rescore"]["verdict"] == "INVALID__smoke_plumbing_only"
    assert smoke["rescore"]["smoke"] is True


def test_r2_the_sealed_blob_and_the_one_bar_edit():
    sealed = BY_NAME["census/papers/frequency-resonance/pmnist_robustness_result.json"]["expect"]
    for leg in ("frozen", "experiment", "rescore"):
        assert (sealed[leg]["verdict"], sealed[leg]["gates_sha256"]) == R2_SEALED, leg
    edited = BY_NAME["r2/working-copy-bar"]["expect"]
    for leg in ("frozen", "experiment", "rescore"):
        assert (edited[leg]["verdict"], edited[leg]["gates_sha256"]) == R2_EDITED, leg


def test_r2_a_re_run_under_the_working_copy_bar_is_refused_twice():
    """DESIGN 4.A, A.2: the receipt records the edited bar's verdict and digest; the sealed bytes
    score it otherwise and hold another digest."""
    v = BY_NAME["r2/a2-re-run-under-the-working-copy-bar"]
    assert v["inputs"]["recorded"]["gates_sha256"] == R2_EDITED[1]
    assert v["inputs"]["sealed"] == [R2_SEALED[1]]
    assert v["expect"]["rescore"]["verdict"] == R2_SEALED[0]
    assert [f["code"] for f in v["expect"]["compare"]] == ["DECISION_DIGEST_UNSEALED", "VERDICT_DIFFERS"]


def test_r2_a_result_that_types_another_rung_is_refused():
    v = BY_NAME["r2/a1-the-result-types-another-rung"]
    assert v["inputs"]["declared"] == R2_EDITED[0]
    assert [f["code"] for f in v["expect"]["compare"]] == ["VERDICT_DIFFERS"]
    prereg, receipt = _inputs(v)
    got = core.compare(core.rescore(prereg, receipt), v["inputs"]["recorded"], sealed=v["inputs"]["sealed"],
                       declared=v["inputs"]["declared"])
    assert got[0]["detail"] == f"the sealed decision scores this receipt {R2_SEALED[0]}; the RESULT prints {R2_EDITED[0]}"


def test_a_crlf_blob_hashes_as_the_experiment_reads_it():
    """PLAN 1.5: prereg-ci reads blobs, Experiment reads a working copy with universal newlines;
    the two digests agree, and decoding without translation finds no fence."""
    lf = BY_NAME["newline/lf-blob"]["expect"]
    for name in ("newline/crlf-blob", "newline/lone-cr-blob"):
        e = BY_NAME[name]["expect"]
        assert e["rescore"]["gates_sha256"] == e["experiment"]["gates_sha256"] == lf["rescore"]["gates_sha256"]
        assert e["frozen"]["raises"] == "GateSpecError" and e["frozen"]["at"] == "select"


def test_rescore_reads_not_gated_only_for_a_prereg_with_no_gates_fence():
    for v in VECTORS:
        if v["expect"]["rescore"]["code"] == "NOT_GATED":
            assert v["expect"]["frozen"]["message"].startswith("prereg has no ```gates block"), v["name"]


def test_rescore_is_score_frozen_wherever_it_reads_neither_smoke_nor_a_pointer_nor_a_newline():
    """C5: the gates form is scored by score_frozen. rescore departs from it in three readings only:
    the smoke branch (3.2(d)), the receipt pointer (3.2(a)) and the newline translation of a blob
    (PLAN 1.5); everywhere else a verdict is score_frozen's and a refusal is a refusal."""
    n = 0
    for v in VECTORS:
        e = v["expect"]
        r = e["rescore"]
        if r["smoke"] is True or r["code"] == "SMOKE_AMBIGUOUS" or v["name"].startswith(
                ("newline/crlf", "newline/lone-cr", "prereg-ci/receipt-pointer")):
            continue
        n += 1
        if "raises" in e["frozen"]:
            assert r["code"] in ("SCORER_REFUSES", "NOT_GATED"), v["name"]
        else:
            assert r["code"] is None, v["name"]
            assert [r[k] for k in ("verdict", "gates", "gates_sha256")] ==                 [e["frozen"][k] for k in ("verdict", "gates", "gates_sha256")], v["name"]
    assert n >= 60, n


# ---- tied to history --------------------------------------------------------------------------------------

def test_the_census_inputs_are_the_blobs_at_their_commits():
    if not _history():
        pytest.skip(f"{REV} is not in the object store (a shallow or partial clone); the census vectors "
                    f"still replay from blobs.json")
    for v in VECTORS:
        if v["family"] != "census":
            continue
        s = v["source"]
        assert sha256(_show(f"{s['prereg_commit']}:{s['prereg_path']}")) == v["inputs"]["prereg"], v["name"]
        assert sha256(_show(f"{s['receipt_rev']}:{s['receipt_path']}")) == v["inputs"]["receipt"], v["name"]


def test_the_generator_regenerates_the_committed_set():
    if not _history():
        pytest.skip(f"{REV} is not in the object store; the generator reads the census from it")
    r = subprocess.run([sys.executable, "-B", str(SET / "gen_vectors.py"), "--check"], cwd=str(ROOT),
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
