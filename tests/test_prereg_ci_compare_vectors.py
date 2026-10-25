# -*- coding: utf-8 -*-
"""prereg-ci B0 fix: the compare vectors (conformance/prereg_ci/vectors_compare.json).

The B0 review found that ``compare`` returned no finding for a NOT_GATED re-score even when the
receipt recorded a gates digest (DESIGN 3.4 C6(i): a recorded digest no seal holds is
DECISION_DIGEST_UNSEALED), so DESIGN 4.A's A.2, a re-run scored under a bar moved only in the
working copy, passed when the sealed PREREG's fence was one the scanner does not count and
CommonMark renders. The vectors pin the fix and every code ``compare`` returns. Each vector holds
the four legs of a score vector; every leg is replayed here, and ``compare`` runs on every vector.
Nothing here reads history, so nothing skips.
"""
from __future__ import annotations

import builtins
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SET = ROOT / "conformance" / "prereg_ci"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conformance.prereg_ci.replay import (  # noqa: E402
    agree, blob_bytes, build_experiment_repo, content_id, experiment_leg, frozen_leg, normal, parse_receipt,
    sha256)
from styxx.prereg_ci import core  # noqa: E402

DOC = json.loads((SET / "vectors_compare.json").read_text(encoding="utf-8"))
BLOBS = DOC["blobs"]
VECTORS = DOC["vectors"]
BY_NAME = {v["name"]: v for v in VECTORS}
RESCORE_KEYS = {"form", "code", "verdict", "gates", "row", "gates_sha256", "smoke", "smoke_verdict", "total",
                "holes", "unreachable_rows"}
HIDDEN = ["a2/hidden-fence/bom-on-line-1", "a2/hidden-fence/block-quote", "a2/hidden-fence/list-item"]
NO_FENCE = "prereg has no ```gates block"


def _ids(vectors):
    return [v["name"] for v in vectors]


def _inputs(v):
    return blob_bytes(BLOBS[v["inputs"]["prereg"]]), parse_receipt(blob_bytes(BLOBS[v["inputs"]["receipt"]]))


def _compare(v):
    prereg, receipt = _inputs(v)
    i = v["inputs"]
    return core.compare(core.rescore(prereg, receipt), i.get("recorded", {}), sealed=i.get("sealed"),
                        declared=i.get("declared"))


def _codes(v):
    c = v["expect"]["compare"]
    return c if isinstance(c, dict) else [f["code"] for f in c]


# ---- the file itself --------------------------------------------------------------------------------------

def test_the_file_names_its_schema_and_rules():
    assert DOC["schema"] == "styxx.prereg-ci.conformance.compare/v0"
    assert set(DOC["rules"]) >= {"id", "receipt", "frozen", "experiment", "rescore", "compare", "agree"}
    assert len(BY_NAME) == len(VECTORS) == 26


def test_every_blob_hashes_to_its_key_and_every_id_is_its_content_address():
    for key, entry in BLOBS.items():
        assert sha256(blob_bytes(entry)) == key, key
    for v in VECTORS:
        assert v["id"] == content_id(v["name"], v["inputs"]), v["name"]
        assert v["inputs"]["prereg"] in BLOBS and v["inputs"]["receipt"] in BLOBS, v["name"]


def test_the_agree_flag_is_what_the_two_scorer_legs_say():
    for v in VECTORS:
        assert v["agree"] == agree(v["expect"]["frozen"], v["expect"]["experiment"]), v["name"]
    assert sorted(v["name"] for v in VECTORS if not v["agree"]) == [
        "smoke/pointer-not-a-string", "smoke/recorded-smoke-verdict"]


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


@pytest.mark.parametrize("v", VECTORS, ids=_ids(VECTORS))
def test_rescore_leg(v):
    prereg, receipt = _inputs(v)
    want = v["expect"]["rescore"]
    assert set(want) == RESCORE_KEYS
    got = normal(core.rescore(prereg, receipt))
    assert {k: got[k] for k in want} == want


@pytest.mark.parametrize("v", VECTORS, ids=_ids(VECTORS))
def test_compare_leg(v):
    want = v["expect"]["compare"]
    if isinstance(want, dict):
        with pytest.raises(getattr(builtins, want["raises"])) as info:
            _compare(v)
        assert type(info.value).__name__ == want["raises"]
        return
    got = _compare(v)
    assert [{"code": f["code"], "kind": f["kind"]} for f in got] == want
    assert all(isinstance(f["detail"], str) and f["detail"] for f in got)


# ---- what the vectors say --------------------------------------------------------------------------------

def test_the_vectors_reach_every_code_compare_returns():
    """Every code compare returns for a gates decision. The three codes of the locks form (step B4)
    come from a locked scorer's run, which this set (committed at B0, never written again) holds no
    leg for; tests/test_prereg_ci_scorer_run.py reaches each of them through compare by name. The
    three licence codes (step B5) are ``licences``'s, which compare never returns;
    tests/test_prereg_ci_licences.py and conformance/prereg_ci/vectors_licence.json reach each."""
    seen = {f["code"] for v in VECTORS if isinstance(v["expect"]["compare"], list) for f in v["expect"]["compare"]}
    kinds = core._KIND
    locks_codes = {core._SCORER_NOT_RUN, core._SCORER_ERROR, core._SCORER_READS_UNLOCKED}
    licence_codes = {core._LICENCE_WITHDRAWN, core._LICENCE_PENDING, core._LICENCE_PATTERN_FINDS_NOTHING}
    compare_codes = {c for c in kinds if c != core._COVER_UNCHECKABLE} - locks_codes - licence_codes
    assert seen == compare_codes
    for v in VECTORS:
        if isinstance(v["expect"]["compare"], list):
            for f in v["expect"]["compare"]:
                assert f["kind"] == kinds[f["code"]], v["name"]


def test_a_recorded_digest_against_a_not_gated_prereg_is_unsealed():
    """DESIGN 3.4 C6(i): a PREREG with no decision holds no sealed digest, so a recorded one is
    DECISION_DIGEST_UNSEALED, with or without a sealed list beside it; with no recorded digest the
    re-score is its own NOTE and nothing recorded is read."""
    for name in ("not-gated/recorded-digest-and-verdict", "not-gated/recorded-digest-a-sealed-list-names"):
        v = BY_NAME[name]
        assert v["expect"]["rescore"]["code"] == "NOT_GATED"
        assert _codes(v) == ["DECISION_DIGEST_UNSEALED"], name
    for name in ("not-gated/recorded-verdict-only", "not-gated/nothing-recorded"):
        assert _codes(BY_NAME[name]) == ["NOT_GATED"], name


def test_a2_through_a_fence_the_scanner_does_not_count_is_refused():
    """The re-run records the working copy's verdict and digest (Experiment's own reading of the
    edited bar). Behind each of the three hidden fences the scanner and Experiment both refuse the
    sealed PREREG and compare refuses the recorded digest; behind the visible fence compare gives
    both A.2 codes."""
    work = BY_NAME["a2/the-working-copy-bar"]
    record = work["inputs"]["recorded"]
    assert work["expect"]["experiment"] == record and _codes(work) == []
    for name in HIDDEN:
        v = BY_NAME[name]
        e = v["expect"]
        assert v["inputs"]["recorded"] == record, name
        assert e["frozen"]["at"] == "select" and e["frozen"]["message"].startswith(NO_FENCE), name
        assert e["experiment"]["at"] == "init" and e["experiment"]["message"].startswith(NO_FENCE), name
        assert e["rescore"]["code"] == "NOT_GATED", name
        assert _codes(v) == ["DECISION_DIGEST_UNSEALED"], name
        got = _compare(v)
        assert record["gates_sha256"] in got[0]["detail"] and "no gates fence" in got[0]["detail"]
    visible = BY_NAME["a2/visible-fence"]
    assert visible["inputs"]["recorded"] == record
    assert _codes(visible) == ["DECISION_DIGEST_UNSEALED", "VERDICT_DIFFERS"]


def test_the_hidden_fences_render_the_sealed_fence():
    """Each hidden-fence PREREG holds the sealed fence's lines, so a reader sees the 0.5 bar."""
    sealed = blob_bytes(BLOBS[BY_NAME["a2/visible-fence"]["inputs"]["prereg"]]).decode("utf-8")
    body = sealed.split("```gates\n", 1)[1].split("\n```", 1)[0]
    for name in HIDDEN:
        text = blob_bytes(BLOBS[BY_NAME[name]["inputs"]["prereg"]]).decode("utf-8")
        for line in body.split("\n"):
            assert line in text, (name, line)
        assert '"value": 0.5' in text


def test_recorded_gates_are_read_without_a_recorded_verdict():
    assert _codes(BY_NAME["recorded/verdict-absent-gates-agree"]) == ["RECORDED_VERDICT_ABSENT"]
    assert _codes(BY_NAME["recorded/verdict-absent-gates-differ"]) == ["RECORDED_VERDICT_ABSENT", "VERDICT_DIFFERS"]
    assert _codes(BY_NAME["recorded/gates-differ-verdict-agrees"]) == ["VERDICT_DIFFERS"]


def test_sealed_is_an_iterable_of_digests_never_one_string():
    v = BY_NAME["sealed/one-digest-as-a-string"]
    assert isinstance(v["inputs"]["sealed"], str) and _codes(v) == {"raises": "TypeError"}
    prereg, receipt = _inputs(v)
    r = core.rescore(prereg, receipt)
    for bad in (r["gates_sha256"], r["gates_sha256"].encode("ascii"), bytearray(b"ab")):
        with pytest.raises(TypeError):
            core.compare(r, v["inputs"]["recorded"], sealed=bad)
    assert core.compare(r, v["inputs"]["recorded"], sealed=(r["gates_sha256"],)) == []
    assert core.compare(r, v["inputs"]["recorded"], sealed=iter([r["gates_sha256"]])) == []
    assert _codes(BY_NAME["sealed/empty-list"]) == ["DECISION_DIGEST_UNSEALED"]


def test_one_table_reads_one_way_for_a_smoke_receipt_and_a_scored_one():
    a = BY_NAME["smoke/pointer-not-a-string"]["expect"]["rescore"]
    b = BY_NAME["smoke/pointer-not-a-string-scored-receipt"]["expect"]["rescore"]
    assert a["code"] == b["code"] == "SCORER_REFUSES"


def test_the_generator_regenerates_the_file():
    r = subprocess.run([sys.executable, "-B", str(SET / "gen_vectors_compare.py"), "--check"], cwd=str(ROOT),
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert r.stdout.startswith("SAME vectors_compare.json=")
