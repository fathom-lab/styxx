# -*- coding: utf-8 -*-
"""prereg-ci B0: totality and row reachability by cube splitting, against brute force.

``research_capsule._table_total`` enumerates every assignment and stops at 12 gates (it returns
None above, R8b). prereg-ci's ``core._cover`` splits the assignment cube on the gates the outcome
rows mention, with no gate cap, and counts exactly: the assignments no row matches (holes) and
the assignments that land on each row, under the scorers' own matching rule (the earliest row
whose every ``when`` literal equals the fired value lands).

``conformance/prereg_ci/vectors_cover.json`` holds synthetic tables from 1 to 15 gates, the 52
gated tables of styxx main at ca347730 (R8: b41 the one table that is not total, no row
unreachable, the 15-gate leg-2 table total over 32,768 assignments) and tables with known answers
past brute force. Here the cover and the brute force are both run live on every vector up to 15
gates, and the brute force is itself checked against ``score_frozen`` on the small tables.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SET = ROOT / "conformance" / "prereg_ci"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conformance.prereg_ci import REV  # noqa: E402
from conformance.prereg_ci.replay import brute_force, matches, normal  # noqa: E402
from styxx.prereg_ci import core  # noqa: E402
from styxx.protocol import GateSpecError, _select_gates_block  # noqa: E402
from styxx.research_capsule import _strict_json, _table_total, score_frozen  # noqa: E402

DOC = json.loads((SET / "vectors_cover.json").read_text(encoding="utf-8"))
VECTORS = DOC["vectors"]
BRUTE = [v for v in VECTORS if v["brute"]]
B41 = "papers/disjoint-worlds/PREREG_b41_bridge_2026_08_05.md"
LEG2 = "papers/sworn/PREREG_sworn_measurement_2026_10_23.md"


def _ids(vectors):
    return [v["name"] for v in vectors]


def _history():
    r = subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", REV + "^{commit}"], capture_output=True)
    return r.returncode == 0


def _counts(out):
    return {k: out[k] for k in ("total", "holes", "landings", "unreachable_rows")}


def test_the_set_names_its_schema_and_rule():
    assert DOC["schema"] == "styxx.prereg-ci.conformance.cover/v0"
    assert DOC["rev"] == REV
    assert set(DOC["rules"]) >= {"match", "expect"}
    assert len({v["name"] for v in VECTORS}) == len(VECTORS)


@pytest.mark.parametrize("v", VECTORS, ids=_ids(VECTORS))
def test_cover_equals_the_vector(v):
    got = core._cover(v["gates"], v["rows"])
    assert normal(_counts(got)) == v["expect"]
    if got["total"]:
        assert got["hole"] is None
    else:
        hole = got["hole"]
        assert sorted(hole) == sorted(v["gates"])
        assert not any(matches(hole, when) for when in v["rows"]), "the hole the cover names is matched"


@pytest.mark.parametrize("v", BRUTE, ids=_ids(BRUTE))
def test_brute_force_equals_the_vector(v):
    assert brute_force(v["gates"], v["rows"]) == v["expect"]


def test_the_brute_force_set_reaches_fifteen_gates_from_one():
    sizes = {len(v["gates"]) for v in BRUTE if v["family"] == "synthetic"}
    assert sizes == set(range(1, 16)), sorted(sizes)
    assert max(len(v["gates"]) for v in BRUTE) == 15


def test_the_brute_force_is_the_scorers_own_reading():
    """On every synthetic table up to 6 gates, score_frozen itself, fed one result per assignment,
    lands on the row the brute force counts, and refuses exactly the holes."""
    checked = 0
    for v in BRUTE:
        if v["family"] != "synthetic" or len(v["gates"]) > 6:
            continue
        gates = {g: {"metric": "m." + g, "op": ">=", "value": 0.5} for g in v["gates"]}
        rows = [{"when": w, "verdict": "ROW_%d" % i} for i, w in enumerate(v["rows"])]
        block = json.dumps({"gates": gates, "outcomes": rows, "smoke_verdict": "INVALID__smoke"})
        landings, holes = [0] * len(rows), 0
        for bits in range(1 << len(v["gates"])):
            result = {"m": {g: (1.0 if bits >> i & 1 else 0.0) for i, g in enumerate(v["gates"])}}
            try:
                landings[int(score_frozen(block, result)["verdict"][4:])] += 1
            except GateSpecError as e:
                assert "not total" in str(e)
                holes += 1
        assert (holes, landings) == (v["expect"]["holes"], v["expect"]["landings"]), v["name"]
        checked += 1
    assert checked >= 40, checked


def test_table_total_agrees_up_to_twelve_gates_and_stops_there():
    """R8b: _table_total is the old check; it agrees with the cover where it answers."""
    for v in VECTORS:
        rows = [{"when": w, "verdict": str(i)} for i, w in enumerate(v["rows"])]
        old = _table_total(v["gates"], rows)
        if len(v["gates"]) > 12:
            assert old is None, v["name"]
        else:
            assert old == v["expect"]["total"], v["name"]


def test_the_census_tables_r8():
    """R8 at ca347730: 52 gated tables; the cover agrees with brute force on 52; b41 is the one
    table that is not total; no row is unreachable; the 15-gate leg-2 table is total over 32,768
    assignments, where _table_total returns None."""
    census = [v for v in VECTORS if v["family"] == "census"]
    assert len(census) == 52
    assert all(v["brute"] for v in census)
    assert [v["source"]["path"] for v in census if not v["expect"]["total"]] == [B41]
    assert sum(len(v["expect"]["unreachable_rows"]) for v in census) == 0
    b41 = next(v for v in census if v["source"]["path"] == B41)
    assert (len(b41["gates"]), len(b41["rows"])) == (4, 5)
    leg2 = next(v for v in census if v["source"]["path"] == LEG2)
    assert (len(leg2["gates"]), len(leg2["rows"])) == (15, 70)
    assert leg2["expect"]["total"] is True and sum(leg2["expect"]["landings"]) == 32768
    assert _table_total(leg2["gates"], [{"when": w} for w in leg2["rows"]]) is None


def test_known_answers_past_brute_force():
    known = [v for v in VECTORS if v["family"] == "known"]
    assert known and all(not v["brute"] for v in known)
    assert max(len(v["gates"]) for v in known) >= 64
    for v in known:
        assert sum(v["expect"]["landings"]) + v["expect"]["holes"] == 2 ** len(v["gates"]), v["name"]


def test_the_cover_reads_literals_as_the_scorers_compare_them():
    """1 == True and 0 == False for the scorers; any other non-boolean literal never matches."""
    assert core._cover(["G"], [{"G": 1}, {"G": 0.0}])["landings"] == [1, 1]
    out = core._cover(["G"], [{"G": "true"}, {"G": None}, {"G": 2}, {}])
    assert out["landings"] == [0, 0, 0, 2] and out["unreachable_rows"] == [0, 1, 2]


def test_the_cover_refuses_what_score_frozen_refuses():
    with pytest.raises(GateSpecError):
        core._cover(["G0"], [{"G1": True}])
    with pytest.raises(GateSpecError):
        core._cover(["G0"], ["not a row"])


def test_rescore_reads_totality_with_no_gate_cap():
    """score_frozen's own ``total`` is None past 12 gates; rescore's comes from the cover."""
    names = ["G%02d" % i for i in range(15)]
    gates = {g: {"metric": g, "op": ">=", "value": 0.5} for g in names}
    rows = [{"when": {names[i]: True}, "verdict": "R%d" % i} for i in range(15)] + [{"when": {}, "verdict": "REST"}]
    block = json.dumps({"gates": gates, "outcomes": rows, "smoke_verdict": "INVALID__smoke"})
    receipt = {g: 0.0 for g in names}
    assert score_frozen(block, receipt)["total"] is None
    got = core.rescore("# plan\n\n```gates\n" + block + "\n```\n", receipt)
    assert (got["verdict"], got["total"], got["holes"], got["row"]) == ("REST", True, 0, 15)


def test_the_census_tables_are_read_from_history():
    if not _history():
        pytest.skip(f"{REV} is not in the object store (a shallow or partial clone); the census tables "
                    f"still replay from vectors_cover.json")
    out = subprocess.run(["git", "--no-replace-objects", "-C", str(ROOT), "ls-tree", "-r", "--name-only", REV,
                          "papers"], capture_output=True, check=True).stdout.decode("utf-8").split("\n")
    seen = {}
    for path in out:
        if not (path.rsplit("/", 1)[-1].startswith("PREREG_") and path.endswith(".md")):
            continue
        blob = subprocess.run(["git", "--no-replace-objects", "-C", str(ROOT), "show", f"{REV}:{path}"],
                              capture_output=True, check=True).stdout
        try:
            block = _select_gates_block(blob.decode("utf-8"))
            spec = _strict_json(block)
        except (GateSpecError, ValueError):
            continue
        seen[path] = (hashlib.sha256(block.encode("utf-8")).hexdigest(), list(spec["gates"]),
                      [r["when"] for r in spec["outcomes"]])
    census = {v["source"]["path"]: v for v in VECTORS if v["family"] == "census"}
    assert sorted(seen) == sorted(census)
    for path, (digest, gates, rows) in seen.items():
        v = census[path]
        assert (digest, gates, rows) == (v["source"]["gates_sha256"], v["gates"], v["rows"]), path
