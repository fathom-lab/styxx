"""The pass-4 survey builder, driven through every rule it applies, on synthetic inputs (no network, no readers)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILDER = os.path.join(REPO, "papers", "plates", "build_sand_survey_pass4.py")
P3 = os.path.join(REPO, "papers", "plates", "sand_prior_art_survey_pass3_2026_09_14.json")
E = ["E1", "E2", "E3", "E4", "E5", "E6"]


def text(sid):
    return "\n".join([f"line {k} of source {sid}" for k in range(1, 30)] + [f"the end of {sid}"]) + "\n"


def reading(sid, *, clause="C4a", verdict="OCCUPIES", true=(), last=None, phrase=None):
    els = {e: {"value": e in true, "quote": f"line {3 + i} of source {sid}" if e in true else ""} for i, e in enumerate(E)}
    v = {"clause": clause, "verdict": verdict, "object": "o", "reason": "r", "neighbour_phrase": phrase or f"{sid} do a thing"}
    if clause == "C4a":
        v["elements"] = els
    return {"id": sid, "read_end_to_end": True, "last_line_quote": last if last is not None else f"the end of {sid}",
            "midpoint_quote": f"line 15 of source {sid}", "verdicts": [v], "bearing_quotes": [], "leads": [], "notes": ""}


def confirm(sid, k, *, clause="C4a", all_five=True, verdict="RETIRES"):
    els = {e: {"value": all_five or e != "E3", "quote": ""} for e in E}
    return {"confirmer": f"confirmer-{k}", "id": sid, "clause": clause, "verdict": verdict, "read_end_to_end": True,
            "last_line_quote": f"the end of {sid}", "midpoint_quote": f"line 15 of source {sid}", "object": "o", "elements": els, "reason": "r"}


def build(tmp_path, sources, readings, confirmations, unfetchable=(), bodies=None, retry=None, expect_fail=False):
    root = tmp_path / "repo"
    (root / "papers" / "plates").mkdir(parents=True)
    shutil.copy(P3, root / "papers" / "plates" / os.path.basename(P3))
    inputs, texts = root / "inputs", root / "texts"
    inputs.mkdir()
    texts.mkdir()
    lst, fetch = [], {}
    for sid, might in sources:
        lst.append({"id": sid, "part": "B", "title": f"title {sid}", "who": f"who {sid}", "year": 2025, "urls": [], "might_occupy": might})
        if sid in unfetchable:
            fetch[sid] = {"status": "UNFETCHABLE", "reason": "no candidate"}
        else:
            (texts / f"{sid}.txt").write_text((bodies or {}).get(sid) or text(sid), encoding="utf-8")
            fetch[sid] = {"status": "FETCHED", "url": f"https://example.org/{sid}", "sha256": "0" * 64}
    (inputs / "list.json").write_text(json.dumps(lst), encoding="utf-8")
    (inputs / "fetch_record.json").write_text(json.dumps(fetch), encoding="utf-8")
    if retry:
        for sid, entry in retry.items():
            if entry.get("status") == "FETCHED":
                (texts / f"{sid}.txt").write_text((bodies or {}).get(sid) or text(sid), encoding="utf-8")
        (inputs / "fetch_record_retry.json").write_text(json.dumps(retry), encoding="utf-8")
    (inputs / "search_record.json").write_text(json.dumps({"searches": [], "list_b": [], "below_cap": [], "n_distinct": 0}), encoding="utf-8")
    (inputs / "readings.json").write_text(json.dumps({"readings": [{"reader": "reader-1", "sources": readings}], "confirmations": confirmations}), encoding="utf-8")
    r = subprocess.run([sys.executable, BUILDER, str(inputs), "2099_01_01", "deadbeef", str(texts)], cwd=root,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if expect_fail:
        return r
    assert r.returncode == 0, r.stderr
    return json.loads((root / "papers" / "plates" / "sand_prior_art_survey_pass4_2099_01_01.json").read_text(encoding="utf-8"))


ALL5 = ("E1", "E2", "E3", "E4", "E5")


def test_confirmed_retirement_retires_the_clause_and_the_sentence(tmp_path):
    out = build(tmp_path, [("X1", ["C4a"])], [reading("X1", verdict="RETIRES", true=ALL5)], [confirm("X1", 1), confirm("X1", 2)])
    v = out["sources"]["X1"]["verdicts"]["C4a"]
    assert v["verdict"] == "RETIRES" and v["retirement"].startswith("CONFIRMED")
    assert out["clauses"]["C4a"]["status"] == "RETIRED"
    assert out["sentence"]["status"] == "RETIRED"
    assert out["counts"]["element_quotes_found"] == out["counts"]["element_quotes"] == 5


def test_one_dissenting_confirmer_disputes_and_unlicenses(tmp_path):
    out = build(tmp_path, [("X1", ["C4a"])], [reading("X1", verdict="RETIRES", true=ALL5)], [confirm("X1", 1), confirm("X1", 2, all_five=False)])
    assert out["sources"]["X1"]["verdicts"]["C4a"]["verdict"] == "DISPUTED"
    assert out["clauses"]["C4a"]["status"] == "UNPRICED"
    assert out["sentence"]["status"] == "UNLICENSED"


def test_all_five_coded_without_confirmers_is_disputed_not_retired(tmp_path):
    out = build(tmp_path, [("X1", ["C4a"])], [reading("X1", verdict="OCCUPIES", true=ALL5)], [])
    assert out["sources"]["X1"]["verdicts"]["C4a"]["verdict"] == "DISPUTED"
    assert out["sentence"]["status"] == "UNLICENSED"


def test_a_reader_who_cannot_quote_the_end_is_skimmed_and_cannot_retire(tmp_path):
    out = build(tmp_path, [("X1", ["C4a"])], [reading("X1", verdict="RETIRES", true=ALL5, last="line 20 of source X1")],
                [confirm("X1", 1), confirm("X1", 2)])
    assert out["sources"]["X1"]["status"] == "SKIMMED"
    assert out["sources"]["X1"]["verdicts"]["C4a"]["verdict"] == "UNCHECKABLE"
    assert out["clauses"]["C4a"]["status"] == "UNPRICED"


def test_a_right_last_line_without_a_middle_line_is_skimmed(tmp_path):
    r = reading("X1", true=("E1",))
    r["midpoint_quote"] = "line 2 of source X1"
    out = build(tmp_path, [("X1", ["C4a"])], [r], [])
    assert out["sources"]["X1"]["status"] == "SKIMMED"
    assert out["sources"]["X1"]["proof_of_reading"] == {"last_line_ok": True, "midpoint_ok": False}


def test_the_middle_third_is_by_line_number_as_readers_see_it(tmp_path):
    body = "\n".join([f"long line {k} " + "x" * 2000 for k in range(1, 8)] + [f"line {k} of source X1" for k in range(8, 30)] + ["the end of X1"]) + "\n"
    out = build(tmp_path, [("X1", ["C4a"])], [reading("X1")], [], bodies={"X1": body})
    assert out["sources"]["X1"]["proof_of_reading"] == {"last_line_ok": True, "midpoint_ok": True}
    assert out["sources"]["X1"]["status"] == "READ"


def test_escaped_ampersands_in_phrases_are_unescaped_in_the_sentence(tmp_path):
    rs = [reading("X1", true=("E1", "E2", "E3", "E4"), phrase="Cao, Jia &amp; Gong grade a thing")]
    out = build(tmp_path, [("X1", ["C4a"])], rs, [])
    assert "Cao, Jia & Gong grade a thing" in out["sentence"]["text"] and "&amp;" not in out["sentence"]["text"]


def test_retires_without_all_five_is_recorded_as_occupies(tmp_path):
    out = build(tmp_path, [("X1", ["C4a"])], [reading("X1", verdict="RETIRES", true=("E1", "E2"))], [])
    v = out["sources"]["X1"]["verdicts"]["C4a"]
    assert v["verdict"] == "OCCUPIES" and "verdict_note" in v and v["distance"] == 3


def test_paraphrased_element_quote_is_not_found(tmp_path):
    r = reading("X1", true=("E1",))
    r["verdicts"][0]["elements"]["E1"]["quote"] = "a paraphrase the source never printed"
    out = build(tmp_path, [("X1", ["C4a"])], [r], [])
    assert out["sources"]["X1"]["verdicts"]["C4a"]["elements"]["E1"]["quote_found"] is False
    assert out["counts"]["element_quotes_found"] == 0


def test_unfetchable_candidate_unprices_c2_and_the_sentence_survives_without_it(tmp_path):
    out = build(tmp_path, [("X1", ["C4a"]), ("X2", ["C2"])], [reading("X1", true=("E1", "E2", "E3", "E4"))], [], unfetchable=("X2",))
    assert out["clauses"]["C2"]["status"] == "UNPRICED"
    assert out["clauses"]["C4a"]["status"] == "OCCUPIED"
    assert out["sentence"]["status"] == "SURVIVES_WITHOUT_C2"
    assert "chained log" not in out["sentence"]["text"]


def test_nearness_is_counted_per_element_and_pass4_wins_ties(tmp_path):
    rs = [reading("X1", true=("E1", "E2", "E3", "E4"), phrase="X1 authors grade a thing"),
          reading("X2", true=("E1", "E2", "E4", "E5"), phrase="X2 authors grade another thing"),
          reading("X3", true=("E1",), phrase="X3 authors do little")]
    out = build(tmp_path, [("X1", ["C4a"]), ("X2", ["C4a"]), ("X3", ["C4a"])], rs, [])
    near = out["clauses"]["C4a"]["nearness"]
    assert near["by_element"]["E5"]["id"] == "X1"
    assert near["by_element"]["E3"]["id"] == "X2"
    assert "X3" not in near["named_in_sentence"]
    assert out["sentence"]["status"] == "SURVIVES"
    assert "without log-probabilities" in out["sentence"]["text"]
    assert "without a floor measured on the same weights" in out["sentence"]["text"]


def test_c5_retirement_needs_two_blind_confirmers(tmp_path):
    ok = build(tmp_path / "a", [("X1", ["C4a", "C5"])], [reading("X1", clause="C5", verdict="RETIRES")],
               [confirm("X1", 1, clause="C5"), confirm("X1", 2, clause="C5")])
    assert ok["clauses"]["C5"]["status"] == "RETIRED" and ok["sentence"]["status"] == "RETIRED"
    no = build(tmp_path / "b", [("X1", ["C4a", "C5"])], [reading("X1", clause="C5", verdict="RETIRES")],
               [confirm("X1", 1, clause="C5"), confirm("X1", 2, clause="C5", verdict="OCCUPIES")])
    assert no["sources"]["X1"]["verdicts"]["C5"]["verdict"] == "DISPUTED"
    assert no["clauses"]["C5"]["status"] == "UNPRICED"


def test_a_later_fetch_record_locates_an_unfetchable_source_and_keeps_the_failed_attempts(tmp_path):
    retry = {"X2": {"status": "FETCHED", "url": "https://doi.org/10.0/x2", "sha256": "1" * 64, "attempts": [{"url": "https://doi.org/10.0/x2"}]}}
    rs = [reading("X1", true=("E1", "E2", "E3", "E4")), reading("X2", true=("E1",))]
    out = build(tmp_path, [("X1", ["C4a"]), ("X2", ["C4a"])], rs, [], unfetchable=("X2",), retry=retry)
    x2 = out["sources"]["X2"]
    assert x2["status"] == "READ" and x2["fetch_record"] == "fetch_record_retry.json"
    assert x2["earlier_attempts"][0]["fetch_record"] == "fetch_record.json"
    assert out["clauses"]["C4a"]["status"] == "OCCUPIED"


def test_a_source_fetched_in_two_records_is_refused(tmp_path):
    retry = {"X1": {"status": "FETCHED", "url": "https://example.org/again", "sha256": "1" * 64}}
    r = build(tmp_path, [("X1", ["C4a"])], [reading("X1")], [], retry=retry, expect_fail=True)
    assert r.returncode != 0 and "fetched in two records" in r.stderr


def test_an_unfetchable_c4a_candidate_unprices_the_fingerprint_clause(tmp_path):
    out = build(tmp_path, [("X1", ["C4a"]), ("X2", ["C4a"])], [reading("X1", true=("E1", "E2", "E3", "E4"))], [], unfetchable=("X2",))
    assert out["sources"]["X2"]["status"] == "UNFETCHABLE"
    assert out["clauses"]["C4a"]["status"] == "UNPRICED"
    assert out["sentence"]["status"] == "UNLICENSED"


@pytest.mark.parametrize("missing", ["C4a"])
def test_a_clause_with_no_verdict_is_unscored_and_unprices(tmp_path, missing):
    r = reading("X1", clause="C5", verdict="SILENT")
    out = build(tmp_path, [("X1", ["C5", missing])], [r], [])
    assert out["sources"]["X1"]["verdicts"][missing]["verdict"] == "UNSCORED"
    assert out["clauses"][missing]["status"] == "UNPRICED"
