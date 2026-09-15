"""The pass-4 CORRECTION builder, driven through every rule the correction protocol marks new, and every repair the
adversarial review of its first version confirmed (synthetic inputs)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILDER = os.path.join(REPO, "papers", "plates", "build_sand_survey_pass4_correction.py")
P2 = os.path.join(REPO, "papers", "plates", "sand_prior_art_survey_pass2_2026_09_13.json")
P3 = os.path.join(REPO, "papers", "plates", "sand_prior_art_survey_pass3_2026_09_14.json")
E = ["E1", "E2", "E3", "E4", "E5", "E6"]
ALL5 = ("E1", "E2", "E3", "E4", "E5")


def text(sid):
    return "\n".join([f"line {k} of source {sid}" for k in range(1, 30)] + [f"the end of {sid}"]) + "\n"


def elements(sid, true, bad_quote=()):
    return {e: {"value": e in true, "quote": (f"words not in {sid}" if e in bad_quote else f"line {3 + i} of source {sid}") if e in true else ""}
            for i, e in enumerate(E)}


def reading(sid, *, clause="C4a", verdict="OCCUPIES", true=(), bad_quote=(), phrase=None, leads=()):
    v = {"clause": clause, "verdict": verdict, "object": "o", "reason": "r", "neighbour_phrase": phrase or f"{sid} do a thing"}
    if clause == "C4a":
        v["elements"] = elements(sid, true, bad_quote)
    return {"id": sid, "read_end_to_end": True, "last_line_quote": f"the end of {sid}", "midpoint_quote": f"line 15 of source {sid}",
            "verdicts": [v], "bearing_quotes": [], "leads": list(leads), "notes": ""}


def confirm(sid, name, *, clause="C4a", verdict="RETIRES", reason="r", bad_quote=()):
    return {"confirmer": name, "id": sid, "clause": clause, "verdict": verdict, "read_end_to_end": True, "last_line_quote": f"the end of {sid}",
            "midpoint_quote": f"line 15 of source {sid}", "object": "o", "elements": elements(sid, ALL5 + ("E6",), bad_quote), "reason": reason}


def build(tmp_path, *, p4_sources=(), p4_readings=(), p4_confirmations=(), p4_unfetchable=(), p4_scope=None, below_cap=(), titles=None, urls=None,
          c_sources=(), c_readings=(), c_confirmations=(), c_fetch=None, extra_files=None, expect_fail=False):
    titles, urls = titles or {}, urls or {}
    root = tmp_path / "repo"
    plates = root / "papers" / "plates"
    plates.mkdir(parents=True)
    shutil.copy(P2, plates / os.path.basename(P2))
    shutil.copy(P3, plates / os.path.basename(P3))
    p4, pc, t4, tc = root / "p4", root / "pc", root / "t4", root / "tc"
    for d in (p4, pc, t4, tc):
        d.mkdir()
    lst, fetch = [], {}
    for sid, might in p4_sources:
        lst.append({"id": sid, "part": "B", "title": titles.get(sid, f"the title of {sid}"), "who": f"who {sid}", "year": 2025,
                    "urls": urls.get(sid, []), "might_occupy": list(might)})
        if sid in p4_unfetchable:
            fetch[sid] = {"status": "UNFETCHABLE", "reason": "no candidate"}
        else:
            (t4 / f"{sid}.txt").write_text(text(sid), encoding="utf-8")
            fetch[sid] = {"status": "FETCHED", "url": f"https://example.org/{sid}", "sha256": "0" * 64, "fetched_at": "2099-01-01T00:00:00Z",
                          "attempts": [{"url": f"https://example.org/{sid}"}], "title_words_found": 1.0}
    (p4 / "list.json").write_text(json.dumps(lst), encoding="utf-8")
    (p4 / "fetch_record.json").write_text(json.dumps(fetch), encoding="utf-8")
    (p4 / "search_record.json").write_text(json.dumps({"list_b": [], "below_cap": [{"title": t} for t in below_cap]}), encoding="utf-8")
    (p4 / "readings.json").write_text(json.dumps({"readings": [{"reader": "reader-1", "sources": list(p4_readings)}],
                                                   "confirmations": list(p4_confirmations)}), encoding="utf-8")
    if p4_scope:
        (p4 / "fetched_text_scope.json").write_text(json.dumps(p4_scope), encoding="utf-8")
    clst, cfetch = [], dict(c_fetch or {})
    for sid, pass_ in c_sources:
        clst.append({"id": sid, "pass": pass_, "title": titles.get(sid, f"the title of {sid}"), "who": f"who {sid}", "year": 2020, "might_occupy": ["C4a"]})
        cfetch.setdefault(sid, {"status": "FETCHED", "sha256": "1" * 64})
    for sid, e in cfetch.items():
        if e.get("status") == "FETCHED":
            (tc / f"{sid}.txt").write_text(text(sid), encoding="utf-8")
    (pc / "list_correction.json").write_text(json.dumps(clst), encoding="utf-8")
    (pc / "fetch_record_correction.json").write_text(json.dumps(cfetch), encoding="utf-8")
    (pc / "readings_correction.json").write_text(json.dumps({"readings": [{"reader": "reader-1", "sources": list(c_readings)}],
                                                             "confirmations": list(c_confirmations)}), encoding="utf-8")
    for name, content in (extra_files or {}).items():
        (pc / name).write_text(json.dumps(content), encoding="utf-8")
    r = subprocess.run([sys.executable, BUILDER, str(p4), str(pc), "2099_01_01", "deadbeef", str(t4), str(tc)], cwd=root,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if expect_fail:
        return r
    assert r.returncode == 0, r.stderr
    return json.loads((plates / "sand_prior_art_survey_pass4_correction_2099_01_01.json").read_text(encoding="utf-8"))


D1 = ("E1", "E2", "E3", "E4")  # a distance-one coding (without E5), to keep a composable sentence


def test_an_element_whose_quote_is_not_found_is_not_carried(tmp_path):
    out = build(tmp_path, p4_sources=[("PA", ["C4a"])], p4_readings=[reading("PA", true=D1, bad_quote=("E4",))])
    v = out["sources"]["PA"]["verdicts"]["C4a"]
    assert v["elements"]["E4"]["quote_found"] is False and v["elements"]["E4"]["carried"] is False
    assert v["missing"] == ["E4", "E5"] and v["distance"] == 2
    assert out["counts"]["element_quotes_not_found"] == ["PA:E4"]


def test_a_c4a_retires_not_carrying_all_five_goes_to_the_confirmers_and_without_them_is_disputed(tmp_path):
    out = build(tmp_path, p4_sources=[("PA", ["C4a"])], p4_readings=[reading("PA", verdict="RETIRES", true=ALL5, bad_quote=("E5",))])
    v = out["sources"]["PA"]["verdicts"]["C4a"]
    assert v["verdict"] == "DISPUTED" and v["distance"] == 1 and v["confirmations"] == []
    assert out["clauses"]["C4a"]["status"] == "UNPRICED" and out["sentence"]["status"] == "UNLICENSED"


def test_a_c4a_retires_not_carrying_all_five_is_retired_by_two_carrying_confirmers(tmp_path):
    out = build(tmp_path, p4_sources=[("PA", ["C4a"])], p4_readings=[reading("PA", verdict="RETIRES", true=D1)],
                p4_confirmations=[confirm("PA", "confirmer-1"), confirm("PA", "confirmer-2", reason="r2")])
    assert out["sources"]["PA"]["verdicts"]["C4a"]["verdict"] == "RETIRES"
    assert out["clauses"]["C4a"]["status"] == "RETIRED" and out["sentence"]["status"] == "RETIRED"


def test_one_return_under_two_confirmer_names_is_one_confirmation(tmp_path):
    out = build(tmp_path, p4_sources=[("PA", ["C4a"])], p4_readings=[reading("PA", verdict="RETIRES", true=ALL5)],
                p4_confirmations=[confirm("PA", "confirmer-1"), confirm("PA", "confirmer-2")])
    assert out["sources"]["PA"]["verdicts"]["C4a"]["verdict"] == "DISPUTED"
    assert out["sentence"]["status"] == "UNLICENSED"


def test_two_confirmations_under_one_confirmers_name_are_disputed(tmp_path):
    out = build(tmp_path, p4_sources=[("PA", ["C4a"])], p4_readings=[reading("PA", verdict="RETIRES", true=ALL5)],
                p4_confirmations=[confirm("PA", "confirmer-1"), confirm("PA", "confirmer-1", reason="r2")])
    assert out["sources"]["PA"]["verdicts"]["C4a"]["verdict"] == "DISPUTED"
    assert out["clauses"]["C4a"]["status"] == "UNPRICED" and out["sentence"]["status"] == "UNLICENSED"


def test_a_confirmation_copied_from_the_first_reading_is_not_a_confirmer(tmp_path):
    first = reading("PA", verdict="RETIRES", true=ALL5 + ("E6",))
    v = first["verdicts"][0]
    copy = {"confirmer": "confirmer-1", "id": "PA", "clause": "C4a", "verdict": "RETIRES", "read_end_to_end": True,
            "last_line_quote": first["last_line_quote"], "midpoint_quote": first["midpoint_quote"], "object": v["object"],
            "elements": v["elements"], "reason": v["reason"]}
    out = build(tmp_path, p4_sources=[("PA", ["C4a"])], p4_readings=[first], p4_confirmations=[copy, confirm("PA", "confirmer-2", reason="r2")])
    entry = out["sources"]["PA"]["verdicts"]["C4a"]
    assert entry["verdict"] == "DISPUTED" and entry["distinct_confirmers_carrying"] == ["pass4/readings:confirmer-2"]
    assert any(row["copies_the_first_reading"] for row in entry["confirmations"])


def test_the_same_confirmation_in_two_files_is_refused(tmp_path):
    c = confirm("PA", "confirmer-1")
    r = build(tmp_path, p4_sources=[("PA", ["C4a"])], p4_readings=[reading("PA", verdict="RETIRES", true=ALL5)], p4_confirmations=[c],
              extra_files={"readings_correction_dup.json": {"readings": [], "confirmations": [c]}}, expect_fail=True)
    assert r.returncode == 2 and "appears twice" in r.stderr


def test_two_distinct_confirmers_retire_the_clause_and_the_sentence(tmp_path):
    out = build(tmp_path, p4_sources=[("PA", ["C4a"])], p4_readings=[reading("PA", verdict="RETIRES", true=ALL5)],
                p4_confirmations=[confirm("PA", "confirmer-1"), confirm("PA", "confirmer-2", reason="r2")])
    assert out["sources"]["PA"]["verdicts"]["C4a"]["verdict"] == "RETIRES"
    assert out["clauses"]["C4a"]["status"] == "RETIRED" and out["sentence"]["status"] == "RETIRED"


def test_a_confirmer_whose_quote_is_not_found_does_not_confirm(tmp_path):
    out = build(tmp_path, p4_sources=[("PA", ["C4a"])], p4_readings=[reading("PA", verdict="RETIRES", true=ALL5)],
                p4_confirmations=[confirm("PA", "confirmer-1"), confirm("PA", "confirmer-2", bad_quote=("E3",))])
    v = out["sources"]["PA"]["verdicts"]["C4a"]
    assert v["verdict"] == "DISPUTED" and v["distinct_confirmers_carrying"] == ["pass4/readings:confirmer-1"]


def test_a_skimmed_retires_is_occupies_from_abstract_and_leaves_the_clause_occupied(tmp_path):
    out = build(tmp_path, p4_sources=[("PA", ["C5"]), ("PB", ["C4a"])], p4_scope={"PA": {"scope": "a landing page", "reason": "menus only"}},
                p4_readings=[reading("PA", clause="C5", verdict="RETIRES"), reading("PB", true=D1)])
    v = out["sources"]["PA"]["verdicts"]["C5"]
    assert out["sources"]["PA"]["status"] == "SKIMMED"
    assert v["verdict"] == "OCCUPIES" and v["from_abstract"] is True and v["reader_verdict"] == "RETIRES"
    assert out["clauses"]["C5"]["status"] == "OCCUPIED" and out["clauses"]["C5"]["correction"]["occupied_from_abstract_by"] == ["PA"]
    assert out["sentence"]["status"] == "SURVIVES"


def test_a_skimmed_silent_with_all_five_coded_stays_silent_from_abstract(tmp_path):
    out = build(tmp_path, p4_sources=[("PA", ["C4a"]), ("PB", ["C4a"])], p4_scope={"PA": {"scope": "a landing page", "reason": "menus"}},
                p4_readings=[reading("PA", verdict="SILENT", true=ALL5), reading("PB", true=D1)])
    v = out["sources"]["PA"]["verdicts"]["C4a"]
    assert v["verdict"] == "SILENT" and v["from_abstract"] is True
    assert out["clauses"]["C4a"]["correction"]["silent"] == ["PA"] and out["clauses"]["C4a"]["status"] == "OCCUPIED"


def test_a_c4a_retirement_on_a_source_listed_only_for_c2_retires_c4a(tmp_path):
    out = build(tmp_path, p4_sources=[("PA", ["C2"])], p4_readings=[reading("PA", verdict="RETIRES", true=ALL5)],
                p4_confirmations=[confirm("PA", "confirmer-1"), confirm("PA", "confirmer-2", reason="r2")])
    assert out["clauses"]["C4a"]["status"] == "RETIRED" and out["sentence"]["status"] == "RETIRED"


def test_a_verdict_word_outside_the_vocabulary_is_refused(tmp_path):
    r = build(tmp_path, p4_sources=[("PA", ["C5"])], p4_readings=[reading("PA", clause="C5", verdict="RETIRED")], expect_fail=True)
    assert r.returncode == 2 and "verdict word" in r.stderr


def test_pass3_flags_never_enter_the_nearness_pool(tmp_path):
    out = build(tmp_path, p4_sources=[("PA", ["C4a"])], p4_readings=[reading("PA", true=("E1", "E2", "E3"))])
    near = out["clauses"]["C4a"]["nearness"]
    assert near["pool"] == ["PA"] and all(not ids for ids in near["at_distance_one_by_element"].values())
    assert near["named_in_sentence"] == ["PA"] and near["rule_used"].startswith("no source at distance one")
    assert "Xu et al." not in out["sentence"]["text"]


def test_every_source_tied_at_distance_one_is_named(tmp_path):
    out = build(tmp_path, p4_sources=[("PA", ["C4a"]), ("PB", ["C4a"])],
                p4_readings=[reading("PA", true=("E1", "E2", "E3", "E5"), phrase="PA authors grade"), reading("PB", true=("E1", "E2", "E3", "E5"), phrase="PB authors grade")])
    near = out["clauses"]["C4a"]["nearness"]
    assert near["at_distance_one_by_element"]["E4"] == ["PA", "PB"] and near["named_in_sentence"] == ["PA", "PB"]
    assert "PA authors grade" in out["sentence"]["text"] and "PB authors grade" in out["sentence"]["text"]


def test_more_than_five_names_unlicense_the_sentence(tmp_path):
    ids = ["PA", "PB", "PC", "PD", "PF", "PG"]
    out = build(tmp_path, p4_sources=[(i, ["C4a"]) for i in ids], p4_readings=[reading(i, true=("E1", "E2", "E3", "E5")) for i in ids])
    assert out["clauses"]["C4a"]["nearness"]["refused_more_than_five"] is True
    assert out["sentence"]["status"] == "UNLICENSED"


def test_the_conjunction_is_recorded(tmp_path):
    out = build(tmp_path, p4_sources=[("PA", ["C4a"])], p4_readings=[reading("PA", true=D1)])
    conj = out["conjunction"]
    assert conj["status"] == "RETIRED" and conj["one_source_does_all"] == []
    assert "C3" in conj["deleted_clauses"] and "C4b" in conj["deleted_clauses"] and "C4a" in conj["surviving_clauses"]


def test_leads_naming_a_listed_source_are_dropped_and_duplicates_merged(tmp_path):
    leads = ["Someone. The title of PB. 2025.", "Foo bar baz qux, arXiv 2501.00001", "Another citation arXiv:2501.00001v2", "A unique lead about something else"]
    out = build(tmp_path, p4_sources=[("PA", ["C4a"]), ("PB", ["C4a"])], below_cap=["A unique lead about something else"],
                p4_readings=[reading("PA", true=D1, leads=leads), reading("PB", true=("E1",))])
    c = out["counts"]
    assert c["leads_raw"] == 4 and c["leads_naming_a_listed_source"] == 1 and c["leads_kept"] == 3 and c["leads_distinct"] == 2


def test_leads_match_listed_sources_through_ampersands_display_suffixes_and_arxiv_ids(tmp_path):
    titles = {"PA": "Cats & Dogs Audit Systems – a standing programme for spotting things", "PB": "Some Other Paper"}
    leads = ["Someone. Cats and Dogs Audit Systems. 2025.", "A lead citing arXiv 2501.00002", "Smith et al. 2024. Token efficient change detection in LLM APIs. arXiv",
             "Token efficient change detection in LLM APIs (2026)"]
    out = build(tmp_path, p4_sources=[("PA", ["C4a"]), ("PB", ["C4a"])], titles=titles, urls={"PB": ["https://arxiv.org/abs/2501.00002"]},
                p4_readings=[reading("PA", true=D1, leads=leads), reading("PB", true=("E1",))])
    dropped = {x["names_listed_source"] for x in out["leads"]["naming_a_listed_source"]}
    assert dropped == {"PA", "PB"}
    assert out["counts"]["leads_kept"] == 2 and out["counts"]["leads_distinct"] == 1


def test_an_unfetchable_bounty_candidate_is_listed_as_able_to_retire_the_sentence(tmp_path):
    out = build(tmp_path, p4_sources=[("PA", ["C4a"]), ("PZ", ["C5"])], p4_unfetchable=("PZ",), p4_readings=[reading("PA", true=D1)])
    assert out["clauses"]["C5"]["status"] == "UNPRICED" and out["sentence"]["status"] == "SURVIVES_WITHOUT_C5"
    assert out["sentence"]["unchecked_candidates_that_could_retire_the_sentence"] == ["PZ"]
    assert out["sentence"]["c4a_unchecked_candidates"] == []


def test_an_unscored_candidate_is_recorded_and_does_not_unprice_its_clause(tmp_path):
    out = build(tmp_path, p4_sources=[("PA", ["C4a"]), ("PB", ["C4a"])], p4_readings=[reading("PA", true=D1)])
    assert out["sources"]["PB"]["status"] == "UNREAD"
    assert out["clauses"]["C4a"]["correction"]["unscored"] == ["PB"] and out["clauses"]["C4a"]["status"] == "OCCUPIED"
    assert out["sentence"]["status"] == "SURVIVES" and out["sentence"]["unchecked_candidates_that_could_retire_the_sentence"] == []


def test_a_not_recoded_earlier_source_is_recorded_and_not_priced(tmp_path):
    out = build(tmp_path, p4_sources=[("PA", ["C4a"])], p4_readings=[reading("PA", true=D1)],
                c_sources=[("SA", 2)], c_fetch={"SA": {"status": "NOT_RECODED", "reason": "bytes do not match"}})
    assert out["sources"]["SA"]["status"] == "NOT_RECODED" and out["counts"]["recoded_status"]["NOT_RECODED"] == 1
    assert out["clauses"]["C4a"]["status"] == "OCCUPIED" and out["sentence"]["status"] == "SURVIVES"


def test_only_a_trailing_without_clause_is_removed_and_only_five_entities_unescaped(tmp_path):
    out = build(tmp_path, p4_sources=[("PA", ["C4a"]), ("PB", ["C4a"]), ("PC", ["C4a"])],
                p4_readings=[reading("PA", true=("E1", "E2", "E3", "E5"), phrase="Doe et al. audit served models without weights, without a floor"),
                             reading("PB", true=D1, phrase="Smith &notes drift &amp; more"),
                             reading("PC", true=("E1", "E2", "E4", "E5"), phrase="Roe et al. audit APIs, without a floor, an interval or log-probabilities")])
    text = out["sentence"]["text"]
    assert "where Doe et al. audit served models without weights, without an interval on the comparison" in text
    assert "where Smith &notes drift & more, without log-probabilities" in text
    assert "where Roe et al. audit APIs, without a floor measured on the same weights" in text


def test_a_located_copy_read_under_the_correction_supersedes_the_landing_page(tmp_path):
    out = build(tmp_path, p4_sources=[("PA", ["C4a"])], p4_scope={"PA": {"scope": "a landing page", "reason": "menus"}},
                p4_readings=[reading("PA", true=("E1",))],
                c_fetch={"PA": {"status": "FETCHED", "sha256": "2" * 64, "located_for_scope": True}},
                c_readings=[reading("PA", true=D1)])
    pa = out["sources"]["PA"]
    assert pa["status"] == "READ" and pa["reading_origin"] == "correction" and len(pa["superseded_readings"]) == 1
    assert pa["fetch_record"].startswith("correction/") and pa["sha256"] == "2" * 64 and pa["earlier_attempts"][0]["fetch_record"].startswith("pass4/")
    assert pa["verdicts"]["C4a"]["distance"] == 1 and out["counts"]["read_from_a_located_copy"] == ["PA"]


def test_a_located_copy_fetched_but_not_read_keeps_the_pass4_fetch_fields(tmp_path):
    out = build(tmp_path, p4_sources=[("PA", ["C4a"]), ("PB", ["C4a"])], p4_scope={"PA": {"scope": "a landing page", "reason": "menus"}},
                p4_readings=[reading("PA", true=("E1",)), reading("PB", true=D1)],
                c_fetch={"PA": {"status": "FETCHED", "sha256": "2" * 64, "located_for_scope": True}})
    pa = out["sources"]["PA"]
    assert pa["status"] == "SKIMMED" and pa["sha256"] == "0" * 64 and pa["fetch_record"].startswith("pass4/")
    assert pa["located_attempt"]["sha256"] == "2" * 64 and pa["located_attempt"]["status"] == "FETCHED"


def test_a_failed_located_fetch_leaves_the_landing_page_skimmed(tmp_path):
    out = build(tmp_path, p4_sources=[("PA", ["C4a"]), ("PB", ["C4a"])], p4_scope={"PA": {"scope": "a landing page", "reason": "menus"}},
                p4_readings=[reading("PA", true=("E1",)), reading("PB", true=D1)],
                c_fetch={"PA": {"status": "UNFETCHABLE", "reason": "403", "located_for_scope": True}})
    pa = out["sources"]["PA"]
    assert pa["status"] == "SKIMMED" and pa["located_attempt"]["status"] == "UNFETCHABLE" and pa["fetch_record"].startswith("pass4/")


def test_a_correction_reading_of_pass4_bytes_is_checked_against_the_pass4_text(tmp_path):
    out = build(tmp_path, p4_sources=[("PA", ["C4a"])], c_readings=[reading("PA", true=D1)])
    pa = out["sources"]["PA"]
    assert pa["reading_origin"] == "correction" and pa["status"] == "READ" and pa["proof_of_reading"]["midpoint_ok"] is True
    assert out["counts"]["element_quotes_not_found"] == []


def test_fetch_time_attempts_and_title_check_are_carried_into_the_record(tmp_path):
    out = build(tmp_path, p4_sources=[("PA", ["C4a"])], p4_readings=[reading("PA", true=D1)])
    pa = out["sources"]["PA"]
    assert pa["fetched_at"] == "2099-01-01T00:00:00Z" and pa["attempts"] and pa["title_words_found"] == 1.0


def test_an_earlier_pass_source_recoded_under_the_correction_enters_the_pool(tmp_path):
    out = build(tmp_path, p4_sources=[("PA", ["C4a"])], p4_readings=[reading("PA", true=("E1", "E2"))],
                c_sources=[("SA", 3)], c_readings=[reading("SA", true=("E1", "E2", "E4", "E5"), phrase="SA authors grade compression")])
    near = out["clauses"]["C4a"]["nearness"]
    assert near["at_distance_one_by_element"]["E3"] == ["SA"] and near["named_in_sentence"] == ["SA"]
    assert out["sources"]["SA"]["listed_in"] == "pass 3" and "SA authors grade compression" in out["sentence"]["text"]
    assert out["counts"]["recoded_status"]["READ"] == 1
