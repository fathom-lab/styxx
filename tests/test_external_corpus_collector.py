"""The external-corpus collector must not lose a repository without saying so.

The pilot (`oath_ext_recon.py`) had four silent-drop paths — a clone timeout, a two-page search
cap that stopped on any error, unparseable receipts, and oversized READMEs — each of which removed
a repository from the denominator without counting it. A population defined by "what survived the
fetch" is the defect this lane has catalogued nine times.

These tests drive every failure branch and assert it produces a STATUS, not an absence. They are
offline: the network calls are monkeypatched.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "oath_external_corpus",
    ROOT / "papers" / "closed-model-frontier" / "oath_external_corpus.py")
M = importlib.util.module_from_spec(_SPEC)
sys.modules["oath_external_corpus"] = M
_SPEC.loader.exec_module(M)


ENTRY = {"repo": "someone/thing", "query": "q", "query_index": 1, "rank_within_query": 0}


def _entries(*paths):
    return [{"path": p, "type": "blob"} for p in paths]


# --- receipt selection ------------------------------------------------------------------------

def test_receipts_are_taken_in_tree_order_not_name_order():
    """The pilot's cap selected by filename; this one selects by position."""
    got = [e["path"] for e in M.receipt_candidates(
        _entries("z/metrics.json", "a/all_results.json", "m/scores.json", "src/main.py"))]
    assert got == ["z/metrics.json", "a/all_results.json", "m/scores.json"]


def test_receipt_matching_is_on_the_basename_not_the_path():
    got = [e["path"] for e in M.receipt_candidates(
        _entries("deep/nested/dir/results.json", "results.json.bak", "notresults.json"))]
    assert got == ["deep/nested/dir/results.json"]


def test_every_frozen_receipt_name_is_reachable():
    paths = [f"d/{n}" for n in sorted(M.RECEIPT_NAMES)]
    assert len(M.receipt_candidates(_entries(*paths))) == len(M.RECEIPT_NAMES)


# --- obligation vocabulary --------------------------------------------------------------------

def test_obligating_words_are_recorded_lowercased_and_deduped():
    words = M._obligating_words("Accuracy and accuracy again, with F1 accuracy")
    assert words == sorted(set(words))
    assert all(w == w.lower() for w in words)


def test_obligating_words_empty_when_nothing_fires():
    assert M._obligating_words("a sentence with no measurement vocabulary at all") == []


# --- every branch of collect_repo returns a status ----------------------------------------------

@pytest.fixture
def stub(monkeypatch, tmp_path):
    """Drive collect_repo without touching the network."""
    state = {"head": (None, "main", "d" * 40), "tree": (_entries("README.md"), False),
             "files": {}}

    monkeypatch.setattr(M, "head_of", lambda repo: state["head"])
    monkeypatch.setattr(M, "tree_paths", lambda repo, sha: state["tree"])

    def _fetch(repo, sha, path, cap):
        v = state["files"].get(path)
        if v is None:
            return None, "fetch_failed"
        if v == "__OVER_CAP__":
            return None, "over_cap"
        raw = v if isinstance(v, bytes) else v.encode()
        import hashlib
        return raw, hashlib.sha256(raw).hexdigest()

    monkeypatch.setattr(M, "fetch", _fetch)
    return state


def test_head_unavailable_is_a_status(stub, tmp_path):
    stub["head"] = (None, None, None)
    assert M.collect_repo(ENTRY, tmp_path)["status"] == "HEAD_UNAVAILABLE"


def test_tree_unavailable_is_a_status(stub, tmp_path):
    stub["tree"] = (None, False)
    assert M.collect_repo(ENTRY, tmp_path)["status"] == "TREE_UNAVAILABLE"


def test_no_readme_is_a_status(stub, tmp_path):
    stub["tree"] = (_entries("src/main.py", "d/metrics.json"), False)
    assert M.collect_repo(ENTRY, tmp_path)["status"] == "NO_DOC"


def test_oversized_readme_is_a_status_not_a_disappearance(stub, tmp_path):
    stub["tree"] = (_entries("README.md"), False)
    stub["files"]["README.md"] = "__OVER_CAP__"
    assert M.collect_repo(ENTRY, tmp_path)["status"] == "DOC_TOO_LARGE"


def test_unfetchable_readme_is_a_status(stub, tmp_path):
    stub["tree"] = (_entries("README.md"), False)
    assert M.collect_repo(ENTRY, tmp_path)["status"] == "FETCH_FAILED"


def test_no_receipt_is_a_status(stub, tmp_path):
    stub["tree"] = (_entries("README.md"), False)
    stub["files"]["README.md"] = "# hi\n"
    assert M.collect_repo(ENTRY, tmp_path)["status"] == "NO_RECEIPT"


def test_receipts_present_but_all_unparseable_is_its_own_status(stub, tmp_path):
    """Distinct from NO_RECEIPT: the author DID publish results, we could not read them."""
    stub["tree"] = (_entries("README.md", "d/metrics.json"), False)
    stub["files"]["README.md"] = "# hi\n"
    stub["files"]["d/metrics.json"] = "{not json"
    rec = M.collect_repo(ENTRY, tmp_path)
    assert rec["status"] == "RECEIPTS_UNPARSEABLE"
    assert rec["receipts_unparseable"] == 1


def test_a_truncated_tree_is_recorded_rather_than_hidden(stub, tmp_path):
    stub["tree"] = (_entries("README.md", "d/metrics.json"), True)
    stub["files"]["README.md"] = "# acc\nAccuracy was 0.5\n"
    stub["files"]["d/metrics.json"] = json.dumps({"accuracy": 0.5})
    rec = M.collect_repo(ENTRY, tmp_path)
    assert rec["tree_truncated"] is True
    assert "truncated" in rec["status_note"]
    assert rec["status"] == "CERTIFIED", "truncation is a caveat, not an exclusion"


def test_happy_path_records_provenance_for_every_file(stub, tmp_path):
    stub["tree"] = (_entries("README.md", "d/metrics.json"), False)
    stub["files"]["README.md"] = "# r\nAccuracy was 0.5 on the test set\n"
    stub["files"]["d/metrics.json"] = json.dumps({"accuracy": 0.5})
    rec = M.collect_repo(ENTRY, tmp_path)
    assert rec["status"] == "CERTIFIED"
    assert {f["role"] for f in rec["files"]} == {"document", "receipt"}
    for f in rec["files"]:
        assert len(f["sha256"]) == 64 and f["bytes"] > 0 and f["path"]
    assert rec["sha"] == "d" * 40


def test_receipt_cap_is_applied_and_the_tree_total_is_still_reported(stub, tmp_path):
    many = [f"d{i}/metrics.json" for i in range(M.MAX_RECEIPTS_PER_DOC + 5)]
    stub["tree"] = (_entries("README.md", *many), False)
    stub["files"]["README.md"] = "# r\nAccuracy was 0.5\n"
    for p in many:
        stub["files"][p] = json.dumps({"accuracy": 0.5})
    rec = M.collect_repo(ENTRY, tmp_path)
    assert sum(1 for f in rec["files"] if f["role"] == "receipt") == M.MAX_RECEIPTS_PER_DOC
    assert rec["receipts_seen_in_tree"] == len(many), "the cap must not hide the true count"


def test_context_excerpt_is_capped(stub, tmp_path):
    stub["tree"] = (_entries("README.md", "d/metrics.json"), False)
    stub["files"]["README.md"] = "# r\nAccuracy " + ("x" * 900) + " 0.5\n"
    stub["files"]["d/metrics.json"] = json.dumps({"accuracy": 0.5})
    rec = M.collect_repo(ENTRY, tmp_path)
    assert rec["tokens"], "expected at least one token"
    assert all(len(t["context"]) <= M.CONTEXT_CHARS for t in rec["tokens"])


def test_pilot_repos_are_flagged_so_they_can_be_reported_separately(stub, tmp_path):
    stub["tree"] = (_entries("README.md"), False)
    pilot = sorted(M.PILOT_REPOS)[0]
    rec = M.collect_repo({**ENTRY, "repo": pilot}, tmp_path)
    assert rec["is_pilot_repo"] is True
    assert M.collect_repo(ENTRY, tmp_path)["is_pilot_repo"] is False


def test_the_pilot_roster_loaded_and_is_the_expected_size():
    assert len(M.PILOT_REPOS) == 14


# --- amendments forced by the pre-collection red team -------------------------------------------

def test_total_cap_can_never_delete_a_trailing_query():
    """The BLOCKER. 20 x 7 = 140 against a 120 total silently deleted query 7 entirely.

    Queries run in frozen order and `taken` counts only repos new to the global seen set, so
    cross-query dedup never reduces a query's take. Six queries filled 20 each, the total cap
    fired, and `filename:evaluation_results.json` -- the non-HuggingFace eval-harness arm the
    protocol exists to reach -- was never issued a request. Arithmetic, not chance.
    """
    assert M.MAX_REPOS_PER_QUERY * len(M.SEARCH_QUERIES) <= M.MAX_REPOS_TOTAL


def test_the_last_query_is_reachable_under_the_frozen_caps(monkeypatch):
    """Simulated with the network stubbed: every arm must be able to take repositories."""
    page = {"total_count": 9999,
            "items": [{"repository": {"full_name": f"o{i}/r{i}"}} for i in range(100)]}
    calls = {"n": 0}

    def _stub(args, timeout=90):
        calls["n"] += 1
        return {"total_count": 9999,
                "items": [{"repository": {"full_name": f"o{calls['n']}_{i}/r{i}"}}
                          for i in range(100)]}

    monkeypatch.setattr(M, "gh_json", _stub)
    monkeypatch.setattr(M.time, "sleep", lambda *_: None)
    _, accounting = M.search_repos(M.MAX_REPOS_TOTAL)
    last = accounting[-1]
    assert last["query"] == M.SEARCH_QUERIES[-1]
    assert last["repos_taken"] > 0, f"trailing arm deleted: {last}"
    assert page  # keep the literal referenced for readers


def test_readme_match_is_case_insensitive_at_the_root(stub, tmp_path):
    """`path in DOC_NAMES` missed ReadMe.md and friends, inflating NO_DOC."""
    stub["tree"] = (_entries("ReadMe.md", "d/metrics.json"), False)
    stub["files"]["ReadMe.md"] = "# r\nAccuracy was 0.5\n"
    stub["files"]["d/metrics.json"] = json.dumps({"accuracy": 0.5})
    assert M.collect_repo(ENTRY, tmp_path)["status"] == "CERTIFIED"


def test_a_nested_readme_is_still_no_doc_but_is_recorded(stub, tmp_path):
    """Root-only stays root-only; the residual is measured rather than assumed."""
    stub["tree"] = (_entries("docs/README.md", "README.rst"), False)
    rec = M.collect_repo(ENTRY, tmp_path)
    assert rec["status"] == "NO_DOC"
    assert "docs/README.md" in rec["readme_like_paths_seen"]
    assert "README.rst" in rec["readme_like_paths_seen"]


def test_no_doc_with_nothing_readme_like_records_an_empty_list(stub, tmp_path):
    stub["tree"] = (_entries("src/main.py"), False)
    rec = M.collect_repo(ENTRY, tmp_path)
    assert rec["status"] == "NO_DOC"
    assert rec["readme_like_paths_seen"] == []


def test_receipt_rejections_separate_three_different_facts(stub, tmp_path):
    """One counter previously conflated malformed JSON, a network failure and over-cap."""
    stub["tree"] = (_entries("README.md", "a/metrics.json", "b/results.json",
                             "c/scores.json"), False)
    stub["files"]["README.md"] = "# r\nAccuracy was 0.5\n"
    stub["files"]["a/metrics.json"] = "{not json"
    stub["files"]["b/results.json"] = "__OVER_CAP__"
    # c/scores.json is absent from the stub -> fetch_failed
    rec = M.collect_repo(ENTRY, tmp_path)
    assert rec["receipts_rejected"] == {"unparseable": 1, "over_cap": 1, "fetch_failed": 1}
    assert rec["receipts_unparseable"] == 1, "must not absorb the other two"


def test_unfetchable_receipts_get_their_own_status(stub, tmp_path):
    """Not the same claim as 'their JSON was malformed'."""
    stub["tree"] = (_entries("README.md", "a/metrics.json", "b/results.json"), False)
    stub["files"]["README.md"] = "# r\n"
    rec = M.collect_repo(ENTRY, tmp_path)
    assert rec["status"] == "RECEIPTS_UNFETCHABLE"


def test_obligation_surface_is_counted_so_arms_can_be_compared(stub, tmp_path):
    """An arm whose READMEs carry no numeral-plus-trigger line has nothing to abstain on."""
    stub["tree"] = (_entries("README.md", "d/metrics.json"), False)
    stub["files"]["README.md"] = (
        "# title\n"
        "Accuracy was 0.5 on the test set\n"
        "just prose with no numbers\n"
        "an unrelated 42 with no trigger word\n")
    stub["files"]["d/metrics.json"] = json.dumps({"accuracy": 0.5})
    rec = M.collect_repo(ENTRY, tmp_path)
    assert rec["obligation_surface_lines"] == 1
    assert rec["doc_lines"] == 4


# --- the control arm ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def control():
    """Re-certifies 186 documents live; computed once for the module."""
    return M.internal_control()


def test_control_arm_is_recertified_live_not_summed_from_stored_counts(control):
    """The protocol promises "the same verifier, at the same verifier_sha256".

    Summing the counts stored in committed certificates made that promise false: the corpus
    carries TEN distinct verifier_sha256 values and only four certificates were produced by the
    current styxx/certify.py. The control is now re-certified under the pinned verifier, and the
    mixture it replaced is recorded so the reader can see why.
    """
    assert control["documents"] > 100
    assert control["tokens"] == sum(control["status_counts"].values())
    assert control["distinct_recorded_verifier_shas"] > 1, (
        "if this is ever 1 the disclosure can be simplified, but it must be MEASURED")
    assert control["recertification_failures"] == 0
    assert isinstance(control["verdicts_changed_vs_recorded"], list)


def test_control_arm_records_a_reproducible_certificate_roster(control):
    """The frame grows under us, so the arm is pinned by roster rather than assumed stable."""
    assert len(control["certificates"]) == control["documents"]
    assert all(c["certificate"] for c in control["certificates"])
    assert "not independent of the treatment" in control["disclosure_accusation_column"]


def test_control_arm_is_not_degenerate(control):
    """If the internal arm ever reported zero of everything it would tie any external number."""
    assert control["status_counts"]["VERIFIED"] > 0, "a control that verifies nothing cannot contrast"
    assert 0.0 <= control["abstain_share"] <= 1.0
    assert 0.0 <= control["accusation_share"] <= 1.0
