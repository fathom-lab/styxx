"""Blinding and tie-direction are the only things making the adjudication arm worth anything.

If a packet leaks the verifier's decision, the panel measures its own agreeableness. If ties
resolve toward the instrument, the ground-truth column flatters it. Both are pinned here.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "oath_adjudication", ROOT / "papers" / "closed-model-frontier" / "oath_adjudication.py")
A = importlib.util.module_from_spec(_SPEC)
sys.modules["oath_adjudication"] = A
_SPEC.loader.exec_module(A)


def _row(status, i, repo="o/r"):
    return {"repo": repo, "sha": "s" * 40, "line": i, "col": 1, "token": str(i),
            "value": float(i), "status": status, "receipt_ref": None,
            "obligating_words": ["accuracy"], "context": f"Accuracy was {i} on the test set"}


@pytest.fixture
def built(tmp_path, monkeypatch):
    rows = ([_row("UNGROUNDED", i) for i in range(1, 18)]
            + [_row("ABSTAIN", 100 + i) for i in range(400)]
            + [_row("VERIFIED", 900 + i) for i in range(200)])
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    monkeypatch.setattr(A, "LEDGER", ledger)
    monkeypatch.setattr(A, "PACKETS", tmp_path / "packets.json")
    monkeypatch.setattr(A, "KEY", tmp_path / "key.json")
    monkeypatch.setattr(A, "RESULT", tmp_path / "result.json")
    return A.build()


# --- blinding -----------------------------------------------------------------------------------

def test_no_packet_item_carries_the_verifier_decision(built):
    """Membership must be the only thing an adjudicator sees, and it must carry nothing."""
    leak = {"status", "receipt_ref", "verdict", "arm", "is_accused"}
    for packet in built["packets"]:
        for item in packet:
            assert not (leak & set(item)), f"packet item leaks: {sorted(leak & set(item))}"


def test_every_arm_is_represented_and_shuffled_together(built):
    """If accused tokens clustered at the front, position would leak status."""
    key = json.loads(Path(A.KEY).read_text(encoding="utf-8"))
    flat = [it for p in built["packets"] for it in p]
    statuses = [key[it["id"]]["status"] for it in flat]
    assert set(statuses) == {"UNGROUNDED", "ABSTAIN", "VERIFIED"}
    first_third = statuses[: len(statuses) // 3]
    assert "UNGROUNDED" in first_third and "ABSTAIN" in first_third


def test_the_key_is_written_separately_and_holds_the_answers(built):
    key = json.loads(Path(A.KEY).read_text(encoding="utf-8"))
    assert all("status" in v for v in key.values())
    assert len(key) == built["composition"]["total_items"]


# --- the frozen sampling rule --------------------------------------------------------------------

def test_every_accusation_is_included_never_sampled(built):
    assert built["composition"]["accused"] == 17
    assert built["composition"]["accused"] == built["available"]["accused"]


def test_decoy_counts_follow_the_frozen_rule(built):
    assert built["composition"]["abstain_decoys"] == A.N_ABSTAIN_DECOYS
    assert built["composition"]["verified_decoys"] == A.N_VERIFIED_DECOYS


def test_decoys_degrade_gracefully_when_the_corpus_is_small(tmp_path, monkeypatch):
    rows = [_row("UNGROUNDED", 1), _row("ABSTAIN", 2), _row("VERIFIED", 3)]
    ledger = tmp_path / "l.jsonl"
    ledger.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    monkeypatch.setattr(A, "LEDGER", ledger)
    monkeypatch.setattr(A, "PACKETS", tmp_path / "p.json")
    monkeypatch.setattr(A, "KEY", tmp_path / "k.json")
    c = A.build()["composition"]
    assert (c["accused"], c["abstain_decoys"], c["verified_decoys"]) == (1, 1, 1)


def test_build_is_deterministic_under_the_frozen_seed(built, tmp_path, monkeypatch):
    again = A.build()
    assert [it["id"] for p in again["packets"] for it in p] == \
           [it["id"] for p in built["packets"] for it in p]


# --- tie direction ------------------------------------------------------------------------------

def test_majority_decides():
    assert A._verdict(["CLAIM", "CLAIM", "NOT_A_CLAIM"]) == "CLAIM"
    assert A._verdict(["NOT_A_CLAIM", "NOT_A_CLAIM", "CLAIM"]) == "NOT_A_CLAIM"


def test_no_majority_resolves_against_the_instrument():
    """The whole point: an unresolvable token must not be scored as a real catch."""
    assert A._verdict(["CLAIM", "NOT_A_CLAIM", "UNSURE"]) == "NOT_A_CLAIM"
    assert A._verdict(["UNSURE", "UNSURE", "UNSURE"]) == "NOT_A_CLAIM"
    assert A._verdict(["CLAIM", "UNSURE", "UNSURE"]) == "NOT_A_CLAIM"


def test_unsure_never_rescues_an_accusation():
    """Two UNSURE plus one CLAIM is not a claim; the instrument does not get the benefit."""
    assert A._verdict(["CLAIM", "UNSURE", "NOT_A_CLAIM"]) == "NOT_A_CLAIM"


# --- scoring ------------------------------------------------------------------------------------

def test_scoring_separates_the_three_arms(built, tmp_path):
    key = json.loads(Path(A.KEY).read_text(encoding="utf-8"))
    judged = []
    for iid, k in key.items():
        # accused -> panel says NOT a claim; abstained -> panel says CLAIM; verified -> CLAIM
        v = {"UNGROUNDED": "NOT_A_CLAIM", "ABSTAIN": "CLAIM", "VERIFIED": "CLAIM"}[k["status"]]
        judged += [{"id": iid, "verdict": v} for _ in range(3)]
    jp = tmp_path / "j.json"
    jp.write_text(json.dumps(judged), encoding="utf-8")
    out = A.score(jp)
    assert out["false_accusation_rate"]["rate"] == 1.0
    assert out["miss_rate"]["rate"] == 1.0
    assert out["verified_arm_sanity"]["rate"] == 1.0
    assert out["agreement"]["split"] == 0


def test_scoring_counts_split_panels(built, tmp_path):
    key = json.loads(Path(A.KEY).read_text(encoding="utf-8"))
    iid = next(iter(key))
    jp = tmp_path / "j.json"
    jp.write_text(json.dumps([{"id": iid, "verdict": v}
                              for v in ("CLAIM", "NOT_A_CLAIM", "UNSURE")]), encoding="utf-8")
    out = A.score(jp)
    assert out["agreement"]["split"] == 1 and out["agreement"]["unanimous"] == 0


def test_scoring_ignores_ids_not_in_the_key(built, tmp_path):
    jp = tmp_path / "j.json"
    jp.write_text(json.dumps([{"id": "NOT_A_REAL_ID", "verdict": "CLAIM"}]), encoding="utf-8")
    assert A.score(jp)["items_scored"] == 0


def test_the_question_forces_unsure_to_cost_the_instrument():
    assert "counted against the instrument" in A.QUESTION
