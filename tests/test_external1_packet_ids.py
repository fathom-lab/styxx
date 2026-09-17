"""EXTERNAL-1's packet ids (#125): the repaired build numbers items after the shuffle, and
`build --as-published` still reproduces the numbering that was published, leak included.

The ledger and the shelf are gitignored, so every build here runs on a synthetic ledger and a
synthetic `f` table in tmp_path. No test writes next to the committed packet, key or digest.
"""
import importlib.util
import json
import random
import sqlite3
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CMF = ROOT / "papers" / "closed-model-frontier"
_SPEC = importlib.util.spec_from_file_location("external1_packet_under_test",
                                               CMF / "external1_packet.py")
E = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(E)

ACC, VER, SYN = "gate_says_contradicted", "decoy_verified", "decoy_synthetic_contradiction"


def _write_ledger(path, verdicts):
    """One claim per verdict, four claims to a PR record, every claim naming a path."""
    with path.open("w", encoding="utf-8") as fh:
        for start in range(0, len(verdicts), 4):
            pr_id = 1000 + start // 4
            claims = []
            for j, verdict in enumerate(verdicts[start:start + 4]):
                p = f"src/m{pr_id}/f{j}.py"
                claims.append({"kind": "file_modified", "text": f"updated `{p}`",
                               "detail": {"path": p}, "verdict": verdict})
            fh.write(json.dumps({"pr_id": pr_id, "agent": f"agent-{pr_id % 3}",
                                 "html_url": f"https://example.invalid/pr/{pr_id}",
                                 "claims": claims}) + "\n")
    return [1000 + s // 4 for s in range(0, len(verdicts), 4)]


def _write_shelf(path, pr_ids):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE f (pr_id INTEGER, filename TEXT, status TEXT, patch TEXT)")
    rows = []
    for pr in pr_ids:
        rows += [(pr, f"src/m{pr}/f0.py", "MODIFIED", "@@"),
                 (pr, f"src/m{pr}/f0.py", "modified", "@@"),      # duplicate filename
                 (pr, f"src/m{pr}/new.py", None, None),           # no status -> modified
                 (pr, None, "added", None)]                       # no filename -> dropped
    con.executemany("INSERT INTO f VALUES (?, ?, ?, ?)", rows)
    con.commit()
    con.close()


def _point_outputs(monkeypatch, d):
    d.mkdir(exist_ok=True)
    monkeypatch.setattr(E, "PACKET", d / "packet.json")
    monkeypatch.setattr(E, "KEY", d / "key_SEALED.json")
    monkeypatch.setattr(E, "DIGEST", d / "key_digest.txt")
    return d


def _read(d):
    return (json.loads((d / "packet.json").read_text(encoding="utf-8")),
            json.loads((d / "key_SEALED.json").read_text(encoding="utf-8")))


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """130 accusations, 45 verified, 9 uncheckable (which the builder must ignore), mixed across PRs."""
    verdicts = ["CONTRADICTED"] * 130 + ["VERIFIED"] * 45 + ["UNCHECKABLE"] * 9
    random.Random(7).shuffle(verdicts)
    pr_ids = _write_ledger(tmp_path / "ledger.jsonl", verdicts)
    _write_shelf(tmp_path / "shelf.sqlite", pr_ids)
    monkeypatch.setattr(E, "LEDGER", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(E, "DB", tmp_path / "shelf.sqlite")
    monkeypatch.setattr(E, "ANSWERS", tmp_path / "answers.json")
    monkeypatch.setattr(E, "RESULT", tmp_path / "adjudication.json")
    _point_outputs(monkeypatch, tmp_path / "default")
    return tmp_path


@pytest.fixture
def built(corpus, monkeypatch):
    out = {}
    for tag, as_published in (("repaired", False), ("published", True)):
        d = _point_outputs(monkeypatch, corpus / tag)
        assert E.build(as_published=as_published) == 0
        out[tag] = _read(d)
    return out


def _arms_in_id_order(key):
    return [key[iid]["truth"] for iid in sorted(key)]


def _perturbed_ids(packet):
    return sorted(it["id"] for it in packet["items"]
                  if "zz_" in it["claim_text"] + json.dumps(it["claim_detail"]))


def _without_id(item):
    return {k: v for k, v in item.items() if k != "id"}


# --- the repair -----------------------------------------------------------------------------------

def test_default_ids_do_not_map_contiguous_ranges_to_arms(built):
    packet, key = built["repaired"]
    arms = _arms_in_id_order(key)
    assert Counter(arms) == {ACC: E.N_ACC, VER: E.N_VER, SYN: E.N_SYN}
    assert E.arm_runs(arms) > len(set(arms)) + 2
    for arm in (ACC, VER, SYN):
        nums = sorted(int(iid[3:]) for iid, v in key.items() if v["truth"] == arm)
        assert nums[-1] - nums[0] + 1 != len(nums), f"{arm} occupies one contiguous id range"
    last = [f"E1-{n:03d}" for n in range(len(key) - E.N_SYN, len(key))]
    assert _perturbed_ids(packet) != last            # the tell the issue read off the packet


def test_default_ids_follow_packet_position(built):
    packet, key = built["repaired"]
    assert [it["id"] for it in packet["items"]] == [f"E1-{n:03d}" for n in range(len(key))]


def test_the_clustering_guard_fires_when_the_shuffle_does_not_take(corpus, monkeypatch):
    """Mutation check of the guard: with a no-op shuffle the default build must refuse, and
    write nothing."""
    class NoShuffle(random.Random):
        def shuffle(self, x):
            return None

    monkeypatch.setattr(E, "random", type("R", (), {"Random": NoShuffle}))
    with pytest.raises(AssertionError, match="cluster by arm"):
        E.build()
    assert not E.PACKET.exists() and not E.KEY.exists() and not E.DIGEST.exists()


# --- the record ------------------------------------------------------------------------------------

def test_as_published_ids_carry_the_arm_pinned_as_history(built):
    packet, key = built["published"]
    n_acc, n_ver = E.N_ACC, E.N_VER
    for iid, v in key.items():
        n = int(iid[3:])
        want = ACC if n < n_acc else VER if n < n_acc + n_ver else SYN
        assert v["truth"] == want, iid
    assert E.arm_runs(_arms_in_id_order(key)) == 3
    assert _perturbed_ids(packet) == [f"E1-{n:03d}" for n in range(n_acc + n_ver, len(key))]


def test_both_modes_draw_the_same_items_and_the_same_key_apart_from_ids(built):
    (p_new, k_new), (p_pub, k_pub) = built["repaired"], built["published"]
    assert {k: v for k, v in p_new.items() if k != "items"} == \
           {k: v for k, v in p_pub.items() if k != "items"}
    # the same shuffle: position by position the same item, only the id differs
    assert [_without_id(i) for i in p_new["items"]] == [_without_id(i) for i in p_pub["items"]]
    canon = lambda o: json.dumps(o, sort_keys=True)  # noqa: E731
    assert Counter(canon(_without_id(i)) for i in p_new["items"]) == \
           Counter(canon(_without_id(i)) for i in p_pub["items"])
    assert Counter(canon(v) for v in k_new.values()) == Counter(canon(v) for v in k_pub.values())
    # and each key entry still describes the item that carries its id
    pairs = lambda p, k: Counter(canon([k[i["id"]], _without_id(i)]) for i in p["items"])  # noqa: E731
    assert pairs(p_new, k_new) == pairs(p_pub, k_pub)
    assert set(k_new) == set(k_pub)
    assert k_new != k_pub                            # the numbering is what changed


def test_the_published_id_order_follows_from_the_population_counts(tmp_path, monkeypatch):
    """The committed packet's id sequence, all 130 positions, is what `--as-published` numbers on
    a ledger with the pre-correction population counts. Only the counts matter: sample and
    shuffle draw on sizes, never on contents."""
    counts = json.loads((CMF / "external1_summary_PREFIX.json").read_text(
        encoding="utf-8"))["claims_by_verdict"]
    verdicts = ["CONTRADICTED"] * counts["CONTRADICTED"] + ["VERIFIED"] * counts["VERIFIED"]
    _write_ledger(tmp_path / "ledger.jsonl", verdicts)
    _write_shelf(tmp_path / "shelf.sqlite", [])
    monkeypatch.setattr(E, "LEDGER", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(E, "DB", tmp_path / "shelf.sqlite")
    committed = json.loads((CMF / "external1_packet.json").read_text(encoding="utf-8"))
    committed_ids = [it["id"] for it in committed["items"]]
    assert _perturbed_ids(committed) == [f"E1-{n:03d}" for n in range(115, 130)]   # the leak

    d = _point_outputs(monkeypatch, tmp_path / "published")
    assert E.build(as_published=True) == 0
    assert [it["id"] for it in _read(d)[0]["items"]] == committed_ids

    d = _point_outputs(monkeypatch, tmp_path / "repaired")
    assert E.build() == 0
    assert [it["id"] for it in _read(d)[0]["items"]] == [f"E1-{n:03d}" for n in range(130)]


# --- the receipts stay receipts --------------------------------------------------------------------

def test_build_refuses_to_rewrite_an_existing_packet_under_other_ids(corpus, monkeypatch, capsys):
    d = _point_outputs(monkeypatch, corpus / "record")
    assert E.build(as_published=True) == 0
    before = {p.name: p.read_bytes() for p in d.iterdir()}
    assert E.build() == 1
    assert "REFUSED" in capsys.readouterr().out
    assert {p.name: p.read_bytes() for p in d.iterdir()} == before
    assert E.build(as_published=True) == 0           # regenerating the same record is allowed
    assert {p.name: p.read_bytes() for p in d.iterdir()} == before


def test_cli_accepts_as_published_and_refuses_anything_else(corpus, monkeypatch):
    d = _point_outputs(monkeypatch, corpus / "cli")
    assert E.main(["build", "bogus"]) == 2
    assert E.main(["score", "--as-published"]) == 2
    assert E.main(["bogus"]) == 2
    assert not any(d.iterdir())
    assert E.main(["build", "--as-published"]) == 0
    assert E.arm_runs(_arms_in_id_order(_read(d)[1])) == 3
