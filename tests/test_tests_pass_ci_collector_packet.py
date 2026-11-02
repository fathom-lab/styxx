# -*- coding: utf-8 -*-
"""The packet builder of tests-pass-ci (P3, P7, P8.5, P9), offline, on invented stores and items only.

The beacon refused unless it is block 975000 (or the late rule's) read from both explorers, the stand-in hash refused
for real; the draw re-derived from the frame as the collector writes it; the cap and the audits by v, the size cut in
P7's order, the rows never cut; the shuffle and the ids; the key sealed by its salt; the packets P2 (a dry run on a
store opened before 2026-11-02), P1 (testreceipt's sheets) and Q; the official build's refusals (seal shut, lock row
unfilled, beacon not named). Nothing here reads a pull request or a network.
"""
from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import json
import sys
import types

import pytest

from tests import _tests_pass_ci_fakes  # noqa: F401  (puts the collector on sys.path)
from tests._tests_pass_ci_fakes import COLLECTOR, ROOT, FakeGitHub, OpenGuard

# the invented decoy fixture is read from the repository's own collector, never from a mutated copy
sys.path.insert(0, str(ROOT / "papers" / "tests-pass-ci" / "collect" / "checks"))

import synthetic_decoys  # noqa: E402
import tpc_beacon  # noqa: E402
import tpc_collect  # noqa: E402
import tpc_decoys  # noqa: E402
import tpc_packet  # noqa: E402
from tpc_draw import membership  # noqa: E402
from tpc_store import Store, dumps  # noqa: E402

from tests.test_tests_pass_ci_collector_screen_rule_reports import GENESIS, GENESIS_ID  # noqa: E402


# ---- the beacon --------------------------------------------------------------------------------------------
def _named(**kw):
    b = {"kind": "bitcoin", "height": 975000, "beacon_hex": GENESIS_ID, "header": GENESIS,
         "header_time": "2009-01-03T18:15:05Z", "rule": "section 3.1, named height",
         "explorers": [{"explorer": e, "block_id": GENESIS_ID, "header": GENESIS, "tip_height": 975010}
                       for e in tpc_packet.EXPLORER_NAMES]}
    b.update(kw)
    return b


def test_a_named_block_read_from_both_explorers_passes():
    assert tpc_packet.beacon_problems(_named()) == []
    assert tpc_packet.require_named_beacon(_named()) == GENESIS_ID


@pytest.mark.parametrize("change,want", [
    ({"kind": "stand_in", "beacon_hex": tpc_packet.STAND_IN_HEX}, "stand-in"),
    ({"kind": "dev"}, "not a Bitcoin block"),
    ({"height": 974999}, "named height"),
    ({"height": 975001}, "late rule"),
    ({"explorers": [{"explorer": tpc_packet.EXPLORER_NAMES[0], "block_id": GENESIS_ID, "header": GENESIS,
                     "tip_height": 975010}]}, "both explorers"),
])
def test_beacons_that_may_not_draw(change, want):
    probs = tpc_packet.beacon_problems(_named(**change))
    assert any(want in p for p in probs), probs
    with pytest.raises(tpc_packet.PacketError):
        tpc_packet.require_named_beacon(_named(**change))


def test_the_explorers_must_agree_and_hold_six_confirmations():
    b = _named()
    b["explorers"][1]["block_id"] = "0" * 64
    assert any("answers" in p for p in tpc_packet.beacon_problems(b))
    b = _named()
    b["explorers"][0]["tip_height"] = 975004
    assert any("confirmations" in p for p in tpc_packet.beacon_problems(b))


def test_the_window_store_draws_only_under_the_named_block(tmp_path):
    day = "2026-11-02"
    for beacon, ok in ((dict(_named(), height=0), False), (tpc_packet.stand_in_beacon(), False), (_named(), True)):
        root = tmp_path / str(len(list(tmp_path.iterdir())))
        st = Store(root)
        st.write(f"frames/frame_{day}.jsonl", b"".join(dumps(r) for r in _frame(day, 30, 10, 30, 0)))
        st.write("draw/beacon.json", json.dumps(beacon).encode())
        ctx = types.SimpleNamespace(store=st, dev=False, stage="", code={"combined": "x"}, say=lambda m: None,
                                    args=types.SimpleNamespace(dev_beacon=None))
        if ok:
            tpc_collect.stage_draw(ctx)
            assert st.exists(f"draw/draw_{day}.jsonl")
        else:
            with pytest.raises(SystemExit, match="beacon refused"):
                tpc_collect.stage_draw(ctx)
            assert not st.exists(f"draw/draw_{day}.jsonl")


def test_the_beacon_command_reads_the_named_height_only(tmp_path, monkeypatch, capsys):
    called = []
    monkeypatch.setattr(tpc_beacon, "read_beacon", lambda h, rule: called.append(h) or dict(_named(), height=h))
    root = tmp_path / "w"
    assert tpc_collect.main(["beacon", "--root", str(root), "--height", "975001", "--rule", "x", "--go"]) == 2
    assert called == []  # refused before any explorer is asked
    assert tpc_collect.main(["beacon", "--root", str(root), "--height", "975001", "--rule", "late rule", "--go",
                             "--pr-opened-at", "2026-11-07T20:00:00Z"]) == 2  # header time not six hours after
    assert called == [975001] and not (root / "draw" / "beacon.json").exists()
    assert tpc_collect.main(["beacon", "--root", str(root), "--height", "975000", "--rule", "named", "--go"]) == 0
    assert json.loads((root / "draw" / "beacon.json").read_bytes())["height"] == 975000
    assert "refused" in capsys.readouterr().out


def test_the_late_rule_needs_the_pull_requests_time_and_six_hours():
    ok = _named(height=975003, rule="section 3.1, late rule", pr_opened_at="2009-01-03T00:00:00Z")
    assert tpc_packet.beacon_problems(ok) == []
    late = _named(height=975003, rule="section 3.1, late rule", pr_opened_at="2009-01-03T15:00:00Z")
    assert any("six hours" in p for p in tpc_packet.beacon_problems(late))
    assert tpc_packet.stand_in_beacon()["beacon_hex"] == tpc_packet.STAND_IN_HEX
    assert tpc_packet.STAND_IN_HEX == hashlib.sha256(b"tests-pass-ci/stand-in: the hash of a dry run, never a beacon").hexdigest()


# ---- the draw, re-derived ----------------------------------------------------------------------------------
def _frame(day, n_cursor=300, n_devin=100, n_cc=300, small=20):
    rows = []
    t = f"{day}T12:00:00Z"
    for agent, n, base in (("Cursor", n_cursor, 10_000), ("Devin", n_devin, 20_000), ("Claude_Code", n_cc, 30_000)):
        for i in range(n):
            pid = base + i
            rows.append({"pr_id": pid, "repo": f"org{i % 40}/repo{i % 40}", "number": i + 1, "stars": 500,
                         "fork": False, "author": "someone", "author_type": "User", "created_at": t, "agent": agent,
                         "slice": None})
    for i in range(small):  # under 100 stars or forks: never kept
        rows.append({"pr_id": 90_000 + i, "repo": "tiny/repo", "number": i, "stars": 5, "fork": False,
                     "author": "x", "author_type": "User", "created_at": t, "agent": "Cursor", "slice": None})
    return rows


def test_the_draw_re_derived_is_the_collectors(tmp_path):
    day = "2026-10-20"
    st = Store(tmp_path)
    frame = _frame(day)
    st.write(f"frames/frame_{day}.jsonl", b"".join(dumps(r) for r in frame))
    ctx = types.SimpleNamespace(store=st, dev=True, stage="", code={"combined": "x"}, say=lambda m: None,
                                args=types.SimpleNamespace(dev_beacon="label"))
    tpc_collect.stage_draw(ctx)
    bh = st.read_json("draw/beacon.json")["beacon_hex"]
    stored = st.read_jsonl(f"draw/draw_{day}.jsonl")
    derived = tpc_packet.derive_draw(frame, bh, day)
    assert derived == stored and len(derived) > 30
    assert tpc_packet.draw_check(tmp_path, bh)["differ"] == []
    assert tpc_packet.draw_check(tmp_path, "f" * 64)["differ"] == [day]


# ---- P7, the selection --------------------------------------------------------------------------------------
def _row(pid, agent="Cursor", repo=None, u=0.01, in_m=True, screen="claimer", outcome="tests failed",
         other_fail=False, fires=False):
    pm = 0.12 if agent == "Claude_Code" else 0.02
    return {"pr_id": pid, "agent": agent, "repo": repo or f"o/r{pid}", "number": pid, "u": u, "v": u / pm,
            "in_M": in_m, "in_D": True, "eligible": True, "screen": screen, "created_at": "2026-10-20T00:00:00Z",
            "ci": {"outcome": outcome, "has_ci_test_result": outcome in ("tests failed", "tests passed"),
                   "any_failed_job_or_check": other_fail}, "rule_r": {"fires": fires}, "claims": []}


def test_the_cap_reads_v_so_claude_code_rows_are_not_pushed_out():
    rows = [_row(i, agent="Claude_Code", repo="big/repo", u=0.003 * (i + 1)) for i in range(4)]  # v 0.025..0.1
    rows += [_row(10 + i, agent="Cursor", repo="big/repo", u=0.0045 + 0.001 * i) for i in range(3)]  # v 0.225..
    s = tpc_packet.analysis_sets(rows)
    assert sorted(r["pr_id"] for r in s["m_cap"]) == [0, 1, 2]
    by_u = sorted(rows, key=lambda r: r["u"])[:3]
    assert sorted(r["pr_id"] for r in by_u) != [0, 1, 2]  # a cap by u would have kept other rows


def _universe():
    rows = [_row(i, u=0.0001 * (i + 1)) for i in range(3)]                                   # candidates in M
    rows += [_row(100 + i, u=0.001 * (i + 1), outcome="tests passed") for i in range(4)]     # AU4 (and AU2 for 2)
    rows[3]["ci"]["any_failed_job_or_check"] = rows[4]["ci"]["any_failed_job_or_check"] = True
    rows += [_row(200 + i, u=0.002 + 0.0001 * i, screen="none") for i in range(5)]          # AU1
    rows += [_row(300 + i, u=0.003 + 0.0001 * i, screen="none", outcome="tests passed") for i in range(5)]  # AU3
    rows += [_row(400 + i, u=0.03 + 0.01 * i, in_m=False) for i in range(4)]               # top-ups (Cursor, not in M)
    rows += [_row(500, u=0.09, in_m=False, fires=True)]                                     # a top-up rule R fires on
    return rows


def test_every_stratum_and_the_steps_of_each():
    sel = tpc_packet.select_p2(_universe())
    it = sel["items"]
    assert sel["counts"]["candidates_M"] == 3 and sel["counts"]["AU4"] == 4 and sel["counts"]["AU2"] == 2
    assert it[0]["strata"] == ["candidate_M"] and it[0]["steps"] == 2
    assert it[100]["strata"] == ["AU2", "AU4"] and it[100]["steps"] == 2   # shown once, with both steps
    assert it[102]["strata"] == ["AU4"] and it[102]["steps"] == 1
    assert it[300]["strata"] == ["AU3"] and it[300]["steps"] == 1
    assert it[200]["strata"] == ["AU1"] and it[200]["steps"] == 2
    assert it[400]["strata"] == ["topup:Cursor"] and it[500]["strata"] == ["topup:Cursor", "rule_R"]
    assert sel["cut"] == {"AU3": [], "AU2": [], "topup": [], "AU1": []}


def test_the_size_cut_runs_in_p7s_order():
    rows = _universe()
    full = len(tpc_packet.select_p2(rows)["items"])  # 3 + 4 + 5 + 5 + 5 = 22
    assert full == 22
    sel = tpc_packet.select_p2(rows, budget=17)
    assert sorted(sel["cut"]["AU3"]) == [300, 301, 302, 303, 304] and sel["cut"]["AU2"] == []
    sel = tpc_packet.select_p2(rows, budget=15)
    assert sel["cut"]["AU2"] == [] and sel["cut"]["topup"] == [403, 402]  # AU2 rows are AU4 rows: nothing leaves
    assert all("AU2" not in v["strata"] for v in sel["items"].values())
    sel = tpc_packet.select_p2(rows, budget=12)
    assert sel["cut"]["topup"] == [403, 402, 401, 400] and len(sel["cut"]["AU1"]) == 5
    assert 500 in sel["items"]  # rule R's row is never cut
    sel = tpc_packet.select_p2(rows, budget=3)
    assert sel["grew"] and set(sel["items"]) == {0, 1, 2, 100, 101, 102, 103, 500}


def test_audit_sizes_take_the_lowest_v():
    rows = [_row(i, u=0.00001 * (i + 1), outcome="tests passed") for i in range(130)]
    sel = tpc_packet.select_p2(rows)
    au4 = [k for k, v in sel["items"].items() if "AU4" in v["strata"]]
    assert sorted(au4) == list(range(100))


# ---- P8.5 and P9, the shuffle and the key ------------------------------------------------------------------
def _decoys(packet="P2"):
    set_obj, probs = tpc_decoys.build_set(synthetic_decoys.spec(packet))
    assert probs == []
    return set_obj["decoys"]


def _reals(n=5):
    import tpc_items
    return [{"key": {"pr_id": i, "repo": f"o/r{i}", "number": i, "agent": "Cursor", "strata": ["candidate_M"]},
             "item": tpc_items.build_item(f"Body {i}. All tests pass.", {"jobs": []})} for i in range(n)]


def test_assemble_shuffles_by_the_seed_and_numbers_after_the_shuffle():
    beacon = {"kind": "bitcoin", "beacon_hex": "a" * 64}
    a = tpc_packet.assemble("P2", "a" * 64, _reals(), _decoys(), beacon=beacon, salt=b"\x03" * 32)
    b = tpc_packet.assemble("P2", "a" * 64, _reals(), _decoys(), beacon=beacon, salt=b"\x03" * 32)
    c = tpc_packet.assemble("P2", "b" * 64, _reals(), _decoys(), beacon=beacon, salt=b"\x03" * 32)
    assert a["public"] == b["public"] and a["key_digest"] == b["key_digest"]
    assert [x["item_text"] for x in a["public"]["items"]] != [x["item_text"] for x in c["public"]["items"]]
    ids = [x["id"] for x in a["public"]["items"]]
    assert ids == [f"P2-{i:03d}" for i in range(1, 36)]
    decoy_pos = [i for i, k in enumerate(ids) if a["key"]["items"][k]["decoy"]]
    assert decoy_pos != list(range(5, 35))  # interleaved, not appended
    for x in a["public"]["items"]:
        assert set(x) == {"id", "steps", "step1_text", "item_text"}  # nothing else reaches a seat
    assert a["key_digest"] == hashlib.sha256(b"\x03" * 32 + a["key_bytes"]).hexdigest()
    import random
    order = list(range(35))
    random.Random(int(hashlib.sha256(("a" * 64 + ":P2").encode()).hexdigest(), 16)).shuffle(order)
    assert len(set(order)) == 35


def test_q_uses_the_prereg_seed_text():
    assert tpc_packet.Q_SEED == "tests-pass-ci:Q"
    a = tpc_packet.assemble("Q", tpc_packet.Q_SEED, _reals(), _decoys("Q"), beacon={"kind": "q_rule"})
    assert a["public"]["seed"] == "tests-pass-ci:Q" and a["public"]["items"][0]["id"] == "Q-001"


def test_the_p2_lock_row_reads_one_digest_or_none():
    row = "| decoy key digest of packet P2 | `sha256(salt \\|\\| key)` | TO-FILL at the signing commit | public |"
    assert tpc_packet.p2_lock_digest(row) is None
    assert tpc_packet.p2_lock_digest(row.replace("TO-FILL at the signing commit", "`" + "c" * 64 + "`")) == "c" * 64
    prereg = (COLLECTOR.parent / "PREREG_tests_pass_ci_2026_11_02.md").read_text(encoding="utf-8")
    assert tpc_packet.p2_lock_digest(prereg) is None  # unfilled until the signing commit


def test_a_decoy_set_loads_only_by_its_digest_and_the_lock_row(tmp_path):
    set_obj, _ = tpc_decoys.build_set(synthetic_decoys.spec("P2"))
    sealed = tpc_decoys.seal_set(set_obj, salt=b"\x05" * 32)
    tpc_packet._write_set(tmp_path, "P2", sealed, [])
    assert len(tpc_packet.load_decoy_set(tmp_path, "P2")) == 30
    assert len(tpc_packet.load_decoy_set(tmp_path, "P2", lock_digest=sealed["digest"])) == 30
    with pytest.raises(tpc_packet.PacketError):
        tpc_packet.load_decoy_set(tmp_path, "P2", lock_digest="0" * 64)
    with pytest.raises(tpc_packet.PacketError):
        tpc_packet._write_set(tmp_path, "P2", sealed, [])  # never resealed over itself
    (tmp_path / "decoyset_P2.json").write_bytes(sealed["set_bytes"].replace(b"WHOLE_CONTRADICTED", b"NOT_DECIDABLE", 1))
    with pytest.raises(tpc_decoys.DecoyError):
        tpc_packet.load_decoy_set(tmp_path, "P2")


# ---- a store opened before 2026-11-02 ----------------------------------------------------------------------
def _gz(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(gzip.compress(json.dumps(obj).encode("utf-8")))


def make_store(root, beacon_hex, day="2026-10-20"):
    """Frames and read rows for the rows drawn under beacon_hex; every pull request invented."""
    st = Store(root)
    frame = _frame(day)
    st.write(f"frames/frame_{day}.jsonl", b"".join(dumps(r) for r in frame))
    rows = []
    for d in tpc_packet.derive_draw(frame, beacon_hex, day):
        i = d["pr_id"]
        kind = i % 5
        screen = "none" if kind == 4 else "claimer"
        outcome = "tests failed" if kind in (0, 4) else "tests passed"
        raw_dir = f"a2/raw/{i}/att"
        body = "All tests pass." if screen == "claimer" else "Refactor only."
        _gz(root / raw_dir / "pr_node.json.gz", {"body": f"Change {i}.\n\n{body}"})
        concl = "failure" if outcome == "tests failed" else "success"
        _gz(root / raw_dir / "check_runs.p1.json.gz", {"check_runs": [
            {"id": i, "name": "test", "app": {"slug": "github-actions"}, "status": "completed", "conclusion": concl}]})
        _gz(root / raw_dir / "status.p1.json.gz", {"statuses": []})
        _gz(root / raw_dir / "jobs.1.p1.json.gz", {"jobs": [
            {"id": i, "name": "test", "status": "completed", "conclusion": concl,
             "steps": [{"number": 1, "name": "Run tests", "conclusion": concl, "status": "completed"}]}]})
        if concl == "failure":
            p = root / raw_dir / "logs" / f"{i}.log.gz"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(gzip.compress(b"##[group]Run pytest\nE assert False\n1 failed, 9 passed\n##[error]exit 1\n"))
        rec = dict(d, eligible=True, exclusions=[], screen=screen, raw_dir=raw_dir, stale=False,
                   claims=[{"kind": "pass", "explicit": True, "scope": "full", "text": "All tests pass."}]
                   if screen == "claimer" else [],
                   ci={"outcome": outcome, "has_ci_test_result": True, "any_failed_job_or_check": concl == "failure"},
                   rule_r={"fires": False})
        st.write_json(f"a2/rows/{i}.json", rec)
        rows.append(rec)
    return rows


def _write_p2_set(d):
    set_obj, _ = tpc_decoys.build_set(synthetic_decoys.spec("P2"))
    sealed = tpc_decoys.seal_set(set_obj)
    tpc_packet._write_set(d, "P2", sealed, [])
    return sealed


def test_p2_dry_run_builds_under_the_stand_in_on_a_pre_window_store(tmp_path):
    root = tmp_path / "store"
    make_store(root, tpc_packet.STAND_IN_HEX)
    dec = tmp_path / "dec"
    _write_p2_set(dec)
    out = tmp_path / "out"
    rc = tpc_packet.main(["p2", "--root", str(root), "--decoys", str(dec), "--out", str(out), "--dry-run"])
    assert rc == 0
    pub = json.loads((out / "packet_P2.json").read_bytes())
    assert pub["beacon"]["kind"] == "stand_in" and pub["DRY_RUN"] and pub["seed"] == tpc_packet.STAND_IN_HEX
    key = json.loads((out / "key_P2.json").read_bytes())
    salt = bytes.fromhex((out / "key_P2.salt").read_bytes().decode().strip())
    assert (out / "key_P2.sha256").read_bytes().decode().strip() == hashlib.sha256(salt + (out / "key_P2.json").read_bytes()).hexdigest()
    assert sum(1 for k in key["items"].values() if k["decoy"]) == 30
    real = [k for k in key["items"].values() if not k["decoy"]]
    assert real and all(k["strata"] for k in real)
    for it in pub["items"]:
        k = key["items"][it["id"]]
        if it["steps"] == 2:
            assert hashlib.sha256(it["item_text"].encode()).hexdigest() == k["item_sha256"]
    rec = json.loads((out / "build_P2.json").read_bytes())
    assert rec["dry_run"] is True and rec["decoys"] == 30
    assert tpc_packet.main(["p2", "--root", str(root), "--decoys", str(dec), "--out", str(out), "--dry-run"]) == 2


def test_dry_runs_refuse_window_rows_and_the_packets_folder(tmp_path):
    root = tmp_path / "store"
    rows = make_store(root, tpc_packet.STAND_IN_HEX)
    r = rows[0]
    r["created_at"] = "2026-11-03T00:00:00Z"
    Store(root).write_json(f"a2/rows/{r['pr_id']}.json", r, overwrite=True)
    dec = tmp_path / "dec"
    _write_p2_set(dec)
    with pytest.raises(tpc_packet.PacketError):
        tpc_packet.build_p2(root, tpc_packet.load_decoy_set(dec, "P2"), beacon=tpc_packet.stand_in_beacon(), dry_run=True)
    a = types.SimpleNamespace(dry_run=True, out=str(tpc_packet.DEFAULT_OUT / "P2"))
    with pytest.raises(tpc_packet.PacketError):
        tpc_packet._official_out(a, "P2")
    with pytest.raises(tpc_packet.PacketError):
        tpc_packet._official_out(types.SimpleNamespace(dry_run=True, out=None), "P2")


def test_the_official_p2_build_refuses_without_the_named_beacon_or_the_open_seal(tmp_path):
    root = tmp_path / "store"
    make_store(root, GENESIS_ID)
    dec = tmp_path / "dec"
    _write_p2_set(dec)
    decoys = tpc_packet.load_decoy_set(dec, "P2")
    with pytest.raises(tpc_packet.PacketError, match="stand-in"):
        tpc_packet.build_p2(root, decoys, beacon=tpc_packet.stand_in_beacon(), dry_run=False, guard=OpenGuard())
    shut = types.SimpleNamespace(open=False)
    with pytest.raises(tpc_packet.PacketError, match="SEAL_GUARD"):
        tpc_packet.build_p2(root, decoys, beacon=_named(), dry_run=False, guard=shut)
    with pytest.raises(tpc_packet.PacketError, match="a dry run builds under the stand-in"):
        tpc_packet.build_p2(root, decoys, beacon=_named(), dry_run=True)
    st = Store(root)
    st.write("draw/beacon.json", json.dumps(_named()).encode())
    day = "2026-10-20"
    st.write(f"draw/draw_{day}.jsonl", b"".join(dumps(r) for r in tpc_packet.derive_draw(
        st.read_jsonl(f"frames/frame_{day}.jsonl"), GENESIS_ID, day)))
    built, record = tpc_packet.build_p2(root, decoys, beacon=_named(), dry_run=False, guard=OpenGuard(), salt=b"\x09" * 32)
    assert record["dry_run"] is False and built["public"]["seed"] == GENESIS_ID
    # the CLI's official path reads the lock row, which is unfilled in this draft
    rc = tpc_packet.main(["p2", "--root", str(root), "--decoys", str(dec), "--out", str(tmp_path / "o")])
    assert rc == 2


def test_q_takes_thirty_pre_window_claimers_and_refuses_window_rows(tmp_path):
    root = tmp_path / "store"
    make_store(root, tpc_packet.STAND_IN_HEX)
    for i in range(40):  # more claimers with a CI test result
        r = _row(800_000 + i, outcome="tests passed")
        r.update(raw_dir=f"a2/raw/{800_000 + i}/att", repo="o/q", number=i)
        _gz(root / r["raw_dir"] / "pr_node.json.gz", {"body": f"Q item {i}. All tests pass."})
        _gz(root / r["raw_dir"] / "check_runs.p1.json.gz", {"check_runs": []})
        Store(root).write_json(f"a2/rows/{r['pr_id']}.json", r)
    q_dec = tmp_path / "qd"
    set_obj, _ = tpc_decoys.build_set(synthetic_decoys.spec("Q"))
    tpc_packet._write_set(q_dec, "Q", tpc_decoys.seal_set(set_obj), [])
    built, rec = tpc_packet.build_q(root, tpc_packet.load_decoy_set(q_dec, "Q"))
    assert rec["items"] == 30 and len(built["public"]["items"]) == 60
    picked = sorted(k["pr_id"] for k in built["key"]["items"].values() if not k["decoy"])
    rows = [r for r in tpc_packet._read_rows(root) if tpc_packet._claim_ci(r)]
    assert picked == sorted(r["pr_id"] for r in sorted(rows, key=lambda r: tpc_packet.q_order(r["pr_id"]))[:30])
    bad = json.loads((root / "a2" / "rows" / f"{picked[0]}.json").read_bytes())
    bad["created_at"] = "2026-11-05T00:00:00Z"
    Store(root).write_json(f"a2/rows/{picked[0]}.json", bad, overwrite=True)
    with pytest.raises(tpc_packet.PacketError):
        tpc_packet.build_q(root, tpc_packet.load_decoy_set(q_dec, "Q"))


# ---- P1 -------------------------------------------------------------------------------------------------------
def make_testreceipt(d, n2025=6, n2026=3):
    r = d / "evals" / "results"
    r.mkdir(parents=True)
    m2 = ["# M2 contradiction cases", ""]
    for n in range(1, n2025 + 1):
        claim = "pass" if n % 3 else "implied pass"
        m2 += [f"## {n}. Claude_Code, merged=True, claim={claim}", f"https://github.com/acme/w{n}/pull/{n}",
               f"- head commit: {'a' * 40}", "- claim line: > All tests pass", "- evidence: test",
               f"- all failing checks: status ci/unit-{n}: failure; test (ubuntu): failure", ""]
    fr = ["# Fresh M2 (2026)", ""]
    for n in range(1, n2026 + 1):
        fr += [f"## {n}. CONTRADICTED — Devin, merged=False", f"https://github.com/acme/f{n}/pull/{n}",
               "- testreceipt says: x", ""]
    (r / "sheet-m2.md").write_text("\n".join(m2), encoding="utf-8")
    (r / "sheet-fresh.md").write_text("\n".join(fr), encoding="utf-8")


def test_parse_sheet_reads_both_sheets(tmp_path):
    make_testreceipt(tmp_path)
    m2 = tpc_packet.parse_sheet((tmp_path / "evals/results/sheet-m2.md").read_text(encoding="utf-8"), "2025")
    assert [x["n"] for x in m2] == [1, 2, 3, 4, 5, 6]
    assert m2[2]["claim"] == "implied pass" and m2[0]["claim"] == "pass" and m2[0]["agent"] == "Claude_Code"
    assert m2[0]["failing"].startswith("status ci/unit-1: failure")
    fr = tpc_packet.parse_sheet((tmp_path / "evals/results/sheet-fresh.md").read_text(encoding="utf-8"), "2026")
    assert fr[0]["agent"] == "Devin" and fr[0]["url"] == "https://github.com/acme/f1/pull/1"
    assert len(tpc_packet.testreceipt_urls(tmp_path)) == 9


def test_p1_dry_run_uses_the_snapshot_or_the_transcription(tmp_path):
    tr = tmp_path / "tr"
    make_testreceipt(tr)
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "2025_1_77.log").write_bytes(b"##[group]Run pytest\n2 failed\n##[error]exit 1\n")
    snap = {"items": [{"year": "2025", "n": 1, "url": "https://github.com/acme/w1/pull/1", "check_runs": [
        {"id": 77, "name": "test", "app": "github-actions", "status": "completed", "conclusion": "failure",
         "steps": [{"name": "Run tests", "conclusion": "failure", "number": 1}], "log_file": "2025_1_77.log"}],
        "statuses": []}]}
    bodies = {f"https://github.com/acme/w{n}/pull/{n}": f"Change {n}. All tests pass." for n in range(1, 7)}
    bodies.update({f"https://github.com/acme/f{n}/pull/{n}": {"body": f"Fresh {n}. Tests pass."} for n in range(1, 4)})
    set_obj, _ = tpc_decoys.build_set(synthetic_decoys.spec("P1"))
    built, rec = tpc_packet.build_p1(tr, snap, logs, {k: (v["body"] if isinstance(v, dict) else v) for k, v in bodies.items()},
                                     set_obj["decoys"], beacon=tpc_packet.stand_in_beacon(), dry_run=True)
    assert len(built["public"]["items"]) == 39 and rec["without_description"] == []
    assert rec["item_builds"]["2025#1"]["names_only"] is False and rec["item_builds"]["2025#2"]["names_only"] is True
    texts = {built["key"]["items"][x["id"]].get("url"): x["item_text"] for x in built["public"]["items"]}
    assert "log: test" in texts["https://github.com/acme/w1/pull/1"]
    assert texts["https://github.com/acme/w2/pull/2"].split("CI AT THE HEAD COMMIT\n")[1].startswith(
        "Only the names and conclusions of the failing checks survive")
    with pytest.raises(tpc_packet.PacketError):
        tpc_packet.build_p1(tr, snap, logs, {}, set_obj["decoys"], beacon=_named(), dry_run=False)


def test_fetch_bodies_asks_the_seal_guard_on_githubs_created_at():
    prs = {"acme/f1#1": {"body": "Tests pass.", "createdAt": "2026-08-01T00:00:00Z"}}
    gh = FakeGitHub(prs=prs)

    class G:
        asked = []

        def check(self, created_at, what):
            self.asked.append(created_at)
    got = tpc_packet.fetch_bodies(gh, ["https://github.com/acme/f1/pull/1"], G())
    assert got["https://github.com/acme/f1/pull/1"]["body"] == "Tests pass." and G.asked == ["2026-08-01T00:00:00Z"]
    gh.prs["acme/f1#1"]["createdAt"] = "2026-11-03T00:00:00Z"
    import tpc_seal

    class Shut:
        def check(self, created_at, what):
            if tpc_seal.SealGuard.in_window(created_at):
                raise tpc_seal.SealRefused("SEAL_GUARD")
    with pytest.raises(tpc_seal.SealRefused):
        tpc_packet.fetch_bodies(gh, ["https://github.com/acme/f1/pull/1"], Shut())


def test_the_decoys_cli_seals_a_set_and_refuses_a_bad_one(tmp_path):
    spec = synthetic_decoys.spec("P2")
    sp = tmp_path / "spec.json"
    sp.write_bytes(json.dumps(spec).encode())
    assert tpc_packet.main(["decoys", "--spec", str(sp), "--packet", "P2", "--out", str(tmp_path / "s")]) == 0
    d = (tmp_path / "s" / "decoyset_P2.sha256").read_bytes().decode().strip()
    assert len(d) == 64
    spec["decoys"].pop()
    sp.write_bytes(json.dumps(spec).encode())
    assert tpc_packet.main(["decoys", "--spec", str(sp), "--packet", "P2", "--out", str(tmp_path / "t")]) == 1
    assert not (tmp_path / "t" / "decoyset_P2.json").exists()


def test_membership_the_builder_reads_is_the_draws():
    m = membership(GENESIS_ID, 12345, "Cursor")
    assert set(m) >= {"u", "v", "in_D", "in_M"}
    assert dt.date(2026, 11, 2) > dt.date(2026, 10, 20)
