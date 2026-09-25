"""EXTERNAL-1's packet ids (#125): the repaired build numbers items after the shuffle, and
`build --as-published` still reproduces the numbering that was published, leak included.

The ledger and the shelf are gitignored, so every build here runs on a synthetic ledger and a
synthetic `f` table in tmp_path. No test writes next to the committed packet, key or digest.
"""
import hashlib
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
    """One claim per verdict, four claims to a PR record, every claim naming a path.

    The claim text carries an EN SPACE (U+2002), as claim texts in the committed packet do, so
    the corpus is not ASCII and the byte pin below holds the packet's `ensure_ascii`."""
    with path.open("w", encoding="utf-8") as fh:
        for start in range(0, len(verdicts), 4):
            pr_id = 1000 + start // 4
            claims = []
            for j, verdict in enumerate(verdicts[start:start + 4]):
                p = f"src/m{pr_id}/f{j}.py"
                claims.append({"kind": "file_modified", "text": f"updated `{p}`",
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


def test_cli_accepts_as_published_and_refuses_anything_else(corpus, monkeypatch, capsys):
    d = _point_outputs(monkeypatch, corpus / "cli")
    for argv in (["build", "bogus"], ["score", "--as-published"], ["bogus"]):
        assert E.main(argv) == 2
        err = capsys.readouterr().err
        # the usage is the module docstring plus RECIPE, the one place the regeneration is written
        assert "EXTERNAL-1 blind adjudication packet" in err, argv
        assert "Item ids (issue #125)" in err and "Then run `build --as-published`" in err, argv
    assert not any(d.iterdir())
    assert E.main(["build", "--as-published"]) == 0
    assert E.arm_runs(_arms_in_id_order(_read(d)[1])) == 3
    # the module's own report that it reproduced the arm-ordered numbering, read back
    assert "AS PUBLISHED: ids numbered in arm order (3 runs over 3 arms)" in capsys.readouterr().out


def test_main_drives_a_plain_build_and_the_no_argument_default(corpus, monkeypatch):
    """`build` and the bare default are the two paths a person actually types, and neither was
    exercised through main() before."""
    for argv, tag in (([], "bare"), (["build"], "named")):
        d = _point_outputs(monkeypatch, corpus / tag)
        assert E.main(argv) == 0
        packet, key = _read(d)
        assert [it["id"] for it in packet["items"]] == [f"E1-{n:03d}" for n in range(len(key))]
        assert E.arm_runs(_arms_in_id_order(key)) > 5
        assert (d / "key_digest.txt").exists()


# --- the bytes, pinned against the builder that wrote the record -----------------------------------

# sha256 of what origin/main's external1_packet.py (blob daf9e7a5, last changed by f015f706, the
# commit that ran EXTERNAL-1 — the builder of the published packet, before the #125 repair) writes
# on the `corpus` fixture's synthetic ledger and shelf, as LF bytes: the exact text it passed to
# Path.write_text, UTF-8 encoded. That is what it wrote on Linux and what git stores for the
# committed receipts; on Windows it wrote the same text with CRLF. The repaired builder writes LF
# on every platform, so the pin is one set of bytes on CI's ubuntu and on a Windows box. The
# repaired `build --as-published` has to reproduce it byte for byte, or RECIPE regenerates nothing.
AS_PUBLISHED_SHA256 = {
    "packet.json": "203ba269f7444a532ccf40abae939fcf846a643e29c962558e797dbc7557a582",
    "key_SEALED.json": "cd55d49b24a4608f89ce483a8c180011d4d3347e45742c58f72917f3ea1a5884",
    "key_digest.txt": "7e318d108d1a7fc99b21707d107818e37719e30936c12bcd43d642aff2ad56e7",
}
AS_PUBLISHED_BYTES = {"packet.json": 52902, "key_SEALED.json": 15036, "key_digest.txt": 136}


def test_as_published_writes_the_bytes_the_pre_repair_builder_wrote(corpus, monkeypatch):
    d = _point_outputs(monkeypatch, corpus / "pinned")
    assert E.build(as_published=True) == 0
    got = {p.name: p.read_bytes() for p in d.iterdir()}
    assert {n: len(b) for n, b in got.items()} == AS_PUBLISHED_BYTES
    assert {n: hashlib.sha256(b).hexdigest() for n, b in got.items()} == AS_PUBLISHED_SHA256


def test_both_modes_write_lf_on_every_platform(corpus, monkeypatch):
    """Path.write_text writes the platform's separator; the receipts must not depend on it."""
    for tag, as_published in (("lf_repaired", False), ("lf_published", True)):
        d = _point_outputs(monkeypatch, corpus / tag)
        assert E.build(as_published=as_published) == 0
        for p in d.iterdir():
            data = p.read_bytes()
            assert b"\r" not in data and data.endswith(b"\n"), (tag, p.name)


def test_the_committed_packet_is_the_builders_serialisation_of_itself():
    """The committed packet carries non-ASCII characters in its claim texts (U+2002 among them),
    so `ensure_ascii` is part of the record. Read as LF, the bytes git stores; a Windows
    checkout renders it with CRLF."""
    text = (CMF / "external1_packet.json").read_bytes().decode("utf-8").replace("\r\n", "\n")
    assert "\r" not in text and any(ord(c) > 127 for c in text)
    assert json.dumps(json.loads(text), indent=1, ensure_ascii=False) + "\n" == text
    assert E.packet_json(json.loads(text)) == text


def test_the_cited_sampling_line_is_still_line_63():
    """ANALYSIS_base_rate_ceiling_2026_09_01.md:45 cites `external1_packet.py:63` for this line.
    The ANALYSIS is certified and frozen, and the #125 repair once moved the line to 102."""
    analysis = (CMF / "ANALYSIS_base_rate_ceiling_2026_09_01.md").read_text(encoding="utf-8")
    assert "`external1_packet.py:63`" in analysis.splitlines()[44]
    line = (CMF / "external1_packet.py").read_text(encoding="utf-8").splitlines()[62]
    assert line == "    sample_acc = rng.sample(acc, N_ACC)", (
        "external1_packet.py:63 is no longer `sample_acc = rng.sample(acc, N_ACC)`, the line "
        "ANALYSIS_base_rate_ceiling_2026_09_01.md:45 cites. Do not edit the ANALYSIS: move the "
        "line back, or commit a new note beside it naming the new line (as "
        "NOTE_external1_id_repair_2026_09_25.md does for this one) and point this test at it.")


def test_the_repaired_build_writes_different_bytes(corpus, monkeypatch):
    """The pin above is a claim about `--as-published` alone: a plain build must not match it,
    or the pin would pass on a module that ignored the flag."""
    d = _point_outputs(monkeypatch, corpus / "pinned_default")
    assert E.build() == 0
    got = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in d.iterdir()}
    assert got["packet.json"] != AS_PUBLISHED_SHA256["packet.json"]
    assert got["key_SEALED.json"] != AS_PUBLISHED_SHA256["key_SEALED.json"]


# --- score, on the renumbered ids ------------------------------------------------------------------

def _answers_for(key, wrong_accusations=4, wrong_decoys=0):
    """`wrong_decoys` of the 30 decoys answered wrong, the rest right; `wrong_accusations` of the
    100 accusations called SUPPORTED, the rest CONTRADICTED."""
    out, wrong, missed = {}, 0, 0
    for iid in sorted(key):
        t = key[iid]["truth"]
        if t in (VER, SYN):
            right = "SUPPORTED" if t == VER else "CONTRADICTED"
            wrong_answer = "CONTRADICTED" if t == VER else "SUPPORTED"
            if missed < wrong_decoys:
                out[iid], missed = wrong_answer, missed + 1
            else:
                out[iid] = right
        elif wrong < wrong_accusations:
            out[iid], wrong = "SUPPORTED", wrong + 1
        else:
            out[iid] = "CONTRADICTED"
    return out


def test_score_round_trips_the_renumbered_ids(corpus, monkeypatch):
    d = _point_outputs(monkeypatch, corpus / "scored")
    assert E.build() == 0
    key = _read(d)[1]
    E.ANSWERS.write_text(json.dumps(_answers_for(key)), encoding="utf-8")
    assert E.score() == 0
    out = json.loads(E.RESULT.read_text(encoding="utf-8"))
    assert out["decoys_correct"] == 30 and out["decoys_total"] == 30
    assert out["adjudicator_reliable"] is True and out["decoy_misses"] == []
    assert out["accusations_scored"] == 100 and out["accusations_upheld"] == 96
    assert out["precision"] == 0.96 and out["gate_G_E1_pass"] is True


VOID = ("ADJUDICATION VOID — fewer than 27/30 decoys correct. "
        "No headline number may be published.")


@pytest.mark.parametrize("wrong_decoys, wrong_accusations, reliable, precision, gate", [
    (3, 4, True, 0.96, True),       # 27/30: on the decoy floor, still reliable
    (4, 4, False, None, False),     # 26/30: below it, the adjudication is void
    (0, 5, True, 0.95, True),       # 95/100 upheld: on the precision gate
    (0, 6, True, 0.94, False),      # 94/100 upheld: below it
])
def test_score_holds_both_thresholds_from_both_sides(corpus, monkeypatch, capsys, wrong_decoys,
                                                     wrong_accusations, reliable, precision, gate):
    """EXTERNAL-1's validity gate (27/30 decoys) and headline gate (precision >= 0.95), each at
    its boundary and one step below it. A single passing point pins neither."""
    d = _point_outputs(monkeypatch, corpus / f"thr_{wrong_decoys}_{wrong_accusations}")
    assert E.build() == 0
    key = _read(d)[1]
    E.ANSWERS.write_text(json.dumps(_answers_for(key, wrong_accusations, wrong_decoys)),
                         encoding="utf-8")
    capsys.readouterr()
    assert E.score() == 0
    stdout = capsys.readouterr().out
    out = json.loads(E.RESULT.read_text(encoding="utf-8"))
    assert out["decoys_correct"] == 30 - wrong_decoys and out["decoys_total"] == 30
    assert out["adjudicator_reliable"] is reliable
    assert out["accusations_scored"] == 100
    assert out["accusations_upheld"] == 100 - wrong_accusations
    assert out["precision"] == precision
    assert out["gate_G_E1_pass"] is gate
    assert len(out["decoy_misses"]) == wrong_decoys
    assert all(key[iid]["truth"] == truth and answer != ("SUPPORTED" if truth == VER
                                                          else "CONTRADICTED")
               for iid, truth, answer in out["decoy_misses"])
    assert (VOID in stdout) is (not reliable)


def test_score_refuses_an_unanswered_item_and_writes_no_result(corpus, monkeypatch, capsys):
    """The prereg's 'Answer every item': a missing answer voids the run rather than being
    scored as a miss or a false positive."""
    d = _point_outputs(monkeypatch, corpus / "unanswered")
    assert E.build() == 0
    key = _read(d)[1]
    answers = _answers_for(key)
    missing = sorted(key)[57]
    del answers[missing]
    E.ANSWERS.write_text(json.dumps(answers), encoding="utf-8")
    capsys.readouterr()
    assert E.score() == 1
    assert f"REFUSED: item {missing} unanswered" in capsys.readouterr().out
    assert not E.RESULT.exists()


def test_score_refuses_when_one_key_byte_changes(corpus, monkeypatch, capsys):
    d = _point_outputs(monkeypatch, corpus / "tampered_key")
    assert E.build() == 0
    key = _read(d)[1]
    E.ANSWERS.write_text(json.dumps(_answers_for(key)), encoding="utf-8")
    text = E.KEY.read_text(encoding="utf-8")
    tampered = text.replace(VER, VER[:-1] + VER[-1].upper(), 1)   # one byte, one arm relabelled
    assert tampered != text and len(tampered) == len(text)
    E.KEY.write_text(tampered, encoding="utf-8")
    assert E.score() == 1
    assert "does not match the committed digest" in capsys.readouterr().out
    assert not E.RESULT.exists()


def test_score_refuses_when_the_committed_digest_changes(corpus, monkeypatch, capsys):
    d = _point_outputs(monkeypatch, corpus / "tampered_digest")
    assert E.build() == 0
    key = _read(d)[1]
    E.ANSWERS.write_text(json.dumps(_answers_for(key)), encoding="utf-8")
    text = E.DIGEST.read_text(encoding="utf-8")
    recorded = text.split(" = ", 1)[1].split("\n", 1)[0]
    assert len(recorded) == 64
    flipped = ("0" if recorded[0] != "0" else "1") + recorded[1:]
    E.DIGEST.write_text(text.replace(recorded, flipped), encoding="utf-8")
    assert E.score() == 1
    assert "does not match the committed digest" in capsys.readouterr().out
    assert not E.RESULT.exists()


# --- a refused build touches nothing ---------------------------------------------------------------

def _fresh_clone(monkeypatch, d):
    """The state every clone is in: packet and digest committed, the sealed key gitignored."""
    _point_outputs(monkeypatch, d)
    assert E.build(as_published=True) == 0
    sealed = E.KEY.read_bytes()
    E.KEY.unlink()
    before = {p.name: p.read_bytes() for p in d.iterdir()}
    assert sorted(before) == ["key_digest.txt", "packet.json"]
    return sealed, before


def test_the_fresh_clone_state_refuses_a_plain_build_and_writes_nothing(corpus, monkeypatch,
                                                                        capsys):
    """A key minted under the repaired numbering would be a different key under a committed
    digest, so a plain build refuses there and names the command that can restore the key."""
    d = corpus / "clone_plain"
    _, before = _fresh_clone(monkeypatch, d)
    capsys.readouterr()
    assert E.build() == 1
    out = capsys.readouterr().out
    assert "REFUSED" in out and "fresh-clone state" in out and "build --as-published" in out
    assert not E.KEY.exists()
    assert {p.name: p.read_bytes() for p in d.iterdir()} == before


def test_as_published_restores_the_sealed_key_from_the_fresh_clone_state(corpus, monkeypatch):
    """RECIPE's regeneration, run from the state the repository ships in. The packet and digest
    it builds match the ones on disk, and the digest is the key's salted hash, so the key it
    writes is the sealed key, byte for byte."""
    d = corpus / "clone_restore"
    sealed, before = _fresh_clone(monkeypatch, d)
    assert E.main(["build", "--as-published"]) == 0
    assert E.KEY.read_bytes() == sealed
    assert {p.name: p.read_bytes() for p in d.iterdir() if p.name != "key_SEALED.json"} == before


def test_as_published_reads_a_crlf_checkout_of_the_record_as_the_record(corpus, monkeypatch):
    """Git stores the receipts with LF; a Windows checkout renders them with CRLF. That rendering
    is the record, so the restore goes ahead, and what it leaves on disk is the LF bytes."""
    d = corpus / "clone_crlf"
    sealed, before = _fresh_clone(monkeypatch, d)
    for name, data in before.items():
        (d / name).write_bytes(data.replace(b"\n", b"\r\n"))
    assert E.build(as_published=True) == 0
    assert E.KEY.read_bytes() == sealed
    assert {p.name: p.read_bytes() for p in d.iterdir() if p.name != "key_SEALED.json"} == before


def test_as_published_from_the_fresh_clone_state_refuses_a_different_record(corpus, monkeypatch,
                                                                            capsys):
    """A different ledger in place (the one regenerated for the CORRECTION, say) puts different
    claims in the sample: the packet and digest it builds are not the committed ones, so it
    writes nothing and mints no key."""
    d = corpus / "clone_wrong_ledger"
    _, before = _fresh_clone(monkeypatch, d)
    verdicts = ["CONTRADICTED"] * 130 + ["VERIFIED"] * 45 + ["UNCHECKABLE"] * 9
    random.Random(8).shuffle(verdicts)
    _write_shelf(corpus / "other_shelf.sqlite",
                 _write_ledger(corpus / "other_ledger.jsonl", verdicts))
    monkeypatch.setattr(E, "LEDGER", corpus / "other_ledger.jsonl")
    monkeypatch.setattr(E, "DB", corpus / "other_shelf.sqlite")
    capsys.readouterr()
    assert E.build(as_published=True) == 1
    out = capsys.readouterr().out
    assert "already exist with different contents" in out
    assert "packet.json" in out and "key_digest.txt" in out
    assert not E.KEY.exists()
    assert {p.name: p.read_bytes() for p in d.iterdir()} == before


def test_as_published_refuses_when_the_record_on_disk_is_not_the_record_it_built(
        corpus, monkeypatch, capsys):
    """The early check cannot compare bytes it has not built yet, so `--as-published` — the one
    mode allowed to write over a complete record — is held by the exact-bytes check at the end.
    A wrong ledger in place is how that goes wrong in practice."""
    d = _point_outputs(monkeypatch, corpus / "mismatch")
    assert E.build(as_published=True) == 0
    capsys.readouterr()
    packet = E.PACKET.read_text(encoding="utf-8")
    E.PACKET.write_text(packet.replace("E1-000", "E1-999", 1), encoding="utf-8")
    before = {p.name: p.read_bytes() for p in d.iterdir()}
    assert E.build(as_published=True) == 1
    assert "already exist with different contents" in capsys.readouterr().out
    assert {p.name: p.read_bytes() for p in d.iterdir()} == before


def test_a_refused_build_reads_neither_the_ledger_nor_the_shelf(corpus, monkeypatch, capsys):
    """The refusal is decided from the output files alone: point the 5 GB shelf and the ledger at
    paths that do not exist and the build still returns 1 instead of raising."""
    _point_outputs(monkeypatch, corpus / "no_inputs")
    assert E.build(as_published=True) == 0
    capsys.readouterr()
    monkeypatch.setattr(E, "LEDGER", corpus / "nowhere" / "ledger.jsonl")
    monkeypatch.setattr(E, "DB", corpus / "nowhere" / "shelf.sqlite")
    assert E.build() == 1
    assert "REFUSED" in capsys.readouterr().out


def test_the_shelf_is_opened_read_only(corpus, monkeypatch):
    """A read-write connection to one of the recipe's inputs can leave journal sidecars beside a
    5 GB file the builder never writes to."""
    seen = []
    real_connect = sqlite3.connect

    def spy(*a, **kw):
        seen.append((a, kw))
        return real_connect(*a, **kw)

    monkeypatch.setattr(E, "sqlite3", type("Shim", (), {"connect": staticmethod(spy)}))
    _point_outputs(monkeypatch, corpus / "readonly")
    assert E.build() == 0
    assert len(seen) == 1
    (uri,), kw = seen[0]
    assert kw == {"uri": True}
    assert uri.startswith("file:") and uri.endswith("?mode=ro")
    con = real_connect(uri, uri=True)
    try:
        with pytest.raises(sqlite3.OperationalError):
            con.execute("CREATE TABLE zz (x)")
    finally:
        con.close()
    assert sorted(p.name for p in corpus.iterdir() if p.name.startswith("shelf")) \
        == ["shelf.sqlite"]


# --- the clustering guard, at its threshold --------------------------------------------------------

def _chunks(seq, n):
    size, extra = divmod(len(seq), n)
    out, i = [], 0
    for j in range(n):
        take = size + (1 if j < extra else 0)
        out.append(seq[i:i + take])
        i += take
    return out


def _shuffle_leaving_runs(monkeypatch, n_runs):
    """Replace the builder's Random with one whose shuffle leaves exactly `n_runs` arm runs:
    alternating accusation / verified-decoy blocks, then the synthetic block."""
    alt = n_runs - 1

    class Staged(random.Random):
        def shuffle(self, x):
            by_arm = {}
            for e in x:
                by_arm.setdefault(e[2]["truth"], []).append(e)
            a_blocks = _chunks(by_arm[ACC], (alt + 1) // 2)
            v_blocks = _chunks(by_arm[VER], alt // 2)
            ordered = []
            for i in range(alt):
                ordered += (a_blocks if i % 2 == 0 else v_blocks)[i // 2]
            x[:] = ordered + by_arm[SYN]

    monkeypatch.setattr(E, "random", type("Shim", (), {"Random": Staged}))


def test_the_clustering_guard_refuses_five_runs_and_accepts_six(corpus, monkeypatch):
    """The guard's bar is `runs <= n_arms + 2`, i.e. 5 with three arms. Only a no-op shuffle
    (3 runs) exercised it before, so any threshold from 3 upward survived."""
    d = _point_outputs(monkeypatch, corpus / "runs5")
    _shuffle_leaving_runs(monkeypatch, 5)
    with pytest.raises(AssertionError, match="5 runs over 3 arms"):
        E.build()
    assert not any(d.iterdir())

    d = _point_outputs(monkeypatch, corpus / "runs6")
    _shuffle_leaving_runs(monkeypatch, 6)
    assert E.build() == 0
    assert E.arm_runs(_arms_in_id_order(_read(d)[1])) == 6


def test_arm_runs_counts_maximal_runs():
    assert E.arm_runs([]) == 0
    assert E.arm_runs(["a"]) == 1
    assert E.arm_runs(["a", "a", "a"]) == 1
    assert E.arm_runs(["a", "b", "a"]) == 3
    assert E.arm_runs(["a", "a", "b", "b", "c"]) == 3
