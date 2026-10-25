# -*- coding: utf-8 -*-
"""styxx.research_capsule — the research capsule, Phase A: builder and verifier over a synthetic chain.

The chain here is SYNTHETIC: a PREREG with a two-gate table, a receipt scored by that table, a
RESULT sworn to the receipt over a snapshot at a nominal commit, a charon line made from the
derived core, an anchor line shaped as ``styxx.clock`` records one. Every offline link holds over
it — which is the disclosed limit (a forger controlling the whole file passes every offline link),
pinned below as a test and not hidden. The load-bearing tests are the two batteries: a digest
mutation flips exactly the link that owns the bytes, and a re-built CONSISTENT capsule with one
link made false flips exactly that link (the hash-blind verifier the red team named, F1, would
pass the digest battery and fail this one). Controls that must NOT fire sit beside them. The
page's own script (layer 1, run under node) is held to the installed verifier (layer 2) link by
link on the clean capsule and on every member of both batteries, the sworn span values included;
on every integer literal of the clean page written as a float literal (``N.0``, which no digest
sees); on fields written with another JSON type where an integer is read; and on the shapes of the
anchor lines, the public clocks and their times.
"""
from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from styxx import charon, clock, sworn
from styxx import research_capsule as rc
from styxx.research_capsule import (FAILS, HOLDS, LABEL, ONLINE, SIZE_CAP, SPEC, STATED, SUMMARY_FAILS,
                                    SUMMARY_OK, build, compose, datasheet, render_html,
                                    score_frozen, verify)

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "a" * 40
BLOB_COMMIT = "b" * 40
AT_COMMIT = "c" * 40
PAYLOAD_RE = re.compile(r'<script type="application/json" id="oath-capsule">(.*?)</script>', re.S)
TS = "2026-10-11T22:00:00Z"

GATES = """```gates
{"gates": {"G_A": {"metric": "gap", "op": ">=", "value": 0.15,
                   "power_basis": "half the measured gap of the flagship receipt"},
           "G_B": {"metric": "dev", "op": "<=", "value": 0.03,
                   "power_basis": "the device tolerance of a 784-step scan"}},
 "outcomes": [{"when": {"G_B": false}, "verdict": "INVALID__anchors_drifted"},
              {"when": {"G_B": true, "G_A": true}, "verdict": "EFFECT_HOLDS"},
              {"when": {"G_B": true, "G_A": false}, "verdict": "EFFECT_ABSENT"}],
 "smoke_verdict": "INVALID__smoke_plumbing_only"}
```"""

PREREG = ("# PREREG — a synthetic experiment — 2026-10-11\n\nFROZEN before data. Runner `run_x.py`.\n\n"
          "## Gates\n\n" + GATES + "\n\n## Discipline\n\nCommitted before the run; RESULT sworn to the receipt.\n")
RUNNER = b"# -*- coding: utf-8 -*-\n# run_x.py -- frozen by PREREG_x.md\nimport json\nprint('a synthetic runner')\n"
RUN_NOTE = (b"# run note\n\nRun of run_x.py as frozen; start 2026-10-11T17:32:47Z, end 2026-10-11T18:54:26Z, "
            b"exit 0.\n")
ANCHOR_RECEIPT = b'{"result": {"free_acc": 0.9195, "clamped_acc": 0.6073}}\n'


def _receipt_bytes(metrics: dict, smoke: bool = False, verdict=None, gates=None) -> bytes:
    sc = score_frozen(GATES.split("\n", 1)[1].rsplit("\n```", 1)[0], metrics)
    obj = {"prereg": "PREREG_x.md",
           "config": {"task": "synthetic", "seeds": [0, 1], "device": "cuda", "torch": "2.5.1", "smoke": smoke},
           "params": {"free": 101002},
           "test_acc": {"free": [0.924, 0.915], "clamped": [0.5529, 0.6617]},
           "metrics": metrics,
           "verdict": verdict or sc["verdict"], "gates": gates or sc["gates"]}
    return (json.dumps(obj, indent=2) + "\n").encode("utf-8")


def _result_bytes(receipt_path: str) -> bytes:
    return (f"# RESULT — synthetic — 2026-10-11\n\nEvery number is sworn to the receipt at commit `{COMMIT[:12]}`.\n"
            f'<sworn r="path:{receipt_path}#/metrics/gap" k="numeric">The gap was 0.3122.</sworn> '
            f'<sworn r="path:{receipt_path}#/params/free" k="numeric">FREE has 101002 parameters.</sworn>\n\n'
            f"## What this does not say\n\nOne task, two seeds, no interval.\n").encode("utf-8")


def _result_every_kind(receipt_path: str, receipt: bytes) -> bytes:
    """A RESULT whose spans are numeric, quote and hash over the snapshot, so the browser's
    snapshot resolver is exercised on more than one kind."""
    return (f"# RESULT — synthetic, every kind — 2026-10-11\n\nSworn at `{COMMIT[:12]}`.\n"
            f'<sworn r="path:{receipt_path}#/metrics/gap" k="numeric">The gap was 0.3122.</sworn> '
            f'<sworn r="path:{receipt_path}#/config/task" k="quote">The task was `synthetic`.</sworn> '
            f'<sworn r="path:{receipt_path}" k="hash">The receipt is {sworn._sha256(receipt)}.</sworn>\n').encode("utf-8")


def _sworn_trio(result_bytes: bytes, tree_files: dict, commit: str = COMMIT):
    side = sworn.to_sidecar(result_bytes, "RESULT_x.md", commit)
    tree = sworn.SnapshotTree.from_memory(sworn.MemoryTree(tree_files, commit))
    core = sworn.verify(sidecar=side, tree=tree, commit=commit)
    rec = sworn.issue_receipt(core, timestamp=TS)
    return side, rec, core


def _charon_entry(side: dict, rec: dict, core: dict, result_path: str, header: dict, seq: int = 1,
                  prev=None, **override):
    """The core ``charon.derive_sworn`` writes, from the carried objects; then ``make_entry``."""
    shas = sorted({s["resolved_sha256"] for s in core["spans"] if s.get("resolved_sha256")})
    verdict = core["document_verdict"]
    counts = dict(core["counts"])
    counts["unresolved_spans"] = core["unresolved"]
    counts["recorded_verdict"] = rec.get("document_verdict")
    counts["receipt_check"] = {"status": "VERIFIED", "digest_match": True, "verdict_reproduces": True,
                               "same_verifier_build": True, "schema": rec["schema"]}
    derived = {"kind": "sworn",
               "subject": {"name": "RESULT_x.md", "path": result_path, "sha256": sworn._sha256(sworn.render(side))},
               "at": {"commit": side["commit"], "manifest_digest": core.get("manifest_digest"),
                      "document_at_commit": True},
               "receipts": {"n": len(shas), "sha256": shas, "cited": None, "vacuous": False},
               "verifier": charon._verifier("sworn"),
               "verdict": verdict,
               "verdict_class": {"SWORN-HELD": "HELD", "SWORN-FAILED": "FAILED", "UNSWORN": "UNSWORN"}[verdict],
               "counts": counts, "floor": core["coverage"].get("sentence_share"), "rungs": core.get("rungs"),
               "reproduced": True}
    derived.update(override)
    return charon.make_entry(derived, seq, prev or charon.header_digest(header), timestamp=TS)


HEADER = {"schema": charon.LOG_SCHEMA, "name": "test", "created": TS, "population": "test",
          "certifies": charon._CERTIFIES}


def _inputs(**kw) -> dict:
    """A consistent synthetic chain. Keyword overrides replace whole inputs AFTER derivation so a
    test can make exactly one link false while every digest stays consistent."""
    metrics = kw.pop("metrics", {"gap": 0.3122, "dev": 0.0048})
    receipt = kw.pop("receipt_bytes", None) or _receipt_bytes(metrics)
    receipt_path = "papers/x/x_result.json"
    result_fn = kw.pop("result_fn", None)
    result = result_fn(receipt_path, receipt) if result_fn else _result_bytes(receipt_path)
    side, rec, core = _sworn_trio(result, {receipt_path: receipt})
    entry = _charon_entry(side, rec, core, "papers/x/RESULT_x.md", HEADER)
    prereg = PREREG.encode("utf-8")
    digest = sworn._sha256(prereg)
    anchor = {"n": 2, "kind": "sealed-prereg", "digest": digest, "memo": clock.memo("sealed-prereg", digest),
              "tx": "5" * 87, "slot": 371_000_000, "block_time": "2026-10-11T20:15:00Z"}
    inputs = {
        "experiment": {"prereg_name": "PREREG_x.md", "repo": "github.com/example/lab", "at_commit": AT_COMMIT},
        "prereg": {"path": "papers/x/PREREG_x.md", "bytes": prereg, "blob_commit": BLOB_COMMIT},
        "anchors": [anchor], "anchors_path": "papers/charon/anchors.jsonl",
        "clock_check": {"status": "ANCHORED", "slot": 371_000_000, "block_time": "2026-10-11T20:15:00Z",
                        "rpc": "https://api.mainnet-beta.solana.com", "beacon": "ab" * 32},
        "receipts": [{"path": receipt_path, "bytes": receipt},
                     {"path": "papers/x/anchor_result.json", "bytes": ANCHOR_RECEIPT}],
        "runners": [{"path": "papers/x/run_x.py", "bytes": RUNNER}],
        "run_note": {"path": "papers/x/x_RUN_NOTE.md", "bytes": RUN_NOTE},
        "result": {"path": "papers/x/RESULT_x.md", "bytes": result},
        "sidecar": side, "sworn_receipt": rec,
        "snapshot": {"commit": COMMIT, "entries": {receipt_path: {"mode": "100644", "size": len(receipt),
                                                                  "sha256": sworn._sha256(receipt),
                                                                  "bytes": receipt}}},
        "charon": {"header": HEADER, "entry": entry, "log_path": "papers/charon/charon.log.jsonl"},
        "public_clocks": [{"name": "the merge of the PREREG's pull request", "binds": "the PREREG blob",
                           "read": "https://api.github.com/repos/example/lab/pulls/1 -> merged_at",
                           "stated": "2026-09-02T10:00:00Z"}],
        "stated": {"run_started_utc": "2026-10-11T17:32:47Z", "run_ended_utc": "2026-10-11T18:54:26Z",
                   "device": "an RTX 4070 Laptop GPU", "data_sha256": "98d68505" + "0" * 56},
    }
    inputs.update(kw)
    inputs["_core"] = core
    return inputs


@pytest.fixture
def minted(tmp_path):
    inp = _inputs()
    inp.pop("_core")
    return build(inp, tmp_path / "x.research.html", created=TS)


def _payload(cap: Path) -> dict:
    return json.loads(PAYLOAD_RE.search(cap.read_text(encoding="utf-8")).group(1))


def _page_outside_payload(cap: Path) -> str:
    html = cap.read_text(encoding="utf-8")
    m = PAYLOAD_RE.search(html)
    return html[:m.start(1)] + html[m.end(1):]


def _reseal(cap: Path, payload: dict, out: Path | None = None, sort_keys: bool = True) -> Path:
    """Write a payload back into the page — the forger's move and no more."""
    html = cap.read_text(encoding="utf-8")
    body = json.dumps(payload, ensure_ascii=False, sort_keys=sort_keys).replace("<", "\\u003c")
    m = PAYLOAD_RE.search(html)
    out = out or cap
    out.write_bytes((html[:m.start(1)] + body + html[m.end(1):]).encode("utf-8"))
    return out


def _write_consistent(tmp: Path, inputs: dict, mutate=None, name="m.research.html") -> Path:
    """A CONSISTENT capsule with one link made false: compose without build's refusals, apply a
    payload-level mutation (recomputing core_sha256), render, write."""
    inputs = dict(inputs)
    inputs.pop("_core", None)
    payload = compose(inputs, created=TS)
    if mutate:
        mutate(payload)
        payload["core_sha256"] = rc._core_sha256(payload)
    out = tmp / name
    out.write_bytes(render_html(payload).encode("utf-8"))
    return out


def _fails(rep: dict) -> set:
    return {l["id"] for l in rep["links"] if l["offline"] == FAILS}


def _flip_b64(entry: dict, needle: bytes) -> None:
    """Flip one byte of the DECODED bytes at `needle` and re-encode: the decoded bytes change."""
    data = base64.b64decode(entry["b64"])
    i = data.index(needle)
    data = data[:i] + bytes([data[i] ^ 0x01]) + data[i + 1:]
    assert data != base64.b64decode(entry["b64"])
    entry["b64"] = base64.b64encode(data).decode("ascii")


# ----------------------------------------------------------------------------- the clean capsule

def test_a_consistent_chain_builds_and_every_offline_link_holds(minted):
    rep = verify(minted)
    assert rep["ok"], [(l["id"], l["problems"]) for l in rep["links"]] + [rep["payload"]]
    assert rep["spec"] == SPEC and rep["summary"] == SUMMARY_OK
    assert [l["id"] for l in rep["links"]] == ["L0", "L1", "L1b", "L2", "L3", "L4", "L5", "L6"]
    assert all(l["offline"] == HOLDS for l in rep["links"])
    assert rep["payload"]["core_sha256_ok"] and rep["payload"]["page_ok"] and rep["payload"]["verifier_js_ok"]
    assert rep["payload"]["size_ok"] and rep["payload"]["disclosure_hits"] == []
    assert rep["bytes"] <= SIZE_CAP
    l3 = next(l for l in rep["links"] if l["id"] == "L3")
    assert any("EFFECT_HOLDS" in c for c in l3["checked"])


def test_the_cli_verify_prints_the_datasheet_and_exits_zero(minted, capsys):
    assert rc.main(["verify", str(minted)]) == 0
    out = capsys.readouterr().out
    assert SUMMARY_OK in out and LABEL in out
    assert "verified" not in out.lower()


def test_three_readings_per_link_with_the_settled_labels(minted):
    rep = verify(minted)
    lines = datasheet(rep)
    for l in rep["links"]:
        assert l["offline"] in (HOLDS, FAILS)
        assert l["online"]["reading"] in (ONLINE, rc.NOTHING_ONLINE)
        assert l["stated"]["reading"] in (STATED, rc.NOTHING_STATED)
        block = lines[lines.index(next(x for x in lines if x.startswith(l["id"] + " "))):]
        assert block[1].startswith("     offline  ") and block[2].startswith("     online   ")
        assert any(x.startswith("     stated   ") for x in block[3:40])
    # the links that reach out of the file print the public read a stranger would make
    by_id = {l["id"]: l for l in rep["links"]}
    assert any(r.startswith(f"git show {BLOB_COMMIT}:papers/x/PREREG_x.md") for r in by_id["L0"]["online"]["reads"])
    assert any("getTransaction" in r for r in by_id["L1"]["online"]["reads"])
    assert any("merged_at" in r for r in by_id["L1b"]["online"]["reads"])
    assert any(r.startswith(f"git show {COMMIT}:papers/x/x_result.json") for r in by_id["L4"]["online"]["reads"])
    assert any("seq 1" in r for r in by_id["L5"]["online"]["reads"])
    assert by_id["L3"]["online"]["reading"] == rc.NOTHING_ONLINE
    # what nothing checks is labelled as the lab's statement, with the anchor's sign against the run
    assert by_id["L1"]["stated"]["reading"] == STATED
    delta = next(f for f in by_id["L1"]["stated"]["facts"] if "block_time − run_end" in f)
    assert "+1 h 20 min" in delta and "after the run ended" in delta
    assert "orders the PREREG's digest before the receipt's reading, not before the run" in delta
    assert any("wallet" in f and "not here" in f for f in by_id["L1"]["stated"]["facts"])
    assert any("run_ended_utc" in f for f in by_id["L2"]["stated"]["facts"])
    assert any("power_basis" in f for f in by_id["L3"]["stated"]["facts"])
    # the page prints the same three labels, and never the whole-chain word sworn uses
    page = _page_outside_payload(minted)
    assert page.count("CHECKABLE-ONLINE") >= 7 and page.count("STATED-BY-THE-LAB") >= 7
    assert "HOLDS / FAILS" in page and "NOT RUN" in page
    assert LABEL in page and SUMMARY_OK in page and SUMMARY_FAILS in page
    assert "verified" not in page.lower()
    assert "sealed before the run" not in page.lower() and "sealed before the run" not in "\n".join(lines).lower()


def test_the_summary_wording_is_internally_consistent_never_a_stronger_word(minted):
    lines = datasheet(verify(minted))
    assert any(x.startswith(SUMMARY_OK + ":") for x in lines)
    assert "nothing here is checked against the world" in "\n".join(lines)
    assert lines[-1].startswith("Nothing above vouches for the run")


# ---------------------------------------------------------------------- the whole-file forger

def test_the_synthetic_chain_is_a_forger_and_passes_every_offline_link_with_the_label(minted):
    """The honest negative, pinned: nothing here ran, and every offline link reads HOLDS. If a
    change made this capsule fail offline, the change would be claiming an offline check tells a
    forger from a lab, and none can."""
    rep = verify(minted)
    assert rep["ok"] and all(l["offline"] == HOLDS for l in rep["links"])
    assert rep["label"] == LABEL
    assert "a forger controlling the whole file passes every offline link" in LABEL
    assert LABEL in minted.read_text(encoding="utf-8")


# ------------------------------------------------------------------ battery (a): digest mutations

DIGEST_BYTES = [
    ("prereg", b"FROZEN before data", "L0"),             # outside the gates block: L3 keeps its table
    ("receipt", b"synthetic", "L2"),                     # inside a config string: metrics unchanged
    ("runner", b"synthetic runner", "L2"),
    ("run_note", b"exit 0", "L2"),
    ("result", b"no interval", "L4"),                    # narrative, outside every span
]


def _digest_mutant(minted: Path, out: Path, where: str, needle: bytes) -> Path:
    p = _payload(minted)
    a = p["artifacts"]
    target = {"prereg": a["prereg"], "receipt": a["receipts"][0], "runner": a["runners"][0],
              "run_note": a["run_note"], "result": a["result"]}[where]
    _flip_b64(target, needle)
    return _reseal(minted, p, out)


@pytest.mark.parametrize("where,needle,link", DIGEST_BYTES)
def test_a_digest_mutation_flips_exactly_the_link_that_owns_the_bytes(minted, tmp_path, where, needle, link):
    cap = _digest_mutant(minted, tmp_path / "t.research.html", where, needle)
    rep = verify(cap)
    assert _fails(rep) == {link}, [(l["id"], l["problems"]) for l in rep["links"]]
    assert rep["payload"]["core_sha256_ok"] is False and rep["ok"] is False
    assert any(x.startswith(SUMMARY_FAILS) for x in datasheet(rep))


FIELD_EDITS = [
    (lambda a: a["anchors"][0].__setitem__("digest", "0" * 64), "L1"),            # memo no longer kind+digest
    (lambda a: a["anchors"][0].__setitem__("memo", a["anchors"][0]["memo"][:-1] + "0"), "L1"),
    (lambda a: a["charon"]["entry"].__setitem__("entry_id", "0" * 64), "L5"),
    (lambda a: a["charon"].__setitem__("head_at_seq", "0" * 64), "L5"),
    (lambda a: a["sworn_receipt"].__setitem__("digest", "0" * 64), "L4"),
    (lambda a: a["prereg"].__setitem__("gates_sha256", "0" * 64), "L0"),
]


@pytest.mark.parametrize("edit,link", FIELD_EDITS)
def test_a_field_mutation_flips_exactly_its_link(minted, tmp_path, edit, link):
    p = _payload(minted)
    edit(p["artifacts"])
    rep = verify(_reseal(minted, p, tmp_path / "t.research.html"))
    assert _fails(rep) == {link}, [(l["id"], l["problems"]) for l in rep["links"]]


def test_a_mutated_slot_is_not_caught_offline_and_the_page_says_the_check_is_online(minted, tmp_path):
    """ATTACK F1 named this miss: nothing offline re-derives a slot. The payload digest shows the
    edit; the link itself holds, and its online reading is where the slot is checked."""
    p = _payload(minted)
    p["artifacts"]["anchors"][0]["slot"] += 1
    rep = verify(_reseal(minted, p, tmp_path / "t.research.html"))
    assert _fails(rep) == set()
    assert rep["payload"]["core_sha256_ok"] is False
    l1 = next(l for l in rep["links"] if l["id"] == "L1")
    assert l1["online"]["reading"] == ONLINE and any("slot" in r for r in l1["online"]["reads"])


def test_a_digest_mutation_inside_the_gates_block_flips_L0_and_the_dependent_L3(minted, tmp_path):
    p = _payload(minted)
    _flip_b64(p["artifacts"]["prereg"], b'"value": 0.15')
    rep = verify(_reseal(minted, p, tmp_path / "t.research.html"))
    assert "L0" in _fails(rep) and _fails(rep) <= {"L0", "L3"}


# --------------------------------------------------------- battery (b): semantic mutations, consistent

def _inputs_other_kind() -> dict:
    inp = _inputs()
    digest = sworn._sha256(PREREG.encode("utf-8"))
    inp["anchors"] = [{"n": 7, "kind": "sealed-canaries", "digest": digest,
                       "memo": clock.memo("sealed-canaries", digest), "tx": "6" * 87, "slot": 1,
                       "block_time": "2026-10-11T20:15:00Z"}]
    return inp


def test_semantic_L1_a_consistent_memo_of_another_kind_is_not_a_sealed_prereg(tmp_path):
    rep = verify(_write_consistent(tmp_path, _inputs_other_kind()))
    assert _fails(rep) == {"L1"} and rep["payload"]["core_sha256_ok"] and rep["payload"]["page_ok"]
    assert any("UNANCHORED" in p for p in next(l for l in rep["links"] if l["id"] == "L1")["problems"])


def test_semantic_L2_a_smoke_receipt_consistent_everywhere_fails_only_L2(tmp_path):
    rep = verify(_write_consistent(tmp_path, _inputs(receipt_bytes=_receipt_bytes({"gap": 0.3122, "dev": 0.0048},
                                                                                     smoke=True))))
    assert _fails(rep) == {"L2"}, [(l["id"], l["problems"]) for l in rep["links"]]
    assert rep["payload"]["core_sha256_ok"] and rep["payload"]["page_ok"]


def test_semantic_L3_a_verdict_from_another_row_of_the_table_fails_only_L3(tmp_path):
    receipt = _receipt_bytes({"gap": 0.3122, "dev": 0.0048}, verdict="EFFECT_ABSENT")
    rep = verify(_write_consistent(tmp_path, _inputs(receipt_bytes=receipt)))
    assert _fails(rep) == {"L3"}, [(l["id"], l["problems"]) for l in rep["links"]]
    l3 = next(l for l in rep["links"] if l["id"] == "L3")
    assert any("EFFECT_HOLDS" in p and "EFFECT_ABSENT" in p for p in l3["problems"])


def _result_gap_010(receipt_path: str, receipt: bytes) -> bytes:
    """The RESULT swears the gap the receipt below records (0.10), so its spans hold and only the
    re-score sees the crossed bar. Until review 2 this member swore 0.3122 against a 0.10 leaf
    and was an honest SWORN-FAILED at L4 too, which L4 then read HOLDS."""
    return _result_bytes(receipt_path).replace(b"The gap was 0.3122.", b"The gap was 0.10.")


def _inputs_metric_across_a_bar() -> dict:
    receipt = _receipt_bytes({"gap": 0.10, "dev": 0.0048}, verdict="EFFECT_HOLDS", gates={"G_A": True, "G_B": True})
    inp = _inputs(receipt_bytes=receipt, result_fn=_result_gap_010)
    assert inp["sworn_receipt"]["document_verdict"] == "SWORN-HELD"
    return inp


def test_semantic_L3_a_metric_that_crosses_a_bar_while_the_gates_map_stays_fails_only_L3(tmp_path):
    rep = verify(_write_consistent(tmp_path, _inputs_metric_across_a_bar()))
    assert _fails(rep) == {"L3"}, [(l["id"], l["problems"]) for l in rep["links"]]


def _redigest(rec: dict) -> dict:
    body = {k: v for k, v in rec.items() if k not in sworn._RECEIPT_OUTSIDE_DIGEST}
    rec["digest"] = sworn._sha256(sworn._jcs(body).encode("utf-8"))
    return rec


def _inputs_verdict_edited(**kw) -> dict:
    """The sworn receipt's document_verdict edited and re-digested; the charon line agrees."""
    inp = _inputs(**kw)
    rec = _redigest(dict(inp["sworn_receipt"], document_verdict="SWORN-FAILED"))
    inp["sworn_receipt"] = rec
    inp["charon"]["entry"] = _charon_entry(inp["sidecar"], rec, inp["_core"], "papers/x/RESULT_x.md", HEADER,
                                           verdict="SWORN-FAILED", verdict_class="FAILED")
    return inp


def _inputs_repointed(old: bytes, new: bytes, **kw) -> dict:
    """The forger types a value the receipt does not hold: the RESULT is rewritten, given its own
    sidecar, and the carried sworn receipt keeps its spans and verdict and is re-pointed at the new
    document (both digests) and re-digested; the charon line names the new document."""
    inp = _inputs(**kw)
    result = inp["result"]["bytes"]
    assert old in result
    result = result.replace(old, new)
    side = sworn.to_sidecar(result, "RESULT_x.md", COMMIT)
    rec = dict(inp["sworn_receipt"])
    rec["document"] = dict(rec["document"], inline_sha256=sworn._sha256(result),
                           canonical_sha256=sworn._sha256(side["text"].encode("utf-8")))
    rec = _redigest(rec)
    inp.update(sidecar=side, sworn_receipt=rec, result={"path": "papers/x/RESULT_x.md", "bytes": result})
    inp["charon"]["entry"] = _charon_entry(side, rec, inp["_core"], "papers/x/RESULT_x.md", HEADER,
                                           subject={"name": "RESULT_x.md", "path": "papers/x/RESULT_x.md",
                                                    "sha256": sworn._sha256(result)})
    return inp


def _inputs_snapshot_swapped() -> dict:
    """A snapshot entry that is another receipt, carried as the run's receipt too."""
    inp = _inputs()
    other = _receipt_bytes({"gap": 0.20, "dev": 0.0048})
    inp["snapshot"]["entries"]["papers/x/x_result.json"] = {"mode": "100644", "size": len(other),
                                                            "sha256": sworn._sha256(other), "bytes": other}
    inp["receipts"][0]["bytes"] = other               # L2 agrees with the snapshot; L3 scores the same row
    return inp


def _result_failed_honestly(receipt_path: str, receipt: bytes) -> bytes:
    """The RESULT types 0.9122 where the leaf holds 0.3122, and its own sworn receipt says so
    honestly: SWORN-FAILED, re-derivable, the charon line agreeing."""
    return _result_bytes(receipt_path).replace(b"The gap was 0.3122.", b"The gap was 0.9122.")


def _result_rn_unresolved(receipt_path: str, receipt: bytes) -> bytes:
    """A second span names r1 and no manifest carries it: UNRESOLVED (manifest_absent), which the
    sworn document rule still reads SWORN-HELD, and no leaf is read for 0.9999."""
    return (b"# RESULT\n\n" + f'<sworn r="path:{receipt_path}#/metrics/gap" k="numeric">The gap was '
            f'0.3122.</sworn>\n'.encode("utf-8") + b'<sworn r="r1" k="numeric">The dev was 0.9999.</sworn>\n')


def _result_prereg_unresolved(receipt_path: str, receipt: bytes) -> bytes:
    """A prereg: span whose digest the snapshot does not hold: UNRESOLVED (prereg_not_in_tree)."""
    return (b"# RESULT\n\n" + f'<sworn r="path:{receipt_path}#/metrics/gap" k="numeric">The gap was '
            f'0.3122.</sworn>\n<sworn r="prereg:{"e" * 64}" k="quote">The bar was frozen at '
            f'`0.99`.</sworn>\n'.encode("utf-8"))


def _result_unsworn(receipt_path: str, receipt: bytes) -> bytes:
    return b"# RESULT\n\nThe gap was 0.9122. Nothing here is sworn.\n"


NOT_ALL_HELD = [
    ("an honest SWORN-FAILED receipt", _result_failed_honestly, "SWORN-FAILED", "FAILED 1"),
    ("an UNRESOLVED rN span", _result_rn_unresolved, "SWORN-HELD", "UNRESOLVED 1"),
    ("an UNRESOLVED prereg span", _result_prereg_unresolved, "SWORN-HELD", "UNRESOLVED 1"),
]


@pytest.mark.parametrize("name,result_fn,carried,count", NOT_ALL_HELD, ids=[x[0] for x in NOT_ALL_HELD])
def test_semantic_L4_a_RESULT_whose_spans_do_not_all_hold_fails_only_L4_and_build_refuses_it(
        tmp_path, name, result_fn, carried, count):
    """Review 2's major: the carried sworn receipt is HONEST here (its core re-derives, its tally
    is its spans'), and still a span did not hold against a leaf, or was read against none. L4
    proves every number inside a sworn span equals the leaf it names, so it reads FAILS."""
    inp = _inputs(result_fn=result_fn)
    assert inp["sworn_receipt"]["document_verdict"] == carried
    rep = verify(_write_consistent(tmp_path, inp))
    assert _fails(rep) == {"L4"}, [(l["id"], l["problems"]) for l in rep["links"]]
    assert rep["payload"]["core_sha256_ok"] and rep["payload"]["page_ok"]
    l4 = next(l for l in rep["links"] if l["id"] == "L4")
    assert [p for p in l4["problems"] if "sworn spans read HELD" in p and count in p], l4["problems"]
    assert len(l4["problems"]) == 1, l4["problems"]
    assert any(c.startswith("the sworn core re-derives over the snapshot") for c in l4["checked"])
    inp.pop("_core")
    with pytest.raises(SystemExit) as e:
        build(inp, tmp_path / "no.research.html", created=TS)
    msg = str(e.value.code)
    assert "REFUSED inconsistent_chain" in msg and "L4" in msg and "sworn spans read HELD" in msg
    assert not (tmp_path / "no.research.html").exists()


def test_semantic_L4_a_RESULT_that_swears_nothing_fails_L4_by_name(tmp_path):
    inp = _inputs(result_fn=_result_unsworn)
    assert inp["sworn_receipt"]["document_verdict"] == "UNSWORN"
    rep = verify(_write_consistent(tmp_path, inp))
    l4 = next(l for l in rep["links"] if l["id"] == "L4")
    assert any("the RESULT swears nothing (UNSWORN)" in p for p in l4["problems"]), l4["problems"]
    # the charon line's receipt set is empty too: L5 names it from its side
    assert _fails(rep) == {"L4", "L5"}, [(l["id"], l["problems"]) for l in rep["links"]]


def test_semantic_L4_a_sworn_receipt_whose_verdict_was_edited_fails_only_L4(tmp_path):
    rep = verify(_write_consistent(tmp_path, _inputs_verdict_edited()))
    assert _fails(rep) == {"L4"}, [(l["id"], l["problems"]) for l in rep["links"]]
    l4 = next(l for l in rep["links"] if l["id"] == "L4")
    assert any("its own spans tally SWORN-HELD" in p for p in l4["problems"])


def test_semantic_L4_a_span_value_the_receipt_does_not_hold_fails_only_L4(tmp_path):
    """ATTACK F1's span-value member: the RESULT prints 0.9122 where the receipt leaf is 0.3122,
    and the carried sworn receipt still says HELD, re-pointed and re-digested. Every digest agrees;
    only re-deriving the span against the snapshot leaf sees it."""
    inp = _inputs_repointed(b"The gap was 0.3122.", b"The gap was 0.9122.")
    rep = verify(_write_consistent(tmp_path, inp))
    assert _fails(rep) == {"L4"}, [(l["id"], l["problems"]) for l in rep["links"]]
    assert rep["payload"]["core_sha256_ok"] and rep["payload"]["page_ok"]
    l4 = next(l for l in rep["links"] if l["id"] == "L4")
    assert any("live SWORN-FAILED" in p for p in l4["problems"])


def test_semantic_L4_a_snapshot_entry_that_is_another_receipt_fails_L4_and_the_line_that_named_it(tmp_path):
    rep = verify(_write_consistent(tmp_path, _inputs_snapshot_swapped()))
    # L5 sees it too, honestly: the carried line's receipt set names the receipt the spans read
    assert _fails(rep) == {"L4", "L5"}, [(l["id"], l["problems"]) for l in rep["links"]]


def _tracked_json_under_papers() -> list:
    """Every ``*.json`` under ``papers/`` that git tracks. A checkout may hold a runner's output
    untracked beside the PREREG that seals it, and the suite never opens it; with no git (an
    unpacked source tree), every one on disk except that runner's output name."""
    papers = ROOT / "papers"
    if not papers.is_dir():
        return []
    if shutil.which("git") is not None:
        p = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z", "--", "papers"], capture_output=True)
        if p.returncode == 0:
            names = [n for n in p.stdout.decode("utf-8", "replace").split("\0") if n.endswith(".json")]
            return sorted(f for f in (ROOT / n for n in names) if f.is_file())
    return sorted(f for f in papers.rglob("*.json") if f.name != "pmnist_untied_result.json")


def test_the_tally_rule_reads_the_receipt_alone_and_holds_on_every_committed_sworn_receipt():
    inp = _inputs()
    assert rc.receipt_tally_problem(inp["sworn_receipt"]) is None
    edited = dict(inp["sworn_receipt"], document_verdict="SWORN-FAILED")
    assert "its own spans tally SWORN-HELD" in rc.receipt_tally_problem(edited)
    recount = dict(inp["sworn_receipt"], counts=dict(inp["sworn_receipt"]["counts"], HELD=1, FAILED=1))
    assert "not its spans' tally" in rc.receipt_tally_problem(recount)
    # a count is an integer as written, and the verdict is the spans' tally
    floated = dict(inp["sworn_receipt"], counts=dict(inp["sworn_receipt"]["counts"], HELD=2.0))
    assert rc.receipt_tally_problem(floated) == ("its counts hold a value that is not an integer as "
                                                 "written (HELD)")
    assert "its sworn_total is not an integer as written" == rc.receipt_tally_problem(
        dict(inp["sworn_receipt"], sworn_total=True))
    # every sworn receipt committed under papers/, chosen by its schema field and not its file name
    # (the measurement dry run names its receipts *.receipt.json); tracked files only
    committed = []
    marker = b"styxx.sworn.verdict-receipt/v"
    for p in _tracked_json_under_papers():
        data = p.read_bytes()
        if marker not in data:
            continue
        try:
            obj = json.loads(data.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            continue
        if isinstance(obj, dict) and obj.get("schema") in sworn.RECEIPT_SCHEMAS:
            committed.append((p, obj))
    if not committed:
        pytest.skip("no committed sworn receipt in this tree")
    bad =[(p.name, rc.receipt_tally_problem(obj)) for p, obj in committed]
    assert [b for b in bad if b[1]] == []


def _inputs_line_about_another_document() -> dict:
    inp = _inputs()
    inp["charon"]["entry"] = _charon_entry(inp["sidecar"], inp["sworn_receipt"], inp["_core"], "papers/x/RESULT_x.md",
                                           HEADER, subject={"name": "RESULT_x.md", "path": "papers/x/RESULT_x.md",
                                                            "sha256": "0" * 64})
    return inp


def _inputs_receipt_set_omits() -> dict:
    inp = _inputs()
    inp["charon"]["entry"] = _charon_entry(inp["sidecar"], inp["sworn_receipt"], inp["_core"], "papers/x/RESULT_x.md",
                                           HEADER, receipts={"n": 0, "sha256": [], "cited": None, "vacuous": False})
    return inp


def _two_fences() -> str:
    return PREREG + "\n```gates\n{}\n```\n"


def _write_two_fences(tmp: Path, name: str = "two.research.html") -> Path:
    """Built around compose's refusal: the PREREG carries a second gates fence, re-sealed."""
    two = _two_fences().encode("utf-8")
    digest = sworn._sha256(two)
    payload = compose({k: v for k, v in _inputs().items() if k != "_core"}, created=TS)
    payload["artifacts"]["prereg"]["b64"] = base64.b64encode(two).decode("ascii")
    payload["artifacts"]["prereg"]["sha256"] = digest
    payload["artifacts"]["anchors"][0].update(digest=digest, memo=clock.memo("sealed-prereg", digest))
    payload["core_sha256"] = rc._core_sha256(payload)
    out = tmp / name
    out.write_bytes(render_html(payload).encode("utf-8"))
    return out


def _edit_label(p: dict) -> None:
    p["label"] = p["label"].replace("passes every offline link", "is caught")


def test_semantic_L5_a_line_about_another_document_fails_only_L5(tmp_path):
    rep = verify(_write_consistent(tmp_path, _inputs_line_about_another_document()))
    assert _fails(rep) == {"L5"}, [(l["id"], l["problems"]) for l in rep["links"]]


def test_semantic_L5_a_receipt_set_that_omits_the_run_receipt_fails_only_L5(tmp_path):
    rep = verify(_write_consistent(tmp_path, _inputs_receipt_set_omits()))
    assert _fails(rep) == {"L5"}


def test_semantic_L0_a_second_gates_fence_fails_L0_and_the_dependent_L3(tmp_path):
    inp = _inputs()
    two = _two_fences().encode("utf-8")
    inp["prereg"]["bytes"] = two
    digest = sworn._sha256(two)
    inp["anchors"][0].update(digest=digest, memo=clock.memo("sealed-prereg", digest))
    with pytest.raises(SystemExit, match="prereg_gates"):
        compose({k: v for k, v in inp.items() if k != "_core"}, created=TS)
    # built around the refusal, the verifier names both: L0 (two blocks) and L3, which has no table
    rep = verify(_write_two_fences(tmp_path))
    assert _fails(rep) == {"L0", "L3"}


def test_semantic_L6_an_edited_label_fails_only_L6(tmp_path):
    rep = verify(_write_consistent(tmp_path, _inputs(), mutate=_edit_label))
    assert _fails(rep) == {"L6"}


# every member of the semantic battery, by name: (what is made false, the links it flips, the capsule)
SEMANTIC = [
    ("L0 a second gates fence", {"L0", "L3"}, lambda tmp, n: _write_two_fences(tmp, n)),
    ("L1 a memo of another kind", {"L1"}, lambda tmp, n: _write_consistent(tmp, _inputs_other_kind(), name=n)),
    ("L2 a smoke receipt", {"L2"}, lambda tmp, n: _write_consistent(
        tmp, _inputs(receipt_bytes=_receipt_bytes({"gap": 0.3122, "dev": 0.0048}, smoke=True)), name=n)),
    ("L3 another row's verdict", {"L3"}, lambda tmp, n: _write_consistent(
        tmp, _inputs(receipt_bytes=_receipt_bytes({"gap": 0.3122, "dev": 0.0048}, verdict="EFFECT_ABSENT")), name=n)),
    ("L3 a metric across a bar", {"L3"}, lambda tmp, n: _write_consistent(tmp, _inputs_metric_across_a_bar(), name=n)),
    ("L4 an edited sworn verdict", {"L4"}, lambda tmp, n: _write_consistent(tmp, _inputs_verdict_edited(), name=n)),
    ("L4 a span value the receipt does not hold", {"L4"}, lambda tmp, n: _write_consistent(
        tmp, _inputs_repointed(b"The gap was 0.3122.", b"The gap was 0.9122."), name=n)),
    ("L4 a snapshot entry that is another receipt", {"L4", "L5"},
     lambda tmp, n: _write_consistent(tmp, _inputs_snapshot_swapped(), name=n)),
    ("L4 an honest SWORN-FAILED receipt", {"L4"},
     lambda tmp, n: _write_consistent(tmp, _inputs(result_fn=_result_failed_honestly), name=n)),
    ("L4 an UNRESOLVED rN span", {"L4"},
     lambda tmp, n: _write_consistent(tmp, _inputs(result_fn=_result_rn_unresolved), name=n)),
    ("L4 an UNRESOLVED prereg span", {"L4"},
     lambda tmp, n: _write_consistent(tmp, _inputs(result_fn=_result_prereg_unresolved), name=n)),
    ("L4 a RESULT that swears nothing", {"L4", "L5"},
     lambda tmp, n: _write_consistent(tmp, _inputs(result_fn=_result_unsworn), name=n)),
    ("L5 a line about another document", {"L5"},
     lambda tmp, n: _write_consistent(tmp, _inputs_line_about_another_document(), name=n)),
    ("L5 a receipt set without the run's receipt", {"L5"},
     lambda tmp, n: _write_consistent(tmp, _inputs_receipt_set_omits(), name=n)),
    ("L6 an edited label", {"L6"}, lambda tmp, n: _write_consistent(tmp, _inputs(), mutate=_edit_label, name=n)),
]


# ------------------------------------------------------------------- controls that must not fire

def test_controls_created_timestamp_key_order_and_charon_note_fire_no_link(minted, tmp_path):
    p = _payload(minted)
    p["created"] = "2030-01-01T00:00:00Z"
    rep = verify(_reseal(minted, p, tmp_path / "c1.research.html"))
    assert _fails(rep) == set() and rep["payload"]["core_sha256_ok"] is True
    # key order: JCS sorts, so a permuted payload is the same payload
    p = _payload(minted)
    shuffled = {k: p[k] for k in reversed(list(p))}
    rep = verify(_reseal(minted, shuffled, tmp_path / "c2.research.html", sort_keys=False))
    assert _fails(rep) == set() and rep["payload"]["core_sha256_ok"] is True and rep["ok"]
    # the charon timestamp and note sit outside the line's digest: the line still re-derives
    p = _payload(minted)
    p["artifacts"]["charon"]["entry"]["timestamp"] = "2030-01-01T00:00:00Z"
    p["artifacts"]["charon"]["entry"]["note"] = "re-ferried"
    rep = verify(_reseal(minted, p, tmp_path / "c3.research.html"))
    assert _fails(rep) == set()


# -------------------------------------------------------------------------- refusals at build

def test_build_refuses_an_inconsistent_chain_by_name(tmp_path):
    inp = _inputs(receipt_bytes=_receipt_bytes({"gap": 0.3122, "dev": 0.0048}, verdict="EFFECT_ABSENT"))
    inp.pop("_core")
    with pytest.raises(SystemExit) as e:
        build(inp, tmp_path / "no.research.html", created=TS)
    assert "REFUSED inconsistent_chain" in str(e.value.code) and "L3" in str(e.value.code)
    assert not (tmp_path / "no.research.html").exists()


def test_build_refuses_a_payload_with_a_token_looking_string(tmp_path):
    inp = _inputs()
    inp.pop("_core")
    inp["run_note"] = {"path": "papers/x/x_RUN_NOTE.md",
                       "bytes": RUN_NOTE + b"\nexport GH=ghp_" + b"A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8" + b"\n"}
    with pytest.raises(SystemExit) as e:
        build(inp, tmp_path / "no.research.html", created=TS)
    assert "REFUSED disclosure" in str(e.value.code) and "token" in str(e.value.code)
    assert not (tmp_path / "no.research.html").exists()


@pytest.mark.parametrize("value,what", [
    ("C:/Users/someone/clawd/wt", "drive-letter path"),
    ("/home/someone/repo", "home directory path"),
    ("CPython 3.12.10", "interpreter version string"),
    ("worktree _frequency_2026_10_11", "worktree or workspace name"),
    ("-----BEGIN RSA PRIVATE KEY-----", "private key"),
])
def test_build_refuses_a_disclosure_in_a_stated_fact(tmp_path, value, what):
    inp = _inputs()
    inp.pop("_core")
    inp["stated"]["note"] = value
    with pytest.raises(SystemExit) as e:
        build(inp, tmp_path / "no.research.html", created=TS)
    assert "REFUSED disclosure" in str(e.value.code) and what in str(e.value.code)


def test_build_refuses_a_path_that_is_not_repository_relative_posix(tmp_path):
    inp = _inputs()
    inp.pop("_core")
    inp["runners"] = [{"path": "C:/x/run_x.py", "bytes": RUNNER}]
    with pytest.raises(SystemExit, match="REFUSED runner"):
        build(inp, tmp_path / "no.research.html", created=TS)
    inp["runners"] = [{"path": "papers/../run_x.py", "bytes": RUNNER}]
    with pytest.raises(SystemExit, match="REFUSED runner"):
        build(inp, tmp_path / "no.research.html", created=TS)


def test_build_refuses_an_oversize_page_and_verify_fails_one(tmp_path, minted):
    inp = _inputs()
    inp.pop("_core")
    inp["runners"] = [{"path": "papers/x/run_x.py", "bytes": RUNNER + b"#" * (SIZE_CAP + 10)}]
    with pytest.raises(SystemExit, match="REFUSED oversize"):
        build(inp, tmp_path / "big.research.html", created=TS)
    assert not (tmp_path / "big.research.html").exists()
    # a page padded past the cap after the mint fails on size alone
    big = tmp_path / "padded.research.html"
    big.write_bytes(minted.read_bytes() + b"<!--" + b" " * (SIZE_CAP + 1) + b"-->")
    rep = verify(big)
    assert rep["payload"]["size_ok"] is False and rep["ok"] is False
    assert "OVER" in "\n".join(datasheet(rep))


def test_build_without_an_anchor_needs_allow_unanchored_and_then_L1_reads_fails(tmp_path):
    inp = _inputs()
    inp.pop("_core")
    inp["anchors"] = []
    with pytest.raises(SystemExit, match="UNANCHORED"):
        build(inp, tmp_path / "u.research.html", created=TS)
    cap = build(inp, tmp_path / "u.research.html", created=TS, allow_unanchored=True)
    rep = verify(cap)
    assert _fails(rep) == {"L1"} and rep["ok"] is False


# ------------------------------------------------------------------------ O9 and the house words

def _without_carried_runtime(p: dict) -> dict:
    """The payload less the carried charon line's ``verifier.runtime``: carried history, which names the
    interpreter its line was derived under (charon's verifier scheme v1) and is carried as is."""
    out = json.loads(json.dumps(p))
    ver = ((out["artifacts"].get("charon") or {}).get("entry") or {}).get("verifier")
    if isinstance(ver, dict):
        ver.pop("runtime", None)
    return out


def test_the_payload_carries_no_interpreter_path_host_or_secret(minted):
    p = _payload(minted)
    assert set(p["verifier"]) == {"styxx_version", "modules", "digest"}
    assert [m[0] for m in p["verifier"]["modules"]][0] == "styxx.research_capsule"
    text = json.dumps(p)
    import platform
    import sys
    # O9 as the operator recorded it: no sys.version anywhere in the payload, carried history included
    assert sys.version not in text and "sys.version" not in text
    # and the interpreter's version nowhere but where carried history names it (the next test)
    own = json.dumps(_without_carried_runtime(p))
    assert sys.version.split()[0] not in own or sys.version.split()[0] == p["verifier"]["styxx_version"]
    assert len(platform.node()) < 4 or platform.node() not in text
    assert "C:\\\\" not in text and "C:/" not in text and "/home/" not in text and "/Users/" not in text
    assert rc.disclosure_hits(p, minted.read_text(encoding="utf-8"), host=platform.node()) == []


def test_a_carried_charon_line_names_its_runtime_as_history_and_dropping_it_fails_L5(minted, tmp_path):
    import platform
    import sys
    import unicodedata
    p = _payload(minted)
    v = p["artifacts"]["charon"]["entry"]["verifier"]
    assert v["scheme"] == charon.VERIFIER_SCHEME
    assert v["runtime"] == {"implementation": sys.implementation.name, "version": platform.python_version(),
                            "unicode": unicodedata.unidata_version}
    assert sys.version not in json.dumps(v) and "runtime" not in p["verifier"]
    # the runtime sits inside the core the line's entry_id digests: a consistent capsule that dropped it
    # fails L5 and L5 alone
    def drop(payload):
        payload["artifacts"]["charon"]["entry"]["verifier"].pop("runtime")
    assert _fails(verify(_write_consistent(tmp_path, _inputs(), drop))) == {"L5"}
    # the DISCLOSURE says so, and no longer says the payload carries no interpreter version
    doc = " ".join(rc.__doc__.split())
    assert "``verifier.runtime``" in doc and "carried as is" in doc
    assert "the payload carries no interpreter version" not in doc


def test_the_capsules_own_verifier_names_module_shas_over_lf_bytes(minted):
    import hashlib
    import importlib
    p = _payload(minted)
    assert [m for m, _ in p["verifier"]["modules"]] == list(rc._MODULES)
    for name, sha in p["verifier"]["modules"]:
        raw = Path(importlib.import_module(name).__file__).read_bytes()
        assert sha == hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest(), name


def test_no_house_word_in_the_module_the_tests_the_page_or_the_datasheet(minted):
    # the list is held reversed in the module, as the lab holds its ordinal, so neither file prints it
    rx = rc._HOUSE_RX
    for f in (ROOT / "styxx" / "research_capsule.py", Path(__file__)):
        assert rx.search(f.read_text(encoding="utf-8")) is None, f
    html = minted.read_text(encoding="utf-8")
    # what the page prints; the inlined verifier is the package's code (sworn_verify.js carries the
    # ordinal as a variable name) and is sealed by digest, not scanned for words
    assert rx.search(rc._without_verifier(html)) is None
    assert rx.search(rc._JS_RESEARCH) is None
    assert rx.search("\n".join(datasheet(verify(minted)))) is None
    assert len(rc._HOUSE_WORDS) == 8
    for word in rc._HOUSE_WORDS:
        assert rx.search("a " + word + " claim")                    # the scanner sees each
    # and never a stronger summary than the settled one
    page = _page_outside_payload(minted).lower()
    for banned in ("verified", "immutable", "proof of", "sealed before the run"):
        assert banned not in page


def test_the_eighth_house_word_is_the_ordinal_the_lab_holds():
    assert rc._HOUSE_WORDS[7] == "tsrif"[::-1]
    lab = ROOT / "tests" / "test_unpriced_priority_sentences_withdrawn.py"
    if not lab.exists():
        pytest.skip("the lab's ordinal test is absent from this tree")
    held = re.search(r'_ORDINAL = "(\w+)"\[::-1\]', lab.read_text(encoding="utf-8"))
    assert held and held.group(1)[::-1] == rc._HOUSE_WORDS[7]


_ORD = "tsrif"[::-1]


@pytest.mark.parametrize("where", ["stated", "run_note"])
def test_build_refuses_the_ordinal_in_what_the_lab_writes_and_never_echoes_it(tmp_path, where):
    inp = _inputs()
    inp.pop("_core")
    if where == "stated":
        inp["stated"]["note"] = f"the {_ORD} seed ran"
    else:
        inp["run_note"] = {"path": "papers/x/x_RUN_NOTE.md", "bytes": RUN_NOTE + f"the {_ORD} seed ran\n".encode()}
    with pytest.raises(SystemExit) as e:
        build(inp, tmp_path / "no.research.html", created=TS)
    msg = str(e.value.code)
    assert "REFUSED disclosure" in msg and "entry 8 of 8" in msg
    assert rc._HOUSE_RX.search(msg) is None


def test_carried_history_is_scanned_for_leaks_and_not_for_the_house_words(tmp_path):
    """A frozen PREREG and a runner are history: one that prints the ordinal in a sentence the
    lab wrote before the list (the pMNIST PREREG does, at its line 55) is carried as is; a leak in
    history is still refused."""
    inp = _inputs()
    inp.pop("_core")
    prereg = (PREREG + f"\nThe anchor digits were {_ORD} typed from the prose.\n").encode("utf-8")
    digest = sworn._sha256(prereg)
    inp["prereg"]["bytes"] = prereg
    inp["anchors"][0].update(digest=digest, memo=clock.memo("sealed-prereg", digest))
    inp["runners"] = [{"path": "papers/x/run_x.py", "bytes": RUNNER + f"# the {_ORD} seed\n".encode()}]
    cap = build(inp, tmp_path / "h.research.html", created=TS)
    assert verify(cap)["ok"]
    inp["runners"] = [{"path": "papers/x/run_x.py", "bytes": RUNNER + b"# C:\\Users\\someone\\run\n"}]
    with pytest.raises(SystemExit, match="drive-letter path"):
        build(inp, tmp_path / "no.research.html", created=TS)


def test_the_disclosure_scan_reads_decoded_strings_not_json_escapes():
    honest = {"artifacts": {"sidecar": {"text": "Let the gap be g:\nthen the bar"}}}
    assert rc.disclosure_hits(honest) == []
    assert any("drive-letter" in h for h in rc.disclosure_hits({"stated": {"note": "C:\\Users\\someone"}}))
    # a carried JSON artifact is read as its decoded strings too
    ok = json.dumps({"note": "g:\nthen"}).encode("utf-8")
    assert b"g:\\n" in ok
    leak = json.dumps({"note": "C:\\Users\\someone"}).encode("utf-8")

    def art(data):
        return {"artifacts": {"receipts": [{"name": "r.json", "b64": base64.b64encode(data).decode()}]}}
    assert rc.disclosure_hits(art(ok)) == []
    assert any(h.startswith("artifacts/receipts[0]/r.json: a drive-letter path") for h in rc.disclosure_hits(art(leak)))


def test_the_disclosure_scan_reads_every_value_of_a_repeated_key(minted):
    """json.loads keeps a repeated key's last value; the scan reads every pair, in a carried JSON
    artifact and in the page's own payload, so a leak or a word behind a duplicate is not hidden."""
    bs = "\\"
    leak = json.dumps("C:" + bs + "Users" + bs + "someone")
    dup = ('{"note": ' + leak + ', "note": "ok"}').encode("utf-8")
    assert json.loads(dup) == {"note": "ok"}
    art = {"artifacts": {"receipts": [{"name": "r.json", "b64": base64.b64encode(dup).decode()}]}}
    assert any(h.startswith("artifacts/receipts[0]/r.json: a drive-letter path") for h in rc.disclosure_hits(art))
    # a repeated name or b64 key: every body is decoded and scanned
    two = {"artifacts": {"runners": [{"name": "a.py", "b64": base64.b64encode(b"x").decode()}]}}
    assert json.dumps(two).endswith('"}]}}')
    page = (rc._BEGIN + json.dumps(two)[:-4] + ', "b64": "' + base64.b64encode(b"/home/someone/x").decode()
            + '"}]}}' + rc._END)
    assert any(h.startswith("artifacts/runners[0]/a.py: a home directory path") for h in rc.disclosure_hits(two, page))
    # the payload read from the page: a duplicate inside stated hides neither a path nor a word
    html = minted.read_text(encoding="utf-8")
    m = PAYLOAD_RE.search(html)
    body = m.group(1)
    assert body.count('"stated": {') == 1
    forged = body.replace('"stated": {', '"stated": {"note": ' + leak + ', "note": "ok", "word": "the '
                          + _ORD + ' seed", "word": "a seed", ', 1)
    payload = json.loads(forged)
    assert payload["stated"]["note"] == "ok" and payload["stated"]["word"] == "a seed"
    page = html[:m.start(1)] + forged + html[m.end(1):]
    hits = rc.disclosure_hits(payload, page)
    assert any(h.startswith("the payload: a drive-letter path") for h in hits), hits
    assert any(h.startswith("the payload: a word the lab never prints (entry 8 of 8") for h in hits), hits
    assert rc.disclosure_hits(payload) == []                      # the parsed payload alone hides both


# -------------------------------------------------------------------------- the pure re-score

def test_score_frozen_matches_the_committed_experiment_scorer():
    """Byte-identical verdicts to ``styxx.protocol.Experiment.score`` on a committed PREREG whose
    receipt is in the tree."""
    from styxx.protocol import Experiment, _select_gates_block
    prereg = ROOT / "papers" / "rhythm-rescue" / "PREREG_untied_magnitudes_2026_09_02.md"
    receipt = json.loads((ROOT / "papers" / "rhythm-rescue" / "untied_control_result.json").read_text(encoding="utf-8"))
    if not prereg.exists():
        pytest.skip("committed PREREG absent")
    try:
        exp = Experiment(prereg, repo_root=ROOT)
    except Exception as e:                                   # noqa: BLE001 - a shallow clone has no history
        pytest.skip(f"Experiment unavailable here: {e}")
    gates_text = _select_gates_block(prereg.read_text(encoding="utf-8"))
    mine = score_frozen(gates_text, receipt["metrics"])
    theirs = exp.score(receipt["metrics"])
    assert mine["verdict"] == theirs.verdict == receipt["verdict"]
    assert mine["gates"] == theirs.gates == receipt["gates"]
    assert mine["gates_sha256"] == theirs.gates_sha256
    assert mine["total"] is True


def test_score_frozen_refuses_duplicate_keys_and_a_partial_table():
    from styxx.protocol import GateSpecError
    dup = '{"gates": {"G": {"metric": "m", "op": ">=", "value": 1}, "G": {"metric": "m", "op": "<", "value": 1}},' \
          ' "outcomes": [{"when": {"G": true}, "verdict": "A"}, {"when": {"G": false}, "verdict": "B"}], "smoke_verdict": "S"}'
    with pytest.raises(GateSpecError, match="duplicate key"):
        score_frozen(dup, {"m": 2})
    partial = '{"gates": {"G": {"metric": "m", "op": ">=", "value": 1}}, "outcomes": [{"when": {"G": true}, "verdict": "A"}],' \
              ' "smoke_verdict": "S"}'
    assert score_frozen(partial, {"m": 2})["total"] is False
    with pytest.raises(GateSpecError, match="not total"):
        score_frozen(partial, {"m": 0})


def test_check_anchor_offline_is_the_pure_half_of_the_clock():
    d = sworn._sha256(b"x")
    good = {"n": 1, "kind": "sealed-prereg", "digest": d, "memo": clock.memo("sealed-prereg", d), "tx": "t",
            "slot": 5, "block_time": "2026-10-11T20:15:00Z"}
    assert rc.check_anchor_offline(good)["status"] == "SHAPE_OK"
    assert rc.check_anchor_offline(dict(good, memo="styxx sealed-prereg " + "0" * 64))["status"] == "MALFORMED_LINE"
    assert rc.check_anchor_offline(dict(good, kind="tweet"))["status"] == "MALFORMED_LINE"
    assert rc.check_anchor_offline(dict(good, slot=-1))["status"] == "MALFORMED_LINE"
    assert rc.check_anchor_offline(dict(good, block_time="yesterday"))["status"] == "MALFORMED_LINE"
    assert rc.check_anchor_offline(dict(good, digest=d.upper()))["status"] == "MALFORMED_LINE"


# ----------------------------------------------------------------------------- the network reading

MEMO_PROGRAM = "MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr"
TOKEN_PROGRAM = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"


def _fake_chain(memo_text, *, slot=371_000_000, block_time=1_791_000_000, err=None, rpc_fails=False):
    """A fetch shaped like the chain's jsonParsed answers, as tests/test_clock.py builds one."""
    tx = {"slot": slot, "blockTime": block_time,
          "transaction": {"message": {"accountKeys": [{"pubkey": clock.CREATOR, "signer": True}],
                                      "instructions": [
                                          {"program": "spl-token", "programId": TOKEN_PROGRAM,
                                           "parsed": {"type": "transferChecked",
                                                      "info": {"mint": clock.MINT, "tokenAmount": {"amount": "1"}}}},
                                          {"program": "spl-memo", "programId": MEMO_PROGRAM, "parsed": memo_text}]}},
          "meta": {"err": err, "innerInstructions": []}}

    def fetch(method, params, rpcs=None):
        if rpc_fails:
            raise RuntimeError("rpc failed: 429")
        if method == "getTransaction":
            return tx
        if method == "getBlock":
            return {"blockhash": clock._b58encode(bytes(range(1, 33)))}
        if method == "getSignaturesForAddress":
            if params[1].get("before"):
                return []
            return [{"signature": "5" * 87, "slot": slot, "err": None, "memo": f"[{len(memo_text)}] {memo_text}"},
                    {"signature": "7" * 87, "slot": slot - 10, "err": None,
                     "memo": "[85] styxx sealed-prereg " + "e" * 64}]
        raise AssertionError(method)
    fetch.last_endpoint = "fixture"
    return fetch


def test_network_readings_confirmed_contradicted_unavailable(minted):
    memo = clock.memo("sealed-prereg", sworn._sha256(PREREG.encode("utf-8")))
    # the recorded block_time must be the chain's for ANCHORED; the fixture's blockTime is 2026-10-11T20:00:00Z
    p = _payload(minted)
    p["artifacts"]["anchors"][0]["block_time"] = clock._iso(1_791_000_000)
    p["core_sha256"] = rc._core_sha256(p)
    cap = minted
    cap.write_bytes(render_html(p).encode("utf-8"))        # the page's L1 sentence names the block time
    rep = verify(cap, network=True, fetch=_fake_chain(memo))
    row = rep["network"]["anchors"][0]
    assert row["status"] == "ANCHORED" and row["reading"] == rc.CONFIRMED and row["beacon"] == bytes(range(1, 33)).hex()
    assert rep["network"]["sealed_prereg_memos"]["listed_through_slot"] == 2
    assert rep["ok"]
    out = "\n".join(datasheet(rep))
    assert "CONFIRMED-ON-CHAIN (clock says ANCHORED" in out and "sealed-prereg memos listed through the anchor's slot: 2" in out
    rep = verify(cap, network=True, fetch=_fake_chain("styxx sealed-prereg " + "0" * 64))
    assert rep["network"]["anchors"][0]["status"] == "MEMO_MISMATCH"
    assert rep["network"]["anchors"][0]["reading"] == rc.CONTRADICTED and rep["ok"] is False
    rep = verify(cap, network=True, fetch=_fake_chain(memo, rpc_fails=True))
    assert rep["network"]["anchors"][0]["status"] == "RPC_ERROR"
    assert rep["network"]["anchors"][0]["reading"] == rc.UNAVAILABLE and rep["ok"] is True


# ---------------------------------------------------------- the CLI over a repository, and --repo

def _git(repo, *args):
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@t")
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, encoding="utf-8",
                          errors="replace", env=env, check=True).stdout.strip()


@pytest.fixture
def chain_repo(tmp_path):
    """A repository holding the whole chain in commits: PREREG (frozen), then the receipt, then the
    sworn RESULT trio, the charon log, the anchors file and the run note."""
    if shutil.which("git") is None:
        pytest.skip("git is not on PATH")
    repo = tmp_path / "repo"
    (repo / "papers" / "x").mkdir(parents=True)
    (repo / "papers" / "charon").mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "core.autocrlf", "false")
    (repo / "papers" / "x" / "PREREG_x.md").write_bytes(PREREG.encode("utf-8"))
    (repo / "papers" / "x" / "run_x.py").write_bytes(RUNNER)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "prereg")
    prereg_commit = _git(repo, "rev-parse", "HEAD")
    receipt = _receipt_bytes({"gap": 0.3122, "dev": 0.0048})
    (repo / "papers" / "x" / "x_result.json").write_bytes(receipt)
    (repo / "papers" / "x" / "anchor_result.json").write_bytes(ANCHOR_RECEIPT)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "receipt")
    receipt_commit = _git(repo, "rev-parse", "HEAD")
    result = (f"# RESULT — synthetic — 2026-10-11\n\nSworn at `{receipt_commit[:12]}`.\n"
              f'<sworn r="path:papers/x/x_result.json#/metrics/gap" k="numeric">The gap was 0.3122.</sworn>\n').encode("utf-8")
    doc = repo / "papers" / "x" / "RESULT_x.md"
    doc.write_bytes(result)
    side = sworn.to_sidecar(result, doc.name, receipt_commit)
    sworn._write_json_lf(repo / "papers" / "x" / "RESULT_x.sworn.json", side)
    core = sworn.verify(sidecar=side, tree=sworn.GitTree(repo, receipt_commit), commit=receipt_commit)
    sworn._write_json_lf(repo / "papers" / "x" / "RESULT_x.sworn-receipt.json", sworn.issue_receipt(core))
    charon.ingest([repo / "papers" / "x" / "RESULT_x.sworn.json"], repo / "papers" / "charon" / "charon.log.jsonl",
                  repo, timestamp=TS, population="test")
    digest = sworn._sha256(PREREG.encode("utf-8"))
    line = {"n": 2, "kind": "sealed-prereg", "digest": digest, "memo": clock.memo("sealed-prereg", digest),
            "tx": "5" * 87, "slot": 371_000_000, "block_time": "2026-10-11T20:15:00Z"}
    (repo / "papers" / "charon" / "anchors.jsonl").write_bytes((json.dumps(line) + "\n").encode("utf-8"))
    (repo / "papers" / "x" / "x_RUN_NOTE.md").write_bytes(RUN_NOTE)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "result, log, anchor, note")
    at = _git(repo, "rev-parse", "HEAD")
    return repo, prereg_commit, at


def test_the_cli_builds_from_the_repository_at_named_commits_and_repo_reads_match(chain_repo, tmp_path, capsys):
    repo, prereg_commit, at = chain_repo
    stated = tmp_path / "stated.json"
    stated.write_text(json.dumps({"run_ended_utc": "2026-10-11T18:54:26Z"}), encoding="utf-8")
    out = tmp_path / "x.research.html"
    rcode = rc.main(["build", "--repo", str(repo), "--at", at, "--prereg", "papers/x/PREREG_x.md",
                     "--prereg-commit", prereg_commit, "--receipt", "papers/x/x_result.json",
                     "--receipt", "papers/x/anchor_result.json", "--runner", "papers/x/run_x.py",
                     "--run-note", "papers/x/x_RUN_NOTE.md", "--result", "papers/x/RESULT_x.md",
                     "--stated", str(stated), "--public-repo", "github.com/example/lab", "--out", str(out)])
    assert rcode == 0, capsys.readouterr().out
    assert out.exists() and out.stat().st_size <= SIZE_CAP
    p = _payload(out)
    assert p["artifacts"]["prereg"]["blob_commit"] == prereg_commit
    assert p["artifacts"]["charon"]["seq"] == 1 and p["experiment"]["at_commit"] == at
    assert str(repo) not in json.dumps(p) and str(tmp_path) not in json.dumps(p)
    # --repo: every public read matches the clone, the charon line is included at seq 1
    rep = verify(out, repo=repo)
    assert rep["ok"] and rep["repo"]["differs"] == 0 and rep["repo"]["not_checked"] == 0
    assert rep["repo"]["matches"] >= 7
    assert rep["repo"]["head_now"] == p["artifacts"]["charon"]["head_at_seq"]
    text = "\n".join(datasheet(rep))
    assert rc.SUMMARY_REPO_OK in text and "MATCHES" in text
    assert rc.main(["verify", str(out), "--repo", str(repo)]) == 0
    capsys.readouterr()
    # a clone whose log was rebuilt with another past: inclusion DIFFERS, nothing else moves
    log = repo / "papers" / "charon" / "charon.log.jsonl"
    lines = log.read_text(encoding="utf-8").splitlines()
    e = json.loads(lines[1])
    e["note"] = "rebuilt"
    e["prev"] = "0" * 64
    log.write_text(lines[0] + "\n" + json.dumps(e) + "\n", encoding="utf-8")
    rep = verify(out, repo=repo)
    rows = {(r["link"], r["path"]): r for r in rep["repo"]["rows"]}
    assert rows[("L5", "papers/charon/charon.log.jsonl")]["reading"] == rc.DIFFERS
    assert rep["ok"] is False and all(l["offline"] == HOLDS for l in rep["links"])
    assert rc.main(["verify", str(out), "--repo", str(repo)]) == 1
    assert "DIFFERS FROM THE REPOSITORY" in capsys.readouterr().out


def test_repo_reads_tell_a_commit_the_clone_lacks_from_a_path_absent_at_a_commit_it_has(chain_repo, minted,
                                                                                         tmp_path, capsys):
    repo, prereg_commit, at = chain_repo
    # the synthetic forger's commits are not in this clone: NOT-CHECKED, and the reason says so
    rep = verify(minted, repo=repo)
    blob_rows = [r for r in rep["repo"]["rows"] if r["link"] in ("L0", "L2", "L4")]
    assert blob_rows and all(r["reading"] == rc.NOT_CHECKED for r in blob_rows)
    assert all(r["detail"].startswith("commit_absent: the clone has no commit with this id") for r in blob_rows)
    # a forger naming commits the clone HAS, with paths absent at them: DIFFERS, not NOT-CHECKED
    inp = _inputs()
    inp.pop("_core")
    inp["experiment"]["at_commit"] = at
    inp["prereg"].update(blob_commit=prereg_commit, path="papers/y/PREREG_x.md")
    inp["runners"] = [{"path": "papers/y/run_x.py", "bytes": RUNNER}]
    forged = build(inp, tmp_path / "f.research.html", created=TS)
    assert verify(forged)["ok"]                                      # every offline link holds
    rep = verify(forged, repo=repo)
    rows = {(r["link"], r["path"]): r for r in rep["repo"]["rows"]}
    for key in (("L0", "papers/y/PREREG_x.md"), ("L2", "papers/y/run_x.py")):
        assert rows[key]["reading"] == rc.DIFFERS, rows[key]
        assert rows[key]["detail"].startswith("path_absent: the clone has commit"), rows[key]
    # the snapshot still names a commit the clone lacks, and reads NOT-CHECKED beside them
    assert rows[("L4", "papers/x/x_result.json")]["reading"] == rc.NOT_CHECKED
    assert rows[("L2", "papers/x/x_result.json")]["reading"] == rc.MATCHES
    assert rep["ok"] is False
    assert rc.main(["verify", str(forged), "--repo", str(repo)]) == 1
    out = capsys.readouterr().out
    assert "DIFFERS: papers/y/PREREG_x.md" in out and "DIFFERS FROM THE REPOSITORY" in out


def test_the_cli_build_refuses_to_read_the_working_tree(chain_repo, tmp_path, capsys):
    repo, prereg_commit, at = chain_repo
    (repo / "papers" / "x" / "x_RUN_NOTE.md").write_bytes(RUN_NOTE + b"edited after the commit\n")
    out = tmp_path / "x.research.html"
    rcode = rc.main(["build", "--repo", str(repo), "--at", at, "--prereg", "papers/x/PREREG_x.md",
                     "--prereg-commit", prereg_commit, "--receipt", "papers/x/x_result.json",
                     "--run-note", "papers/x/x_RUN_NOTE.md", "--result", "papers/x/RESULT_x.md", "--out", str(out)])
    assert rcode == 0
    p = _payload(out)
    assert base64.b64decode(p["artifacts"]["run_note"]["b64"]) == RUN_NOTE          # the blob, not the edit
    bad = rc.main(["build", "--repo", str(repo), "--at", at, "--prereg", "papers/x/PREREG_x.md",
                   "--prereg-commit", prereg_commit, "--receipt", "papers/x/x_result.json",
                   "--result", "papers/x/RESULT_x.md", "--run-note", "papers/x/not_committed.md", "--out", str(out)])
    assert bad == 2 and "REFUSED read" in capsys.readouterr().out


# ------------------------------------------------------------------------- layer 1 under node

_NODE_RUN = r"""
const fs=require('fs');
function run(file){
const html=fs.readFileSync(file,'utf8');
const payload=html.match(/<script type="application\/json" id="oath-capsule">([\s\S]*?)<\/script>/)[1];
const js=html.match(/<script id="research-verify">([\s\S]*?)<\/script>/)[1];
const els={};
function el(id){if(!els[id])els[id]={id,textContent:"",innerHTML:"",className:""};return els[id];}
global.document={
  getElementById:id=>id==="oath-capsule"?{textContent:payload}:id==="research-verify"?{textContent:js}:el(id),
  querySelector:sel=>{const m=/\[data-(offline|checked)="([^"]+)"\]/.exec(sel);return el(m[1]+":"+m[2]);},
  body:{textContent:html.replace(/<script[\s\S]*?<\/script>/g," ").replace(/<[^>]+>/g," ")}};
new Function(js)();
const out={summary:el("summary").textContent,summary_class:el("summary").className,links:{}};
["L0","L1","L1b","L2","L3","L4","L5","L6"].forEach(k=>{out.links[k]={offline:el("offline:"+k).textContent,checked:el("checked:"+k).textContent};});
return out;}
"""
_NODE_SHIM = _NODE_RUN + "console.log(JSON.stringify(run(process.argv[1])));\n"
_NODE_SHIM_MANY = _NODE_RUN + "console.log(JSON.stringify(JSON.parse(fs.readFileSync(process.argv[1],'utf8')).map(run)));\n"


def _layer1(cap: Path) -> dict:
    if shutil.which("node") is None:
        pytest.skip("node is not on PATH")
    p = subprocess.run(["node", "-e", _NODE_SHIM, str(cap)], capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=60)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout)


def _layer1_many(caps: list, tmp: Path) -> list:
    """Layer 1 over many capsules in one node process, in order."""
    if shutil.which("node") is None:
        pytest.skip("node is not on PATH")
    listing = tmp / "layer1_files.json"
    listing.write_text(json.dumps([str(c) for c in caps]), encoding="utf-8")
    p = subprocess.run(["node", "-e", _NODE_SHIM_MANY, str(listing)], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=600)
    assert p.returncode == 0, p.stderr
    out = json.loads(p.stdout)
    assert len(out) == len(caps)
    return out


def _l1_fails(l1: dict) -> set:
    return {k for k, v in l1["links"].items() if v["offline"] == FAILS}


def test_layer_1_agrees_with_layer_2_on_the_clean_capsule(minted):
    l1 = _layer1(minted)
    rep = verify(minted)
    assert l1["summary"].startswith(SUMMARY_OK), l1
    assert l1["summary_class"] == "summary ok"
    for l in rep["links"]:
        assert l1["links"][l["id"]]["offline"] == l["offline"] == HOLDS, (l["id"], l1["links"][l["id"]])
    assert "re-scored verdict EFFECT_HOLDS" in l1["links"]["L3"]["checked"]
    assert "verified" not in l1["summary"].lower()


def _agree(l1: dict, rep: dict, name: str) -> None:
    for l in rep["links"]:
        assert l1["links"][l["id"]]["offline"] == l["offline"], (name, l["id"], l1["links"][l["id"]], l["problems"])


@pytest.mark.parametrize("where,needle,link", DIGEST_BYTES)
def test_layer_1_agrees_with_layer_2_on_every_byte_mutation(minted, tmp_path, where, needle, link):
    cap = _digest_mutant(minted, tmp_path / "t.research.html", where, needle)
    l1, rep = _layer1(cap), verify(cap)
    assert l1["summary"].startswith(SUMMARY_FAILS)
    _agree(l1, rep, where)
    assert l1["links"][link]["offline"] == FAILS


@pytest.mark.parametrize("edit,link", FIELD_EDITS)
def test_layer_1_agrees_with_layer_2_on_every_field_mutation(minted, tmp_path, edit, link):
    p = _payload(minted)
    edit(p["artifacts"])
    cap = _reseal(minted, p, tmp_path / "t.research.html")
    l1, rep = _layer1(cap), verify(cap)
    assert _fails(rep) == {link}
    assert l1["summary"].startswith(SUMMARY_FAILS)
    _agree(l1, rep, link)


def test_layer_1_agrees_with_layer_2_on_a_byte_inside_the_gates_block(minted, tmp_path):
    p = _payload(minted)
    _flip_b64(p["artifacts"]["prereg"], b'"value": 0.15')
    cap = _reseal(minted, p, tmp_path / "t.research.html")
    l1, rep = _layer1(cap), verify(cap)
    assert "L0" in _fails(rep)
    _agree(l1, rep, "gates block")


def test_layer_1_agrees_with_layer_2_on_the_controls(minted, tmp_path):
    def created(p):
        p["created"] = "2030-01-01T00:00:00Z"

    def note(p):
        p["artifacts"]["charon"]["entry"].update(timestamp="2030-01-01T00:00:00Z", note="re-ferried")
    for name, edit, sort_keys in (("created", created, True), ("note", note, True), ("key order", None, False)):
        p = _payload(minted)
        if edit:
            edit(p)
        else:
            p = {k: p[k] for k in reversed(list(p))}
        cap = _reseal(minted, p, tmp_path / f"{name}.research.html", sort_keys=sort_keys)
        l1, rep = _layer1(cap), verify(cap)
        assert _fails(rep) == set(), name
        _agree(l1, rep, name)
        if name == "note":          # outside the line's entry_id, inside the payload digest: both say so
            assert rep["payload"]["core_sha256_ok"] is False and "core_sha256 does not re-derive" in l1["summary"]
        else:
            assert rep["ok"] and l1["summary"].startswith(SUMMARY_OK), (name, l1["summary"])


def test_layer_1_agrees_with_layer_2_on_the_slot_neither_catches_offline(minted, tmp_path):
    p = _payload(minted)
    p["artifacts"]["anchors"][0]["slot"] += 1
    cap = _reseal(minted, p, tmp_path / "t.research.html")
    l1, rep = _layer1(cap), verify(cap)
    assert _fails(rep) == set()
    _agree(l1, rep, "slot")
    # both summaries fail on the payload digest alone, and the browser says which
    assert l1["summary"].startswith(SUMMARY_FAILS) and "core_sha256 does not re-derive" in l1["summary"]


@pytest.mark.parametrize("name,links,make", SEMANTIC, ids=[s[0] for s in SEMANTIC])
def test_layer_1_agrees_with_layer_2_on_every_semantic_mutation(tmp_path, name, links, make):
    """Every member of the semantic battery, each L4 member included (a lying receipt, an honest
    SWORN-FAILED one, UNRESOLVED spans, a RESULT that swears nothing): the browser reads each link
    as the installed layer does, and never prints the consistent summary over a false link."""
    cap = make(tmp_path, "s.research.html")
    l1, rep = _layer1(cap), verify(cap)
    assert _fails(rep) == links, (name, [(l["id"], l["problems"]) for l in rep["links"]])
    _agree(l1, rep, name)
    assert l1["summary"].startswith(SUMMARY_FAILS) and l1["summary_class"] == "summary bad", (name, l1["summary"])
    if name == "L4 a span value the receipt does not hold":
        assert "live SWORN-FAILED" in l1["links"]["L4"]["checked"], l1["links"]["L4"]
    if name == "L4 an edited sworn verdict":
        assert "its own spans tally SWORN-HELD" in l1["links"]["L4"]["checked"], l1["links"]["L4"]
    if name in ("L4 an honest SWORN-FAILED receipt", "L4 an UNRESOLVED rN span", "L4 an UNRESOLVED prereg span"):
        assert "sworn spans read HELD" in l1["links"]["L4"]["checked"], l1["links"]["L4"]
    if name == "L4 a RESULT that swears nothing":
        assert "the RESULT swears nothing (UNSWORN)" in l1["links"]["L4"]["checked"], l1["links"]["L4"]


def test_layer_1_re_derives_every_span_kind_over_the_snapshot(tmp_path):
    """numeric, quote and hash spans resolved through the page's snapshot resolver: both layers
    hold on the honest chain and both fail L4 alone when a quoted needle is one the leaf lacks."""
    clean = _write_consistent(tmp_path, _inputs(result_fn=_result_every_kind), name="kinds.research.html")
    l1, rep = _layer1(clean), verify(clean)
    assert rep["ok"], [(l["id"], l["problems"]) for l in rep["links"]]
    l4 = next(l for l in rep["links"] if l["id"] == "L4")
    assert any("SWORN-HELD {'HELD': 3," in c for c in l4["checked"]), l4["checked"]
    assert l1["summary"].startswith(SUMMARY_OK), l1
    assert all(v["offline"] == HOLDS for v in l1["links"].values())
    assert "the sworn core re-derives here over the carried snapshot: SWORN-HELD" in l1["links"]["L4"]["checked"]
    forged = _write_consistent(tmp_path, _inputs_repointed(b"`synthetic`", b"`synthetiq`", result_fn=_result_every_kind),
                               name="kinds-forged.research.html")
    l1, rep = _layer1(forged), verify(forged)
    assert _fails(rep) == {"L4"}
    assert l1["links"]["L4"]["offline"] == FAILS and l1["summary"].startswith(SUMMARY_FAILS)
    assert all(l1["links"][l["id"]]["offline"] == l["offline"] for l in rep["links"])


def test_layer_1_and_layer_2_resolve_alike_when_the_sidecar_commit_is_not_the_snapshot_commit(tmp_path):
    """A snapshot taken at another commit resolves nothing (commit_absent); both layers fail L4."""
    def move(p):
        p["artifacts"]["snapshot"]["commit"] = "d" * 40
    cap = _write_consistent(tmp_path, _inputs(), mutate=move, name="moved.research.html")
    l1, rep = _layer1(cap), verify(cap)
    assert "L4" in _fails(rep)
    assert all(l1["links"][l["id"]]["offline"] == l["offline"] for l in rep["links"])


def _span0(p: dict) -> dict:
    return p["artifacts"]["sidecar"]["spans"][0]


def _entry0(p: dict) -> dict:
    return next(iter(p["artifacts"]["snapshot"]["entries"].values()))


def _srec(p: dict) -> dict:
    return p["artifacts"]["sworn_receipt"]


def _anchor0(p: dict) -> dict:
    return p["artifacts"]["anchors"][0]


def _run_receipt_second(p: dict, index) -> None:
    """The run's receipt carried second and named by ``index``: a JS array reads 1, "1" and true
    alike, Python's json does not."""
    r = p["artifacts"]["receipts"]
    r[0], r[1] = r[1], r[0]
    p["experiment"]["score"]["receipt"] = index


_RUN = {"L2", "L3", "L5"}       # no run receipt: L2 names it, L3 has nothing to score, L5 no receipt to find

# a field whose JSON type is not the one the instrument reads: JSON.parse writes 102.0 as 102, so the
# page reads every integer the instruments read through a typed parse of its payload that keeps a
# float literal apart, as Python's json does. (name, mutation, the links it flips); a control flips none
TYPE_CONFUSED = [
    ("a span start written as a float", lambda p: _span0(p).__setitem__("start", float(_span0(p)["start"])), {"L4"}),
    ("a span end written as a float", lambda p: _span0(p).__setitem__("end", float(_span0(p)["end"])), {"L4"}),
    ("a snapshot size written as a float", lambda p: _entry0(p).__setitem__("size", float(_entry0(p)["size"])), {"L4"}),
    ("a snapshot size written as a string", lambda p: _entry0(p).__setitem__("size", str(_entry0(p)["size"])), {"L4"}),
    ("a snapshot size written as a list", lambda p: _entry0(p).__setitem__("size", [_entry0(p)["size"]]), {"L4"}),
    ("a snapshot size that is not the byte count", lambda p: _entry0(p).__setitem__("size", _entry0(p)["size"] - 1),
     {"L4"}),
    ("a sworn count written as a float", lambda p: _srec(p)["counts"].__setitem__("HELD", 2.0), {"L4"}),
    ("a sworn count of zero written as a float", lambda p: _srec(p)["counts"].__setitem__("FAILED", 0.0), {"L4"}),
    ("the sworn total written as a float", lambda p: _srec(p).__setitem__("sworn_total", 2.0), {"L4"}),
    ("the sworn unresolved count written as a float", lambda p: _srec(p).__setitem__("unresolved", 0.0), {"L4"}),
    ("an anchor slot written as a float", lambda p: _anchor0(p).__setitem__("slot", float(_anchor0(p)["slot"])), {"L1"}),
    ("an anchor slot written as true", lambda p: _anchor0(p).__setitem__("slot", True), {"L1"}),
    ("an anchor slot written as a string", lambda p: _anchor0(p).__setitem__("slot", str(_anchor0(p)["slot"])), {"L1"}),
    ("the run receipt's index written as a float", lambda p: p["experiment"]["score"].__setitem__("receipt", 0.0), _RUN),
    ("the run receipt's index written as null", lambda p: p["experiment"]["score"].__setitem__("receipt", None), _RUN),
    ("the run receipt carried second, its index written 1.0", lambda p: _run_receipt_second(p, 1.0), _RUN),
    ("the run receipt carried second, its index written as a string", lambda p: _run_receipt_second(p, "1"), _RUN),
    ("the run receipt carried second, its index written as true", lambda p: _run_receipt_second(p, True), _RUN),
    ("control: the run receipt carried second, its index written 1", lambda p: _run_receipt_second(p, 1), set()),
]


@pytest.mark.parametrize("name,mutate,links", TYPE_CONFUSED, ids=[x[0] for x in TYPE_CONFUSED])
def test_a_type_confused_field_fails_exactly_its_links_by_name(tmp_path, name, mutate, links):
    cap = _write_consistent(tmp_path, _inputs(), mutate=mutate, name="t.research.html")
    rep = verify(cap)
    assert _fails(rep) == links, (name, [(l["id"], l["problems"]) for l in rep["links"]])
    assert rep["payload"]["core_sha256_ok"] and rep["payload"]["page_ok"]
    problems = [p for l in rep["links"] for p in l["problems"]]
    assert not any("Error:" in p for p in problems), problems        # named, never a TypeError
    assert rep["ok"] is (not links)
    if links:
        assert any("integer" in p or "byte count" in p for p in problems), problems


@pytest.mark.parametrize("name,mutate,links", TYPE_CONFUSED, ids=[x[0] for x in TYPE_CONFUSED])
def test_layer_1_agrees_with_layer_2_on_a_type_confused_field(tmp_path, name, mutate, links):
    cap = _write_consistent(tmp_path, _inputs(), mutate=mutate, name="t.research.html")
    l1, rep = _layer1(cap), verify(cap)
    _agree(l1, rep, name)
    assert _l1_fails(l1) == links
    if links:
        assert l1["summary"].startswith(SUMMARY_FAILS) and l1["summary_class"] == "summary bad", (name, l1["summary"])
    else:
        assert l1["summary"].startswith(SUMMARY_OK), (name, l1["summary"])


def _int_literals(body: str) -> list:
    """(start, end, key) of every integer literal outside a string in a JSON text; key is the
    nearest object key written before it."""
    out, i, n, key = [], 0, len(body), None
    while i < n:
        c = body[i]
        if c == '"':
            j = i + 1
            while body[j] != '"':
                j += 2 if body[j] == "\\" else 1
            k = j + 1
            while k < n and body[k] in " \t\r\n":
                k += 1
            if k < n and body[k] == ":":
                key = json.loads(body[i:j + 1])
            i = j + 1
        elif c == "-" or c in "0123456789":
            j = i + 1
            while j < n and body[j] in "0123456789+-.eE":
                j += 1
            if re.fullmatch(r"-?[0-9]+", body[i:j]):
                out.append((i, j, key))
            i = j
        else:
            i += 1
    return out


def test_every_integer_literal_written_as_a_float_reads_alike_in_both_layers(minted, tmp_path):
    """Review 3's sweep: each integer literal of the clean page rewritten ``N.0`` by hand. JCS writes
    both the same, so no digest moves; where an instrument reads an integer, both layers fail that
    link (by name in the terminal), and everywhere else both hold. Never a traceback."""
    html = minted.read_text(encoding="utf-8")
    m = PAYLOAD_RE.search(html)
    body = m.group(1)
    lits = _int_literals(body)
    assert len(lits) >= 30, len(lits)
    caps, reps = [], []
    for k, (i, j, key) in enumerate(lits):
        cap = tmp_path / f"f{k:02d}.research.html"
        cap.write_bytes((html[:m.start(1)] + body[:j] + ".0" + body[j:] + html[m.end(1):]).encode("utf-8"))
        rep = verify(cap)
        assert rep["payload"]["core_sha256_ok"] is True, key
        problems = [p for l in rep["links"] for p in l["problems"]]
        assert not any("Error:" in p for p in problems), (key, problems)
        caps.append(cap)
        reps.append(rep)
    failing = {key for (_i, _j, key), rep in zip(lits, reps) if _fails(rep)}
    # the integers the instruments read: span offsets, the snapshot size, the sworn receipt's counts,
    # sworn_total and unresolved, the anchor slot, the run receipt's index
    assert {"start", "end", "size", "HELD", "FAILED", "sworn_total", "unresolved", "slot",
            "receipt"} <= failing, failing
    l1s = _layer1_many(caps, tmp_path)
    split = [(key, sorted(_fails(rep)), sorted(_l1_fails(l1)))
             for (_i, _j, key), rep, l1 in zip(lits, reps, l1s) if _fails(rep) != _l1_fails(l1)]
    assert split == []
    for (_i, _j, key), rep, l1 in zip(lits, reps, l1s):
        if _fails(rep):
            assert l1["summary"].startswith(SUMMARY_FAILS), (key, l1["summary"])


# a field written in a shape one layer reads and the other did not, or a time that fits the format
# and names no time: (name, mutation, the links it flips)
def _clock0(p: dict) -> dict:
    return p["artifacts"]["public_clocks"][0]


_FULLWIDTH = "\uff12\uff10\uff12\uff16-10-11T20:15:00Z"
SHAPES = [
    ("an anchor line that is not an object", lambda p: p["artifacts"]["anchors"].__setitem__(0, "a line"), {"L1"}),
    ("an anchor block_time on February 30", lambda p: _anchor0(p).__setitem__("block_time", "2026-02-30T20:15:00Z"),
     {"L1"}),
    ("an anchor block_time at hour 24", lambda p: _anchor0(p).__setitem__("block_time", "2026-10-11T24:00:00Z"), {"L1"}),
    ("an anchor block_time at second 60", lambda p: _anchor0(p).__setitem__("block_time", "2026-10-11T23:59:60Z"),
     {"L1"}),
    ("an anchor block_time in year 0", lambda p: _anchor0(p).__setitem__("block_time", "0000-10-11T20:15:00Z"), {"L1"}),
    ("an anchor block_time in fullwidth digits", lambda p: _anchor0(p).__setitem__("block_time", _FULLWIDTH), {"L1"}),
    ("control: an anchor block_time in year 50", lambda p: _anchor0(p).__setitem__("block_time", "0050-10-11T20:15:00Z"),
     set()),
    ("control: an anchor block_time on a leap day", lambda p: _anchor0(p).__setitem__("block_time", "2024-02-29T20:15:00Z"),
     set()),
    ("a public clock stated in month 13", lambda p: _clock0(p).__setitem__("stated", "2026-13-01T00:00:00Z"), {"L1b"}),
    ("a public clock stated in fullwidth digits", lambda p: _clock0(p).__setitem__("stated", _FULLWIDTH), {"L1b"}),
    ("a public clock with an empty name", lambda p: _clock0(p).__setitem__("name", ""), {"L1b"}),
    ("a public clock whose binds is ideographic space", lambda p: _clock0(p).__setitem__("binds", "\u3000"), {"L1b"}),
    ("a public clock whose read is an information separator", lambda p: _clock0(p).__setitem__("read", "\x1c"),
     {"L1b"}),
    ("a public clock whose does_not_prove is a number", lambda p: _clock0(p).__setitem__("does_not_prove", 5), {"L1b"}),
    ("a public clock whose does_not_prove is null", lambda p: _clock0(p).__setitem__("does_not_prove", None), {"L1b"}),
    ("control: a public clock whose name is a byte order mark", lambda p: _clock0(p).__setitem__("name", "\ufeff"),
     set()),
    ("control: a stated run end on February 30", lambda p: p["stated"].__setitem__("run_ended_utc", "2026-02-30T18:54:26Z"),
     set()),
]


@pytest.mark.parametrize("name,mutate,links", SHAPES, ids=[x[0] for x in SHAPES])
def test_a_shape_or_an_impossible_time_fails_exactly_its_link_by_name(tmp_path, name, mutate, links):
    cap = _write_consistent(tmp_path, _inputs(), mutate=mutate, name="s.research.html")
    rep = verify(cap)                                              # never a traceback
    assert _fails(rep) == links, (name, [(l["id"], l["problems"]) for l in rep["links"]])
    assert rep["payload"]["core_sha256_ok"] and rep["payload"]["page_ok"]
    assert rep["ok"] is (not links)
    if "stated run end" in name:
        l1 = next(l for l in rep["links"] if l["id"] == "L1")
        assert any("the stated run end '2026-02-30T18:54:26Z' is not an ISO-8601 UTC time that exists" in f
                   for f in l1["stated"]["facts"]), l1["stated"]["facts"]


@pytest.mark.parametrize("name,mutate,links", SHAPES, ids=[x[0] for x in SHAPES])
def test_layer_1_agrees_with_layer_2_on_a_shape_or_an_impossible_time(tmp_path, name, mutate, links):
    cap = _write_consistent(tmp_path, _inputs(), mutate=mutate, name="s.research.html")
    l1, rep = _layer1(cap), verify(cap)
    _agree(l1, rep, name)
    assert _l1_fails(l1) == links
    if links:
        assert l1["summary"].startswith(SUMMARY_FAILS), (name, l1["summary"])
        for lid in links & {"L1b"}:              # the page names the public clock's problem word for word
            for p in next(l for l in rep["links"] if l["id"] == lid)["problems"]:
                assert p.split(" — ", 1)[-1] in l1["links"][lid]["checked"], (name, p, l1["links"][lid])
    else:
        assert l1["summary"].startswith(SUMMARY_OK), (name, l1["summary"])


@pytest.mark.parametrize("where,edit", [
    ("anchor", lambda i: i["anchors"][0].__setitem__("block_time", "2026-02-30T20:15:00Z")),
    ("public_clock", lambda i: i["public_clocks"][0].__setitem__("stated", "2026-02-30T00:00:00Z")),
    ("stated", lambda i: i["stated"].__setitem__("run_ended_utc", "2026-02-30T18:54:26Z")),
])
def test_build_refuses_a_time_that_does_not_exist_by_name(tmp_path, where, edit):
    inp = _inputs()
    inp.pop("_core")
    edit(inp)
    with pytest.raises(SystemExit) as e:
        build(inp, tmp_path / "no.research.html", created=TS)
    msg = str(e.value.code)
    assert "an ISO-8601 UTC time that exists" in msg, msg
    assert {"anchor": "REFUSED inconsistent_chain", "public_clock": "REFUSED public_clock",
            "stated": "REFUSED stated"}[where] in msg
    assert not (tmp_path / "no.research.html").exists()


def test_parse_iso_names_only_a_time_that_exists():
    assert rc._parse_iso("2026-10-11T20:15:00Z") is not None
    assert rc._parse_iso("2024-02-29T00:00:00Z") is not None
    assert rc._parse_iso("0050-01-01T00:00:00Z") is not None
    for s in ("2026-02-30T00:00:00Z", "2026-13-01T00:00:00Z", "2026-00-10T00:00:00Z", "2026-10-00T00:00:00Z",
              "2026-10-11T24:00:00Z", "2026-10-11T23:60:00Z", "2026-10-11T23:59:60Z", "0000-01-01T00:00:00Z",
              _FULLWIDTH, "2026-10-11 20:15:00Z", "2026-10-11T20:15:00", None, 5):
        assert rc._parse_iso(s) is None, s


def test_the_page_blank_class_is_python_str_isspace():
    """The page's public-clock rule calls a string blank as Python's ``str.strip`` does; its class
    is written out, since ``String.prototype.trim`` strips another set."""
    m = re.search(r"var PYWS=/\^\[(.*?)\]\*\$/;", rc._JS_RESEARCH)
    assert m, "PYWS not found in the page script"
    rx = re.compile("[" + m.group(1) + "]")
    page = {c for c in range(0x110000) if rx.fullmatch(chr(c))}
    py = {c for c in range(0x110000) if chr(c).isspace()}
    assert page == py, (sorted(page - py), sorted(py - page))


def test_an_exponent_literal_offset_is_a_float_in_both_layers(minted, tmp_path):
    """Written by hand into the page: ``102e0`` parses to the same number in both languages and
    leaves the payload digest unchanged, and it is a float to Python's json, so neither layer
    reads it as an offset."""
    html = minted.read_text(encoding="utf-8")
    m = PAYLOAD_RE.search(html)
    body = m.group(1)
    at = body.index('"sidecar": {')
    s = body.index('"start": ', at) + len('"start": ')
    e = s + len(re.match(r"\d+", body[s:]).group(0))
    cap = tmp_path / "e.research.html"
    cap.write_bytes((html[:m.start(1)] + body[:e] + "e0" + body[e:] + html[m.end(1):]).encode("utf-8"))
    rep = verify(cap)
    assert rep["payload"]["core_sha256_ok"] is True and _fails(rep) == {"L4"}
    assert any("carry an offset that is not an integer" in p
               for p in next(l for l in rep["links"] if l["id"] == "L4")["problems"])
    l1 = _layer1(cap)
    _agree(l1, rep, "exponent")
    assert l1["summary"].startswith(SUMMARY_FAILS) and "core_sha256" not in l1["summary"]


def test_the_page_before_its_script_runs_claims_nothing(minted):
    page = _page_outside_payload(minted)
    assert page.count('>NOT RUN<') == 8
    assert "layer 1 has not run: nothing on this page is checked" in page
    # the inlined verifier is the installed one and is sealed by digest in the payload
    p = _payload(minted)
    assert p["verifier_js"]["sha256"] == sworn._sha256(rc._JS.encode("utf-8"))
    html = minted.read_text(encoding="utf-8")
    assert rc._JS in html
    # the sworn layer is the package's second implementation, inlined byte for byte (LF), not edited
    sworn_js = (ROOT / "styxx" / "_data" / "sworn_verify.js").read_bytes().decode("utf-8").replace("\r\n", "\n")
    assert sworn_js in html and rc._JS_RESEARCH in html


# --------------------------------------------------------------------- a /v1 sworn receipt, carried

def _inputs_v1_receipt(certifies=None) -> dict:
    """The synthetic chain with its sworn receipt as a /v1 verifier issued it: the core projected onto
    styxx.sworn.verdict-receipt/v1 and re-digested; the charon line agrees with it."""
    inp = _inputs()
    v1 = sworn._core_as(inp["sworn_receipt"], "styxx.sworn.verdict-receipt/v1")
    if certifies is not None:
        v1["certifies"] = certifies
    v1 = _redigest(v1)
    inp["sworn_receipt"] = v1
    inp["charon"]["entry"] = _charon_entry(inp["sidecar"], v1, inp["_core"], "papers/x/RESULT_x.md", HEADER)
    inp.pop("_core")
    return inp


def test_a_research_capsule_carries_a_v1_sworn_receipt(tmp_path):
    """Every sworn receipt committed before 2026-10-24 is /v1, and the CLI builds from a committed one
    (papers/sworn/NOTE_certifies_correction_2026_10_24.md). L4 projects the live core onto the carried
    receipt's own schema, in Python and in the page, so a /v1 receipt holds as it did under a /v1
    verifier; one whose certifies was re-worded and re-digested does not."""
    inp = _inputs_v1_receipt()
    assert inp["sworn_receipt"]["schema"] == "styxx.sworn.verdict-receipt/v1"
    assert inp["sworn_receipt"]["certifies"] == sworn._CERTIFIES_V1
    cap = build(inp, tmp_path / "v1.research.html", created=TS)
    rep = verify(cap)
    assert rep["ok"], [(l["id"], l["problems"]) for l in rep["links"]]
    l1 = _layer1(cap)
    assert l1["summary"].startswith(SUMMARY_OK), l1
    _agree(l1, rep, "v1 receipt")
    assert "the /v1 sentence, withdrawn: papers/sworn/NOTE_certifies_correction_2026_10_24.md" in rc._JS_RESEARCH

    bad = _write_consistent(tmp_path, _inputs_v1_receipt(certifies=sworn._CERTIFIES), name="v1bad.research.html")
    rep = verify(bad)
    assert _fails(rep) == {"L4"}, [(l["id"], l["problems"]) for l in rep["links"]]
    l1 = _layer1(bad)
    _agree(l1, rep, "v1 receipt carrying the v2 sentence")
