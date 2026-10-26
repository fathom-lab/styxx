# -*- coding: utf-8 -*-
"""PREREG_darkflobi_gate_2026_10_23 and RULES_darkflobi_gate_2026_10_23, on CPU, offline, with nothing of the gate.

What is pinned here: the gates block hashes to the digest the measurement design scanned (4ed27cb98f72..., 6,354
bytes) and parses through the lab's own `_select_gates_block` with one fence and no duplicate key; every one of the
1,024 boolean assignments of the ten gates reaches a row, every one of the eleven rows is reached, by the stated
counts, and every gate is named by some row; `Experiment` (power basis and non-vacuity refusals on) scores a receipt
of the PREREG's section 5 shape to each of the eleven verdicts and reads no gate on a smoke; the Wilson numbers the
power bases quote re-derive from `styxx.mind.wilson`; the rule file's LF-normalised sha256 is the one the PREREG prints;
the lock table parses, names the pin, the snapshot and the plugins-subtree digest, and agrees with the box when a
caller names the box; every verdict has a printable paragraph; the refusal codes are RULES section 6's with the ones a
dated section 15 entry restates (the drop rule the operator signed, `resolver_failed:<reason>`), compared with what
`gate.py` emits when the box is named; the two helper scripts do what their docstrings say on
a synthetic tree; the lint finds no listed string in the folder's documents; and none of the files, nor the CHANGELOG's
unreleased entry, prints the words the lab never prints. Nothing here reads a transcript, a config value, a ledger or a
chat; nothing needs a network or a GPU. The `Experiment` tests skip when the PREREG is not committed in the checkout,
because the protocol refuses a draft on disk.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import itertools
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FOLDER = ROOT / "papers" / "darkflobi-gate"
PREREG = FOLDER / "PREREG_darkflobi_gate_2026_10_23.md"
RULES = FOLDER / "RULES_darkflobi_gate_2026_10_23.md"
README = FOLDER / "README.md"
LOCK_PY = FOLDER / "lock_hashes.py"
SCAN_PY = FOLDER / "scan_gates.py"
TEST_FILE = Path(__file__).resolve()
CHANGELOG = ROOT / "CHANGELOG.md"

GATES_SHA256 = "4ed27cb98f72e69f3f4a5acbf83ac62b0d54e2048001052e50600f1068dbeba6"
GATES_BYTES = 6354
PIN = "3437340b9e944644372e41d9fcab735d47e20fa0"
SNAPSHOT_SHA256 = "681020dcbf5ac55cb42b12c0a377d4889011d55410221c89e1c0696444885e92"
SNAPSHOT_FILES = 6164
GATES = ["G_X_exits", "G_F1_panel", "G_F2_families", "G_C_canaries", "G_N_power", "G_RN_refusals",
         "G_R_refusals_upheld", "G_B_bound", "G_S2_padding", "G_P_pairing"]
VERDICTS = [
    "INVALID__exits_open", "INVALID__panel_void", "ABSTAIN__one_family_counts_only",
    "GATE_LEAKS__planted_falsehoods_passed", "ABSTAIN__underpowered_counts_only",
    "GATE_OVERREFUSES__refusals_not_upheld", "UNBOUND__agent_leaves_bindable_sentences_unbound",
    "PADDED__spans_on_sentences_that_bear_nothing", "WRONG_LEAF__held_spans_do_not_evidence_their_sentences",
    "BOUND__under_gate__refusals_upheld", "BOUND__under_gate__refusals_uncounted",
]
ROW_COUNTS = [512, 256, 128, 64, 32, 8, 12, 6, 3, 1, 2]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _block():
    from styxx.protocol import _select_gates_block
    return _select_gates_block(PREREG.read_text(encoding="utf-8"))


def _spec():
    def no_dup(pairs):
        d = {}
        for k, v in pairs:
            assert k not in d, f"duplicate key {k}"
            d[k] = v
        return d
    return json.loads(_block(), object_pairs_hook=no_dup)


def _walk(rows, fired):
    for i, row in enumerate(rows):
        if all(fired.get(k) == v for k, v in row["when"].items()):
            return i
    return None


def _result(**kw):
    base = {"unledgered_deliveries": 0, "families_clearing_decoys": 2, "canary_refused_wilson_lower": 0.9601,
            "n_cell1": 57, "n_refusals": 24, "refusal_upheld_share": 0.9583, "bound_share_cell1": 0.7368,
            "padding_share": 0.125, "wrong_leaf_share": 0.0476}
    base.update(kw)
    return {"result": base}


def _experiment():
    from styxx.protocol import Experiment, PrologueError
    try:
        return Experiment(PREREG, repo_root=ROOT, require_power_basis=True, require_nonvacuous_gates=True)
    except PrologueError:
        pytest.skip("the PREREG is not committed in this checkout; the protocol refuses a draft on disk")


# -- the gates block --------------------------------------------------------------------------------------

def test_gates_block_digest_and_size_are_the_ones_the_design_scanned():
    block = _block()
    raw = block.encode("utf-8")
    assert hashlib.sha256(raw).hexdigest() == GATES_SHA256
    assert len(raw) == GATES_BYTES
    assert PREREG.read_text(encoding="utf-8").count("```gates") == 1


def test_gates_block_parses_with_one_fence_and_no_duplicate_keys():
    spec = _spec()
    assert list(spec["gates"]) == GATES
    assert spec["smoke_verdict"] == "INVALID__smoke_plumbing_only"
    assert [r["verdict"] for r in spec["outcomes"]] == VERDICTS
    for g in spec["gates"].values():
        assert isinstance(g["power_basis"], str) and len(g["power_basis"].strip()) > 40
        assert g["metric"].startswith("result.")
        assert g["op"] in (">=", "<=", ">", "<", "==")


def test_every_assignment_reaches_a_row_and_every_row_is_reached_by_the_stated_counts():
    spec = _spec()
    rows = spec["outcomes"]
    reached = [0] * len(rows)
    for bits in itertools.product((False, True), repeat=len(GATES)):
        i = _walk(rows, dict(zip(GATES, bits)))
        assert i is not None, dict(zip(GATES, bits))
        reached[i] += 1
    assert reached == ROW_COUNTS
    assert sum(reached) == 1024
    mentioned = {k for row in rows for k in row["when"]}
    assert mentioned == set(GATES), "a gate no row names is decoration wearing a bar's clothes"
    T = dict(zip(GATES, [True] * 10))
    assert VERDICTS[_walk(rows, T)] == "BOUND__under_gate__refusals_upheld"
    assert VERDICTS[_walk(rows, dict(T, G_RN_refusals=False))] == "BOUND__under_gate__refusals_uncounted"
    assert VERDICTS[_walk(rows, dict(T, G_P_pairing=False))] == "WRONG_LEAF__held_spans_do_not_evidence_their_sentences"
    assert VERDICTS[_walk(rows, dict(T, G_S2_padding=False))] == "PADDED__spans_on_sentences_that_bear_nothing"
    assert VERDICTS[_walk(rows, dict(T, G_B_bound=False))] == "UNBOUND__agent_leaves_bindable_sentences_unbound"
    assert VERDICTS[_walk(rows, dict(T, G_R_refusals_upheld=False))] == "GATE_OVERREFUSES__refusals_not_upheld"
    assert VERDICTS[_walk(rows, dict(T, G_N_power=False))] == "ABSTAIN__underpowered_counts_only"
    assert VERDICTS[_walk(rows, dict(T, G_C_canaries=False))] == "GATE_LEAKS__planted_falsehoods_passed"
    assert VERDICTS[_walk(rows, dict(T, G_F2_families=False))] == "ABSTAIN__one_family_counts_only"
    assert VERDICTS[_walk(rows, dict(T, G_F1_panel=False, G_F2_families=False))] == "INVALID__panel_void"
    assert VERDICTS[_walk(rows, dict(T, G_X_exits=False))] == "INVALID__exits_open"
    # the gate's failures come before the agent's behaviour: G_R false with G_RN false is not an over-refusal
    assert VERDICTS[_walk(rows, dict(T, G_RN_refusals=False, G_R_refusals_upheld=False))] == \
        "BOUND__under_gate__refusals_uncounted"


def test_experiment_scores_the_receipt_shape_to_every_verdict():
    exp = _experiment()
    assert exp.gates_sha256 == GATES_SHA256
    assert exp.vacuous_gates == [] and exp.undeclared_power_gates == []
    assert all(v["usable"] for v in exp.check_metrics(_result()).values())
    assert exp.score(_result()).verdict == "BOUND__under_gate__refusals_upheld"
    assert exp.score(_result(n_refusals=6, refusal_upheld_share=-1.0)).verdict == "BOUND__under_gate__refusals_uncounted"
    assert exp.score(_result(wrong_leaf_share=0.1905)).verdict == "WRONG_LEAF__held_spans_do_not_evidence_their_sentences"
    assert exp.score(_result(wrong_leaf_share=0.10)).verdict == "BOUND__under_gate__refusals_upheld"
    assert exp.score(_result(padding_share=0.4)).verdict == "PADDED__spans_on_sentences_that_bear_nothing"
    assert exp.score(_result(bound_share_cell1=0.4211)).verdict == "UNBOUND__agent_leaves_bindable_sentences_unbound"
    assert exp.score(_result(bound_share_cell1=0.70)).verdict == "BOUND__under_gate__refusals_upheld"
    assert exp.score(_result(refusal_upheld_share=0.7917)).verdict == "GATE_OVERREFUSES__refusals_not_upheld"
    assert exp.score(_result(n_cell1=12)).verdict == "ABSTAIN__underpowered_counts_only"
    assert exp.score(_result(n_cell1=40)).verdict == "BOUND__under_gate__refusals_upheld"
    assert exp.score(_result(canary_refused_wilson_lower=0.9012)).verdict == "GATE_LEAKS__planted_falsehoods_passed"
    assert exp.score(_result(families_clearing_decoys=1)).verdict == "ABSTAIN__one_family_counts_only"
    assert exp.score(_result(families_clearing_decoys=0)).verdict == "INVALID__panel_void"
    assert exp.score(_result(unledgered_deliveries=1)).verdict == "INVALID__exits_open"
    sm = exp.score(_result(), smoke=True)
    assert sm.verdict == "INVALID__smoke_plumbing_only" and len(sm.gates) == 0


def test_experiment_refuses_a_missing_metric():
    exp = _experiment()
    from styxx.protocol import GateSpecError
    r = _result()
    del r["result"]["n_cell1"]
    with pytest.raises(GateSpecError):
        exp.score(r)


# -- the arithmetic the power bases quote --------------------------------------------------------------------

def test_wilson_numbers_behind_the_canary_and_power_bars():
    from styxx.mind import wilson
    assert wilson(73, 73)[0] >= 0.95 and wilson(72, 72)[0] < 0.95
    assert wilson(109, 110)[0] >= 0.95 and wilson(108, 109)[0] < 0.95       # one miss clears from n = 110
    assert wilson(140, 142)[0] >= 0.95 and wilson(139, 141)[0] < 0.95       # two misses from n = 142
    assert wilson(111, 112)[0] >= 0.95 and wilson(110, 112)[0] < 0.95       # four a day for 28 days: one miss, not two
    lo, hi = wilson(28, 40)
    assert round(lo, 4) == 0.5457 and round(hi, 4) == 0.8193
    assert round((0.7 * 0.3 / 40) ** 0.5, 4) == 0.0725
    assert round(wilson(30, 30)[0], 4) == 0.8865 and round(wilson(27, 30)[0], 4) == 0.7438


# -- the rule file and the lock table ---------------------------------------------------------------------------

def _lf_sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def test_rule_file_hash_is_the_one_the_prereg_prints():
    text = PREREG.read_text(encoding="utf-8")
    rows = _load(LOCK_PY, "lock_hashes").parse_lock_rows(text)
    h, _ = rows["papers/darkflobi-gate/RULES_darkflobi_gate_2026_10_23.md"]
    assert h == _lf_sha256(RULES)
    assert re.search(r"RULES_darkflobi_gate_2026_10_23\.md` beside this file, sha256 \(git blob, LF\)\n`" + h, text)
    assert "@@" not in text, "a template placeholder survived"


def test_lock_table_names_the_pin_the_snapshot_the_block_and_the_plugins_subtree():
    text = PREREG.read_text(encoding="utf-8")
    lock = _load(LOCK_PY, "lock_hashes")
    rows = lock.parse_lock_rows(text)
    assert rows["pin:3437340b"][0] == PIN
    assert rows["PREREG_darkflobi_gate_2026_10_23.md#gates"][0] == GATES_SHA256
    assert re.fullmatch(r"[0-9a-f]{64}", rows["plugins-subtree:clawdbot.json"][0])
    assert re.fullmatch(r"[0-9a-f]{64}", rows["W:gate/snapshot_3437340b/"][0])
    hex_or_blank = re.compile(r"[0-9a-f]{64}|" + re.escape(lock.NOT_BUILT))
    for path in ("W:gate/config.json", "W:gate/salt", "W:gate/gate.py", "W:gate/selector.py", "W:gate/reconcile.py",
                 "W:gate/canary_shadow.py", "W:gate/gate_verify.py", "W:gate/page.py", "W:gate/patch_dist.js",
                 "W:ops/darkflobi-tweet.js", "~/.clawdbot/extensions/styxx-gate/index.js"):
        assert path in rows, path
        assert hex_or_blank.fullmatch(rows[path][0]), path
    paths = [m.group("path") for m in map(lock.ROW_RE.match, text.splitlines()) if m]
    section15 = text.split("## 15. amendments before the merge", 1)[1]
    above = text.split("## 15. amendments before the merge", 1)[0]
    above_paths = [m.group("path") for m in map(lock.ROW_RE.match, above.splitlines()) if m]
    assert len(above_paths) == len(set(above_paths)), "a path is locked twice above section 15"
    assert len(paths) >= 12
    for line in section15.splitlines():
        if line.startswith("|") and not line.startswith("| what") and not line.startswith("|---"):
            assert lock.ROW_RE.match(line), f"a section 15 row is not in the lock row format: {line}"


# every AMENDMENT file of this folder, by topic, with the number of lock rows its own lock-row table states; the
# lock-rows AMENDMENT prints the same table for the five before-D0 files it was written with, and a test below reads it
# there too; an AMENDMENT after it states its own count in its own lock-rows section
AMENDMENT_ROW_COUNTS = {"canary_replan": 2, "count_rule": 4, "lint_overclaims": 0, "lock_rows": 0, "posting_skill": 7,
                        "anchor_native": 9}
LOCK_ROWS_TABLE_TOPICS = {"canary_replan", "count_rule", "lint_overclaims", "lock_rows", "posting_skill"}
# the paths the amendments lock, together: the before-D0 move's ten files under W:gate/, the posting skill, and the
# anchor's nine plugin files (F-1: index.js, native_import.cjs and README.md in both places, and three test files)
AMENDED_PATHS = {
    "W:gate/gate.py", "W:gate/gate_verify.py", "W:gate/README.md", "W:gate/README_exhibit.md",
    "W:gate/canary_plan.py", "W:gate/tests/test_trailer_rule.py", "W:gate/tests/test_prepared_guard.py",
    "W:gate/tests/test_canary_plan.py", "W:gate/tests/test_gate.py", "W:gate/tests/test_gate_round2.py",
    "W:skills/styxx-post/SKILL.md",
    "W:gate/plugin/index.js", "~/.clawdbot/extensions/styxx-gate/index.js",
    "W:gate/plugin/native_import.cjs", "~/.clawdbot/extensions/styxx-gate/native_import.cjs",
    "W:gate/plugin/README.md", "~/.clawdbot/extensions/styxx-gate/README.md", "W:gate/plugin/tests/run_tests.mjs",
    "W:gate/plugin/tests/f1_native_deliver_probe.mjs", "W:gate/plugin/tests/fakes/import_deliver_v040.mjs",
}
AMENDMENT_NAME = re.compile(r"AMENDMENT_darkflobi_gate_([a-z0-9_]+?)_(\d{4}_\d{2}_\d{2})\.md")


def _amendment_files() -> list:
    return sorted(FOLDER.glob("AMENDMENT_darkflobi_gate_*.md"))


def test_lock_rows_agree_with_the_box_when_a_caller_names_it():
    """The PREREG's rows, then every AMENDMENT file of this folder (`check --amendment`, in name order), against the box.

    With STYXX_GATE_LOCK_ROWS=prereg the PREREG's rows alone are read: the reading of a box before the before-D0
    move, which the amendment rows then read CHANGED for the files the move is to bring."""
    w_root = os.environ.get("STYXX_GATE_W_ROOT")
    if not w_root:
        pytest.skip("set STYXX_GATE_W_ROOT to the box's W: folder to compare the lock rows with the files there")
    lock = _load(LOCK_PY, "lock_hashes")
    amendments = [] if os.environ.get("STYXX_GATE_LOCK_ROWS") == "prereg" else _amendment_files()
    rc = lock.check(PREREG, Path(w_root), Path.home(), amendments=amendments)
    assert rc == 0, "a lock row reads CHANGED against the box (see the printed rows)"


def test_each_amendment_parses_to_the_lock_rows_its_table_states():
    lock = _load(LOCK_PY, "lock_hashes")
    files = _amendment_files()
    topics = {}
    for f in files:
        m = AMENDMENT_NAME.fullmatch(f.name)
        assert m, f.name
        topics[m.group(1)] = f
    assert set(topics) == set(AMENDMENT_ROW_COUNTS), sorted(topics)
    full_digest_row = re.compile(r"^\|[^|]*\|\s*`[^`]+`\s*\|\s*`[0-9a-f]{64}`")
    for topic, f in topics.items():
        text = f.read_text(encoding="utf-8")
        lines = text.splitlines()
        form = [ln for ln in lines if lock.ROW_RE.match(ln)]
        assert len(form) == AMENDMENT_ROW_COUNTS[topic], (f.name, len(form))
        assert len(lock.parse_lock_rows(text)) == len(form), f"{f.name} names a path twice"
        for ln in lines:
            if full_digest_row.match(ln):
                assert lock.ROW_RE.match(ln), f"{f.name}: a table line with a full digest is not in the row form: {ln[:100]}"
    rows, via, _read, clashes = lock.amendment_rows(files)
    assert clashes == [], clashes
    assert set(rows) == AMENDED_PATHS, sorted(set(rows) ^ AMENDED_PATHS)
    assert via["W:gate/gate.py"] == ("AMENDMENT_darkflobi_gate_count_rule_2026_10_25.md + "
                                     "AMENDMENT_darkflobi_gate_posting_skill_2026_10_25.md")
    prereg = lock.parse_lock_rows(PREREG.read_text(encoding="utf-8"))
    # 84 for the before-D0 files, and the anchor's four new paths (F-1: the helper in both places, two test files)
    assert len(set(prereg) | set(rows)) == 88
    # the lock-rows AMENDMENT states the same counts for the five files it was written with, one row of its table each
    stated = dict(re.findall(r"^[ ]*\| `(AMENDMENT_darkflobi_gate_[a-z0-9_]+\.md)` \| (\d+) \|",
                             topics["lock_rows"].read_text(encoding="utf-8"), flags=re.M))
    assert stated == {f.name: str(AMENDMENT_ROW_COUNTS[t]) for t, f in topics.items() if t in LOCK_ROWS_TABLE_TOPICS}, stated
    # a later AMENDMENT states its own count, once, in its lock-rows section
    for t, f in topics.items():
        if t in LOCK_ROWS_TABLE_TOPICS:
            continue
        own = re.findall(r"^This file carries (\d+) rows? in `lock_hashes\.py`'s row form\.$",
                         f.read_text(encoding="utf-8"), flags=re.M)
        assert own == [str(AMENDMENT_ROW_COUNTS[t])], (f.name, own)


def test_check_reads_amendment_rows_after_the_prereg_and_exits_2_on_a_clash(tmp_path, capsys):
    lock = _load(LOCK_PY, "lock_hashes")
    w = tmp_path / "W"
    gate = w / "gate"
    (gate / "snapshot_3437340b").mkdir(parents=True)
    (gate / "snapshot_3437340b" / "x.py").write_bytes(b"print(1)\n")
    (gate / "gate.py").write_bytes(b"# gate\n")
    home = tmp_path / "home"
    home.mkdir()
    doc = tmp_path / "P.md"
    doc.write_text("## 11. lock hashes\n\n| what | path | sha256 | note |\n|---|---|---|---|\n"
                   + lock.markdown_rows(lock.rows_for_box(w, home)) + "\n", encoding="utf-8")
    assert lock.check(doc, w, home, strict=True) == 0
    n0 = len(lock.parse_lock_rows(doc.read_text(encoding="utf-8")))
    # the box moves: an edited file, a new file, and a file the PREREG froze blank
    (gate / "gate.py").write_bytes(b"# gate, amended\n")
    (gate / "plan.py").write_bytes(b"# new\n")
    (w / "skills" / "styxx-post").mkdir(parents=True)
    (w / "skills" / "styxx-post" / "SKILL.md").write_bytes(b"---\nname: styxx-post\n---\n")
    h_gate = hashlib.sha256(b"# gate, amended\n").hexdigest()
    h_plan = hashlib.sha256(b"# new\n").hexdigest()
    h_skill = hashlib.sha256(b"---\nname: styxx-post\n---\n").hexdigest()
    assert lock.check(doc, w, home) == 1, "without the amendment's row the edited file reads CHANGED"
    head = "# AMENDMENT\n\n| what | path | sha256 | note |\n|---|---|---|---|\n"
    a1 = tmp_path / "AMENDMENT_a.md"
    a1.write_text(head + f"| gate file | `W:gate/gate.py` | `{h_gate}` | before it the PREREG's row |\n"
                  f"| the posting skill | `W:skills/styxx-post/SKILL.md` | `{h_skill}` | before it not built |\n",
                  encoding="utf-8")
    a2 = tmp_path / "AMENDMENT_b.md"
    a2.write_text(head + f"| gate file | `W:gate/gate.py` | `{h_gate}` | the same digest, named again |\n"
                  f"| gate file | `W:gate/plan.py` | `{h_plan}` | new, no earlier row |\n", encoding="utf-8")
    a0 = tmp_path / "AMENDMENT_none.md"
    a0.write_text("# AMENDMENT with no lock rows\n\n| what | n |\n|---|---|\n| a table | 1 |\n", encoding="utf-8")
    capsys.readouterr()
    assert lock.check(doc, w, home, amendments=[a1]) == 0, "with the amendment's rows the box reads SAME"
    out = capsys.readouterr().out
    assert "AMENDMENT    AMENDMENT_a.md  rows 2  sha256 " in out
    assert "via=AMENDMENT_a.md" in out and f"rows {n0}," in out
    assert lock.check(doc, w, home, strict=True, amendments=[a1]) == 1, "plan.py has no row yet: UNLOCKED"
    capsys.readouterr()
    assert lock.check(doc, w, home, strict=True, amendments=[a1, a2, a0]) == 0
    out = capsys.readouterr().out
    assert "via=AMENDMENT_a.md + AMENDMENT_b.md" in out and "AMENDMENT_none.md  rows 0" in out and f"rows {n0 + 1}," in out
    assert lock.main(["check", str(doc), "--w-root", str(w), "--home", str(home), "--strict",
                      "--amendment", str(a1), "--amendment", str(a2)]) == 0
    assert lock.main(["check", str(doc), "--w-root", str(w), "--home", str(home), "--strict"]) == 1
    # a path two files name with two digests: exit 2, both files named, no row read
    a3 = tmp_path / "AMENDMENT_c.md"
    a3.write_text(head + "| gate file | `W:gate/gate.py` | `" + "0" * 64 + "` | another digest |\n", encoding="utf-8")
    capsys.readouterr()
    assert lock.check(doc, w, home, amendments=[a1, a3]) == 2
    cap = capsys.readouterr()
    assert "AMENDMENT_a.md" in cap.err and "AMENDMENT_c.md" in cap.err and "CLASH" in cap.out
    assert h_gate in cap.err and "0" * 64 in cap.err
    assert not re.search(r"^(SAME|CHANGED)", cap.out, flags=re.M), "no row is read once a clash is found"
    assert lock.main(["check", str(doc), "--w-root", str(w), "--home", str(home),
                      "--amendment", str(a1), "--amendment", str(a3)]) == 2
    # one file naming a path twice with two digests clashes too
    a4 = tmp_path / "AMENDMENT_d.md"
    a4.write_text(head + f"| gate file | `W:gate/gate.py` | `{h_gate}` | one |\n"
                  "| gate file | `W:gate/gate.py` | `" + "1" * 64 + "` | two |\n", encoding="utf-8")
    assert lock.check(doc, w, home, amendments=[a4]) == 2
    # a missing amendment file is a refusal, never a silent PREREG-only reading
    assert lock.check(doc, w, home, amendments=[tmp_path / "AMENDMENT_absent.md"]) == 2
    # an amendment's row replaces the PREREG's even where the PREREG restates the path under its section 15
    doc.write_text(doc.read_text(encoding="utf-8") + "\n## 15. amendments before the merge\n\n"
                   "| gate file | `W:gate/gate.py` | `" + "2" * 64 + "` | a later PREREG row |\n", encoding="utf-8")
    assert lock.check(doc, w, home) == 1
    assert lock.check(doc, w, home, strict=True, amendments=[a1, a2]) == 0


def test_every_verdict_has_a_printable_paragraph():
    text = PREREG.read_text(encoding="utf-8")
    printable = text.split("## 10. the printable sentences", 1)[1].split("## 11. lock hashes", 1)[0]
    for v in VERDICTS:
        assert f"`{v}`" in printable, v
    for token in ("Not run", "2026-11-05", "G-1", "G-2", "the anchor check", "the capsule", "[cf]", "media_no_rule"):
        assert token in printable, token


def test_the_population_and_the_cells_are_frozen_in_words():
    text = PREREG.read_text(encoding="utf-8")
    for token in ("anchor_channel_hash", "canary_chat_hash", "operator_dm_hash", "HMAC-SHA256", "28 calendar days",
                  "America/New_York", "pre-gate", "shadow", "live", "never pooled", "four G_C items", "4 × 28 = 112",
                  "Qwen2.5-3B-Instruct", "claude -p", "27 of 30", "9 of 15", "## 15. amendments before the merge"):
        assert token in text, token
    rules = RULES.read_text(encoding="utf-8")
    for token in ("## 3. what counts as a claim", "### 3.1 the claim shapes", "## 4. what counts as bound", "self:count",
                  "media_no_rule", "anchor_form_spoof", "gate anchor", "gateway_native",
                  "## 12. the census normalisation rule", "receipt-bearing", "| S14 ", "| S15 "):
        assert token in rules, token
    shapes = re.findall(r"^\| S(\d+) ", rules, re.M)
    assert [int(s) for s in shapes] == list(range(1, 16)), shapes
    assert rules.count("**checked**") == 5


# -- the helper scripts on synthetic input ------------------------------------------------------------------

def test_tree_digest_is_order_free_and_byte_sensitive(tmp_path):
    lock = _load(LOCK_PY, "lock_hashes")
    (tmp_path / "b").mkdir()
    (tmp_path / "a.txt").write_bytes(b"alpha\n")
    (tmp_path / "b" / "c.txt").write_bytes(b"gamma\n")
    d1, n1 = lock.tree_digest(tmp_path)
    assert n1 == 2 and re.fullmatch(r"[0-9a-f]{64}", d1)
    h_a = hashlib.sha256(b"alpha\n").hexdigest()
    h_c = hashlib.sha256(b"gamma\n").hexdigest()
    expected = hashlib.sha256((h_a + "  a.txt\n" + h_c + "  b/c.txt\n").encode("utf-8")).hexdigest()
    assert d1 == expected
    (tmp_path / "b" / "c.txt").write_bytes(b"gamma\r\n")
    assert lock.tree_digest(tmp_path)[0] != d1, "a CR byte must move the digest; nothing is normalised"
    r = subprocess.run([sys.executable, str(LOCK_PY), "tree", str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.split()[1] == "2"


def test_lock_table_and_check_on_a_synthetic_box(tmp_path):
    lock = _load(LOCK_PY, "lock_hashes")
    w = tmp_path / "W"
    gate = w / "gate"
    (gate / "snapshot_3437340b").mkdir(parents=True)
    (gate / "snapshot_3437340b" / "x.py").write_bytes(b"print(1)\n")
    (gate / "gate.py").write_bytes(b"# gate\n")
    (gate / "ledger.jsonl").write_bytes(b"{}\n")                 # live state: never locked
    (gate / "sources").mkdir()
    (gate / "sources" / "deadbeef.md").write_bytes(b"text\n")   # live state: never locked
    (gate / "__pycache__").mkdir()
    (gate / "__pycache__" / "gate.cpython-312.pyc").write_bytes(b"\x00")
    (gate / "canary_sealed").mkdir()
    (gate / "canary_sealed" / "seed.json").write_bytes(b"{}")      # a sealed folder: never listed, never a row
    (gate / "_test_stdout.txt").write_bytes(b"scratch\n")         # a scratch file by the folder's convention
    (gate / "plugin" / "tests" / "_tmp" / "case1").mkdir(parents=True)
    (gate / "plugin" / "tests" / "_tmp" / "case1" / "ledger.jsonl").write_bytes(b"{}\n")   # test scratch, any depth
    (gate / "plugin" / "tests" / ".pytest_cache").mkdir(parents=True)
    (gate / "plugin" / "tests" / ".pytest_cache" / "v").write_bytes(b"x")
    (gate / "plugin" / "tests" / "run_tests.mjs").write_bytes(b"// tests\n")             # a real file, any depth
    (gate / ".manifest_snapshot_3437340b.json").write_bytes(b"{}")                         # the snapshot's cache
    for live in ("ledger.jsonl.lock", "ledger.jsonl.lock.break", "ledger_failures.jsonl", "anchor_state.json"):
        (gate / live).write_bytes(b"{}\n")                                                  # live state: never rows
    (gate / "tmp").mkdir()
    (gate / "tmp" / "x.json").write_bytes(b"{}")                                            # the plugin's scratch
    (gate / "canary" / "keys").mkdir(parents=True)
    (gate / "canary" / "outbox").mkdir()
    (gate / "canary" / "outbox" / "item.json").write_bytes(b"{}")                          # harness working state
    (gate / "canary" / "keys" / "sworn_measurement_darkflobi_gate_canary_seed.sha256").write_bytes(b"ab  k.json\n")
    home = tmp_path / "home"
    home.mkdir()
    rows = lock.rows_for_box(w, home)
    paths = {p for _, p, _, _ in rows}
    assert "W:gate/gate.py" in paths and "W:gate/snapshot_3437340b/" in paths and "pin:3437340b" in paths
    assert "W:gate/plugin/tests/run_tests.mjs" in paths
    assert "W:gate/ledger.jsonl" not in paths and "W:gate/sources/deadbeef.md" not in paths
    assert not any("pycache" in p or "pytest_cache" in p or "/_tmp/" in p for p in paths)
    assert not any("sealed" in p for p in paths), "a sealed folder must never become a lock row"
    assert "W:gate/_test_stdout.txt" not in paths and "W:gate/.manifest_snapshot_3437340b.json" not in paths
    assert "W:gate/canary/outbox/item.json" not in paths
    for live in ("ledger.jsonl.lock", "ledger.jsonl.lock.break", "ledger_failures.jsonl", "anchor_state.json",
                 "tmp/x.json"):
        assert "W:gate/" + live not in paths, live
    by_path = {p: h for _, p, h, _ in rows}
    assert by_path["W:gate/canary/keys/sworn_measurement_darkflobi_gate_canary_seed.sha256"] == \
        hashlib.sha256(b"ab  k.json\n").hexdigest(), "the harness's public digest file is a row of its own"
    assert by_path["W:gate/canary/schedule.json"] == lock.NOT_BUILT
    assert by_path["W:gate/salt"] == lock.NOT_BUILT and by_path["W:gate/exhibit_mint.py"] == lock.NOT_BUILT
    assert by_path["W:gate/reconcile.py"] == lock.NOT_BUILT, "a plan-named file that is absent prints a blank row, not nothing"
    assert by_path["W:gate/config.json"] == lock.NOT_BUILT
    assert {p for _, p, _, _ in lock.rows_for_box(w, home, hashed=False)} == paths
    md = lock.markdown_rows(rows)
    parsed = lock.parse_lock_rows(md)
    assert parsed["W:gate/gate.py"][0] == hashlib.sha256(b"# gate\n").hexdigest()
    doc = tmp_path / "P.md"
    doc.write_text("## 11. lock hashes\n\n| what | path | sha256 | note |\n|---|---|---|---|\n" + md + "\n",
                   encoding="utf-8")
    assert lock.check(doc, w, home) == 0
    # a bytecode cache beside a locked source: printed always, a failure under --strict until it is gone
    assert lock.bytecode_paths(gate) == ["__pycache__/"]
    assert lock.check(doc, w, home, strict=True) == 1, "a cache Python would run in place of gate.py must fail"
    (gate / "__pycache__" / "gate.cpython-312.pyc").unlink()
    (gate / "__pycache__").rmdir()
    assert lock.check(doc, w, home, strict=True) == 0
    (gate / "gate.py").write_bytes(b"# edited\n")
    assert lock.check(doc, w, home) == 1, "an edited lock-hashed file must read CHANGED"
    # a later row under section 15 overrides the earlier one for the same path
    h_edited = hashlib.sha256(b"# edited\n").hexdigest()
    later = md + "\n\n## 15. amendments before the merge\n\n| gate file | `W:gate/gate.py` | `" + h_edited + "` | restated |\n"
    doc.write_text("## 11. lock hashes\n\n" + later, encoding="utf-8")
    assert lock.check(doc, w, home) == 0
    # a plan-named file built after the fill, with no later row, is BUILT SINCE: a failure
    (gate / "reconcile.py").write_bytes(b"# census\n")
    assert lock.check(doc, w, home) == 1, "a row frozen blank whose file now exists must fail until restated"
    later += "| gate file | `W:gate/reconcile.py` | `" + hashlib.sha256(b"# census\n").hexdigest() + "` | restated |\n"
    doc.write_text("## 11. lock hashes\n\n" + later, encoding="utf-8")
    assert lock.check(doc, w, home) == 0
    # a file added under W:gate/ after the fill that no row names is UNLOCKED: printed always, a failure under --strict
    (gate / "helpers").mkdir()
    (gate / "helpers" / "x.py").write_bytes(b"# imported by gate.py\n")
    assert lock.check(doc, w, home) == 0
    assert lock.check(doc, w, home, strict=True) == 1, "a file no row covers must fail the merge-time check"
    # a locked file that is gone is ABSENT: printed, a failure under --strict
    (gate / "helpers" / "x.py").unlink()
    (gate / "plugin" / "tests" / "run_tests.mjs").unlink()
    assert lock.check(doc, w, home) == 0
    assert lock.check(doc, w, home, strict=True) == 1


def test_check_re_derives_the_gates_block_row_and_the_repository_rows(tmp_path):
    lock = _load(LOCK_PY, "lock_hashes")
    # the committed PREREG itself: the block row and the rule-file row re-derive without styxx on the path
    assert lock.gates_block_sha256(PREREG.read_text(encoding="utf-8")) == GATES_SHA256
    rows = lock.parse_lock_rows(PREREG.read_text(encoding="utf-8"))
    assert rows["papers/darkflobi-gate/RULES_darkflobi_gate_2026_10_23.md"][0] == \
        hashlib.sha256(RULES.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    # a synthetic repository: the rule-file row is hashed LF whatever the checkout's line ends
    repo = tmp_path / "repo"
    folder = repo / "papers" / "x"
    folder.mkdir(parents=True)
    (folder / "RULES.md").write_bytes(b"a\r\nb\r\n")
    h_lf = hashlib.sha256(b"a\nb\n").hexdigest()
    block = '{"gates": {}, "outcomes": [], "smoke_verdict": "x"}'
    h_block = hashlib.sha256(block.encode("utf-8")).hexdigest()
    doc = folder / "P.md"
    doc.write_text("```gates\n" + block + "\n```\n\n| r | `papers/x/RULES.md` | `" + h_lf + "` | blob |\n"
                   "| g | `P.md#gates` | `" + h_block + "` | block |\n", encoding="utf-8")
    assert lock.check(doc, tmp_path / "nowhere", tmp_path / "nohome") == 0
    doc.write_text(doc.read_text(encoding="utf-8").replace(h_block, "0" * 64), encoding="utf-8")
    assert lock.check(doc, tmp_path / "nowhere", tmp_path / "nohome") == 1
    with pytest.raises(ValueError):
        lock.gates_block_sha256("```gates\n{}\n```\n```gates\n{}\n```\n")


def test_the_walker_never_lists_a_sealed_folder_or_a_pruned_one(tmp_path, monkeypatch):
    lock = _load(LOCK_PY, "lock_hashes")
    gate = tmp_path / "W" / "gate"
    for d in ("canary_sealed", "panel_secret_keys", "plugin/tests/_tmp", "node_modules/x", "exhibit_out", "canary"):
        (gate / d).mkdir(parents=True)
        (gate / d / "f.txt").write_bytes(b"x")
    (gate / "gate.py").write_bytes(b"# gate\n")
    listed = []
    real = os.scandir

    def spy(path="."):
        listed.append(os.fspath(path).replace("\\", "/"))
        return real(path)

    monkeypatch.setattr(os, "scandir", spy)
    files = [rel for rel, _, _ in lock.gate_files(gate)]
    assert files == ["gate.py"]
    assert lock.bytecode_paths(gate) == []
    assert listed, "the spy saw no listing; the walker no longer uses os.scandir"
    for bad in ("canary_sealed", "panel_secret_keys", "_tmp", "node_modules", "exhibit_out", "/canary"):
        assert not any(p.endswith(bad) or (bad + "/") in p for p in listed), (bad, listed)


def test_the_bytecode_scan_names_every_cache_a_locked_script_could_run(tmp_path):
    lock = _load(LOCK_PY, "lock_hashes")
    gate = tmp_path / "W" / "gate"
    for d in ("__pycache__", "helpers/__pycache__", "plugin/tests/_tmp/__pycache__", "snapshot_3437340b/x/__pycache__",
              "canary_sealed/__pycache__", "node_modules/m/__pycache__"):
        (gate / d).mkdir(parents=True)
        (gate / d / "m.cpython-312.pyc").write_bytes(b"\x00")
    (gate / "helpers" / "loose.pyc").write_bytes(b"\x00")     # a sourceless module beside the helpers
    (gate / "gate.py").write_bytes(b"# gate\n")
    assert lock.bytecode_paths(gate) == ["__pycache__/", "helpers/__pycache__/", "helpers/loose.pyc"]
    assert lock.bytecode_paths(tmp_path / "nowhere") == []


def test_the_walker_refuses_what_it_cannot_read_and_reads_long_paths(tmp_path, monkeypatch):
    lock = _load(LOCK_PY, "lock_hashes")
    # a path past 260 characters: written and read through the long-path prefix on Windows, counted, not skipped
    deep = tmp_path / "long"
    segs = ["s" * 48 + str(i) for i in range(6)]
    target_dir = deep.joinpath(*segs)
    os.makedirs(lock._long(target_dir), exist_ok=True)
    target = target_dir / "f.txt"
    with open(lock._long(target), "wb") as fh:
        fh.write(b"far\n")
    assert len(str(target)) > 260
    d, n = lock.tree_digest(deep)
    rel = "/".join(segs) + "/f.txt"
    assert n == 1
    assert d == hashlib.sha256((hashlib.sha256(b"far\n").hexdigest() + "  " + rel + "\n").encode("utf-8")).hexdigest()
    # a path the walker cannot stat is an error, never a silent skip
    flat = tmp_path / "flat"
    flat.mkdir()
    (flat / "ok.txt").write_bytes(b"ok\n")
    (flat / "bad.txt").write_bytes(b"bad\n")
    real_lstat = os.lstat

    def lstat(p, *a, **k):
        if os.fspath(p).endswith("bad.txt"):
            raise PermissionError("denied")
        return real_lstat(p, *a, **k)

    monkeypatch.setattr(os, "lstat", lstat)
    with pytest.raises(lock.WalkError):
        lock.tree_digest(flat)
    monkeypatch.setattr(os, "lstat", real_lstat)
    # a folder the walker cannot list is an error too
    real_scandir = os.scandir

    def scandir(p="."):
        if os.fspath(p).endswith("flat"):
            raise PermissionError("denied")
        return real_scandir(p)

    monkeypatch.setattr(os, "scandir", scandir)
    with pytest.raises(lock.WalkError):
        lock.tree_digest(flat)


def test_the_snapshot_row_is_the_digest_the_pin_gives():
    lock = _load(LOCK_PY, "lock_hashes")
    rows = lock.parse_lock_rows(PREREG.read_text(encoding="utf-8"))
    h, _ = rows["W:gate/snapshot_3437340b/"]
    assert h == SNAPSHOT_SHA256
    text = PREREG.read_text(encoding="utf-8")
    assert "tree digest over 6,164 files" in text and "6165" not in text
    have_pin = subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", PIN + "^{commit}"],
                              capture_output=True).returncode == 0
    if not have_pin:
        pytest.skip("the pin is not in this clone's object store")
    d, n = lock.pin_tree_digest(ROOT, PIN)
    assert (d, n) == (SNAPSHOT_SHA256, SNAPSHOT_FILES)


# -- the rule file against the pieces it names --------------------------------------------------------------------

def _rules_section(n: int) -> str:
    rules = RULES.read_text(encoding="utf-8")
    start = rules.index(f"\n## {n}. ")
    end = rules.index("\n## ", start + 5)
    return rules[start:end]


SELECTOR_LINE = re.compile(r"^(?P<flag>[I-])  (?P<name>\S(?:.*?\S)?) {2,}(?P<pattern>\S.*)$")


def _rules_selector():
    sec = _rules_section(10)
    fence = sec.split("```\n", 2)[1]
    out = []
    for line in fence.splitlines():
        m = SELECTOR_LINE.match(line)
        assert m, f"not a selector line: {line!r}"
        out.append((m.group("name"), m.group("flag"), m.group("pattern")))
    return out


def test_the_rules_selector_is_a_prefix_rule_and_pins_its_cases():
    rows = _rules_selector()
    assert len(rows) == 17
    rx = [re.compile(p, re.IGNORECASE if f == "I" else 0) for _, f, p in rows]

    def selected(text):
        return any(r.search(text) for r in rx)

    for text in ("the preregistration is merged", "we preregistered it", "receipted deliveries", "capsules shipped",
                 "oathkeeper", "styxxes everywhere", "a#12", "##5", "version 7.50.1", "see PR #224", "Styxx",
                 "see papers/foo", "run.py", "fathom-lab", "diff gate", "pull requests", "precision 0.23"):
        assert selected(text), text
    for text in ("version 7.50.10", "the weather is fine", "a draft for the operator", "17.5.1", "issue#x"):
        assert not selected(text), text
    sec = _rules_section(10)
    assert "**The source is `W:gate/selector.py`**" in sec and "draw_miss_sample" in sec
    assert "20261023" in sec and "\\bprereg\\b" not in sec


def test_the_rules_selector_is_the_locked_selector_py_when_the_box_is_named():
    w_root = os.environ.get("STYXX_GATE_W_ROOT")
    if not w_root:
        pytest.skip("set STYXX_GATE_W_ROOT to compare RULES section 10 with W:gate/selector.py")
    sel = _load(Path(w_root) / "gate" / "selector.py", "darkflobi_gate_selector")
    have = [(name, "I" if rx.flags & re.IGNORECASE else "-", rx.pattern) for name, rx in sel.STEM_PATTERNS.items()]
    assert _rules_selector() == have


# the codes RULES section 6 names, by class, as gate.py, the plugin and its helper spell them
TRUTH_CODES = ("sworn_failed:", "certify_ungrounded", "audit_fail:")
FORM_CODES = ("sworn_malformed:", "sworn_unresolved:", "sworn_withheld:", "receipts_missing", "receipts_unreadable",
              "text_not_utf8:", "audit_unresolved:", "audit_err:", "media_no_rule", "unbound_on_x")
PLUGIN_CODES = ("gate_unavailable", "gate_timeout", "gate_no_verdict", "gate_not_registered", "gate_not_wired",
                "gate_hook_foreign", "gate_result_unbound")
LEDGER_CODES = ("ledger_locked", "tamper", "ledger_error:")
CLASS_ROWS = {"truth": "truth refusal", "form": "form refusal", "spoof": "spoof", "lint": "lint refusal",
              "unavailable": "unavailability on the line", "internal": "not a refusal"}


def _rules_code_rows() -> dict:
    out = {}
    for line in _rules_section(6).splitlines():
        m = re.match(r"^\| \*\*(?P<cls>[^*]+)\*\*[^|]*\| (?P<codes>[^|]+) \|", line)
        if m:
            out[m.group("cls")] = m.group("codes")
    return out


# the codes a dated entry under the PREREG's section 15 restates for a RULES section 6 row (the rule file is never
# edited after the pull request opens): the drop rule as the operator signed it on 2026-10-24 (PREREG 15.5)
RESTATED_TRUTH_CODES = ("resolver_failed:",)
RESOLVER_TRUTH_REASONS = ("value_differs", "commit_not_pin_or_ancestor", "commit_unknown")
RESTATED_ROW = re.compile(r"^- \*\*RULES §6, the (?P<cls>[a-z ]+) row, restated\.\*\*(?P<body>.*?)(?=^- \*\*|^#|\n\n|\Z)",
                          re.M | re.S)


def _section15() -> str:
    return PREREG.read_text(encoding="utf-8").split("## 15. amendments before the merge", 1)[1]


def _restated_code_rows() -> dict:
    """{RULES section 6 class row: the text of the section 15 bullet that restates it}; the last entry for a row wins."""
    return {m.group("cls"): m.group("body") for m in RESTATED_ROW.finditer(_section15())}


def _code_rows_as_restated() -> dict:
    """RULES section 6's rows, each with the codes section 15 restates for it appended."""
    rows = _rules_code_rows()
    for cls, body in _restated_code_rows().items():
        assert cls in rows, f"section 15 restates a row RULES section 6 does not have: {cls}"
        rows[cls] = rows[cls] + " " + body
    return rows


def test_the_rules_codes_and_the_prereg_truth_codes_agree():
    rows = _rules_code_rows()
    assert list(rows) == ["truth refusal", "form refusal", "spoof", "lint refusal", "unavailability on the line",
                          "unavailability before any line", "the ledger could not take the line", "not a refusal"]
    for code in TRUTH_CODES:
        assert f"`{code}" in rows["truth refusal"], code
    for code in FORM_CODES:
        assert f"`{code}" in rows["form refusal"], code
    for code in PLUGIN_CODES:
        assert f"`{code}`" in rows["unavailability before any line"], code
    assert "`anchor_form_spoof`" in rows["spoof"] and "`lint_hit:" in rows["lint refusal"]
    assert "`gate_unavailable:" in rows["unavailability on the line"]
    for code in LEDGER_CODES:
        assert f"`{code}" in rows["the ledger could not take the line"], code
    truth_cell = re.findall(r"`([a-z_]+:?)", rows["truth refusal"])
    assert tuple(truth_cell) == TRUTH_CODES, truth_cell
    prereg = PREREG.read_text(encoding="utf-8")
    sec5 = prereg.split("## 5. metrics", 1)[1].split("## 6. ", 1)[0]
    n_ref = sec5.split("`n_refusals` (int)", 1)[1].split("\n- `", 1)[0]
    for c in TRUTH_CODES:
        assert f"`{c}" in n_ref, c
    for stale in ("read_from_failed", "audit_claims_fail", "read_from_malformed", "ledger_append_failed",
                  "STYXX_SEALED_DIR"):
        assert stale not in prereg and stale not in RULES.read_text(encoding="utf-8"), stale


def test_section_15_restates_the_signed_drop_rule_for_rules_4_6_7_and_prereg_5():
    sec15 = _section15()
    entry = sec15.split("### 15.5 ", 1)[1].split("\n### ", 1)[0]
    assert "> Refuse the post (Recommended)" in entry and "2026-10-24T15:51:31Z" in entry
    restated = _restated_code_rows()
    assert list(restated) == ["truth refusal"], list(restated)
    truth = restated["truth refusal"]
    order = [m.group(1) for m in re.finditer(r"`([a-z_]+:?)", truth)]
    assert tuple(order[:4]) == TRUTH_CODES + RESTATED_TRUTH_CODES, order
    for reason in RESOLVER_TRUTH_REASONS:
        assert f"`{reason}`" in truth, reason
    assert "`n_refusals`" in truth and "G_R" in truth
    # RULES section 4 restated: the full drop-reason list, with the one the rule file did not name
    s4 = entry.split("- **RULES §4,", 1)[1].split("\n- **", 1)[0]
    rules4 = " ".join(_rules_section(4).split())
    listed = re.findall(r"`([a-z_]+(?::<[^`]*>)?)`", s4.split("The reasons,", 1)[1].split(". Three of them", 1)[0])
    for reason in listed:
        assert reason == "commit_unknown" or f"`{reason}`" in rules4, reason
    assert "commit_unknown" in listed and "`commit_unknown`" not in rules4
    assert all(r in listed for r in RESOLVER_TRUTH_REASONS)
    # PREREG section 5's n_refusals restated: every truth code, the restated one among them
    s5 = entry.split("- **PREREG §5, `n_refusals`, restated.**", 1)[1].split("\n- **", 1)[0]
    for code in TRUTH_CODES + RESTATED_TRUTH_CODES:
        assert f"`{code}" in s5, code
    assert "- **RULES §7, restated.**" in entry and "- **§13's drop-rule row, as signed.**" in entry


def _gate_findings(gate_py: str) -> list:
    """(class or None, code prefix or None, source of the code expression) for every `findings.append((cls, code))`
    in gate.py, read with ast: a literal code whole, a `%`-formatted or f-string code up to its opening placeholder, and
    any other expression as its source text (the class None when it is not a literal)."""
    out = []
    for node in ast.walk(ast.parse(gate_py)):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "append"
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "findings" and node.args
                and isinstance(node.args[0], ast.Tuple) and len(node.args[0].elts) == 2):
            continue
        c, e = node.args[0].elts
        cls = c.value if isinstance(c, ast.Constant) and isinstance(c.value, str) else None
        prefix = None
        if isinstance(e, ast.Constant) and isinstance(e.value, str):
            prefix = e.value
        elif isinstance(e, ast.BinOp) and isinstance(e.op, ast.Mod) and isinstance(e.left, ast.Constant):
            prefix = e.left.value.split("%", 1)[0]
        elif isinstance(e, ast.JoinedStr) and e.values and isinstance(e.values[0], ast.Constant):
            prefix = e.values[0].value
        out.append((cls, prefix, ast.get_source_segment(gate_py, e)))
    return out


def test_the_rules_codes_are_the_codes_the_gate_emits_when_the_box_is_named():
    w_root = os.environ.get("STYXX_GATE_W_ROOT")
    if not w_root:
        pytest.skip("set STYXX_GATE_W_ROOT to read W:gate/gate.py and the plugin against RULES section 6")
    gate = Path(w_root) / "gate"
    gate_py = (gate / "gate.py").read_text(encoding="utf-8")
    plugin = (gate / "plugin" / "index.js").read_text(encoding="utf-8") + (gate / "gate.js").read_text(encoding="utf-8")
    rows = _code_rows_as_restated()
    found = _gate_findings(gate_py)
    assert len(found) >= 15, found
    computed = sorted({src for cls, prefix, src in found if prefix is None})
    # the one computed code: gate_verify's replay hands the receipts finding back, and it is one of two literals
    assert computed == ['replay["receipts_finding"]'], computed
    for literal in ("receipts_missing", "receipts_unreadable"):
        assert f'"{literal}"' in gate_py and f"`{literal}`" in rows["form refusal"], literal
    for cls, prefix, src in found:
        if prefix is None:
            continue
        if cls is None:
            # the sworn span's class is computed: FAILED is truth, every other non-HELD verdict form
            assert prefix == "sworn_" and '"truth" if s["verdict"] == "FAILED" else "form"' in gate_py, src
            continue
        assert f"`{prefix}" in rows[CLASS_ROWS[cls]], (cls, prefix)
    # every code the RULES names, or section 15 restates, is one the files can emit
    literals = {p for _, p, _ in found if p}
    for code in TRUTH_CODES + RESTATED_TRUTH_CODES + FORM_CODES:
        if code in ("sworn_failed:", "sworn_unresolved:", "sworn_withheld:"):
            assert '"sworn_%s:%s"' in gate_py, code
            continue
        assert any(lit.startswith(code) for lit in literals) or f'"{code}"' in gate_py \
            or ("PUBLIC_MARK_X = \"%s\"" % code) in gate_py, code
    for code in PLUGIN_CODES:
        assert f'"{code}"' in plugin, code
    for code in ("ledger_locked", "tamper", "unbound_marked", "internal_not_gated", "anchor_form_spoof"):
        assert f'"{code}"' in gate_py, code
    assert '"ledger_error:%s"' in gate_py
    assert ("unavailable", "gate_unavailable:") in {(c, p) for c, p, _ in found}


def test_the_restated_drop_rule_is_the_gates_when_the_box_is_named():
    w_root = os.environ.get("STYXX_GATE_W_ROOT")
    if not w_root:
        pytest.skip("set STYXX_GATE_W_ROOT to read W:gate/gate.py against the drop rule section 15 restates")
    gate_py = (Path(w_root) / "gate" / "gate.py").read_text(encoding="utf-8")
    reasons = None
    for node in ast.parse(gate_py).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "RESOLVER_TRUTH_REASONS"
                                                for t in node.targets):
            reasons = ast.literal_eval(node.value)
    assert reasons == RESOLVER_TRUTH_REASONS, reasons
    assert ("truth", "resolver_failed:") in {(c, p) for c, p, _ in _gate_findings(gate_py)}
    truth = _restated_code_rows()["truth refusal"]
    assert [r for r in re.findall(r"`([a-z_]+)`", truth) if r in reasons] == list(reasons)
    for why in ("pin_not_in_repo", "merge_base_exit%d", "cat_file_exit%d"):
        assert f'GateUnavailable("{why}")' in gate_py or f'GateUnavailable("{why}" %' in gate_py, why
    # 15.10: git's own "no such object" is `cat-file -e` exit 1, and a sign `_NUM` does not read makes a fragment
    assert '"cat-file", "-e", commit' in gate_py and "_OTHER_MINUS" in gate_py


def test_section_15_restates_the_resolver_after_the_review():
    """15.10 restates RULES section 4's fragment and git's own no; 15.11 and 15.12 say the drop rule's readings and
    component (iv)'s plainly. The RULES section 6 row restated is still the one 15.5 restates."""
    sec15 = _section15()
    entry = sec15.split("### 15.10 ", 1)[1].split("\n### ", 1)[0]
    assert "**`RULES` §4's fragment, restated.**" in entry
    for s in ("`range_malformed:fragment`", "U+2212", "`gate_unavailable:cat_file_exit<n>`", "`commit_unknown`",
              "`W:gate/gate.py`", "`W:gate/tests/test_drop_rule.py`"):
        assert s in entry, s
    for n in ("15.11", "15.12"):
        assert f"\n### {n} — 2026-10-24: " in sec15, n
    assert list(_restated_code_rows()) == ["truth refusal"]


def test_the_rules_kinds_are_the_ledger_kinds_when_the_box_is_named():
    w_root = os.environ.get("STYXX_GATE_W_ROOT")
    if not w_root:
        pytest.skip("set STYXX_GATE_W_ROOT to read W:gate/ledger.py against RULES sections 2 and 13")
    tree = ast.parse((Path(w_root) / "gate" / "ledger.py").read_text(encoding="utf-8"))
    kinds = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(x, ast.Name) and x.id == "KINDS" for x in node.targets):
            kinds = ast.literal_eval(node.value)
    assert kinds and "send" in kinds, kinds
    s2, s13 = _rules_section(2), _rules_section(13)
    for k in kinds:
        assert f"`{k}`" in s2, ("RULES section 2", k)
        assert f"`{k}`" in s13 or k in ("send",), ("RULES section 13", k)


END_FORM_NAMES = ("TRAILER_END", "MARK_END", "CAPSULE_FOOTER_END", "NARRATIVE_FOOTER_END")


def _rules_end_forms() -> list:
    sec = _rules_section(12)
    fence = sec.split("   ```\n", 2)[1]
    return [ln.strip() for ln in fence.splitlines() if ln.strip()]


def test_the_rules_census_forms_pin_their_cases():
    forms = _rules_end_forms()
    assert len(forms) == 4
    rx = [re.compile(f) for f in forms]
    footer = ("body\n\ncapsule: https://x.example/c/1.html\npost sha256 0123456789abcdef \u00b7 styxx 7.50.0 (pip install "
              "styxx==7.50.0; pin 3437340b) \u00b7 nothing signed; the words are not checked")
    desktop = footer + ("\nthe page does not print this digest yet: compare it on a desktop with python -m "
                        "styxx.capsule verify")
    assert rx[2].search(footer) and rx[2].search(desktop), "the mint's footer, with and without the desktop line"
    assert rx[0].search("t\n\n[gate: 3 of 7 numerals bound \u00b7 ledger 0042]")
    assert rx[1].search("t\n\n[unchecked: styxx unavailable]") and not rx[1].search("t [gate: x]")
    assert rx[3].search("t\n\nno number in this post; no receipt stands behind it")
    sec = " ".join(_rules_section(12).split())
    for token in ("**The source is `W:gate/reconcile.py`**", "180 s after the one before it (`JOIN_WINDOW_S`)",
                  "up to 8 in all (`JOIN_MAX`)", "600 s (`MEDIA_MATCH_WINDOW_S`)", "120 s (`LINE_AFTER_MESSAGE_SLACK_S`",
                  "`sent_not_in_export`", "the shortest join"):
        assert token in sec, token


def test_the_rules_census_rule_is_reconcile_py_when_the_box_is_named():
    w_root = os.environ.get("STYXX_GATE_W_ROOT")
    if not w_root:
        pytest.skip("set STYXX_GATE_W_ROOT to compare RULES section 12 with W:gate/reconcile.py")
    tree = ast.parse((Path(w_root) / "gate" / "reconcile.py").read_text(encoding="utf-8"))
    consts, patterns = {}, {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if isinstance(node.value, ast.Constant):
                consts[name] = node.value.value
            elif (isinstance(node.value, ast.Call) and getattr(node.value.func, "attr", "") == "compile"
                  and node.value.args):
                patterns[name] = ast.literal_eval(node.value.args[0])
    assert [patterns[n] for n in END_FORM_NAMES] == _rules_end_forms()
    for name, value in (("JOIN_WINDOW_S", 180), ("JOIN_MAX", 8), ("MEDIA_MATCH_WINDOW_S", 600),
                        ("LINE_AFTER_MESSAGE_SLACK_S", 120)):
        assert consts.get(name) == value, (name, consts.get(name))
    assert patterns["_LINE_LEAD"] in _rules_section(12)


def test_the_canary_section_names_the_cadence_and_the_catch_rule():
    sec7 = " ".join(PREREG.read_text(encoding="utf-8").split("## 7. the canaries", 1)[1].split("## 8. ", 1)[0].split())
    for token in ("cadence `s7`", "four G_C items, one each of constructions A, B, C and D", "4 × 28 = 112",
                  "Two a week of kinds E and F", "4 of E and 4 of F", "`sworn_failed:`", "`lint_hit:withdrawn/`",
                  "0 canaries planted", "--cadence s7 --window-start <D0>", "`fallbacks_to_D` 10"):
        assert token in sec7, token


def test_the_canary_section_is_the_harness_when_the_box_is_named():
    w_root = os.environ.get("STYXX_GATE_W_ROOT")
    if not w_root:
        pytest.skip("set STYXX_GATE_W_ROOT to compare PREREG section 7 with W:gate/canary_shadow.py and its schedule")
    gate = Path(w_root) / "gate"
    src = (gate / "canary_shadow.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    consts = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                consts[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    assert consts.get("PER_DAY_GC") == 4 and "s7" in consts.get("CADENCES", ())
    assert consts.get("CATCH_PREFIXES") == {"A": ("sworn_failed:",), "B": ("sworn_failed:",), "C": ("sworn_failed:",),
                                            "D": ("lint_hit:withdrawn/",)}, consts.get("CATCH_PREFIXES")
    sched = json.loads((gate / "canary" / "schedule.json").read_text(encoding="utf-8"))
    c = sched["counts"]
    if sched.get("cadence") == "s7":
        assert (c["gc_items"], c["ef_items"], c["live_items"], c["fallbacks_to_D"]) == (112, 8, 4, 0), c
    else:
        # the v0 plan for the example start that section 7 describes, until the lead re-plans at the sitting
        assert sched["window_start"] == "2026-11-05" and sched["days"] == 28
        assert (c["items"], c["gc_items"], c["ef_items"], c["live_items"], c["fallbacks_to_D"]) == (144, 112, 28, 4, 10), c
        assert sched["capacity_by_construction"]["C"] == 18


def test_the_canary_seed_digest_is_committed_and_its_location_said_once():
    key = FOLDER / "keys" / "sworn_measurement_darkflobi_gate_canary_seed.sha256"
    raw = key.read_bytes().replace(b"\r\n", b"\n")
    assert re.fullmatch(rb"[0-9a-f]{64}  sworn_measurement_darkflobi_gate_canary_seed\.json\n", raw), raw
    lock = _load(LOCK_PY, "lock_hashes")
    rows = lock.parse_lock_rows(PREREG.read_text(encoding="utf-8"))
    h, _ = rows["papers/darkflobi-gate/keys/sworn_measurement_darkflobi_gate_canary_seed.sha256"]
    assert h == hashlib.sha256(raw).hexdigest()
    sec7 = PREREG.read_text(encoding="utf-8").split("## 7. the canaries", 1)[1].split("## 8. ", 1)[0]
    assert sec7.count("`W:gate/canary_sealed/`") == 1
    assert "STYXX_SEALED_DIR" not in sec7
    w_root = os.environ.get("STYXX_GATE_W_ROOT")
    if w_root:
        box = Path(w_root) / "gate" / "canary" / "keys" / key.name
        assert box.read_bytes().replace(b"\r\n", b"\n") == raw, "the committed digest must equal the harness's copy"


def test_scan_script_prints_the_pinned_digest_and_the_row_counts(tmp_path):
    r = subprocess.run([sys.executable, str(SCAN_PY), str(PREREG), "--styxx", str(ROOT), "--scratch", str(tmp_path)],
                       capture_output=True, text=True, encoding="utf-8", timeout=300)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    out = r.stdout
    assert f"gates block sha256: {GATES_SHA256}" in out
    assert f"gates block bytes : {GATES_BYTES}" in out
    for i, (n, v) in enumerate(zip(ROW_COUNTS, VERDICTS)):
        assert f"row {i:2d} reached by {n:4d} assignments -> {v}" in out, v
    assert "rows never realised by a result dict: none" in out
    assert "two gates fences refused:" in out
    assert "smallest n with 1 miss  clearing 0.95: 110" in out


# -- words and lint ---------------------------------------------------------------------------------------------

def _unreleased_entry() -> str:
    text = CHANGELOG.read_text(encoding="utf-8")
    m = re.search(r"^## \[Unreleased\].*?(?=^## \[)", text, re.S | re.M)
    return m.group(0) if m else ""


def test_the_lab_never_prints_these_words():
    src = ROOT / "scripts" / "zenodo_publish_v7_48_0.py"
    if not src.exists():
        pytest.skip("the lab's BANNED tuple is not in this checkout")
    m = re.search(r"^BANNED = \((.*?)\)$", src.read_text(encoding="utf-8"), re.S | re.M)
    assert m, "BANNED tuple not found"
    words = re.findall(r'"([^"]+)"', m.group(1))
    assert len(words) >= 8
    banned = re.compile(r"\b(" + "|".join(re.escape(w) + "s?" for w in words) + r")\b", re.I)
    for p in (PREREG, RULES, README, LOCK_PY, SCAN_PY, TEST_FILE):
        raw = p.read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf"), f"{p.name} has a BOM"
        hits = [m.group(0) for m in banned.finditer(raw.decode("utf-8"))]
        assert not hits, f"{p.name}: {hits}"
    entry = _unreleased_entry()
    if "darkflobi" in entry:
        hits = [m.group(0) for m in banned.finditer(entry)]
        assert not hits, f"CHANGELOG [Unreleased]: {hits}"


def test_the_surface_lint_finds_nothing_in_the_folder_documents():
    lint = ROOT / "scripts" / "surface_lint.py"
    if not lint.exists():
        pytest.skip("scripts/surface_lint.py is not in this checkout")
    r = subprocess.run([sys.executable, str(lint), "--files", str(PREREG), str(RULES), str(README), "--json"],
                       capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    report = json.loads(r.stdout)
    assert report["files_read"] == 3
    assert report["hits"] == [], report["hits"]


def test_the_documents_are_lf_without_a_bom_in_the_blob():
    for p in (PREREG, RULES, README):
        raw = p.read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf")
        assert raw.decode("utf-8")


def test_rows_for_box_reads_every_file_of_the_installed_plugin_folder(tmp_path, capsys):
    """F-1: gate.py's gate_sha256 hashes every file of the installed folder, so the lock reads every file there.

    A file no row names (a stray `.new` left by a failed move) reads UNLOCKED under --strict; a named file the folder
    does not hold reads NOT BUILT in the table."""
    lock = _load(LOCK_PY, "lock_hashes")
    w = tmp_path / "W"
    (w / "gate").mkdir(parents=True)
    home = tmp_path / "home"
    inst = home / ".clawdbot" / "extensions" / "styxx-gate"
    inst.mkdir(parents=True)
    (inst / "index.js").write_bytes(b"// index\n")
    (inst / "native_import.cjs").write_bytes(b"// helper\n")
    pre = "~/.clawdbot/extensions/styxx-gate/"
    rows = {p: (h, label) for label, p, h, _n in lock.rows_for_box(w, home)}
    assert rows[pre + "index.js"] == (hashlib.sha256(b"// index\n").hexdigest(), "plugin as installed")
    assert rows[pre + "native_import.cjs"] == (hashlib.sha256(b"// helper\n").hexdigest(), "plugin as installed")
    assert rows[pre + "gate.js"][0] == lock.NOT_BUILT
    assert pre + "native_import.cjs" in lock.PLUGIN_HOME
    doc = tmp_path / "P.md"
    doc.write_text("## 11. lock hashes\n\n| what | path | sha256 | note |\n|---|---|---|---|\n"
                   + lock.markdown_rows(lock.rows_for_box(w, home)) + "\n", encoding="utf-8")
    assert lock.check(doc, w, home, strict=True) == 0
    capsys.readouterr()
    (inst / "index.js.new").write_bytes(b"// a stray\n")
    assert lock.check(doc, w, home, strict=True) != 0
    assert "UNLOCKED     " + pre + "index.js.new" in capsys.readouterr().out
