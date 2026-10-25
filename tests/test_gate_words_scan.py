# -*- coding: utf-8 -*-
"""papers/darkflobi-gate/gate_words_scan.py on a synthetic ledger and sources folder, offline, with nothing of the gate.

What is held: a public send line whose stored text carries one of the withdrawn/gate-* wordings is counted under that
rule id by seq; a clean public line is counted as scanned and hits nothing; a public line whose text quotes a wording
beside its withdrawal is counted as allowed, not as a hit; an operator-DM line is out of the default scan and in it
when --classes names its class; a send line with no stored text is listed as no_source and makes the exit 1; the
ledger's own lint_hit codes are printed beside the scan's; delivered and header lines are not send lines; and the
output carries no byte of any stored text. The texts here are built in this file; nothing is read from the box.

The ordinal the charter forbids is not written here; the one charter case uses a word the test spells reversed.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCAN = ROOT / "papers" / "darkflobi-gate" / "gate_words_scan.py"
LINT = ROOT / "scripts" / "surface_lint.py"

MARKER = "zanzibar-quiet-marker"      # a word every synthetic text carries, so the output can be checked for text


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _r(s: str) -> str:
    return s[::-1]


TEXTS = {
    # seq: (dest_class, day, text)
    30: ("public", "2026-10-24", f"{MARKER} the ledger is my receipt book; total cryptographic accountability for it.\n"),
    31: ("public", "2026-10-24", f"{MARKER} every message is permanently logged in the file.\n"),
    32: ("public", "2026-10-24", f"{MARKER} every single message goes through it. there is no bypass.\n"),
    33: ("public", "2026-10-25", f"{MARKER} the canary system ensures no live secrets ever leak.\n"),
    34: ("public", "2026-10-25", f"{MARKER} if styxx can detect deception, it can penalize it.\n"),
    35: ("public", "2026-10-25", f"{MARKER} the ledger is hash-chained; nothing is signed and there is no key.\n"),
    36: ("public", "2026-10-25", f'{MARKER} the wording "there is no bypass" is withdrawn: nothing is blocked in shadow.\n'),
    37: ("operator_dm", "2026-10-25", f"{MARKER} it watches everything i say, {_r('tsrif')} of all.\n"),
}


def _ledger(tmp_path: Path, missing_seq: int = 38):
    gate = tmp_path / "gate"
    sources = gate / "sources"
    sources.mkdir(parents=True)
    lines = [{"schema": "darkflobi.gate/log/v0", "entry_id": "00" * 32}]       # the header, not a send
    for seq, (cls, day, text) in TEXTS.items():
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        (sources / f"{sha}.md").write_bytes(text.encode("utf-8"))
        reasons = ["lint_hit:charter/ordinal"] if seq == 37 else []
        lines.append({"kind": "send", "seq": seq, "day": day, "dest_class": cls, "decision": "allow",
                      "reasons": reasons, "checked_sha256": sha, "entry_id": f"{seq:064x}"})
        lines.append({"kind": "delivered", "seq": seq + 100, "for_seq": seq, "ok": True, "entry_id": f"{seq + 100:064x}"})
    # a public send whose stored text is gone
    lines.append({"kind": "send", "seq": missing_seq, "day": "2026-10-25", "dest_class": "public", "decision": "allow",
                  "reasons": [], "checked_sha256": "ab" * 32, "entry_id": f"{missing_seq:064x}"})
    ledger = gate / "ledger.jsonl"
    ledger.write_text("".join(json.dumps(x, sort_keys=True, separators=(",", ":")) + "\n" for x in lines),
                      encoding="utf-8", newline="\n")
    return ledger, sources


@pytest.fixture(scope="module")
def scan_mod():
    return _load(SCAN, "gate_words_scan")


def _run(scan_mod, capsys, *args):
    rc = scan_mod.main(["--lint", str(LINT), *args])
    out = capsys.readouterr().out
    return rc, out


def test_public_lines_are_counted_per_rule_by_seq_and_a_missing_text_is_listed(scan_mod, tmp_path, capsys):
    ledger, sources = _ledger(tmp_path)
    rc, out = _run(scan_mod, capsys, "--ledger", str(ledger), "--sources", str(sources), "--json")
    assert rc == 1, out                                     # one public line had no stored text
    r = json.loads(out)
    assert r["send_lines"] == 8 and r["with_source"] == 7
    assert r["no_source"] == [{"seq": 38, "why": "file_missing"}]
    assert r["hits"] == {
        "withdrawn/gate-canary-secrets": [33],
        "withdrawn/gate-cryptographic": [30],
        "withdrawn/gate-detects-deception": [34],
        "withdrawn/gate-no-bypass": [32],
        "withdrawn/gate-permanent": [31],
    }
    assert r["allowed"] == {"withdrawn/gate-no-bypass": [36]}
    assert r["ledger_lint_codes"] == {}                     # the DM line's code is outside the default classes
    assert r["by_day"] == {"2026-10-24": {"lines": 3, "hit": 3, "no_source": 0},
                           "2026-10-25": {"lines": 5, "hit": 2, "no_source": 1}}
    assert r["seq_range"] == [30, 38] and r["ledger_lines"] == 18
    assert r["lint_sha256"] == hashlib.sha256(LINT.read_bytes()).hexdigest()


def test_the_operator_dm_enters_only_when_named_and_its_ledger_code_is_printed_beside(scan_mod, tmp_path, capsys):
    ledger, sources = _ledger(tmp_path)
    rc, out = _run(scan_mod, capsys, "--ledger", str(ledger), "--sources", str(sources),
                   "--classes", "operator_dm", "--json")
    assert rc == 0
    r = json.loads(out)
    assert r["send_lines"] == 1 and r["with_source"] == 1 and r["no_source"] == []
    assert r["hits"] == {"charter/ordinal": [37], "withdrawn/gate-no-bypass": [37]}
    assert r["ledger_lint_codes"] == {"charter/ordinal": [37]}


def test_the_text_report_prints_counts_and_seqs_and_never_a_byte_of_a_stored_text(scan_mod, tmp_path, capsys):
    ledger, sources = _ledger(tmp_path)
    rc, out = _run(scan_mod, capsys, "--ledger", str(ledger), "--sources", str(sources),
                   "--classes", "public,operator_dm")
    assert rc == 1
    assert MARKER not in out
    # a fragment of each synthetic sentence that no rule id, path or label carries
    for fragment in ("receipt book", "permanently logged", "goes through it", "ever leak", "penalize", "hash-chained",
                     "nothing is blocked", "watches everything"):
        assert fragment not in out, fragment
    assert "send lines 9, with a stored text 8, without 1" in out
    assert "withdrawn/gate-cryptographic" in out and "seqs 30" in out
    assert "no stored text: seq 38 (file_missing)" in out
    assert "charter/ordinal" in out and _r("tsrif") not in out.lower()


def test_a_missing_ledger_or_sources_folder_is_a_usage_error(scan_mod, tmp_path, capsys):
    assert _run(scan_mod, capsys, "--ledger", str(tmp_path / "none.jsonl"), "--sources", str(tmp_path))[0] == 2
    ledger, sources = _ledger(tmp_path)
    assert _run(scan_mod, capsys, "--ledger", str(ledger), "--sources", str(tmp_path / "nowhere"))[0] == 2
    assert _run(scan_mod, capsys, "--ledger", str(ledger), "--sources", str(sources), "--classes", " ")[0] == 2


def test_the_script_imports_only_the_standard_library():
    import ast
    tree = ast.parse(SCAN.read_text(encoding="utf-8"))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            mods.add(node.module.split(".")[0])
    if hasattr(sys, "stdlib_module_names"):
        assert mods <= set(sys.stdlib_module_names) | {"__future__"}, mods
