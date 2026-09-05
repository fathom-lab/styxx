# -*- coding: utf-8 -*-
"""styxx.harness.junit — a test report through styxx.evidence, minted at the declared rung.

Built to papers/sworn/DESIGN_harness_adapters_2026_09_02.md. Every manifest here is verified
in-process and through the sworn CLI against a canned document; the adapter derives nothing,
withholds r1 and r2 on a report the reader could not parse, keeps failures apart from harness
errors, refuses the reserved rung, and writes LF-only.

LOAD-BEARING: test_a_report_the_agent_wrote_is_malformed_end_to_end,
test_an_unparsed_report_withholds_r1_and_r2.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from styxx import sworn
from styxx.attestation import jcs
from styxx.evidence import load_evidence
from styxx.harness import junit
from styxx.sworn import Manifest, verify

GREEN = b"""<?xml version="1.0" encoding="utf-8"?>
<testsuites name="pytest tests"><testsuite name="pytest" errors="0" failures="0" \
skipped="0" tests="3" time="0.012" timestamp="2026-09-05T00:00:00.000000" hostname="runner">
<testcase classname="tests.test_app" name="test_one" time="0.001" />
<testcase classname="tests.test_app" name="test_two" time="0.001" />
<testcase classname="tests.test_app" name="test_three" time="0.001" />
</testsuite></testsuites>
"""

# One suite of real tests and one suite that failed to load: a harness error, not a failure.
ERRORS_APART = b"""<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="ok" tests="2" failures="0" errors="0">
<testcase classname="t" name="a" /><testcase classname="t" name="b" />
</testsuite><testsuite name="broken" tests="0" failures="0" errors="1">
<error message="ImportError: no module named thing">traceback</error>
</testsuite></testsuites>
"""

UNPARSED = b"<testsuites><unclosed"


def sp(text, receipt, kind="numeric"):
    return '<sworn r="%s" k="%s">%s</sworn>' % (receipt, kind, text)


def w(tmp_path, name, data: bytes) -> Path:
    p = tmp_path / name
    p.write_bytes(data)
    return p


class TestMint:
    def test_a_green_report_mints_four_test_report_receipts_at_the_declared_rung(self, tmp_path):
        m = junit.mint(w(tmp_path, "junit.xml", GREEN), rung="L1", turn="t1")
        assert sorted(m.receipts) == ["r1", "r2", "r3", "r4"]
        assert all(e["kind_of_source"] == "test_report" and e["complete"] is True
                   for e in m.receipts.values())
        assert m.rung_status() == ("ok", "L1")
        assert m.turn == "t1"
        assert m.authored_sha256 == []
        assert "rung L1 declared by the caller, not detected" in m.harness
        assert "weak" in m.harness
        assert "authored bytes recorded: none" in m.harness
        assert "adapters, never a recorder" in m.harness

    def test_r1_and_r2_are_the_readers_counts_as_ascii(self, tmp_path):
        m = junit.mint(w(tmp_path, "junit.xml", GREEN), rung="L2")
        import base64
        assert base64.b64decode(m.receipts["r1"]["bytes"]) == b"3"
        assert base64.b64decode(m.receipts["r2"]["bytes"]) == b"0"
        assert m.rung_status() == ("ok", "L2")
        assert "weak" not in m.harness

    def test_r4_is_the_readers_output_in_canonical_form_re_derivable_from_the_same_path(self, tmp_path):
        p = w(tmp_path, "junit.xml", GREEN)
        m = junit.mint(p, rung="L1")
        import base64
        r4 = base64.b64decode(m.receipts["r4"]["bytes"])
        assert r4 == (jcs(load_evidence([str(p)])) + "\n").encode("utf-8")
        assert r4.endswith(b"\n") and b"\r" not in r4
        assert str(p) in m.receipts["r4"]["harness_note"]

    def test_a_reserved_rung_is_a_value_error_not_a_manifest(self, tmp_path):
        p = w(tmp_path, "junit.xml", GREEN)
        with pytest.raises(ValueError):
            junit.mint(p, rung="L3")
        with pytest.raises(ValueError):
            junit.mint(p, rung="l1")

    def test_a_missing_report_propagates_as_an_os_error(self, tmp_path):
        with pytest.raises(OSError):
            junit.mint(tmp_path / "absent.xml", rung="L1")


class TestVerifiesEndToEnd:
    def test_in_process_every_span_holds_and_the_rung_count_agrees(self, tmp_path):
        m = junit.mint(w(tmp_path, "junit.xml", GREEN), rung="L1")
        doc = ("\n".join([
            sp("3 tests passed", "r1"),
            sp("0 failed", "r2"),
            sp("0 harness errors", "r4#/totals/errors"),
            sp("the reader's outcome reads `PASSED`", "r4#/outcome", "quote"),
            sp("2 tests are listed? no: 3", "r4#/resolved/passed"),
        ]) + "\n").encode()
        core = verify(doc, name="d.md", manifest=m)
        verdicts = [(s["receipt"], s["verdict"], s["reason"]) for s in core["spans"]]
        assert verdicts[:4] == [("r1", "HELD", None), ("r2", "HELD", None),
                                ("r4#/totals/errors", "HELD", None), ("r4#/outcome", "HELD", None)]
        assert verdicts[4][1] == "MALFORMED" and verdicts[4][2] == "number_count"
        # A span refused at the grammar never reaches the manifest, so it carries no provenance
        # and the verifier counts it under "unresolved", not under the rung (sworn.py, R7).
        assert core["rungs"] == {"L1": 4, "unresolved": 1}
        assert all(s["provenance"]["rung"] == "L1" for s in core["spans"] if s.get("provenance"))
        assert "provenance" not in core["spans"][4]

    def test_through_the_cli_the_receipt_prints_the_rung(self, tmp_path, capsys):
        m = junit.mint(w(tmp_path, "junit.xml", GREEN), rung="L2")
        mp = m.write(tmp_path / "m.json")
        doc = tmp_path / "d.md"
        doc.write_bytes((sp("3 tests passed", "r1") + " narrative.\n").encode())
        rec = tmp_path / "rec.json"
        assert sworn.main(["verify", str(doc), "--manifest", str(mp), "--out", str(rec)]) == 0
        out = capsys.readouterr().out
        assert "SWORN-HELD  held=1 failed=0 unresolved=0 malformed=0" in out
        assert "rungs L2=1" in out
        receipt = json.loads(rec.read_text(encoding="utf-8"))
        assert receipt["rungs"] == {"L2": 1}
        assert b"\r" not in rec.read_bytes()

    def test_a_report_the_agent_wrote_is_malformed_end_to_end(self, tmp_path):
        """LOAD-BEARING. Invariant 2 by set membership: the caller hands the report bytes as
        authored, and any span over r3 is the agent swearing to itself."""
        m = junit.mint(w(tmp_path, "junit.xml", GREEN), rung="L1", authored=[GREEN])
        assert m.authored_sha256 == [hashlib.sha256(GREEN).hexdigest()]
        assert "authored bytes recorded: 1" in m.harness
        doc = (sp("the report names `tests.test_app`", "r3", "quote") + "\n").encode()
        s = verify(doc, name="d.md", manifest=m)["spans"][0]
        assert (s["verdict"], s["reason"]) == ("MALFORMED", "receipt_author_minted")
        # r1 was minted from the same bytes' counts, not from the bytes: it still resolves
        s = verify((sp("3 passed", "r1") + "\n").encode(), name="d.md", manifest=m)["spans"][0]
        assert s["verdict"] == "HELD"

    def test_an_unparsed_report_withholds_r1_and_r2(self, tmp_path):
        """LOAD-BEARING. A zero from a report that did not parse is absence printed as a number
        (styxx.evidence M7); the verifier must say it could not see, never HELD on a hollow 0."""
        m = junit.mint(w(tmp_path, "junit.xml", UNPARSED), rung="L1")
        assert sorted(m.receipts) == ["r3", "r4"]
        s = verify((sp("0 failed", "r2") + "\n").encode(), name="d.md", manifest=m)["spans"][0]
        assert (s["verdict"], s["reason"]) == ("UNRESOLVED", "manifest_id_missing")
        s = verify((sp("the reader's outcome reads `EMPTY`", "r4#/outcome", "quote") + "\n").encode(),
                   name="d.md", manifest=m)["spans"][0]
        assert s["verdict"] == "HELD"
        ev = json.loads(__import__("base64").b64decode(m.receipts["r4"]["bytes"]))
        assert ev["unparsed"] and ev["sources"] == []

    def test_failures_and_harness_errors_are_kept_apart(self, tmp_path):
        m = junit.mint(w(tmp_path, "junit.xml", ERRORS_APART), rung="L1")
        import base64
        assert base64.b64decode(m.receipts["r2"]["bytes"]) == b"0"
        core = verify(("\n".join([sp("0 failed", "r2"), sp("1 harness error", "r4#/totals/errors"),
                                  sp("2 passed", "r1")]) + "\n").encode(), name="d.md", manifest=m)
        assert [s["verdict"] for s in core["spans"]] == ["HELD", "HELD", "HELD"]


class TestCLI:
    def test_main_writes_an_lf_only_intact_manifest_and_returns_zero(self, tmp_path, capsys):
        p = w(tmp_path, "junit.xml", GREEN)
        out = tmp_path / "m.json"
        assert junit.main([str(p), "--rung", "L1", "--turn", "t", "--out", str(out)]) == 0
        assert "minted" in capsys.readouterr().out
        raw = out.read_bytes()
        assert b"\r" not in raw and raw.endswith(b"\n")
        m = Manifest.load(out)
        assert m.intact() and m.rung_status() == ("ok", "L1") and m.turn == "t"

    def test_authored_files_enter_authored_sha256_through_the_cli(self, tmp_path):
        p = w(tmp_path, "junit.xml", GREEN)
        a = w(tmp_path, "agent_wrote.txt", b"hello\n")
        out = tmp_path / "m.json"
        assert junit.main([str(p), "--rung", "L1", "--authored", str(a), "--out", str(out)]) == 0
        assert Manifest.load(out).authored_sha256 == [hashlib.sha256(b"hello\n").hexdigest()]

    def test_l3_is_refused_by_argparse_with_exit_two(self, tmp_path):
        p = w(tmp_path, "junit.xml", GREEN)
        with pytest.raises(SystemExit) as ex:
            junit.main([str(p), "--rung", "L3", "--out", str(tmp_path / "m.json")])
        assert ex.value.code == 2

    def test_a_missing_report_is_a_usage_error_exit_two_and_no_manifest(self, tmp_path, capsys):
        out = tmp_path / "m.json"
        assert junit.main([str(tmp_path / "absent.xml"), "--rung", "L1", "--out", str(out)]) == 2
        assert not out.exists()
        assert "usage" in capsys.readouterr().err

    def test_the_cli_exits_zero_whatever_the_report_says(self, tmp_path):
        p = w(tmp_path, "junit.xml", UNPARSED)
        assert junit.main([str(p), "--rung", "L1", "--out", str(tmp_path / "m.json")]) == 0
