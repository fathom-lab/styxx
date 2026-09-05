# -*- coding: utf-8 -*-
"""Layer 2 of a v0.1 capsule compares verdict CLASSES, not verdict strings.

The v0.13 UNCOVERED band appends ", N uncovered" to a verdict — a coverage report in the
headline, not a verdict change (styxx.corpus_audit.verdict_class, 2026-09-01). Until
2026-09-02 `verify_capsule` compared the raw strings, so six sound capsules minted before the
band read as "verdict not reproduced" under the installed verifier; the charon log recorded them
as not reproduced at ingest and the red team named the cause. This pins the repair and the
boundary it keeps: a class change is still a failure, and the string difference is reported.
"""
from __future__ import annotations

import json
from pathlib import Path

from styxx.capsule import _verdict_class, verify_capsule

ROOT = Path(__file__).resolve().parent.parent
FRONTIER = ROOT / "papers" / "closed-model-frontier"
SUFFIXED = FRONTIER / "RESULT_v14_naming_the_defects_did_not_save_it_2026_09_01.capsule.html"
CLEAN = FRONTIER / "CORPUS_STATE_2026_08_31.capsule.html"


def test_the_class_strips_the_coverage_suffix_and_nothing_else():
    assert _verdict_class("OATH-HELD, 5 uncovered") == "OATH-HELD"
    assert _verdict_class("OATH-FAILED, 12 uncovered") == "OATH-FAILED"
    assert _verdict_class("OATH-HELD") == "OATH-HELD"
    assert _verdict_class("OATH-HELD uncovered") == "OATH-HELD uncovered"     # not the band's form


def test_a_capsule_minted_before_the_uncovered_band_verifies_with_the_suffix_reported():
    rep = verify_capsule(SUFFIXED)
    assert rep["ok"] is True and rep["problems"] == []
    assert rep["verdict"] == "OATH-HELD" and rep["live_verdict"].startswith("OATH-HELD")
    if rep["live_verdict"] != rep["verdict"]:
        assert any("coverage suffix" in a for a in rep["advisory"])


def test_a_forged_class_still_fails(tmp_path):
    html = CLEAN.read_text(encoding="utf-8")
    i = html.index('id="oath-capsule">') + len('id="oath-capsule">')
    j = html.index("</script>", i)
    payload = json.loads(html[i:j])
    payload["certificate"]["verdict"] = ("OATH-FAILED" if "HELD" in payload["certificate"]["verdict"]
                                         else "OATH-HELD")
    forged = tmp_path / "forged.capsule.html"
    forged.write_text(html[:i] + json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c") + html[j:],
                      encoding="utf-8")
    rep = verify_capsule(forged)
    assert rep["ok"] is False and any("verdict not reproduced" in p for p in rep["problems"])
