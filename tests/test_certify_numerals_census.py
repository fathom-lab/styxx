# -*- coding: utf-8 -*-
"""The numeral reader v2 census: what the committed result says, re-checked against the live tree.

`papers/closed-model-frontier/oath_v15_numeral_reader_census.py` re-derives every committed certify
output under each flag configuration and writes `oath_v15_numeral_reader_census_result.json`. This
suite pins what the NOTE and the CHANGELOG say from it: under the shipped default (RULE_final 0.1) no
committed certificate re-derives differently in any compared field, no capsule stops verifying, no
charon line reads anything but what it read before, and the REPLICATIONS.md audit line re-derives; each
rule that ships on moves nothing alone; the two that ship off for what they move (the bare magnitude
arm and the form rule) are measured moving it, with no verdict class among it; under
`V15_SENTENCE_FINAL` the certificates the result lists flip, every one of them OATH-HELD to OATH-FAILED,
none of them in `KNOWN_VERDICT_DRIFT` (the flag is off, so the committed certificates still reproduce).
The two flag arms are re-derived live here on the flip documents, so the committed result cannot go
stale without this test saying so.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

# importlib, not `import styxx.certify as C`: the package attribute `styxx.certify` is the FUNCTION
# styxx/__init__ exports (from styxx.provenance), and which one an import binds depends on the order
# the suite imported things in (the earlier session's band test failed in a full run on exactly that)
import importlib  # noqa: E402

C = importlib.import_module("styxx.certify")
from styxx.corpus_audit import _resolve_receipts

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "papers" / "closed-model-frontier" / "oath_v15_numeral_reader_census_result.json"
DOC = json.loads(RESULT.read_text(encoding="utf-8"))
CFG = DOC["configurations"]


def _vclass(v) -> str:
    return str(v or "").split(",")[0].strip()


def _flags(**kw):
    saved = {k: getattr(C, k) for k in ("V15_READER_V2", *C._V15_RULE_FLAGS)}
    for k, v in kw.items():
        setattr(C, k, v)
    C._v15_reset()
    return saved


def _restore(saved):
    for k, v in saved.items():
        setattr(C, k, v)
    C._v15_reset()


SUM = DOC["summary"]["configs"]
SHIPPED_ON = [k for k in C._V15_RULE_FLAGS if getattr(C, k)]


def test_the_result_was_measured_at_the_shipped_flags_on_a_clean_tree():
    assert DOC["schema"] == "styxx-oath/numeral-reader-census/v1"
    assert DOC["shipped_rules"] == {k: bool(getattr(C, k)) for k in C._V15_RULE_FLAGS}
    assert DOC["tree_clean"] is True and len(DOC["head"]) == 40
    assert CFG["default"]["flags"]["V15_READER_V2"] is True
    assert CFG["v1"]["flags"]["V15_READER_V2"] is False
    assert DOC["population"]["resolved"] >= 230 and DOC["summary"]["certificates"] == DOC["population"]["resolved"]


def test_the_shipped_default_moves_nothing_committed():
    """The scope rule: the tree reads exactly as today under the shipped flags."""
    d, v1 = SUM["default"], SUM["v1"]
    for k in ("certificates_moved", "class_flips", "rows_added", "rows_removed", "rows_status_changed",
              "magnitude_rows", "capsules_stop_verifying", "charon_skew", "charon_drift",
              "charon_neither_same_nor_moved_verifier"):
        assert d[k] == 0, k
    assert v1["certificates_moved"] == 0 and v1["charon_neither_same_nor_moved_verifier"] == 0
    assert d["capsules_ok"] == v1["capsules_ok"] and d["capsules"] == v1["capsules"]
    assert CFG["default"]["charon"]["by_status"] == CFG["v1"]["charon"]["by_status"]
    assert [c["ok"] for c in CFG["default"]["capsules"]["rows"]] == [c["ok"] for c in CFG["v1"]["capsules"]["rows"]]


@pytest.mark.parametrize("flag", [k for k in C._V15_RULE_FLAGS if getattr(C, k)])
def test_each_rule_that_ships_on_moves_nothing_alone(flag):
    assert SUM[f"only_{flag}"]["certificates_moved"] == 0, flag


def test_what_the_flags_that_ship_off_would_move_is_measured_and_no_verdict_is_among_it():
    for name in ("explicit_list", "only_V15_MAGNITUDE_BARE", "only_V15_FORM_NOISE"):
        x = SUM[name]
        assert x["certificates_moved"] > 0 and x["charon_skew"] > 0, name
        assert x["class_flips"] == 0 and x["capsules_stop_verifying"] == 0, name
    assert SUM["only_V15_MAGNITUDE_BARE"]["rows_added"] > 0 and SUM["only_V15_FORM_NOISE"]["rows_removed"] > 0
    assert SUM["sentence_final"]["class_flips"] > 0 and SUM["sentence_final"]["capsules_stop_verifying"] > 0


def test_the_shipped_default_re_derives_the_pinned_audit_line():
    rep_text = (ROOT / "REPLICATIONS.md").read_text(encoding="utf-8")
    m = re.search(r"corpus papers: \d+ certificates \| HELD \d+  FAILED \d+  unresolved \d+  verdict-drift \d+"
                  r"  receipt-drift \d+  incomplete \d+  receipt-changed \d+", rep_text)
    assert m, "REPLICATIONS.md no longer carries the expected audit line"
    assert CFG["default"]["audit_line"].splitlines()[0] == m.group(0)
    assert CFG["v1"]["audit_line"].splitlines()[0] == m.group(0)


def test_the_sentence_final_flips_are_held_to_failed_and_none_is_a_known_drift():
    from tests.test_certificate_reproduces import KNOWN_VERDICT_DRIFT
    flips = CFG["sentence_final"]["class_flips_list"]
    assert len(flips) == CFG["sentence_final"]["class_flips"] > 0
    for f in flips:
        assert _vclass(f["v1"]) == "OATH-HELD" and _vclass(f["live"]) == "OATH-FAILED", f["doc"]
        assert f["accused"], f["doc"]
        assert f["doc"] not in KNOWN_VERDICT_DRIFT, (
            "a sentence-final flip is listed as a known drift: the flag is off, so the committed certificate "
            "still reproduces and the entry would be wrong")


@pytest.mark.parametrize("flip", CFG["sentence_final"]["class_flips_list"], ids=lambda f: Path(f["doc"]).name)
def test_each_sentence_final_flip_re_derives_live(flip):
    cp = ROOT / flip["cert"]
    doc = ROOT / flip["doc"]
    cert = json.loads(cp.read_text(encoding="utf-8"))
    receipts, missing, _ = _resolve_receipts(cp, cert, ROOT / "papers")
    if not receipts or missing:
        pytest.skip(f"{flip['doc']}: receipts do not resolve in this checkout")
    saved = _flags()
    try:
        _flags(V15_SENTENCE_FINAL=False)
        held = C.certify_doc(doc, receipts)["verdict"]
        _flags(V15_SENTENCE_FINAL=True)
        failed = C.certify_doc(doc, receipts)["verdict"]
    finally:
        _restore(saved)
    assert _vclass(held) == "OATH-HELD", (flip["doc"], held)
    assert _vclass(failed) == "OATH-FAILED", (flip["doc"], failed)
