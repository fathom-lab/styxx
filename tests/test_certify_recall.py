"""Regression tests for the OATH v0.4 trigger-recall extension (shipped autopilot cycle 25).

Locks the shipped invariant so a future edit can't silently regress it: the correlation/similarity
register obligates a number to ground ONLY when it is a fractional correlation (decimals > 0 AND
value in [-1, 1]). Ordinals, counts, API constants, whole-percents, and out-of-range values in that
register must NOT be obligated. See papers/closed-model-frontier/PREREG_oath_v04_recall_decimalguard_2026_07_04.md.
"""
import json
from pathlib import Path

from styxx.certify import certify_doc


def _status(tmp_path, line, receipt_obj, token):
    doc = tmp_path / "d.md"
    doc.write_text(f"# t\n\nsome preamble sentence to avoid line-start filters.\n\n{line}\n",
                   encoding="utf-8")
    rp = tmp_path / "r.json"
    rp.write_text(json.dumps(receipt_obj), encoding="utf-8")
    cert = certify_doc(doc, [rp])
    for e in cert["ledger"]:
        if e["token"] == token:
            return e["status"]
    return None


def test_fractional_correlation_ungrounded_when_no_receipt(tmp_path):
    # RSA 0.264 with no receipt holding it -> the correlation register obligates it -> UNGROUNDED
    assert _status(tmp_path, "The RSA-to-brain correlation was 0.264 at the final layer.",
                   {"unrelated": 1}, "0.264") == "UNGROUNDED"


def test_fractional_correlation_verified_when_grounded(tmp_path):
    assert _status(tmp_path, "The RSA-to-brain correlation was 0.264 at the final layer.",
                   {"rsa_to_brain": 0.264}, "0.264") == "VERIFIED"


def test_ordinal_under_register_word_is_spared(tmp_path):
    # "drift, stage 1" -- the ordinal 1 (decimals==0) must NOT be obligated by "drift"
    assert _status(tmp_path, "run_geometry_integrity.py handles drift at stage 1 of the pipeline.",
                   {"unrelated": 42}, "1") != "UNGROUNDED"


def test_api_constant_under_register_word_is_spared(tmp_path):
    # "entropy ... cap 20" -- the whole-integer 20 must NOT be obligated by "entropy"
    assert _status(tmp_path, "The entropy proxy is truncated because the API caps top_logprobs at 20.",
                   {"unrelated": 3}, "20") != "UNGROUNDED"


def test_out_of_range_value_in_register_is_spared(tmp_path):
    # a correlation word next to an out-of-range decimal (1.7) is not a plausible correlation
    assert _status(tmp_path, "The reliability figure quoted was 1.7 in that table.",
                   {"unrelated": 9}, "1.7") != "UNGROUNDED"


def test_auroc_register_still_obligates(tmp_path):
    # sanity: the pre-existing AUROC register is untouched by the recall extension
    assert _status(tmp_path, "The detector reached AUROC 0.884 on the held-out split.",
                   {"unrelated": 5}, "0.884") == "UNGROUNDED"
