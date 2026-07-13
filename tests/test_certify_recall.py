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


# --- OATH v0.5 precision classes (autopilot cycle 38); A dropped by the battery, B/C/D/E/F ship ---

def test_v05_self_scoped_n_frees_line_siblings(tmp_path):
    # class F: "N=4" must NOT obligate the unrelated integer 8 on its own line
    assert _status(tmp_path, "Feasibility: 3 OpenAI models, N=4, 8 items/tier, single run.",
                   {"unrelated": 1}, "8") != "UNGROUNDED"


def test_v05_self_scoped_n_own_token_abstains_via_spec_rule(tmp_path):
    # class F frees LINE SIBLINGS; the glued value itself ABSTAINs -- the trailing "=" in "N="
    # matches the pre-existing comparison-operator spec rule, so a sample size reads as a spec
    # constant, not a provenance gap. (Corrects the prereg/RESULT overclaim that it stays UNGROUNDED.)
    assert _status(tmp_path, "The holdout was scored with N=5 samples per item.",
                   {"unrelated": 42}, "5") == "ABSTAIN"


def test_v05_unit_suffixed_range_abstains(tmp_path):
    # class B: "2-3B" model-size range is not a measurement
    assert _status(tmp_path, "one strong plus two weak (2-3B) members in the council.",
                   {"unrelated": 9}, "2") == "ABSTAIN"


def test_v05_at_parameter_abstains(tmp_path):
    # class D: "cosine@0.90" is a config threshold
    assert _status(tmp_path, "inconsistency is cosine@0.90 cross-sample entropy on answered items.",
                   {"unrelated": 7}, "0.90") == "ABSTAIN"


def test_v05_derived_percent_verifies_from_grounded_operands(tmp_path):
    # class E: "12.7% (19/150" verifies iff both 19 and 150 ground and 100*19/150 rounds to 12.7
    st = _status(tmp_path, "Base error rate 12.7% (19/150 wrong).",
                 {"n_incorrect": 19, "n": 150}, "12.7")
    assert st == "VERIFIED"


def test_v05_derived_percent_rejects_ungrounded_operands(tmp_path):
    # no fabricated-pair verification: if the operands do not ground, the percent does not verify
    st = _status(tmp_path, "Base error rate 12.7% (19/150 wrong).",
                 {"unrelated": 1}, "12.7")
    assert st != "VERIFIED"


def test_v05_class_a_is_dropped():
    # class A (approx-notation) was dropped by the severability procedure; the flag must stay off
    import styxx.certify as C
    assert C.V05_APPROX_NOTATION is False
    assert C.V05_SELF_SCOPED_N is True and C.V05_DERIVED_PCT is True
