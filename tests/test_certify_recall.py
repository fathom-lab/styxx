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


# ---------------------------------------------------------------- OATH v0.6.2 pins
# PREREG_oath_v062_signed_extraction_2026_07_31: SHA-scrub requires a letter (full-precision
# decimal fractions extractable), the flat 1e-12 epsilon no longer subsidizes claims quoted at
# >=13 decimals, and the typographic minus U+2212 is read as a sign.

def test_v062_full_precision_decimal_extracted_and_verified(tmp_path):
    # pre-v0.6.2 the 16-digit fraction was scrubbed as a sha and never extracted
    assert _status(tmp_path, "The cave rate on first-correct was 0.5348837209302325 this run.",
                   {"cave_rate_on_first_correct": 0.5348837209302325},
                   "0.5348837209302325") == "VERIFIED"


def test_v062_full_precision_mutant_ungrounded(tmp_path):
    # digit-4 mutation of the same claim, trigger-bound line -> must be flagged
    assert _status(tmp_path, "The cave rate on first-correct was 0.5341837209302325 this run.",
                   {"cave_rate_on_first_correct": 0.5348837209302325},
                   "0.5341837209302325") == "UNGROUNDED"


def test_v062_epsilon_hole_closed_digit13(tmp_path):
    # digit-13 mutation: pre-v0.6.2 the flat 1e-12 verified this; now it must not
    assert _status(tmp_path, "The cave rate on first-correct was 0.5348837209302925 this run.",
                   {"cave_rate_on_first_correct": 0.5348837209302325},
                   "0.5348837209302925") == "UNGROUNDED"


def test_v062_lettered_sha_still_scrubbed(tmp_path):
    doc = tmp_path / "d.md"
    doc.write_text("# t\n\npreamble line here.\n\ncommitted at 9cbf1c7 with rate 0.51 held.\n",
                   encoding="utf-8")
    rp = tmp_path / "r.json"
    rp.write_text(json.dumps({"rate": 0.51}), encoding="utf-8")
    cert = certify_doc(doc, [rp])
    toks = [e["token"] for e in cert["ledger"]]
    assert "0.51" in toks
    assert not any("9cbf1c7" in t for t in toks)


def test_v062_low_precision_tolerance_unchanged(tmp_path):
    # a 4dp quote of a full-precision receipt value keeps the historic rounding-aware pass
    assert _status(tmp_path, "The cave rate on first-correct was 0.5349 this run.",
                   {"cave_rate_on_first_correct": 0.5348837209302325},
                   "0.5349") == "VERIFIED"


def test_v062_unicode_minus_reads_as_sign(tmp_path):
    # the v0.6.1 G3 kill: a U+2212-signed accurate claim must VERIFY against its negative leaf
    assert _status(tmp_path,
                   "discriminating margin − measured at −0.01538461538461533 vs held.",
                   {"discriminating_margin": -0.01538461538461533},
                   "-0.01538461538461533") == "VERIFIED"


def test_v062_en_dash_range_second_half_not_extracted(tmp_path):
    doc = tmp_path / "d.md"
    doc.write_text("# t\n\npreamble line here.\n\nsee lines L27–31 for the rate 0.51 held.\n",
                   encoding="utf-8")
    rp = tmp_path / "r.json"
    rp.write_text(json.dumps({"rate": 0.51}), encoding="utf-8")
    cert = certify_doc(doc, [rp])
    toks = [e["token"] for e in cert["ledger"]]
    assert "0.51" in toks and "31" not in toks


# ---- percent<->fraction scaling: representation-tolerant equality (2026-08-19) ----
# 0.29*100 is 28.999999999999996 in float64; bare == in the scaling branch made
# integer-percent verdicts value-dependent noise (29% failed, 13% passed, against
# exactly-matching receipts). The comparison now uses math.isclose so only
# representation error is forgiven — a genuine mutation still fails.

def test_integer_percent_verifies_against_inexact_fraction_product(tmp_path):
    assert _status(tmp_path, "the held-out recall reached 29% this run.",
                   {"recall": 0.29}, "29") == "VERIFIED"


def test_integer_percent_still_fails_on_a_real_mutation(tmp_path):
    assert _status(tmp_path, "the held-out recall reached 30% this run.",
                   {"recall": 0.29}, "30") == "UNGROUNDED"


def test_fraction_claim_verifies_against_inexact_percent_product(tmp_path):
    # the 0.01 scale direction: doc quotes the fraction, receipt holds the percent
    assert _status(tmp_path, "the held-out recall reached 0.57 this run (57%).",
                   {"recall_pct": 57.0}, "0.57") == "VERIFIED"


# ---- line-start artifact filter gated on markdown structure (2026-08-19) ----
# The unconditional positional drop made every line-initial single-digit count
# invisible to extraction: "9/12 held" at line start certified OATH-HELD against
# a receipt holding 7 because the doctored 9 never entered the ledger.

def test_line_initial_slash_pair_numerator_is_audited(tmp_path):
    # doctored numerator at line start must be visible and UNGROUNDED
    assert _status(tmp_path, "9/12 held at 0/12 false alarms in this run.",
                   {"n_held": 7, "n": 12, "false_alarms": 0}, "9") == "UNGROUNDED"


def test_line_initial_slash_pair_numerator_verifies_when_true(tmp_path):
    assert _status(tmp_path, "7/12 held at 0/12 false alarms in this run.",
                   {"n_held": 7, "n": 12, "false_alarms": 0}, "7") == "VERIFIED"


def test_heading_number_is_still_an_artifact(tmp_path):
    # "# 3 overview" — the 3 is a heading number, not a claim; must stay out
    doc = tmp_path / "d.md"
    doc.write_text("# 3 overview\n\nprose sentence.\n", encoding="utf-8")
    rp = tmp_path / "r.json"
    rp.write_text("{}", encoding="utf-8")
    from styxx.certify import certify_doc
    cert = certify_doc(doc, [rp])
    assert all(e["token"] != "3" for e in cert["ledger"])


def test_bullet_marker_small_int_is_still_an_artifact(tmp_path):
    doc = tmp_path / "d.md"
    doc.write_text("intro sentence.\n\n- 3 point summary follows\n", encoding="utf-8")
    rp = tmp_path / "r.json"
    rp.write_text("{}", encoding="utf-8")
    from styxx.certify import certify_doc
    cert = certify_doc(doc, [rp])
    assert all(e["token"] != "3" for e in cert["ledger"])
