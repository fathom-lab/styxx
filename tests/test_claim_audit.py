# -*- coding: utf-8 -*-
"""Tests for styxx.audit_grounding — statistical-claim grounding."""
from styxx.claim_audit import audit_grounding


def test_grounded_decimal_and_pvalue():
    r = audit_grounding("The RSA was 0.222 and the effect held (p=0.005).",
                     {"rsa": 0.2221, "p": 0.005})
    assert r.n_unsourced == 0 and r.verdict == "ALL GROUNDED"


def test_ci_grounded():
    r = audit_grounding("ventral 0.051, 95% CI [0.030, 0.072].",
                     {"point": 0.051, "ci": [0.03, 0.072]})
    assert r.n_unsourced == 0
    assert all(n.status == "grounded" for n in r.items)


def test_percent_derived_from_ratio():
    # 0.294 / 0.339 = 86.7% -> rounds to 87%, should be DERIVED not unsourced
    r = audit_grounding("reaches 87% of the lower ceiling.",
                     {"full": 0.294, "ceiling_lo": 0.339})
    assert r.n_unsourced == 0
    assert any(n.kind == "percent" and n.status in ("grounded", "derived") for n in r.items)


def test_multiplier_derived():
    r = audit_grounding("the meaning signal is ~3× stronger in high-level cortex.",
                     {"high": 0.148, "low": 0.05})  # 0.148/0.05 = 2.96 ~ 3
    assert r.n_unsourced == 0


def test_unsourced_is_flagged():
    r = audit_grounding("the model reached an accuracy of 0.991.",
                     {"rsa": 0.222, "p": 0.005})
    assert r.n_unsourced == 1
    assert r.unsourced[0].value == 0.991
    assert r.verdict.startswith("UNSOURCED")


def test_identifiers_and_years_ignored():
    # years, arXiv ids, DOIs, section refs, version tags must NOT be extracted as claims
    r = audit_grounding("Mitchell 2008; arXiv:2501.12547; doi:10.1073/pnas.2512514122; see §2.5; gpt-2.",
                     {})
    assert r.n_total == 0, [n.raw for n in r.items]


def test_overclaim_caught_on_wrong_baseline():
    # claims 90% but the real ratio is 0.294/0.339=87% and 0.9 is nowhere -> unsourced (the failure mode we care about)
    r = audit_grounding("reaches 90% of ceiling.", {"full": 0.294, "ceiling_lo": 0.339})
    assert r.n_unsourced == 1


# --- 7.24.1 regression tests (from the post-release fuzz sweep) ---

def test_iso_date_not_a_claim():
    # YYYY-MM-DD must not produce phantom range claims from the MM-DD portion
    r = audit_grounding("Published 2026-06-30; see also 2024-01-05.", {})
    assert r.n_total == 0, [n.raw for n in r.items]


def test_scientific_notation_value_and_no_false_match():
    r = audit_grounding("the rate was 1.2e-5 (p=1.2e-5).", {"rate": 1.2e-5, "p": 1.2e-5})
    assert r.n_unsourced == 0
    # a far-off source must NOT falsely ground a tiny sci-notation value (exponent-aware tolerance)
    r2 = audit_grounding("the rate was 1.2e-5.", {"x": 0.04})
    assert r2.n_unsourced == 1


def test_numeric_dict_keys_as_sources():
    r = audit_grounding("reaches 87% of ceiling.", {0.294: "full", 0.339: "ceiling"})
    assert r.n_unsourced == 0


def test_set_sources():
    r = audit_grounding("RSA 0.222.", {0.222, 0.5})
    assert r.n_unsourced == 0


# --- 7.24.2 regression tests (dotted-identifier fuzz fix) ---

def test_dotted_identifiers_not_claims():
    # semver / IP / tool versions (3+ dotted segments) must not leak sub-parts as decimal claims
    r = audit_grounding("Python 3.11.2 on CUDA 12.4.1, host 192.168.1.1.", {})
    assert r.n_total == 0, [n.raw for n in r.items]


def test_single_dot_decimal_still_extracted():
    # the dotted-identifier mask must NOT swallow a genuine two-part decimal statistic
    r = audit_grounding("RSA was 0.222.", {})
    assert r.n_total == 1 and r.items[0].raw == "0.222"


# --- 7.24.3 regression tests (deep fuzz sweep: datetime, dash-runs, unicode ×) ---

def test_iso_datetime_not_a_claim():
    # a T-suffixed timestamp must not defeat the ISO-date mask (the MM-DD used to leak as a range)
    r = audit_grounding("logged 2026-06-30T14:30:00Z and 2026-06-30 14:30.", {})
    assert r.n_total == 0, [n.raw for n in r.items]


def test_dash_run_identifiers_not_claims():
    # 3+ dash-joined groups are never a statistical range (which is always exactly two numbers):
    # intl phone / DD-MM-YYYY dates must not yield phantom range claims
    r = audit_grounding("call +1-555-123-4567 on 30-06-2026.", {})
    assert r.n_total == 0, [n.raw for n in r.items]


def test_genuine_two_number_range_preserved():
    # the dash-run mask must NOT suppress a real range (exactly two numbers)
    r = audit_grounding("collected 12-15 samples.", {})
    assert r.n_total == 2 and all(n.kind == "range" for n in r.items)
    assert {n.raw for n in r.items} == {"12", "15"}


def test_unicode_multiplier_extracted_and_grounded():
    # "3×" (unicode ×) was silently dropped by a trailing \b; it must extract as a multiplier and derive
    r = audit_grounding("the signal is ~3× stronger.", {"high": 0.148, "low": 0.05})  # 0.148/0.05 ≈ 3
    assert any(n.kind == "multiplier" for n in r.items), [(n.raw, n.kind) for n in r.items]
    assert r.n_unsourced == 0


def test_multiplier_not_matched_in_dimension():
    # "3x2" is a dimension, not a 3-fold multiplier — must not extract a multiplier claim
    r = audit_grounding("a 3x2 grid.", {})
    assert not any(n.kind == "multiplier" for n in r.items), [(n.raw, n.kind) for n in r.items]
