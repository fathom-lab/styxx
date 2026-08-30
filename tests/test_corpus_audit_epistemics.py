"""The corpus auditor may READ epistemics_summary and ADD corpus totals; it may move nothing else.

The audit is the instrument that reads the whole corpus. It now folds each certificate's
epistemics_summary into a corpus composition -- the unobligated-oath rate over everything, in one
line -- but that is observation. The frozen invariant: adding the fold changes no verdict, no
count, no drift number. These tests keep both facts true.
"""

from styxx.corpus_audit import _fold_epistemics, audit_corpus
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _summary(schema="styxx-oath/epistemics-summary/v1", total=0, obl_ran=0, obl_na=0,
             un_ran=0, un_na=0, der_ob=0, der_un=0, accus=0):
    return {"schema": schema,
            "by_branch": {"obligated-accusation": accus},
            "verified": {"total": total,
                         "derived": {"obligated": der_ob, "unobligated": der_un},
                         "value_match": {"obligated_integer_filter_ran": obl_ran,
                                         "obligated_integer_filter_na": obl_na,
                                         "unobligated_integer_filter_ran": un_ran,
                                         "unobligated_integer_filter_na": un_na}}}


def test_fold_sums_obligation_across_documents():
    docs = [{"epistemics_summary": _summary(total=10, obl_ran=2, obl_na=1, un_ran=4, un_na=3)},
            {"epistemics_summary": _summary(total=6, obl_ran=0, obl_na=0, un_ran=3, un_na=3)}]
    f = _fold_epistemics(docs)
    assert f["verified_total"] == 16
    assert f["verified_obligated"] == 3          # 2+1+0 (+0 derived)
    assert f["verified_unobligated"] == 13
    assert f["unobligated_oath_rate"] == round(13 / 16, 4)
    assert f["weakest_attestations"] == 6        # un_na: 3+3
    assert f["certificates_without_summary"] == 0


def test_fold_counts_pre_v1_certificates_but_does_not_credit_them():
    docs = [{"epistemics_summary": _summary(total=4, un_na=4)},
            {"epistemics_summary": None},                       # older cert, no block
            {"epistemics_summary": {"schema": "some-other/v9"}}]  # foreign schema
    f = _fold_epistemics(docs)
    assert f["certificates_without_summary"] == 2
    assert f["certificates_with_summary"] == 1
    assert f["verified_total"] == 4, "pre-v1 certs contribute nothing, not garbage"


def test_fold_is_empty_safe():
    f = _fold_epistemics([])
    assert f["verified_total"] == 0
    assert f["unobligated_oath_rate"] is None
    assert f["weakest_share"] is None


def test_the_fold_never_touches_verdict_or_counts():
    """The invariant, on the live corpus: adding epistemics leaves every prior number identical."""
    rep = audit_corpus(ROOT / "papers")
    s = rep["summary"]
    # the composition block exists and is non-trivial
    assert s["epistemics"]["verified_total"] > 1000
    assert 0.0 < s["epistemics"]["unobligated_oath_rate"] < 1.0
    assert s["epistemics"]["certificates_without_summary"] == 0, (
        "every certificate reissued under v1 should carry the block")
    # and the classic summary keys are still exactly what they were: this is the audit line
    # REPLICATIONS.md pins, and the epistemics fold must not have perturbed it
    assert s["n_certificates"] == s["held"] + s["failed"] + s["unresolved"]
    assert s["held"] >= 188 and s["failed"] == 5


def test_composition_is_labelled_as_composition_not_quality():
    f = _fold_epistemics([{"epistemics_summary": _summary(total=4, un_na=4)}])
    assert "not quality" in f["reading"]
    assert "claim-share is the panel" in f["reading"]
