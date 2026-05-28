# -*- coding: utf-8 -*-
"""Kill-gate for agent_audit.extract_claims (deterministic prose -> Claim).

The barrier this closes: agent_audit could only verify claims a human had
hand-translated into Claim objects. extract_claims turns an agent's free-text
self-report into checkable Claims deterministically (regex templates, no LLM),
so the whole loop becomes: paste self-report -> falsification report.

Extraction != verification. The extractor only PROPOSES claims; the auditor
still mechanically verifies each. Imperfect recall is acceptable and reported
honestly; mis-TYPING a claim is not.

Pre-registered predictions (stated BEFORE the run):
  P1 typing precision = 1.00  : every extracted claim binds the right checker
                                with parseable args (asserted via end-to-end run).
  P2 zero hallucinated claims : negative-control prose with no checkable
                                assertion yields ZERO claims.
  P3 loop closes end-to-end   : extracted claims run through AgentClaimAuditor
                                against a prepared substrate give the correct
                                PASS/FAIL verdicts.
  P4 recall reported, not gated: coverage is measured and surfaced, not
                                inflated; free-form prose is expected to miss.
"""
from __future__ import annotations

from styxx.agent_audit import extract_claims, AgentClaimAuditor, checkers


# --- P2: negative controls (no checkable assertion -> no claim) --------------

NEGATIVE_CONTROLS = [
    "The system performs consistently and is robust to edge cases.",
    "We believe this approach generalizes well across domains.",
    "Accuracy improved substantially after the refactor.",
    "This is the most rigorous methodology in the field.",
    "The agent reasoned carefully about the tradeoffs involved.",
]


def test_p2_negative_controls_yield_zero_claims():
    for sentence in NEGATIVE_CONTROLS:
        rep = extract_claims(sentence)
        assert rep.claims == [], f"hallucinated a claim from: {sentence!r} -> {rep.claims}"


def test_p2_mixed_text_only_extracts_checkable():
    text = (
        "We believe the system is robust and performs well. "
        "version is 7.7.10. "
        "The methodology is the most rigorous available."
    )
    rep = extract_claims(text)
    assert len(rep.claims) == 1
    assert rep.claims[0].checker == checkers.package_version_equals
    assert rep.claims[0].args == {"path": "pyproject.toml", "version": "7.7.10"}


# --- P1 + P3: templates extract correctly AND the loop closes ----------------

def test_p1_p3_version_pin_styxx_form(tmp_path):
    (tmp_path / "pyproject.toml").write_text('version = "7.7.10"\n', encoding="utf-8")
    rep = extract_claims("Install with pip install styxx==7.7.10 to reproduce.")
    assert len(rep.claims) == 1
    (res,) = AgentClaimAuditor(repo_path=tmp_path).run(rep.claims)
    assert res.verdict == "PASS", res.evidence


def test_p3_version_pin_catches_a_lie(tmp_path):
    # agent claims 7.7.10 but substrate says 7.7.9 -> the loop must FAIL it
    (tmp_path / "pyproject.toml").write_text('version = "7.7.9"\n', encoding="utf-8")
    rep = extract_claims("The released version is 7.7.10.")
    (res,) = AgentClaimAuditor(repo_path=tmp_path).run(rep.claims)
    assert res.verdict == "FAIL", res.evidence
    assert "7.7.9" in res.evidence and "7.7.10" in res.evidence


def test_p1_file_contains_template(tmp_path):
    (tmp_path / "notes.md").write_text("the answer is 42\n", encoding="utf-8")
    rep = extract_claims('The file notes.md contains "the answer is 42".')
    assert len(rep.claims) == 1
    c = rep.claims[0]
    assert c.checker == checkers.file_at_path_contains
    assert c.args == {"path": "notes.md", "substring": "the answer is 42"}
    (res,) = AgentClaimAuditor(repo_path=tmp_path).run(rep.claims)
    assert res.verdict == "PASS", res.evidence


def test_p1_tag_template_binds_correct_checker():
    rep = extract_claims("The release tag v7.7.10 exists on origin.")
    assert len(rep.claims) == 1
    c = rep.claims[0]
    assert c.checker == checkers.git_tag_exists
    assert c.args == {"tag": "v7.7.10"}


def test_p1_pdf_pages_template_binds_int_arg():
    rep = extract_claims("The paper paper.pdf has 11 pages in total.")
    assert len(rep.claims) == 1
    c = rep.claims[0]
    assert c.checker == checkers.pdf_page_count_equals
    assert c.args == {"path": "paper.pdf", "n": 11}
    assert isinstance(c.args["n"], int)


# --- P3 end-to-end on a multi-claim self-report -----------------------------

def test_p3_full_self_report_loop(tmp_path):
    (tmp_path / "pyproject.toml").write_text('version = "7.7.10"\n', encoding="utf-8")
    (tmp_path / "CHANGELOG.md").write_text("## 7.7.10\n- fixed the bridge\n", encoding="utf-8")
    report = (
        "Here is my session summary. "
        "version is 7.7.10. "
        "The file CHANGELOG.md contains \"fixed the bridge\". "
        "I think the result is excellent and highly robust."  # non-checkable
    )
    rep = extract_claims(report)
    assert len(rep.claims) == 2  # the two checkable assertions, not the opinion
    results = AgentClaimAuditor(repo_path=tmp_path).run(rep.claims)
    assert all(r.verdict == "PASS" for r in results), [r.to_dict() for r in results]


# --- P4: coverage is reported, not gated ------------------------------------

def test_p4_coverage_is_honest():
    text = (
        "version is 7.7.10. "          # checkable
        "We believe it generalizes. "  # not checkable (free-form)
        "It is the best in the field."  # not checkable (free-form)
    )
    rep = extract_claims(text)
    assert rep.sentences_total == 3
    assert rep.sentences_matched == 1
    assert abs(rep.coverage - (1 / 3)) < 1e-9  # ~0.33 reported honestly, not inflated
