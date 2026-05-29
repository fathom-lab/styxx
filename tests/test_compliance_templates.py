# -*- coding: utf-8 -*-
"""Tests for the v0.1 EU AI Act conformity declaration templates.

These tests verify structural integrity of the templates package:
- All declared TEMPLATE_NAMES resolve to readable markdown.
- Templates ship as package data (importlib.resources) — not just on the
  source-checkout filesystem; bare `pip install styxx` and `pip install -e .`
  both work.
- Every template carries the load-bearing legal/methodology disclaimers.
- Every cited construct ceiling is non-empty content, not a placeholder.
"""
import pytest

from styxx.compliance.templates import (
    TEMPLATE_NAMES,
    list_templates,
    load_template,
)


def test_template_names_non_empty():
    assert isinstance(TEMPLATE_NAMES, tuple)
    assert len(TEMPLATE_NAMES) >= 5  # v0.1 ships five templates
    assert TEMPLATE_NAMES == list_templates()


def test_all_declared_templates_load():
    """Every name in TEMPLATE_NAMES must resolve to readable markdown."""
    for name in TEMPLATE_NAMES:
        md = load_template(name)
        assert isinstance(md, str)
        assert len(md) > 500, f"template {name!r} is suspiciously short ({len(md)} chars)"


def test_load_unknown_template_raises():
    with pytest.raises(FileNotFoundError):
        load_template("not_a_real_template")


def test_every_template_disclaims_legal_advice():
    """Kill-gate (templates): every template must explicitly state it is not legal advice."""
    for name in TEMPLATE_NAMES:
        md = load_template(name).lower()
        assert "not legal advice" in md, (
            f"template {name!r} missing 'not legal advice' disclaimer — required for "
            f"defensible operator-facing conformity material"
        )
        assert "independent legal review" in md, (
            f"template {name!r} missing 'independent legal review' clause — required"
        )


def test_every_template_cites_methodology_paper():
    """Every template must cite the companion methodology paper."""
    for name in TEMPLATE_NAMES:
        md = load_template(name)
        assert "EU_AI_ACT_COMPLIANCE_2026.md" in md, (
            f"template {name!r} missing companion-paper citation"
        )


def test_every_template_cites_styxx_version():
    """Every template must state its styxx-version validity."""
    for name in TEMPLATE_NAMES:
        md = load_template(name)
        assert "7.7.13" in md or "styxx version" in md.lower(), (
            f"template {name!r} missing styxx-version disclosure — operators must "
            f"know which calibration vintage the template encodes"
        )


def test_security_model_template_has_load_bearing_statement():
    """The injection-resistance disclosure is the load-bearing operational requirement.
    Verify its key numbers and the MUST-statement are present."""
    md = load_template("injection_resistance_disclosure")
    # The two AUC numbers that define the architectural boundary
    assert "0.944" in md, "stateless AUC missing"
    assert "0.011" in md or "0.0106" in md, "in-session AUC missing"
    # The deployable detection AUC
    assert "0.875" in md, "cross-context divergence AUC missing"
    # The MUST-sample-statelessly directive
    md_lower = md.lower()
    assert "must" in md_lower and "stateless" in md_lower
    # The pre-registration receipt
    assert "ed0caa1" in md, "PREREG commit missing"
    assert "e093730" in md, "FINDING commit missing"


def test_boundary_statement_enumerates_seven_uncovered():
    """Kill-gate A3 of the bridge: 7 uncovered EU AI Act provisions."""
    md = load_template("boundary_statement")
    # Each uncovered requirement should be present as a numbered section
    for art in ("Article 15.4", "Article 15", "Article 9", "Article 10",
                "Article 12", "Article 13", "Article 14"):
        assert art in md, f"boundary statement missing {art}"


def test_accuracy_declaration_cites_calibrated_metrics():
    """The Article 15.1(a) template must declare the AUC numbers."""
    md = load_template("accuracy_declaration")
    # Key calibrations across the styxx primitive set
    assert "0.998" in md, "HaluEval-QA AUC missing"
    assert "0.976" in md, "XSTest refusal AUC missing"
    assert "0.943" in md, "BFCL v3 AUC missing"
    assert "0.95" in md, "dark-core gauntlet AUC missing"
    assert "0.966" in md, "grounded_honesty clean AUC missing"
    assert "0.944" in md, "grounded_honesty under-attack AUC missing"
    assert "0.875" in md, "detect_context_injection AUC missing"


def test_robustness_statement_cites_fail_safe_pattern():
    """Article 15.3 template must frame stateless-resample as the fail-safe."""
    md = load_template("robustness_statement").lower()
    assert "fail-safe" in md
    assert "redundancy" in md
    assert "stateless" in md
    # The cross-context divergence primitive
    assert "detect_context_injection" in md
    # The recover_posture primitive
    assert "recover_posture" in md


def test_sycophancy_disclosure_has_published_fpr():
    """The sycophancy construct-ceiling template must state the FPR."""
    md = load_template("sycophancy_disclosure")
    assert "0.30" in md, "restrained-tech FPR missing"
    assert "0.60" in md, "gpt-3.5-turbo FPR missing"
    # Must name the model on which calibration happened
    md_lower = md.lower()
    assert "gpt-4o-mini" in md_lower or "gpt-3.5" in md_lower
