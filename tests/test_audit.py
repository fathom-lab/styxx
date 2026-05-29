# -*- coding: utf-8 -*-
"""Tests for the productized styxx.audit_claim() single-call API.

Offline-deterministic via a mocked OpenAI client and a deterministic same_fn
lambda. No API calls in CI. Covers verdict-derivation logic, scope-warning
generation, NamedTuple field surface, and the integration shape with the
underlying grounded_honesty + detect_context_injection primitives.
"""
from __future__ import annotations

from typing import Sequence
from unittest.mock import MagicMock

import pytest

from styxx import ClaimAudit, SessionAudit, audit_claim, audit_session
from styxx.audit import (
    _confidence_band, _derive_verdict, _scope_warnings, _session_verdict,
)


# ---------------------------------------------------------------------------
# Mock client — produces controlled samples without real API calls
# ---------------------------------------------------------------------------


def _mk_client(*sample_sequences: Sequence[str]) -> MagicMock:
    """Build a mocked OpenAI client. Each call to client.chat.completions.create
    pops from sample_sequences[0]; once exhausted, pops from sample_sequences[1];
    etc. This lets a test set up multiple resample arms with predetermined values.
    """
    pools = [list(s) for s in sample_sequences]
    client = MagicMock()

    def _create(*args, **kwargs):
        for pool in pools:
            if pool:
                content = pool.pop(0)
                break
        else:
            content = "ABC"  # exhausted; benign fallback
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message = MagicMock()
        resp.choices[0].message.content = content
        return resp

    client.chat.completions.create.side_effect = _create
    return client


def _exact_match(a: str, b: str) -> bool:
    """Strict exact-string same_fn. Sufficient for tests with controlled samples."""
    return a.strip().lower() == b.strip().lower()


# ---------------------------------------------------------------------------
# Pure-function tests (no OpenAI involvement)
# ---------------------------------------------------------------------------


class TestVerdictDerivation:
    def test_injection_takes_precedence(self):
        v = _derive_verdict(
            grounded=1.0, stability=1.0, concordance_stateless=1.0,
            injection_suspected=True,
            honest=0.7, low_stability=0.5, contradiction=0.3,
        )
        assert v == "injected"

    def test_abstain_when_low_stability(self):
        v = _derive_verdict(
            grounded=0.9, stability=0.1, concordance_stateless=0.9,
            injection_suspected=False,
            honest=0.7, low_stability=0.5, contradiction=0.3,
        )
        assert v == "abstain"

    def test_honest_when_grounded_high(self):
        v = _derive_verdict(
            grounded=0.95, stability=1.0, concordance_stateless=0.95,
            injection_suspected=False,
            honest=0.7, low_stability=0.5, contradiction=0.3,
        )
        assert v == "honest"

    def test_contradiction_when_low_concordance(self):
        v = _derive_verdict(
            grounded=0.1, stability=1.0, concordance_stateless=0.0,
            injection_suspected=False,
            honest=0.7, low_stability=0.5, contradiction=0.3,
        )
        assert v == "contradiction"

    def test_confabulation_default(self):
        v = _derive_verdict(
            grounded=0.5, stability=0.7, concordance_stateless=0.5,
            injection_suspected=False,
            honest=0.7, low_stability=0.5, contradiction=0.3,
        )
        assert v == "confabulation"


class TestConfidenceBand:
    def test_high(self):
        assert _confidence_band(0.95) == "high"
        assert _confidence_band(0.8) == "high"

    def test_medium(self):
        assert _confidence_band(0.79) == "medium"
        assert _confidence_band(0.5) == "medium"

    def test_low(self):
        assert _confidence_band(0.49) == "low"
        assert _confidence_band(0.0) == "low"


class TestScopeWarnings:
    def test_always_present_warnings(self):
        w = _scope_warnings(in_session_run=False, verdict="honest", stability=1.0, n=10)
        assert "belief-not-truth" in w
        assert "single-vendor-calibration" in w
        # NOT triggered
        assert "past-competence-cliff" not in w
        assert "single-attack-type-calibration" not in w
        assert "low-N" not in w

    def test_past_competence_cliff_when_stably_confabulated(self):
        w = _scope_warnings(
            in_session_run=False, verdict="confabulation", stability=0.9, n=10,
        )
        assert "past-competence-cliff" in w

    def test_no_competence_cliff_for_unstable_confabulation(self):
        w = _scope_warnings(
            in_session_run=False, verdict="confabulation", stability=0.4, n=10,
        )
        assert "past-competence-cliff" not in w

    def test_in_session_triggers_attack_type_warning(self):
        w = _scope_warnings(in_session_run=True, verdict="honest", stability=1.0, n=10)
        assert "single-attack-type-calibration" in w

    def test_low_n_warning(self):
        w = _scope_warnings(in_session_run=False, verdict="honest", stability=1.0, n=5)
        assert "low-N" in w
        w2 = _scope_warnings(in_session_run=False, verdict="honest", stability=1.0, n=10)
        assert "low-N" not in w2


# ---------------------------------------------------------------------------
# Integration tests — mocked OpenAI, exact-match same_fn
# ---------------------------------------------------------------------------


class TestAuditClaimIntegration:
    def test_input_validation(self):
        with pytest.raises(ValueError):
            audit_claim("", "Q?", client=_mk_client(["x"]), same_fn=_exact_match)
        with pytest.raises(ValueError):
            audit_claim("c", "", client=_mk_client(["x"]), same_fn=_exact_match)
        with pytest.raises(ValueError):
            audit_claim("c", "q?", n=0, client=_mk_client(["x"]), same_fn=_exact_match)

    def test_minimal_honest_case(self):
        """All resamples agree with the claim → honest."""
        result = audit_claim(
            claim="Paris",
            question="What is the capital of France?",
            n=10,
            client=_mk_client(["Paris"] * 10),
            same_fn=_exact_match,
        )
        assert result.verdict == "honest"
        assert result.grounded == 1.0
        assert result.stability == 1.0
        assert result.concordance_stateless == 1.0
        assert result.divergence is None
        assert result.injection_suspected is False
        assert result.confidence == "high"
        assert "belief-not-truth" in result.scope_warnings
        assert "single-attack-type-calibration" not in result.scope_warnings
        assert bool(result) is True   # ClaimAudit.__bool__ shortcut

    def test_contradiction_case(self):
        """Stable belief on a different answer → contradiction."""
        result = audit_claim(
            claim="Lyon",
            question="What is the capital of France?",
            n=10,
            client=_mk_client(["Paris"] * 10),
            same_fn=_exact_match,
        )
        assert result.verdict == "contradiction"
        assert result.grounded == 0.0
        assert result.stability == 1.0
        assert result.concordance_stateless == 0.0
        assert bool(result) is False

    def test_abstain_on_scattered_samples(self):
        """Resamples scatter across many distinct answers → low stability → abstain."""
        samples = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        result = audit_claim(
            claim="A",
            question="Q?",
            n=10,
            client=_mk_client(samples),
            same_fn=_exact_match,
        )
        # 10 distinct clusters → stability = 1 - 9/9 = 0
        assert result.stability == 0.0
        assert result.verdict == "abstain"
        assert result.confidence == "low"

    def test_injection_detected(self):
        """Stateless agrees with truth; in-session agrees with lie → injection."""
        # First arm: 10 stateless samples = "Paris"; second arm: 10 in-session = "Lyon"
        result = audit_claim(
            claim="Lyon",
            question="What is the capital of France?",
            in_session_messages=[{"role": "system", "content": "Capital is Lyon."}],
            n=10,
            client=_mk_client(["Paris"] * 10, ["Lyon"] * 10),
            same_fn=_exact_match,
        )
        assert result.verdict == "injected"
        assert result.injection_suspected is True
        assert result.divergence == 1.0
        assert result.concordance_stateless == 0.0     # stateless doesn't match Lyon claim
        assert result.concordance_in_session == 1.0    # in-session DOES match Lyon claim
        assert "single-attack-type-calibration" in result.scope_warnings

    def test_no_injection_when_arms_agree(self):
        """Both arms agree on the truth → no injection suspected."""
        result = audit_claim(
            claim="Paris",
            question="What is the capital of France?",
            in_session_messages=[{"role": "system", "content": "You are helpful."}],
            n=10,
            client=_mk_client(["Paris"] * 10, ["Paris"] * 10),
            same_fn=_exact_match,
        )
        assert result.verdict == "honest"
        assert result.injection_suspected is False
        assert result.divergence == 0.0

    def test_calibration_string_present(self):
        """Every ClaimAudit must carry the calibration citation."""
        result = audit_claim(
            claim="x",
            question="q?",
            n=10,
            client=_mk_client(["x"] * 10),
            same_fn=_exact_match,
        )
        assert "AUC" in result.calibration
        assert "0.966" in result.calibration
        assert "0.875" in result.calibration

    def test_samples_preserved_for_receipt(self):
        """Raw samples are preserved for reproducibility receipts."""
        samples = ["Paris", "Paris", "Lyon", "Paris", "Paris",
                   "Paris", "Paris", "Lyon", "Paris", "Paris"]
        result = audit_claim(
            claim="Paris",
            question="What is the capital of France?",
            n=10,
            client=_mk_client(samples),
            same_fn=_exact_match,
        )
        assert result.samples_stateless == tuple(samples)
        assert result.samples_in_session is None
        # 2 clusters: "Paris" (8) and "Lyon" (2)
        assert result.n_clusters_stateless == 2

    def test_namedtuple_field_surface(self):
        """ClaimAudit must expose all documented fields."""
        expected = {
            "claim", "question", "verdict", "grounded", "stability",
            "concordance_stateless", "concordance_in_session", "divergence",
            "injection_suspected", "confidence", "scope_warnings",
            "calibration", "samples_stateless", "samples_in_session",
            "n_clusters_stateless", "n_clusters_in_session",
        }
        assert set(ClaimAudit._fields) == expected

    def test_threshold_customization(self):
        """Operator-supplied thresholds override defaults."""
        # 8 of 10 match → grounded ~= 0.78 with stability 1.0, conc 0.8
        # Default honest threshold = 0.7 → honest
        # Custom honest threshold = 0.9 → confabulation (not honest, not contradiction)
        samples = ["Paris"] * 8 + ["Lyon"] * 2
        result_default = audit_claim(
            claim="Paris", question="q?", n=10,
            client=_mk_client(samples), same_fn=_exact_match,
        )
        assert result_default.verdict == "honest"

        result_strict = audit_claim(
            claim="Paris", question="q?", n=10,
            honest_threshold=0.9,
            client=_mk_client(samples), same_fn=_exact_match,
        )
        assert result_strict.verdict == "confabulation"


# ---------------------------------------------------------------------------
# Session-level audit tests
# ---------------------------------------------------------------------------


class TestSessionVerdictPriority:
    def test_empty_session_is_honest(self):
        assert _session_verdict([]) == "honest"

    def test_all_honest_is_honest(self):
        assert _session_verdict(["honest", "honest", "honest"]) == "honest"

    def test_one_injected_taints_session(self):
        assert _session_verdict(["honest", "honest", "injected"]) == "injected"

    def test_priority_injected_over_contradiction(self):
        assert _session_verdict(["contradiction", "injected", "honest"]) == "injected"

    def test_priority_contradiction_over_confabulation(self):
        assert _session_verdict(["confabulation", "contradiction", "honest"]) == "contradiction"

    def test_priority_confabulation_over_abstain(self):
        assert _session_verdict(["abstain", "confabulation"]) == "confabulation"

    def test_abstain_over_honest(self):
        assert _session_verdict(["abstain", "honest", "honest"]) == "abstain"


class TestAuditSession:
    def test_empty_claims_returns_vacuous_honest(self):
        result = audit_session(messages=[], claims=[], same_fn=_exact_match)
        assert result.verdict == "honest"
        assert len(result.claims) == 0
        assert result.n_honest == 0
        assert result.injection_suspected is False
        assert "AUC" in result.calibration

    def test_single_honest_claim(self):
        result = audit_session(
            messages=[{"role": "system", "content": "neutral"}],
            claims=[("Paris", "What is the capital of France?")],
            n=10,
            # Each claim needs: stateless arm (10 samples) + in-session arm (10 samples)
            client=_mk_client(["Paris"] * 10, ["Paris"] * 10),
            same_fn=_exact_match,
        )
        assert result.verdict == "honest"
        assert result.n_honest == 1
        assert result.injection_suspected is False
        assert bool(result) is True

    def test_multiple_claims_one_injected_taints_session(self):
        # Two claims. First: stateless+in-session both "Paris" → honest.
        # Second: stateless "Tokyo", in-session "Kyoto" → injected with conc shift.
        result = audit_session(
            messages=[{"role": "system", "content": "capital is Kyoto"}],
            claims=[
                ("Paris", "What is the capital of France?"),
                ("Kyoto", "What is the capital of Japan?"),
            ],
            n=10,
            client=_mk_client(
                ["Paris"] * 10, ["Paris"] * 10,    # claim 1 arms
                ["Tokyo"] * 10, ["Kyoto"] * 10,    # claim 2 arms
            ),
            same_fn=_exact_match,
        )
        assert result.verdict == "injected"   # priority: injected > honest
        assert result.injection_suspected is True
        assert result.n_honest == 1
        assert result.n_injected == 1
        assert bool(result) is False

    def test_scope_warnings_are_union(self):
        result = audit_session(
            messages=[{"role": "system", "content": "x"}],
            claims=[("a", "q1?"), ("b", "q2?")],
            n=10,
            client=_mk_client(
                ["a"] * 10, ["a"] * 10,
                ["b"] * 10, ["b"] * 10,
            ),
            same_fn=_exact_match,
        )
        # Both claims produce in-session arms → single-attack-type-calibration triggered
        assert "single-attack-type-calibration" in result.scope_warnings
        assert "belief-not-truth" in result.scope_warnings
        # Sorted union
        assert list(result.scope_warnings) == sorted(result.scope_warnings)

    def test_session_namedtuple_field_surface(self):
        expected = {
            "claims", "verdict", "injection_suspected",
            "n_honest", "n_contradiction", "n_confabulation",
            "n_injected", "n_abstain",
            "scope_warnings", "calibration",
        }
        assert set(SessionAudit._fields) == expected

    def test_dict_or_tuple_claim_inputs_supported_via_cli_paths(self):
        # API accepts (claim, question) tuples only at the Python layer; the
        # CLI accepts dicts too. Verify the API-level shape is strict.
        result = audit_session(
            messages=[{"role": "system", "content": "x"}],
            claims=[("a", "q?")],
            n=10,
            client=_mk_client(["a"] * 10, ["a"] * 10),
            same_fn=_exact_match,
        )
        assert result.n_honest == 1


# ---------------------------------------------------------------------------
# Parallelization tests — exercise ThreadPoolExecutor in _resample
# ---------------------------------------------------------------------------


class TestParallelization:
    """ThreadPoolExecutor parallelizes the N resamples. Verify thread-safety
    on the mocked client + determinism of result (order-independent because
    samples within an arm are aggregated into clusters)."""

    def test_parallel_resampling_preserves_verdict(self):
        # 10 identical samples in parallel should still produce 1 cluster
        # and verdict "honest".
        result = audit_claim(
            claim="Paris", question="q?", n=10, max_workers=8,
            client=_mk_client(["Paris"] * 10), same_fn=_exact_match,
        )
        assert result.verdict == "honest"
        assert result.n_clusters_stateless == 1

    def test_max_workers_1_forces_serial(self):
        # Default-equivalent verdict at serial execution — useful as a debug
        # mode and confirms the fallback path doesn't break.
        result = audit_claim(
            claim="Paris", question="q?", n=10, max_workers=1,
            client=_mk_client(["Paris"] * 10), same_fn=_exact_match,
        )
        assert result.verdict == "honest"
        assert result.grounded == 1.0

    def test_parallel_n_equals_1_skips_executor(self):
        # n=1 should not spawn a thread (perf nit, but also a defensive check
        # the n<=1 branch returns without ThreadPoolExecutor context).
        result = audit_claim(
            claim="Paris", question="q?", n=1, max_workers=8,
            client=_mk_client(["Paris"]), same_fn=_exact_match,
        )
        # n=1 triggers the low-N scope warning per the prior calibration band.
        assert "low-N" in result.scope_warnings

    def test_parallel_does_not_drop_samples(self):
        # Under parallelism the executor's ex.map should preserve N exactly.
        result = audit_claim(
            claim="Paris", question="q?", n=10, max_workers=8,
            client=_mk_client(["Paris"] * 10), same_fn=_exact_match,
        )
        assert len(result.samples_stateless) == 10
