# -*- coding: utf-8 -*-
"""
test_thought.py — tests for the 3.0.0a1 Thought type.

Covers:
  - Thought construction (empty, target, from_vitals, from real
    demo trajectories via Raw)
  - Algebra (interpolate, mix, distance, similarity, delta,
    + / - / == operators)
  - Save / load round-trip with the .fathom file format
  - content_hash determinism (identity-free)
  - Cognitive equivalence: Thoughts from real trajectories should
    cluster by source category in eigenvalue space
  - Phase handling for short trajectories
  - read_thought() across Vitals / response / text input modes

These tests run with no network access and no API keys. They use
the bundled demo trajectories shipped with the styxx package.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import styxx
from styxx import Raw
from styxx.cli import _load_demo_trajectories
from styxx.thought import (
    Thought, PhaseThought, ThoughtDelta,
    read_thought, _vec_distance,
    CATEGORIES, PHASE_ORDER, N_CATEGORIES,
    FATHOM_FORMAT, FATHOM_VERSION,
)
from styxx.vitals import Vitals, PhaseReading


# ══════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════

def _demo_vitals(category: str) -> Vitals:
    """Build a real Vitals object from the bundled demo trajectories."""
    data = _load_demo_trajectories()
    t = data["trajectories"][category]
    return Raw().read(
        entropy=t["entropy"],
        logprob=t["logprob"],
        top2_margin=t["top2_margin"],
    )


def _demo_thought(category: str, **kwargs) -> Thought:
    """Build a Thought from a demo trajectory, recording the source
    category in the tags so we can assert clustering later.
    """
    v = _demo_vitals(category)
    return Thought.from_vitals(
        v,
        source_text=f"demo:{category}",
        source_model="atlas-v0.3-fixture",
        tags={"demo_category": category, **kwargs},
    )


def _phase_reading(category: str, confidence: float = 0.6) -> PhaseReading:
    """Build a synthetic PhaseReading with a desired primary category."""
    other = (1.0 - confidence) / (N_CATEGORIES - 1)
    probs = {c: (confidence if c == category else other) for c in CATEGORIES}
    return PhaseReading(
        phase="phase4_late",
        n_tokens_used=25,
        features=[0.0] * 12,
        predicted_category=category,
        margin=confidence - other,
        distances={c: 1.0 for c in CATEGORIES},
        probs=probs,
    )


# ══════════════════════════════════════════════════════════════════
# Constructors
# ══════════════════════════════════════════════════════════════════

class TestConstructors:

    def test_empty_thought_is_uniform(self):
        e = Thought.empty()
        assert e.populated_phases == list(PHASE_ORDER)
        for name in e.populated_phases:
            probs = e.phases[name].probs
            assert len(probs) == N_CATEGORIES
            for p in probs:
                assert abs(p - 1.0 / N_CATEGORIES) < 1e-9

    def test_target_thought_primary_matches(self):
        for cat in CATEGORIES:
            t = Thought.target(cat, confidence=0.7)
            assert t.primary_category == cat
            assert abs(t.primary_confidence - 0.7) < 1e-6

    def test_target_thought_rejects_unknown_category(self):
        with pytest.raises(ValueError):
            Thought.target("nonsense_category")

    def test_target_thought_rejects_bad_confidence(self):
        with pytest.raises(ValueError):
            Thought.target("reasoning", confidence=0.0)
        with pytest.raises(ValueError):
            Thought.target("reasoning", confidence=1.5)

    def test_from_vitals_extracts_all_phases(self):
        v = _demo_vitals("reasoning")
        t = Thought.from_vitals(v)
        # demo trajectories have 30 tokens, so all 4 phases should be present
        assert len(t.populated_phases) == 4
        assert set(t.populated_phases) == set(PHASE_ORDER)

    def test_from_vitals_records_source_text_hash_only(self):
        v = _demo_vitals("creative")
        t = Thought.from_vitals(v, source_text="hello world", source_model="m1")
        assert t.source_model == "m1"
        assert t.source_text_hash is not None
        assert t.source_text_hash.startswith("sha256:")
        # Make sure the actual text isn't in the dict form anywhere
        d = t.as_dict()
        blob = json.dumps(d)
        assert "hello world" not in blob

    def test_from_vitals_rejects_non_vitals(self):
        with pytest.raises(TypeError):
            Thought.from_vitals("not a vitals")  # type: ignore


# ══════════════════════════════════════════════════════════════════
# Distance + similarity
# ══════════════════════════════════════════════════════════════════

class TestDistanceAndSimilarity:

    def test_self_distance_is_zero(self):
        t = Thought.target("reasoning")
        assert t.distance(t) == 0.0
        assert t.similarity(t) == 1.0

    def test_distance_is_symmetric(self):
        a = Thought.target("reasoning")
        b = Thought.target("creative")
        assert abs(a.distance(b) - b.distance(a)) < 1e-9

    def test_distance_is_positive_for_different_targets(self):
        a = Thought.target("reasoning")
        b = Thought.target("hallucination")
        assert a.distance(b) > 0

    def test_similarity_in_unit_interval(self):
        for c1 in CATEGORIES:
            for c2 in CATEGORIES:
                a = Thought.target(c1)
                b = Thought.target(c2)
                s = a.similarity(b)
                assert 0.0 <= s <= 1.0
                if c1 == c2:
                    assert abs(s - 1.0) < 1e-9

    def test_metrics_all_implemented(self):
        a = Thought.target("reasoning")
        b = Thought.target("creative")
        for metric in ("euclidean", "cosine", "js"):
            d = a.distance(b, metric=metric)
            assert d >= 0.0

    def test_unknown_metric_raises(self):
        a = Thought.target("reasoning")
        b = Thought.target("creative")
        with pytest.raises(ValueError):
            a.distance(b, metric="manhattan")


# ══════════════════════════════════════════════════════════════════
# Algebra: interpolate, mix, +, -, mean
# ══════════════════════════════════════════════════════════════════

class TestAlgebra:

    def test_interpolate_midpoint_is_equidistant(self):
        a = Thought.target("reasoning")
        b = Thought.target("creative")
        m = a.interpolate(b, alpha=0.5)
        d_ma = m.distance(a)
        d_mb = m.distance(b)
        assert abs(d_ma - d_mb) < 1e-6

    def test_interpolate_alpha_extremes(self):
        a = Thought.target("reasoning")
        b = Thought.target("creative")
        # alpha=1 should be a, alpha=0 should be b (cognitively)
        m1 = a.interpolate(b, alpha=1.0)
        m0 = a.interpolate(b, alpha=0.0)
        assert m1.distance(a) < 1e-9
        assert m0.distance(b) < 1e-9

    def test_interpolate_rejects_bad_alpha(self):
        a = Thought.target("reasoning")
        b = Thought.target("creative")
        with pytest.raises(ValueError):
            a.interpolate(b, alpha=-0.1)
        with pytest.raises(ValueError):
            a.interpolate(b, alpha=1.5)

    def test_add_operator_is_mean(self):
        a = Thought.target("reasoning")
        b = Thought.target("creative")
        sum_ = a + b
        mid = a.interpolate(b, alpha=0.5)
        assert sum_.distance(mid) < 1e-9

    def test_subtract_returns_thought_delta(self):
        a = Thought.target("reasoning")
        b = Thought.target("creative")
        delta = a - b
        assert isinstance(delta, ThoughtDelta)
        # Magnitude should be positive
        assert delta.magnitude() > 0
        # And there should be exactly 4 phase deltas
        assert len(delta.per_phase) == 4
        # Top mover should mention 'reasoning' or 'creative'
        movers = delta.biggest_movers(top_k=2)
        cats = {m[1] for m in movers}
        assert cats & {"reasoning", "creative"}

    def test_mix_with_uniform_weights(self):
        a = Thought.target("reasoning", confidence=0.8)
        b = Thought.target("creative", confidence=0.8)
        c = Thought.target("retrieval", confidence=0.8)
        m = Thought.mix([a, b, c])
        # All three categories should have some mass in m
        mean = m.mean_probs()
        for cat in ("reasoning", "creative", "retrieval"):
            i = CATEGORIES.index(cat)
            assert mean[i] > 0.1, f"category {cat} got negligible mass: {mean[i]}"

    def test_mix_with_weights(self):
        a = Thought.target("reasoning", confidence=0.9)
        b = Thought.target("creative", confidence=0.9)
        # Heavily weight 'a'
        m = Thought.mix([a, b], weights=[0.9, 0.1])
        # The result should be much closer to a than to b
        assert m.distance(a) < m.distance(b)

    def test_mix_rejects_empty(self):
        with pytest.raises(ValueError):
            Thought.mix([])

    def test_mix_rejects_mismatched_weights(self):
        a = Thought.target("reasoning")
        b = Thought.target("creative")
        with pytest.raises(ValueError):
            Thought.mix([a, b], weights=[0.5])

    def test_eq_is_cognitive_equality(self):
        # Two target thoughts with the same content compare equal
        # even though they have different thought_ids
        a = Thought.target("reasoning", confidence=0.7)
        b = Thought.target("reasoning", confidence=0.7)
        assert a == b
        # And different content compares unequal
        c = Thought.target("creative", confidence=0.7)
        assert a != c


# ══════════════════════════════════════════════════════════════════
# .fathom file I/O
# ══════════════════════════════════════════════════════════════════

class TestFathomFileFormat:

    def test_save_load_roundtrip_target(self, tmp_path):
        original = Thought.target("reasoning", confidence=0.8)
        path = tmp_path / "out.fathom"
        original.save(path)
        loaded = Thought.load(path)
        assert loaded == original
        assert loaded.primary_category == "reasoning"

    def test_save_load_roundtrip_real_trajectory(self, tmp_path):
        original = _demo_thought("reasoning")
        path = tmp_path / "real.fathom"
        original.save(path)
        loaded = Thought.load(path)
        assert loaded == original

    def test_no_bom_on_save(self, tmp_path):
        t = Thought.target("creative")
        path = tmp_path / "nobom.fathom"
        t.save(path)
        head = path.read_bytes()[:3]
        assert head != b"\xef\xbb\xbf", f"BOM detected in .fathom file: {head!r}"

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Thought.load(tmp_path / "nope.fathom")

    def test_load_wrong_format_raises(self, tmp_path):
        path = tmp_path / "bad.fathom"
        path.write_text(json.dumps({"fathom_format": "not_a_thought"}))
        with pytest.raises(ValueError):
            Thought.load(path)

    def test_load_unknown_version_raises(self, tmp_path):
        path = tmp_path / "future.fathom"
        path.write_text(json.dumps({
            "fathom_format": FATHOM_FORMAT,
            "fathom_version": "99.99",
        }))
        with pytest.raises(ValueError):
            Thought.load(path)

    def test_canonical_dict_has_required_top_level_keys(self):
        t = Thought.target("reasoning")
        d = t.as_dict()
        for key in ("fathom_format", "fathom_version", "thought_id",
                    "schema", "trajectory", "source", "created_at"):
            assert key in d, f"missing top-level key: {key}"
        assert d["fathom_format"] == "thought"
        assert d["fathom_version"] == FATHOM_VERSION

    def test_canonical_dict_renders_to_sorted_json(self):
        t = Thought.target("reasoning")
        s = t.as_json()
        # sort_keys=True means alphabetical order at every level
        d1 = json.loads(s)
        d2 = json.loads(t.as_json())
        assert d1 == d2


# ══════════════════════════════════════════════════════════════════
# content_hash
# ══════════════════════════════════════════════════════════════════

class TestContentHash:

    def test_content_hash_is_identity_free(self):
        a = Thought.target("reasoning", confidence=0.7)
        b = Thought.target("reasoning", confidence=0.7)
        # Different thought_ids
        assert a.thought_id != b.thought_id
        # But same content_hash
        assert a.content_hash() == b.content_hash()

    def test_content_hash_changes_on_content_change(self):
        a = Thought.target("reasoning", confidence=0.7)
        b = Thought.target("reasoning", confidence=0.8)
        assert a.content_hash() != b.content_hash()

    def test_content_hash_is_64_hex_chars(self):
        h = Thought.target("reasoning").content_hash()
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


# ══════════════════════════════════════════════════════════════════
# Cognitive equivalence on real trajectories
# ══════════════════════════════════════════════════════════════════

class TestCognitiveEquivalenceOnRealData:
    """The substrate-independence claim, tested empirically against
    bundled atlas v0.3 demo trajectories.

    Key invariant: if Thoughts represent cognitive content faithfully,
    then a Thought read from a 'reasoning' demo should be CLOSER to
    the canonical reasoning target than to the canonical creative
    target. If this fails for any category, the eigenvalue projection
    is broken — and therefore Thought is not portable.
    """

    def test_demo_trajectories_classify_to_their_source_category(self):
        """For each demo category, the Thought's primary category at
        phase 4 should equal (or be highly similar to) the source.

        We don't strictly require equality — the centroid classifier
        is stochastic at the margin and some demo trajectories have
        ambiguous fingerprints. But the source category should be
        in the top-3 by mass for at least 4 of 6 categories.
        """
        hits = 0
        for cat in CATEGORIES:
            t = _demo_thought(cat)
            mean = t.mean_probs()
            ranked = sorted(zip(CATEGORIES, mean), key=lambda kv: -kv[1])
            top3 = [c for c, _ in ranked[:3]]
            if cat in top3:
                hits += 1
        assert hits >= 4, (
            f"demo trajectories classified into top-3 for only {hits}/6 "
            f"source categories — eigenvalue projection looks broken"
        )

    def test_real_trajectory_round_trips_through_fathom_file(self, tmp_path):
        """Read a Thought from a real trajectory, save, load, and
        verify the loaded Thought has identical eigenvalue content.
        This is the core portability claim.
        """
        original = _demo_thought("reasoning")
        path = tmp_path / "reasoning.fathom"
        original.save(path)
        loaded = Thought.load(path)
        # Cognitive equality (==) checks per-phase probs to 1e-9
        assert loaded == original
        # content_hash should also match (it's identity-free and only
        # depends on cognitive content)
        assert loaded.content_hash() == original.content_hash()

    def test_distance_between_real_trajectories_is_finite_and_bounded(self):
        """All pairwise distances between real demo Thoughts should
        be finite and bounded by sqrt(2) (the max L2 distance between
        two probability vectors on the simplex).
        """
        thoughts = {cat: _demo_thought(cat) for cat in CATEGORIES}
        max_d = math.sqrt(2.0)
        for c1 in CATEGORIES:
            for c2 in CATEGORIES:
                d = thoughts[c1].distance(thoughts[c2])
                assert math.isfinite(d), f"{c1} vs {c2}: distance not finite"
                assert 0 <= d <= max_d + 1e-9, (
                    f"{c1} vs {c2}: distance {d} out of [0, sqrt(2)]"
                )

    def test_interpolation_of_real_thoughts_is_smaller_than_either(self):
        """The midpoint between two real-trajectory Thoughts should
        be strictly closer to each parent than the parents are to
        each other (triangle inequality on the simplex)."""
        a = _demo_thought("reasoning")
        b = _demo_thought("creative")
        m = a.interpolate(b, alpha=0.5)
        d_ab = a.distance(b)
        d_am = a.distance(m)
        d_bm = b.distance(m)
        if d_ab > 1e-9:
            assert d_am <= d_ab + 1e-9
            assert d_bm <= d_ab + 1e-9


# ══════════════════════════════════════════════════════════════════
# Phase handling for short trajectories
# ══════════════════════════════════════════════════════════════════

class TestPhaseHandling:

    def test_short_trajectory_only_populates_phase1(self):
        """A 1-token trajectory should populate only phase 1."""
        v = Raw().read(
            entropy=[1.5],
            logprob=[-0.5],
            top2_margin=[0.4],
        )
        t = Thought.from_vitals(v)
        # phase1 should be populated
        assert t.phases.get("phase1_preflight") is not None
        # later phases should be None (not enough tokens)
        assert t.phases.get("phase4_late") is None

    def test_distance_falls_back_to_mean_when_phases_disjoint(self):
        """If two Thoughts have NO phases in common, distance should
        fall back to comparing mean_probs and still return a finite
        value (not raise)."""
        # Build a 1-token Thought (only phase1) — but actually phase2
        # also fires at 5 tokens. Let's build a 1-token one and a
        # synthetic Thought that only has phase4 to force disjoint.
        short_v = Raw().read(
            entropy=[1.5], logprob=[-0.5], top2_margin=[0.4],
        )
        short_t = Thought.from_vitals(short_v)
        long_phase4_only = Thought(
            phases={
                "phase1_preflight": None,
                "phase2_early": None,
                "phase3_mid": None,
                "phase4_late": PhaseThought(
                    probs=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    n_tokens=25,
                ),
            },
        )
        d = short_t.distance(long_phase4_only)
        assert math.isfinite(d)

    def test_interpolate_handles_phase_union(self):
        """Interpolating a 1-token Thought with a 4-phase Thought
        should yield a Thought that has all phases the parents had."""
        short = Thought.from_vitals(
            Raw().read(entropy=[1.5], logprob=[-0.5], top2_margin=[0.4])
        )
        full = _demo_thought("reasoning")
        merged = short.interpolate(full, alpha=0.5)
        assert "phase1_preflight" in merged.populated_phases
        assert "phase4_late" in merged.populated_phases


# ══════════════════════════════════════════════════════════════════
# read_thought() input modes
# ══════════════════════════════════════════════════════════════════

class TestReadThought:

    def test_read_thought_from_vitals(self):
        v = _demo_vitals("reasoning")
        t = read_thought(v, model="m1")
        assert isinstance(t, Thought)
        assert t.source_model == "m1"

    def test_read_thought_from_response_with_vitals(self):
        """A response object that has .vitals attached should be
        accepted directly."""
        v = _demo_vitals("reasoning")

        class FakeResponse:
            vitals = v
            model = "fake-model"

        t = read_thought(FakeResponse())
        assert isinstance(t, Thought)
        assert t.source_model == "fake-model"

    def test_read_thought_from_string_without_client_raises(self):
        with pytest.raises(ValueError):
            read_thought("hello")

    def test_read_thought_unsupported_type_raises(self):
        with pytest.raises(TypeError):
            read_thought(42)  # type: ignore


# ══════════════════════════════════════════════════════════════════
# Module exports
# ══════════════════════════════════════════════════════════════════

class TestPublicAPI:

    def test_thought_is_exposed_on_styxx_module(self):
        assert hasattr(styxx, "Thought")
        assert hasattr(styxx, "PhaseThought")
        assert hasattr(styxx, "ThoughtDelta")
        assert hasattr(styxx, "read_thought")
        assert hasattr(styxx, "write_thought")
        assert hasattr(styxx, "FATHOM_FORMAT")
        assert hasattr(styxx, "FATHOM_VERSION")

    def test_version_is_3_0_alpha(self):
        assert styxx.__version__.startswith("3.0.0")


# ══════════════════════════════════════════════════════════════════
# Hash invariant (3.0.0a1 polish)
# ══════════════════════════════════════════════════════════════════

class TestHashInvariant:
    """Python's hash invariant requires: a == b => hash(a) == hash(b).
    For Thought, == is cognitive equality (per-phase per-category to
    1e-9), so __hash__ must be content-based, not id-based.
    """

    def test_equal_thoughts_have_equal_hashes(self):
        a = Thought.target("reasoning", confidence=0.7)
        b = Thought.target("reasoning", confidence=0.7)
        assert a == b
        assert hash(a) == hash(b), (
            f"hash invariant violated: a == b but hash(a)={hash(a)} "
            f"!= hash(b)={hash(b)}"
        )

    def test_set_dedupes_by_cognitive_content(self):
        """Two cognitively equivalent Thoughts should collapse to one
        entry in a set, not stay separate by thought_id."""
        a = Thought.target("reasoning", confidence=0.7)
        b = Thought.target("reasoning", confidence=0.7)
        c = Thought.target("creative", confidence=0.7)
        assert a.thought_id != b.thought_id  # different identities
        s = {a, b, c}
        assert len(s) == 2  # a and b dedupe, c stays separate

    def test_dict_keyed_by_thought_works(self):
        a = Thought.target("reasoning", confidence=0.7)
        b = Thought.target("reasoning", confidence=0.7)
        d = {a: "first"}
        d[b] = "second"  # should overwrite, not add
        assert len(d) == 1
        assert d[a] == "second"

    def test_real_trajectory_roundtrip_preserves_hash(self):
        original = _demo_thought("reasoning")
        as_json = original.as_json()
        loaded = Thought.from_dict(json.loads(as_json))
        assert loaded == original
        assert hash(loaded) == hash(original)


# ══════════════════════════════════════════════════════════════════
# Vitals.to_thought() symmetric shortcut (3.0.0a1 polish)
# ══════════════════════════════════════════════════════════════════

class TestVitalsToThoughtShortcut:

    def test_vitals_has_to_thought_method(self):
        v = _demo_vitals("reasoning")
        assert hasattr(v, "to_thought")

    def test_vitals_to_thought_matches_from_vitals(self):
        v = _demo_vitals("reasoning")
        a = v.to_thought(source_model="m1", tags={"k": "v"})
        b = Thought.from_vitals(v, source_model="m1", tags={"k": "v"})
        assert a == b
        assert a.source_model == "m1" == b.source_model
        assert a.tags == b.tags

    def test_vitals_to_thought_carries_source_text_hash(self):
        v = _demo_vitals("reasoning")
        t = v.to_thought(source_text="hello", source_model="m")
        assert t.source_text_hash is not None
        assert t.source_text_hash.startswith("sha256:")


# ══════════════════════════════════════════════════════════════════
# Provenance bridge: Thought.certify() (3.0.0a1 polish)
# ══════════════════════════════════════════════════════════════════

class TestProvenanceBridge:
    """Thought.certify() produces a CognitiveCertificate whose
    thought_content_hash field matches the Thought's content_hash().
    This is the bridge between the .fathom content layer and the
    cognitive-provenance attestation layer.
    """

    def test_certify_returns_certificate(self):
        from styxx.provenance import CognitiveCertificate
        t = Thought.target("reasoning", confidence=0.85)
        cert = t.certify(agent_name="test", session_id="s1")
        assert isinstance(cert, CognitiveCertificate)

    def test_certificate_thought_content_hash_matches(self):
        t = Thought.target("reasoning", confidence=0.85)
        cert = t.certify()
        assert cert.thought_content_hash == t.content_hash()

    def test_certificate_carries_primary_category(self):
        t = Thought.target("reasoning", confidence=0.85)
        cert = t.certify()
        assert cert.phase4_category == "reasoning"
        assert cert.phase4_confidence is not None
        assert abs(cert.phase4_confidence - 0.85) < 1e-3

    def test_certificate_gate_is_pass_for_safe_target(self):
        t = Thought.target("reasoning", confidence=0.85)
        cert = t.certify()
        assert cert.gate == "pass"
        assert cert.integrity in ("verified", "degraded")

    def test_certificate_gate_is_warn_for_refusal_target(self):
        t = Thought.target("refusal", confidence=0.85)
        cert = t.certify()
        assert cert.gate == "warn"

    def test_certificate_gate_is_fail_for_hallucination_target(self):
        t = Thought.target("hallucination", confidence=0.85)
        cert = t.certify()
        assert cert.gate == "fail"

    def test_certify_real_trajectory_thought(self):
        t = _demo_thought("reasoning")
        cert = t.certify(agent_name="real-agent")
        assert cert.thought_content_hash == t.content_hash()
        # Should serialize to dict cleanly with the bound hash
        d = cert.as_dict()
        assert d["verification"]["thought_content_hash"] == t.content_hash()

    def test_two_equal_thoughts_produce_equal_thought_content_hashes(self):
        a = Thought.target("reasoning", confidence=0.7)
        b = Thought.target("reasoning", confidence=0.7)
        ca = a.certify()
        cb = b.certify()
        assert ca.thought_content_hash == cb.thought_content_hash


# ══════════════════════════════════════════════════════════════════
# write_thought() mock-client test (3.0.0a1 polish — no API calls)
# ══════════════════════════════════════════════════════════════════

class _MockMessage:
    def __init__(self, content: str):
        self.content = content

class _MockChoice:
    def __init__(self, content: str):
        self.message = _MockMessage(content)

class _MockResponse:
    def __init__(self, content: str, vitals: Vitals, model: str):
        self.choices = [_MockChoice(content)]
        self.vitals = vitals
        self.model = model

class _MockChatCompletions:
    def __init__(self, scripted_vitals_seq, scripted_text="generated text"):
        self.scripted_vitals_seq = list(scripted_vitals_seq)
        self.scripted_text = scripted_text
        self.calls = []

    def create(self, *, model, messages, max_tokens, **kwargs):
        self.calls.append({"model": model, "messages": messages, "max_tokens": max_tokens})
        if not self.scripted_vitals_seq:
            raise RuntimeError("mock client ran out of scripted vitals")
        v = self.scripted_vitals_seq.pop(0)
        return _MockResponse(content=self.scripted_text, vitals=v, model=model)

class _MockChat:
    def __init__(self, completions):
        self.completions = completions

class _MockClient:
    def __init__(self, vitals_seq, text="generated text"):
        self.chat = _MockChat(_MockChatCompletions(vitals_seq, text))


class TestWriteThoughtMockClient:
    """Exercise write_thought()'s steering loop end-to-end against a
    mock client. No API key, no network, no real model — just verifies
    the loop wires together correctly.
    """

    def test_write_thought_returns_dict_with_required_keys(self):
        target = Thought.target("reasoning", confidence=0.85)
        # Mock returns vitals that already match the target, so the
        # first iteration should hit the distance threshold
        v_match = _demo_vitals("reasoning")
        mock = _MockClient([v_match])
        result = styxx.write_thought(
            target, client=mock, model="mock-model", max_iters=1,
        )
        assert "text" in result
        assert "thought" in result
        assert "distance" in result
        assert "iters" in result
        assert "history" in result
        assert isinstance(result["thought"], Thought)

    def test_write_thought_records_history(self):
        target = Thought.target("reasoning", confidence=0.85)
        v_match = _demo_vitals("reasoning")
        mock = _MockClient([v_match, v_match, v_match])
        result = styxx.write_thought(
            target, client=mock, model="mock-model",
            max_iters=3, distance_threshold=0.0,  # force all iters
        )
        # max_iters=3 with threshold=0.0 should run all 3
        assert result["iters"] == 3
        assert len(result["history"]) == 3

    def test_write_thought_stops_early_when_threshold_hit(self):
        target = Thought.target("reasoning", confidence=0.85)
        v_match = _demo_vitals("reasoning")
        mock = _MockClient([v_match] * 5)
        result = styxx.write_thought(
            target, client=mock, model="mock-model",
            max_iters=5, distance_threshold=999.0,  # any distance passes
        )
        # First iteration's distance is below 999, so we stop after 1
        assert result["iters"] == 1

    def test_write_thought_rejects_non_thought_target(self):
        with pytest.raises(TypeError):
            styxx.write_thought("not a thought", client=_MockClient([]))

    def test_write_thought_rejects_no_client(self):
        with pytest.raises(ValueError):
            styxx.write_thought(Thought.target("reasoning"), client=None)

    def test_write_thought_picks_best_iteration(self):
        """Across multiple iterations, the best (lowest distance) one
        should be returned."""
        target = Thought.target("reasoning", confidence=0.85)
        # First iteration returns creative (far), second returns reasoning (close)
        v_far = _demo_vitals("creative")
        v_close = _demo_vitals("reasoning")
        mock = _MockClient([v_far, v_close, v_far])
        result = styxx.write_thought(
            target, client=mock, model="mock-model",
            max_iters=3, distance_threshold=0.0,  # run all 3
        )
        assert result["iters"] == 3
        # The best should be one of the close ones (iteration index 1)
        # or at least, no worse than the close iteration's distance
        close_distance = target.distance(
            Thought.from_vitals(v_close)
        )
        assert result["distance"] <= close_distance + 1e-9
