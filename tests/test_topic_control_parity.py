# -*- coding: utf-8 -*-
"""
test_topic_control_parity
=========================

§8 integrity tripwire for the topic-overlap control preregistration.
API-free (no embedding calls). Verifies:

  1. topic_control_analysis reuses the LOCKED drift_axis_alignment from
     drift_axis_scorer (same function object) — not a reimplementation.
  2. The stopword list is a frozenset (locked, immutable).
  3. content_words strips stopwords deterministically.
  4. The stratified regime permutation test is deterministic given seed
     and returns a valid p-value, and behaves correctly on a planted
     strong / null effect.

If (1) fails, the analysis has drifted from the locked scorer — the DAA
in the topic-control bet would no longer equal the DAA in the parent
positive. If (2) fails, the topic-overlap measure's definition moved
after lock. Either is a methodology change requiring a new lock-hash.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load(modname: str):
    repo = Path(__file__).resolve().parent.parent
    path = repo / "scripts" / f"{modname}.py"
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def test_reuses_locked_drift_axis_alignment():
    """topic_control_analysis.drift_axis_alignment must BE the locked
    scorer's function, imported, not a copy."""
    scorer = _load("drift_axis_scorer")
    analysis = _load("topic_control_analysis")
    assert analysis.drift_axis_alignment is scorer.drift_axis_alignment, (
        "topic_control_analysis must import the locked drift_axis_alignment, "
        "not reimplement it. See test docstring."
    )


def test_stopwords_locked_frozenset():
    analysis = _load("topic_control_analysis")
    assert isinstance(analysis.STOPWORDS, frozenset), "STOPWORDS must be frozen"
    # spot-check a few canonical function words are present
    for w in ("the", "and", "of", "to", "is", "a"):
        assert w in analysis.STOPWORDS


def test_content_words_strips_stopwords():
    analysis = _load("topic_control_analysis")
    out = analysis.content_words("The library is on the corner of Main and Oak")
    toks = out.split()
    assert "the" not in toks and "is" not in toks and "on" not in toks and "of" not in toks
    # content words survive
    assert "library" in toks and "corner" in toks and "main" in toks and "oak" in toks


def test_permutation_deterministic_and_valid():
    analysis = _load("topic_control_analysis")
    # planted STRONG regime effect: cooperative cells high, non-coop low
    strong = {
        "coop_shared":        [0.80, 0.82, 0.78, 0.81, 0.79],
        "coop_independent":   [0.76, 0.74, 0.77, 0.75, 0.73],
        "noncoop_shared":     [0.40, 0.38, 0.42, 0.41, 0.39],
        "noncoop_independent":[0.35, 0.37, 0.33, 0.36, 0.34],
    }
    p1 = analysis.stratified_regime_permutation_p(strong, n_resamples=2000, seed=7)
    p2 = analysis.stratified_regime_permutation_p(strong, n_resamples=2000, seed=7)
    assert p1 == p2, "permutation p must be deterministic given seed"
    assert 0.0 < p1 < 1.0
    assert p1 < 0.01, f"strong planted effect should be significant, got p={p1}"

    # planted NULL effect: all cells same distribution
    null = {
        "coop_shared":        [0.50, 0.52, 0.48, 0.51, 0.49],
        "coop_independent":   [0.51, 0.49, 0.50, 0.52, 0.48],
        "noncoop_shared":     [0.49, 0.51, 0.50, 0.48, 0.52],
        "noncoop_independent":[0.50, 0.50, 0.49, 0.51, 0.49],
    }
    p_null = analysis.stratified_regime_permutation_p(null, n_resamples=2000, seed=7)
    assert p_null > 0.05, f"null planted effect should not be significant, got p={p_null}"


def test_independent_seeds_distinct_from_shared():
    """The independent-topic seeds must be genuinely separate-deliverable
    (not the same shared-deliverable tasks renamed)."""
    corpus = _load("topic_control_corpus")
    assert len(corpus.INDEPENDENT_TASK_SEEDS) == 5
    for seed in corpus.INDEPENDENT_TASK_SEEDS.values():
        sp = seed.seed_prompt.lower()
        # independent seeds must signal separateness, not convergence
        assert ("separate" in sp or "own" in sp or "different" in sp), seed.name
        assert "do not" in sp  # explicit "do not merge/co-author/plan one shared"
