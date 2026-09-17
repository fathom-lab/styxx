"""styxx.islands — the cohort survey, and the refusals that keep it honest."""
import inspect

import numpy as np
import pytest

import styxx.islands as islands_mod
from styxx.islands import survey, cliff, rescue, frame, affinity, MIN_COHORT


def _cohort(n_items=60, seed=0, island=True):
    """Three members sharing a latent geometry, plus one rotated away from it."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_items, 8))
    reps = {}
    for i, name in enumerate(["a", "b", "c"]):
        W = rng.standard_normal((8, 16 + i))
        reps[name] = Z @ W + 0.05 * rng.standard_normal((n_items, 16 + i))
    if island:
        Zi = rng.standard_normal((n_items, 8))          # its own latent: not the shared one
        reps["island"] = Zi @ rng.standard_normal((8, 20))
    else:
        reps["island"] = Z @ rng.standard_normal((8, 20)) + 0.05 * rng.standard_normal((n_items, 20))
    return reps


def test_frame_is_orthonormal_and_item_space():
    X = np.random.default_rng(0).standard_normal((40, 12))
    U = frame(X, k=5)
    assert U.shape == (40, 5)                      # item space, not feature space
    assert np.allclose(U.T @ U, np.eye(5), atol=1e-8)


def test_affinity_bounds():
    U = frame(np.random.default_rng(1).standard_normal((30, 9)), k=4)
    assert affinity(U, U) == pytest.approx(1.0, abs=1e-9)
    assert 0.0 <= affinity(U, frame(np.random.default_rng(2).standard_normal((30, 9)), k=4)) <= 1.0


def test_survey_finds_the_planted_island():
    s = survey(_cohort(), k=6, n_null=100, n_perm=100)
    assert "island" in s.islands
    assert s.mean_affinity["island"] < min(s.mean_affinity[m] for m in ("a", "b", "c"))


def test_survey_finds_no_island_when_none_planted():
    """Assert the VERDICT, not an empty `.islands` list.

    The MAD rule `median - 1.4826*MAD` sits ~1 SD below the median, so it names roughly 15% of
    members unconditionally — on a cohort of 8 the list is non-empty about 80% of the time even
    on pure noise (red team 2026-08-06). The list is a lead; only the bimodality gate is a claim.
    """
    s = survey(_cohort(island=False), k=6, n_null=100, n_perm=1000)
    assert s.verdict != "ISLANDS_PRESENT"


def test_survey_refuses_a_verdict_below_min_cohort():
    """Bimodality is not testable on four points and the instrument must say so, not guess."""
    s = survey(_cohort(), k=6, n_null=50, n_perm=50)
    assert len(s.members) < MIN_COHORT
    assert s.verdict == "UNDERPOWERED__n_below_8"
    assert any("not testable" in c for c in s.caveats)


def test_survey_rejects_mismatched_item_sets():
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError, match="share the item set"):
        survey({"a": rng.standard_normal((30, 5)), "b": rng.standard_normal((31, 5))})


def test_survey_flags_a_cohort_that_shares_no_frame_at_all():
    rng = np.random.default_rng(4)
    reps = {n: rng.standard_normal((40, 10)) for n in "abcd"}     # independent: no shared frame
    s = survey(reps, k=5, n_null=200, n_perm=100)
    assert any("do not share a frame" in c for c in s.caveats)


def test_cliff_refuses_when_the_endpoint_is_at_chance():
    """A knee read off a noise curve is worse than no knee — the documented failure that got
    the module's own internal legibility measure removed before release."""
    reps = _cohort()
    out = cliff(reps["a"], reps["island"], legibility_fn=lambda r, i: 1.0 / len(r), k=6)
    assert out["shape"].startswith("REFUSED__")
    assert out["knee_t_half"] is None and out["transition_width"] is None


def test_rescue_refuses_when_no_rank_beats_chance():
    reps = _cohort()
    out = rescue(reps["a"], reps["island"], legibility_fn=lambda r, i: 1.0 / len(r), ranks=(2, 8))
    assert out["reading"].startswith("REFUSED__")
    assert out["min_sufficient_rank"] is None


def test_cliff_reads_a_switch_like_curve_when_one_is_present():
    """With a legibility measure that fires only near full correction, the shape must read
    switch-like — the b46 signature, on a synthetic stand-in."""
    reps = _cohort()
    doses = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    calls = {"n": 0}

    def fn(reader, corrected):
        calls["n"] += 1
        return [0.01, 0.01, 0.02, 0.05, 0.90, 0.95][calls["n"] - 1]
    out = cliff(reps["a"], reps["island"], legibility_fn=fn, k=6, doses=doses)
    assert out["shape"] == "switch-like"
    assert out["knee_t_half"] == 0.8


def test_rescue_credits_a_recovery_only_against_its_matched_random_null():
    """A correction that a random frame of the same rank reproduces is not a rescue."""
    reps = _cohort()
    seq = iter([0.01,                     # baseline
                0.80, 0.80,               # rank 2: corrected AND its random null both high
                0.85, 0.02])              # rank 8: corrected high, null at floor
    out = rescue(reps["a"], reps["island"], legibility_fn=lambda r, i: next(seq), ranks=(2, 8))
    assert out["min_sufficient_rank"] == 8      # rank 2 rejected: its null matched it


# --- the island rule's z: the demo, the CLI, and the frozen library default (#93) ------------------

def test_library_default_island_z_is_still_one():
    """Preregistered runs (b47, h1a) called survey() with its defaults; the default is frozen."""
    assert inspect.signature(survey).parameters["island_z"].default == 1.0
    s = survey(_cohort(), k=6, n_null=20, n_perm=20)
    assert s.island_rule.startswith("mean affinity < median - 1.0*1.4826*MAD")


def test_demo_cohort_helper_keeps_the_demo_data():
    """The factored helper builds exactly what _demo built inline before #93."""
    rng = np.random.default_rng(0)
    n = 120
    shared = rng.standard_normal((n, 8))
    want = {f"mind_{i}": shared @ rng.standard_normal((8, 24 + i))
                         + 0.05 * rng.standard_normal((n, 24 + i)) for i in range(6)}
    want["mind_6"] = shared @ rng.standard_normal((8, 24)) + 0.05 * rng.standard_normal((n, 24))
    want["ISLAND"] = (rng.standard_normal((n, 8)) @ rng.standard_normal((8, 24)))
    got = islands_mod._demo_cohort()
    assert list(got) == list(want)
    assert all(np.array_equal(got[m], want[m]) for m in want)


def test_demo_cohort_at_island_z_three_flags_exactly_the_planted_island():
    s = survey(islands_mod._demo_cohort(), n_null=400, n_perm=400, island_z=3.0)
    assert s.islands == ["ISLAND"]
    assert s.island_rule.startswith("mean affinity < median - 3.0*1.4826*MAD")


def _recording_survey(monkeypatch):
    calls, real = [], islands_mod.survey

    def fake(reps, **kw):
        calls.append(kw)
        return real(reps, k=kw.get("k", 20), island_z=kw.get("island_z", 1.0), n_null=20, n_perm=20)
    monkeypatch.setattr(islands_mod, "survey", fake)
    return calls


def test_demo_calls_survey_at_island_z_three_and_prints_the_rule(monkeypatch, capsys):
    calls = _recording_survey(monkeypatch)
    assert islands_mod.main(["--demo"]) == 0
    assert [c["island_z"] for c in calls] == [3.0]
    out = capsys.readouterr().out
    assert "island rule used: mean affinity < median - 3.0*1.4826*MAD" in out
    assert "Frame affinity alone separates the planted island from the clique." in out


def test_demo_honours_an_explicit_island_z(monkeypatch, capsys):
    calls = _recording_survey(monkeypatch)
    assert islands_mod.main(["--demo", "--island-z", "1.0"]) == 0
    assert [c["island_z"] for c in calls] == [1.0]
    assert "island rule used: mean affinity < median - 1.0*1.4826*MAD" in capsys.readouterr().out


def test_cli_accepts_island_z_and_passes_it_for_npz_input(tmp_path, monkeypatch, capsys):
    reps = _cohort()
    npz = tmp_path / "cohort.npz"
    np.savez(npz, **reps)
    calls = _recording_survey(monkeypatch)
    assert islands_mod.main([str(npz), "--k", "6", "--island-z", "2.5"]) == 0
    assert islands_mod.main([str(npz), "--k", "6"]) == 0
    assert [c["island_z"] for c in calls] == [2.5, 1.0]      # default = survey()'s own default
    out = capsys.readouterr().out
    assert "island rule used: mean affinity < median - 2.5*1.4826*MAD" in out
    assert "island rule used: mean affinity < median - 1.0*1.4826*MAD" in out
