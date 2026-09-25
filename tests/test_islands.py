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


def _demo_survey_at_three():
    s = survey(islands_mod._demo_cohort(), n_null=400, n_perm=400, island_z=3.0)
    cut = float(s.island_rule.rsplit("= ", 1)[1])          # the stated rule, as printed
    lowest = min(v for m, v in s.mean_affinity.items() if m != "ISLAND")
    return s, round(lowest - cut, 4)


# survey()'s docstring: the same cohort gives affinities up to about 0.005 apart between the two
# machines it has been run on, cause not isolated. CI runs neither of them.
DRIFT = 0.005


def test_demo_cohort_at_island_z_three_lists_the_planted_island():
    """ISLAND sits 61.5 robust deviations below the median in survey()'s docstring: a drift in
    the third decimal does not move it across the cut."""
    s, _ = _demo_survey_at_three()
    assert "ISLAND" in s.islands
    assert s.island_rule.startswith("mean affinity < median - 3.0*1.4826*MAD")


def test_demo_cohort_at_island_z_three_lists_only_the_planted_island():
    """The exact list, which is what the demo's separation claim rests on. It depends on the
    clique's lowest member staying above the cut, which the next test records."""
    s, margin = _demo_survey_at_three()
    assert s.islands == ["ISLAND"], (
        f"at z=3 the demo cohort lists {s.islands}. The lowest clique member sits {margin} "
        f"above the cut here. survey()'s docstring records affinities up to about {DRIFT} apart "
        f"between machines, cause not isolated: a failure here with a margin near or under that "
        f"is that drift reaching the demo, not a broken rule.")


def test_demo_cohort_clique_margin_is_wider_than_the_documented_drift():
    """The margin between the clique's lowest member and the z=3 cut: 0.0116 on the lab's
    Windows box (Python 3.12.10, numpy 2.4.4), 0.0086 from the reporter's median and MAD in
    survey()'s docstring. Both clear the documented drift; the gap between them (0.003) is of
    the same order, which is why this is recorded apart from the exact list above."""
    _, margin = _demo_survey_at_three()
    assert margin > DRIFT, (
        f"the demo's clique margin at z=3 is {margin}, not wider than the ~{DRIFT} "
        f"between-machine affinity drift survey()'s docstring documents (cause not isolated). "
        f"The exact island list is no longer safe on this machine: read a failure of the "
        f"exact-list test as that drift, and isolate its cause before trusting the demo's claim.")


def _recording_survey(monkeypatch):
    calls, real = [], islands_mod.survey

    def fake(reps, **kw):
        calls.append(kw)
        return real(reps, k=kw.get("k", 20), island_z=kw.get("island_z", 1.0), n_null=20, n_perm=20)
    monkeypatch.setattr(islands_mod, "survey", fake)
    return calls


def test_demo_calls_survey_at_island_z_three_and_prints_the_rule(monkeypatch, capsys):
    """Which sentence follows the rule depends on the exact list, which the tests above hold
    apart from the drift; the branches themselves are driven with a fixed list below."""
    calls = _recording_survey(monkeypatch)
    assert islands_mod.main(["--demo"]) == 0
    assert [c["island_z"] for c in calls] == [3.0]
    out = capsys.readouterr().out
    assert "island rule used: mean affinity < median - 3.0*1.4826*MAD" in out


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


def test_demo_with_no_arguments_uses_three_and_names_surveys_default(monkeypatch, capsys):
    """_demo() is public enough to be called directly; its own default must be the demo z, and
    it must say that the library default is a different number."""
    calls = _recording_survey(monkeypatch)
    assert islands_mod._demo() == 0
    assert [c["island_z"] for c in calls] == [3.0]
    out = capsys.readouterr().out
    assert "survey()'s default is island_z=1.0" in out
    assert "a tight clique's low members can be listed too" in out


def test_demo_does_not_print_the_default_note_when_the_z_is_the_library_default(monkeypatch,
                                                                               capsys):
    _recording_survey(monkeypatch)
    assert islands_mod._demo(island_z=1.0) == 0
    assert "survey()'s default is island_z=" not in capsys.readouterr().out


def _survey_reporting_islands(monkeypatch, islands):
    """Run the real survey cheaply, then overwrite its island list — the branch under test reads
    that list and nothing else."""
    real = islands_mod.survey

    def fake(reps, **kw):
        s = real(reps, k=kw.get("k", 20), island_z=kw.get("island_z", 1.0), n_null=20, n_perm=20)
        s.islands = list(islands)
        return s
    monkeypatch.setattr(islands_mod, "survey", fake)


SEPARATION_CLAIM = "Frame affinity alone separates the planted island from the clique."


def test_demo_withholds_the_separation_claim_when_the_list_has_extra_members(monkeypatch, capsys):
    _survey_reporting_islands(monkeypatch, ["mind_0", "ISLAND"])
    assert islands_mod.main(["--demo"]) == 0
    out = capsys.readouterr().out
    assert SEPARATION_CLAIM not in out
    assert "the rule also lists ['mind_0']" in out
    assert "Only ISLAND was planted." in out


def test_demo_withholds_the_separation_claim_when_the_list_is_empty(monkeypatch, capsys):
    _survey_reporting_islands(monkeypatch, [])
    assert islands_mod.main(["--demo"]) == 0
    out = capsys.readouterr().out
    assert SEPARATION_CLAIM not in out
    assert "too strict to list the planted ISLAND" in out


def test_demo_makes_the_separation_claim_only_for_the_planted_island_alone(monkeypatch, capsys):
    _survey_reporting_islands(monkeypatch, ["ISLAND"])
    assert islands_mod.main(["--demo"]) == 0
    out = capsys.readouterr().out
    assert SEPARATION_CLAIM in out
    assert "the rule also lists" not in out and "missed the planted ISLAND" not in out


@pytest.mark.parametrize("listed", [["mind_0"], ["mind_0", "mind_1"]])
def test_demo_names_the_miss_when_the_list_leaves_out_the_planted_island(monkeypatch, capsys,
                                                                         listed):
    """Clique members listed and ISLAND not: the demo must say it missed, not that the rule
    'also' lists them."""
    _survey_reporting_islands(monkeypatch, listed)
    assert islands_mod.main(["--demo"]) == 0
    out = capsys.readouterr().out
    assert SEPARATION_CLAIM not in out
    assert f"the rule missed the planted ISLAND and listed {listed} instead" in out
    assert "the rule also lists" not in out and "Only ISLAND was planted." not in out


FINITE = "island z must be finite and greater than 0"


@pytest.mark.parametrize("argv, fragment", [
    (["--island-z", "0"], FINITE),
    (["--island-z", "0.0"], FINITE),
    (["--island-z", "-1"], FINITE),
    (["--island-z", "-0.5"], FINITE),
    (["--island-z", "nan"], FINITE),
    (["--island-z", "NaN"], FINITE),
    (["--island-z", "inf"], FINITE),
    (["--island-z=-inf"], FINITE),
    (["--island-z", "-inf"], "expected one argument"),
    (["--island-z", "abc"], "is not a number"),
    (["--island-z", ""], "is not a number"),
])
def test_island_z_rejects_values_the_rule_is_not_defined_for(argv, fragment, capsys):
    """At z<=0 the cut sits at or above the median and lists every member below it; nan and
    +inf empty the list and -inf lists everyone. None announced itself before this check."""
    with pytest.raises(SystemExit) as e:
        islands_mod.main(["--demo", *argv])
    assert e.value.code == 2
    err = capsys.readouterr().err
    # "-1" and "-0.5" reach the type: argparse reads them as negative numbers. "-inf" after a
    # space does not: a leading "-" that is not a negative decimal number is read as an option
    # string, so argparse refuses it one step earlier. "--island-z=-inf" hands it to the type.
    assert fragment in err
    if fragment == FINITE:
        assert "at nan or +inf the list is empty whatever the data says" in err
        assert "at -inf it names every member" in err


@pytest.mark.parametrize("good", ["1", "0.5", "3.0", "1e-6", "100"])
def test_island_z_accepts_any_finite_positive_value(good, monkeypatch):
    calls = _recording_survey(monkeypatch)
    assert islands_mod.main(["--demo", "--island-z", good]) == 0
    assert calls[-1]["island_z"] == float(good)


def test_survey_docstring_states_the_measured_spread_not_a_coin_flip():
    """The docstring used to call the between-machine move 'the fourth decimal' and the
    resulting listing 'a coin flip'. What was observed is about 0.005, cause not isolated."""
    doc = " ".join(survey.__doc__.split())
    assert "fourth decimal" not in doc and "coin flip" not in doc
    assert "0.005" in doc and "third decimal" in doc
    assert "the cause has not been isolated" in doc
    assert "Python 3.11" in doc and "Python 3.12.10" in doc and "numpy 2.4.4" in doc
