"""Tests for version 2 of the geometry axis (styxx.mind_geometry, 2026-11-10).

One test per acceptance gate V1 to V7 (papers/mind-instrument/NOTE_geometry_axis_v2_2026_11_10.md lists them),
then edge cases. Committed files only: papers/real-convergence/contextual_reps.npz, lex_tokens.json,
papers/real-convergence/artifact_2026_11_10/score_v2_out.json, papers/mind-instrument/atlas_live_reps.npz and
papers/mind-instrument/anchors_v2_2026_11_10.npz. No model, no tokenizer, no network. Where git and commit
4bbe4b3e are at hand, V1 and V7 compare with that commit's files; otherwise V1 uses values pinned from it.

styxx/mind.py itself is not changed (two lock tables pin its sha256); V7 checks that.
"""
import ast
import hashlib
import importlib.util
import itertools
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from styxx import mind  # noqa: E402
from styxx import mind_geometry as mg  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
JUNE = REPO / "papers" / "real-convergence" / "contextual_reps.npz"
LEX = REPO / "papers" / "real-convergence" / "lex_tokens.json"
SCORE = REPO / "papers" / "real-convergence" / "artifact_2026_11_10" / "score_v2_out.json"
ATLAS = REPO / "papers" / "mind-instrument" / "atlas_live_reps.npz"
V2 = REPO / "papers" / "mind-instrument" / "anchors_v2_2026_11_10.npz"
TPC_SCORE = REPO / "papers" / "tests-pass-ci" / "collect" / "tpc_score.py"
GATE_SCORE = REPO / "papers" / "darkflobi-gate" / "score.py"
FIVE = ["Qwen2.5-1.5B", "Qwen2.5-3B", "Llama-3.2-1B", "Llama-3.2-3B", "Phi-3.5-mini"]
FAMILY = dict(mind.ANCHOR_MODELS)
OLD_COMMIT = "4bbe4b3e"
MIND_PIN = "c4c5091b991061f22592e1353da3a9d14889e4817aecf9bb15645de720577115"

need = pytest.mark.skipif(not all(p.exists() for p in (JUNE, LEX, SCORE, ATLAS, V2)),
                          reason="the committed arrays are not in this checkout (wheel install)")

# geometry_citizenship at commit 4bbe4b3e, each June anchor used as a candidate against contextual_reps.npz:
# (citizenship_xfam_partial_lex, per_anchor partial_lex_rsa in ANCHOR_MODELS order). Used when git or that
# commit is not at hand.
JUNE_PINNED = {
    "Qwen2.5-1.5B": (0.1801, (1.0, 0.918, 0.133, 0.143, 0.298, 0.147)),
    "Qwen2.5-3B": (0.1972, (0.918, 1.0, 0.152, 0.164, 0.299, 0.173)),
    "Llama-3.2-1B": (0.3477, (0.133, 0.152, 1.0, 0.966, 0.169, 0.937)),
    "Llama-3.2-3B": (0.3577, (0.143, 0.164, 0.966, 1.0, 0.18, 0.944)),
    "Phi-3.5-mini": (0.233, (0.298, 0.299, 0.169, 0.18, 1.0, 0.219)),
    "gemma-2-2b": (0.484, (0.147, 0.173, 0.937, 0.944, 0.219, 1.0)),
}
OLD_KEYS = ("per_anchor", "citizenship_xfam_partial_lex", "n_anchors", "anchors_sha256")


def _git_show(rev_path: str):
    try:
        r = subprocess.run(["git", "-C", str(REPO), "show", rev_path], capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


def _old_mind(tmp_path):
    src = _git_show(OLD_COMMIT + ":styxx/mind.py")
    if src is None:
        return None
    p = tmp_path / "mind_4bbe4b3e.py"
    p.write_bytes(src)
    spec = importlib.util.spec_from_file_location("mind_4bbe4b3e", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _npz(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


# ---------------------------------------------------------------------------------------------------------------
# the gates
# ---------------------------------------------------------------------------------------------------------------

@need
def test_v1_june_convention_reproduces_commit_4bbe4b3e(tmp_path):
    june = _npz(JUNE)
    old = _old_mind(tmp_path)
    for name, fam in mind.ANCHOR_MODELS:
        reps = june["fixed__" + name].astype(float)
        got = mg.geometry_citizenship(reps, JUNE, fam, convention="june")
        assert got["convention"] == "june"
        assert "withdrawn" in got["notice"] and "ERRATUM_real_convergence_2026_11_10.md" in got["notice"]
        # the june path is styxx.mind.geometry_citizenship, called unchanged
        same = mind.geometry_citizenship(reps, JUNE, fam)
        assert {k: got[k] for k in OLD_KEYS} == same
        if old is not None:
            want = old.geometry_citizenship(reps, JUNE, fam)
            assert {k: got[k] for k in OLD_KEYS} == want
        else:
            cit, per = JUNE_PINNED[name]
            assert got["citizenship_xfam_partial_lex"] == cit
            assert tuple(a["partial_lex_rsa"] for a in got["per_anchor"]) == per
    if old is None:
        return
    # the 4 arrays of atlas_live_reps.npz as candidates too (not pinned: compared with the old function only)
    for key, arr in _npz(ATLAS).items():
        reps = arr.astype(float)
        got = mg.geometry_citizenship(reps, JUNE, "?", convention="june")
        want = old.geometry_citizenship(reps, JUNE, "?")
        assert {k: got[k] for k in OLD_KEYS} == want, key


@need
def test_v2_dominant_share_separates_the_known_arrays():
    june, atlas, v2 = _npz(JUNE), _npz(ATLAS), _npz(V2)
    thr = mg.DOMINANT_SHARE_REFUSE
    assert thr == 0.5
    above = {"Qwen2.5-1.5B", "Qwen2.5-3B", "Phi-3.5-mini"}
    for name, _f in mind.ANCHOR_MODELS:
        s = mg.dominant_direction_share(june["fixed__" + name])
        assert (s >= thr) == (name in above), (name, s)
    assert len(atlas) == 4
    for key in ("Qwen2.5-0.5B", "pythia-410m", "gpt2", "gpt2-large"):
        assert mg.dominant_direction_share(atlas[key]) >= thr, key
    assert sorted(v2) == sorted("fixed__" + m for m in FIVE)
    for key, arr in v2.items():
        assert mg.dominant_direction_share(arr) < 0.11, key      # 0.0758 to 0.0977 when written
    # two bare-template arrays read at another layer are below the cut: the share does not detect the template
    assert mg.dominant_direction_share(june["best__Qwen2.5-1.5B"]) < thr
    assert mg.dominant_direction_share(june["best__Qwen2.5-3B"]) < thr
    # the share does not depend on the scale of the array
    a = june["fixed__Llama-3.2-1B"].astype(float)
    assert mg.dominant_direction_share(a) == pytest.approx(mg.dominant_direction_share(1000.0 * a), abs=1e-12)


@need
def test_v3_refusal_exclusion_and_allow_dominant():
    atlas = _npz(ATLAS)
    q05 = atlas["Qwen2.5-0.5B"].astype(float)
    out = mg.geometry_citizenship(q05, V2, "qwen")
    assert out["convention"] == "v2"
    assert out["status"] == "REFUSED" and out["citizenship_xfam_partial_lex"] is None
    assert out["per_anchor"] == [] and "0.9896" in out["reason"]
    assert "can have other causes" in out["reason"]
    assert out["threshold"] == 0.5 and out["candidate_share"] >= 0.5
    assert set(out["anchor_shares"]) == set(FIVE)
    assert out["category_structure_note"] == mg.GEOMETRY_CATEGORY_NOTE
    # a clean candidate against the June file: the three dominated anchors are left out and listed
    june = _npz(JUNE)
    out = mg.geometry_citizenship(june["fixed__Llama-3.2-1B"].astype(float), JUNE, "llama")
    assert out["status"] == "OK"
    excl = {e["anchor"]: e["share"] for e in out["anchors_excluded"]}
    assert set(excl) == {"Qwen2.5-1.5B", "Qwen2.5-3B", "Phi-3.5-mini"} and min(excl.values()) >= 0.5
    assert [a["anchor"] for a in out["per_anchor"]] == ["Llama-3.2-1B", "Llama-3.2-3B", "gemma-2-2b"]
    assert out["n_anchors"] == 3 and out["n_anchors_xfam"] == 1
    assert out["citizenship_xfam_partial_lex"] == out["per_anchor"][2]["partial_lex_rsa"]
    # allow_dominant turns both refusals off and says so
    off = mg.geometry_citizenship(q05, JUNE, "qwen", allow_dominant=True)
    assert off["status"] == "OK" and off["anchors_excluded"] == [] and off["n_anchors"] == 6
    assert off["dominance_check"].startswith("OFF")
    assert round(off["citizenship_xfam_partial_lex"], 4) == 0.1833     # the documented artifact number
    assert mg.geometry_citizenship(q05, V2, "qwen")["dominance_check"] == "on"


@need
def test_v4_pairs_equal_the_sealed_scorer():
    lex = json.loads(LEX.read_text(encoding="utf-8"))["original"]
    assert lex["words"] == mind.BATTERY and lex["counts"] == mind.BATTERY_REF_TOKLEN   # the same lexical control
    table = json.loads(SCORE.read_text(encoding="utf-8"))["tables"]["original"]
    v2 = _npz(V2)
    n = 0
    for a, b in itertools.combinations(FIVE, 2):
        row = [r for r in table if {r["a"], r["b"]} == {a, b}]
        assert len(row) == 1
        out = mg.geometry_citizenship(v2["fixed__" + a], V2, FAMILY[a])
        assert out["status"] == "OK" and out["anchors_excluded"] == []
        got = [p["partial_lex_rsa"] for p in out["per_anchor"] if p["anchor"] == b]
        assert abs(got[0] - row[0]["partial_lex_7"]) <= 1e-6, (a, b, got[0], row[0]["partial_lex_7"])
        n += 1
    assert n == 10


@need
def test_v5_within_category_null_matches_the_sealed_scorer():
    table = json.loads(SCORE.read_text(encoding="utf-8"))["tables"]["original"]
    v2 = _npz(V2)
    rows = [r for r in table if r["cross_family"] and r["a"] in FIVE and r["b"] in FIVE]
    assert len(rows) == 8
    for r in rows:
        # score_v2.py shuffled the second array of its pair; here that array is the candidate
        cand, anchor = v2["fixed__" + r["b"]], v2["fixed__" + r["a"]]
        run_a = mg.within_category_null(cand, anchor, n_perm=200, seed=0)
        again = mg.within_category_null(cand, anchor, n_perm=200, seed=0)
        assert run_a == again
        assert run_a["n_perm"] == 200 and run_a["seed"] == 0 and run_a["sd"] > 0
        assert abs(run_a["mean"] - r["within_category_null_7"]) <= 0.02, (r["a"], r["b"], run_a["mean"])
    other = mg.within_category_null(cand, anchor, n_perm=200, seed=1)
    assert other != run_a


def _no_start_tokenizer():
    vocab = {}

    def no_start(text):          # stand-in: a word of up to 5 letters is one token, a longer word two
        ids = []
        for piece in text.split(" "):
            for j in range(1 if len(piece) <= 5 else 2):
                ids.append(vocab.setdefault((piece, j), 100 + len(vocab)))
        return ids
    return no_start


def test_v6_start_token_exposure_with_stand_in_tokenizers():
    no_start = _no_start_tokenizer()

    def with_start(text):
        return [1] + no_start(text)

    single = [w for w in mind.BATTERY if len(w) <= 5]
    assert 0 < len(single) < len(mind.BATTERY)
    e = mg.start_token_exposure(no_start, templates=mind.TEMPLATES)
    assert e["n_exposed"] == len(single) and e["n_inputs"] == 8 * 96
    assert e["exposed"] == [("{w}", w) for w in single]
    assert e["by_template"]["{w}"] == len(single) and sum(e["by_template"].values()) == len(single)
    assert mg.start_token_exposure(with_start, templates=mind.TEMPLATES, bos_id=1)["n_exposed"] == 0
    assert mg.start_token_exposure(no_start)["n_exposed"] == 0
    assert mg.start_token_exposure(with_start, bos_id=1)["n_exposed"] == 0
    assert mg.start_token_exposure(no_start)["n_inputs"] == 7 * 96
    assert mg.TEMPLATES_V2 == mind.TEMPLATES[1:] and "{w}" not in mg.TEMPLATES_V2
    assert len(mind.TEMPLATES) == 8
    # a start token alone is not counted; an empty encoding is an error
    assert mg.start_token_exposure(lambda t: [1], words=["x"], templates=["{w}"], bos_id=1)["n_exposed"] == 0
    with pytest.raises(ValueError):
        mg.start_token_exposure(lambda t: [], words=["x"], templates=["{w}"])
    with pytest.raises(ValueError):
        mg.start_token_exposure(no_start, templates=["no slot"])


def test_v7_styxx_mind_is_unchanged(tmp_path):
    # styxx/mind.py is the file both lock tables pin (blob sha256, CRLF folded to LF as git stores it)
    blob = (REPO / "styxx" / "mind.py").read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(blob).hexdigest() == MIND_PIN
    for scorer in (TPC_SCORE, GATE_SCORE):
        if scorer.exists():
            assert MIND_PIN in scorer.read_text(encoding="utf-8"), scorer.name
    assert mind.geometry_citizenship is not mg.geometry_citizenship
    # numpy is imported inside the functions only: mind.py loaded alone pulls in no numpy
    code = ("import importlib.util, sys; spec = importlib.util.spec_from_file_location('m', sys.argv[1]); "
            "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); print('numpy' in sys.modules)")
    r = subprocess.run([sys.executable, "-B", "-c", code, str(REPO / "styxx" / "mind.py")],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "False"
    # mind_geometry.py keeps the same rule: no numpy import at module level
    tree = ast.parse((REPO / "styxx" / "mind_geometry.py").read_text(encoding="utf-8"))
    top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = [a.name for n in top for a in n.names] + [n.module or "" for n in top if isinstance(n, ast.ImportFrom)]
    assert not any(re.match(r"numpy\b", x) for x in names), names
    # tests/test_mind.py is byte for byte as at commit 4bbe4b3e (line endings aside)
    old_test = _git_show(OLD_COMMIT + ":tests/test_mind.py")
    if old_test is None:
        pytest.skip("git or commit 4bbe4b3e is not at hand")
    now_test = (REPO / "tests" / "test_mind.py").read_bytes().replace(b"\r\n", b"\n")
    assert now_test == old_test.replace(b"\r\n", b"\n")


# ---------------------------------------------------------------------------------------------------------------
# edge cases
# ---------------------------------------------------------------------------------------------------------------

def _clean(seed=0, d=16):
    return np.random.default_rng(seed).normal(size=(len(mind.BATTERY), d))


def test_non_finite_input_raises(tmp_path):
    bad = _clean()
    bad[3, 2] = np.nan
    with pytest.raises(ValueError):
        mg.dominant_direction_share(bad)
    bad[3, 2] = np.inf
    with pytest.raises(ValueError):
        mg.dominant_direction_share(bad)
    p = tmp_path / "anchors.npz"
    np.savez(p, **{"fixed__Llama-3.2-1B": _clean(1)})
    with pytest.raises(ValueError):
        mg.geometry_citizenship(bad, p, "?")
    with pytest.raises(ValueError):
        mg.within_category_null(bad, _clean(1), n_perm=2)
    np.savez(p, **{"fixed__Llama-3.2-1B": bad})
    with pytest.raises(ValueError):
        mg.geometry_citizenship(_clean(), p, "?")


def test_wrong_rows_and_degenerate_arrays_raise(tmp_path):
    p = tmp_path / "anchors.npz"
    np.savez(p, **{"fixed__Llama-3.2-1B": _clean(1)})
    with pytest.raises(ValueError):
        mg.geometry_citizenship(_clean()[:95], p, "?")
    with pytest.raises(ValueError):
        mg.within_category_null(_clean()[:95], _clean(1), n_perm=2)
    with pytest.raises(ValueError):
        mg.dominant_direction_share(np.ones((2, 5)) * np.arange(5))
    with pytest.raises(ValueError, match="below 1e-20"):
        mg.dominant_direction_share(np.zeros((10, 4)))
    with pytest.raises(ValueError, match="below 1e-20"):
        mg.dominant_direction_share(np.tile(np.arange(4.0), (10, 1)))
    with pytest.raises(ValueError):
        mg.dominant_direction_share(np.arange(10.0))
    with pytest.raises(ValueError):
        mg.within_category_null(_clean(), _clean(1), n_perm=0)
    # a rank-one array is all on one direction
    rank1 = np.outer(np.arange(1.0, 11.0), np.ones(6))
    assert mg.dominant_direction_share(rank1) == pytest.approx(1.0)


def test_zero_variance_error_names_the_array(tmp_path):
    const = np.tile(np.arange(16.0), (len(mind.BATTERY), 1))
    p = tmp_path / "anchors.npz"
    np.savez(p, **{"fixed__Llama-3.2-1B": _clean(1)})
    with pytest.raises(ValueError, match="^candidate_reps: .*below 1e-20"):
        mg.geometry_citizenship(const, p, "?")
    np.savez(p, **{"fixed__Llama-3.2-1B": const})
    with pytest.raises(ValueError, match="^anchor fixed__Llama-3.2-1B: .*below 1e-20"):
        mg.geometry_citizenship(_clean(), p, "?")


def test_complex_input_raises(tmp_path):
    z = _clean().astype(complex)
    with pytest.raises(ValueError, match="complex"):
        mg.dominant_direction_share(z)
    with pytest.raises(ValueError, match="complex"):
        mg.within_category_null(z, _clean(1), n_perm=2)
    p = tmp_path / "anchors.npz"
    np.savez(p, **{"fixed__Llama-3.2-1B": _clean(1)})
    with pytest.raises(ValueError, match="complex"):
        mg.geometry_citizenship(z, p, "?")


def test_integer_arguments_reject_floats_and_bools(tmp_path):
    p = tmp_path / "anchors.npz"
    np.savez(p, **{"fixed__Llama-3.2-1B": _clean(1)})
    for bad in (2.7, True, "3"):
        with pytest.raises(TypeError):
            mg.geometry_citizenship(_clean(), p, "?", null_perms=bad)
        with pytest.raises(TypeError):
            mg.within_category_null(_clean(), _clean(1), n_perm=bad)
    assert mg.within_category_null(_clean(), _clean(1), n_perm=np.int64(2))["n_perm"] == 2


def test_start_token_exposure_rejects_a_string_and_duplicates():
    no_start = _no_start_tokenizer()
    with pytest.raises(TypeError):
        mg.start_token_exposure(no_start, templates="{w}")
    with pytest.raises(TypeError):
        mg.start_token_exposure(no_start, words="dog")
    with pytest.raises(ValueError, match="more than once"):
        mg.start_token_exposure(no_start, templates=["{w}", "a {w}", "{w}"])


def test_anchors_file_with_none_of_the_expected_keys(tmp_path):
    p = tmp_path / "anchors.npz"
    np.savez(p, foo=_clean(1), fixed__unknown=_clean(2))
    out = mg.geometry_citizenship(_clean(), p, "?")
    assert out["status"] == "NO_ANCHORS" and out["citizenship_xfam_partial_lex"] is None
    assert out["n_anchors"] == 0 and out["anchor_shares"] == {} and "fixed__Qwen2.5-1.5B" in out["reason"]
    june = mg.geometry_citizenship(_clean(), p, "?", convention="june")
    assert june["citizenship_xfam_partial_lex"] is None and june["n_anchors"] == 0


def test_only_same_family_anchor_left(tmp_path):
    p = tmp_path / "anchors.npz"
    np.savez(p, **{"fixed__Llama-3.2-1B": _clean(1)})
    out = mg.geometry_citizenship(_clean(), p, "llama")
    assert out["status"] == "NO_ANCHORS" and out["n_anchors"] == 1 and out["n_anchors_xfam"] == 0
    assert out["citizenship_xfam_partial_lex"] is None


def test_candidate_equal_to_an_anchor(tmp_path):
    p = tmp_path / "anchors.npz"
    a, b = _clean(1), _clean(2)
    np.savez(p, **{"fixed__Llama-3.2-1B": a, "fixed__Phi-3.5-mini": b})
    out = mg.geometry_citizenship(a.copy(), p, "?")
    assert out["status"] == "OK" and out["candidate_matches_anchor"] == ["Llama-3.2-1B"]
    rsa = {r["anchor"]: r["partial_lex_rsa"] for r in out["per_anchor"]}
    assert rsa["Llama-3.2-1B"] == pytest.approx(1.0, abs=1e-9) and abs(rsa["Phi-3.5-mini"]) < 0.5
    out = mg.geometry_citizenship(a.copy(), p, "llama")
    assert out["n_anchors_xfam"] == 1 and out["citizenship_xfam_partial_lex"] == rsa["Phi-3.5-mini"]


def test_null_perms_adds_the_null_and_the_excess(tmp_path):
    p = tmp_path / "anchors.npz"
    np.savez(p, **{"fixed__Llama-3.2-1B": _clean(1), "fixed__Phi-3.5-mini": _clean(2)})
    out = mg.geometry_citizenship(_clean(), p, "?", null_perms=5)
    for row in out["per_anchor"]:
        assert row["within_category_null"]["n_perm"] == 5 and row["within_category_null"]["seed"] == 0
        assert row["excess_over_null"] == row["partial_lex_rsa"] - row["within_category_null"]["mean"]
    assert out["excess_xfam_mean"] == pytest.approx(np.mean([r["excess_over_null"] for r in out["per_anchor"]]))
    assert "excess_xfam_mean" not in mg.geometry_citizenship(_clean(), p, "?")


def test_convention_arguments(tmp_path):
    p = tmp_path / "anchors.npz"
    np.savez(p, **{"fixed__Llama-3.2-1B": _clean(1)})
    with pytest.raises(ValueError):
        mg.geometry_citizenship(_clean(), p, "?", convention="june", allow_dominant=True)
    with pytest.raises(ValueError):
        mg.geometry_citizenship(_clean(), p, "?", convention="june", null_perms=3)
    with pytest.raises(ValueError):
        mg.geometry_citizenship(_clean(), p, "?", convention="v3")
    with pytest.raises(ValueError):
        mg.geometry_citizenship(_clean(), p, "?", null_perms=-1)


@need
def test_command_line_prints_the_shares(capsys):
    assert mg.main(["geometry", "--anchors", str(JUNE)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["convention"] == "v2" and out["status"] == "REFUSED"          # fixed__Qwen2.5-1.5B, share 0.9971
    assert set(out["anchor_shares"]) == {n for n, _f in mind.ANCHOR_MODELS}
    assert mg.main(["geometry", "--anchors", str(V2)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "OK" and out["n_anchors_xfam"] == 3 and out["anchors_excluded"] == []
    assert mg.main(["geometry", "--anchors", str(JUNE), "--convention", "june"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["citizenship_xfam_partial_lex"] == 0.1801 and "anchor_shares" in out
    # the CLI of styxx.mind is the June one, unchanged
    assert mind.main(["geometry", "--anchors", str(JUNE)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["citizenship_xfam_partial_lex"] == 0.1801 and "convention" not in out
