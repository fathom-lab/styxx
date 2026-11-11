"""styxx.mind_geometry: version 2 of the geometry axis of styxx.mind (2026-11-10).

styxx.mind.geometry_citizenship scores a candidate's concept geometry against stored anchor arrays. Every array it
used was the mean over 8 templates, and template 0 is the bare word "{w}". In a tokenizer that adds no start token a
word that is one token then sits alone at position 0. The correction of the June 2026 result that axis was built on
is papers/real-convergence/ERRATUM_real_convergence_2026_11_10.md.

This module is version 2 of that axis. It uses 7 templates (TEMPLATES_V2: the bare word dropped), refuses a
candidate and leaves out an anchor whose across-concept variance sits on one direction (DOMINANT_SHARE_REFUSE),
and says with every number that most of the agreement it reports is category structure.

styxx/mind.py is not changed: it stays byte for byte as at commit 4bbe4b3e, because two lock tables pin its
sha256 (section 14 of papers/tests-pass-ci/PREREG_tests_pass_ci_2026_11_02.md, read by
papers/tests-pass-ci/collect/tpc_score.py, and papers/darkflobi-gate/score.py). So
styxx.mind.geometry_citizenship is the withdrawn June convention, and its docstrings still carry the June wording.
geometry_citizenship(..., convention="june") here calls it unchanged. What changed and what was measured:
papers/mind-instrument/NOTE_geometry_axis_v2_2026_11_10.md.

CLI:
  python -m styxx.mind_geometry geometry --anchors REPS.npz    # anchor-set self-RSA (validation view), with the shares
"""
from __future__ import annotations

import hashlib
import json
import operator
from pathlib import Path

from . import mind as _mind
from .mind import ANCHOR_MODELS, BATTERY, BATTERY_CATEGORY, TEMPLATES, _lexical_Z, distmat, partial_corr

__all__ = [
    "TEMPLATES_V2", "DOMINANT_SHARE_REFUSE", "GEOMETRY_CATEGORY_NOTE", "JUNE_CONVENTION_NOTICE",
    "dominant_direction_share", "start_token_exposure", "within_category_null", "geometry_citizenship",
]

# The same templates as styxx.mind.TEMPLATES without the bare word "{w}". In a tokenizer that adds no start token
# (Qwen2.5, Phi-3.5, GPT-2 and Pythia among those this lab used), a word that is one token sits alone at position 0
# in the bare template. Qwen2.5 and Phi-3.5 carry a very large activation there (measured in the erratum of
# 2026-11-10; described by Sun, Chen, Kolter, Liu, arXiv 2402.17762); for GPT-2 and Pythia it was not measured.
# The 8-template mean of such a word is then that one state and little else. In every template kept here the word
# comes after at least one other token. start_token_exposure() counts the inputs that would still sit alone at
# position 0.
TEMPLATES_V2 = TEMPLATES[1:]

# A representation array whose across-concept variance has at least this share on one direction
# (dominant_direction_share) is refused as a candidate and left out as an anchor by
# geometry_citizenship(convention="v2"). How it was set: on 2026-11-10, after the shares below were known, so it is
# a cut chosen on these arrays and not a prediction. A screen that day covered the 47 stored 2-D float arrays with
# at least 20 rows and 32 columns in the 21 .npz and .npy files tracked at commit 4bbe4b3e. Of the stored arrays
# built with the bare template from tokenizers that add no start token, 8 are at or above 0.5 (0.8161 to 0.9984):
# fixed__Qwen2.5-1.5B, fixed__Qwen2.5-3B, fixed__Phi-3.5-mini and best__Phi-3.5-mini in
# papers/real-convergence/contextual_reps.npz, and Qwen2.5-0.5B, pythia-410m, gpt2 and gpt2-large in
# papers/mind-instrument/atlas_live_reps.npz. Two others, best__Qwen2.5-1.5B (0.1369) and best__Qwen2.5-3B
# (0.1358) of contextual_reps.npz, read at the layer with the best category structure and not at
# max(1, int(0.66 L)), are below it. So are the 20 unit-normalized arrays of papers/mind-instrument/normeq_reps.npz
# and telepathy_reps.npz, which average the bare template in after normalizing each template's state (0.0553 to
# 0.1074; the 14 of them from tokenizers that add no start token 0.0577 to 0.1074). No other screened array reached
# 0.37. The June anchors of Llama-3.2-1B, Llama-3.2-3B and gemma-2-2b have 0.0821 to 0.0966, and the 5 anchors of
# papers/mind-instrument/anchors_v2_2026_11_10.npz have 0.0758 to 0.0977. For the GPT-2 and Pythia arrays the large
# activation at position 0 was not measured; the share is the only evidence there.
# What this does not show: that 0.5 suits model families other than those named here; that an array below it is
# free of every artifact; that a candidate built with the bare template is refused (at another layer, or after
# per-template normalization, it can fall below 0.5); or that an array above it carries an artifact (an array
# whose concept geometry really lies on one direction would be refused too).
DOMINANT_SHARE_REFUSE = 0.5

GEOMETRY_CATEGORY_NOTE = (
    "Most cross-model agreement on this battery is category structure. In the sealed test of "
    "2026-11-10 (papers/real-convergence/ERRATUM_real_convergence_2026_11_10.md), on 96 fresh words, "
    "the 8 cross-family pairs among five models agreed at a raw RSA of 0.88 to 0.93, and shuffling "
    "words inside their own category still gave 0.63 to 0.69 (the within-category null). Only the "
    "part above that null, 0.22 to 0.28 on the r scale, can be agreement finer than the categories."
)

JUNE_CONVENTION_NOTICE = (
    "convention='june' reproduces the June 2026 numbers: 8 templates with the bare word, scored "
    "against the June anchors, by styxx.mind.geometry_citizenship unchanged. That convention is "
    "withdrawn. For tokenizers that add no start token its numbers carry the start-token artifact. See "
    "papers/real-convergence/ERRATUM_real_convergence_2026_11_10.md."
)

_NO_VARIANCE = "the variance across rows is below 1e-20 of the array's sum of squares: no direction can be measured"


def _real_array(reps, what: str):
    """`reps` as a float64 numpy array; ValueError naming `what` on complex input."""
    import numpy as np
    if np.iscomplexobj(reps):
        raise ValueError(f"{what}: the array is complex; a real-valued array is expected")
    return np.asarray(reps, dtype=float)


def _battery_array(reps, what: str):
    """`reps` as a float64 (len(BATTERY) x d) array, or ValueError naming `what`."""
    import numpy as np
    X = _real_array(reps, what)
    if X.ndim != 2 or X.shape[0] != len(BATTERY) or X.shape[1] < 1:
        raise ValueError(f"{what}: expected a ({len(BATTERY)} x d) array on the frozen battery, "
                         f"got shape {X.shape}")
    if not np.isfinite(X).all():
        raise ValueError(f"{what}: the array holds a non-finite value (nan or inf)")
    return X


def _count(value, name: str) -> int:
    """An integer argument: ints and numpy integers only (no bool, no float)."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got a bool")
    try:
        return operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}") from None


def dominant_direction_share(reps) -> float:
    """Share of the across-row variance that lies on the single largest direction.

    Centers the (rows x dims) array on its column means and returns the largest squared singular
    value over the sum of all of them: 1.0 when the rows differ from their mean along one direction
    only, near 1 / min(rows - 1, dims) when the variance is spread evenly. On the battery arrays
    screened on 2026-11-10, a share at or above DOMINANT_SHARE_REFUSE is what the start-token
    artifact gave; a share this high can have other causes (see the comment there for what that cut
    does and does not show). The share does not depend on the scale of the array. Raises ValueError
    on complex input, on fewer than 3 rows, on a non-finite value, and when the variance across rows
    is below 1e-20 of the array's sum of squares (rows that do not vary, or vary only at that level).
    """
    import numpy as np
    X = _real_array(reps, "array")
    if X.ndim != 2 or X.shape[1] < 1:
        raise ValueError(f"expected a 2-D (rows x dims) array, got shape {X.shape}")
    if X.shape[0] < 3:
        raise ValueError(f"need at least 3 rows to measure a dominant direction, got {X.shape[0]}")
    if not np.isfinite(X).all():
        raise ValueError("the array holds a non-finite value (nan or inf)")
    scale = float(np.abs(X).max())
    if scale == 0.0:
        raise ValueError(_NO_VARIANCE)
    X = X / scale
    s2 = np.linalg.svd(X - X.mean(0), compute_uv=False) ** 2
    total = float(s2.sum())
    if not total > 1e-20 * float((X * X).sum()):
        raise ValueError(_NO_VARIANCE)
    return float(s2[0] / total)


def _named_share(reps, what: str) -> float:
    try:
        return dominant_direction_share(reps)
    except ValueError as e:
        raise ValueError(f"{what}: {e}") from None


def start_token_exposure(encode, words=BATTERY, templates=TEMPLATES_V2, bos_id=None) -> dict:
    """Count the template inputs whose read-out token would sit alone at position 0.

    `encode` is any callable from text to a list of token ids, exactly as the model will see the
    input (with its start token, if the tokenizer adds one); no tokenizer library is imported here.
    An input is exposed when its encoding is exactly one token and that token is not `bos_id`:
    the representation is read at the last token, and that token is then alone at position 0.
    With the default TEMPLATES_V2 every input has at least two tokens in any tokenizer that gives
    each word of a phrase at least one token, so the count is 0; with styxx.mind.TEMPLATES it counts
    the bare words that are one token in a tokenizer that adds no start token. A count of 0 does not
    show that an extraction is free of other position effects.

    Returns n_inputs, n_exposed, n_words, n_templates, bos_id, by_template (template -> count) and
    exposed (the list of (template, word) pairs). Raises TypeError when `encode` is not callable or
    `words` or `templates` is a single string, and ValueError when a template has no "{w}" slot, a
    template is given twice, or `encode` returns no token.
    """
    if not callable(encode):
        raise TypeError("encode must be a callable from text to a list of token ids")
    for value, name in ((words, "words"), (templates, "templates")):
        if isinstance(value, (str, bytes)):
            raise TypeError(f"{name} must be a list of strings, not a single string")
    words, templates = list(words), list(templates)
    if len(set(templates)) != len(templates):
        dup = sorted({t for t in templates if templates.count(t) > 1})
        raise ValueError(f"template given more than once: {dup}")
    exposed, by_template, n = [], {}, 0
    for t in templates:
        if "{w}" not in t:
            raise ValueError(f"template {t!r} has no {{w}} slot")
        k = 0
        for w in words:
            text = t.replace("{w}", w)
            ids = list(encode(text))
            if not ids:
                raise ValueError(f"encode returned no token for {text!r}")
            n += 1
            if len(ids) == 1 and ids[0] != bos_id:
                exposed.append((t, w))
                k += 1
        by_template[t] = k
    return {"n_inputs": n, "n_exposed": len(exposed), "n_words": len(words),
            "n_templates": len(templates), "bos_id": bos_id, "by_template": by_template,
            "exposed": exposed}


def within_category_null(candidate_reps, anchor_reps, n_perm=200, seed=0) -> dict:
    """Partial-lexical RSA after shuffling the candidate's rows inside their own category.

    Each draw permutes the candidate's 96 rows within each of the 8 categories of
    styxx.mind.BATTERY_CATEGORY (so every word moves only to another word of its category) and
    computes the same partial-lexical RSA that geometry_citizenship reports. The mean over the draws
    is the agreement that category structure alone gives; the RSA minus that mean is the agreement
    finer than the categories. Deterministic for a given `seed` (numpy default_rng).

    Ported from the within-category null of the sealed scorer
    papers/real-convergence/artifact_2026_11_10/score_v2.py (within_cat_null), with the same
    permutation scheme: categories taken in battery order, one rng.permutation per category per
    draw, 200 draws and seed 0 by default. Where it differs: score_v2.py computes the raw Pearson
    r of the two distance vectors, this function the partial-lexical RSA (the number the axis
    reports); score_v2.py shuffles the second array of its pair, this function the candidate; and
    score_v2.py ran on its fresh words, this function on the frozen battery. On the 8 cross-family
    pairs of the original words its mean is within 0.02 of score_v2.py's (gate V5,
    tests/test_mind_geometry_v2.py).

    Returns mean, sd (population), n_perm, seed and the statistic's name. `n_perm` must be an
    integer of at least 1 (TypeError on a float or bool, ValueError below 1).
    """
    import numpy as np
    C = _battery_array(candidate_reps, "candidate_reps")
    A = _battery_array(anchor_reps, "anchor_reps")
    n_perm = _count(n_perm, "n_perm")
    if n_perm < 1:
        raise ValueError(f"n_perm must be at least 1, got {n_perm}")
    Zlex, iu = _lexical_Z()
    Da = distmat(A)[iu]
    cats = np.array(BATTERY_CATEGORY)
    groups = [np.where(cats == c)[0] for c in dict.fromkeys(BATTERY_CATEGORY)]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_perm):
        perm = np.arange(len(cats))
        for g in groups:
            perm[g] = rng.permutation(g)
        vals.append(partial_corr(distmat(C[perm])[iu], Da, Zlex))
    return {"mean": float(np.mean(vals)), "sd": float(np.std(vals)), "n_perm": n_perm, "seed": seed,
            "statistic": "partial-lexical RSA, candidate rows shuffled inside their own category"}


def _share_or_error(reps):
    try:
        return dominant_direction_share(reps)
    except ValueError as e:
        return f"not measured: {e}"


def geometry_citizenship(candidate_reps, anchors_npz: Path, candidate_family: str = "?", *,
                         convention: str = "v2", allow_dominant: bool = False,
                         null_perms: int = 0) -> dict:
    """Partial-lexical RSA of a candidate's battery geometry against each stored anchor model.

    `candidate_reps`: (96 x d) array on the frozen battery, fixed 0.66-layer, last token, averaged
    over the templates of the convention. Citizenship = mean partial-lexical RSA over the anchors
    whose family differs from `candidate_family`.

    convention="v2" (the default): the candidate should be averaged over TEMPLATES_V2, and the
    anchors file should be built the same way (papers/mind-instrument/anchors_v2_2026_11_10.npz).
    The share of the across-concept variance on one direction (dominant_direction_share) is
    measured for the candidate and for every anchor present. A candidate at or above
    DOMINANT_SHARE_REFUSE gets status REFUSED and no number; an anchor at or above it is left out of
    the mean and listed in anchors_excluded with its share. allow_dominant=True turns both off and
    says so in `dominance_check`. The output always carries the shares, the threshold, how many
    anchors were used and GEOMETRY_CATEGORY_NOTE. null_perms > 0 adds, per used anchor,
    within_category_null(candidate, anchor, null_perms, seed=0) and the excess of the RSA over its
    mean. Values are not rounded. status is OK (a number is given), REFUSED, or NO_ANCHORS (no
    cross-family anchor left to average; reason says why). Raises ValueError on a candidate or a
    stored anchor that is not a finite real (96 x d) array, or whose rows do not vary (the message
    names the array), and TypeError on a `null_perms` that is not an integer.
    candidate_matches_anchor names every anchor equal to the candidate, element for element. Such an
    anchor is still scored (its RSA is 1) and counts in the mean unless candidate_family names its
    family, so pass the candidate's family when it is one of the anchors. A scaled or near copy of
    an anchor is not named there.

    convention="june": styxx.mind.geometry_citizenship, the function as it shipped at commit
    4bbe4b3e, called unchanged (rounded as it was: per anchor to 3 decimals, citizenship to 4),
    8 templates, no refusal. This function adds to its output that the convention is withdrawn
    (JUNE_CONVENTION_NOTICE) and the shares for reading only. It takes neither allow_dominant nor
    null_perms.
    """
    import numpy as np
    if convention == "june":
        if allow_dominant or null_perms:
            raise ValueError("convention='june' reproduces the withdrawn numbers as they were and "
                             "takes neither allow_dominant nor null_perms")
        out = _mind.geometry_citizenship(candidate_reps, anchors_npz, candidate_family)
        with np.load(anchors_npz) as z:
            shares = {name: _share_or_error(z[f"fixed__{name}"])
                      for name, _f in ANCHOR_MODELS if f"fixed__{name}" in z}
        out.update({"convention": "june", "notice": JUNE_CONVENTION_NOTICE,
                    "templates": list(TEMPLATES), "threshold": DOMINANT_SHARE_REFUSE,
                    "candidate_share": _share_or_error(candidate_reps), "anchor_shares": shares})
        return out
    if convention != "v2":
        raise ValueError(f"unknown convention {convention!r}: use 'v2' (default) or 'june'")
    null_perms = _count(null_perms, "null_perms")
    if null_perms < 0:
        raise ValueError(f"null_perms must be 0 or more, got {null_perms}")
    C = _battery_array(candidate_reps, "candidate_reps")
    thr = DOMINANT_SHARE_REFUSE
    with np.load(anchors_npz, allow_pickle=False) as z:
        present = [(name, family, _battery_array(z[f"fixed__{name}"], f"anchor fixed__{name}"))
                   for name, family in ANCHOR_MODELS if f"fixed__{name}" in z]
    cand_share = _named_share(C, "candidate_reps")
    anchor_shares = {name: _named_share(A, f"anchor fixed__{name}") for name, _f, A in present}
    out = {
        "convention": "v2",
        "status": None,
        "reason": None,
        "citizenship_xfam_partial_lex": None,
        "templates": list(TEMPLATES_V2),
        "threshold": thr,
        "dominance_check": ("OFF: allow_dominant=True, so no candidate was refused and no anchor "
                            "left out for its share" if allow_dominant else "on"),
        "candidate_share": cand_share,
        "anchor_shares": anchor_shares,
        "anchors_excluded": [],
        "per_anchor": [],
        "n_anchors": 0,
        "n_anchors_xfam": 0,
        "candidate_matches_anchor": [name for name, _f, A in present
                                     if A.shape == C.shape and bool(np.array_equal(A, C))],
        "category_structure_note": GEOMETRY_CATEGORY_NOTE,
        "anchors_sha256": hashlib.sha256(Path(anchors_npz).read_bytes()).hexdigest(),
    }
    used = []
    for name, family, A in present:
        if not allow_dominant and anchor_shares[name] >= thr:
            out["anchors_excluded"].append({"anchor": name, "family": family,
                                            "share": anchor_shares[name]})
        else:
            used.append((name, family, A))
    if not present:
        out["status"] = "NO_ANCHORS"
        out["reason"] = ("none of the expected keys (" + ", ".join(f"fixed__{n}" for n, _f in ANCHOR_MODELS)
                         + ") is in the anchors file")
        return out
    if not allow_dominant and cand_share >= thr:
        out["status"] = "REFUSED"
        out["reason"] = (f"the candidate's across-concept variance has a share of {cand_share:.4f} on "
                         f"one direction, at or above the threshold {thr}. A share this high is what "
                         f"the start-token artifact (a word alone at position 0) gave in the arrays "
                         f"screened on 2026-11-10; it can have other causes. Rebuild the candidate "
                         f"over TEMPLATES_V2 and check start_token_exposure; allow_dominant=True "
                         f"scores it anyway.")
        return out
    Zlex, iu = _lexical_Z()
    Dc = distmat(C)[iu]
    xfam, xfam_excess = [], []
    for name, family, A in used:
        r = partial_corr(Dc, distmat(A)[iu], Zlex)
        row = {"anchor": name, "family": family, "share": anchor_shares[name],
               "cross_family": family != candidate_family, "partial_lex_rsa": r}
        if null_perms:
            nl = within_category_null(C, A, n_perm=null_perms, seed=0)
            row["within_category_null"] = nl
            row["excess_over_null"] = r - nl["mean"]
        out["per_anchor"].append(row)
        if family != candidate_family:
            xfam.append(r)
            if null_perms:
                xfam_excess.append(row["excess_over_null"])
    out["n_anchors"], out["n_anchors_xfam"] = len(used), len(xfam)
    if xfam:
        out["status"] = "OK"
        out["citizenship_xfam_partial_lex"] = float(np.mean(xfam))
        if null_perms:
            out["excess_xfam_mean"] = float(np.mean(xfam_excess))
    else:
        out["status"] = "NO_ANCHORS"
        out["reason"] = (f"no cross-family anchor is left to average: {len(present)} present, "
                         f"{len(out['anchors_excluded'])} left out for a share at or above {thr}, "
                         f"the rest of family {candidate_family!r}")
    return out


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="styxx.mind_geometry")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("geometry", help="anchor-set self-RSA (validation view): the anchor of the file "
                                        "listed earliest in ANCHOR_MODELS, scored against the whole file, "
                                        "with every share printed")
    g.add_argument("--anchors", required=True)
    g.add_argument("--convention", choices=["v2", "june"], default="v2",
                   help="v2 (default) or june (the withdrawn 8-template convention, to reproduce it)")
    g.add_argument("--allow-dominant", action="store_true",
                   help="v2 only: score arrays dominated by one direction instead of refusing them")
    g.add_argument("--null-perms", type=int, default=0,
                   help="v2 only: add the within-category null with this many draws (seed 0)")
    a = ap.parse_args(argv)
    import numpy as np
    if a.convention == "june" and (a.allow_dominant or a.null_perms):
        ap.error("--allow-dominant and --null-perms apply to --convention v2 only")
    with np.load(a.anchors) as z:
        top = next(((n, f) for n, f in ANCHOR_MODELS if f"fixed__{n}" in z), None)
        if top is None:
            print("no expected anchor key (fixed__<model> of ANCHOR_MODELS) in " + a.anchors)
            return 2
        reps = z[f"fixed__{top[0]}"].astype(float)
    kw = {} if a.convention == "june" else {"allow_dominant": a.allow_dominant, "null_perms": a.null_perms}
    out = geometry_citizenship(reps, Path(a.anchors), top[1], convention=a.convention, **kw)
    out = {"candidate": f"fixed__{top[0]} (an anchor of the file, scored against the whole file)", **out}
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
