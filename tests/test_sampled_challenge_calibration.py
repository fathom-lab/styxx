"""Guards for the sampled-challenge calibration under papers/v8/sampled_challenge_2026_09_09/.

The calibration's claims are claims about a published artifact, so these tests pin them to it:
that the honest certificate is clean, that the detection formula is the one Monte Carlo agrees
with, that the classes reported as misses really are missed by both detectors, and that the
classes reported as free catches really are caught.

A test here failing means either the artifact moved or the calibration's report is wrong.  Both
are findings; neither is repaired by loosening the assertion.

Run only this file:  python -m pytest tests/test_sampled_challenge_calibration.py -q
"""
from __future__ import annotations

import itertools
import pathlib
import random
import sys

import pytest

CAL = (pathlib.Path(__file__).resolve().parents[1]
       / "papers" / "v8" / "sampled_challenge_2026_09_09" / "calibration")

if not CAL.exists():                                            # pragma: no cover
    pytest.skip("calibration directory absent", allow_module_level=True)

sys.path.insert(0, str(CAL))

import artifact as A                                            # noqa: E402
import forgeries as F                                           # noqa: E402


@pytest.fixture(scope="module")
def honest():
    return A.load_cert(A.BF16[0])


@pytest.fixture(scope="module")
def items(honest):
    return A.items_of(honest)


@pytest.fixture(scope="module")
def check_item():
    return A.load_battery_module().check_item


# --------------------------------------------------------------------- the artifact

def test_target_shape(items):
    assert len(items) == 64
    partial = [i for i, it in items.items() if F.is_partial(it)]
    free = sum(it["n_generated"] - F.coverage(it)[1] for it in items.values())
    # the numbers the calibration report quotes for this one certificate
    assert len(partial) == 27
    assert len(items) - len(partial) == 37
    assert free == 193


def test_honest_certificate_is_clean(honest, check_item):
    caught, fails = A.internal_verdict(honest, check_item)
    assert not caught, f"the published certificate fails its own battery: {fails}"


def test_same_batch_pairs_agree_exactly():
    """The false-alarm measurement the challenge protocol rests on."""
    pairs = A.same_batch_reproducibility()["pairs"]
    same = [p for p in pairs if p["batch_size"][0] == p["batch_size"][1]]
    diff = [p for p in pairs if p["batch_size"][0] != p["batch_size"][1]]
    assert len(same) == 3 and len(diff) == 7
    for p in same:
        assert p["digest_disagreements"] == 0
        assert p["full_record_disagreements"] == 0
    for p in diff:
        assert 1 <= p["digest_disagreements"] <= 3
        assert p["full_record_disagreements"] == 64


# --------------------------------------------------------------------- the detector

@pytest.mark.parametrize("b,k", [(1, 8), (2, 4), (3, 16), (8, 4), (27, 2)])
def test_closed_form_matches_drawing(b, k):
    assert A.catch_prob(b, k) == pytest.approx(A.montecarlo_catch(b, k), abs=0.015)


def test_catch_prob_edges():
    assert A.catch_prob(0, 64) == 0.0          # nothing marked -> nothing to find
    assert A.catch_prob(1, 0) == 0.0
    for b in (1, 2, 27, 64):
        assert A.catch_prob(b, 64) == 1.0      # k=64 is the whole battery
    assert A.catch_prob(1, 16) == pytest.approx(0.25)


# --------------------------------------------------------------------- the miss list

MISSED_BY_BOTH = ["seqlp_partial_under", "seqlp_partial_to_prefix", "topk_reshape_coordinated"]


@pytest.mark.parametrize("name", MISSED_BY_BOTH)
@pytest.mark.parametrize("f", [1, 4])
def test_reported_misses_are_missed(name, f, honest, items, check_item):
    """Both detectors blind, on every forgery in a small population."""
    fn = dict((n, g) for n, g, _, _ in F.CLASSES)[name]
    rng = random.Random(f"guard:{name}:{f}")
    for _ in range(25):
        forged = fn(honest, f, rng)
        assert forged["body"]["items"] != honest["body"]["items"], "vacuous forgery"
        caught, _ = A.internal_verdict(forged, check_item)
        assert not caught, f"{name} f={f}: the battery caught it; the miss list is stale"
        d = A.differing(A.items_of(forged), items, A.DIGEST_FIELDS)
        assert d == [], f"{name} f={f}: {len(d)} digests moved; a challenge would reach it"


@pytest.mark.parametrize("name", ["fabricate_careless", "seqlp_tied", "seqlp_partial_over",
                                  "truncate_careless"])
def test_reported_free_catches_are_caught(name, honest, check_item):
    """The classes the report says not to spend a re-run on."""
    fn = dict((n, g) for n, g, _, _ in F.CLASSES)[name]
    rng = random.Random(f"guard:{name}")
    for _ in range(25):
        caught, _ = A.internal_verdict(fn(honest, 1, rng), check_item)
        assert caught, f"{name}: the battery missed it; it is not a free catch"


@pytest.mark.parametrize("f", [1, 2, 4, 8])
def test_fabrication_is_invisible_internally_and_visible_to_a_challenge(f, honest, items,
                                                                       check_item):
    rng = random.Random(f"guard:fab:{f}")
    for _ in range(25):
        forged = F.fabricate_careful(honest, f, rng)
        caught, _ = A.internal_verdict(forged, check_item)
        assert not caught, "a carefully fabricated item should pass an item-local battery"
        assert len(A.differing(A.items_of(forged), items, A.DIGEST_FIELDS)) == f


def test_text_rewrite_moves_no_token_id(honest, items):
    """H1: the rendered answer is replaceable without touching anything a channel compares."""
    forged = F.text_only_rewrite(honest, 8, random.Random("guard:text"))
    fi = A.items_of(forged)
    assert len(A.differing(fi, items, ("output_sha256",))) == 8
    assert A.differing(fi, items, ("token_ids", "token_ids_sha256", "seq_logprob")) == []


# --------------------------------------------------------------------- the worked attack

def test_published_verdict_reproduces_and_one_float_flips_a_channel(items):
    """The seqlp channel of the published verdict, moved by one permitted write."""
    from styxx.v8 import distances as D

    roles = A.roles_of()
    runs = {i: A.items_of(A.load_cert(p)) for i, p in A.BF16.items()}
    fp16 = A.items_of(A.load_cert(A.FP16))

    assert D.seqlp(runs[0], fp16) == pytest.approx(0.058564664, abs=5e-9)
    assert D.exact(runs[0], fp16, roles)[0] == pytest.approx(0.0625, abs=5e-9)
    assert D.topk(runs[0], fp16) == pytest.approx(1.407087824, abs=5e-9)
    floor = max(D.seqlp(runs[i], runs[j]) for i, j in itertools.combinations(range(5), 2))
    assert floor == pytest.approx(0.036070694, abs=5e-9)
    assert D.seqlp(runs[0], fp16) > floor                       # published: exceeds_floor

    iid = "a1f4a82542815cc4"
    it = runs[0][iid]
    prefix, covered = F.coverage(it)
    assert covered == 8 and it["n_generated"] == 14              # in the unrecorded-tail region
    assert fp16[iid]["seq_logprob"] <= prefix                    # P8b permits the write

    forged = {i: dict(r) for i, r in runs.items()}
    forged[0] = dict(runs[0])
    forged[0][iid] = dict(it, seq_logprob=fp16[iid]["seq_logprob"])
    new_floor = max(D.seqlp(forged[i], forged[j]) for i, j in itertools.combinations(range(5), 2))
    assert D.seqlp(forged[0], fp16) < new_floor                 # flipped to: same
    # and no digest moved
    assert A.differing(forged[0], runs[0], A.DIGEST_FIELDS) == []
