import math
from styxx.three_axis.regen_scorer import _entropy_topk


class TLP:
    def __init__(self, lp): self.logprob = lp


def test_no_alternatives_refuses_instead_of_claiming_certainty():
    assert math.isnan(_entropy_topk([]))


def test_real_alternatives_still_measured():
    h = _entropy_topk([TLP(math.log(0.5)), TLP(math.log(0.5))])
    assert abs(h - math.log(2)) < 1e-9


def test_the_upstream_guard_is_no_longer_defeated():
    """The regression this fixes: a provider returning tokens WITHOUT
    top_logprobs used to fill Hs with real-looking zeros, so
    `sum(Hs)/len(Hs) if Hs else nan` saw a non-empty list and reported 0.0 --
    maximum certainty -- instead of refusing."""
    content = [object()] * 4
    Hs = [_entropy_topk([]) for _ in content]
    assert Hs and all(math.isnan(h) for h in Hs)
    mean = sum(Hs) / len(Hs) if Hs else float("nan")
    assert math.isnan(mean), "an unmeasured mean must not read as certainty"
