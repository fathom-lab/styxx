# -*- coding: utf-8 -*-
"""
EXPLORE_anchor_threshold_2026_07_23.py  (exploratory; not a gated claim)

The paper's impossibility: at J=3 a correlated (shared-blind-spot) panel and an independent (good)
panel can induce IDENTICAL unlabeled verdict distributions -- consensus cannot tell them apart. This
probe asks the creative follow-up: HOW LITTLE does it take to break that blindness? Concretely, how
many KNOWN-NEGATIVE anchors are needed to distinguish "independent & good" from "synchronized &
secretly wrong", and is there a sharp threshold / a single-anchor smoking gun?

Construction (exchangeable J=3, exact-matchable on the vote-count sufficient statistic):
  BENIGN world A: judges conditionally independent; sensitivity s, false-positive f, prevalence piA.
  MALIGN world B: a fraction tau of the true NEGATIVES are "traps" on which ALL judges vote 1
    (unanimous wrong -- a shared blind spot); the rest behave like A. piB, s, f chosen so the
    UNLABELED distribution of (# judges voting 1) EXACTLY matches world A.
Then: (1) verify the two worlds are unlabeled-identical; (2) show a consensus estimate is the same in
both (blind); (3) measure the power to separate them from K known-negative anchors, and the
single-anchor likelihood ratio of a unanimous-wrong negative.
"""
import numpy as np
from math import comb
from itertools import product

J = 3
RNG = np.random.default_rng(0)


def votecount_dist_A(piA, s, f):
    """Unlabeled P(k judges vote 1), k=0..J, for the independent world."""
    d = np.zeros(J + 1)
    for k in range(J + 1):
        d[k] = piA * comb(J, k) * s**k * (1 - s)**(J - k) + (1 - piA) * comb(J, k) * f**k * (1 - f)**(J - k)
    return d


def votecount_dist_B(piB, s, f, tau):
    """Malign world: negatives are trap (all-1) w.p. tau, else independent at f; positives at s."""
    d = np.zeros(J + 1)
    for k in range(J + 1):
        pos = comb(J, k) * s**k * (1 - s)**(J - k)
        neg_indep = comb(J, k) * f**k * (1 - f)**(J - k)
        neg = (tau if k == J else 0.0) + (1 - tau) * neg_indep
        d[k] = piB * pos + (1 - piB) * neg
    return d


def solve_malign(target, tau=0.20):
    """Find (piB, sB, fB) so world B's vote-count dist matches `target` (world A's) EXACTLY, at fixed
    tau. 3 free params vs 3 free cells (4th pinned by sum-to-1) -> generically an exact solution.
    Returns params and the max cell residual so the caller can VERIFY the match before claiming it."""
    from scipy.optimize import least_squares
    def resid(x):
        piB, sB, fB = x
        return votecount_dist_B(*[np.clip(v, 1e-4, 1 - 1e-4) for v in (piB, sB, fB)], tau)[:3] - target[:3]
    best = None
    for p0 in np.linspace(0.15, 0.85, 8):
        for s0 in (0.7, 0.8, 0.9):
            for f0 in (0.05, 0.15, 0.3):
                r = least_squares(resid, [p0, s0, f0], bounds=([1e-4] * 3, [1 - 1e-4] * 3))
                if best is None or r.cost < best.cost:
                    best = r
    piB, sB, fB = [float(np.clip(v, 1e-4, 1 - 1e-4)) for v in best.x]
    dB = votecount_dist_B(piB, sB, fB, tau)
    return piB, sB, fB, float(np.abs(dB - target).max())


print("=" * 84)
print("1. TWO UNLABELED-IDENTICAL WORLDS  (benign-independent vs malign-synchronized)")
print("=" * 84)
piA, s, f = 0.40, 0.90, 0.10     # non-symmetric so an exact malign match exists
tau = 0.15                       # 15% of negatives are shared-blind-spot traps (unanimous wrong)
dA = votecount_dist_A(piA, s, f)
piB, sB, fB, resid = solve_malign(dA, tau=tau)
dB = votecount_dist_B(piB, sB, fB, tau)
print(f"  BENIGN A:  prevalence={piA:.3f}  s={s:.2f}  f={f:.2f}  (independent)")
print(f"  MALIGN B:  prevalence={piB:.3f}  s={sB:.3f}  f={fB:.3f}  tau(traps)={tau:.2f}")
print(f"  unlabeled vote-count dist A: {np.round(dA,6)}")
print(f"  unlabeled vote-count dist B: {np.round(dB,6)}")
print(f"  max |A-B| over cells: {resid:.2e}   ", end="")
MATCHED = resid < 1e-6
print("<-- EXACT match: the two worlds are unlabeled-identical" if MATCHED
      else "<-- NOT matched; claims below are UNSUPPORTED (honesty guard)")
print(f"  TRUE prevalence differs by: {abs(piA-piB):.3f}  (they are genuinely different worlds)")
if not MATCHED:
    import sys; print("\n  ABORT: construction did not produce unlabeled-identical worlds. No claim made."); sys.exit(0)

print()
print("=" * 84)
print("2. CONSENSUS IS BLIND: an unlabeled estimator returns the SAME answer in both worlds")
print("=" * 84)
# a majority-vote 'fraction positive' consensus reads the same distribution in both -> same estimate
def majority_positive_rate(d):
    return d[2] + d[3]          # >= 2 of 3 judges vote 1
print(f"  majority-vote positive rate:  A={majority_positive_rate(dA):.4f}   B={majority_positive_rate(dB):.4f}")
print(f"  any unlabeled statistic is identical -> consensus CANNOT distinguish A from B.")
print(f"  it will report the same prevalence for a good panel and a secretly-broken one.")

print()
print("=" * 84)
print("3. ONE KNOWN-NEGATIVE BREAKS IT: unanimous-wrong on a true negative is a smoking gun")
print("=" * 84)
# P(a known-negative -> all J judges vote 1)
pA_unanim = f**J
pB_unanim = tau + (1 - tau) * fB**J
LR = pB_unanim / pA_unanim
print(f"  P(known-negative is unanimously mis-voted '1'):  A={pA_unanim:.4f}   B={pB_unanim:.4f}")
print(f"  single-anchor likelihood ratio (B vs A) of ONE unanimous-wrong negative: {LR:.1f}x")
print(f"  -> in the benign world this almost never happens; in the malign world it happens ~{pB_unanim*100:.0f}% of the time.")

print()
print("=" * 84)
print("4. THE THRESHOLD: power to separate the worlds vs number K of known-negative anchors")
print("=" * 84)
print("  Draw K known-negatives from the true world; test H0:benign vs H1:malign via the count of")
print("  unanimous-wrong anchors (a one-sided binomial test at alpha=0.05). Power = P(detect B | B).")
def power_detect(K, trials=4000):
    # decision rule: reject benign if #unanimous-wrong > threshold t, where t is the smallest count
    # with P(Binom(K, pA_unanim) >= t) <= 0.05
    from scipy.stats import binom
    t = 0
    while binom.sf(t - 1, K, pA_unanim) > 0.05:
        t += 1
    # power under malign
    return 1 - binom.cdf(t, K, pB_unanim), t
print(f"{'K anchors':>10} {'reject-if >':>12} {'power (detect B)':>18}")
for K in (1, 3, 5, 10, 20, 30, 50):
    pw, t = power_detect(K)
    print(f"{K:>10} {t:>12} {pw:>18.3f}")
print()
print("  reading: the impossibility is an impossibility about the UNLABELED MARGINAL. The class-")
print("  conditional -- what one known-negative samples -- is exactly where the worlds diverge. So the")
print("  cost of seeing a shared blind spot is not labels; it is a HANDFUL of known-negatives.")

# --- committed receipt so the numbers are OATH-certifiable ---
import json
from scipy.stats import binom as _binom
def _power(K):
    t = 0
    while _binom.sf(t - 1, K, pA_unanim) > 0.05:
        t += 1
    return round(1 - _binom.cdf(t, K, pB_unanim), 4)
receipt = {
    "match_residual": float(f"{resid:.2e}"),
    "benign": {"prevalence": round(piA, 4), "s": s, "f": f},
    "malign": {"prevalence": round(piB, 4), "s": round(sB, 4), "f": round(fB, 4), "tau": tau},
    "prevalence_diff": round(abs(piA - piB), 4),
    "majority_rate_A": round(float(majority_positive_rate(dA)), 4),
    "majority_rate_B": round(float(majority_positive_rate(dB)), 4),
    "p_unanimous_wrong_benign": round(pA_unanim, 4),
    "p_unanimous_wrong_malign": round(pB_unanim, 4),
    "single_anchor_LR": round(LR, 1),
    "power_by_K": {str(K): _power(K) for K in (1, 3, 5, 10, 20, 30, 50)},
}
import pathlib
pathlib.Path(pathlib.Path(__file__).parent / "anchor_threshold_result.json").write_text(json.dumps(receipt, indent=2))
print("\n  wrote anchor_threshold_result.json (receipt)")
