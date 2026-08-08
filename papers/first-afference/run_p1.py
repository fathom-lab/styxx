"""P1 — the styxx.power exam, per PREREG_p1_power_refusal_2026_08_08.md.

Two families scored separately: three documented historical bar failures (IN-SAMPLE, disclosed)
and a synthetic battery whose ground truth follows from its construction. Constant-answer
baselines are scored on the same synthetic battery, because the apparatus failure was invisible
until a constant was scored alongside it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from styxx.power import PowerError, effective_n, reachable   # noqa: E402

PREREG = "PREREG_p1_power_refusal_2026_08_08.md"
SMOKE = "--smoke" in sys.argv
SEED = 11081


def _call(**kw):
    """Return (flagged_unreachable, verdict) or (None, error) if the module refused."""
    try:
        r = reachable(**kw)
        return (not r.reachable), r.verdict
    except PowerError as e:
        return None, f"REFUSED: {e}"


# ---------------------------------------------------------------- historical (IN-SAMPLE, n=3)
def historical(rng):
    """The three documented bar failures, reconstructed from their committed receipts."""
    cases = []

    # b48-G2: a single-draw bar applied to the MAXIMUM of 45 null draws, on a 96-item
    # proportion. Receipt b48_result.json: chance 0.0104, bar 0.0208, max null 0.0521.
    nulls96 = rng.integers(0, 6, size=4000) / 96.0        # discrete, floored at 1/96
    f, v = _call(null=nulls96, bar=0.0208, op="<=", alt=None, n_draws=45, n_items=96)
    cases.append({"case": "b48-G2", "source": "papers/disjoint-worlds/b48_result.json",
                  "defect": "single-draw bar read as the max of 45 draws",
                  "truth_unreachable": True, "flagged": f, "verdict": v})

    # C5-G1: 80% of individual pairs significant, at an effective sample size of 6.9 to 45.8
    # against a nominal 300. The alternative is what the apparatus actually produces.
    n_eff = 6.9
    alt_c5 = rng.beta(2.0, 5.0, size=2000) * (n_eff / 300.0) + 0.02
    null_c5 = rng.beta(2.0, 5.0, size=2000) * 0.05
    f, v = _call(null=null_c5, bar=0.80, op=">=", alt=alt_c5)
    cases.append({"case": "C5-G1", "source": "papers/first-afference/c5_result.json",
                  "defect": "80% of pairs significant at effective n between 6.9 and 45.8",
                  "truth_unreachable": True, "flagged": f, "verdict": v})

    # b37-G2: a bar above the ceiling the apparatus can produce at all.
    null_b37 = rng.normal(0.0, 0.05, size=2000)
    alt_b37 = rng.normal(0.30, 0.08, size=2000)
    f, v = _call(null=null_b37, bar=0.90, op=">=", alt=alt_b37)
    cases.append({"case": "b37-G2", "source": "papers/disjoint-worlds/b37_result.json",
                  "defect": "bar demanded an effect the apparatus cannot produce",
                  "truth_unreachable": True, "flagged": f, "verdict": v})
    return cases


# ---------------------------------------------------------------- synthetic (held out)
def synthetic(rng):
    """Bars whose reachability follows from the construction, assigned before the module runs."""
    cases = []

    def add(name, truth, **kw):
        f, v = _call(**kw)
        cases.append({"case": name, "truth_unreachable": truth, "flagged": f, "verdict": v})

    for i in range(3):
        null = rng.normal(0.0, 1.0, size=1500)
        alt = rng.normal(3.0, 1.0, size=1500)
        # UNREACHABLE: bar beneath the null median concedes ~half the nulls
        add(f"below_null_median_{i}", True, null=null, bar=-0.2, op=">=", alt=alt)
        # REACHABLE: bar in the null's upper tail, well inside the alternative
        add(f"clean_separation_{i}", False, null=null, bar=2.0, op=">=", alt=alt)

    for i in range(3):
        # UNREACHABLE: bar above the ceiling of a bounded proportion
        null = rng.integers(0, 8, size=1500) / 50.0
        alt = rng.integers(20, 46, size=1500) / 50.0
        add(f"above_bounded_ceiling_{i}", True, null=null, bar=1.4, op=">=", alt=alt, n_items=50)
        # REACHABLE: bar on an attainable value inside the alternative
        add(f"bounded_attainable_{i}", False, null=null, bar=0.40, op=">=", alt=alt, n_items=50)

    for i, n in enumerate((10, 45, 200)):
        null = rng.integers(0, 6, size=1500) / 96.0
        alt = rng.integers(10, 30, size=1500) / 96.0
        naive = float(np.quantile(null, 0.95))
        # UNREACHABLE: alpha-correct for one draw, read as the max of n
        add(f"order_stat_naive_n{n}_{i}", True, null=null, bar=naive, op=">=", alt=alt,
            n_draws=n, n_items=96)
        # REACHABLE: the same construction with the order-statistic-corrected bar
        corr = float(np.quantile(null, (1.0 - 0.05) ** (1.0 / n)))
        add(f"order_stat_corrected_n{n}_{i}", False, null=null, bar=max(corr, 0.0625), op=">=",
            alt=alt, n_draws=n, n_items=96)

    for i in range(3):
        # UNREACHABLE: the effect exists but is far too small for the bar
        null = rng.normal(0.0, 1.0, size=1500)
        add(f"underpowered_{i}", True, null=null, bar=2.5, op=">=",
            alt=rng.normal(0.4, 1.0, size=1500))
        # REACHABLE: same bar, an effect large enough to clear it
        add(f"powered_{i}", False, null=null, bar=2.5, op=">=",
            alt=rng.normal(5.0, 1.0, size=1500))

    # UNREACHABLE: no alternative supplied at all -- alpha alone cannot license a bar
    for i in range(3):
        add(f"no_alternative_{i}", True, null=rng.normal(0, 1, size=1500), bar=2.0, op=">=")
    return cases


# ---------------------------------------------------------------- degenerate inputs
def degenerate(rng):
    """Inputs on which any verdict would have no basis. The module must refuse all of them."""
    good = rng.normal(0, 1, size=500)
    trials = [
        ("too_few_draws", dict(null=[0.1, 0.2, 0.3], bar=0.5, op=">=")),
        ("zero_variance", dict(null=np.full(500, 0.7), bar=0.5, op=">=")),
        ("nan_in_null", dict(null=np.r_[good, np.nan], bar=0.5, op=">=")),
        ("inf_in_null", dict(null=np.r_[good, np.inf], bar=0.5, op=">=")),
        ("empty_null", dict(null=[], bar=0.5, op=">=")),
        ("nan_bar", dict(null=good, bar=float("nan"), op=">=")),
        ("bool_bar", dict(null=good, bar=True, op=">=")),
        ("bad_op", dict(null=good, bar=0.5, op="~=")),
        ("alpha_zero", dict(null=good, bar=0.5, op=">=", alpha=0.0)),
        ("alpha_one", dict(null=good, bar=0.5, op=">=", alpha=1.0)),
        ("too_few_alt", dict(null=good, bar=0.5, op=">=", alt=[0.9, 0.95])),
        ("nan_in_alt", dict(null=good, bar=0.5, op=">=", alt=np.r_[good, np.nan])),
    ]
    out = []
    for name, kw in trials:
        f, v = _call(**kw)
        out.append({"case": name, "refused": f is None, "verdict": v})
    return out


def balanced_accuracy(cases):
    """Mean of sensitivity and specificity. A refusal counts as a miss, never as a pass."""
    pos = [c for c in cases if c["truth_unreachable"]]
    neg = [c for c in cases if not c["truth_unreachable"]]
    sens = np.mean([c["flagged"] is True for c in pos]) if pos else float("nan")
    spec = np.mean([c["flagged"] is False for c in neg]) if neg else float("nan")
    return round(float((sens + spec) / 2), 4), round(float(sens), 4), round(float(spec), 4)


def main() -> int:
    rng = np.random.default_rng(SEED)
    hist, syn, deg = historical(rng), synthetic(rng), degenerate(rng)
    if SMOKE:
        syn, deg = syn[:4], deg[:2]

    ba, sens, spec = balanced_accuracy(syn)
    n_pos = sum(c["truth_unreachable"] for c in syn)
    n_neg = len(syn) - n_pos
    # A constant answer scores sensitivity 1 / specificity 0, or the reverse: 0.5 either way.
    const_all_unreachable = round((1.0 + 0.0) / 2, 4)
    const_all_reachable = round((0.0 + 1.0) / 2, 4)
    best_const = max(const_all_unreachable, const_all_reachable)

    res = {"prereg": PREREG, "smoke": SMOKE, "seed": SEED,
           "historical_cases": hist, "synthetic_cases": syn, "degenerate_cases": deg,
           "n_cases_scored": len(hist) + len(syn),
           "n_synthetic": len(syn), "n_synthetic_unreachable": int(n_pos),
           "n_synthetic_reachable": int(n_neg),
           "historical_caught": int(sum(c["flagged"] is True for c in hist)),
           "historical_n": len(hist),
           "synthetic_balanced_accuracy": ba,
           "synthetic_sensitivity": sens, "synthetic_specificity": spec,
           "constant_all_unreachable_balanced_accuracy": const_all_unreachable,
           "constant_all_reachable_balanced_accuracy": const_all_reachable,
           "best_constant_balanced_accuracy": best_const,
           "synthetic_balanced_accuracy_minus_best_constant": round(ba - best_const, 4),
           "degenerate_refusal_rate": round(
               float(np.mean([c["refused"] for c in deg])), 4) if deg else 0.0,
           "n_degenerate": len(deg),
           "disclosure": ("historical cases are IN-SAMPLE: the module was written by someone who "
                          "knew all three. They are a sanity check on the implementation and "
                          "carry no evidential weight about generalisation.")}

    # The C-series construction, reported but ungated: effective n on a known-autocorrelated series
    try:
        ar = np.zeros(300)
        for i in range(1, 300):
            ar[i] = 0.8 * ar[i - 1] + rng.normal()
        res["effective_n_demo"] = effective_n(ar)
    except PowerError as e:
        res["effective_n_demo"] = {"refused": str(e)}

    try:
        from styxx.protocol import Experiment
        e = Experiment(HERE / PREREG, require_power_basis=True)
        res["metric_check"] = e.check_metrics(res)
        unusable = sorted(n for n, d in res["metric_check"].items() if not d["usable"])
        if unusable and not SMOKE:
            raise SystemExit(f"gate metrics unresolvable before scoring: {unusable}")
        v = e.score(res, smoke=SMOKE)
        res["verdict"], res["gates"] = v.verdict, v.gates
        res["prereg_commit"] = v.prereg_commit
    except Exception as exc:
        res["verdict"] = f"UNSCORED__{type(exc).__name__}: {exc}"

    (HERE / f"p1_result{'_smoke' if SMOKE else ''}.json").write_text(
        json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"historical (IN-SAMPLE) {res['historical_caught']}/{res['historical_n']}")
    print(f"synthetic balanced accuracy {ba} (sens {sens} spec {spec}) "
          f"over {n_pos} unreachable / {n_neg} reachable")
    print(f"best constant {best_const} | margin {res['synthetic_balanced_accuracy_minus_best_constant']}")
    print(f"degenerate refusal rate {res['degenerate_refusal_rate']} over {len(deg)} inputs")
    for c in syn:
        if c["flagged"] is not (not c["truth_unreachable"]) and \
           c["flagged"] is not c["truth_unreachable"]:
            print(f"  REFUSED unexpectedly: {c['case']} -> {c['verdict'][:70]}")
        elif bool(c["flagged"]) is not bool(c["truth_unreachable"]):
            print(f"  MISS: {c['case']} truth_unreachable={c['truth_unreachable']} "
                  f"-> {c['verdict'][:70]}")
    print(f"VERDICT: {res['verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
