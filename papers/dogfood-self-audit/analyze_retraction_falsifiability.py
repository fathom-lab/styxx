"""The pre-registered test: do dead instruments predict retracted results?

Written BEFORE the exposure ledger finished building and before any module-level dead
rate was inspected. `PREREG_dead_gates_predict_retraction_2026_08_13.md` fixes every
choice this file implements; nothing here may be tuned after seeing output. The day's
own record contains one instance of a floor being applied to the wrong denominator to
rescue a p-value, and it was withdrawn -- writing the analysis first is the cheapest
defence against doing that again.

    python analyze_retraction_falsifiability.py --probe probe_e_styxx_v2.json \
        --ledger RETRACTION_LEDGER.json --json PREREG_RESULT.json

Reports, in this order and always together:
  - the primary test (H1): Mann-Whitney U on module-level adjudicative dead rate
  - the covariates the prereg named, each of which could produce H1 spuriously
  - H2 explicitly: is the exposed group simply less EXERCISED rather than more dead?

Refuses rather than reports when the pre-registered floor is not met.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

MIN_MODULES = 5          # from the prereg. NOT adjustable after seeing data.
MIN_POWERED_PER_MODULE = 1


def module_rates(probe_path):
    """Per-module adjudicative dead rate, plus the covariates the prereg named."""
    with open(probe_path, encoding="utf-8") as f:
        rep = json.load(f)
    by_mod = defaultdict(lambda: {"adj_powered": 0, "adj_dead": 0,
                                  "terms": 0, "powered": 0})
    for r in rep.get("rows", []):
        m = r.get("module") or "?"
        d = by_mod[m]
        d["terms"] += 1
        v = r.get("verdict")
        powered = v in ("LIVE", "CONSTANT_TRUE", "CONSTANT_FALSE")
        if powered:
            d["powered"] += 1
        if r.get("pos") == "adjudicative" and powered:
            d["adj_powered"] += 1
            if v != "LIVE":
                d["adj_dead"] += 1
    out = {}
    for m, d in by_mod.items():
        if d["adj_powered"] < MIN_POWERED_PER_MODULE:
            continue
        out[m] = {
            "dead_rate": d["adj_dead"] / d["adj_powered"],
            "adj_powered": d["adj_powered"],
            "adj_dead": d["adj_dead"],
            # H2's mechanism: coverage, not deadness
            "exercised_frac": (d["powered"] / d["terms"]) if d["terms"] else 0.0,
            "terms": d["terms"],
        }
    return out, rep


def normalise(m):
    """styxx.guardrail.deception -> guardrail/deception, so ledger and probe agree."""
    m = str(m or "").strip()
    for pre in ("styxx.", "styxx/"):
        if m.startswith(pre):
            m = m[len(pre):]
    return m.replace("\\", "/").replace(".py", "").replace(".", "/").lower()


def mannwhitney(a, b):
    """U test with a normal approximation and tie correction; returns (U, p, rbc).

    rbc is the rank-biserial correlation -- the effect size the prereg demands be
    reported instead of a bare p. Implemented here rather than imported so the test has
    no dependency that could silently differ between machines.
    """
    import math
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return None, None, None
    pooled = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    ranks, i = [0.0] * len(pooled), 0
    ties = []
    while i < len(pooled):
        j = i
        while j + 1 < len(pooled) and pooled[j + 1][0] == pooled[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = avg
        ties.append(j - i + 1)
        i = j + 1
    r1 = sum(rk for rk, (_, g) in zip(ranks, pooled) if g == 0)
    u1 = r1 - n1 * (n1 + 1) / 2
    u2 = n1 * n2 - u1
    u = min(u1, u2)
    mu = n1 * n2 / 2
    n = n1 + n2
    tie_term = sum(t ** 3 - t for t in ties)
    sd = math.sqrt(n1 * n2 / 12 * ((n + 1) - tie_term / (n * (n - 1)))) if n > 1 else 0
    if sd == 0:
        return u, None, None
    z = (u - mu) / sd
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    rbc = 1 - (2 * u) / (n1 * n2)          # rank-biserial
    return u, round(p, 5), round(rbc, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--json")
    a = ap.parse_args()

    rates, rep = module_rates(a.probe)
    if rep.get("n_adjudicative_powered", 0) == 0:
        print("REFUSED__no_position_data — the probe run carries no term positions, so "
              "the pre-registered outcome (adjudicative dead rate) cannot be computed.")
        return 2

    with open(a.ledger, encoding="utf-8") as f:
        led = json.load(f)
    exposed_raw = set()
    for e in led.get("entries", led if isinstance(led, list) else []):
        for m in (e.get("modules") or []):
            exposed_raw.add(normalise(m))

    idx = {normalise(m): m for m in rates}
    exposed = sorted({idx[m] for m in exposed_raw if m in idx})
    unmatched = sorted(exposed_raw - set(idx))
    comparison = sorted(set(rates) - set(exposed))

    print(f"  modules with >=1 powered adjudicative term : {len(rates)}")
    print(f"  ledger modules matched to probe            : {len(exposed)}")
    print(f"  ledger modules NOT matched (reported, not dropped silently): "
          f"{len(unmatched)}")
    for m in unmatched[:10]:
        print(f"      unmatched: {m}")

    if len(exposed) < MIN_MODULES:
        print(f"\n  REFUSED__underpowered — {len(exposed)} exposed modules against a "
              f"pre-registered floor of {MIN_MODULES}. The floor was fixed before the "
              f"ledger was built and is not lowered to obtain a result.")
        out = {"verdict": "REFUSED__underpowered", "n_exposed": len(exposed),
               "min_modules": MIN_MODULES, "exposed": exposed,
               "unmatched_ledger_modules": unmatched}
        if a.json:
            with open(a.json, "w", encoding="utf-8", newline="\n") as f:
                json.dump(out, f, indent=1)
        return 3

    ex = [rates[m]["dead_rate"] for m in exposed]
    co = [rates[m]["dead_rate"] for m in comparison]
    u, p, rbc = mannwhitney(ex, co)

    def med(v):
        s = sorted(v)
        n = len(s)
        return None if not n else (s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2)

    # H2 must be tested on the SAME data, in the same run, or the primary result can be
    # reported as falsifiability when it is coverage.
    ex_cov = [rates[m]["exercised_frac"] for m in exposed]
    co_cov = [rates[m]["exercised_frac"] for m in comparison]
    u2, p2, rbc2 = mannwhitney(ex_cov, co_cov)
    ex_pow = [rates[m]["adj_powered"] for m in exposed]
    co_pow = [rates[m]["adj_powered"] for m in comparison]
    u3, p3, rbc3 = mannwhitney(ex_pow, co_pow)

    print(f"\n  H1 PRIMARY — adjudicative dead rate")
    print(f"    exposed    n={len(ex):3d}  median {med(ex):.4f}")
    print(f"    comparison n={len(co):3d}  median {med(co):.4f}")
    print(f"    U={u}  p={p}  rank-biserial={rbc}")
    print(f"\n  H2 ALTERNATIVE — exercised fraction (is the exposed group just less run?)")
    print(f"    exposed median {med(ex_cov):.4f} vs comparison {med(co_cov):.4f}"
          f"   p={p2}  rbc={rbc2}")
    print(f"\n  COVARIATE — powered adjudicative terms per module")
    print(f"    exposed median {med(ex_pow):.1f} vs comparison {med(co_pow):.1f}"
          f"   p={p3}  rbc={rbc3}")

    verdict = ("H1_SUPPORTED" if (p is not None and p < 0.05 and rbc and rbc > 0)
               else "H1_NOT_SUPPORTED")
    if verdict == "H1_SUPPORTED" and p2 is not None and p2 < 0.05:
        verdict = "H1_CONFOUNDED_BY_COVERAGE"
    print(f"\n  VERDICT: {verdict}")
    if verdict == "H1_CONFOUNDED_BY_COVERAGE":
        print("    The exposed group also differs on exercised fraction, which is H2. "
              "This must be reported as a coverage finding, not a falsifiability one.")

    out = {
        "verdict": verdict,
        "h1": {"u": u, "p": p, "rank_biserial": rbc,
               "median_exposed": med(ex), "median_comparison": med(co),
               "n_exposed": len(ex), "n_comparison": len(co)},
        "h2_coverage": {"p": p2, "rank_biserial": rbc2,
                        "median_exposed": med(ex_cov),
                        "median_comparison": med(co_cov)},
        "covariate_powered_terms": {"p": p3, "rank_biserial": rbc3,
                                    "median_exposed": med(ex_pow),
                                    "median_comparison": med(co_pow)},
        "exposed_modules": exposed,
        "unmatched_ledger_modules": unmatched,
        "limits": [
            "Retrospective and observational. Retracted claims received MORE attention, "
            "which may drive both the retraction and the discovery of dead gates near "
            "it; this design cannot separate that.",
            "One repository, one suite, one day's snapshot.",
            "Module-level exposure is coarse: a retraction implicates a pipeline, and "
            "every module on that pipeline is marked exposed whether or not it "
            "contributed the error.",
        ],
    }
    if a.json:
        with open(a.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
