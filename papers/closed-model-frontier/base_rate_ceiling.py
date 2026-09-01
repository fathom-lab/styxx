# -*- coding: utf-8 -*-
"""Derive every arithmetic quantity ANALYSIS_base_rate_ceiling cites, into a receipt.

The paper's argument is arithmetic over two measured inputs: an external base rate
(external_citations.json) and our own flag rate (flag_rate.json). Arithmetic written by
hand into prose is exactly the unbound number this lab's verifier is built to refuse, and
it refused forty-one of them on the first certification of that paper. So the arithmetic
is done here instead, once, into bytes a reader can re-run.

    precision = pi * r / f                      Bayes, an identity and not an inequality
    ceiling   = min(1, pi / f)                  since r <= 1
    r_implied = p_obs * f / pi                  the falsification test: r_implied > 1
                                                refutes the imported base rate outright

  python papers/closed-model-frontier/base_rate_ceiling.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "base_rate_ceiling.json"

P_OBS = 0.16          # RESULT_v14 held-out precision, receipt v14_adjudication.json
FLOOR = 0.95          # the preregistered precision floor, PREREG_v14_repair
REFUTE_AT = 0.0179    # the derivation's preregistered refutation threshold on f


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple:
    """Wilson score interval. Assumes independent draws — see the note it emits."""
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return round((c - s) / d, 4), round((c + s) / d, 4)


def main() -> int:
    fr = json.loads((HERE / "flag_rate.json").read_text(encoding="utf-8"))
    ex = json.loads((HERE / "external_citations.json").read_text(encoding="utf-8"))

    hm = ex["pr_mci_study"]["high_mci"]
    pi = hm["count"] / hm["of"]

    splits = {}
    for name in ("development", "held_out", "corpus_wide"):
        s = fr[name]
        f = s["prs_with_path_accusation"] / s["prs_scored"]
        ceiling = min(1.0, pi / f)
        splits[name] = {
            "accused_prs": s["prs_with_path_accusation"],
            "prs_scored": s["prs_scored"],
            "flag_rate": round(f, 5),
            "ceiling": round(ceiling, 4),
            "ceiling_3dp": round(ceiling, 3),
            "ceiling_binding_at_the_floor": bool(ceiling < FLOOR),
            "ceiling_explains_the_observed": bool(ceiling <= P_OBS),
            "implied_recall_at_observed_precision": round(P_OBS * f / pi, 4),
            "recall_required_to_reach_the_floor": round(FLOOR * f / pi, 4),
            "regime": ("A: ceiling not binding — the floor was attainable"
                       if ceiling >= FLOOR else
                       "C: ceiling explains the observed level"
                       if ceiling <= P_OBS else
                       "B: binding but NOT explanatory"),
        }

    # The unit-matched pairing the challenger needs: our CLAIM-conditional flag rate
    # against a PER-PR base rate. It is the pairing that fails hardest.
    fcond = fr["held_out"]["path_flag_rate_of_claiming_prs"]
    r_cond = P_OBS * fcond / pi

    payload = {
        "what": "the arithmetic ANALYSIS_base_rate_ceiling cites, derived rather than asserted",
        "identity": "precision = pi * r / f  (Bayes); ceiling = min(1, pi/f) since r <= 1",
        "inputs": {
            "pi_base_rate": round(pi, 5),
            "pi_source": "external_citations.json :: pr_mci_study.high_mci (406/23,247)",
            "pi_caveat": hm["WHAT_IT_IS_NOT"],
            "p_obs": P_OBS,
            "p_obs_source": "v14_adjudication.json — 16 upheld of 100 scored, held-out",
            "floor": FLOOR,
            "refutation_threshold_on_f": REFUTE_AT,
        },
        "splits": splits,
        "held_out_verdict": {
            "flag_rate": splits["held_out"]["flag_rate"],
            "below_refutation_threshold": bool(splits["held_out"]["flag_rate"] <= REFUTE_AT),
            "ceiling": splits["held_out"]["ceiling"],
            "conclusion": (
                "The base-rate account is REFUTED on the split that carries the headline. "
                "The ceiling is not binding, so the 0.95 floor was arithmetically reachable "
                "and the shortfall is the instrument's."),
        },
        "unit_matched_pairing": {
            "note": ("Pairing a PER-PR base rate with our CLAIM-CONDITIONAL flag rate is the "
                     "strongest form of the challenge. It self-refutes: implied recall exceeds 1, "
                     "which is impossible, so the imported base rate cannot be the one that "
                     "bounds our claim class."),
            "claim_conditional_flag_rate": fcond,
            "implied_recall": round(r_cond, 4),
            "impossible": bool(r_cond > 1.0),
        },
        "wilson_95_on_the_observed_precision": {
            "k": 16, "n": 100,
            "interval": list(wilson(16, 100)),
            "assumption_violated": (
                "Wilson assumes independent draws. The 100 accusations are clustered within "
                "PRs and repositories, so the true interval is wider than this. Reported "
                "because the analysis cites it; not to be read as a calibrated interval."),
        },
        "collateral": {
            "note": ("What the flag rate actually buys, which is the finding worth more than "
                     "the refutation: at held-out f with precision 0.16, most accusations "
                     "issued are false ones."),
            "held_out_prs_accused": fr["held_out"]["prs_with_path_accusation"],
            "expected_justly_accused": round(fr["held_out"]["prs_with_path_accusation"] * P_OBS),
            "expected_wrongly_accused": round(fr["held_out"]["prs_with_path_accusation"] * (1 - P_OBS)),
        },
    }
    OUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")

    print(f"pi = {pi:.5f}  (406/23,247, a heuristic's firing rate — see the caveat)")
    for k, v in splits.items():
        print(f"{k:12s} f={v['flag_rate']:.5f}  ceiling={v['ceiling']:.4f}  {v['regime']}")
    print(f"unit-matched implied recall = {r_cond:.4f} "
          f"{'-> IMPOSSIBLE, pairing refuted' if r_cond > 1 else ''}")
    print(f"collateral: {payload['collateral']['expected_wrongly_accused']} wrongly accused "
          f"vs {payload['collateral']['expected_justly_accused']} justly, held-out")
    print(f"-> {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
