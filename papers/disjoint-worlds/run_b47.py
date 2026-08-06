"""B47 — eight minds, one battery, per PREREG_b47_eight_minds_2026_08_06.md.

Dogfood: the arc's recurrence question answered with the arc's own shipped product
(`styxx.islands`, styxx 7.30.0) on representations committed months earlier for another purpose.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from styxx.islands import survey                # noqa: E402

BANK = ROOT / "papers" / "mind-instrument" / "normeq_reps.npz"


def main() -> int:
    smoke = "--smoke" in sys.argv
    z = np.load(BANK, allow_pickle=True)
    names = list(z.files)[:3] if smoke else list(z.files)
    reps = {n: np.asarray(z[n], dtype=float) for n in names}
    s = survey(reps, n_null=100 if smoke else 1000, n_perm=100 if smoke else 1000, seed=343)
    print(s, flush=True)

    results = {"prereg": "PREREG_b47_eight_minds_2026_08_06.md", "smoke": smoke,
               "bank": BANK.name, "n_members": len(reps), "n_items": s.n_items, "k": s.k,
               "members": s.members, "pairwise": s.pairwise, "mean_affinity": s.mean_affinity,
               "islands": s.islands, "island_rule": s.island_rule, "null": s.null,
               "bimodality_p": s.bimodality_p, "cohort_median": s.cohort_median,
               "survey_verdict": s.verdict, "caveats": s.caveats,
               "median_minus_null_p95": round(s.cohort_median - s.null["p95"], 4)}
    try:
        from styxx.protocol import Experiment
        v = Experiment(HERE / "PREREG_b47_eight_minds_2026_08_06.md").score(results, smoke=smoke)
        results["verdict"], results["gates"] = v.verdict, v.gates
        results["prereg_commit"] = v.prereg_commit
    except Exception as e:
        results["verdict"] = f"UNSCORED__{type(e).__name__}: {e}"
    (HERE / f"b47_result{'_smoke' if smoke else ''}.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nVERDICT: {results['verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
