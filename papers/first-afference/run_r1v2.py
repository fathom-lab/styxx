"""R1-v2 apparatus — room-agent coupling detection, per PREREG_r1v2_room_coupling_2026_08_06.md.

Loaders from run_r1 (frozen 2026-08-05), permutation test from run_r0v2 (licensed by R0-v2).
Nothing is fit, tuned, or chosen at analysis time.

  python run_r1v2.py --room room_record.jsonl [--agent ~/.styxx/chart.jsonl]
  python run_r1v2.py --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_r1 import load_room, load_agent, to_bins, pair_streams, synthetic_smoke  # noqa: E402
import run_r0v2                                   # noqa: E402
from run_r0v2 import perm_test                    # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", default=None)
    ap.add_argument("--agent", default=str(Path.home() / ".styxx" / "chart.jsonl"))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    results = {"prereg": "PREREG_r1v2_room_coupling_2026_08_06.md", "smoke": a.smoke}
    t0 = time.time()
    if a.smoke:
        run_r0v2.N_PERM = 50
        XR, XA, hours = synthetic_smoke()
        n = len(XR)
    else:
        if not a.room:
            raise SystemExit("--room required for a scored run")
        rts, rX = load_room(Path(a.room))
        ats, aX = load_agent(Path(a.agent))
        XR, XA, hours, n = pair_streams(to_bins(rts, rX), to_bins(ats, aX))
        print(f"paired bins: {n} (room {len(rts)} obs, agent {len(ats)} usable obs)", flush=True)
    results["n_paired_bins"] = n
    results["n_perm"] = run_r0v2.N_PERM
    results.update(perm_test(XR, XA, hours, seed=343))
    print(f"measured in {time.time()-t0:.0f}s: {results}", flush=True)

    try:
        from styxx.protocol import Experiment
        v = Experiment(HERE / "PREREG_r1v2_room_coupling_2026_08_06.md").score(results,
                                                                              smoke=a.smoke)
        results["verdict"], results["gates"] = v.verdict, v.gates
        results["prereg_commit"] = v.prereg_commit
    except Exception as e:
        results["verdict"] = f"UNSCORED__{type(e).__name__}: {e}"
    out = HERE / ("r1v2_result_smoke.json" if a.smoke else "r1v2_result.json")
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nVERDICT: {results['verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
