"""R1 apparatus — is the room legible to the agent? Per PREREG_r1_room_legibility_2026_08_05.md.

Committed BEFORE any room-side data exists. Apparatus decisions fixed here, pre-data:

  * agent vector (24-dim): [tier_active, phase1_conf, phase4_conf] + features_v2 (21 dims).
    Records without a populated features_v2 are treated as unobserved on the agent side --
    a 3-dim cloud cannot support discovery, and mixing vector widths would let bin coverage
    vary by field. At the measured logging cadence this makes 200 paired bins roughly 5-7
    days of joint observation, not hours; the prereg's G0 gates on bin count, so nothing moves.
  * room vector (12-dim): the recorder's raw feats line ({"ts": ..., "feats": [11 bands + rms_db]}),
    written by `coil_daemon`/`room_cortex --record` every emit interval.
  * grid: 60 s bins, mean-pooled within bin; bins missing either side dropped (prereg).
  * discovery: TransferMap.fit + Hungarian (b34-v3 machinery verbatim; rank self-clamps to
    the smaller cloud). The true pairing is destroyed by uniform shuffle and must be recovered.
  * hour-matched null: agent rows permuted only within hour-of-day bins, then the same
    discovery is scored against the pre-permutation truth -- circadian structure survives
    into the null by construction.
  * disc_over_chance_ratio = disc * n_paired_bins  (chance for n-way assignment = 1/n).

Usage:
  python run_r1.py --room room_record.jsonl --agent ~/.styxx/chart.jsonl
  python run_r1.py --smoke          # synthetic plumbing check, INVALID-only by prereg
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "papers" / "disjoint-worlds"))
from styxx_transfer import TransferMap          # noqa: E402

AGENT_SCALARS = ["tier_active", "phase1_conf", "phase4_conf"]
FEATURES_V2_DIM = 21
BIN_S = 60


def load_room(path: Path) -> tuple[np.ndarray, np.ndarray]:
    ts, X = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "feats" in r and r.get("ts"):
            ts.append(float(r["ts"]))
            X.append([float(v) for v in r["feats"]])
    if not ts:
        raise SystemExit(f"no room records with feats in {path}")
    widths = {len(x) for x in X}
    if len(widths) != 1:
        raise SystemExit(f"inconsistent room vector widths {sorted(widths)} in {path}")
    return np.asarray(ts), np.asarray(X)


def load_agent(path: Path) -> tuple[np.ndarray, np.ndarray]:
    ts, X = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        fv = r.get("features_v2")
        if not fv or len(fv) != FEATURES_V2_DIM:
            continue                       # unobserved on the agent side (see module docstring)
        row = []
        ok = True
        for k in AGENT_SCALARS:
            v = r.get(k)
            if not isinstance(v, (int, float)):
                ok = False
                break
            row.append(float(v))
        if not ok:
            continue
        ts.append(float(r["ts"]))
        X.append(row + [float(v) for v in fv])
    if not ts:
        raise SystemExit(f"no usable agent records in {path}")
    return np.asarray(ts), np.asarray(X)


def to_bins(ts: np.ndarray, X: np.ndarray) -> dict[int, np.ndarray]:
    out: dict[int, list] = {}
    for t, x in zip(ts, X):
        out.setdefault(int(t // BIN_S), []).append(x)
    return {b: np.mean(np.asarray(v), axis=0) for b, v in out.items()}


def pair_streams(room_bins: dict, agent_bins: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    common = sorted(set(room_bins) & set(agent_bins))
    XR = np.asarray([room_bins[b] for b in common])
    XA = np.asarray([agent_bins[b] for b in common])
    hours = np.asarray([datetime.fromtimestamp(b * BIN_S).hour for b in common])
    return XR, XA, hours, len(common)


def zscore(X: np.ndarray) -> np.ndarray:
    return (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)


def discover(XR: np.ndarray, XA: np.ndarray, rng: np.random.Generator) -> float:
    """Destroy the pairing, recover it from geometry alone, score against truth."""
    perm = rng.permutation(len(XR))
    XAs = XA[perm]
    true_col = np.argsort(perm)
    k = min(XR.shape[1], XA.shape[1])
    tm = TransferMap.fit(XR, XAs, k=k)
    MR = np.stack([tm.transfer_point(x) for x in XR])
    _, col = linear_sum_assignment(np.linalg.norm(MR[:, None, :] - XAs[None, :, :], axis=-1))
    return float((col == true_col).mean())


def hour_matched_permutation(n: int, hours: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    idx = np.arange(n)
    for h in np.unique(hours):
        m = np.where(hours == h)[0]
        idx[m] = m[rng.permutation(len(m))]
    return idx


def measure(XR: np.ndarray, XA: np.ndarray, hours: np.ndarray,
            seed: int = 343) -> dict:
    XR, XA = zscore(XR), zscore(XA)
    n = len(XR)
    disc = discover(XR, XA, np.random.default_rng(seed + 500000))
    hm = discover(XR, XA[hour_matched_permutation(n, hours, np.random.default_rng(seed + 100))],
                  np.random.default_rng(seed + 900000))
    free = discover(XR, XA[np.random.default_rng(seed + 200).permutation(n)],
                    np.random.default_rng(seed + 700000))
    return {"n_paired_bins": n, "disc": round(disc, 4),
            "hourmatched_null": round(hm, 4), "free_null": round(free, 4),
            "chance": round(1.0 / n, 4),
            "disc_minus_hourmatched_null": round(disc - hm, 4),
            "disc_minus_free_null": round(disc - free, 4),
            "disc_over_chance_ratio": round(disc * n, 4)}


def synthetic_smoke(n: int = 30, seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 4))
    XR = np.tanh(z @ rng.standard_normal((4, 12))) + 0.1 * rng.standard_normal((n, 12))
    XA = np.tanh(z @ rng.standard_normal((4, 24))) + 0.1 * rng.standard_normal((n, 24))
    hours = rng.integers(0, 24, n)
    return XR, XA, hours


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", default=None)
    ap.add_argument("--agent", default=str(Path.home() / ".styxx" / "chart.jsonl"))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    results = {"prereg": "PREREG_r1_room_legibility_2026_08_05.md", "smoke": a.smoke}
    t0 = time.time()
    if a.smoke:
        XR, XA, hours = synthetic_smoke()
    else:
        if not a.room:
            raise SystemExit("--room required for a scored run")
        rts, rX = load_room(Path(a.room))
        ats, aX = load_agent(Path(a.agent))
        XR, XA, hours, n = pair_streams(to_bins(rts, rX), to_bins(ats, aX))
        print(f"paired bins: {n} (room {len(rts)} obs, agent {len(ats)} usable obs)", flush=True)
    results.update(measure(XR, XA, hours))
    print(f"measured in {time.time()-t0:.0f}s: {results}", flush=True)

    try:
        from styxx.protocol import Experiment
        v = Experiment(HERE / "PREREG_r1_room_legibility_2026_08_05.md").score(results,
                                                                              smoke=a.smoke)
        results["verdict"], results["gates"] = v.verdict, v.gates
        results["prereg_commit"] = v.prereg_commit
    except Exception as e:
        results["verdict"] = f"UNSCORED__{type(e).__name__}: {e}"
    out = HERE / ("r1_result_smoke.json" if a.smoke else "r1_result.json")
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nVERDICT: {results['verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
