# -*- coding: utf-8 -*-
"""SP-6 replay against styxx.contract using the REAL pre-fix source.

The whole ``styxx`` package is extracted from git at ``<fix_commit>~1`` -- the
last commit where the defect was live -- imported in isolation, and its real
function wrapped, unmodified, with @measures. Nothing is reconstructed.

An earlier version of this replay used hand-written reproductions and scored
4/5. That number was inflated: two reproductions passed an empty sequence at the
call boundary where the shipped code received a well-formed argument and
produced the emptiness internally. This file exists because the difference
matters, and it is the difference between measuring the contract and measuring
my memory of the bug.

Kill criterion, published before the module was written:
    catches >= 4 of the 5 SP-6 cases, or the idea dies and the number gets published.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

from styxx.contract import Violation, clear_violations, is_degenerate, looks_confident, measures, violations

REPO = Path(__file__).resolve().parent.parent
_ROOT = Path(tempfile.mkdtemp(prefix="styxx_sp6_"))


def load_prefix_pkg(commit: str):
    """Import the whole styxx package as it existed at <commit>~1, in isolation."""
    dest = _ROOT / commit
    if not dest.exists():
        dest.mkdir(parents=True)
        tar = subprocess.run(["git", "archive", f"{commit}~1", "styxx"],
                             cwd=REPO, capture_output=True, check=True).stdout
        p = subprocess.Popen(["tar", "-x", "-C", str(dest)], stdin=subprocess.PIPE)
        p.communicate(tar)
        if p.returncode != 0:
            raise RuntimeError(f"tar failed for {commit}")
    for name in [k for k in sys.modules if k == "styxx" or k.startswith("styxx.")]:
        del sys.modules[name]
    sys.path.insert(0, str(dest))
    import styxx as pre                                  # noqa: PLC0415
    return pre


def restore():
    for name in [k for k in sys.modules if k == "styxx" or k.startswith("styxx.")]:
        del sys.modules[name]
    sys.path[:] = [p for p in sys.path if not str(p).startswith(str(_ROOT))]


CASES = []


def case(cid, commit, note):
    def deco(fn):
        CASES.append((cid, commit, note, fn))
        return fn
    return deco


@case("SP-2026-0008", "2c5eff3", "empty trajectories -> confident low-risk forecast")
def sp8(pre):
    from styxx.forecast import CognitiveForecaster
    fc = CognitiveForecaster.bootstrap()   # trained: the shipped configuration
    w = measures(inputs=["trajectories"], min_n=1)(type(fc).forecast)
    return w(fc, {"entropy": [], "logprob": [], "top2_margin": []})


@case("SP-2026-0011", "5dd3949", "response with no completed gate -> valid=True")
def sp11(pre):
    from styxx.cognometrics import tool_verify_response
    w = measures(inputs=["args"], min_n=1)(tool_verify_response)
    # the shipped failure: a well-formed 20-token response whose scoring never
    # reached phase4, so gate stays 'pending' and `gate != "fail"` reads True.
    resp = {"choices": [{"logprobs": {"content":
            [{"logprob": -0.2 - 0.01 * i} for i in range(20)]}}]}
    return w({"response": resp})


@case("SP-2026-0012", "5dd3949", "empty trajectory -> confab_ratio 0.0, 'steady'")
def sp12(pre):
    from styxx.temperature import TruthMap
    w = measures(inputs=["entropy", "logprob", "top2_margin"], min_n=1)(
        TruthMap.from_trajectories.__func__)
    return w(TruthMap, [], [], [])


@case("SP-2026-0016", "ed91621", "both resampling arms empty -> suspected=False")
def sp16(pre):
    from styxx.divergence import detect_context_injection
    w = measures(inputs=["samples_stateless", "samples_in_session"], min_n=1)(
        detect_context_injection)
    return w([], [], "Paris")


@case("SP-2026-0020", "deeb7e4", "4 distinct Japanese answers -> entropy 0.0")
def sp20(pre):
    from styxx.divergence import semantic_entropy
    w = measures(inputs=["samples"], min_n=2)(semantic_entropy)
    return w(["東京", "大阪", "京都", "札幌"], method="lexical")


def main() -> int:
    print("SP-6 replay -- REAL pre-fix source (git archive <fix>~1, isolated import)")
    print("kill criterion, published in advance: >= 4 of 5\n")
    caught = 0
    rows = []
    for cid, commit, note, fn in CASES:
        clear_violations()
        status, detail = "MISSED ", ""
        try:
            pre = load_prefix_pkg(commit)
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                out = fn(pre)
            if violations():
                v = violations()[0]
                status, detail = "CAUGHT ", f"{v.why_degenerate}  ->  {v.what_was_returned}"
                caught += 1
            else:
                detail = f"returned {str(out)[:80]}"
        except Exception as e:
            status, detail = "UNRUN  ", f"{type(e).__name__}: {e}"   # scored as a miss
        finally:
            restore()
        rows.append((status, cid, note, detail))
        print(f"  {status} {cid}  {note}\n           {detail}")

    print(f"\n  {caught}/5 caught")
    print(f"  kill criterion >= 4 of 5 -> "
          f"{'SURVIVES' if caught >= 4 else 'DIES -- published as such'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        shutil.rmtree(_ROOT, ignore_errors=True)
