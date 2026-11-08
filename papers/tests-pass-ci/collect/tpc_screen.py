# -*- coding: utf-8 -*-
"""The screen (PREREG P5) and the CI test result at the head commit (P6), over testreceipt's own
code, vendored unchanged under vendor/testreceipt (MIT; its LICENSE beside it).

P5: a claimer has at least one pass claim with explicit=True, of any scope; implied-only rows have
only explicit=False pass claims and enter no denominator.

P6: testreceipt's `CiResult.tests` over the fully paged, latest-attempt listings, built exactly as
testreceipt's `fetch_pr` builds them (check runs; for a github-actions check run whose conclusion
is success or failure, its job's steps; the combined status's contexts), with the PREREG's one
change: `timed_out` on a check run or step reads as failure. So a timed-out Actions job also gets
its steps, as a failed one does. `cancelled`, `skipped`, `neutral`, `stale` and in-progress read
as no result, as in testreceipt's code; a status `error` reads as failure, as in its code.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Optional

_VENDOR = Path(__file__).resolve().parent / "vendor"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

from testreceipt import ci as tr_ci  # noqa: E402
from testreceipt import claims as tr_claims  # noqa: E402
from testreceipt import reconcile as tr_reconcile  # noqa: E402

# the lock table's pins (PREREG section 14); the collector refuses other bytes
VENDOR_PINS = {
    "claims.py": "434688dce2d33e720754b937f4cf85669ecf1fcec046ba04177d9edb794c554e",
    "ci.py": "df554bf3f0e1b3e5a5437fdcb7c506d50c90f25a40f222af7e590168178741b0",
    "reconcile.py": "e77f2ee308660373e1ffe89102b5b89970f04363926bf4778769edf5e9c160bb",
    "LICENSE": "94ba9b60a42b3913b37575d0986e99f1f729f8619e4517bc1a69bbb2c8486624",
}


def vendor_digests() -> dict:
    d = _VENDOR / "testreceipt"
    return {name: hashlib.sha256((d / name).read_bytes()).hexdigest() for name in VENDOR_PINS}


def check_vendor() -> None:
    got = vendor_digests()
    bad = {k: v for k, v in got.items() if v != VENDOR_PINS[k]}
    if bad:
        raise RuntimeError(f"vendored testreceipt differs from the PREREG's pins: {sorted(bad)}")


check_vendor()

is_test_step = tr_ci.is_test_step
Check = tr_ci.Check
CiResult = tr_ci.CiResult
claims = tr_claims.claims


# ---- P5 -----------------------------------------------------------------------------------------
def screen(body: str) -> dict:
    found = claims(body or "")
    explicit = [c for c in found if c.kind == "pass" and c.explicit]
    implied = [c for c in found if c.kind == "pass" and not c.explicit]
    fails = [c for c in found if c.kind == "fail"]
    if explicit:
        kind = "claimer"
    elif implied:
        kind = "implied_only"
    else:
        kind = "none"
    return {
        "screen": kind,
        "fail_claim": bool(fails),
        "claims": [{"kind": c.kind, "explicit": c.explicit, "scope": c.scope, "why_part": c.why_part,
                    "count": c.count, "text": c.text[:300]} for c in found],
        "_objects": found,
    }


# ---- P6 -----------------------------------------------------------------------------------------
def _map(conclusion: Optional[str]) -> Optional[str]:
    """The PREREG's one change: timed_out reads as failure."""
    return "failure" if conclusion == "timed_out" else conclusion


STEPS_FOR = ("success", "failure")  # after the mapping; testreceipt's fetch_pr condition


def build_ci(head: str, check_runs: list, jobs_by_id: dict, statuses: list) -> tuple:
    """(CiResult, steps_source {check_run_id: 'listing'|'missing'|'none'}). `jobs_by_id` maps a
    job id to the job object (with steps) from the Actions listings or the per-job fallback."""
    checks = []
    source = {}
    for run in check_runs:
        app = (run.get("app") or {}).get("slug")
        concl = _map(run.get("conclusion"))
        steps: tuple = ()
        if app == "github-actions" and concl in STEPS_FOR:
            job = jobs_by_id.get(run.get("id"))
            if job is not None:
                steps = tuple(((s.get("name") or ""), _map(s.get("conclusion"))) for s in (job.get("steps") or []))
                source[run.get("id")] = "listing"
            else:
                source[run.get("id")] = "missing"
        checks.append(Check(run.get("name") or "", concl, app, steps))
    sts = [((s.get("context") or ""), (s.get("state") or "")) for s in statuses]
    return CiResult(head, checks, sts), source


def ci_outcome(result) -> tuple:
    return result.tests()


def test_like_job(job: dict) -> bool:
    if is_test_step(job.get("name") or ""):
        return True
    return any(is_test_step(s.get("name") or "") for s in (job.get("steps") or []))


def failing_test_steps(job: dict) -> list:
    """Test-like steps of an Actions job whose conclusion (after the mapping) is failure."""
    return [s.get("name") or "" for s in (job.get("steps") or [])
            if is_test_step(s.get("name") or "") and _map(s.get("conclusion")) == "failure"]


def reconcile_verdict(body: str, outcome: str, evidence: str) -> str:
    return tr_reconcile.reconcile(body or "", outcome, evidence).verdict
