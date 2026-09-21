# -*- coding: utf-8 -*-
"""The `styxx ci-audit` card. Plain text, one screen, then every finding on one line."""
from __future__ import annotations

_MEANING = {
    "RED": "the workflow goes red when this step's tools fail",
    "FAIL_OPEN": "a check that would have run is silently not run",
    "SWALLOWED": "a check ran, its tools failed, and nothing went red",
    "ABSORBED": "every check reaches its runner as before",
    "NO_CHECK": "no check in this step's job or downstream of it",
    "BASELINE_RED": "fails on its own logic under the model: not read",
    "BASELINE_SKIPPED": "does not run under the model: not read",
}
_ORDER = ("RED", "FAIL_OPEN", "SWALLOWED", "ABSORBED", "NO_CHECK", "BASELINE_RED", "BASELINE_SKIPPED")
_MECHANISM = {
    "step-if": "the check's `if:` turns false",
    "job-if": "the check's job `if:` turns false",
    "job-needs": "the check's job is skipped through `needs:`",
    "job-empty-matrix": "the check's job runs zero times (empty matrix)",
    "unreached": "the check runs but never reaches its runner",
    "fewer": "the check reaches its runner fewer times",
    "not-run": "the check is not run",
}


def _where(f: dict) -> str:
    name = f.get("name") or f"step {f['index']}"
    return f"{f['workflow']} › {f['job']} › {name}"


def card(rec: dict, *, counted: bool = False, width: int = 96) -> str:
    key = "verdict_counted" if counted else "verdict"
    s = rec.get("summary") or {}
    by = s.get("by_verdict", {})
    faults = rec.get("faults", [])
    lines = []
    what = rec.get("target") or rec.get("repo") or "."
    head = (rec.get("head") or "")[:8]
    lines.append(f"styxx ci-audit — {what}" + (f" @ {head}" if head else "") + f"  ·  {s.get('workflows', 0)} workflows, {s.get('fault_sites', 0)} fault sites"
                 + (f"  ·  {rec['seconds']} s" if rec.get("seconds") is not None else ""))
    lines.append("one fault at a time: every external command of one `run:` step fails, everything else stays healthy"
                 + ("  ·  counted reading" if counted else ""))
    lines.append("")
    for v in _ORDER:
        n = by.get(v, 0)
        if n or v in ("RED", "FAIL_OPEN", "SWALLOWED"):
            lines.append(f"  {v:<17}{n:>6}   {_MEANING[v]}")
    lines.append("")
    if s.get("live_checks"):
        lines.append(f"  checks: {s['checks_reached_in_healthy_world']} of {s['checks']} reach their runner in the healthy world; "
                     f"{s['live_checks_red_under_own_fault']} of {s['live_checks']} are RED under their own fault")
    elif s.get("checks"):
        lines.append(f"  checks: {s['checks']} recognised, none reaches its runner in the healthy world (actions and non-bash steps are not executed)")
    else:
        lines.append("  checks: none recognised (a known test / lint / typecheck runner, a test-named script, or a check-word in the step name)")
    art = s.get("artifact_failures_in_healthy_world", {})
    if any(art.values()):
        lines.append(f"  model artifacts: {art.get('x', 0)} steps fail on their own logic in the x flavour, {art.get('empty', 0)} in the empty flavour (carried past, not counted)")
    if rec.get("capped"):
        lines.append("  CAPPED: the deadline passed before every workflow was read; the numbers above are partial")

    findings = [f for f in faults if (f.get(key) or f["verdict"]) in ("FAIL_OPEN", "SWALLOWED")]
    lines.append("")
    if not findings:
        lines.append("nothing hidden, nothing dropped." if by.get("RED") else "nothing hidden, nothing dropped — and nothing read: see the counts above.")
    else:
        lines.append("findings:")
        for f in sorted(findings, key=lambda f: (0 if (f.get(key) or f["verdict"]) == "FAIL_OPEN" else 1, f["workflow"], f["job"], f["index"])):
            v = f.get(key) or f["verdict"]
            how = []
            if f.get("continue_on_error"):
                how.append("continue-on-error")
            if f.get("alone") == "SWALLOWS":
                how.append("exits 0 alone with its tools failed")
            head_line = f"  {v:<10} {_where(f)}"
            if how:
                head_line += "   [" + ", ".join(how) + "]"
            lines.append(head_line)
            drops = f.get("dropped_counted" if counted else "dropped", [])
            for d in drops:
                mech = _MECHANISM.get(d.get("mechanism"), d.get("mechanism"))
                runs = f" ({d['runs_minus']} of {d['runs_plus']} runs)" if counted and "runs_plus" in d else ""
                lines.append(f"             drops {d['job']} › {d.get('name') or 'step ' + str(d['index'])}: {mech}{runs}")
            if f.get("run_head"):
                lines.append(f"             {f['run_head'][:width - 13]}")
    lines.append("")
    lines.append("RED is loud, not correct. A check that is an action (`uses:`) is never executed here. The receipt (--format json) keeps every step.")
    return "\n".join(lines)
