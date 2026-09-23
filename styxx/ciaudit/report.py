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


def _said(readings) -> str | None:
    """SWALLOW-13's readings of an unrepaired finding's script, in one line."""
    if not readings:
        return None
    from .repair_frontier import say
    return say(readings)


_ORDER = ("RED", "FAIL_OPEN", "SWALLOWED", "ABSORBED", "NO_CHECK", "BASELINE_RED", "BASELINE_SKIPPED")
_MECHANISM = {
    "step-if": "the check's `if:` turns false",
    "job-if": "the check's job `if:` turns false",
    "job-needs": "the check's job is skipped through `needs:`",
    "job-empty-matrix": "the check's job runs zero times (empty matrix)",
    "unreached": "the check runs but never reaches its runner",
    "fewer": "the check reaches its runner fewer times",
    "not-run": "the check is not run",
    "after-failure": "an earlier step of the job failed",
}


def _where(f: dict) -> str:
    name = f.get("name") or f"step {f['index']}"
    return f"{f['workflow']} › {f['job']} › {name}"


def _since_line(h: dict) -> str:
    """One line of history for a finding: since when, by whose hand, and whether the commit said why."""
    import datetime as dt
    if not h.get("placed"):
        return f"history: not placed — {h.get('why', '')}"
    day = dt.datetime.fromtimestamp(h["since"], dt.timezone.utc).strftime("%Y-%m-%d")
    age = f", {int(h['age_days'])} days" if h.get("age_days") is not None else ""
    sha = (h.get("sha") or "")[:7]
    if h.get("born_hidden"):
        how = "born hidden" + (" at the clone's oldest commit (may be older)" if h.get("at_boundary") else "")
    else:
        how = "acquired: " + ", ".join(h.get("mechanism") or []) + (f' — "{h["subject"][:60]}"' if h.get("subject") else "")
    ack = f" [the commit says: {h['acknowledged']}]" if h.get("acknowledged") else ""
    rep = f" (made loud {h['repaired_before']}× before)" if h.get("repaired_before") else ""
    return f"hidden since {day} ({sha}{age}): {how}{ack}{rep}"


def _differential_section(d: dict, width: int) -> list[str]:
    """What this HEAD hides that its base did not, what it made loud or removed, and what it left as it was."""
    out = []
    if d.get("error"):
        out.append(f"since {d.get('base_ref')}: could not read the base — {d['error']}")
        return out
    head = f"since {d.get('base_ref')} (merge-base {(d.get('merge_base') or '')[:8]}): "
    if d.get("pr") is not None:
        head = f"{d.get('base_ref')} (head {(d.get('pr_head') or '')[:8]}, base {(d.get('merge_base') or '')[:8]}; {d.get('reading')}): "
    wfs = d.get("workflows", [])
    if not wfs:
        out.append(head + "no workflow changed; the gate has nothing to read.")
        return out
    n, r, still, unread = d.get("new_hidden", 0), d.get("removed_hidden", 0), d.get("still_hidden", 0), d.get("hidden_after_unread", 0)
    out.append(head + f"{len(wfs)} workflow{'s' if len(wfs) != 1 else ''} changed · "
               + (f"{n} check{'s' if n != 1 else ''} newly hidden — THE GATE FIRES" if n else "nothing newly hidden")
               + (f" · {r} hidden check{'s' if r != 1 else ''} made loud or removed" if r else "")
               + (f" · {still} hidden on both sides (not this change's doing)" if still else "")
               + (f" · {unread} hidden here, unreadable at the base" if unread else ""))
    for w in wfs:
        for x in w.get("new_hidden", []):
            what = x.get("name") or "step " + str(x["index"])
            how = x["kind"] + (": " + ", ".join(x["mechanism"]) if x.get("mechanism") else "")
            out.append(f"  {x['verdict']:<10} {w['workflow']} › {x['job']} › {what}   [{how}]")
            if x.get("run_head"):
                out.append(f"             {x['run_head'][:width - 13]}")
            fx = x.get("fix") or {}
            if fx.get("verified_repair"):
                out.append(f"             repair: {fx['verified_repair']}, {fx['lines_changed']} line{'s' if fx['lines_changed'] != 1 else ''} (verified: loud under the same fault, healthy run unchanged)")
                for dl in (fx.get("diff") or "").splitlines():
                    if (dl.startswith("+") or dl.startswith("-")) and not dl.startswith(("+++", "---")):
                        out.append("                 " + dl[: width + 60])
            elif fx:
                out.append(f"             no verified repair: {(fx.get('why_not') or '')[: width + 40]}")
                said = _said(fx.get("readings"))
                if said:
                    out.append(f"             reading: {said[: width + 60]}")
        for x in w.get("removed_hidden", []):
            what = x.get("name") or "step " + str(x["index"])
            out.append(f"  {'loud' if x['kind'] == 'repaired' else 'gone':<10} {w['workflow']} › {x['job']} › {what}   [{x['kind']}"
                       + (": " + ", ".join(x["mechanism"]) if x.get("mechanism") else "") + "]")
        for x in w.get("hidden_after_unread", []):
            what = x.get("name") or "step " + str(x["index"])
            out.append(f"  {x['verdict']:<10} {w['workflow']} › {x['job']} › {what}   [hidden here; the base could not be read: {x.get('verdict_before')}]")
    return out


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
    if rec.get("confinement") is not None:
        from .confine import describe
        lines.append(describe(rec["confinement"]))
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
        lines.append(f"  checks: {s['checks']} recognised, none reaches its runner in the healthy world (non-bash steps are not executed)")
    else:
        lines.append("  checks: none recognised in run: steps (a known test / lint / typecheck runner, a test-named script, or a check-word in the step name)")
    if s.get("action_checks"):
        unv = f"; {s['action_checks_unverified']} on an unverified catalogue entry" if s.get("action_checks_unverified") else ""
        lines.append(f"  action checks: {s['action_checks_reached_in_healthy_world']} of {s['action_checks']} reached in the healthy world "
                     f"(never executed: a catalogued action is a check that is reached or not){unv}")
    elif rec.get("actions", True):
        lines.append("  action checks: none catalogued (see styxx/ciaudit/actions.py for the list)")
    if s.get("unreadable_steps"):
        lines.append(f"  unreadable: {s['unreadable_steps']} steps are local actions, reusable workflows, docker images or github-script, which cannot be read")
    art = s.get("artifact_failures_in_healthy_world", {})
    if any(art.values()):
        lines.append(f"  model artifacts: {art.get('x', 0)} steps fail on their own logic in the x flavour, {art.get('empty', 0)} in the empty flavour (carried past, not counted)")
    if rec.get("capped"):
        lines.append("  CAPPED: the deadline passed before every workflow was read; the numbers above are partial")

    findings = [f for f in faults if (f.get(key) or f["verdict"]) in ("FAIL_OPEN", "SWALLOWED")]
    hist = {(h["workflow"], h["job"], h["index"]): h for h in (rec.get("history") or {}).get("findings", [])}
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
                runs = f" ({d['runs_minus']} of {d['runs_plus']} runs)" if counted and "runs_plus" in d and not d.get("action") else ""
                what = d.get("name") or "step " + str(d["index"])
                act = f" [{d['action']}, {d.get('kind')}{'' if d.get('verified', True) else ', unverified entry'}]" if d.get("action") else ""
                lines.append(f"             drops {d['job']} › {what}{act}: {mech}{runs}")
            if f.get("run_head"):
                lines.append(f"             {f['run_head'][:width - 13]}")
            h = hist.get((f["workflow"], f["job"], f["index"]))
            if h is not None:
                lines.append("             " + _since_line(h)[: width + 60])
    if rec.get("history") is not None:
        hm = rec["history"]
        note = []
        if hm.get("error"):
            note.append(hm["error"])
        if hm.get("shallow") and any(h.get("at_boundary") for h in hm.get("findings", [])):
            since = (rec.get("history_clone") or {}).get("since")
            note.append(f"history fetched since {since}; a step hidden at that boundary may be older" if since
                        else "the clone is shallow: a step hidden at its oldest commit may be older")
        if hm.get("capped"):
            note.append("the deadline passed before every workflow's history was read")
        lines.append("")
        lines.append(f"history: {hm.get('revisions_read', 0)} workflow revisions read on the mainline (first-parent from HEAD)"
                     + (f"; {'; '.join(note)}" if note else ""))
    if rec.get("differential") is not None:
        lines.append("")
        lines.extend(_differential_section(rec["differential"], width))
    if rec.get("repairs") is not None:
        lines.append("")
        lines.append("repairs (verified: RED under the same fault, and a healthy run unchanged in both flavours):")
        if not rec["repairs"]:
            lines.append("  nothing to repair.")
        for t in rec["repairs"]:
            where = f"{t['workflow']} › {t['job']} › {t.get('name') or 'step ' + str(t['index'])}"
            if t.get("verified_repair"):
                c = next(c for c in t["candidates"] if c["repair"] == t["verified_repair"])
                lines.append(f"  {where} — {t['verified_repair']}, {c['lines_changed']} line{'s' if c['lines_changed'] != 1 else ''}")
                for dl in c["diff"].splitlines():
                    if (dl.startswith("+") or dl.startswith("-")) and not dl.startswith(("+++", "---")):
                        lines.append("      " + dl[: width + 80])
            else:
                tried = [c for c in t["candidates"] if c.get("applies")]
                if not tried:
                    why = "no repair applies: " + "; ".join(sorted({c.get("why", "") for c in t["candidates"] if c.get("why")}))[: width + 40]
                else:
                    rej = next((c for c in tried if c.get("unchanged") is False), None)
                    if rej is not None:
                        why = f"{rej['repair']} is loud but {rej['why']}"
                    elif len({c.get("why") for c in tried}) == 1:
                        why = f"{' / '.join(c['repair'] for c in tried)}: {tried[0].get('why', '')}"
                    else:
                        why = "; ".join(f"{c['repair']}: {c.get('why', '')}" for c in tried)
                lines.append(f"  {where} — no verified repair: {why[: width + 80]}")
                said = _said(t.get("readings"))
                if said:
                    lines.append(f"      reading: {said[: width + 60]}")
    lines.append("")
    lines.append("RED is loud, not correct. An action check is counted, never executed: it can be dropped here, not seen to fail. The receipt (--format json) keeps every step.")
    return "\n".join(lines)
