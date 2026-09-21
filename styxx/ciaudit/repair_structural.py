# -*- coding: utf-8 -*-
"""`styxx ci-audit --repair`, second stage: for the findings SWALLOW-4's repairs leave unverified, two
edits to the script's logic -- a guard whose failing tool is not its green path (`guard-status`),
a query without its `|| echo` default (`no-default`), and both -- verified on the same two halves.
The living copy of SWALLOW-5's instrument, `benchmarks/harness_mutation/repair_structural.py`,
frozen at the sha256 the SWALLOW-5 receipt names; `tests/test_ciaudit.py` holds the two to
identical outcomes.
"""
from __future__ import annotations

import re
from pathlib import Path

from . import engine as faults          # the living copy of the instrument; the names below read as in benchmarks/harness_mutation/repair_structural.py
from . import repair
from .repair import _first_change, _healthy_shape, analyse, diff_size, locate, strict_shell, unified_diff

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "styxx.ci-audit-repair-structural/v1"
REPAIRS = ("guard-status", "no-default", "both-structural")

# ----------------------------------------------------------------------------- the two edits, on text

_IF_LINE = re.compile(r"^(?P<ind>\s*)(?P<kw>if)\s+(?P<neg>!\s*)?(?P<cmd>.+?)\s*;\s*then\s*$")
_IF_OPEN = re.compile(r"^(?P<ind>\s*)(?P<kw>if)\s+(?P<neg>!\s*)?(?P<cmd>.+?)\s*$")
_THEN = re.compile(r"^\s*then\s*$")
_BUILTIN_HEAD = re.compile(r"^(?:\[\[?|test|true|false|:|\(\(|\$\(\(|-[a-z]|[A-Za-z_][A-Za-z0-9_]*=)")
_GUARD = ('{ind}__rc=0; {cmd} || __rc=$?',
          '{ind}if [ "$__rc" -gt 1 ]; then echo "guard command failed (exit $__rc): {what}" >&2; exit "$__rc"; fi',
          '{ind}if [ "$__rc" {test} 0 ]; then')


def _guardable(cmd: str) -> bool:
    c = cmd.strip()
    return bool(c) and not _BUILTIN_HEAD.match(c) and not c.startswith(("$", "\"", "'")) and "__rc" not in c


def guard_status(run: str) -> str:
    """The guard-status repair; the script unchanged when no guardable `if` is found."""
    lines = run.splitlines()
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = _IF_LINE.match(line)
        two = False
        if not m and i + 1 < len(lines) and _THEN.match(lines[i + 1]) and not line.rstrip().endswith("\\"):
            m = _IF_OPEN.match(line)
            two = m is not None
        if m and _guardable(m.group("cmd")):
            ind, cmd, neg = m.group("ind"), m.group("cmd").strip(), bool(m.group("neg"))
            what = cmd.replace("\\", "\\\\").replace('"', '\\"')[:60]
            out.append(_GUARD[0].format(ind=ind, cmd=cmd))
            out.append(_GUARD[1].format(ind=ind, what=what))
            out.append(_GUARD[2].format(ind=ind, test="-ne" if neg else "-eq"))
            i += 2 if two else 1
            continue
        out.append(line)
        i += 1
    return "\n".join(out) + ("\n" if run.endswith("\n") else "")


_DEFAULT_START = re.compile(r"\s*\|\|\s*echo\b")
_DEFAULT_LINE = re.compile(r"^\s*\|\|\s*echo\b.*$")


def _strip_defaults(line: str) -> str:
    """Remove every `|| echo …` simple command from one line, reading quotes: the fallback ends at
    an unquoted `;`, `)`, `|`, `&` or the end of the line (a `>&2` redirection belongs to it)."""
    out = line
    while True:
        m = _DEFAULT_START.search(out)
        if not m:
            return out
        j, q = m.end(), None
        while j < len(out):
            ch = out[j]
            if q:
                if ch == "\\" and q == '"':
                    j += 2
                    continue
                if ch == q:
                    q = None
            elif ch in "'\"":
                q = ch
            elif ch == ">" and out[j:j + 2] == ">&" and j + 2 < len(out) and out[j + 2].isdigit():
                j += 3
                continue
            elif ch in ";)|&":
                break
            j += 1
        if q:
            return out                                   # an unterminated quote: leave the line alone
        out = out[: m.start()] + (" " if j < len(out) and out[j] in ";|&" else "") + out[j:]


def no_default(run: str) -> str:
    """The no-default repair (without the strict shell); the script unchanged when no `|| echo …` is found."""
    lines = run.splitlines()
    out: list[str] = []
    for line in lines:
        if _DEFAULT_LINE.match(line):
            # a fallback on a continuation line of its own: drop it, and the previous line's continuation
            if out and out[-1].rstrip().endswith("\\"):
                out[-1] = out[-1].rstrip()[:-1].rstrip()
            continue
        out.append(_strip_defaults(line))
    return "\n".join(out) + ("\n" if run.endswith("\n") else "")


def transform(run: str, name: str) -> tuple[str | None, str | None]:
    """(repaired script, reason it does not apply)."""
    if name == "guard-status":
        new = guard_status(run)
        return (new, None) if new != run else (None, "no single-line `if CMD; then` with a guardable command")
    if name == "no-default":
        nd = no_default(run)
        if nd == run:
            return None, "no `|| echo …` fallback"
        return strict_shell(nd), None
    if name == "both-structural":
        g = guard_status(run)
        nd = no_default(g)
        if g == run or nd == g:
            return None, "both edits must apply"
        return strict_shell(nd), None
    return None, "unknown repair"


def apply_structural(text: str, jid: str, i: int, name: str) -> tuple[str | None, str | None]:
    pos = locate(text, jid, i)
    if pos is None or pos["run"] is None:
        return None, "step not located in the workflow text, or no run:"
    new_run, why = transform(pos["run"]["value"], name)
    if new_run is None:
        return None, why
    lines = text.splitlines()
    out = repair._replace_run(lines, pos["run"], new_run)
    if out is None:
        return None, "run: value could not be rewritten in place"
    return "\n".join(out) + "\n", None


# ----------------------------------------------------------------------------- verification (SWALLOW-4's, over these candidates)

def try_structural(text: str, wf_name: str, jid: str, i: int, runner: faults.Runner, baseline: dict | None = None) -> dict:
    import yaml
    doc = yaml.safe_load(text)
    base_faults, base_plus = analyse(doc, wf_name, runner)
    base = base_faults.get((jid, i))
    rec = {"job": jid, "index": i, "baseline": None, "candidates": [], "verified_repair": None}
    if base is None:
        rec["baseline"] = {"verdict": None, "note": "not a fault site on this machine"}
        return rec
    rec["baseline"] = {"verdict": base["verdict"], "verdict_runs_only": base.get("verdict_runs_only"), "by_flavour": base["by_flavour"],
                       "alone": base["alone"], "name": base["name"], "continue_on_error": base["continue_on_error"]}
    if baseline is not None and baseline.get("verdict") != base["verdict"]:
        rec["baseline"]["differs_from_receipt"] = baseline.get("verdict")
    interpretable = [f for f, v in base["by_flavour"].items() if v in faults._PRECEDENCE]
    base_shape = {f: _healthy_shape(base_plus[f]) for f in faults.FLAVOURS}
    for name in REPAIRS:
        cand = {"repair": name}
        new_text, why = apply_structural(text, jid, i, name)
        if new_text is None:
            cand.update(applies=False, why=why)
            rec["candidates"].append(cand)
            continue
        try:
            new_doc = yaml.safe_load(new_text)
        except Exception as e:  # noqa: BLE001
            cand.update(applies=False, why=f"repaired text does not parse: {str(e)[:80]}")
            rec["candidates"].append(cand)
            continue
        new_faults, new_plus = analyse(new_doc, wf_name, runner)
        after = new_faults.get((jid, i))
        cand.update(applies=True, diff=unified_diff(text, new_text, wf_name), lines_changed=diff_size(text, new_text))
        if after is None:
            cand.update(loud=False, unchanged=None, verified=False, why="the repaired step is no longer a fault site (reaches no tool alone)")
            rec["candidates"].append(cand)
            continue
        cand["verdict_after"] = after["verdict"]
        cand["by_flavour_after"] = after["by_flavour"]
        unchanged_by = {f: _healthy_shape(new_plus[f]) == base_shape[f] for f in faults.FLAVOURS}
        loud_by = {f: after["by_flavour"].get(f) == "RED" for f in interpretable}
        unchanged = all(unchanged_by.values())
        loud = bool(loud_by) and all(loud_by.values())
        cand.update(loud=loud, loud_by_flavour=loud_by, unchanged=unchanged, unchanged_by_flavour=unchanged_by, verified=loud and unchanged)
        if not unchanged:
            bad = [f for f, ok in unchanged_by.items() if not ok]
            cand["why"] = "changes the healthy run in flavour " + ", ".join(bad) + ": what the repaired line was protecting"
            cand["healthy_change"] = _first_change(base_plus, new_plus, bad[0])
        elif not loud:
            cand["why"] = "not loud: " + ", ".join(f"{f}={after['by_flavour'].get(f)}" for f in interpretable)
        rec["candidates"].append(cand)
        if cand["verified"] and rec["verified_repair"] is None:
            rec["verified_repair"] = name
    return rec


# ----------------------------------------------------------------------------- drivers

def structural_tree(tree: Path, repo: str | None = None) -> dict:
    """Every hidden or dropped check of one checkout that SWALLOW-4's repairs leave unverified,
    with its structural candidates."""
    first = repair.repair_tree(tree, repo)
    out = {"repo": repo, "workflows": first["workflows"], "targets": [], "unparseable": first["unparseable"], "first_stage": first["targets"]}
    residue = [t for t in first["targets"] if not t["verified_repair"]]
    by_wf: dict = {}
    for t in residue:
        by_wf.setdefault(t["workflow"], []).append(t)
    for wf_name, ts in by_wf.items():
        text = (tree / ".github" / "workflows" / wf_name).read_text(encoding="utf-8", errors="replace")
        runner = faults.Runner()
        for t in ts:
            rec = try_structural(text, wf_name, t["job"], t["index"], runner)
            rec.update({k: t[k] for k in ("workflow", "name", "generated", "verdict", "verdict_counted", "stratum", "continue_on_error", "run_head", "check", "fewer")})
            rec["first_stage"] = [(c["repair"], (c.get("why") or "verified")[:80]) for c in t["candidates"]]
            out["targets"].append(rec)
    return out


def summary(results: dict) -> dict:
    ts = [t for r in results["repos"] for t in r["targets"] if "candidates" in t]

    def block(sel):
        sel = list(sel)
        ver = [t for t in sel if t["verified_repair"]]
        return {"targets": len(sel), "verified": len(ver), "by_repair": faults._count(t["verified_repair"] for t in ver),
                "baseline_differs_from_receipt": sum(1 for t in sel if t["baseline"].get("differs_from_receipt") or t["baseline"]["verdict"] is None),
                "rejected_changes_healthy_run": sum(1 for t in sel if not t["verified_repair"] and any(c.get("applies") and c.get("unchanged") is False for c in t["candidates"])),
                "no_candidate_applies": sum(1 for t in sel if all(not c.get("applies") for c in t["candidates"])),
                "not_loud": sum(1 for t in sel if not t["verified_repair"] and any(c.get("applies") for c in t["candidates"])
                                and all(c.get("unchanged") is not False for c in t["candidates"])),
                "lines_changed": sorted(next(c["lines_changed"] for c in t["candidates"] if c["repair"] == t["verified_repair"]) for t in ver)}
    hand = [t for t in ts if not t["generated"]]
    return {"all": block(ts), "hand_written": block(hand), "generated": block(t for t in ts if t["generated"]),
            "hand_written_swallowed": block(t for t in hand if t["verdict"] == "SWALLOWED"),
            "hand_written_fail_open": block(t for t in hand if t["verdict"] == "FAIL_OPEN")}
