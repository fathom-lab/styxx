# -*- coding: utf-8 -*-
"""SWALLOW-4 -- the repair is loud.

SWALLOW-2 and SWALLOW-3 read, per fault, whether a workflow goes red, drops a check, or hides a
check's own failure. This module asks the next question of every hidden and dropped check: is
there a small change to the workflow that makes the same fault loud, and leaves a healthy run
exactly as it was? It tries two stated repairs at the fault site, re-simulates the same fault on
the repaired workflow with the same instrument, and records both halves of the verdict:

- LOUD:      the fault's verdict on the repaired workflow is RED, in every healthy flavour that
             could interpret it before;
- UNCHANGED: the healthy worlds of the repaired workflow are indistinguishable from the original's
             -- every job has the same result, every step is run or not run for the same reason,
             reaches a tool or not, and is a model artifact or not, in both flavours.

A repair is VERIFIED when both hold. A repair that is loud but changes a healthy run is what a
`|| true` was protecting; it is recorded as rejected, with the flavour that rejected it.

The two repairs, in order (the first verified one is the fault's repair):

  no-continue-on-error   the fault step's `continue-on-error: true` (or its job's) is removed
  strict-shell           in the fault step's script: every trailing `|| true` / `|| :` is removed,
                         `set +e` lines are removed, and `set -eo pipefail` is ensured
  both                   the two together, when both apply

Repairs are edits to the workflow's TEXT (located through the YAML parser's marks), so the receipt
carries a real unified diff per candidate. The instrument underneath is the SWALLOW-3 reading of
the frozen `faults.py` (`action_checks.analyse_workflow`), unchanged.

    python -m benchmarks.harness_mutation.repair --tree .                        # every hidden / dropped check of a checkout
    python -m benchmarks.harness_mutation.repair --receipt papers/harness/swallow3_receipt.json.gz \
        --clones <clones> --out papers/harness/swallow4_receipt.json               # the population's, from SWALLOW-3's receipt
"""
from __future__ import annotations

import argparse
import difflib
import gzip
import hashlib
import json
import re
import sys
import time
from pathlib import Path

from . import action_checks as ac
from . import faults

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "styxx.harness-repair/v1"
TARGET_VERDICTS = ("SWALLOWED", "FAIL_OPEN")
REPAIRS = ("no-continue-on-error", "strict-shell", "both")

# ----------------------------------------------------------------------------- the two repairs, on text

_TRAIL_TRUE = re.compile(r"\s*\|\|\s*(?:true|:)\s*$")
_INNER_TRUE = re.compile(r"\s*\|\|\s*(?:true|:)(?=\s*[;)&|])")
_SET_PLUS_E = re.compile(r"^\s*set\s+\+e\b.*$")
_SET_E = re.compile(r"^\s*set\s+-[a-zA-Z]*e[a-zA-Z]*(\s|$)")
_SET_PIPEFAIL = re.compile(r"^\s*set\s+.*\bpipefail\b")
_SHEBANG = re.compile(r"^#!")


def strict_shell(run: str) -> str:
    """The strict-shell repair of a script; returns the script unchanged when nothing applies."""
    out = []
    for line in run.splitlines():
        if _SET_PLUS_E.match(line):
            continue
        line = _TRAIL_TRUE.sub("", line)
        line = _INNER_TRUE.sub("", line)
        out.append(line)
    has_e = any(_SET_E.match(line) for line in out)
    has_pipefail = any(_SET_PIPEFAIL.match(line) for line in out)
    head = []
    if not has_e and not has_pipefail:
        head = ["set -eo pipefail"]
    elif not has_e:
        head = ["set -e"]
    elif not has_pipefail:
        head = ["set -o pipefail"]
    if head:
        i = 1 if out and _SHEBANG.match(out[0]) else 0
        out = out[:i] + head + out[i:]
    new = "\n".join(out) + ("\n" if run.endswith("\n") else "")
    return new


# ----------------------------------------------------------------------------- locating a step in the text

def _get(mapping, key):
    for k, v in getattr(mapping, "value", []) or []:
        if getattr(k, "value", None) == key:
            return k, v
    return None, None


def locate(text: str, jid: str, i: int) -> dict | None:
    """Line positions of a step's `run:` value, its `continue-on-error:` key, and its job's, from
    the parser's marks. None when the workflow or the step cannot be located."""
    import yaml
    try:
        root = yaml.compose(text)
    except Exception:  # noqa: BLE001
        return None
    _, jobs = _get(root, "jobs")
    _, job = _get(jobs, jid) if jobs is not None else (None, None)
    _, steps = _get(job, "steps") if job is not None else (None, None)
    if steps is None or not hasattr(steps, "value") or i >= len(steps.value):
        return None
    st = steps.value[i]
    rk, rv = _get(st, "run")
    ck, cv = _get(st, "continue-on-error")
    jk, jv = _get(job, "continue-on-error")
    out = {"run": None, "step_coe": None, "job_coe": None}
    if rv is not None:
        out["run"] = {"style": rv.style, "line": rv.start_mark.line, "end": rv.end_mark.line, "col": rk.start_mark.column, "value": rv.value}
    if ck is not None and cv is not None and getattr(cv, "value", None) in ("true", True):
        out["step_coe"] = {"line": ck.start_mark.line, "end": cv.end_mark.line}
    if jk is not None and jv is not None and getattr(jv, "value", None) in ("true", True):
        out["job_coe"] = {"line": jk.start_mark.line, "end": jv.end_mark.line}
    return out


def _delete_key_line(lines: list[str], pos: dict) -> list[str] | None:
    """Delete a `key: value` that sits alone on its line; None when it does not."""
    if pos["end"] != pos["line"] and pos["end"] != pos["line"] + 1:
        return None
    line = lines[pos["line"]]
    if not re.match(r"^\s*continue-on-error:\s*(true|True|yes)\s*(#.*)?$", line):
        return None
    return lines[: pos["line"]] + lines[pos["line"] + 1:]


def _replace_run(lines: list[str], pos: dict, new_run: str) -> list[str] | None:
    """Replace the text of a `run:` value in place, as a block scalar; None when the value cannot
    be located as one line or one block."""
    line = pos["line"]
    style = pos["style"]
    indent = " " * pos["col"]
    body_indent = indent + "  "
    if style in ("|", ">"):
        # content lines are line+1 .. end-1; keep their indentation
        content = lines[line + 1: pos["end"]]
        first = next((c for c in content if c.strip()), None)
        if first is not None:
            body_indent = first[: len(first) - len(first.lstrip())]
        end = pos["end"]
    else:
        if pos["end"] != line:
            return None
        end = line + 1
    head = lines[line]
    m = re.match(r"^(\s*(?:-\s+)?run:)", head)
    if not m:
        return None
    new_lines = [m.group(1) + " |"] + [(body_indent + l) if l.strip() else "" for l in new_run.rstrip("\n").splitlines()]
    return lines[:line] + new_lines + lines[end:]


def apply_repair(text: str, jid: str, i: int, repair: str) -> tuple[str | None, str | None]:
    """(repaired text, reason it does not apply). A repair that changes nothing does not apply."""
    pos = locate(text, jid, i)
    if pos is None:
        return None, "step not located in the workflow text"
    lines = text.splitlines()
    changed = False
    if repair in ("no-continue-on-error", "both"):
        coe_positions = [p for p in (pos["step_coe"], pos["job_coe"]) if p]
        if not coe_positions:
            if repair == "no-continue-on-error":
                return None, "no continue-on-error on the step or its job"
        else:
            # delete from the bottom up so earlier line numbers stay valid
            for p in sorted(coe_positions, key=lambda p: -p["line"]):
                out = _delete_key_line(lines, p)
                if out is None:
                    return None, "continue-on-error is not alone on its line"
                lines = out
            changed = True
            # positions moved: relocate the run block on the edited text
            pos = locate("\n".join(lines) + "\n", jid, i)
            if pos is None:
                return None, "step not located after the edit"
    if repair in ("strict-shell", "both"):
        if pos["run"] is None:
            return None, "no run: on the step"
        new_run = strict_shell(pos["run"]["value"])
        if new_run == pos["run"]["value"]:
            if repair == "strict-shell":
                return None, "nothing to make strict: no || true, no set +e, and set -e / pipefail already present"
        else:
            out = _replace_run(lines, pos["run"], new_run)
            if out is None:
                return None, "run: value could not be rewritten in place"
            lines = out
            changed = True
    if not changed:
        return None, "the repair changes nothing"
    return "\n".join(lines) + "\n", None


def unified_diff(before: str, after: str, name: str) -> str:
    return "".join(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True),
                                        fromfile=f"a/.github/workflows/{name}", tofile=f"b/.github/workflows/{name}", n=2))


def diff_size(before: str, after: str) -> int:
    """Lines added plus lines removed."""
    return sum(1 for l in difflib.unified_diff(before.splitlines(), after.splitlines(), n=0, lineterm="")
               if (l.startswith("+") or l.startswith("-")) and not l.startswith(("+++", "---")))


# ----------------------------------------------------------------------------- verification

def _healthy_shape(sim: dict) -> list:
    """What a healthy run looks like, to the reader: job results; for every step, whether it ran,
    why not, whether it reached a tool, and whether it is a model artifact."""
    out = []
    for jid, job in sim.items():
        out.append((jid, job["result"], job.get("skipped")))
        for s in job["steps"]:
            out.append((jid, s["index"], s["ran"], s["why"], bool(s.get("n_reached")), bool(s.get("artifact"))))
    return out


def _alone(doc: dict, runner: faults.Runner) -> dict:
    alone = {}
    for jid, job in (doc.get("jobs") or {}).items():
        if not isinstance(job, dict):
            continue
        for i, st in enumerate(job.get("steps") or []):
            if isinstance(st, dict) and isinstance(st.get("run"), str) and faults._shell_for(st, job, doc) in ("bash", "sh"):
                alone[(jid, i)] = faults.alone_verdict(st["run"], runner)
    return alone


def analyse(doc: dict, wf_name: str, runner: faults.Runner) -> tuple[dict, dict]:
    """The SWALLOW-3 reading of one workflow: faults by (job, index), and the healthy worlds."""
    fl, _ = ac.analyse_workflow(doc, wf_name, runner, _alone(doc, runner))
    plus = {f: faults.simulate(doc, runner, None, f) for f in faults.FLAVOURS}
    return {(f["job"], f["index"]): f for f in fl}, plus


def try_repairs(text: str, wf_name: str, jid: str, i: int, runner: faults.Runner, baseline: dict | None = None) -> dict:
    """Every candidate repair of one fault site, verified. `baseline` is the fault's record on the
    original text when already known (else it is computed)."""
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
    interpretable_flavours = [f for f, v in base["by_flavour"].items() if v in faults._PRECEDENCE]
    base_shape = {f: _healthy_shape(base_plus[f]) for f in faults.FLAVOURS}
    for repair in REPAIRS:
        cand = {"repair": repair}
        new_text, why = apply_repair(text, jid, i, repair)
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
        loud_by = {f: after["by_flavour"].get(f) == "RED" for f in interpretable_flavours}
        unchanged = all(unchanged_by.values())
        loud = bool(loud_by) and all(loud_by.values())
        cand.update(loud=loud, loud_by_flavour=loud_by, unchanged=unchanged, unchanged_by_flavour=unchanged_by, verified=loud and unchanged)
        if not unchanged:
            bad = [f for f, ok in unchanged_by.items() if not ok]
            cand["why"] = "changes the healthy run in flavour " + ", ".join(bad) + ": what the repaired line was protecting"
            cand["healthy_change"] = _first_change(base_plus, new_plus, bad[0])
        elif not loud:
            cand["why"] = "not loud: " + ", ".join(f"{f}={after['by_flavour'].get(f)}" for f in interpretable_flavours)
        rec["candidates"].append(cand)
        if cand["verified"] and rec["verified_repair"] is None:
            rec["verified_repair"] = repair
    return rec


def _first_change(base_plus: dict, new_plus: dict, flavour: str) -> dict | None:
    a, b = _healthy_shape(base_plus[flavour]), _healthy_shape(new_plus[flavour])
    for x, y in zip(a, b):
        if x != y:
            return {"before": list(x), "after": list(y)}
    return {"before": None, "after": None, "note": "different length"}


# ----------------------------------------------------------------------------- drivers

def repair_tree(tree: Path, repo: str | None = None) -> dict:
    """Every hidden or dropped check of one checkout, with its candidate repairs."""
    import yaml
    out = {"repo": repo, "workflows": 0, "targets": [], "unparseable": []}
    wdir = tree / ".github" / "workflows"
    if not wdir.exists():
        return out
    for wf in sorted(list(wdir.glob("*.yml")) + list(wdir.glob("*.yaml"))):
        text = wf.read_text(encoding="utf-8", errors="replace")
        try:
            doc = yaml.safe_load(text) or {}
        except Exception as e:  # noqa: BLE001
            out["unparseable"].append({"workflow": wf.name, "error": str(e)[:120]})
            continue
        if not isinstance(doc, dict) or not isinstance(doc.get("jobs"), dict):
            continue
        out["workflows"] += 1
        runner = faults.Runner()
        fl, _ = ac.analyse_workflow(doc, wf.name, runner, _alone(doc, runner))
        for f in fl:
            if f["verdict"] in TARGET_VERDICTS:
                rec = try_repairs(text, wf.name, f["job"], f["index"], runner)
                rec.update(workflow=wf.name, name=f["name"], generated=f["generated"], verdict=f["verdict"], verdict_counted=f.get("verdict_counted"),
                           stratum="frozen", continue_on_error=f["continue_on_error"], run_head=f["run_head"], check=f.get("check"),
                           fewer=any(d.get("mechanism") == "fewer" for d in f.get("dropped_counted", [])))
                out["targets"].append(rec)
    return out


def repair_population(receipt: dict, clones: Path, counted: bool = True) -> dict:
    """The population's hidden and dropped checks, taken from a SWALLOW-3 receipt, repaired on the
    clones at the same HEADs. `counted` adds the faults that are FAIL_OPEN only under the counted
    reading, as a second stratum."""
    out = {"repos": [], "targets": 0}
    for rep in receipt["repos"]:
        targets = [f for f in rep.get("faults", []) if f["verdict"] in TARGET_VERDICTS or (counted and f.get("verdict_counted") == "FAIL_OPEN")]
        if not targets:
            continue
        dest = clones / rep["repo"].replace("/", "__")
        rrec = {"repo": rep["repo"], "head": rep.get("head"), "targets": [], "missing_clone": not dest.exists()}
        if not dest.exists():
            out["repos"].append(rrec)
            continue
        by_wf: dict = {}
        for f in targets:
            by_wf.setdefault(f["workflow"], []).append(f)
        for wf_name, fl in by_wf.items():
            path = dest / ".github" / "workflows" / wf_name
            if not path.exists():
                for f in fl:
                    rrec["targets"].append({"workflow": wf_name, "job": f["job"], "index": f["index"], "missing_workflow": True})
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            runner = faults.Runner()
            t1 = time.time()
            for f in fl:
                rec = try_repairs(text, wf_name, f["job"], f["index"], runner, baseline=f)
                rec.update(workflow=wf_name, name=f["name"], generated=f["generated"], verdict=f["verdict"], verdict_counted=f.get("verdict_counted"),
                           stratum="frozen" if f["verdict"] in TARGET_VERDICTS else "counted-only",
                           continue_on_error=f["continue_on_error"], run_head=f["run_head"], check=f.get("check"),
                           fewer=any(d.get("mechanism") == "fewer" for d in f.get("dropped_counted", [])))
                rrec["targets"].append(rec)
            rrec.setdefault("seconds", 0.0)
            rrec["seconds"] = round(rrec["seconds"] + time.time() - t1, 1)
        out["repos"].append(rrec)
        out["targets"] += len(rrec["targets"])
        sys.stderr.write(f"{rep['repo']}: {len(rrec['targets'])} targets, verified {sum(1 for t in rrec['targets'] if t.get('verified_repair'))}\n")
    return out


def summary(results: dict) -> dict:
    ts = [t for r in results["repos"] for t in r["targets"] if "candidates" in t]
    def block(sel):
        sel = list(sel)
        ver = [t for t in sel if t["verified_repair"]]
        return {"targets": len(sel), "verified": len(ver),
                "by_repair": faults._count(t["verified_repair"] for t in ver),
                "baseline_differs_from_receipt": sum(1 for t in sel if t["baseline"].get("differs_from_receipt") or t["baseline"]["verdict"] is None),
                "rejected_changes_healthy_run": sum(1 for t in sel if not t["verified_repair"] and any(c.get("applies") and c.get("unchanged") is False for c in t["candidates"])),
                "no_candidate_applies": sum(1 for t in sel if all(not c.get("applies") for c in t["candidates"])),
                "not_loud": sum(1 for t in sel if not t["verified_repair"] and any(c.get("applies") for c in t["candidates"])
                                and all(c.get("unchanged") is not False for c in t["candidates"])),
                "lines_changed": sorted(next(c["lines_changed"] for c in t["candidates"] if c["repair"] == t["verified_repair"]) for t in ver)}
    hand = [t for t in ts if not t["generated"]]
    return {"all": block(ts), "hand_written": block(hand),
            "hand_written_frozen": block(t for t in hand if t["stratum"] == "frozen"),
            "hand_written_swallowed": block(t for t in hand if t["verdict"] == "SWALLOWED"),
            "hand_written_fail_open": block(t for t in hand if t["verdict"] == "FAIL_OPEN"),
            "hand_written_counted_only": block(t for t in hand if t["stratum"] == "counted-only"),
            "hand_written_fewer": block(t for t in hand if t.get("fewer")),
            "hand_written_swallowed_by_continue_on_error": block(t for t in hand if t["verdict"] == "SWALLOWED" and t["continue_on_error"]),
            "hand_written_swallowed_by_the_shell": block(t for t in hand if t["verdict"] == "SWALLOWED" and not t["continue_on_error"]),
            "generated": block(t for t in ts if t["generated"])}


def instrument_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree")
    ap.add_argument("--receipt", help="a SWALLOW-3 receipt (.json or .json.gz) naming the faults to repair")
    ap.add_argument("--clones", help="the clones the receipt was made from")
    ap.add_argument("--no-counted", action="store_true", help="leave out the counted-only stratum")
    ap.add_argument("--out", default="repair_receipt.json")
    a = ap.parse_args(argv)
    t0 = time.time()
    if a.tree:
        rec = repair_tree(Path(a.tree).resolve())
        rec["summary"] = summary({"repos": [rec]})
        Path(a.out).write_text(json.dumps(rec, indent=1, default=str) + "\n", encoding="utf-8")
        print(json.dumps(rec["summary"]["all"], indent=1))
        return 0
    opener = gzip.open if a.receipt.endswith(".gz") else open
    with opener(a.receipt, "rt", encoding="utf-8") as f:
        receipt = json.load(f)
    res = repair_population(receipt, Path(a.clones), counted=not a.no_counted)
    res.update(schema=SCHEMA, instrument="benchmarks/harness_mutation/repair.py", instrument_sha256=instrument_sha256(),
               faults_sha256=hashlib.sha256((Path(__file__).parent / "faults.py").read_bytes()).hexdigest(),
               action_checks_sha256=hashlib.sha256((Path(__file__).parent / "action_checks.py").read_bytes()).hexdigest(),
               source_receipt=a.receipt, source_receipt_sha256=hashlib.sha256(Path(a.receipt).read_bytes()).hexdigest(),
               repairs=list(REPAIRS), seconds=round(time.time() - t0, 1))
    res["summary"] = summary(res)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str) + "\n", encoding="utf-8")
    print(json.dumps(res["summary"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
