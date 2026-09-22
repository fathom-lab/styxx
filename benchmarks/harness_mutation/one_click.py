# -*- coding: utf-8 -*-
"""SWALLOW-11 -- one click from loud: every hidden check the gate has caught, replayed as the Action
would have reported it.

Three receipts hold every BASE..HEAD pair the gate fired on: SWALLOW-7 (mainline commits of 96
repositories, 2019-2026; BASE the first parent), SWALLOW-9 (pull requests five coding agents
opened; BASE from the pull request's own commit list) and SWALLOW-10 (the pull requests people
opened in the same repositories and months, and the agents' it read again; BASE the closest
branch). For each pair this module re-reads the change with the product -- the living
`styxx.ciaudit.differential.audit_commit` -- and, for every newly hidden check, asks what
`styxx/ciaudit/action.py` would have put in front of the author:

  located     the annotation's line was found in HEAD's text (`positions`, `target`)
  visible     that line lies inside the change's diff as a pull request shows it (three lines of
              context): the Files-changed view shows the annotation where the reader is reading
  rebuilt     the verified repair's text was rebuilt, identical to the diff the gate printed
  reproduces  the suggestion, applied to HEAD's text, is that repaired text, line for line
  one-click   rebuilt, reproduced, and the suggestion's lines lie inside one hunk of the change's
              diff -- where GitHub's review API accepts a suggestion, and one click applies it
  size        the number of HEAD's lines the suggestion replaces

The receipt carries repository, shas, the check (job, step, kind, verdict, mechanism), the repair,
and the five flags with the lines they were decided on. No name, no address, no text.

    python -m benchmarks.harness_mutation.one_click --r7 papers/harness/swallow7_receipt.json.gz \
        --r9 papers/harness/swallow9_receipt.json.gz --r10 papers/harness/swallow10_receipt.json.gz \
        --clones7 <swallow-6 clones> --work <clones> --out papers/harness/swallow11_receipt.json
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from styxx.ciaudit import action as A          # noqa: E402  -- the product, read as it ships
from styxx.ciaudit import differential as D    # noqa: E402

SCHEMA = "styxx.harness-one-click/v1"
AGENTS = ("OpenAI_Codex", "Copilot", "Devin", "Cursor", "Claude_Code", "Google_Jules")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def instrument_sha256() -> str:
    return _sha(Path(__file__))


def _load(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    return json.loads((gzip.decompress(raw) if path.suffix == ".gz" else raw).decode("utf-8")), hashlib.sha256(raw).hexdigest()


def _expected(detail: list[dict]) -> list[list[str]]:
    return sorted([d["workflow"], n["job"], n["step"], n["kind"]] for d in detail or [] for n in d.get("new_hidden", []))


# ----------------------------------------------------------------------------- the population

def population(r7: dict, r9: dict, r10: dict) -> list[dict]:
    """Every firing pair of the three receipts, once: SWALLOW-9's reading of an agent's pull request
    is kept over SWALLOW-10's second reading of it."""
    pairs = []
    for x in r7.get("repos", []):
        for c in x["commits"]:
            if c.get("fires") and not c.get("root"):
                pairs.append({"source": "swallow7", "group": "mainline", "repo": x["repo"], "head": c["sha"], "base": None,
                              "time": c.get("time"), "expected": _expected(c.get("detail"))})
    seen = set()
    for x in r9.get("repos", []):
        for p in x["prs"]:
            if p.get("audited") and p.get("fires"):
                seen.add((x["repo"], p["head"]))
                pairs.append({"source": "swallow9", "group": p["agent"], "repo": x["repo"], "number": p["number"], "base": p["base"], "head": p["head"],
                              "merged": p.get("merged"), "expected": _expected(p.get("detail"))})
    for x in r10.get("repos", []):
        for p in x["prs"]:
            if p.get("audited") and p.get("touching") and p.get("fires") and (x["repo"], p["head"]) not in seen:
                pairs.append({"source": "swallow10", "group": p["group"], "repo": x["repo"], "number": p["number"], "base": p["base"], "head": p["head"],
                              "merged": p.get("merged_heuristic"), "expected": _expected(p.get("detail"))})
    return pairs


# ----------------------------------------------------------------------------- clones

def pr_clone(repo: str, work: Path, shas: list[str]) -> Path:
    """A blobless partial clone holding the named commits (fetched by name, depth 1); blobs arrive
    on demand when a workflow is read."""
    dest = work / repo.replace("/", "__")
    if not (dest / ".git").exists():
        p = subprocess.run(["git", "clone", "--quiet", "--filter=blob:none", "--no-checkout", "--depth=1", f"https://github.com/{repo}.git", str(dest)],
                           capture_output=True, text=True, timeout=1800)
        if p.returncode != 0:
            raise RuntimeError("clone: " + p.stderr.strip()[-160:])
        subprocess.run(["git", "-C", str(dest), "config", "gc.auto", "0"], capture_output=True)
    missing = [s for s in dict.fromkeys(shas) if s and not A.have(dest, s)]
    if missing:
        subprocess.run(["git", "-C", str(dest), "fetch", "--quiet", "--filter=blob:none", "--depth=1", "--no-tags", "--no-write-fetch-head", "origin", *missing],
                       capture_output=True, text=True, timeout=1800)
    return dest


# ----------------------------------------------------------------------------- one pair

def replay_pair(clone: Path, base: str, head: str) -> dict:
    """The change re-read by the product, and every newly hidden check as the Action would report it."""
    A.readable(clone, base, head)
    renamed = {ch["path"]: ch["from"] for ch in D.changed_workflows(clone, base, head)}
    rec = D.audit_commit(clone, base, head, readers={}, fix=True)
    checks = []
    for w in rec["workflows"]:
        if not w["new_hidden"]:
            continue
        text = D._text(clone, head, w["path"])
        hk = A.hunks(clone, base, head, w["path"], before=renamed.get(w["path"]))
        for x in w["new_hidden"]:
            fx = x.get("fix") or {}
            c = {"workflow": w["workflow"], "path": w["path"], "status": w["status"], "job": x["job"], "step": x["step"], "index": x["index"],
                 "kind": x["kind"], "verdict": x["verdict"], "mechanism": x.get("mechanism"), "continue_on_error": x.get("continue_on_error"),
                 "repair": fx.get("verified_repair"), "stage": fx.get("stage"), "lines_changed": fx.get("lines_changed"),
                 "located": False, "visible": False, "rebuilt": False, "reproduces": False, "one_click": False}
            pos = A.positions(text, x["job"], x["index"]) if text is not None else None
            if pos is not None:
                a, b, what = A.target(pos, x)
                c.update(located=True, target=[a, b], target_what=what, visible=A.within((a, a), hk))
            new = A.repaired_text(text, w["workflow"], x) if text is not None and fx.get("verified_repair") else None
            if new is not None:
                c["rebuilt"] = True
                sg = A.suggestion(text, new)
                if sg is not None:
                    c["reproduces"] = A.apply_suggestion(text, sg).splitlines() == new.splitlines()
                    c.update(span=[sg["start_line"], sg["line"]], size=len(sg["replaces"]), added=len(sg["lines"]))
                    c["one_click"] = bool(c["reproduces"] and A.within((sg["start_line"], sg["line"]), hk))
            if not c["one_click"]:
                c["why_not"] = ("no verified repair" if not fx.get("verified_repair") else "not rebuilt" if not c["rebuilt"] else
                                "not reproduced" if not c["reproduces"] else "outside the change's diff")
            checks.append(c)
    return {"workflows_changed": len(rec["workflows"]), "new_hidden": rec["new_hidden"], "checks": checks,
            "got": sorted([c["workflow"], c["job"], c["step"], c["kind"]] for c in checks)}


# ----------------------------------------------------------------------------- the run

def run(pairs: list[dict], clones7: Path, work: Path, deadline: float | None = None) -> list[dict]:
    out = []
    by_repo: dict = {}
    for p in pairs:
        by_repo.setdefault((p["source"] == "swallow7", p["repo"]), []).append(p)
    for (is7, repo), ps in sorted(by_repo.items(), key=lambda kv: (not kv[0][0], kv[0][1])):
        t0 = time.time()
        try:
            clone = clones7 / repo.replace("/", "__") if is7 else pr_clone(repo, work, [s for p in ps for s in (p["base"], p["head"])])
            if is7 and not (clone / ".git").exists():
                raise RuntimeError("no SWALLOW-6 clone")
        except (RuntimeError, OSError, subprocess.SubprocessError) as e:
            for p in ps:
                out.append(dict(p, error=str(e)[:200]))
            continue
        for p in ps:
            q = dict(p)
            if deadline and time.time() > deadline:
                q["error"] = "capped"
                out.append(q)
                continue
            try:
                if is7:
                    q["base"] = A.git(clone, "rev-parse", f"{p['head']}^1").strip() or None
                    if not q["base"]:
                        raise RuntimeError("the first parent is not in the clone")
                r = replay_pair(clone, q["base"], q["head"])
                q.update(r)
                q["reproduced"] = r["got"] == p["expected"]
            except Exception as e:  # noqa: BLE001 -- one pair's failure is recorded, not fatal
                q["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            out.append(q)
        print(f"  {repo}: {len(ps)} pairs, {sum(len(q.get('checks', [])) for q in out[-len(ps):])} checks, "
              f"{sum(1 for q in out[-len(ps):] if q.get('reproduced'))} reproduced, {round(time.time() - t0, 1)}s", file=sys.stderr, flush=True)
    return out


def summary(pairs: list[dict]) -> dict:
    ok = [p for p in pairs if "error" not in p]
    checks = [dict(c, group=p["group"], source=p["source"]) for p in ok for c in p["checks"]]
    def share(sel):
        xs = [c for c in checks if sel(c)]
        return {"n": len(xs), "one_click": sum(1 for c in xs if c["one_click"])}
    return {"pairs": len(pairs), "re_read": len(ok), "reproduced": sum(1 for p in ok if p.get("reproduced")), "errors": len(pairs) - len(ok),
            "checks": len(checks), "located": sum(1 for c in checks if c["located"]), "visible": sum(1 for c in checks if c["visible"]),
            "with_repair": sum(1 for c in checks if c["repair"]), "rebuilt": sum(1 for c in checks if c["rebuilt"]),
            "reproduces": sum(1 for c in checks if c["reproduces"]), "one_click": sum(1 for c in checks if c["one_click"]),
            "by_source": {s: share(lambda c, s=s: c["source"] == s) for s in ("swallow7", "swallow9", "swallow10")},
            "by_group": {"mainline": share(lambda c: c["group"] == "mainline"), "agents": share(lambda c: c["group"] in AGENTS),
                         "human": share(lambda c: c["group"] == "human"), "agent-signed": share(lambda c: c["group"] == "agent-signed")},
            "by_kind": {k: share(lambda c, k=k: c["kind"] == k) for k in ("born hidden", "acquired", "became a check, hidden")}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--r7", required=True)
    ap.add_argument("--r9", required=True)
    ap.add_argument("--r10", required=True)
    ap.add_argument("--clones7", required=True, help="the SWALLOW-6 clones (owner__repo)")
    ap.add_argument("--work", default="s11prs")
    ap.add_argument("--out")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args(argv)
    t0 = time.time()
    (r7, h7), (r9, h9), (r10, h10) = _load(Path(a.r7)), _load(Path(a.r9)), _load(Path(a.r10))
    pairs = population(r7, r9, r10)
    pairs = pairs[: a.limit] if a.limit else pairs
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    out = run(pairs, Path(a.clones7), work)
    rec = {"schema": SCHEMA, "instrument": "benchmarks/harness_mutation/one_click.py", "instrument_sha256": instrument_sha256(),
           "action_sha256": _sha(ROOT / "styxx" / "ciaudit" / "action.py"), "differential_living_sha256": _sha(ROOT / "styxx" / "ciaudit" / "differential.py"),
           "sources_sha256": {"swallow7": h7, "swallow9": h9, "swallow10": h10}, "pairs": out, "summary": summary(out), "seconds": round(time.time() - t0, 1)}
    if a.out:
        Path(a.out).write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(rec["summary"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
