# -*- coding: utf-8 -*-
"""styxx.ciaudit.history -- since when has this check been hiding its failure, and by whose hand?

The living copy of SWALLOW-6's instrument (`benchmarks/harness_mutation/history.py`, frozen; the
test suite holds this file to it). Every revision of a workflow on the checkout's own mainline
(first-parent from HEAD) is read with the engine, and every check step is followed as a LINEAGE --
the same job, the same step name (or id, or first line of its script) -- from the commit that
created it. For a finding of `styxx ci-audit`, the history says:

  hidden since <date> (<commit>): born hidden       the `|| true` was written with the step
  hidden since <date> (<commit>): acquired          a later commit hid it -- by continue-on-error,
                                                    `|| true`, `set +e`, a `|| echo` default, gating,
                                                    a rewrite, or a change elsewhere (context) --
                                                    and whether that commit's message says why
                                                    ("flaky", "for now", "non-blocking", ...)

Nothing is inferred from the message beyond the stated word list; nothing is read that is not on
the mainline; a shallow clone's oldest commit is a boundary, and a lineage born there may be older.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
import time
from pathlib import Path

from . import engine as faults          # the living copy of the instrument; the names below read as in benchmarks/harness_mutation/history.py
from . import repair
from . import repair_structural as rs

SCHEMA = "styxx.ci-audit.history/v1"
SINCE = "2019-08-01"
HIDDEN = ("SWALLOWED", "FAIL_OPEN")
LOUD = ("RED",)
UNREAD = ("BASELINE_RED", "BASELINE_SKIPPED")     # the reading could not interpret the step at that revision: carried over, counted
STATES = ("hidden", "loud", "other", "unread")   # other: not a check at that revision (no fault site, NO_CHECK, ABSORBED)
MECHANISMS = ("continue-on-error", "or-true", "set+e", "default", "strict-shell", "gating", "rewrite", "context")
# a candidate repair of SWALLOW-4 / SWALLOW-5 agrees with a wild repair when the wild mechanism is one it edits
CANDIDATE_MECHANISMS = {"no-continue-on-error": {"continue-on-error"}, "strict-shell": {"or-true", "set+e", "strict-shell"},
                        "both": {"continue-on-error", "or-true", "set+e", "strict-shell"},
                        "guard-status": {"rewrite"}, "no-default": {"default", "strict-shell"},
                        "both-structural": {"default", "strict-shell", "rewrite"}}
ACK = re.compile(r"\bflak|\btemporar|\bfor now\b|\bunblock|non[- ]?blocking|\bignor|\bskip|allow[- _]?fail|work[- ]?around|"
                 r"don'?t fail|\bnot fail|soft[- _]?fail|best[- _]?effort|\boptional\b|\bnoisy\b|\bunstable\b|\bintermittent|"
                 r"\bbroken\b|\bdisable|\bsilence|\bquiet\b|\bwarn", re.I)
_OR_TRUE = re.compile(r"\|\|\s*(?:true|:)(?=\s*$|\s*[;)&|])", re.M)
_SET_PLUS_E = re.compile(r"^\s*set\s+\+e\b", re.M)
_SET_E = re.compile(r"^\s*set\s+-[a-zA-Z]*e[a-zA-Z]*(\s|$)", re.M)
_PIPEFAIL = re.compile(r"^\s*set\s+.*\bpipefail\b", re.M)
_DEFAULT = re.compile(r"\|\|\s*echo\b")


# ----------------------------------------------------------------------------- git

def _git(clone: Path, args: list[str], timeout: int = 900, check: bool = True, data: str | None = None) -> str:
    p = subprocess.run(["git", "-C", str(clone), *args], capture_output=True, text=True, encoding="utf-8", errors="replace",
                       timeout=timeout, input=data)
    if check and p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args[:3])}: {p.stderr.strip()[:200]}")
    return p.stdout


def mainline(clone: Path, tip: str) -> list[dict]:
    """The first-parent commits from the oldest fetched to `tip` that touch .github/workflows, oldest
    first, each with the workflow paths it changed (renames followed)."""
    raw = _git(clone, ["log", "--first-parent", "--reverse", "--name-status", "-M", "--format=%x01%H%x00%ct%x00%P%x00%s%x00%b%x02",
                       tip, "--", ".github/workflows"])
    commits = []
    for chunk in raw.split("\x01")[1:]:
        head, _, files = chunk.partition("\x02")
        sha, ct, parents, subject, body = (head.split("\x00") + ["", "", "", "", ""])[:5]
        changed = []
        for line in files.strip().splitlines():
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            status = parts[0][0]
            if status == "R" and len(parts) >= 3:
                changed.append({"status": "R", "path": parts[2], "from": parts[1]})
            else:
                changed.append({"status": status, "path": parts[-1]})
        changed = [c for c in changed if _is_workflow(c["path"]) or (c.get("from") and _is_workflow(c["from"]))]
        if changed:
            commits.append({"sha": sha, "time": int(ct or 0), "parents": parents.split(), "subject": subject.strip(),
                            "body": body.strip(), "changed": changed})
    return commits


def _is_workflow(path: str) -> bool:
    return path.startswith(".github/workflows/") and path.count("/") == 2 and path.endswith((".yml", ".yaml"))


def merged_subjects(clone: Path, commit: dict, limit: int = 50) -> list[str]:
    """For a merge commit on the mainline, the subjects of the commits it merged."""
    if len(commit["parents"]) < 2:
        return []
    out = _git(clone, ["log", "--format=%s", f"-{limit}", f"{commit['parents'][0]}..{commit['parents'][1]}"], check=False)
    return [s.strip() for s in out.splitlines() if s.strip()]


def texts_at(clone: Path, wants: list[tuple[str, str]]) -> dict[tuple[str, str], str | None]:
    """The text of every (commit, path) in one cat-file batch; None where the path does not exist."""
    if not wants:
        return {}
    p = subprocess.run(["git", "-C", str(clone), "cat-file", "--batch"], input="".join(f"{sha}:{path}\n" for sha, path in wants).encode(),
                       capture_output=True, timeout=1800)
    data = p.stdout
    out: dict = {}
    pos = 0
    for want in wants:
        nl = data.index(b"\n", pos)
        header = data[pos:nl].decode("utf-8", "replace")
        pos = nl + 1
        if header.endswith(" missing") or header.endswith("ambiguous"):
            out[want] = None
            continue
        size = int(header.split()[2])
        out[want] = data[pos:pos + size].decode("utf-8", "replace")
        pos += size + 1
    return out


# ----------------------------------------------------------------------------- the reading, through time

def state_of(verdict: str | None) -> str:
    if verdict in HIDDEN:
        return "hidden"
    if verdict in LOUD:
        return "loud"
    if verdict in UNREAD:
        return "unread"
    return "other"


def step_key(step: dict) -> str:
    """The name a lineage is followed by: the step's name, else its id, else the first line of its script."""
    if isinstance(step.get("name"), str) and step["name"].strip():
        return "name:" + step["name"].strip()
    if isinstance(step.get("id"), str) and step["id"].strip():
        return "id:" + step["id"].strip()
    run = step.get("run") if isinstance(step.get("run"), str) else ""
    first = next((ln.strip() for ln in run.splitlines() if ln.strip()), "")
    return "run:" + first[:120]


def _features(run: str) -> dict:
    return {"or-true": bool(_OR_TRUE.search(run)), "set+e": bool(_SET_PLUS_E.search(run)), "default": bool(_DEFAULT.search(run)),
            "strict": bool(_SET_E.search(run) or _PIPEFAIL.search(run))}


def mechanism(before: dict, after: dict) -> list[str]:
    """What changed at the step between two revisions, as the stated classes, in order. `before`
    and `after` are {"step": ..., "job": ...} snapshots (the job without its steps)."""
    m = []
    bs, bj, as_, aj = before["step"], before["job"], after["step"], after["job"]
    coe = lambda s, j: s.get("continue-on-error") is True or j.get("continue-on-error") is True  # noqa: E731
    if coe(bs, bj) != coe(as_, aj):
        m.append("continue-on-error")
    rb = bs.get("run") if isinstance(bs.get("run"), str) else ""
    ra = as_.get("run") if isinstance(as_.get("run"), str) else ""
    fb, fa = _features(rb), _features(ra)
    for k in ("or-true", "set+e", "default"):
        if fb[k] != fa[k]:
            m.append(k)
    if fb["strict"] != fa["strict"]:
        m.append("strict-shell")
    if (bs.get("if") != as_.get("if")) or (bj.get("if") != aj.get("if")) or (bj.get("needs") != aj.get("needs")) \
            or (bj.get("strategy") != aj.get("strategy")):
        m.append("gating")
    if not m and rb != ra:
        m.append("rewrite")
    if not m:
        m.append("context")
    return m


def acknowledged(commit: dict, merged: list[str]) -> str | None:
    """The first acknowledgement in the commit's subject, body or merged subjects; None if none."""
    for text in [commit.get("subject", ""), commit.get("body", "")] + list(merged):
        hit = ACK.search(text or "")
        if hit:
            return hit.group(0).lower()
    return None


def _snapshot(doc: dict) -> dict:
    """Every step of every job, keyed (job, index): the step, its job without steps, and its key."""
    out = {}
    for jid, job in (doc.get("jobs") or {}).items():
        if not isinstance(job, dict):
            continue
        j = {k: v for k, v in job.items() if k != "steps"}
        for i, st in enumerate(job.get("steps") or []):
            if isinstance(st, dict):
                out[(jid, i)] = {"step": st, "job": j, "key": step_key(st),
                                 "run_sha": hashlib.sha256(st["run"].encode()).hexdigest()[:16] if isinstance(st.get("run"), str) else None}
    return out


class Reader:
    """The frozen reading, memoised by workflow text; one runner per workflow lineage."""

    def __init__(self) -> None:
        self.runner = faults.Runner()
        self.cache: dict[str, tuple] = {}
        self.reads = 0

    def read(self, text: str, wf_name: str) -> tuple[dict | None, dict, str | None]:
        """(faults by (job, index), snapshot of every step, parse error)."""
        import yaml
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if h in self.cache:
            return self.cache[h]
        try:
            doc = yaml.safe_load(text) or {}
        except Exception as e:  # noqa: BLE001
            res = (None, {}, str(e)[:120])
            self.cache[h] = res
            return res
        if not isinstance(doc, dict) or not isinstance(doc.get("jobs"), dict):
            res = (None, {}, "no jobs")
            self.cache[h] = res
            return res
        try:
            fl, _plus = repair.analyse(doc, wf_name, self.runner)
        except RecursionError:
            res = (None, {}, "recursion")
            self.cache[h] = res
            return res
        self.reads += 1
        res = (fl, _snapshot(doc), None)
        self.cache[h] = res
        return res


def follow(revisions: list[dict], reader: Reader, wf_name: str) -> list[dict]:
    """The lineages of one workflow through its revisions (oldest first; each {"sha", "time",
    "subject", "body", "merged", "text"}). A lineage is a job id and a step key; a step whose key
    changed but whose script did not is followed under its new key (a rename). A lineage's `state`
    and `verdict` are its last INTERPRETABLE reading (events are read between those); `state_last`
    and `verdict_last` are the reading at its most recent revision as it is, unread included -- what
    a check "is" at HEAD is `state_last`."""
    live: dict[tuple[str, str], dict] = {}
    done: list[dict] = []
    for n, rev in enumerate(revisions):
        text = rev["text"]
        if text is None:                                   # the workflow was removed at this revision
            for lin in live.values():
                _die(lin, rev, "workflow removed")
                done.append(lin)
            live = {}
            continue
        fl, snap, err = reader.read(text, wf_name)
        if err is not None:
            for lin in live.values():
                lin["unread"] = lin.get("unread", 0) + 1
            continue
        present: dict[tuple[str, str], tuple] = {}
        for (jid, i), s in snap.items():
            k = (jid, s["key"])
            if k in present:                               # two steps share a key: the first is followed
                continue
            present[k] = (jid, i)
        # renames: a live lineage whose key vanished, and a new key in the same job with the same script
        vanished = {k: lin for k, lin in live.items() if k not in present}
        new_keys = {k: ji for k, ji in present.items() if k not in live}
        for k, lin in list(vanished.items()):
            if lin.get("run_sha") is None:
                continue
            match = next((nk for nk, ji in new_keys.items() if nk[0] == k[0] and snap[ji]["run_sha"] == lin["run_sha"]), None)
            if match is not None:
                lin["renames"] = lin.get("renames", []) + [{"sha": rev["sha"], "from": k[1], "to": match[1], "revision": n}]
                lin["key"] = match[1]
                live[match] = live.pop(k)
                del new_keys[match]
                del vanished[k]
        for k, lin in vanished.items():
            _die(lin, rev, "step removed")
            done.append(lin)
            del live[k]
        for k, (jid, i) in present.items():
            f = (fl or {}).get((jid, i))
            verdict = f["verdict"] if f else None
            st = state_of(verdict)
            s = snap[(jid, i)]
            here = {"step": s["step"], "job": s["job"], "index": i, "sha": rev["sha"]}
            if st == "unread" and k in live:                   # uninterpretable here: the state is carried over, the revision counted
                lin = live[k]
                lin["revisions"] += 1
                lin["unread"] = lin.get("unread", 0) + 1
                lin["index"], lin["run_sha"], lin["_prev"] = i, s["run_sha"], here
                lin["state_last"], lin["verdict_last"] = st, verdict           # the reading at this revision, as it is
                continue
            if k not in live:
                live[k] = {"job": jid, "key": k[1], "born": {"sha": rev["sha"], "time": rev["time"], "subject": rev["subject"][:120], "state": st,
                                                          "verdict": verdict, "revision": n}, "state": st, "verdict": verdict, "events": [],
                           "revisions": 1, "run_sha": s["run_sha"], "index": i, "check": bool(f and f.get("check")),
                           "name": s["step"].get("name") if isinstance(s["step"].get("name"), str) else None,
                           "run_head": (s["step"]["run"].strip().splitlines() or [""])[0][:160] if isinstance(s["step"].get("run"), str) else "",
                           "state_last": st, "verdict_last": verdict, "_prev": here}
                continue
            lin = live[k]
            lin["revisions"] += 1
            lin["index"] = i
            lin["check"] = lin["check"] or bool(f and f.get("check"))
            if lin["state"] == "unread":                       # never interpretable before: this reading is the birth state
                lin["born"]["first_read"] = {"sha": rev["sha"], "time": rev["time"], "revision": n}
                lin["born"]["state"], lin["born"]["verdict"] = st, verdict
            elif st != lin["state"] and (st in ("hidden", "loud") or lin["state"] in ("hidden", "loud")):
                kind = {("loud", "hidden"): "acquisition", ("hidden", "loud"): "repair"}.get((lin["state"], st), f"{lin['state']}->{st}")
                before = lin["_prev"]
                ev = {"kind": kind, "sha": rev["sha"], "time": rev["time"], "subject": rev["subject"][:120], "from": lin["state"], "to": st,
                      "verdict_from": lin["verdict"], "verdict_to": verdict, "revision": n,
                      "mechanism": mechanism(before, here), "before": {"sha": before["sha"], "index": before["index"]},
                      "acknowledged": acknowledged(rev, rev.get("merged") or [])}
                lin["events"].append(ev)
            lin["state"], lin["verdict"] = st, verdict
            lin["state_last"], lin["verdict_last"] = st, verdict
            lin["run_sha"] = s["run_sha"]
            lin["_prev"] = here
    for lin in live.values():
        lin["alive"] = True
    out = done + list(live.values())
    for lin in out:
        lin.pop("_prev", None)
        lin.setdefault("alive", False)
    return out


def _die(lin: dict, rev: dict, why: str) -> None:
    lin["alive"] = False
    lin["death"] = {"sha": rev["sha"], "time": rev["time"], "subject": rev["subject"][:120], "state": lin["state"], "why": why}


def wild_repair_agreement(before_text: str, wf_name: str, jid: str, i: int, runner, wild: list[str]) -> dict:
    """What SWALLOW-4 (then SWALLOW-5) would have proposed on the revision before a wild repair, and
    whether the first verified candidate edits what the author edited."""
    rec = repair.try_repairs(before_text, wf_name, jid, i, runner)
    stage = "swallow-4"
    if rec.get("verified_repair") is None:
        rec2 = rs.try_structural(before_text, wf_name, jid, i, runner)
        if rec2.get("verified_repair") is not None:
            rec, stage = rec2, "swallow-5"
    v = rec.get("verified_repair")
    agrees = bool(v) and bool(CANDIDATE_MECHANISMS.get(v, set()) & set(wild))
    return {"verified_repair": v, "stage": stage if v else None, "agrees": agrees,
            "baseline": (rec.get("baseline") or {}).get("verdict"),
            "candidates": [(c["repair"], c.get("applies"), c.get("verified"), (c.get("why") or "")[:80]) for c in rec.get("candidates", [])]}


# ----------------------------------------------------------------------------- one repository

def history_of(clone: Path, tip: str, repo: str | None = None, deadline: float | None = None, agreement: bool = True) -> dict:
    """Every hand-written workflow's lineages on the mainline of `clone` up to `tip`."""
    t0 = time.time()
    out = {"repo": repo, "tip": tip, "since": SINCE, "workflows": {}, "capped": False, "unparseable": []}
    commits = mainline(clone, tip)
    out["mainline_commits_touching_workflows"] = len(commits)
    by_path: dict[str, list[dict]] = {}
    alias: dict[str, str] = {}
    for c in commits:
        for ch in c["changed"]:
            path = ch["path"]
            if ch["status"] == "R" and ch.get("from") in by_path:
                by_path[path] = by_path.pop(ch["from"])
                alias[ch["from"]] = path
            by_path.setdefault(path, []).append({"sha": c["sha"], "time": c["time"], "subject": c["subject"], "body": c["body"],
                                                 "parents": c["parents"], "status": ch["status"], "path": path})   # the path AT this revision
    merged_cache: dict[str, list[str]] = {}
    for path in sorted(by_path):
        if path.endswith(".lock.yml"):
            continue
        if deadline and time.time() > deadline:
            out["capped"] = True
            break
        wf_name = path.rsplit("/", 1)[-1]
        texts = texts_at(clone, [(r["sha"], r["path"]) for r in by_path[path] if r["status"] != "D"])   # one workflow's texts at a time, each at its path then
        revs = []
        for r in by_path[path]:
            text = None if r["status"] == "D" else texts.get((r["sha"], r["path"]))
            if len(r["parents"]) > 1:
                if r["sha"] not in merged_cache:
                    merged_cache[r["sha"]] = merged_subjects(clone, r)
                merged = merged_cache[r["sha"]]
            else:
                merged = []
            revs.append({"sha": r["sha"], "time": r["time"], "subject": r["subject"], "body": r["body"], "merged": merged, "text": text})
        reader = Reader()
        lineages = follow(revs, reader, wf_name)
        path_at = {r["sha"]: r["path"] for r in by_path[path]}                 # a revision before a rename lived at another path
        if agreement:
            for lin in lineages:
                for ev in lin["events"]:
                    if ev["kind"] == "repair" and ev.get("before"):
                        before_text = texts.get((ev["before"]["sha"], path_at.get(ev["before"]["sha"], path)))
                        if before_text is not None and ev["before"]["index"] is not None:
                            try:
                                ev["agreement"] = wild_repair_agreement(before_text, wf_name, lin["job"], ev["before"]["index"], reader.runner, ev["mechanism"])
                            except Exception as e:  # noqa: BLE001
                                ev["agreement"] = {"error": str(e)[:120]}
        checks = [lin for lin in lineages if lin["check"] or lin["born"]["state"] in ("hidden", "loud") or lin["state"] in ("hidden", "loud")
                  or lin["state_last"] in ("hidden", "loud")
                  or any(ev["from"] in ("hidden", "loud") or ev["to"] in ("hidden", "loud") for ev in lin["events"])]
        out["workflows"][wf_name] = {"path": path, "revisions": len(revs), "distinct_texts": len(reader.cache), "reads": reader.reads,
                                     "steps_followed": len(lineages), "lineages": checks,
                                     "unparseable_revisions": sum(1 for r in revs if r["text"] is not None and reader.read(r["text"], wf_name)[2])}
    boundary = shallow_boundary(clone)
    out["shallow_boundary_commits"] = len(boundary)
    for w in out["workflows"].values():
        for lin in w["lineages"]:
            if lin["born"]["sha"] in boundary:
                lin["born"]["at_boundary"] = True           # left-censored: the step may be older than the fetched history
    out["head_time"] = int(_git(clone, ["show", "-s", "--format=%ct", tip]).strip() or 0)
    out["seconds"] = round(time.time() - t0, 1)
    return out


def shallow_boundary(clone: Path) -> set[str]:
    """The commits at which a shallow clone's history stops."""
    path = _git(clone, ["rev-parse", "--git-path", "shallow"], check=False).strip()
    f = Path(path) if Path(path).is_absolute() else clone / path
    if not f.exists():
        return set()
    return {ln.strip() for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()}


def summary(rec: dict) -> dict:
    """The counts one repository's (or the population's merged) lineages add up to."""
    lins = [(wf, lin) for wf, w in rec["workflows"].items() for lin in w["lineages"]]
    head_time = rec.get("head_time") or 0
    alive_hidden = [(wf, l) for wf, l in lins if l.get("alive") and l["state_last"] == "hidden"]
    ever_hidden = [(wf, l) for wf, l in lins if l["born"]["state"] == "hidden" or any(ev["to"] == "hidden" for ev in l["events"])]
    acq = [(wf, l, ev) for wf, l in lins for ev in l["events"] if ev["kind"] == "acquisition"]
    rep = [(wf, l, ev) for wf, l in lins for ev in l["events"] if ev["kind"] == "repair"]
    died_hidden = [(wf, l) for wf, l in lins if not l.get("alive") and l["state_last"] == "hidden"]

    def first_hidden_time(l):
        """When the lineage last became hidden: its birth, or its last arrival in that state."""
        arrivals = [ev["time"] for ev in l["events"] if ev["to"] == "hidden"]
        if arrivals:
            return arrivals[-1]
        return l["born"]["time"] if l["born"]["state"] == "hidden" else None

    ages = sorted((head_time - first_hidden_time(l)) / 86400 for _, l in alive_hidden if first_hidden_time(l) is not None)
    return {"lineages": len(lins), "checks_followed": sum(1 for _, l in lins if l["check"]),
            "alive": sum(1 for _, l in lins if l.get("alive")), "alive_hidden": len(alive_hidden),
            "alive_hidden_born_hidden": sum(1 for _, l in alive_hidden if l["born"]["state"] == "hidden" and not any(ev["kind"] == "repair" for ev in l["events"])),
            "alive_hidden_acquired": sum(1 for _, l in alive_hidden if any(ev["kind"] == "acquisition" for ev in l["events"])),
            "ever_hidden": len(ever_hidden), "died_hidden": len(died_hidden), "died_loud": sum(1 for _, l in lins if not l.get("alive") and l["state_last"] == "loud"),
            "alive_unread": sum(1 for _, l in lins if l.get("alive") and l["state_last"] == "unread"),
            "acquisitions": len(acq), "acquisitions_acknowledged": sum(1 for *_, ev in acq if ev.get("acknowledged")),
            "acquisition_mechanisms": faults._count(ev["mechanism"][0] for *_, ev in acq),
            "repairs": len(rep), "repairs_acknowledged": sum(1 for *_, ev in rep if ev.get("acknowledged")),
            "repair_mechanisms": faults._count(ev["mechanism"][0] for *_, ev in rep),
            "repairs_with_agreement_read": sum(1 for *_, ev in rep if "agreement" in ev and "error" not in ev["agreement"]),
            "repairs_agreeing": sum(1 for *_, ev in rep if (ev.get("agreement") or {}).get("agrees")),
            "repairs_instrument_verified": sum(1 for *_, ev in rep if (ev.get("agreement") or {}).get("verified_repair")),
            "ages_days": [round(a, 1) for a in ages], "median_age_days": (ages[len(ages) // 2] if ages else None)}


def tree_history(tree: Path, repo: str | None = None, agreement: bool = True) -> dict:
    """This checkout's lineages, from its own git history (first-parent from HEAD)."""
    tip = _git(tree, ["rev-parse", "HEAD"]).strip()
    h = history_of(tree, tip, repo, agreement=agreement)
    h["summary"] = summary(h)
    h["shallow"] = (tree / ".git" / "shallow").exists()
    return h


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def instrument_sha256() -> str:
    return _sha(Path(__file__))


# ----------------------------------------------------------------------------- the product: since when

def deepen(tree: Path, since: str = SINCE) -> dict:
    """Give a shallow clone (`styxx ci-audit owner/repo` makes one) its mainline history since
    `since`, blob-less, and fetch every workflow blob along it in one batch."""
    info = {"since": since}
    p = subprocess.run(["git", "-C", str(tree), "fetch", "--quiet", "--filter=blob:none", f"--shallow-since={since}", "origin", "HEAD"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800)
    if p.returncode != 0:
        info["error"] = p.stderr.strip()[:200]
        return info
    tip = _git(tree, ["rev-parse", "HEAD"]).strip()
    out = _git(tree, ["rev-list", "--objects", "--missing=print", "--first-parent", tip, "--", ".github/workflows"], check=False)
    missing = [line[1:].split()[0] for line in out.splitlines() if line.startswith("?")]
    if missing:
        _git(tree, ["-c", "fetch.negotiationAlgorithm=noop", "fetch", "--quiet", "origin", "--no-tags", "--no-write-fetch-head",
                    "--recurse-submodules=no", "--filter=blob:none", "--stdin"], data="\n".join(missing) + "\n", check=False)
    info["blobs_fetched"] = len(missing)
    info["commits"] = int(_git(tree, ["rev-list", "--count", "--first-parent", tip]).strip() or 0)
    return info


def since(tree: Path, faults_list: list, deadline: float | None = None) -> dict:
    """For every finding of an audit (a fault hidden at HEAD), the commit it has been hidden since,
    read from the checkout's own history. Returns {"findings": [...], "shallow", "capped",
    "revisions_read", "seconds"}; a finding whose lineage the history cannot place says so."""
    t0 = time.time()
    hidden = [f for f in faults_list if f.get("verdict") in HIDDEN]
    out = {"findings": [], "shallow": False, "capped": False, "revisions_read": 0, "seconds": 0.0}
    if not hidden:
        out["seconds"] = round(time.time() - t0, 1)
        return out
    try:
        tip = _git(tree, ["rev-parse", "HEAD"]).strip()
        commits = mainline(tree, tip)
    except (RuntimeError, subprocess.SubprocessError, OSError) as e:
        out["error"] = f"no git history to read: {str(e)[:120]}"
        out["findings"] = [{"workflow": f["workflow"], "job": f["job"], "index": f["index"], "placed": False, "why": "no history"} for f in hidden]
        return out
    boundary = shallow_boundary(tree)
    out["shallow"] = bool(boundary)
    head_time = int(_git(tree, ["show", "-s", "--format=%ct", tip]).strip() or 0)
    by_path: dict[str, list[dict]] = {}
    for c in commits:
        for ch in c["changed"]:
            path = ch["path"]
            if ch["status"] == "R" and ch.get("from") in by_path:
                by_path[path] = by_path.pop(ch["from"])
            by_path.setdefault(path, []).append({"sha": c["sha"], "time": c["time"], "subject": c["subject"], "body": c["body"],
                                                 "parents": c["parents"], "status": ch["status"], "path": path})   # the path AT this revision
    wanted = {f["workflow"] for f in hidden}
    paths = {p: revs for p, revs in by_path.items() if p.rsplit("/", 1)[-1] in wanted}
    placed: dict[tuple, dict] = {}
    for path, revs_raw in sorted(paths.items()):
        if deadline and time.time() > deadline:
            out["capped"] = True
            break
        wf_name = path.rsplit("/", 1)[-1]
        texts = texts_at(tree, [(r["sha"], r["path"]) for r in revs_raw if r["status"] != "D"])
        revs = []
        for r in revs_raw:
            merged = merged_subjects(tree, r) if len(r["parents"]) > 1 else []
            revs.append({"sha": r["sha"], "time": r["time"], "subject": r["subject"], "body": r["body"], "merged": merged,
                         "text": None if r["status"] == "D" else texts.get((r["sha"], r["path"]))})
        reader = Reader()
        for lin in follow(revs, reader, wf_name):
            if lin.get("alive"):
                placed[(wf_name, lin["job"], lin["index"])] = lin
        out["revisions_read"] += reader.reads
    for f in hidden:
        lin = placed.get((f["workflow"], f["job"], f["index"]))
        row = {"workflow": f["workflow"], "job": f["job"], "index": f["index"], "name": f.get("name"), "verdict": f["verdict"]}
        if lin is None:
            row.update(placed=False, why="capped before this workflow was read" if out["capped"] else "not on the mainline history (uncommitted, or the step could not be followed)")
        else:
            arrivals = [e for e in lin["events"] if e["to"] == "hidden"]
            if arrivals:
                e = arrivals[-1]
                row.update(placed=True, since=e["time"], sha=e["sha"], subject=e["subject"], born_hidden=False, kind=e["kind"],
                           mechanism=e["mechanism"], acknowledged=e.get("acknowledged"))
            else:
                row.update(placed=True, since=lin["born"]["time"], sha=lin["born"]["sha"], subject=lin["born"]["subject"], born_hidden=True,
                           kind="born", mechanism=None, acknowledged=None)
            row.update(revisions=lin["revisions"], at_boundary=bool(lin["born"].get("at_boundary")),
                       age_days=round((head_time - row["since"]) / 86400, 1) if head_time and row.get("since") else None,
                       repaired_before=sum(1 for e in lin["events"] if e["kind"] == "repair"))
        out["findings"].append(row)
    out["seconds"] = round(time.time() - t0, 1)
    return out
