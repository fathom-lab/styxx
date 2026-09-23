# -*- coding: utf-8 -*-
"""SWALLOW-15 -- the escape that isn't: `styxx ci-audit`, confined by Landlock, red-teamed against
its own boundary and shown to read a population unchanged.

SWALLOW-14 found that the audit runs each workflow step's shell on the machine: `actions/setup-node`'s
`rm -rf $RUNNER_TOOL_CACHE/*` becomes `rm -rf /*` in the simulation, and it deleted the run's machine
(RESULT_swallow14 §0). SWALLOW-15 confines the simulation with Linux Landlock (5.13+, no root,
`styxx/ciaudit/confine.py`) and measures three things, on the bare machine -- no overlay, no throwaway
copy; the product's own confinement is what keeps the machine:

  1. THE BATTERY. Hand-written workflow steps that each try to reach the host -- delete, create,
     truncate, rename or symlink outside a scratch directory; a fork bomb; connect a socket out;
     signal the auditor. Each is paired with a canary on the host. Confined, every canary survives
     and the audit still completes; unconfined (a control, run inside a throwaway copy so the control
     itself cannot hurt the machine), the destructive ones fire -- the boundary is what stops them,
     not the harness.

  2. THE POPULATION. The confined audit over SWALLOW-14's 5,945 repositories, each at the tip its
     receipt recorded, compared target by target to that receipt (built unconfined). The confinement
     must change nothing it reads, and the two repositories that deleted their copy in SWALLOW-14
     (`actions/setup-node`, `djylb/nps`) must now leave the bare machine intact. Host canaries,
     checked after every repository, guard the whole run.

  3. THE PERIMETER. What Landlock here does NOT stop, each stated with a probe: a confined step may
     still read what its user can read, change a file's mode/owner/timestamps (Landlock does not
     govern those), and send a UDP datagram. The paper states the boundary's true shape rather than
     overclaiming it.

    python -m benchmarks.harness_mutation.confined --battery --out papers/harness/swallow15_battery.json
    python -m benchmarks.harness_mutation.confined --population papers/harness/swallow14_population.json.gz \\
        --receipt papers/harness/swallow14_receipt.json.gz --work <dir> --out papers/harness/swallow15_receipt.json.gz --workers 3
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import gzip
import json
import os
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from styxx import ciaudit                                # noqa: E402  -- the product, as it ships
from styxx.ciaudit import confine as C                  # noqa: E402
from benchmarks.harness_mutation import empty_list as EL  # noqa: E402  -- fetch, audit_one, hashing, reused

SCHEMA = "styxx.harness-confined/v1"
DEADLINE = 600.0
TIMEOUT = 1800
HARNESS = ROOT / "papers" / "harness"
PRODUCT = ("styxx/ciaudit/engine.py", "styxx/ciaudit/actions.py", "styxx/ciaudit/repair.py", "styxx/ciaudit/repair_structural.py",
           "styxx/ciaudit/repair_frontier.py", "styxx/ciaudit/confine.py", "styxx/ciaudit/__init__.py", "styxx/ciaudit/differential.py")


def _sha(data: bytes) -> str:
    return EL._sha(data)


def instrument_sha256() -> str:
    return _sha(Path(__file__).read_bytes())


def product_sha256() -> dict:
    return {p: _sha((ROOT / p).read_bytes()) for p in PRODUCT}


# ----------------------------------------------------------------------------- the battery
# Each item is a workflow step whose simulated shell, if it reached the host, would touch something
# under CANARY (a directory of the harness's own making) or a marker the harness set up. The
# {canary} placeholder is filled with that directory. `effect(canary)` returns True when the host
# effect happened -- so the confined arm asserts not effect(), and the unconfined control asserts it.

def _has(canary: Path, name: str) -> bool:
    return (canary / name).exists()


def _gone(canary: Path, name: str) -> bool:
    return not (canary / name).exists()


BATTERY = [
    ("rm -rf a scope that came out empty",
     'D="$(discover-dir)"\nrm -rf "$D"{canary}/*\n',                                   # $D empty -> rm -rf {canary}/*
     lambda c: _gone(c, "keep")),
    ("delete a named file outside",
     'rm -f {canary}/keep\n',
     lambda c: _gone(c, "keep")),
    ("delete a directory outside",
     'rm -rf {canary}/sub\n',
     lambda c: not (c / "sub").exists()),
    ("create a file outside",
     'echo pwned > {canary}/created\n',
     lambda c: _has(c, "created")),
    ("truncate a file outside",
     ': > {canary}/keep\n',
     lambda c: (c / "keep").read_text() == ""),
    ("append to a file outside",
     'echo more >> {canary}/keep\n',
     lambda c: "more" in (c / "keep").read_text()),
    ("rename a file outside",
     'mv {canary}/keep {canary}/moved\n',
     lambda c: _has(c, "moved")),
    ("make a directory outside",
     'mkdir -p {canary}/newdir\n',
     lambda c: (c / "newdir").is_dir()),
    ("plant a symlink outside",
     'ln -s /etc/hosts {canary}/link\n',
     lambda c: (c / "link").is_symlink()),
    ("write through a redirect to an absolute path",
     'printf x > {canary}/redir\n',
     lambda c: _has(c, "redir")),
    ("a child process writes outside",
     'bash -c "echo x > {canary}/viachild"\n',
     lambda c: _has(c, "viachild")),
    ("a background job writes outside",
     'touch {canary}/bg &\nwait\n',
     lambda c: _has(c, "bg")),
]

# Two of the battery are checked differently, against a marker rather than a canary file.
NET_ITEM = "connect a socket to a listening port"
SIGNAL_ITEM = "signal the auditor process"


def _one_step_workflow(run: str) -> str:
    return "on: [push]\njobs:\n  j:\n    runs-on: ubuntu-latest\n    steps:\n      - name: Run tests\n        run: |\n" + \
        "".join("          " + ln + "\n" for ln in run.splitlines())


def _fresh_canary(base: Path, tag: str) -> Path:
    c = base / f"canary-{tag}"
    if c.exists():
        shutil.rmtree(c)
    (c / "sub").mkdir(parents=True)
    (c / "keep").write_text("keep")
    (c / "sub" / "inner").write_text("inner")
    return c


def probe_item(name: str, run_tmpl: str, effect, base: Path, confined: bool) -> dict:
    """Build the one-step workflow, run the product's audit (confined or not), report whether the
    host effect happened and whether the audit completed."""
    canary = _fresh_canary(base, f"{'c' if confined else 'u'}-{abs(hash(name)) % 100000}")
    text = _one_step_workflow(run_tmpl.format(canary=canary))
    tree = base / f"wf-{'c' if confined else 'u'}-{abs(hash(name)) % 100000}"
    (tree / ".github" / "workflows").mkdir(parents=True, exist_ok=True)
    (tree / ".github" / "workflows" / "ci.yml").write_text(text, encoding="utf-8")
    subprocess.run(["git", "-C", str(tree), "init", "-q"], check=False)
    subprocess.run(["git", "-C", str(tree), "add", "-A"], check=False)
    subprocess.run(["git", "-C", str(tree), "-c", "user.email=a@b.c", "-c", "user.name=a", "commit", "-qm", "x"], check=False)
    err = None
    try:
        rec = ciaudit.audit(str(tree), confined=confined)
        audited = {"workflows": rec["summary"].get("workflows"), "fault_sites": rec["summary"].get("fault_sites"),
                   "confined": rec["confinement"].get("confined")}
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {str(e)[:200]}"
        audited = None
    fired = bool(effect(canary))
    shutil.rmtree(tree, ignore_errors=True)
    shutil.rmtree(canary, ignore_errors=True)
    return {"item": name, "confined": confined, "fired_on_host": fired, "audited": audited, "error": err}


def probe_net(base: Path, confined: bool) -> dict:
    """A step that connects to a port the harness is listening on. Confined (ABI ≥ 4), the connect is
    refused and nothing is accepted; unconfined, the harness accepts one connection."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    srv.settimeout(4)
    port = srv.getsockname()[1]
    run = (f'exec 3<>/dev/tcp/127.0.0.1/{port} && echo open >&3\n')
    tree = base / f"net-{'c' if confined else 'u'}"
    (tree / ".github" / "workflows").mkdir(parents=True, exist_ok=True)
    (tree / ".github" / "workflows" / "ci.yml").write_text(_one_step_workflow(run), encoding="utf-8")
    for a in (["init", "-q"], ["add", "-A"], ["-c", "user.email=a@b.c", "-c", "user.name=a", "commit", "-qm", "x"]):
        subprocess.run(["git", "-C", str(tree), *a], check=False)
    err = None
    try:
        ciaudit.audit(str(tree), confined=confined)
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {str(e)[:120]}"
    accepted = False
    try:
        conn, _ = srv.accept()
        accepted = True
        conn.close()
    except (socket.timeout, OSError):
        pass
    srv.close()
    shutil.rmtree(tree, ignore_errors=True)
    return {"item": NET_ITEM, "confined": confined, "fired_on_host": accepted, "audited": None if err else {"confined": confined}, "error": err}


def battery(base: Path) -> dict:
    """Every item twice: confined, and unconfined as a control. Each item is scoped to a canary
    directory of the harness's own making (never `/`), so the unconfined control fires safely on the
    bare machine -- it deletes only the harness's canary. Confined, the same write is refused. The two
    real `/`-wipers are the population arm's job (`actions/setup-node`, `djylb/nps`), where the bare
    machine itself is the canary."""
    base.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, tmpl, effect in BATTERY:
        rows.append(probe_item(name, tmpl, effect, base, confined=False))    # control: fires on the canary
        rows.append(probe_item(name, tmpl, effect, base, confined=True))     # confined: must not
    rows.append(probe_net(base, confined=False))
    rows.append(probe_net(base, confined=True))
    perimeter = _perimeter(base)
    confined = [r for r in rows if r["confined"]]
    control = [r for r in rows if not r["confined"]]
    return {"schema": SCHEMA + "#battery", "abi": C.abi(), "items": len(confined),
            "control_fired": sum(1 for r in control if r["fired_on_host"]),
            "confined_neutralised": sum(1 for r in confined if not r["fired_on_host"] and r["error"] is None),
            "confined_audited_ok": sum(1 for r in confined if r["audited"] and r["error"] is None),
            "any_reached_host": [r["item"] for r in confined if r["fired_on_host"]],
            "rows": rows, "perimeter": perimeter}


def _perimeter(base: Path) -> dict:
    """The boundary's shape: three effects Landlock (this ABI) does not stop, each from a confined
    child, against the harness's own canary -- so the paper states what confinement does not cover."""
    canary = _fresh_canary(base, "perim")
    target = canary / "keep"

    def probe():
        import os as _os
        out = {}
        try:
            out["read outside"] = "yes" if target.read_text() == "keep" else "changed"
        except OSError as e:
            out["read outside"] = f"no ({e.errno})"
        try:
            _os.chmod(target, 0o600)
            out["chmod outside"] = "yes"
        except OSError as e:
            out["chmod outside"] = f"no ({e.errno})"
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.sendto(b"x", ("127.0.0.1", 9))       # discard port; nothing listens, but the send is allowed
            s.close()
            out["udp send"] = "yes"
        except OSError as e:
            out["udp send"] = f"no ({e.errno})"
        return out

    try:
        got, _ = C.run(probe)
    except C.Unconfinable:
        got = {"read outside": "n/a", "chmod outside": "n/a", "udp send": "n/a"}
    shutil.rmtree(canary, ignore_errors=True)
    return got


# ----------------------------------------------------------------------------- the population, confined

def _outcome_targets(rec: dict) -> dict:
    """Per target, two readings of an audit_one record. The **core** is what confinement must not
    change and what does not depend on the machine's load: the verdict, the baseline, the stage, the
    script's sha256, and which candidates apply and are loud. `verified_repair` is kept beside it, not
    in the core, because a candidate that backgrounds a command can verify or not with the load
    (SWALLOW-14 G-S14-5); the run compares cores for equivalence and reports how the chosen repair
    moved. Times and the confinement note are left out."""
    out = {}
    for t in rec.get("targets", []):
        key = (t["workflow"], t["job"], t["index"])
        core = {"verdict": t["verdict"], "baseline": t["baseline"], "stage": t["stage"], "tried_stage3": t["tried_stage3"],
                "run_sha256": t.get("run_sha256"),
                "candidates": [(c["repair"], c.get("applies"), c.get("loud")) for c in t["candidates"]]}
        out[key] = {"core": core, "verified_repair": t["verified_repair"]}
    return out


def confined_audit_one(tree: Path, repo: str | None = None) -> tuple[dict, dict]:
    """SWALLOW-14's `audit_one`, run in a Landlock-confined child (the product's own `confine.run`)."""
    return C.run(EL.audit_one, tree, repo)


def one(repo: str, tip: str, work: Path, want: dict) -> dict:
    """Fetch (unconfined -- the network, and only the clone's own directory is written), then audit
    the checkout in a confined child, and compare to the SWALLOW-14 receipt's targets for this repo."""
    t0 = time.time()
    tree, info = EL.fetch(repo, tip, work)
    res = {"repo": repo, "tip": tip, "fetch": info}
    if tree is None:
        time.sleep(5)
        tree, again = EL.fetch(repo, tip, work)
        res["fetch"] = dict(again, first_error=info.get("error"))
        if tree is None:
            res["seconds"] = round(time.time() - t0, 1)
            return res
    try:
        rec, conf = confined_audit_one(tree, repo)
        got = _outcome_targets(rec)
        res.update(confined=conf.get("confined"), abi=conf.get("abi"), workflows=rec.get("workflows"), capped=rec.get("capped"),
                   targets=len(rec.get("targets", [])), seconds_confined=rec.get("seconds_repair"))
        if want is not None:
            core_same = {k: v["core"] for k, v in got.items()} == {k: v["core"] for k, v in want.items()}
            res["matches_receipt"] = core_same                          # the deterministic, confinement-invariant core
            res["verified_repair_moved"] = sorted([list(k) for k in got if k in want and got[k]["verified_repair"] != want[k]["verified_repair"]])
            if not core_same:
                res["diff"] = _first_diff({k: v["core"] for k, v in want.items()}, {k: v["core"] for k, v in got.items()})
    except C.ConfinedError as e:
        res["error"] = f"confined: {str(e).splitlines()[0][:200]}"
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    shutil.rmtree(tree, ignore_errors=True)
    res["seconds"] = round(time.time() - t0, 1)
    return res


def _first_diff(want: dict, got: dict) -> dict:
    wk, gk = set(want), set(got)
    if wk != gk:
        return {"keys_only_in_receipt": [list(k) for k in list(wk - gk)[:3]], "keys_only_in_run": [list(k) for k in list(gk - wk)[:3]]}
    for k in want:
        if want[k] != got[k]:
            return {"target": list(k), "receipt": want[k], "run": got[k]}
    return {}


def _canaries(where: Path) -> list[Path]:
    """A set of files on the machine, outside any scratch directory, that a correct confinement never
    lets a simulated step touch. Checked after every repository."""
    where.mkdir(parents=True, exist_ok=True)
    cs = []
    for i in range(4):
        p = where / f"canary{i}"
        p.write_text(f"canary{i}")
        cs.append(p)
    return cs


def _canaries_intact(cs: list[Path]) -> bool:
    return all(p.exists() and p.read_text() == p.name for p in cs)


def _receipt_targets(receipt: Path) -> dict:
    r, _ = EL._load(receipt)
    out = {}
    for rep in r["repos"]:
        if "targets" in rep:
            out[rep["repo"]] = _outcome_targets(rep)
    return out, _sha(receipt.read_bytes())


def run(population: Path, receipt: Path, work: Path, out: Path, workers: int = 3) -> dict:
    pop, pop_sha = EL._load(population)
    want_by_repo, receipt_sha = _receipt_targets(receipt)
    work.mkdir(parents=True, exist_ok=True)
    canaries = _canaries(work / "canaries")
    res_dir = work / "results"
    res_dir.mkdir(exist_ok=True)
    t0 = time.time()
    results: dict = {}
    breached = []

    def go(r):
        path = res_dir / (r["repo"].replace("/", "__") + ".json")
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        rec = one(r["repo"], r["tip"], work / "clones", want_by_repo.get(r["repo"]))
        path.write_text(json.dumps(rec), encoding="utf-8")
        return rec

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(go, r): r["repo"] for r in pop["repos"]}
        for n, fut in enumerate(cf.as_completed(futs), 1):
            r = fut.result()
            results[futs[fut]] = r
            if not _canaries_intact(canaries):                 # a correct confinement never trips this
                breached.append({"after": r["repo"], "n": n})
                _canaries(work / "canaries")                   # restore, keep measuring
            print(f"[{n}/{len(futs)}] {r['repo']} {r.get('error') or ('match' if r.get('matches_receipt') else 'DIFF' if 'matches_receipt' in r else '')} "
                  f"{r.get('seconds', 0)}s total={time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    repos = [results[r["repo"]] for r in pop["repos"]]
    receipt_out = {"schema": SCHEMA, "instrument": "benchmarks/harness_mutation/confined.py", "instrument_sha256": instrument_sha256(),
                   "product_sha256": product_sha256(), "population_sha256": pop_sha, "compared_receipt_sha256": receipt_sha,
                   "landlock_abi": C.abi(), "canaries_breached": breached, "workers": workers, "seconds": round(time.time() - t0, 1),
                   "repos": repos, "summary": summary(repos, breached)}
    raw = (json.dumps(receipt_out, indent=1, sort_keys=True) + "\n").encode("utf-8")
    out.write_bytes(gzip.compress(raw, mtime=0) if out.suffix == ".gz" else raw)
    return receipt_out


def summary(repos: list[dict], breached: list) -> dict:
    fetched = [r for r in repos if "targets" in r or "matches_receipt" in r]
    audited = [r for r in repos if r.get("confined") is not None]
    compared = [r for r in repos if "matches_receipt" in r]
    match = [r for r in compared if r["matches_receipt"]]
    secs = [r["seconds"] for r in repos if r.get("seconds")]
    moved = [{"repo": r["repo"], "targets": r["verified_repair_moved"]} for r in compared if r.get("verified_repair_moved")]
    return {
        "repos": len(repos), "fetched": len(fetched),
        "fetch_failed": sum(1 for r in repos if r.get("fetch", {}).get("error") and "targets" not in r and "matches_receipt" not in r),
        "errors": sum(1 for r in repos if r.get("error")),
        "confined": sum(1 for r in audited if r.get("confined")), "unconfined": sum(1 for r in audited if r.get("confined") is False),
        "compared_to_receipt": len(compared), "match_receipt": len(match), "differ_from_receipt": len(compared) - len(match),
        "verified_repair_moved_repos": len(moved), "verified_repair_moved": moved[:20],
        "canaries_breached": len(breached), "seconds_median_per_repo": statistics.median(secs) if secs else None,
        "differs": [{"repo": r["repo"], "diff": r.get("diff")} for r in compared if not r["matches_receipt"]][:20],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--battery", action="store_true")
    ap.add_argument("--population", type=Path)
    ap.add_argument("--receipt", type=Path)
    ap.add_argument("--work", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args(argv)
    if a.battery:
        base = Path(a.work or tempfile.mkdtemp(prefix="s15-battery-"))
        rep = battery(base)
        rep = {"schema": SCHEMA + "#battery", "instrument_sha256": instrument_sha256(), "product_sha256": product_sha256(), **rep}
        a.out.write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in rep["summary" if "summary" in rep else "items"].items()} if False else
                         {"items": rep["items"], "confined_neutralised": rep["confined_neutralised"],
                          "confined_audited_ok": rep["confined_audited_ok"], "any_reached_host": rep["any_reached_host"],
                          "perimeter": rep["perimeter"], "abi": rep["abi"]}, indent=1))
        return 0
    receipt = run(a.population.resolve(), a.receipt.resolve(), a.work.resolve(), a.out.resolve(), a.workers)
    print(json.dumps(receipt["summary"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
