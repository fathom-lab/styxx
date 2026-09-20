"""Score the frozen MUTE-1 predictions against the receipt, in the open.

    python papers/harness/mute1_score.py                       # reads papers/harness/mute1_receipt.json
    python papers/harness/mute1_score.py --receipt X.json

The predictions P1-P8 are transcribed from the frozen preregistration; each is a function of the
receipt and nothing else. The gates G-M1-1 and G-M1-2 are applied first, and a receipt that fails
either is reported INVALID and scored anyway, so the reader can see what an invalid run looked like.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREREG_SHA256_AT_FREEZE = "f569701e1a2fc6b71bbff8336d51d413f6e1291d94d3224a1ec525c392df9fbc"


def _find(vs, op, path=None, job=None, step_name=None, name=None):
    out = []
    for v in vs:
        c = v["check"]
        if v["operator"] != op:
            continue
        if path and c["path"] != path:
            continue
        if job and c.get("job") != job:
            continue
        if step_name and c.get("name") != step_name:
            continue
        if name and c.get("name") != name:
            continue
        out.append(v)
    return out


def score(r: dict) -> dict:
    vs = r["verdicts"]
    t = r["totals"]
    gates = {
        "G-M1-1": {"pass": r["oracle"]["tests_passing_on_baseline"] >= 1 and t["KILLED"] >= 1,
                   "detail": f"{r['oracle']['tests_passing_on_baseline']} tests pass on baseline; {t['KILLED']} KILLED"},
        "G-M1-2": {"pass": t["UNREACHED"] <= 5, "detail": f"UNREACHED = {t['UNREACHED']}"},
        "G-M1-3": {"pass": all(v["verdict"] in ("KILLED", "SURVIVED", "UNREACHED") for v in vs)
                            and all((v["verdict"] == "KILLED") == bool(v["killed_by"]) for v in vs),
                   "detail": "every KILLED names a test and every SURVIVED names none"},
    }
    P = {}

    def one(op, **kw):
        m = _find(vs, op, **kw)
        return m[0]["verdict"] if len(m) == 1 else f"ambiguous({len(m)})"

    # P1
    p1 = {
        "M-JOB typecheck-js": one("M-JOB", path=".github/workflows/test.yml", job="typecheck-js"),
        "M-SUBJECT .gitignore line": one("M-SUBJECT", path=".gitignore"),
        "M-SUBJECT py_side PINNED": one("M-SUBJECT", path="web/gate/differential/py_side.py"),
        "M-STEP gauntlet discover": one("M-STEP", path=".github/workflows/gauntlet-pr.yml", step_name="Discover changed submissions"),
        "M-SWALLOW gauntlet discover": one("M-SWALLOW", path=".github/workflows/gauntlet-pr.yml", step_name="Discover changed submissions"),
    }
    P["P1"] = {"hit": all(v == "KILLED" for v in p1.values()), "observed": p1, "predicted": "all KILLED"}
    # P2
    jobs = _find(vs, "M-JOB")
    killed_jobs = sorted(f"{v['check']['path']}:{v['check']['job']}" for v in jobs if v["verdict"] == "KILLED")
    P["P2"] = {"hit": killed_jobs == [".github/workflows/gauntlet-pr.yml:verify-submissions",
                                      ".github/workflows/test.yml:typecheck-js"],
               "observed": {"killed": killed_jobs, "survived": len(jobs) - len(killed_jobs)},
               "predicted": "exactly 2 KILLED: gauntlet-pr verify-submissions, test.yml typecheck-js; 12 SURVIVED"}
    # P3
    p3 = {"M-STEP Run tests": one("M-STEP", path=".github/workflows/test.yml", job="test", step_name="Run tests"),
          "M-SWALLOW Run tests": one("M-SWALLOW", path=".github/workflows/test.yml", job="test", step_name="Run tests")}
    P["P3"] = {"hit": all(v == "SURVIVED" for v in p3.values()), "observed": p3, "predicted": "both SURVIVED"}
    # P4
    trig = _find(vs, "M-TRIGGER")
    killed_trig = sorted(v["check"]["path"] for v in trig if v["verdict"] == "KILLED")
    P["P4"] = {"hit": killed_trig == [".github/workflows/leaderboard-submission.yml"],
               "observed": {"killed": killed_trig, "survived": len(trig) - len(killed_trig)},
               "predicted": "8 SURVIVED, leaderboard-submission KILLED"}
    # P5
    g = _find(vs, "M-GUARD")
    P["P5"] = {"hit": all(v["verdict"] == "SURVIVED" for v in g),
               "observed": {v["mutant"]: v["verdict"] for v in g}, "predicted": "all 8 SURVIVED"}
    # P6
    sc = _find(vs, "M-SCRIPT")
    obs6 = {v["check"]["name"]: v["verdict"] for v in sc}
    P["P6"] = {"hit": obs6.get("typecheck") == "KILLED" and all(obs6[k] == "SURVIVED" for k in obs6 if k != "typecheck"),
               "observed": obs6, "predicted": "typecheck KILLED; build, test, test:watch SURVIVED"}
    # P7
    p7 = {v["check"]["name"]: v["verdict"] for v in _find(vs, "M-SWALLOW", path=".github/workflows/test.yml", job="typecheck-js")}
    P["P7"] = {"hit": len(p7) == 2 and all(x == "SURVIVED" for x in p7.values()), "observed": p7,
               "predicted": "both typecheck-js run steps SURVIVE a swallow"}
    # P8
    P["P8"] = {"hit": t == {"KILLED": 17, "SURVIVED": 102, "UNREACHED": 1}, "observed": t,
               "predicted": {"KILLED": 17, "SURVIVED": 102, "UNREACHED": 1},
               "direction_hit": t["SURVIVED"] > 0.8 * (t["KILLED"] + t["SURVIVED"] + t["UNREACHED"])}
    valid = gates["G-M1-1"]["pass"] and gates["G-M1-2"]["pass"]
    return {"valid": valid, "gates": gates, "predictions": P,
            "hits": sum(1 for p in P.values() if p["hit"]), "of": len(P)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipt", default=str(HERE / "mute1_receipt.json"))
    ap.add_argument("--out", default=str(HERE / "mute1_scored.json"))
    a = ap.parse_args(argv)
    r = json.loads(Path(a.receipt).read_text(encoding="utf-8"))
    s = score(r)
    s["receipt"] = Path(a.receipt).name
    s["receipt_tree"] = r["tree"]
    s["harness_fingerprint"] = r["harness_fingerprint"]
    s["prereg_sha256_at_freeze"] = PREREG_SHA256_AT_FREEZE
    Path(a.out).write_text(json.dumps(s, indent=2), encoding="utf-8")
    print("VALID" if s["valid"] else "INVALID", "| gates:", {k: v["pass"] for k, v in s["gates"].items()})
    for k, p in s["predictions"].items():
        print(f"  {k}: {'HIT ' if p['hit'] else 'MISS'}  predicted={p['predicted']}  observed={json.dumps(p['observed'])[:160]}")
    print(f"{s['hits']} of {s['of']} predictions HIT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
