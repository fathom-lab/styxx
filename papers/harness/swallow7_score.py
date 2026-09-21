"""Score the SWALLOW-7 preregistration against the differential-audit receipt. Every prediction and gate is mechanical.

    python papers/harness/swallow7_score.py [CLONES]   # reads swallow7_receipt.json(.gz); writes swallow7_scored.json; with the
                                                        # SWALLOW-6 clones, also resolves the frozen join's misses through renames (post hoc)
"""
from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREREG = HERE / "PREREG_swallow7_the_differential_audit_2026_09_21.md"
PREREG_SHA256_FROZEN = "a8f4f675ced0ad319c9866eab5881bd023f22901ee28a432bd823fa20e27aba8"
INSTRUMENT_SHA256_FROZEN = "91e4a4a755b80027ed2e3e8c96d72be299ec44e2afce5ca35b402eecdba11c9e"
# the prereg names history.py at 4b961880… and the SWALLOW-6 receipt at 703583d3…; run 1 (INVALID on G-S7-3) exposed a defect in
# that history.py, which was amended (SWALLOW-6 runs 4 and 5) -- run 2 is scored against the amended history and its receipt, stated in the RESULT
HISTORY_SHA256_FROZEN = "93efb4a947a18e61d4457af12c17bf266ca5680ec89156da6935035211f93d1c"
FAULTS_SHA256_FROZEN = "d26a407ca276d1b1c218fd32bc6bf853544fc394ce5ebc98b2b6ae4bd0fa3543"
ACTIONS_SHA256_FROZEN = "0e723694d459ca2368799e3fc21a26d06e70fb89ad09bd2b0152466bf2a68f72"
REPAIR_SHA256_FROZEN = "7b9a1695d316c2ce109495cf60c5a9a1de03bac204e9bf48fdbc2a1ac66012b9"
STRUCTURAL_SHA256_FROZEN = "77067a71fa41e48089b4c3e68fae17f02322fd892fc71d604d83f23cb693982c"
SOURCE_SHA256_FROZEN = "4c6920b4aa3d7cded78328b59c3925a5239c8d9876c7a9fce2a18fc5533fc73b"
RECEIPT = HERE / "swallow7_receipt.json"
RECEIPT_GZ = HERE / "swallow7_receipt.json.gz"
SOURCE = HERE / "swallow6_receipt.json.gz"
OUT = HERE / "swallow7_scored.json"


def load_receipt() -> tuple[dict, str]:
    raw = RECEIPT.read_bytes() if RECEIPT.exists() else gzip.decompress(RECEIPT_GZ.read_bytes())
    return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


def _count(it) -> dict:
    out: dict = {}
    for x in it:
        out[x] = out.get(x, 0) + 1
    return out


def _pct(a: int, b: int) -> float | None:
    return round(100.0 * a / b, 1) if b else None


def _day(t: int | None) -> str | None:
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%d") if t else None


def main() -> int:
    if hashlib.sha256(PREREG.read_bytes()).hexdigest() != PREREG_SHA256_FROZEN:
        print("INVALID__PREREG_MOVED")
        return 2
    r, receipt_sha = load_receipt()
    src = json.loads(gzip.decompress(SOURCE.read_bytes()).decode("utf-8"))
    gates, P = {}, {}

    t = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(ROOT / "tests" / "test_harness_differential.py")],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(ROOT), timeout=1800)
    gates["G-S7-1"] = {"pass": t.returncode == 0, "detail": (t.stdout.strip().splitlines() or [""])[-1][:120]}

    read = [x for x in r["repos"] if not x.get("capped")]
    gates["G-S7-2"] = {"pass": len(read) >= 90, "detail": f"{len(read)} of {r.get('population_size')} read uncapped; capped: {sum(1 for x in r['repos'] if x.get('capped'))}; "
                                                        f"failed: {len(r.get('clone_failures', []))}", "failed": [f.get('repo') for f in r.get("clone_failures", [])]}

    # the gate's records: firings and hidden-after-unread, keyed (repo, workflow, job, step key, sha)
    commits = [(x["repo"], c) for x in read for c in x["commits"]]
    nonroot = [(repo, c) for repo, c in commits if not c.get("root")]
    firings: dict = {}
    after_unread: set = set()
    fired_at: dict = {}
    new_checks = []
    for repo, c in nonroot:
        fired_at[(repo, c["sha"])] = c["fires"]
        for d in c.get("detail", []):
            for x in d.get("new_hidden", []):
                firings[(repo, d["workflow"], x["job"], x["step"], c["sha"])] = x
                new_checks.append((repo, d["workflow"], c, x))
            for x in d.get("hidden_after_unread", []):
                after_unread.add((repo, d["workflow"], x["job"], x["step"], c["sha"]))

    # SWALLOW-6's arrivals in hidden, at non-boundary commits; a lineage's key is resolved to what it was AT that
    # revision (the receipt keeps the lineage under its last name, with every rename and its revision index)
    def key_at(lin, revision):
        rn = lin.get("renames") or []
        key = rn[0]["from"] if rn else lin["key"]
        for r_ in rn:
            if r_.get("revision", 10**9) <= revision:
                key = r_["to"]
        return key

    roots = {(repo, c["sha"]) for repo, c in commits if c.get("root")}
    arrivals, via_unread_birth, at_roots = [], 0, []
    for h in src["repos"]:
        for wf, w in h["workflows"].items():
            for lin in w["lineages"]:
                b = lin["born"]
                if b["state"] == "hidden" and not b.get("at_boundary"):
                    if b.get("first_read"):
                        via_unread_birth += 1
                        arrivals.append((h["repo"], wf, lin["job"], key_at(lin, b["first_read"]["revision"]), b["first_read"]["sha"], "born unread, read hidden"))
                    else:
                        arrivals.append((h["repo"], wf, lin["job"], key_at(lin, b["revision"]), b["sha"], "born hidden"))
                for ev in lin["events"]:
                    if ev["to"] == "hidden":
                        arrivals.append((h["repo"], wf, lin["job"], key_at(lin, ev["revision"]), ev["sha"], ev["kind"]))
    read_repos = {x["repo"] for x in read}
    arrivals = [a for a in arrivals if a[0] in read_repos]
    at_roots = [a for a in arrivals if (a[0], a[4]) in roots]                 # a root has no base to compare against: excluded, as every count is
    arrivals = [a for a in arrivals if (a[0], a[4]) not in roots]
    agree = [a for a in arrivals if a[:5] in firings]
    via_unread = [a for a in arrivals if a[:5] not in firings and a[:5] in after_unread]
    missed = [a for a in arrivals if a[:5] not in firings and a[:5] not in after_unread]
    arrival_commits = {(a[0], a[4]) for a in arrivals}
    quiet = [(repo, c) for repo, c in nonroot if (repo, c["sha"]) not in arrival_commits]
    false_fires = [(repo, c) for repo, c in quiet if c["fires"]]
    recall = (len(agree) + len(via_unread)) / len(arrivals) if arrivals else 0
    ff = len(false_fires) / len(quiet) if quiet else 0
    # post hoc, not gating: a lineage the receipt keeps under a workflow's LAST name, fired on by the gate under the
    # name the workflow had at that commit -- resolved through the file's rename history when the clones are at hand
    post_hoc = None
    clones = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if clones is not None and clones.exists() and missed:
        import re
        r7 = {(x["repo"], c["sha"]): c for x in read for c in x["commits"]}
        resolved = 0
        for a in missed:
            repo, wf, job, key, sha, _kind = a
            hits = [d["workflow"] for d in r7[(repo, sha)].get("detail", []) for x in d["new_hidden"] if x["job"] == job and x["step"] == key]
            clone = clones / repo.replace("/", "__")
            log = subprocess.run(["git", "-C", str(clone), "log", "--first-parent", "--follow", "--name-status", "--format=%H", "-M", "--", f".github/workflows/{wf}"],
                                 capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
            cur, names_at = None, {}
            for line in log.splitlines():
                if re.fullmatch(r"[0-9a-f]{40}", line.strip()):
                    cur = line.strip()
                    names_at.setdefault(cur, [])
                elif line.strip() and cur:
                    names_at[cur].append(line.split("\t")[-1].rsplit("/", 1)[-1])
            resolved += any(name in names_at.get(sha, []) for name in hits)
        post_hoc = {"resolved_through_renames": resolved, "of_missed": len(missed),
                    "recall_if_resolved": _pct(len(agree) + len(via_unread) + resolved, len(arrivals))}
    gates["G-S7-3"] = {"pass": recall >= 0.99 and ff <= 0.005, "post_hoc": post_hoc,
                       "detail": f"(a) {len(agree)} of {len(arrivals)} arrivals are firings, {len(via_unread)} more are hidden-after-unread at the same commit "
                                 f"({_pct(len(agree) + len(via_unread), len(arrivals))}%), {len(missed)} missed; (b) {len(false_fires)} of {len(quiet)} quiet commits fire ({_pct(len(false_fires), len(quiet))}%)",
                       "missed": [list(a) for a in missed[:40]], "false_fires": [(repo, c["sha"][:12], c["subject"][:80], c["new_hidden"]) for repo, c in false_fires[:40]],
                       "via_unread_birth": via_unread_birth, "arrivals_at_roots_excluded": len(at_roots)}

    src_sha = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    gates["G-S7-4"] = {"pass": r.get("instrument_sha256") == INSTRUMENT_SHA256_FROZEN and r.get("history_sha256") == HISTORY_SHA256_FROZEN
                       and r.get("faults_sha256") == FAULTS_SHA256_FROZEN and r.get("action_checks_sha256") == ACTIONS_SHA256_FROZEN
                       and r.get("repair_sha256") == REPAIR_SHA256_FROZEN and r.get("repair_structural_sha256") == STRUCTURAL_SHA256_FROZEN
                       and r.get("source_receipt_sha256") == SOURCE_SHA256_FROZEN and src_sha == SOURCE_SHA256_FROZEN,
                       "detail": f"differential {str(r.get('instrument_sha256'))[:16]}…, history {str(r.get('history_sha256'))[:16]}…, faults {str(r.get('faults_sha256'))[:16]}…, "
                                 f"action_checks {str(r.get('action_checks_sha256'))[:16]}…, repair {str(r.get('repair_sha256'))[:16]}…, structural {str(r.get('repair_structural_sha256'))[:16]}…, "
                                 f"source {str(r.get('source_receipt_sha256'))[:16]}… (file {src_sha[:16]}…)"}

    firing_commits = [(repo, c) for repo, c in nonroot if c["fires"]]
    P["P1"] = {"hit": bool(nonroot) and len(firing_commits) / len(nonroot) <= 0.01, "predicted": "the gate fires on ≤ 1.0% of mainline commits touching a hand-written workflow",
               "observed": {"commits": len(nonroot), "firing": len(firing_commits), "pct": _pct(len(firing_commits), len(nonroot)), "roots_excluded": len(commits) - len(nonroot)}}
    fixed = [x for *_, x in new_checks if (x.get("fix") or {}).get("verified_repair")]
    P["P2"] = {"hit": bool(new_checks) and len(fixed) / len(new_checks) >= 0.70, "predicted": "≥ 70% of newly hidden checks have a verified repair on that revision's text",
               "observed": {"newly_hidden": len(new_checks), "with_verified_repair": len(fixed), "pct": _pct(len(fixed), len(new_checks)),
                            "by_repair": _count(x["fix"]["verified_repair"] for x in fixed), "by_stage": _count(x["fix"]["stage"] for x in fixed)}}
    sizes = sorted(x["fix"]["lines_changed"] for x in fixed)
    med = sizes[len(sizes) // 2] if sizes else None
    P["P3"] = {"hit": med is not None and med <= 2, "predicted": "median lines_changed of the verified repairs ≤ 2",
               "observed": {"n": len(sizes), "median": med, "by_size": _count(sizes)}}
    born = [x for *_, x in new_checks if x["kind"] == "born hidden"]
    P["P4"] = {"hit": bool(new_checks) and len(born) / len(new_checks) >= 0.70, "predicted": "≥ 70% of newly hidden checks are born hidden",
               "observed": {"newly_hidden": len(new_checks), "born_hidden": len(born), "pct": _pct(len(born), len(new_checks)), "by_kind": _count(x["kind"] for *_, x in new_checks)}}
    sample = sorted(s["seconds"] for x in read for s in x.get("sample", []))
    s_med = sample[len(sample) // 2] if sample else None
    s_p90 = sample[int(len(sample) * 0.9)] if sample else None
    P["P5"] = {"hit": s_med is not None and s_med <= 3 and s_p90 <= 15, "predicted": "sample median ≤ 3 s and p90 ≤ 15 s",
               "observed": {"n": len(sample), "median_s": s_med, "p90_s": s_p90, "max_s": sample[-1] if sample else None,
                            "reads_median": sorted(s["reads"] for x in read for s in x.get("sample", []))[len(sample) // 2] if sample else None}}
    batches = [(repo, c) for repo, c in firing_commits if c["new_hidden"] >= 3]
    P["P6"] = {"hit": len(batches) >= 4, "predicted": "≥ 4 firing commits bring ≥ 3 newly hidden checks at once",
               "observed": {"batches": len(batches), "which": [(repo, c["sha"][:10], _day(c["time"]), c["new_hidden"], c["subject"][:70]) for repo, c in sorted(batches, key=lambda rc: -rc[1]["new_hidden"])]}}
    coe = [x for *_, x in new_checks if x.get("continue_on_error")]
    P["P7"] = {"hit": bool(new_checks) and len(coe) / len(new_checks) >= 0.50, "predicted": "≥ 50% of newly hidden checks carry continue-on-error at the firing revision",
               "observed": {"newly_hidden": len(new_checks), "continue_on_error": len(coe), "pct": _pct(len(coe), len(new_checks)), "by_verdict": _count(x["verdict"] for *_, x in new_checks)}}

    valid = all(gates[g]["pass"] for g in ("G-S7-1", "G-S7-2", "G-S7-3", "G-S7-4"))
    hits = sum(1 for p in P.values() if p["hit"])
    payload = {"valid": valid, "gates": gates, "predictions": P, "hits": hits, "of": len(P),
               "population": {"repos_read": len(read), "commits": len(commits), "nonroot_commits": len(nonroot), "firing_commits": len(firing_commits),
                              "newly_hidden": len(new_checks), "removed_hidden": sum(c["removed_hidden"] for _, c in nonroot),
                              "hidden_after_unread": sum(c["hidden_after_unread"] for _, c in nonroot), "sample_n": len(sample)},
               "firings": [{"repo": repo, "sha": c["sha"][:12], "date": _day(c["time"]), "subject": c["subject"][:100], "workflow": wf, "job": x["job"], "step": x["step"],
                            "kind": x["kind"], "verdict": x["verdict"], "mechanism": x.get("mechanism"), "continue_on_error": x.get("continue_on_error"),
                            "repair": (x.get("fix") or {}).get("verified_repair"), "lines": (x.get("fix") or {}).get("lines_changed"),
                            "why_not": (x.get("fix") or {}).get("why_not")}
                           for repo, wf, c, x in sorted(new_checks, key=lambda t: (t[2]["time"], t[0]))],
               "summary": r.get("summary"), "seconds": r.get("seconds"), "instrument_sha256": r.get("instrument_sha256"), "receipt_sha256": receipt_sha,
               "receipt_file": RECEIPT.name if RECEIPT.exists() else RECEIPT_GZ.name, "prereg_sha256_at_freeze": PREREG_SHA256_FROZEN}
    OUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(("VALID" if valid else "INVALID") + f"  {hits}/{len(P)} HIT")
    for k, p in P.items():
        print(f"  {k}: {'HIT' if p['hit'] else 'MISS'}  {json.dumps(p.get('observed'))[:230]}")
    for g, d in gates.items():
        print(f"  {g}: {'pass' if d['pass'] else 'FAIL'} - {d['detail'][:260]}")
    if gates["G-S7-3"].get("post_hoc"):
        print(f"  G-S7-3 post hoc (not gating): {json.dumps(gates['G-S7-3']['post_hoc'])}")
    return 0 if valid else 1


if __name__ == "__main__":
    sys.exit(main())
