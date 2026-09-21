"""Score the SWALLOW-6 preregistration against the natural-history receipt. Every prediction and gate is mechanical.

    python papers/harness/swallow6_score.py     # reads swallow6_receipt.json(.gz); writes swallow6_scored.json
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
PREREG = HERE / "PREREG_swallow6_where_hidden_checks_come_from_2026_09_21.md"
PREREG_SHA256_FROZEN = "d19eff1fc5a018b68d1d5d46984f67f3731b437fefb3f616d638043e94233725"
# amended after the freeze (stated in the RESULT): the prereg names d92b1b32…; the run is at the sha below (run 3): one workflow's texts fetched at a time, and the reading at the latest revision kept as it is beside the last interpretable one
INSTRUMENT_SHA256_FROZEN = "4b961880c88c6f288b1df694df3cc8225f269cc70169cc83a8dcd18b8295c3d7"
FAULTS_SHA256_FROZEN = "d26a407ca276d1b1c218fd32bc6bf853544fc394ce5ebc98b2b6ae4bd0fa3543"
ACTIONS_SHA256_FROZEN = "0e723694d459ca2368799e3fc21a26d06e70fb89ad09bd2b0152466bf2a68f72"
REPAIR_SHA256_FROZEN = "7b9a1695d316c2ce109495cf60c5a9a1de03bac204e9bf48fdbc2a1ac66012b9"
STRUCTURAL_SHA256_FROZEN = "77067a71fa41e48089b4c3e68fae17f02322fd892fc71d604d83f23cb693982c"
SOURCE_SHA256_FROZEN = "609e664549df085bf8f3c11243ce781d5b0c107009893afbd14ca9fea50ad21e"
RECEIPT = HERE / "swallow6_receipt.json"
RECEIPT_GZ = HERE / "swallow6_receipt.json.gz"
SOURCE = HERE / "swallow3_receipt.json.gz"
OUT = HERE / "swallow6_scored.json"
HIDDEN = ("SWALLOWED", "FAIL_OPEN")


def load_receipt() -> tuple[dict, str]:
    raw = RECEIPT.read_bytes() if RECEIPT.exists() else gzip.decompress(RECEIPT_GZ.read_bytes())
    return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


def _count(it) -> dict:
    out: dict = {}
    for x in it:
        out[x] = out.get(x, 0) + 1
    return out


def _day(t: int | None) -> str | None:
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%d") if t else None


def _pct(a: int, b: int) -> float | None:
    return round(100.0 * a / b, 1) if b else None


def main() -> int:
    if hashlib.sha256(PREREG.read_bytes()).hexdigest() != PREREG_SHA256_FROZEN:
        print("INVALID__PREREG_MOVED")
        return 2
    r, receipt_sha = load_receipt()
    src = json.loads(gzip.decompress(SOURCE.read_bytes()).decode("utf-8"))
    gates, P = {}, {}

    t = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(ROOT / "tests" / "test_harness_history.py")],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(ROOT), timeout=1800)
    gates["G-S6-1"] = {"pass": t.returncode == 0, "detail": (t.stdout.strip().splitlines() or [""])[-1][:120]}

    read = [h for h in r["repos"] if not h.get("capped")]
    capped = [h["repo"] for h in r["repos"] if h.get("capped")]
    failed = [f.get("repo") for f in r.get("clone_failures", [])]
    gates["G-S6-2"] = {"pass": len(read) >= 90, "detail": f"{len(read)} of {r.get('population_size')} read to their pinned HEAD uncapped; capped: {len(capped)}; failed: {len(failed)}",
                       "capped": capped, "failed": failed}

    # G-S6-3: HEAD agrees with the SWALLOW-3 receipt on the hand-written faults it can match
    at_head = {}
    for h in read:
        for wf, w in h["workflows"].items():
            for lin in w["lineages"]:
                if lin.get("alive"):
                    at_head[(h["repo"], wf, lin["job"], lin["index"])] = lin["verdict_last"]
    matched = same = 0
    differ = []
    for x in src["repos"]:
        for f in x.get("faults", []):
            if f.get("generated"):
                continue
            k = (x["repo"], f["workflow"], f["job"], f["index"])
            if k in at_head:
                matched += 1
                if at_head[k] == f["verdict"]:
                    same += 1
                else:
                    differ.append((*k, f["verdict"], at_head[k]))
    gates["G-S6-3"] = {"pass": matched > 0 and same / matched >= 0.99, "detail": f"{same} of {matched} matched hand-written faults carry the same verdict at HEAD ({_pct(same, matched)}%)",
                       "differ": differ[:40], "differ_count": len(differ)}

    src_sha = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    gates["G-S6-4"] = {"pass": r.get("instrument_sha256") == INSTRUMENT_SHA256_FROZEN and r.get("faults_sha256") == FAULTS_SHA256_FROZEN
                       and r.get("action_checks_sha256") == ACTIONS_SHA256_FROZEN and r.get("repair_sha256") == REPAIR_SHA256_FROZEN
                       and r.get("repair_structural_sha256") == STRUCTURAL_SHA256_FROZEN and r.get("source_receipt_sha256") == SOURCE_SHA256_FROZEN
                       and src_sha == SOURCE_SHA256_FROZEN,
                       "detail": f"history {str(r.get('instrument_sha256'))[:16]}…, faults {str(r.get('faults_sha256'))[:16]}…, action_checks {str(r.get('action_checks_sha256'))[:16]}…, "
                                 f"repair {str(r.get('repair_sha256'))[:16]}…, structural {str(r.get('repair_structural_sha256'))[:16]}…, source {str(r.get('source_receipt_sha256'))[:16]}… (file {src_sha[:16]}…)"}

    # the population of lineages: hand-written, in repositories read uncapped
    lins = [(h["repo"], wf, lin, h.get("head_time") or 0) for h in read for wf, w in h["workflows"].items() for lin in w["lineages"]]
    alive_hidden = [(repo, wf, l, ht) for repo, wf, l, ht in lins if l.get("alive") and l["state_last"] == "hidden"]
    born_hidden = [(repo, wf, l, ht) for repo, wf, l, ht in alive_hidden if l["born"]["state"] == "hidden" and not any(e["kind"] == "repair" for e in l["events"])]
    acquired = [(repo, wf, l, ht) for repo, wf, l, ht in alive_hidden if any(e["kind"] == "acquisition" for e in l["events"])]
    acq = [(repo, wf, l, e) for repo, wf, l, _ in lins for e in l["events"] if e["kind"] == "acquisition"]
    rep = [(repo, wf, l, e) for repo, wf, l, _ in lins for e in l["events"] if e["kind"] == "repair"]
    ever_hidden = [(repo, wf, l) for repo, wf, l, _ in lins if l["born"]["state"] == "hidden" or any(e["to"] == "hidden" for e in l["events"])]
    repaired_loud = [(repo, wf, l) for repo, wf, l in ever_hidden if l.get("alive") and l["state_last"] == "loud" and any(e["kind"] == "repair" for e in l["events"])]
    died_hidden = [(repo, wf, l) for repo, wf, l in ever_hidden if not l.get("alive") and l["state_last"] == "hidden"]

    def last_hidden_time(l):
        arrivals = [e["time"] for e in l["events"] if e["to"] == "hidden"]
        if arrivals:
            return arrivals[-1]
        return l["born"]["time"] if l["born"]["state"] == "hidden" else None

    ages = sorted((ht - last_hidden_time(l)) / 86400 for _, _, l, ht in alive_hidden if last_hidden_time(l) is not None)
    median_age = ages[len(ages) // 2] if ages else None

    P["P1"] = {"hit": bool(alive_hidden) and len(born_hidden) / len(alive_hidden) >= 0.60, "predicted": "≥ 60% of the hidden checks alive at HEAD were born hidden and never loud",
               "observed": {"alive_hidden": len(alive_hidden), "born_hidden_never_loud": len(born_hidden), "acquired": len(acquired), "pct": _pct(len(born_hidden), len(alive_hidden)),
                            "other": len(alive_hidden) - len(born_hidden) - len(acquired)}}
    acked = [x for x in acq if x[3].get("acknowledged")]
    P["P2"] = {"hit": bool(acq) and len(acked) / len(acq) >= 0.50, "predicted": "≥ 50% of acquisitions acknowledged in the commit message",
               "observed": {"acquisitions": len(acq), "acknowledged": len(acked), "pct": _pct(len(acked), len(acq)), "words": _count(x[3]["acknowledged"] for x in acked)}}
    P["P3"] = {"hit": bool(ever_hidden) and len(repaired_loud) / len(ever_hidden) <= 0.25, "predicted": "≤ 25% of ever-hidden lineages are alive, loud and repaired at HEAD",
               "observed": {"ever_hidden": len(ever_hidden), "repaired_and_loud": len(repaired_loud), "pct": _pct(len(repaired_loud), len(ever_hidden)),
                            "alive_hidden": len(alive_hidden), "died_hidden": len(died_hidden)}}
    P["P4"] = {"hit": median_age is not None and median_age >= 180, "predicted": "median age of the hidden checks alive at HEAD ≥ 180 days",
               "observed": {"n": len(ages), "median_days": round(median_age, 1) if median_age is not None else None,
                            "quartiles": [round(ages[len(ages) * q // 4], 1) for q in (1, 2, 3)] if ages else None,
                            "over_180": sum(1 for a in ages if a >= 180), "over_365": sum(1 for a in ages if a >= 365)}}
    prim = _count(x[3]["mechanism"][0] for x in acq)
    two = prim.get("continue-on-error", 0) + prim.get("or-true", 0)
    P["P5"] = {"hit": bool(acq) and two / len(acq) >= 0.50, "predicted": "continue-on-error + or-true are the primary mechanism of ≥ 50% of acquisitions",
               "observed": {"acquisitions": len(acq), "the_two": two, "pct": _pct(two, len(acq)), "by_mechanism": prim}}
    readable = [x for x in rep if isinstance(x[3].get("agreement"), dict) and "error" not in x[3]["agreement"]]
    agree = [x for x in readable if x[3]["agreement"].get("agrees")]
    P["P6"] = {"hit": bool(readable) and len(agree) / len(readable) >= 0.40, "predicted": "≥ 40% of readable wild repairs agree with the instrument's verified candidate",
               "observed": {"repairs": len(rep), "readable": len(readable), "agree": len(agree), "pct": _pct(len(agree), len(readable)),
                            "verified_any": sum(1 for x in readable if x[3]["agreement"].get("verified_repair")),
                            "by_candidate": _count(x[3]["agreement"].get("verified_repair") for x in readable),
                            "wild_mechanisms": _count(x[3]["mechanism"][0] for x in rep)}}
    P["P7"] = {"hit": len(acq) > len(rep), "predicted": "acquisitions strictly outnumber repairs", "observed": {"acquisitions": len(acq), "repairs": len(rep)}}

    valid = all(gates[g]["pass"] for g in ("G-S6-1", "G-S6-2", "G-S6-3", "G-S6-4"))
    hits = sum(1 for p in P.values() if p["hit"])

    def ev_row(repo, wf, l, e):
        ag = e.get("agreement") or {}
        return {"repo": repo, "workflow": wf, "job": l["job"], "step": l["key"], "kind": e["kind"], "sha": e["sha"][:12], "date": _day(e["time"]),
                "subject": e["subject"][:100], "mechanism": e["mechanism"], "acknowledged": e.get("acknowledged"),
                "verdicts": f"{e['verdict_from']} -> {e['verdict_to']}",
                **({"instrument": ag.get("verified_repair"), "stage": ag.get("stage"), "agrees": ag.get("agrees")} if e["kind"] == "repair" else {})}

    payload = {"valid": valid, "gates": gates, "predictions": P, "hits": hits, "of": len(P),
               "population": {"repos_read": len(read), "lineages": len(lins), "alive_hidden": len(alive_hidden), "ever_hidden": len(ever_hidden),
                              "births_at_boundary": sum(1 for _, _, l, _ in lins if l["born"].get("at_boundary")),
                              "revisions_read": r.get("summary", {}).get("revisions_read")},
               "alive_hidden": [{"repo": repo, "workflow": wf, "job": l["job"], "step": l["key"], "verdict": l["verdict_last"], "born": l["born"]["state"],
                                 "born_date": _day(l["born"]["time"]), "hidden_since": _day(last_hidden_time(l)),
                                 "age_days": round((ht - last_hidden_time(l)) / 86400, 1) if last_hidden_time(l) else None,
                                 "revisions": l["revisions"], "events": [(e["kind"], e["mechanism"][0], e["acknowledged"], _day(e["time"])) for e in l["events"]],
                                 "at_boundary": bool(l["born"].get("at_boundary"))}
                               for repo, wf, l, ht in sorted(alive_hidden, key=lambda x: (x[0], x[1], x[2]["job"]))],
               "acquisitions": [ev_row(*x) for x in sorted(acq, key=lambda x: x[3]["time"])],
               "repairs": [ev_row(*x) for x in sorted(rep, key=lambda x: x[3]["time"])],
               "died_hidden": [{"repo": repo, "workflow": wf, "job": l["job"], "step": l["key"], "born": l["born"]["state"], "death": l["death"]["why"],
                                "date": _day(l["death"]["time"]), "subject": l["death"]["subject"][:80]} for repo, wf, l in died_hidden],
               "summary": r.get("summary"), "seconds": r.get("seconds"),
               "instrument_sha256": r.get("instrument_sha256"), "receipt_sha256": receipt_sha,
               "receipt_file": RECEIPT.name if RECEIPT.exists() else RECEIPT_GZ.name, "prereg_sha256_at_freeze": PREREG_SHA256_FROZEN}
    OUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(("VALID" if valid else "INVALID") + f"  {hits}/{len(P)} HIT")
    for k, p in P.items():
        print(f"  {k}: {'HIT' if p['hit'] else 'MISS'}  {json.dumps(p.get('observed'))[:230]}")
    for g, d in gates.items():
        print(f"  {g}: {'pass' if d['pass'] else 'FAIL'} - {d['detail'][:240]}")
    return 0 if valid else 1


if __name__ == "__main__":
    sys.exit(main())
