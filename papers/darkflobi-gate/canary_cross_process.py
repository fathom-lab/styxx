# -*- coding: utf-8 -*-
"""canary_cross_process.py: the canary schedule is the same in every process, from probe seeds only.

The generator of `receipts/canary_replan_cross_process_2026_10_25.json` (the canary re-plan amendment,
"What was measured"). At D0 `canary_plan.py expect` and `canary_shadow.py plan` run in two processes, and
after release a stranger re-derives the schedule in a third, so the property the amendment needs is that one
seed gives one schedule across interpreter processes, not only twice inside one. This script:

1. runs `canary_plan.py derive` for two probe seeds, each in four separate processes (PYTHONHASHSEED 0, 1,
   77 and unset), and compares every digest with the committed probe receipt;
2. seals a PROBE seed in a temporary folder with the harness's own `seal-seed`, then runs `check`, `plan`
   and `canary_plan.py expect` as separate processes, and compares the file `plan` wrote with the digest
   `expect` printed (and checks that `expect` never printed the seed).

It reads a COPY of the gate folder given on the command line and refuses any argument naming the live gate
folder; every `--sealed`, `--out` and `--keys-dir` is the temporary folder, so no sealed folder and no
`canary/` folder of the box is read or written. Paths in the receipt are written with receipt keys.

usage: py -3.12 -B -I papers/darkflobi-gate/canary_cross_process.py <gate copy> <receipt out>
exit 0 only when every comparison holds.
"""
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

GC = Path(sys.argv[1])            # the gate copy
OUT = Path(sys.argv[2])           # receipt path
SNAP = GC / "snapshot_3437340b"
SEEDS = (424242001, 424242002)
START = "2026-11-01"
COMMITTED = {424242001: "32a50c9d6133788cff6b7991321a17950dff7afdcfd0807053ec2f282c740f0a",
             424242002: "b7691d079265dc3e860209bf1d5537c5ec30515699dd2f413f48049530f7a8a9"}
HASHSEEDS = ("0", "1", "77", None)   # None: the variable unset, so the interpreter draws a random hash seed
LIVE_GATE = "C:/Users/heyzo/clawd/gate"


def sha(b):
    return hashlib.sha256(b).hexdigest()


def run(argv, hashseed, keyed_argv):
    for a in argv:
        if str(a).replace("\\", "/").startswith(LIVE_GATE + "/") or str(a).replace("\\", "/") == LIVE_GATE:
            raise SystemExit("REFUSED: an argument names the live gate folder: %s" % a)
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("PYTHONHASHSEED", None)
    if hashseed is not None:
        env["PYTHONHASHSEED"] = hashseed
    t0 = dt.datetime.now(dt.timezone.utc)
    r = subprocess.run([sys.executable, "-B"] + [str(a) for a in argv], capture_output=True, env=env, timeout=600)
    if r.returncode != 0:
        sys.stderr.write(r.stderr.decode("utf-8", errors="replace"))
    return {"argv": keyed_argv, "PYTHONHASHSEED": hashseed if hashseed is not None else "unset",
            "exit": r.returncode, "stdout_sha256": sha(r.stdout),
            "started_utc": t0.replace(microsecond=0).isoformat()}, r


def main():
    rec = {"schema": "darkflobi.gate/canary-plan-cross-process/v0",
           "what": "the canary schedule's digest from two probe seeds, each derived in separate interpreter processes "
                   "under four PYTHONHASHSEED settings, and the file canary_shadow.py plan writes from a sealed probe "
                   "seed against the digest canary_plan.py expect prints in another process; never the sealed seed",
           "generator": "papers/darkflobi-gate/canary_cross_process.py",
           "generator_sha256": sha(Path(__file__).read_bytes()),
           "interpreter": sys.version.split()[0], "implementation": platform.python_implementation(),
           "platform": platform.platform(),
           "canary_plan_sha256": sha((GC / "canary_plan.py").read_bytes()),
           "canary_shadow_sha256": sha((GC / "canary_shadow.py").read_bytes()),
           "pin": (GC / "pin.txt").read_text(encoding="utf-8").strip(),
           "sources": "the nine population documents at the pin, read from the read-only styxx checkout (canary_shadow.py's default --repo)",
           "window_start": START, "days": 28, "cadence": "s7",
           "committed_probe_receipt": "papers/darkflobi-gate/receipts/canary_replan_probe_2026_10_25.json",
           "derive_runs": [], "sealed_probe": {}}
    common = ["--snapshot", str(SNAP)]
    kcommon = ["--snapshot", "BD:gate_copy/snapshot_3437340b"]
    by_seed = {}
    for seed in SEEDS:
        for hs in HASHSEEDS:
            tail = ["derive", "--seed", str(seed), "--window-start", START, "--cadence", "s7"]
            row, r = run([GC / "canary_plan.py"] + common + tail, hs, ["BD:gate_copy/canary_plan.py"] + kcommon + tail)
            out = json.loads(r.stdout.decode("utf-8"))
            row.update({"seed": seed, "schedule_sha256": out["schedule_sha256"], "counts": out["counts"]})
            rec["derive_runs"].append(row)
            by_seed.setdefault(seed, set()).add(out["schedule_sha256"])
    rec["per_seed"] = {str(s): {"runs": len(HASHSEEDS), "distinct_digests_over_runs": len(d),
                                "digest": sorted(d)[0] if len(d) == 1 else sorted(d),
                                "equals_committed_probe_receipt": d == {COMMITTED[s]}} for s, d in by_seed.items()}
    rec["two_seeds_two_schedules"] = by_seed[SEEDS[0]].isdisjoint(by_seed[SEEDS[1]])
    rec["one_seed_same_schedule_in_every_process"] = all(len(d) == 1 for d in by_seed.values())

    # a sealed PROBE seed in a scratch folder: seal-seed, check, plan (one process each), then expect (another)
    with tempfile.TemporaryDirectory(prefix="canary_xproc_") as td:
        td = Path(td)
        sealed, out_dir = td / "sealed", td / "out"
        keys = out_dir / "keys"
        seed = SEEDS[0]
        base = [GC / "canary_shadow.py"] + common + ["--sealed", sealed, "--out", out_dir, "--keys-dir", keys]
        kbase = ["BD:gate_copy/canary_shadow.py"] + kcommon + ["--sealed", "TMP:sealed", "--out", "TMP:out", "--keys-dir", "TMP:out/keys"]
        steps = []
        row, r = run(base + ["seal-seed", "--seed", str(seed)], "1", kbase + ["seal-seed", "--seed", str(seed)])
        steps.append(row)
        row, r = run(base + ["check"], "1", kbase + ["check"])
        row["printed"] = "MATCH" if b": MATCH " in r.stdout else "MISMATCH"
        steps.append(row)
        tail = ["plan", "--cadence", "s7", "--window-start", START]
        row, r = run(base + tail, "77", kbase + tail)
        steps.append(row)
        on_disk = (out_dir / "schedule.json").read_bytes()
        tail = ["expect", "--sealed", sealed, "--keys-dir", keys, "--window-start", START, "--cadence", "s7"]
        ktail = ["expect", "--sealed", "TMP:sealed", "--keys-dir", "TMP:out/keys", "--window-start", START, "--cadence", "s7"]
        row, r = run([GC / "canary_plan.py"] + common + tail, None, ["BD:gate_copy/canary_plan.py"] + kcommon + ktail)
        exp = json.loads(r.stdout.decode("utf-8"))
        row["seed_string_in_stdout"] = str(seed) in r.stdout.decode("utf-8")
        steps.append(row)
        rec["sealed_probe"] = {"seed": seed, "note": "a probe seed sealed in a temporary folder by the harness's own seal-seed",
                               "steps": steps,
                               "plan_file_sha256": sha(on_disk), "plan_file_bytes": len(on_disk),
                               "plan_file_crlf": b"\r\n" in on_disk,
                               "expect_schedule_sha256": exp["schedule_sha256"], "expect_counts": exp["counts"],
                               "plan_file_equals_expect": sha(on_disk) == exp["schedule_sha256"],
                               "plan_file_equals_committed_probe_receipt": sha(on_disk) == COMMITTED[seed]}
    rec["generated_at_utc"] = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    text = json.dumps(rec, indent=1, ensure_ascii=False, sort_keys=True) + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print(text, end="")
    sp = rec["sealed_probe"]
    ok = (rec["two_seeds_two_schedules"] and rec["one_seed_same_schedule_in_every_process"]
          and sp["plan_file_equals_expect"] and sp["plan_file_equals_committed_probe_receipt"]
          and not any(s["seed_string_in_stdout"] for s in sp["steps"] if "seed_string_in_stdout" in s)
          and all(v["equals_committed_probe_receipt"] for v in rec["per_seed"].values())
          and all(row["exit"] == 0 for row in rec["derive_runs"] + sp["steps"]))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
