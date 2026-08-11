"""Finalizer: waits for OVERNIGHT_DONE.json, then closes the day-zero eval
end to end — scores all legs, fills REPORT.md / THREAD_DRAFT.md slots, exports
the styxx appendix, attests the report and verifies the attestation round-trip.

Never posts, never publishes. The last mile (reviewing FINAL_SUMMARY.md and
firing the thread) is the operator's.
"""
import json
import os
import re
import subprocess
import sys
import time

HERE = r"C:\Users\heyzo\.styxx\glimmer-day-zero"
PY = sys.executable
STYXX_REPO = r"C:\Users\heyzo\clawd\styxx"
LOG = os.path.join(HERE, "finalize.log")


def log(msg):
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(argv, timeout_s=1800, **kw):
    r = subprocess.run(argv, cwd=HERE, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=timeout_s, **kw)
    return r


def wait_for_done(max_hours=20):
    marker = os.path.join(HERE, "OVERNIGHT_DONE.json")
    deadline = time.time() + max_hours * 3600
    while time.time() < deadline:
        if os.path.exists(marker):
            return True
        time.sleep(120)
    return False


def fmt(x, digits=3):
    if isinstance(x, (int, float)):
        return f"{x:.{digits}f}".rstrip("0").rstrip(".")
    return "n/a"


def fill(path, mapping):
    txt = open(path, encoding="utf-8").read()
    for k, v in mapping.items():
        txt = txt.replace("{" + k + "}", str(v))
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(txt)


def main():
    log("finalizer waiting for overnight chain")
    if not wait_for_done():
        log("FATAL: overnight chain never wrote OVERNIGHT_DONE.json within 20h")
        sys.exit(1)
    log("overnight chain done — scoring")

    r = run([PY, "score_legs.py",
             "runs/baseline-gemini-probes", "runs/baseline-gemini-battery",
             "runs/glimmer-q4-probes", "runs/glimmer-q4-battery",
             "--fixtures", "probes_fixtures.jsonl", "--fixtures", "battery.jsonl",
             "--out", "verdict_all.json"])
    log(f"score_legs exit={r.returncode}")
    verdict = {}
    vpath = os.path.join(HERE, "verdict_all.json")
    if os.path.exists(vpath):
        verdict = json.load(open(vpath, encoding="utf-8"))

    gp = verdict.get("glimmer-q4-probes", {})
    gb = verdict.get("glimmer-q4-battery", {})

    ks_line = "datasheet missing — knowsay step did not produce output"
    ks_verdict = "MISSING"
    kpath = os.path.join(HERE, "knowsay_glimmer_q4.json")
    if os.path.exists(kpath):
        try:
            ks = json.load(open(kpath, encoding="utf-8"))
            ks_verdict = str(ks.get("verdict") or ks.get("datasheet", {}).get("verdict") or "see file")
            core = ks.get("datasheet", ks)
            keys = ["verdict", "n", "first_correct", "cave_rate", "rescue_rate",
                    "recovery_caved", "recovery_held"]
            ks_line = "; ".join(f"{k}={core[k]}" for k in keys if k in core) or ks_verdict
        except (json.JSONDecodeError, OSError) as e:
            ks_line = f"datasheet unreadable: {e}"

    mapping = {
        "GLIM_PROBES_PASS": fmt(gp.get("pass_rate")),
        "GLIM_PROBES_COMP": fmt(gp.get("mean_composite")),
        "GLIM_BAT_PASS": fmt(gb.get("pass_rate")),
        "GLIM_BAT_COMP": fmt(gb.get("mean_composite")),
        "KNOWSAY_VERDICT": ks_verdict,
        "KNOWSAY_ONE_LINE": ks_line,
    }
    fill(os.path.join(HERE, "REPORT.md"), mapping)
    fill(os.path.join(HERE, "THREAD_DRAFT.md"), mapping)
    log(f"slots filled: {mapping}")

    r = run([PY, "-m", "styxx.cli", "export", "--days", "2", "--name",
             "glimmer-day-zero", "--format", "markdown", "--out", "REPORT_APPENDIX.md"])
    if r.returncode != 0:
        r = run(["styxx", "export", "--days", "2", "--name", "glimmer-day-zero",
                 "--format", "markdown", "--out", "REPORT_APPENDIX.md"])
    log(f"export exit={r.returncode}")

    ref = run(["git", "-C", STYXX_REPO, "rev-parse", "HEAD"]).stdout.strip()
    r = run(["styxx", "attest", "REPORT.md", "--vitals",
             "--prompt", "Day-zero cognometric evaluation of Muse-Glimmer-30B UD-Q4_K_XL (text-only) vs gemini-2.5-flash comparator",
             "--repo", STYXX_REPO, "--ref", ref, "--out", "attestation.json"])
    log(f"attest exit={r.returncode} tail={r.stdout[-300:]}{r.stderr[-200:]}")
    verified = "NOT RUN"
    if r.returncode == 0 and os.path.exists(os.path.join(HERE, "attestation.json")):
        rv = run(["styxx", "verify-attestation", "attestation.json"])
        verified = f"exit={rv.returncode} {rv.stdout[-200:].strip()}"
        log(f"verify-attestation {verified}")

    summary = f"""# glimmer day-zero — FINAL SUMMARY ({time.strftime('%Y-%m-%d %H:%M')})

glimmer UD-Q4_K_XL: probes pass {mapping['GLIM_PROBES_PASS']} (comp {mapping['GLIM_PROBES_COMP']}),
battery pass {mapping['GLIM_BAT_PASS']} (comp {mapping['GLIM_BAT_COMP']}).
baseline gemini-flash: probes 0.625, battery 0.950.
knowsay: {ks_line}

attestation: {verified}
styxx repo ref: {ref}

OPERATOR NEXT: read REPORT.md (slots filled), check THREAD_DRAFT.md numbers against
verdict_all.json, then post if it holds. nothing has been published or posted.
optional next legs: Q3_K_XL dose curve (delete Q4 first), reasoning-channel audit.
"""
    with open(os.path.join(HERE, "FINAL_SUMMARY.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(summary)
    log("FINAL_SUMMARY.md written — finalizer done")


if __name__ == "__main__":
    main()
