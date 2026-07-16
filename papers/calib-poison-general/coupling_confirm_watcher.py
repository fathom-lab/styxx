"""Hands-free chain for the B2-coupling CONFIRMATION run -- fires the moment the GPU frees, never
contends with an in-flight scored run (a hard autopilot rail).

A pre-registered sentiment_probe_parity scored run held the 8GB card at launch time. This watcher
polls until the card is genuinely free, then runs the frozen chain:
  1) coupling_confirm.py --calibrate   (base-only disjoint-battery selection; needs >=3 survivors)
  2) coupling_confirm.py --smoke        (confirm the clean-battery guard passes with the selected set)
  3) coupling_confirm.py                (the scored 5-seed run, detached, crash-safe per-seed)

It STOPS at the raw result JSON. It does NOT write a RESULT.md, does NOT certify, does NOT commit or
push -- turning a raw verdict into a certified claim is a reviewed morning step, by design. Every
stage writes _confirm_chain_status.json for the morning report; any failed gate stops the chain with
a BLOCKED status (an honest stop, never a forced run). GPU-free = compute-app VRAM < FREE_MIB and no
other python training process on the card.

Usage: nohup python coupling_confirm_watcher.py > _confirm_watcher.log 2>&1 &
"""
from __future__ import annotations
import json, subprocess, sys, time, os
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATUS = HERE / "_confirm_chain_status.json"
PY = sys.executable
FREE_MIB = 1500            # card considered free when compute-app VRAM is below this
POLL_S = 120
MAX_WAIT_S = 8 * 3600      # give up after 8h of never-free
SELF_PID = os.getpid()


def _t():
    # Date.now-free clock is fine here (real process, not a resumable workflow)
    return time.strftime("%Y-%m-%d %H:%M:%S")


def write_status(stage, **kw):
    STATUS.write_text(json.dumps({"stage": stage, "ts": _t(), **kw}, indent=2) + "\n", encoding="utf-8")
    print(f"[{_t()}] {stage} {kw}", flush=True)


def gpu_free():
    """True iff no compute app is using >= FREE_MIB and no OTHER python process is on the card."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout
    except Exception as e:
        print("nvidia-smi failed:", e, flush=True); return False
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        pid, mem = parts[0], parts[1]
        try:
            mem_i = int(mem)
        except ValueError:
            mem_i = 0            # "[N/A]" (permission) -> treat as unknown-small
        if pid.isdigit() and int(pid) != SELF_PID and mem_i >= FREE_MIB:
            return False
    return True


def run(stage, args, log_name, detach=False):
    logf = HERE / log_name
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    if detach:
        with open(logf, "w") as fh:
            subprocess.Popen([PY, str(HERE / "coupling_confirm.py"), *args], stdout=fh, stderr=subprocess.STDOUT,
                             env=env, creationflags=getattr(subprocess, "DETACHED_PROCESS", 0))
        return None
    with open(logf, "w") as fh:
        rc = subprocess.run([PY, str(HERE / "coupling_confirm.py"), *args], stdout=fh, stderr=subprocess.STDOUT,
                            env=env).returncode
    return rc


def main():
    write_status("WAITING_FOR_GPU", note="a pre-registered scored run holds the card; will not contend")
    waited = 0
    while not gpu_free():
        if waited >= MAX_WAIT_S:
            write_status("TIMED_OUT", waited_s=waited); return 0
        time.sleep(POLL_S); waited += POLL_S
    write_status("GPU_FREE", waited_s=waited)

    # 1) calibrate (base-only disjoint-battery selection)
    run("calibrate", ["--calibrate"], "_confirm_calibrate.log")
    sel = json.loads((HERE / "coupling_confirm_disjoint_selected.json").read_text(encoding="utf-8"))
    if not sel.get("ok"):
        write_status("BLOCKED_calibration", selected=sel.get("selected_disjoint"),
                     base_scores=sel.get("base_scores"),
                     note="fewer than MIN_DISJOINT survivors cleared the floor; not launching a run that would VOID")
        return 0
    write_status("CALIBRATED", selected=sel["selected_disjoint"], aggregate=sel.get("aggregate_selected"))

    # 2) smoke -- confirm the clean-battery guard passes with the selected set
    run("smoke", ["--smoke"], "_confirm_smoke.log")
    smoke = json.loads((HERE / "coupling_confirm_result_SMOKE_INVALID.json").read_text(encoding="utf-8"))
    if not smoke.get("guard_battery"):
        write_status("BLOCKED_smoke_guard", guard_battery=smoke.get("guard_battery"),
                     clean_battery=smoke.get("clean_battery"),
                     note="selected battery still under the clean floor on-device; not launching")
        return 0
    write_status("SMOKE_OK", guard_read=smoke.get("guard_read"), guard_battery=smoke.get("guard_battery"))

    # 3) the scored run, detached + crash-safe. Watcher's job ends here; scoring is a reviewed morning step.
    run("scored_run", [], "_confirm_run.log", detach=True)
    write_status("SCORED_RUN_LAUNCHED",
                 note="5-seed coupling confirmation training detached; raw result -> coupling_confirm_result.json. "
                      "Morning: review the raw verdict, then write+certify RESULT and commit. No auto-certify/commit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
