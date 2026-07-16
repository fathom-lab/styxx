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


def _other_training_python():
    """True iff a python process OTHER than this watcher is running a GPU experiment script from this
    dir (belt to the total-VRAM suspenders: per-process VRAM is permission-restricted here [N/A])."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"],
            capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return False
    if not out:
        return False
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return False
    procs = data if isinstance(data, list) else [data]
    keys = ("sentiment_probe_parity", "b2_", "b7_", "coupling_confirm.py", "honesty_parity",
            "attack_sweep", "adaptive_erasure", "subspace_erasure")
    for p in procs:
        pid = p.get("ProcessId"); cmd = (p.get("CommandLine") or "")
        if pid and pid != SELF_PID and any(k in cmd for k in keys) and "watcher" not in cmd:
            return True
    return False


def gpu_free():
    """Free iff TOTAL GPU memory in use < FREE_MIB AND no other training-python is on the card.
    Uses total memory.used (reliable) because per-process used_memory is [N/A] under this driver's
    permissions -- a per-process check alone would false-positive 'free' while a run trains."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        used = int(out.splitlines()[0].strip())
    except Exception as e:
        print("nvidia-smi failed:", e, flush=True); return False
    if used >= FREE_MIB:
        return False
    return not _other_training_python()


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
