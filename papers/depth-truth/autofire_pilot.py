"""Auto-fire the keystone pilot when the GPU frees (best-effort monitor — PREREG §6, order item 2).

Polls the card. When the live probe (PID 35256 / rung2w_phi35) releases it AND free VRAM clears the
gemma-2-2b + gemmascope threshold, runs run_pilot.py ONCE and stops. Writes a heartbeat/status to
pilot/autofire_status.json every poll and the run output to pilot/pilot_run.log.

NOT a guaranteed daemon: if this process is killed (session/container stop), the pilot simply doesn't
fire until re-launched or run by hand on the next resume. It writes a .fired sentinel so it never
double-runs — a resume can check `pilot/autofire_status.json` and the sentinel to see what happened.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT_PID = 35256          # rung2w_phi35 (disjoint-worlds), the live holder of the card
# Real headroom for gemma-2-2b bf16 (~5.2GB) + gemmascope transcoders + the attribution graph peak.
# The original n=30 run sat at ~7.4GB; require most of the 8GB card free so we don't OOM into a
# half-released card. VRAM release lags process death, so the floor (not just probe-exit) is the gate.
THRESH_MIB = 7000
POLL_S = 60
SETTLE_S = 10              # let VRAM stabilize a beat after the probe exits before firing
OOM_RETRY_WAIT_S = 180     # OOM on a half-freed card -> wait for full release, retry ONCE
PILOT_DIR = os.path.join(HERE, "pilot")
SENTINEL = os.path.join(PILOT_DIR, ".fired")
STATUS = os.path.join(PILOT_DIR, "autofire_status.json")
LOG = os.path.join(PILOT_DIR, "pilot_run.log")


def _smi(query):
    return subprocess.run(["nvidia-smi", f"--query-{query}", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout


def gpu_state():
    free_line = _smi("gpu=memory.free").strip().splitlines()[0]
    free_mib = int("".join(c for c in free_line if c.isdigit()))
    pid_on = str(PILOT_PID) in _smi("compute-apps=pid")
    return free_mib, pid_on


def write_status(**kw):
    kw["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(STATUS, "w", encoding="utf-8") as f:
        json.dump(kw, f, indent=2)


def _fire_pilot():
    with open(LOG, "w", encoding="utf-8") as lf:
        return subprocess.run([sys.executable, os.path.join(HERE, "run_pilot.py")],
                              stdout=lf, stderr=subprocess.STDOUT).returncode


def _log_has_oom():
    try:
        with open(LOG, encoding="utf-8", errors="ignore") as f:
            t = f.read().lower()
    except Exception:
        return False
    return any(m in t for m in ("out of memory", "outofmemory", "cuda error", "cublas", "cudnn",
                                "cuda out of memory"))


def main():
    os.makedirs(PILOT_DIR, exist_ok=True)
    if os.path.exists(SENTINEL):
        write_status(state="already_fired", note="sentinel present; not re-running")
        return
    write_status(state="starting", pilot_pid=PILOT_PID, threshold_mib=THRESH_MIB)
    while True:
        try:
            free, pid_on = gpu_state()
        except Exception as e:
            write_status(state="poll_error", error=str(e))
            time.sleep(POLL_S)
            continue
        if (not pid_on) and free >= THRESH_MIB:
            open(SENTINEL, "w").close()
            time.sleep(SETTLE_S)                      # VRAM release can lag the probe's exit
            write_status(state="firing", free_mib=free)
            rc = _fire_pilot()
            if rc != 0 and _log_has_oom():
                write_status(state="oom_retry_wait", first_returncode=rc)
                time.sleep(OOM_RETRY_WAIT_S)          # let the card fully release, then retry ONCE
                rc = _fire_pilot()
                write_status(state="done_after_oom_retry", returncode=rc,
                             log=os.path.relpath(LOG, HERE))
            else:
                write_status(state="done", returncode=rc, log=os.path.relpath(LOG, HERE))
            # HALT at the pilot boundary — the monitor fires the pilot ONCE and stops. It NEVER chains
            # into the main run: A1 (adaptation freeze) + A0 (sample sizes) are deliberate human commits
            # made on pilot data first (PREREG §9). This return is that halt.
            return
        write_status(state="waiting", free_mib=free, probe_on_gpu=pid_on)
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
