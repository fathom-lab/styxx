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
THRESH_MIB = 6000          # gemma-2-2b bf16 (~5.2GB) + transcoders load headroom on the 8GB card
POLL_S = 60
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
            write_status(state="firing", free_mib=free)
            with open(LOG, "w", encoding="utf-8") as lf:
                rc = subprocess.run([sys.executable, os.path.join(HERE, "run_pilot.py")],
                                    stdout=lf, stderr=subprocess.STDOUT).returncode
            write_status(state="done", returncode=rc, log=os.path.relpath(LOG, HERE))
            return
        write_status(state="waiting", free_mib=free, probe_on_gpu=pid_on)
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
