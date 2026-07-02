"""Auto-fire the keystone v2 pilot when the GPU is free — best-effort monitor (PREREG_v2 §6, PHASE 3).

Queues BEHIND the rung2 phi+gemma re-runs (wait-don't-kill): while any `run_g0_stage1b.py` is running, it
waits. It fires `run_pilot.py` ONCE when NO rung2 process is on the box AND free VRAM clears the gemma-2-2b +
gemmascope threshold, then stops. §9 halt: fires the pilot only; never chains to the main run.

Zombie rule (as specified): if a rung2 run's log is stale > 60 min AND GPU util is ~0%, it is a genuine idle
hang — snapshot the state to status, SIGTERM it (taskkill /F), log the reap, and let the loop proceed. A
compute-spinning run (high util) is NOT reaped — wait-don't-kill.

NOT a guaranteed daemon: a session/container stop kills it; re-launch or run run_pilot.py by hand on resume.
Idempotent via a .fired sentinel.
"""
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RUNG2_LOG = r"C:\Users\heyzo\clawd\styxx\papers\disjoint-worlds\_rung2_write.log"
THRESH_MIB = 7000          # gemma-2-2b bf16 + transcoders + attribution peak headroom on the 8GB card
POLL_S = 60
SETTLE_S = 10
OOM_RETRY_WAIT_S = 180
ZOMBIE_STALE_S = 3600      # 60-min stale log ...
ZOMBIE_UTIL_MAX = 5        # ... AND ~0% util => idle hang, reap it
PILOT_DIR = os.path.join(HERE, "pilot")
SENTINEL = os.path.join(PILOT_DIR, ".fired")
STATUS = os.path.join(PILOT_DIR, "autofire_status.json")
LOG = os.path.join(PILOT_DIR, "pilot_run.log")


def _smi(query):
    return subprocess.run(["nvidia-smi", f"--query-{query}", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout


def gpu_free_util():
    line = _smi("gpu=memory.free,utilization.gpu").strip().splitlines()[0]
    nums = re.findall(r"\d+", line)
    return (int(nums[0]), int(nums[1])) if len(nums) >= 2 else (0, 100)


def rung2_pids():
    """PIDs of any running run_g0_stage1b.py (the rung2 sweep we queue behind)."""
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
         "Where-Object { $_.CommandLine -like '*run_g0_stage1b*' } | ForEach-Object { $_.ProcessId }"],
        capture_output=True, text=True).stdout
    return [int(x) for x in re.findall(r"\d+", out)]


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
        t = open(LOG, encoding="utf-8", errors="ignore").read().lower()
    except Exception:
        return False
    return any(m in t for m in ("out of memory", "outofmemory", "cuda error", "cublas", "cudnn"))


def main():
    os.makedirs(PILOT_DIR, exist_ok=True)
    if os.path.exists(SENTINEL):
        write_status(state="already_fired", note="sentinel present; not re-running")
        return
    write_status(state="starting", threshold_mib=THRESH_MIB)
    while True:
        try:
            free, util = gpu_free_util()
            r2 = rung2_pids()
        except Exception as e:
            write_status(state="poll_error", error=str(e)); time.sleep(POLL_S); continue

        if r2:  # rung2 present -> wait behind it, unless it's an idle zombie
            stale = (time.time() - os.path.getmtime(RUNG2_LOG)) if os.path.exists(RUNG2_LOG) else 0
            if stale > ZOMBIE_STALE_S and util < ZOMBIE_UTIL_MAX:
                write_status(state="zombie_reap", rung2_pids=r2, log_stale_s=int(stale), util=util,
                             note="60-min stale log + ~0 util => idle hang; SIGTERM per zombie rule")
                for pid in r2:
                    subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
            else:
                write_status(state="waiting_behind_rung2", rung2_pids=r2, free_mib=free, util=util,
                             log_stale_s=int(stale))
            time.sleep(POLL_S); continue

        if free >= THRESH_MIB:  # card free of rung2 -> fire the v2 pilot once
            open(SENTINEL, "w").close()
            time.sleep(SETTLE_S)
            write_status(state="firing", free_mib=free)
            rc = _fire_pilot()
            if rc != 0 and _log_has_oom():
                write_status(state="oom_retry_wait", first_returncode=rc)
                time.sleep(OOM_RETRY_WAIT_S)
                rc = _fire_pilot()
                write_status(state="done_after_oom_retry", returncode=rc, log=os.path.relpath(LOG, HERE))
            else:
                write_status(state="done", returncode=rc, log=os.path.relpath(LOG, HERE))
            # §9 HALT: pilot fired ONCE; never chains to the main run. This return is that halt.
            return

        write_status(state="waiting", free_mib=free, util=util)
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
