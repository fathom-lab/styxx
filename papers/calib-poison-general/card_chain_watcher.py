"""Card-chain watcher: B7 (3B erasure) -> B2-coupling (dose-response), hands-free.

The GPU schedule for 2026-07-14: B7's 4-cell scored run (~15h) finishes near midnight; the
B2-coupling scored run (prereg 48e064a, frozen blind to B7) is queued behind it. This watcher
keeps the card hot with zero operator attention:

  poll every POLL_S seconds:
    - B7 process alive            -> keep waiting (never contend with a scored run).
    - B7 gone + result JSON exists -> the run completed: launch b2_coupling_dose.py DETACHED,
                                      write the chain log, exit.
    - B7 gone + NO result          -> it died mid-run: relaunch b7_erasure_3b.py DETACHED
                                      (crash-safe resume skips completed cells), track the new
                                      PID, keep watching. Max MAX_RELAUNCH relaunches, then stop
                                      loudly (a machine that cannot finish B7 needs a human).

Science-neutral: launches the two frozen harnesses exactly as a human would; changes no bar,
no guard, no verdict. The coupling --dry verdict validation passed before this watcher was armed
(all three branches OK). Runtime state (log/pid) is gitignored; this script is committed.

Usage: python papers/calib-poison-general/card_chain_watcher.py [--b7-pid N] [--poll 300]
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
B7_RESULT = HERE / "b7_erasure_3b_result.json"
B7_SCRIPT = HERE / "b7_erasure_3b.py"
COUPLING_SCRIPT = HERE / "b2_coupling_dose.py"
COUPLING_RESULT = HERE / "b2_coupling_dose_result.json"
LOG = HERE / "_card_chain_watcher.log"
PIDFILE = HERE / "_card_chain_watcher.pid"
B7_PIDFILE = HERE / "_b7_run.pid"
COUPLING_PIDFILE = HERE / "_coupling_run.pid"
MAX_RELAUNCH = 3


def log(msg: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def pid_alive(pid: int) -> bool:
    """Windows-safe liveness check via tasklist (no signals)."""
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                             capture_output=True, text=True, timeout=30).stdout
        return f'"{pid}"' in out
    except Exception as e:  # tasklist hiccup: report alive (safe: we just wait longer)
        log(f"pid_alive({pid}) check failed ({e}); assuming alive")
        return True


def launch_detached(script: Path, log_path: Path, pid_path: Path) -> int:
    """Start a harness detached from this process group so it survives session end."""
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("CUDA_VISIBLE_DEVICES", None)   # the harness owns the card
    creation = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
    with log_path.open("ab") as out:
        p = subprocess.Popen([sys.executable, str(script)], cwd=str(ROOT), env=env,
                             stdout=out, stderr=out, creationflags=creation)
    pid_path.write_text(str(p.pid), encoding="ascii")
    return p.pid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b7-pid", type=int, default=None,
                    help="PID of the running B7 process (default: read _b7_run.pid)")
    ap.add_argument("--poll", type=int, default=300, help="poll interval seconds")
    a = ap.parse_args()

    PIDFILE.write_text(str(os.getpid()), encoding="ascii")
    b7_pid = a.b7_pid
    if b7_pid is None and B7_PIDFILE.exists():
        b7_pid = int(B7_PIDFILE.read_text().strip())
    if b7_pid is None:
        log("no B7 PID given or on disk; aborting (arm the watcher with --b7-pid)")
        return 2

    log(f"armed: watching B7 pid={b7_pid}, poll={a.poll}s, max_relaunch={MAX_RELAUNCH}")
    relaunches = 0

    while True:
        if COUPLING_RESULT.exists():
            log("coupling result already exists; nothing to do. exiting.")
            return 0
        alive = pid_alive(b7_pid)
        done = B7_RESULT.exists()
        if alive and not done:
            time.sleep(a.poll)
            continue
        if done:
            # B7 finished (result written even if the process lingers briefly)
            try:
                verdict = json.loads(B7_RESULT.read_text(encoding="utf-8")).get("verdict", "?")
            except Exception:
                verdict = "unreadable"
            log(f"B7 COMPLETE: verdict={verdict}. waiting 60s for the card to settle, then chaining.")
            time.sleep(60)
            cpid = launch_detached(COUPLING_SCRIPT, HERE / "_coupling_run.log", COUPLING_PIDFILE)
            log(f"B2-coupling LAUNCHED detached, pid={cpid} (prereg 48e064a; result -> {COUPLING_RESULT.name})")
            log("chain complete. B7 RESULT doc + certification is the next agent cycle's job. exiting.")
            return 0
        # B7 process gone, no result -> died mid-run
        if relaunches >= MAX_RELAUNCH:
            log(f"B7 died again and relaunch budget ({MAX_RELAUNCH}) is spent. STOPPING LOUDLY — human needed.")
            return 1
        relaunches += 1
        log(f"B7 died without a result (relaunch {relaunches}/{MAX_RELAUNCH}). "
            f"crash-safe resume will skip completed cells. relaunching detached.")
        b7_pid = launch_detached(B7_SCRIPT, HERE / "_b7_run.log", B7_PIDFILE)
        log(f"B7 RELAUNCHED detached, pid={b7_pid}")
        time.sleep(a.poll)


if __name__ == "__main__":
    sys.exit(main())
