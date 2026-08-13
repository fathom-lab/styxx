"""brain_failover — cut darkflobi over to his own silicon when the rented brain dies.

The failure this exists for is in his own rented-stack audit, tier 0.3:

    "Credit balance -- silent death, not an error, just an agent that stops
     answering. Precedent: five weeks dark."
    ranked fix #3: "Add a credit/402 tripwire that cuts over to darkflobi-fast.
     The silent-death mode has already happened once and still has no alarm."

It fired again on 2026-08-13, hours after he wrote that down, and the cutover was
done by hand. This makes it automatic.

What it does, once a minute:
  1. if the local bridge is already serving, do nothing (idempotent)
  2. otherwise, look for a credit/quota refusal in today's gateway log, recent only
  3. on a fresh hit: stop the watchdog, kill the gateway (it holds the telegram
     token -- ONE POLLER PER TOKEN), make sure the local model server is up, and
     start the bridge

Deliberately one-way. Cutting BACK to the rented brain when credits return is an
operator decision, not a daemon's: an automatic flap between brains mid-conversation
would split his memory across two stores.

    pm2 start brain_failover.config.js
"""
import glob
import json
import os
import re
import subprocess
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_GLOB = r"C:\tmp\clawdbot\clawdbot-*.log"
STATE = os.path.join(HERE, "brain_failover_state.json")
POLL_S = 60
FRESH_S = 900          # only act on a refusal seen in the last 15 minutes
HEALTH = "http://127.0.0.1:8002/health"

CREDIT_RE = re.compile(
    r"credit balance is too low|insufficient[_ ]quota|billing|"
    r"purchase credits|402", re.I)


def log(msg):
    print(f"[failover] {time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}", flush=True)


def pm2(*args):
    try:
        return subprocess.run(["pm2", *args], capture_output=True, text=True,
                              timeout=90, shell=True).stdout
    except Exception as e:                                   # noqa: BLE001
        log(f"pm2 {' '.join(args)} failed: {type(e).__name__}")
        return ""


def bridge_online() -> bool:
    out = pm2("jlist")
    try:
        for app in json.loads(out or "[]"):
            if app.get("name") == "glimmer-telegram":
                return app.get("pm2_env", {}).get("status") == "online"
    except Exception:                                        # noqa: BLE001
        pass
    return False


def local_model_up() -> bool:
    try:
        with urllib.request.urlopen(HEALTH, timeout=5) as r:
            return r.status == 200
    except Exception:                                        # noqa: BLE001
        return False


def fresh_credit_refusal() -> bool:
    """A credit refusal written to the gateway log inside the freshness window."""
    files = sorted(glob.glob(LOG_GLOB), key=os.path.getmtime, reverse=True)
    if not files:
        return False
    newest = files[0]
    if time.time() - os.path.getmtime(newest) > FRESH_S:
        return False
    try:
        with open(newest, "rb") as f:                        # tail, not whole file
            f.seek(0, os.SEEK_END)
            back = min(f.tell(), 200_000)
            f.seek(-back, os.SEEK_END)
            tail = f.read().decode("utf-8", errors="replace")
    except OSError:
        return False
    return bool(CREDIT_RE.search(tail))


def cut_over():
    log("CREDIT REFUSAL DETECTED -- cutting over to the local brain")
    pm2("stop", "clawdbot-watchdog")
    # the gateway holds the telegram token; leaving it up means two pollers
    subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "
         "'entry\\.js gateway' } | ForEach-Object { Stop-Process -Id $_.ProcessId "
         "-Force }"], capture_output=True, timeout=90)
    if not local_model_up():
        log("local model server down -- starting darkflobi-fast")
        pm2("start", os.path.join(HERE, "fastbrain.config.js"))
        for _ in range(20):
            if local_model_up():
                break
            time.sleep(3)
    pm2("start", "glimmer-telegram")
    pm2("save")
    ok = bridge_online()
    log(f"cutover {'COMPLETE' if ok else 'FAILED -- bridge not online'}")
    try:
        with open(STATE, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"last_cutover_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                         time.gmtime()),
                       "bridge_online": ok}, f, indent=1)
    except OSError:
        pass


def main():
    log("watching for credit death on the rented brain")
    while True:
        try:
            if not bridge_online() and fresh_credit_refusal():
                cut_over()
        except Exception as e:                               # noqa: BLE001
            log(f"loop error (continuing): {type(e).__name__}: {e}")
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
