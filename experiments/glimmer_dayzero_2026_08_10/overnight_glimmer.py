"""Overnight orchestrator for the Glimmer day-zero local legs. Runs under pm2
so it survives the interactive session. Steps are sequential (one GPU) and
resumable; a failed step logs and the chain continues.

  1. wait for llama-server /health on :8001
  2. glimmer probes leg    (run_fixtures --resume)
  3. glimmer battery leg   (run_fixtures --resume)
  4. knowsay datasheet     (knowsay_glimmer.py, frozen protocol, limit 200, --probe)

Writes overnight.log + OVERNIGHT_DONE.json in this directory.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

HERE = r"C:\Users\heyzo\.styxx\glimmer-day-zero"
PY = sys.executable
ENDPOINT = "http://127.0.0.1:8001"
LOG = os.path.join(HERE, "overnight.log")


def log(msg):
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def wait_health(timeout_s=900):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{ENDPOINT}/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except OSError:
            pass
        time.sleep(10)
    return False


def run_step(name, argv, env_extra=None, timeout_s=16 * 3600):
    log(f"step {name}: start")
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    t0 = time.time()
    try:
        r = subprocess.run(argv, cwd=HERE, env=env, timeout=timeout_s,
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        tail = (r.stdout or "")[-2000:] + (r.stderr or "")[-1000:]
        log(f"step {name}: exit={r.returncode} in {(time.time()-t0)/60:.1f} min\n{tail}")
        return {"step": name, "exit": r.returncode, "minutes": round((time.time()-t0)/60, 1)}
    except subprocess.TimeoutExpired:
        log(f"step {name}: TIMEOUT after {timeout_s}s")
        return {"step": name, "exit": "timeout", "minutes": round(timeout_s/60, 1)}
    except Exception as e:
        log(f"step {name}: ERROR {type(e).__name__}: {e}")
        return {"step": name, "exit": f"error:{type(e).__name__}", "minutes": round((time.time()-t0)/60, 1)}


def main():
    log("overnight chain starting")
    if not wait_health():
        log("FATAL: llama-server never became healthy; aborting")
        sys.exit(1)
    log("llama-server healthy")

    results = []
    results.append(run_step("glimmer-probes", [
        PY, "run_fixtures.py", "--resume",
        "--endpoint", ENDPOINT, "--model", "glimmer-30b",
        "--outdir", "runs/glimmer-q4-probes", "--fixtures", "probes_fixtures.jsonl",
        "--max-tokens", "1024", "--timeout", "600",
    ], env_extra={"STYXX_SESSION_ID": "glimmer-q4-probes"}))

    results.append(run_step("glimmer-battery", [
        PY, "run_fixtures.py", "--resume",
        "--endpoint", ENDPOINT, "--model", "glimmer-30b",
        "--outdir", "runs/glimmer-q4-battery", "--fixtures", "battery.jsonl",
        "--max-tokens", "1024", "--timeout", "600",
    ], env_extra={"STYXX_SESSION_ID": "glimmer-q4-battery"}))

    results.append(run_step("knowsay", [
        PY, "knowsay_glimmer.py", "knowsay_items.jsonl",
        "--base-url", f"{ENDPOINT}/v1", "--model", "glimmer-30b",
        "--probe", "--limit", "200", "--out", "knowsay_glimmer_q4.json",
    ]))

    with open(os.path.join(HERE, "OVERNIGHT_DONE.json"), "w", encoding="utf-8") as f:
        json.dump({"finished_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "steps": results}, f, indent=2)
    log("overnight chain finished")


if __name__ == "__main__":
    main()
