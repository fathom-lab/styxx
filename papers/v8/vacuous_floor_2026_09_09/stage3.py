"""Stage 3: the complete ladder, and the first real verdict.

Runs once the `prereg` verb exists. The order is the one the spec fixes and the one the first
attempt could not satisfy:

    pool battery -> noise plan (prereg) -> R floor runs referencing it -> canonical fingerprint
    carrying the floor -> a second subject at a different precision -> verify --diff against a
    floor that was measured rather than assumed.

The diff is the point. Every verdict this system has produced so far has been on a mock. This one
compares two real fingerprints of the same weights at two precisions, against a floor measured on
the same battery and recipe, and the answer is whatever it is: `same` would say the precision
change is inside the noise this machine produces, `exceeds_floor` would say it is not. Both are
results. Neither is decided here.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WT = Path(r"C:\Users\heyzo\clawd\wt\v8")
OUT = HERE / "out3"
LOG = OUT / "log"
SNAPSHOT = r"C:\Users\heyzo\.cache\huggingface\hub\models--google--gemma-2-2b-it\snapshots\299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8"
sys.path.insert(0, str(WT))

TRANSCRIPT = []
T0 = time.time()


def run(*args, expect=0, note=""):
    cmd = [sys.executable, "-m", "styxx.v8", *[str(a) for a in args]]
    env = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONIOENCODING="utf-8")
    t = time.time()
    p = subprocess.run(cmd, cwd=str(WT), capture_output=True, text=True, encoding="utf-8", env=env)
    try:
        parsed = json.loads(p.stdout) if p.stdout.strip() else None
    except json.JSONDecodeError:
        parsed = {"UNPARSEABLE": p.stdout[:1500]}
    TRANSCRIPT.append({"argv": [str(a) for a in args], "exit": p.returncode,
                       "expected_exit": expect, "seconds": round(time.time() - t, 1),
                       "note": note, "stdout": parsed})
    flag = "OK " if p.returncode == expect else "!! "
    print(f"{flag}exit={p.returncode} (want {expect}) {time.time()-t:6.1f}s  "
          f"{' '.join(str(a) for a in args[:3])}{'  <- ' + note if note else ''}", flush=True)
    if p.returncode != expect:
        print(f"     {json.dumps(parsed)[:700]}", flush=True)
    return parsed, p.returncode


def save(path, obj):
    path.write_text(json.dumps(obj, indent=1), encoding="utf-8", newline="\n")
    return path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    src = HERE / "out"

    # reuse the battery items from the first run so the two are comparable
    items = json.loads((src / "items.json").read_text(encoding="utf-8"))
    items_path = save(OUT / "items.json", items)
    print(f"[items] {len(items)} prompts reused from the first run", flush=True)

    logkey, issuerkey = OUT / "log.pem", OUT / "issuer.pem"
    run("key", "generate", "--out", logkey, "--force")
    run("key", "generate", "--out", issuerkey, "--force")
    pub, _ = run("key", "show", "--key", issuerkey)
    run("log", "init", "--log", LOG, "--key", logkey,
        "--issuer", f"fathom lab={pub['public'] if pub and 'public' in pub else ''}")

    from styxx.v8.runner_hf import subject_from_snapshot
    subj_bf16 = save(OUT / "subject_bf16.json", subject_from_snapshot(SNAPSHOT, "gemma-2", "bf16"))
    subj_fp16 = save(OUT / "subject_fp16.json", subject_from_snapshot(SNAPSHOT, "gemma-2", "fp16"))

    empty = save(OUT / "empty.json", {})
    pool = OUT / "pool.json"
    run("battery", "pool", "--source", items_path, "--key", issuerkey,
        "--subject", subj_bf16, "--recipe", empty, "--out", pool, note="root battery")
    run("log", "append", pool, "--log", LOG, note="index 0")

    # the recipe, with the materials as bytes and BARE-hex material hashes
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    template = tok.chat_template or ""
    lock = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                          capture_output=True, text=True, encoding="utf-8").stdout
    pool_id = json.loads(pool.read_text(encoding="utf-8"))["id"]

    def h(t):
        return hashlib.sha256(t.encode("utf-8")).hexdigest()

    recipe = {
        "battery": pool_id,
        "decoding": {"temperature": 0, "top_p": 1.0, "max_new_tokens": 16, "stop": [],
                     "seed": 7, "batch_size": 1, "padding_side": "left"},
        "materials": {"chat_template": template,
                      "chat_template_source": "tokenizer_config.json@299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8",
                      "system_prompt": "", "env_lock": lock},
        "chat_template_sha256": h(template), "system_prompt_sha256": h(""), "env_lock_sha256": h(lock),
        "harness": {"name": "styxx", "version": "8.0.0", "commit": "worktree-uncommitted"},
    }
    recipe_path = save(OUT / "recipe.json", recipe)

    # ---- the noise plan, minted BEFORE any floor run -------------------------
    plan = OUT / "plan.json"
    res, code = run("prereg", "noise-plan", "--runs", "5",
                    "--nuisance", "batch_size=1|8|32,item_order=canonical|perm11|perm12",
                    "--subject", subj_bf16, "--battery", pool, "--recipe", recipe_path,
                    "--key", issuerkey, "--out", plan,
                    note="THE PLAN: fixed before the runs, which is what stops a chosen nuisance set")
    if code != 0:
        print("\n*** the prereg verb is not available yet; stage 3 stops here ***", flush=True)
        save(HERE / "transcript_stage3.json", {"transcript": TRANSCRIPT, "stopped": "no prereg verb"})
        return
    run("log", "append", plan, "--log", LOG, note="the plan is in the log before any run")
    plan_id = json.loads(plan.read_text(encoding="utf-8"))["id"]

    # ---- the floor runs and the canonical fingerprint ------------------------
    fp_dir = OUT / "fp_bf16"
    fp_dir.mkdir(exist_ok=True)
    run("fingerprint", "--subject", subj_bf16, "--battery", pool, "--recipe", recipe_path,
        "--key", issuerkey, "--runner", "hf", "--runs", "5", "--plan", plan_id,
        "--snapshot", SNAPSHOT, "--dtype", "bfloat16", "--out", fp_dir,
        note="five real runs, this time under the plan")
    for f in sorted(fp_dir.glob("fingerprint-run*.json")):
        run("log", "append", f, "--log", LOG, note=f"floor run {f.name.split('-')[1]}")
    canon = next(fp_dir.glob("fingerprint-canonical-*.json"), None)
    if canon:
        run("log", "append", canon, "--log", LOG, note="the canonical fingerprint, carrying the floor")
        c = json.loads(canon.read_text(encoding="utf-8"))
        nf = c["body"].get("noise_floor")
        print(f"\n[FLOOR] present={nf is not None}", flush=True)
        if nf:
            print(f"[FLOOR] {json.dumps(nf.get('per_channel', {}))[:600]}", flush=True)

    # ---- the second precision, and the first real verdict --------------------
    fp16_dir = OUT / "fp_fp16"
    fp16_dir.mkdir(exist_ok=True)
    run("fingerprint", "--subject", subj_fp16, "--battery", pool, "--recipe", recipe_path,
        "--key", issuerkey, "--runner", "hf", "--runs", "1",
        "--snapshot", SNAPSHOT, "--dtype", "float16", "--out", fp16_dir,
        note="the same weights at a different precision: a declared change, not a nuisance")
    other = next(fp16_dir.glob("fingerprint-*.json"), None)
    if canon and other:
        res, code = run("verify", "--diff", canon, other, "--log", LOG, expect=2,
                        note="THE FIRST REAL VERDICT: bf16 against fp16, judged against a measured floor")
        print(f"\n[VERDICT] {json.dumps(res)[:1200]}", flush=True)

    run("log", "sth", "--log", LOG, "--key", logkey, "--timestamp", "2026-09-09T05:00:00Z")
    save(HERE / "transcript_stage3.json",
         {"transcript": TRANSCRIPT, "wallclock_s": round(time.time() - T0, 1)})
    ok = sum(1 for e in TRANSCRIPT if e["exit"] == e["expected_exit"])
    print(f"\n{ok} of {len(TRANSCRIPT)} commands as expected ({round(time.time()-T0,1)}s)", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        (HERE / "FAILED_stage3.txt").write_text(traceback.format_exc(), encoding="utf-8", newline="\n")
        save(HERE / "transcript_stage3.json", {"transcript": TRANSCRIPT})
        print(traceback.format_exc())
        sys.exit(1)
