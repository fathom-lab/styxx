"""Stage 4: a floor that measured something, and the first verdict that means anything.

Stage 3 produced a floor of exactly zero because the runner could not apply its own plan. That is
repaired: the plan's factors are now written into each run's recipe, each run cert records the
assignment it used, and the log refuses a canonical fingerprint whose runs did not vary what their
plan declared.

So this run asks the question stage 3 could not: with a floor measured across batch sizes 1, 8 and
32 on real weights, does a precision change from bf16 to fp16 exceed the noise this machine
produces on its own? The answer is whatever it is. `same` would mean the precision change hides
inside the machine's own variability, which would be a real limit on what this channel can detect.
`exceeds_floor` would mean the channel separates them. Neither is decided here, and the floor is
measured before the comparison is looked at.
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
OUT = HERE / "out4"
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
        print(f"     {json.dumps(parsed)[:800]}", flush=True)
    return parsed, p.returncode


def save(path, obj):
    path.write_text(json.dumps(obj, indent=1), encoding="utf-8", newline="\n")
    return path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    items = json.loads((HERE / "out" / "items.json").read_text(encoding="utf-8"))
    items_path = save(OUT / "items.json", items)
    print(f"[items] {len(items)} prompts\n", flush=True)

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
    run("log", "append", pool, "--log", LOG)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    template = tok.chat_template or ""
    lock = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                          capture_output=True, text=True, encoding="utf-8").stdout
    pool_id = json.loads(pool.read_text(encoding="utf-8"))["id"]
    h = lambda t: hashlib.sha256(t.encode("utf-8")).hexdigest()
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

    # ---- the plan: batch size is the factor this lab measured moving outputs -------
    plan = OUT / "plan.json"
    run("prereg", "noise-plan", "--runs", "5",
        "--nuisance", "batch_size=1|8|32,item_order=canonical|perm11|perm12",
        "--subject", subj_bf16, "--battery", pool, "--recipe", recipe_path,
        "--key", issuerkey, "--out", plan,
        note="THE PLAN, fixed before any run")
    run("log", "append", plan, "--log", LOG, note="the plan is logged before the runs it governs")

    # ---- five runs that actually apply it -----------------------------------------
    fp_dir = OUT / "fp_bf16"
    fp_dir.mkdir(exist_ok=True)
    res, _ = run("fingerprint", "--subject", subj_bf16, "--battery", pool, "--recipe", recipe_path,
                 "--key", issuerkey, "--runner", "hf", "--runs", "5", "--plan", plan,
                 "--snapshot", SNAPSHOT, "--dtype", "bfloat16", "--out", fp_dir,
                 note="five runs, each under a different assignment of the plan")
    if res and res.get("assignments"):
        print(f"\n[ASSIGNMENTS ACTUALLY USED] {json.dumps(res['assignments'])}\n", flush=True)

    for f in sorted(fp_dir.glob("fingerprint-run*.json")):
        run("log", "append", f, "--log", LOG, note=f"floor run {f.name.split('-')[1]}")
    canon = next(fp_dir.glob("fingerprint-canonical-*.json"), None)
    run("log", "append", canon, "--log", LOG,
        note="the canonical fingerprint; the log now checks that its runs honoured the plan")

    c = json.loads(canon.read_text(encoding="utf-8"))
    nf = c["body"]["noise_floor"]
    print("\n[THE FLOOR, measured across batch sizes]", flush=True)
    for ch, blk in nf["per_channel"].items():
        print(f"    {ch:6s} floor={blk.get('floor')}  distances={blk.get('distances')}", flush=True)
    print(f"    covers={nf.get('covers')}  not_covered={nf.get('not_covered')}", flush=True)
    print(f"    alpha_overall={nf.get('alpha_overall')}\n", flush=True)

    # ---- the comparison ------------------------------------------------------------
    fp16_dir = OUT / "fp_fp16"
    fp16_dir.mkdir(exist_ok=True)
    run("fingerprint", "--subject", subj_fp16, "--battery", pool, "--recipe", recipe_path,
        "--key", issuerkey, "--runner", "hf", "--runs", "1",
        "--snapshot", SNAPSHOT, "--dtype", "float16", "--out", fp16_dir,
        note="the same weights at fp16: a declared change, judged against the measured floor")
    other = next(fp16_dir.glob("fingerprint-*.json"), None)
    res, code = run("verify", "--diff", canon, other, "--log", LOG, expect=2,
                    note="THE VERDICT")
    print("\n" + "=" * 70, flush=True)
    print(res.get("report", json.dumps(res)[:1500]) if res else "no result", flush=True)
    print("=" * 70 + "\n", flush=True)

    run("log", "sth", "--log", LOG, "--key", logkey, "--timestamp", "2026-09-09T07:00:00Z")
    save(HERE / "transcript_stage4.json",
         {"transcript": TRANSCRIPT, "wallclock_s": round(time.time() - T0, 1)})
    ok = sum(1 for e in TRANSCRIPT if e["exit"] == e["expected_exit"])
    print(f"{ok} of {len(TRANSCRIPT)} commands as expected ({round(time.time()-T0,1)}s)", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        (HERE / "FAILED_stage4.txt").write_text(traceback.format_exc(), encoding="utf-8", newline="\n")
        save(HERE / "transcript_stage4.json", {"transcript": TRANSCRIPT})
        print(traceback.format_exc())
        sys.exit(1)
