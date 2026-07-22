"""Part 2c: heterogeneous cross-model judge panel (0.5B/1.5B/3B/7B-4bit Qwen2.5-Instruct).
Attacks the model-generality residual Fable-free. PREREG_part2c_crossmodel_2026_07_21.md is the
frozen contract. Two phases: (1) --judge loads each model once and caches its verdict columns
per seed; (2) --score assembles the 4-model panels and runs styxx.anchors.audit_panel over both
the blatant and ladder arms. Crash-safe: --judge skips (model, seed) cells already cached.
`--smoke` uses seed 19996 and writes *_SMOKE_INVALID*. ASCII only.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))
from stage_b_corpus import build_corpus
from styxx.anchors import audit_panel

PERSONA = "you are a meticulous analyst. judge only logical compatibility."
MODELS = [("0.5B", "Qwen/Qwen2.5-0.5B-Instruct", False),
          ("1.5B", "Qwen/Qwen2.5-1.5B-Instruct", False),
          ("3B", "Qwen/Qwen2.5-3B-Instruct", False),
          ("7B4b", "Qwen/Qwen2.5-7B-Instruct", True)]
KILL_JUDGES = {"1.5B", "3B"}          # named in the prereg from diagnostics
SEEDS = list(range(12001, 12013))     # R=12
N_ORG, K_ANCHOR, PI = 200, 80, 0.35
CACHE = HERE / "p2c_crossmodel_cache.jsonl"


def load(mid, quant):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(mid); tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = dict(device_map="cuda")
    if quant:
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    else:
        kw["torch_dtype"] = torch.float16
    m = AutoModelForCausalLM.from_pretrained(mid, **kw); m.eval()
    yi = sorted({tok.encode(v, add_special_tokens=False)[0]
                 for v in ("YES", " YES", "Yes", " Yes", "yes")})
    ni = sorted({tok.encode(v, add_special_tokens=False)[0]
                 for v in ("NO", " NO", "No", " No", "no")})
    return tok, m, yi, ni


def judge(tok, m, yi, ni, items, redact=False, batch=16):
    import torch
    prompts = []
    for it in items:
        a = "[withheld]" if redact else it["A"]
        b = "[withheld]" if redact else it["B"]
        user = (f"statement A: {a}\nstatement B: {b}\n"
                "can statement A and statement B both be true at the same time? "
                "answer with exactly YES or NO.")
        prompts.append(tok.apply_chat_template(
            [{"role": "system", "content": PERSONA}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True))
    out = np.zeros(len(items), int)
    with torch.no_grad():
        for i in range(0, len(prompts), batch):
            enc = tok(prompts[i:i + batch], return_tensors="pt", padding=True).to(m.device)
            lg = m(**enc).logits[:, -1, :]
            y = lg[:, yi].max(1).values; n = lg[:, ni].max(1).values
            out[i:i + batch] = (n > y).long().cpu().numpy()      # fire = NO = contradiction
    return out.tolist()


def corpus(seed, style):
    org, anc, truth = build_corpus(seed, n_organic=N_ORG, k_anchor=K_ANCHOR, pi=PI,
                                   family="attr", anchor_style=style)
    neg = [a for a in anc if a["role"] == "neg_anchor"]
    pos = [a for a in anc if a["role"] == "pos_anchor"]
    return org, neg, pos, truth


def do_judge(smoke):
    import torch
    seeds = [19996] if smoke else SEEDS
    done = set()
    if CACHE.exists() and not smoke:
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line); done.add((d["model"], d["seed"]))
            except Exception:
                pass
    for tag, mid, quant in MODELS:
        if all((tag, s) in done for s in seeds):
            print(f"{tag}: all cached, skip"); continue
        tok, m, yi, ni = load(mid, quant)
        for seed in seeds:
            if (tag, seed) in done:
                continue
            t0 = time.time()
            org, negB, posB, _ = corpus(seed, "blatant")
            _, negL, posL, _ = corpus(seed, "ladder")
            rec = {"model": tag, "seed": seed,
                   "org": judge(tok, m, yi, ni, org),
                   "negB": judge(tok, m, yi, ni, negB), "posB": judge(tok, m, yi, ni, posB),
                   "negL": judge(tok, m, yi, ni, negL), "posL": judge(tok, m, yi, ni, posL),
                   "org_d": judge(tok, m, yi, ni, org, redact=True),
                   "negB_d": judge(tok, m, yi, ni, negB, redact=True),
                   "posB_d": judge(tok, m, yi, ni, posB, redact=True)}
            dest = (HERE / "p2c_crossmodel_SMOKE_INVALID.jsonl") if smoke else CACHE
            with open(dest, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"  {tag} seed {seed} ({time.time() - t0:.0f}s)")
        del m; torch.cuda.empty_cache()
    print("JUDGE DONE")


def do_score():
    cache = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        try:
            d = json.loads(line); cache[(d["model"], d["seed"])] = d
        except Exception:
            pass
    tags = [t for t, _, _ in MODELS]
    kill_idx = [i for i, t in enumerate(tags) if t in KILL_JUDGES]
    rows = []
    for seed in SEEDS:
        if not all((t, seed) in cache for t in tags):
            continue
        org, negB, posB, truth = corpus(seed, "blatant")
        _, negL, posL, _ = corpus(seed, "ladder")
        y = np.array([truth[it["id"]] for it in org]); pit = float(y.mean())
        col = lambda key: np.array([cache[(t, seed)][key] for t in tags]).T
        V, VnB, VpB = col("org"), col("negB"), col("posB")
        VnL, VpL = col("negL"), col("posL")
        Vd, VnBd, VpBd = col("org_d"), col("negB_d"), col("posB_d")
        aB = audit_panel(V, VnB, VpB, n_boot=300, null_sims=200, seed=seed)
        aL = audit_panel(V, VnL, VpL, n_boot=300, null_sims=200, seed=seed)
        deaf = audit_panel(Vd, VnBd, VpBd, n_boot=100, null_sims=0, seed=seed)
        # 7B-only reference oracle: audit the single competent judge in isolation (J=1 -> no
        # pair moments, but the first-moment system still identifies pi from its own anchors)
        i7 = tags.index("7B4b")
        acc = [float((V[:, j] == y).mean()) for j in range(4)]
        rows.append({"seed": seed, "pi_true": pit, "judge_acc": acc,
                     "blatant": {k: aB.get(k) for k in
                                 ("verdict", "pi", "ci", "kept", "misfit")},
                     "blatant_cov": (bool(aB["ci"][0] <= pit <= aB["ci"][1])
                                     if aB.get("verdict") == "ESTIMATED" else None),
                     "ladder": {k: aL.get(k) for k in
                                ("verdict", "pi", "ci", "kept", "misfit")},
                     "ladder_cov": (bool(aL["ci"][0] <= pit <= aL["ci"][1])
                                    if aL.get("verdict") == "ESTIMATED" else None),
                     "deaf_verdict": deaf["verdict"]})
    out = {"prereg": "PREREG_part2c_crossmodel_2026_07_21.md", "n_replicates": len(rows),
           "models": tags, "kill_judges": sorted(KILL_JUDGES), "gates": [], "rows": rows}
    ok = True

    def gate(name, cond, detail):
        nonlocal ok
        ok = ok and bool(cond)
        out["gates"].append({"gate": name, "ok": bool(cond), "detail": detail})
        print(f"  [{'OK ' if cond else 'FAIL'}] {name}: {detail}")

    R = len(rows)
    deafv = sum(1 for r in rows if str(r["deaf_verdict"]).startswith("VOID"))
    gate("X1:deaf_void", deafv >= 11, f"{deafv}/{R} deaf VOID")

    bl_est = [r for r in rows if r["blatant"]["verdict"] == "ESTIMATED"]
    bl_cov = sum(1 for r in bl_est if r["blatant_cov"])
    if len(bl_est) < 8:
        gate("X2:kill_transfers", False, f"VOID_UNDERPOWERED {len(bl_est)}/{R} ESTIMATED")
    else:
        gate("X2:kill_transfers", bl_cov <= 3,
             f"blatant coverage {bl_cov}/{len(bl_est)} ESTIMATED (<= 3)")

    ld_est = [r for r in rows if r["ladder"]["verdict"] == "ESTIMATED"]
    ld_cov = sum(1 for r in ld_est if r["ladder_cov"])
    ld_void = sum(1 for r in rows if r["ladder"]["verdict"] == "VOID_PANEL__uninformative")
    dropped_kill = sum(1 for r in ld_est
                       if not any(r["ladder"]["kept"][i] for i in kill_idx))
    pass_a = len(ld_est) >= 1 and ld_cov >= 9 and dropped_kill > len(ld_est) / 2
    pass_b = ld_void >= 9
    gate("X3:ladder_does_right", pass_a or pass_b,
         f"(a) est {len(ld_est)} cov {ld_cov} kill-dropped {dropped_kill} | (b) VOID {ld_void}/{R}"
         f" -> pass_a={pass_a} pass_b={pass_b}")

    out["all_gates_ok"] = ok
    (HERE / "stage_b_crossmodel_result.json").write_text(json.dumps(out, indent=1),
                                                         encoding="utf-8")
    print(f"\nRESULT: all_gates_ok={ok} -> stage_b_crossmodel_result.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.judge or a.smoke:
        do_judge(a.smoke)
    if a.score:
        do_score()


if __name__ == "__main__":
    main()
