"""Part 2d: the pricing result on a SECOND task family, heterogeneous cross-model judge panel
(0.5B/1.5B/3B/7B-4bit Qwen2.5-Instruct). PREREG_part2d_pricing_second_family_2026_07_22.md is the
frozen contract. Judge/prompt/decoding code is byte-identical to the frozen part-2c harness
(stage_b_crossmodel.py) -- only the corpus `family` (CLI, default numeric) and the cache path
change. Two phases: (1) --judge caches each model's verdict columns per seed; (2) --score
assembles the 4-model panels and runs styxx.anchors.audit_panel over blatant/ladder/deaf arms and
the frozen PD1/PD2/PD3 gates. Crash-safe: --judge skips (model, seed) cells already cached.
`--smoke` uses seed 19995 and writes *_SMOKE_INVALID*. ASCII only.
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
SEEDS = list(range(12013, 12025))     # fresh disjoint base -- confirmatory design from draw 1
N_ORG, K_ANCHOR, PI = 200, 80, 0.35


def cache_path(family):
    return HERE / f"p2c_fam2_{family}_cache.jsonl"


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


def corpus(seed, style, family):
    org, anc, truth = build_corpus(seed, n_organic=N_ORG, k_anchor=K_ANCHOR, pi=PI,
                                   family=family, anchor_style=style)
    neg = [a for a in anc if a["role"] == "neg_anchor"]
    pos = [a for a in anc if a["role"] == "pos_anchor"]
    return org, neg, pos, truth


def do_judge(smoke, family):
    import torch
    seeds = [19995] if smoke else SEEDS
    cache = cache_path(family)
    done = set()
    if cache.exists() and not smoke:
        for line in cache.read_text(encoding="utf-8").splitlines():
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
            org, negB, posB, _ = corpus(seed, "blatant", family)
            _, negL, posL, _ = corpus(seed, "ladder", family)
            rec = {"model": tag, "seed": seed, "family": family,
                   "org": judge(tok, m, yi, ni, org),
                   "negB": judge(tok, m, yi, ni, negB), "posB": judge(tok, m, yi, ni, posB),
                   "negL": judge(tok, m, yi, ni, negL), "posL": judge(tok, m, yi, ni, posL),
                   "org_d": judge(tok, m, yi, ni, org, redact=True),
                   "negB_d": judge(tok, m, yi, ni, negB, redact=True),
                   "posB_d": judge(tok, m, yi, ni, posB, redact=True)}
            dest = (HERE / f"p2c_fam2_{family}_SMOKE_INVALID.jsonl") if smoke else cache
            with open(dest, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"  {tag} seed {seed} ({time.time() - t0:.0f}s)")
        del m; torch.cuda.empty_cache()
    print("JUDGE DONE")


def do_score(family):
    cache = cache_path(family)
    data = {}
    for line in cache.read_text(encoding="utf-8").splitlines():
        try:
            d = json.loads(line); data[(d["model"], d["seed"])] = d
        except Exception:
            pass
    tags = [t for t, _, _ in MODELS]
    rows = []
    for seed in SEEDS:
        if not all((t, seed) in data for t in tags):
            continue
        org, negB, posB, truth = corpus(seed, "blatant", family)
        _, negL, posL, _ = corpus(seed, "ladder", family)
        y = np.array([truth[it["id"]] for it in org]); pit = float(y.mean())
        col = lambda key: np.array([data[(t, seed)][key] for t in tags]).T
        aB = audit_panel(col("org"), col("negB"), col("posB"), n_boot=300, null_sims=200,
                         seed=seed)
        aL = audit_panel(col("org"), col("negL"), col("posL"), n_boot=300, null_sims=200,
                         seed=seed)
        deaf = audit_panel(col("org_d"), col("negB_d"), col("posB_d"), n_boot=100, null_sims=0,
                           seed=seed)
        acc = [float((col("org")[:, j] == y).mean()) for j in range(4)]
        rows.append({"seed": seed, "pi_true": pit, "judge_acc": acc,
                     "blatant_verdict": aB["verdict"], "blatant_pi": aB.get("pi"),
                     "blatant_cov": (bool(aB["ci"][0] <= pit <= aB["ci"][1])
                                     if aB.get("verdict") == "ESTIMATED" else None),
                     "blatant_err": (abs(aB["pi"] - pit) if aB.get("pi") is not None else None),
                     "ladder_verdict": aL["verdict"], "ladder_pi": aL.get("pi"),
                     "ladder_kept": aL.get("kept"),
                     "ladder_cov": (bool(aL["ci"][0] <= pit <= aL["ci"][1])
                                    if aL.get("verdict") == "ESTIMATED" else None),
                     "ladder_err": (abs(aL["pi"] - pit) if aL.get("pi") is not None else None),
                     "deaf_verdict": deaf["verdict"]})
    R = len(rows)
    out = {"prereg": "PREREG_part2d_pricing_second_family_2026_07_22.md", "family": family,
           "seeds": SEEDS, "n_replicates": R, "models": tags, "gates": [], "rows": rows}
    ok = True

    def gate(name, cond, detail):
        nonlocal ok
        ok = ok and bool(cond)
        out["gates"].append({"gate": name, "ok": bool(cond), "detail": detail})
        print(f"  [{'OK ' if cond else 'FAIL'}] {name}: {detail}")

    bl_est = [r for r in rows if r["blatant_verdict"] == "ESTIMATED"]
    bl_cov = sum(1 for r in bl_est if r["blatant_cov"])
    gate("PD1:kill_transfers", bl_cov <= 3,
         f"blatant coverage {bl_cov}/{len(bl_est)} ESTIMATED (<= 3)")

    ld_est = [r for r in rows if r["ladder_verdict"] == "ESTIMATED"]
    ld_cov = sum(1 for r in ld_est if r["ladder_cov"])
    ld_err = float(np.median([r["ladder_err"] for r in ld_est])) if ld_est else None
    bl_err = float(np.median([r["blatant_err"] for r in bl_est])) if bl_est else None
    margin = (bl_err - ld_err) if (ld_err is not None and bl_err is not None) else None
    ld_void = sum(1 for r in rows if str(r["ladder_verdict"]).startswith("VOID"))
    out["ladder_median_err"] = ld_err
    out["blatant_median_err"] = bl_err
    out["err_margin"] = margin
    out["ladder_void"] = ld_void
    gate("PD2:pricing_recovers",
         len(ld_est) >= 8 and ld_cov >= 10 and margin is not None and margin >= 0.08,
         f"ladder cov {ld_cov}/{len(ld_est)} (>=10, >=8 est), ladder err {ld_err} vs blatant "
         f"{bl_err}, margin {None if margin is None else round(margin, 3)} (>= 0.08); "
         f"ladder VOID {ld_void}/{R}")

    deafv = sum(1 for r in rows if str(r["deaf_verdict"]).startswith("VOID"))
    gate("PD3:deaf_void", deafv >= 11, f"{deafv}/{R} deaf VOID")

    out["all_gates_ok"] = ok
    (HERE / f"stage_b_fam2_{family}_pricing_result.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nRESULT: all_gates_ok={ok} -> stage_b_fam2_{family}_pricing_result.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--family", default="numeric")
    a = ap.parse_args()
    if a.judge or a.smoke:
        do_judge(a.smoke, a.family)
    if a.score:
        do_score(a.family)


if __name__ == "__main__":
    main()
