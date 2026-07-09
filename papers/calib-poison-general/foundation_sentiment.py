"""Stage-1 foundation for the calibration-poisoning GENERALIZATION test (SENTIMENT construct).

Second attempt after refusal returned UNTESTABLE. Sentiment on 2-star vs 4-star Amazon reviews
(human star labels = ground truth, length-matched, boundary ratings = graded not lexical). Establishes
a graded sentiment READ (last-review-token) and a behavioral sentiment JUDGMENT both exist. Forward
passes only, no training.

PREREG: papers/calib-poison-general/PREREG_calib_poison_sentiment_2026_07_09.md (frozen before run)
Usage: python papers/calib-poison-general/foundation_sentiment.py [--smoke]
"""
from __future__ import annotations
import argparse, importlib.util, json, math, sys, gc, random
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


SYK = _load("syk_v1", ROOT / "papers/showcase-viz/run_says_yes_knows_no.py")
FND = _load("rnw_found", ROOT / "papers/read-neq-write/foundation.py")

MODELS = ["Qwen/Qwen2.5-1.5B-Instruct", "meta-llama/Llama-3.2-1B-Instruct"]
SCAN = {"Qwen/Qwen2.5-1.5B-Instruct": [12, 14, 16, 18, 20, 22],
        "meta-llama/Llama-3.2-1B-Instruct": [6, 8, 10, 12, 14]}
NEG_STAR, POS_STAR = 1, 3      # 2-star vs 4-star (label 0..4 == 1..5 stars)
WMIN, WMAX = 10, 120
PER_CLASS = 110                # capped pool per class (keeps forward-pass budget modest)
BIN = 8


def load_matched_reviews(smoke):
    from datasets import load_dataset
    ds = load_dataset("SetFit/amazon_reviews_multi_en", split="train")
    cap = PER_CLASS if not smoke else 30
    pool = {0: [], 1: []}
    for ex in ds:
        lab = int(ex["label"])
        if lab not in (NEG_STAR, POS_STAR):
            continue
        txt = (ex["text"] or "").replace("\\n", " ").replace("\n", " ").strip()
        wc = len(txt.split())
        if not (WMIN <= wc <= WMAX):
            continue
        y = 0 if lab == NEG_STAR else 1
        if len(pool[y]) < cap:
            pool[y].append((txt, wc))
        if len(pool[0]) >= cap and len(pool[1]) >= cap:
            break
    # length-match by 8-word bins across classes
    def binned(items):
        b = {}
        for t, wc in items:
            b.setdefault(wc // BIN, []).append((t, wc))
        return b
    bneg, bpos = binned(pool[0]), binned(pool[1])
    rows = []
    for k in sorted(set(bneg) & set(bpos)):
        take = min(len(bneg[k]), len(bpos[k]))
        for t, wc in bneg[k][:take]:
            rows.append((t, 0, wc))
        for t, wc in bpos[k][:take]:
            rows.append((t, 1, wc))
    random.Random(0).shuffle(rows)
    return rows


def sentiment_prompt(review):
    return (f'Is the sentiment of the following product review positive?\n\n"{review}"\n\n'
            f'Answer with exactly one word: Yes or No.')


def yes_no_ids(tok):
    def ids(words):
        out = set()
        for w in words:
            for v in (w, " " + w):
                t = tok(v, add_special_tokens=False).input_ids
                if t:
                    out.add(t[0])
        return sorted(out)
    return ids(["Yes", "YES", "yes"]), ids(["No", "NO", "no"])


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    models = MODELS[:1] if a.smoke else MODELS

    rows = load_matched_reviews(a.smoke)
    y_all = np.array([l for _, l, _ in rows]); wc = np.array([w for _, _, w in rows])
    corr = float(np.corrcoef(y_all, np.log(wc))[0, 1]) if len(set(y_all)) > 1 else float("nan")
    n = len(rows)
    idx = np.arange(n)
    np.random.default_rng(0).shuffle(idx)
    a_i = idx[:n // 2]; c_i = idx[n // 2:3 * n // 4]; e_i = idx[3 * n // 4:]
    attack = [rows[i] for i in a_i]; calib = [rows[i] for i in c_i]; evl = [rows[i] for i in e_i]
    print(f"matched n={n} corr(label,log_len)={corr:+.3f} | ATTACK {len(attack)} CALIB {len(calib)} EVAL {len(evl)}", flush=True)

    results = {}
    for M in models:
        scan = SCAN[M]
        tok = AutoTokenizer.from_pretrained(M)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map="cuda")
        model.eval()
        yids, nids = yes_no_ids(tok)

        c_txt = [t for t, _, _ in calib]; c_y = np.array([l for _, l, _ in calib])
        e_txt = [t for t, _, _ in evl]; e_y = np.array([l for _, l, _ in evl])
        c_res = SYK.resid_all(model, tok, c_txt, scan)
        e_res = SYK.resid_all(model, tok, e_txt, scan)
        per_layer = {}
        best_auc, best_L = 0.0, scan[0]
        first_layer_auc = None
        for L in scan:
            d = FND.dom_direction(c_res[L], c_y)
            if SYK.auroc(c_res[L] @ d, c_y) < 0.5:
                d = -d
            au = SYK.auroc(e_res[L] @ d, e_y)
            per_layer[f"L{L}"] = round(float(au), 4)
            if first_layer_auc is None:
                first_layer_auc = float(au)
            if au > best_auc:
                best_auc, best_L = au, L
        rng = np.random.default_rng(0)
        rd = rng.standard_normal(e_res[best_L].shape[1]); rd /= np.linalg.norm(rd)
        rand = float(SYK.auroc(e_res[best_L] @ rd, e_y))

        m = SYK.behavioral_margin(model, tok, e_txt, sentiment_prompt, yids, nids)
        beh_acc = float(np.mean((m > 0).astype(int) == e_y))
        shuf = e_y.copy(); np.random.default_rng(1).shuffle(shuf)
        beh_shuf = float(np.mean((m > 0).astype(int) == shuf))

        g1 = best_auc >= 0.75
        g2 = beh_acc >= 0.80
        g3 = True  # index split is disjoint by construction
        g4 = (0.35 <= rand <= 0.65) and (0.3 <= beh_shuf <= 0.7)
        lexical_trivial = (first_layer_auc == 1.0) and not (0.35 <= rand <= 0.65)
        valid = g1 and g2 and g3 and g4
        results[M] = {
            "best_read_auroc": round(float(best_auc), 4), "best_layer": best_L,
            "first_layer_auroc": round(float(first_layer_auc), 4), "per_layer": per_layer,
            "rand_auroc": round(rand, 4), "behavioral_sentiment_acc": round(beh_acc, 4),
            "behavioral_shuffled_acc": round(beh_shuf, 4),
            "lexical_trivial_flag": bool(lexical_trivial),
            "guards": {"g1_read>=0.75": g1, "g2_behavior>=0.80": g2,
                       "g3_disjoint": g3, "g4_chance_floors": g4},
            "foundation_valid": bool(valid),
        }
        print(f"[{M.split('/')[-1]}] read EVAL AUROC={best_auc:.3f} @L{best_L} (L0={first_layer_auc:.3f}, "
              f"rand {rand:.3f}) | behavioral sent acc={beh_acc:.3f} (shuf {beh_shuf:.3f}) | "
              f"lexical={lexical_trivial} VALID={valid}", flush=True)
        del model; gc.collect(); torch.cuda.empty_cache()

    any_valid = any(r["foundation_valid"] for r in results.values())
    verdict = "FOUNDATION_VALID" if any_valid else "UNTESTABLE_ON_THIS_DATA"
    out = {"what": "calibration-poisoning generalization -- sentiment foundation",
           "verdict": verdict,
           "prereg": "papers/calib-poison-general/PREREG_calib_poison_sentiment_2026_07_09.md",
           "dataset": "SetFit/amazon_reviews_multi_en (2* vs 4*, human stars=gold, length-matched)",
           "n": n, "corr_label_length": round(corr, 4), "models": results}
    suffix = "_SMOKE_INVALID" if a.smoke else ""
    (HERE / f"foundation_sentiment_result{suffix}.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nFOUNDATION VERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
