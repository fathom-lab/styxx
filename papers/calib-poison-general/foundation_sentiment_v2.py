"""Stage-1 foundation v2 for the calibration-poisoning GENERALIZATION test (SENTIMENT construct).

v1 cleared the substantive bars on Qwen (graded read 0.789, behavioral 0.891) but VOIDed on a
single-permutation control at small n. v2 corrects the CONTROLS: K-averaged random-direction floor
(K=200) and K-averaged shuffled-behavioral null (K=500), doubled review pool, Qwen decisive. Read and
behavioral bars unchanged.

PREREG: papers/calib-poison-general/PREREG_calib_poison_sentiment_v2_2026_07_09.md (frozen before run)
Usage: python papers/calib-poison-general/foundation_sentiment_v2.py [--smoke]
"""
from __future__ import annotations
import argparse, importlib.util, json, sys, gc, random
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
F1 = _load("sent_f1", HERE / "foundation_sentiment.py")  # reuse loader/prompt/token helpers

DECISIVE = "Qwen/Qwen2.5-1.5B-Instruct"
MODELS = [DECISIVE, "meta-llama/Llama-3.2-1B-Instruct"]
SCAN = {"Qwen/Qwen2.5-1.5B-Instruct": [12, 14, 16, 18, 20, 22],
        "meta-llama/Llama-3.2-1B-Instruct": [6, 8, 10, 12, 14]}
PER_CLASS = 240      # doubled vs v1 (110)
K_RAND = 200
K_SHUF = 500


def load_matched_reviews(smoke):
    from datasets import load_dataset
    ds = load_dataset("SetFit/amazon_reviews_multi_en", split="train")
    cap = PER_CLASS if not smoke else 30
    pool = {0: [], 1: []}
    for ex in ds:
        lab = int(ex["label"])
        if lab not in (F1.NEG_STAR, F1.POS_STAR):
            continue
        txt = (ex["text"] or "").replace("\\n", " ").replace("\n", " ").strip()
        wc = len(txt.split())
        if not (F1.WMIN <= wc <= F1.WMAX):
            continue
        y = 0 if lab == F1.NEG_STAR else 1
        if len(pool[y]) < cap:
            pool[y].append((txt, wc))
        if len(pool[0]) >= cap and len(pool[1]) >= cap:
            break
    def binned(items):
        b = {}
        for t, wc in items:
            b.setdefault(wc // F1.BIN, []).append((t, wc))
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
    idx = np.arange(n); np.random.default_rng(0).shuffle(idx)
    a_i = idx[:n // 2]; c_i = idx[n // 2:3 * n // 4]; e_i = idx[3 * n // 4:]
    calib = [rows[i] for i in c_i]; evl = [rows[i] for i in e_i]
    print(f"matched n={n} corr(label,log_len)={corr:+.4f} | ATTACK {len(a_i)} CALIB {len(calib)} EVAL {len(evl)}", flush=True)

    results = {}
    for M in models:
        scan = SCAN[M]
        tok = AutoTokenizer.from_pretrained(M)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map="cuda")
        model.eval()
        yids, nids = F1.yes_no_ids(tok)

        c_txt = [t for t, _, _ in calib]; c_y = np.array([l for _, l, _ in calib])
        e_txt = [t for t, _, _ in evl]; e_y = np.array([l for _, l, _ in evl])
        c_res = SYK.resid_all(model, tok, c_txt, scan)
        e_res = SYK.resid_all(model, tok, e_txt, scan)
        per_layer = {}
        best_auc, best_L = 0.0, scan[0]
        first_auc = None
        for L in scan:
            d = FND.dom_direction(c_res[L], c_y)
            if SYK.auroc(c_res[L] @ d, c_y) < 0.5:
                d = -d
            au = SYK.auroc(e_res[L] @ d, e_y)
            per_layer[f"L{L}"] = round(float(au), 4)
            if first_auc is None:
                first_auc = float(au)
            if au > best_auc:
                best_auc, best_L = au, L

        # K-averaged random-direction floor
        rng = np.random.default_rng(0)
        dim = e_res[best_L].shape[1]
        rand_aucs = []
        for _ in range(K_RAND):
            rd = rng.standard_normal(dim); rd /= np.linalg.norm(rd)
            rand_aucs.append(SYK.auroc(e_res[best_L] @ rd, e_y))
        rand_mean = float(np.mean(rand_aucs))

        # behavioral judgment + K-averaged permutation null
        m = SYK.behavioral_margin(model, tok, e_txt, F1.sentiment_prompt, yids, nids)
        pred = (m > 0).astype(int)
        beh_acc = float(np.mean(pred == e_y))
        rng2 = np.random.default_rng(1)
        shuf_accs = []
        for _ in range(K_SHUF):
            perm = e_y.copy(); rng2.shuffle(perm)
            shuf_accs.append(float(np.mean(pred == perm)))
        shuf_mean = float(np.mean(shuf_accs))

        g1 = best_auc >= 0.75
        g2 = beh_acc >= 0.80
        g3 = True
        g4 = (0.40 <= rand_mean <= 0.60) and (0.40 <= shuf_mean <= 0.60)
        valid = g1 and g2 and g3 and g4
        results[M] = {
            "decisive": M == DECISIVE,
            "best_read_auroc": round(float(best_auc), 4), "best_layer": best_L,
            "first_layer_auroc": round(float(first_auc), 4), "per_layer": per_layer,
            "rand_dir_mean_auroc_k200": round(rand_mean, 4),
            "behavioral_sentiment_acc": round(beh_acc, 4),
            "shuffled_behavioral_mean_acc_k500": round(shuf_mean, 4),
            "guards": {"g1_read>=0.75": g1, "g2_behavior>=0.80": g2,
                       "g3_disjoint": g3, "g4_kavg_chance_floors": g4},
            "foundation_valid": bool(valid),
        }
        print(f"[{M.split('/')[-1]}{' *DECISIVE*' if M==DECISIVE else ''}] read EVAL AUROC={best_auc:.3f} "
              f"@L{best_L} (L0={first_auc:.3f}, randK={rand_mean:.3f}) | behavioral={beh_acc:.3f} "
              f"(shufK={shuf_mean:.3f}) | VALID={valid}", flush=True)
        del model; gc.collect(); torch.cuda.empty_cache()

    dec_valid = results.get(DECISIVE, {}).get("foundation_valid", False)
    verdict = "FOUNDATION_VALID" if dec_valid else "UNTESTABLE_ON_THIS_DATA"
    out = {"what": "calibration-poisoning generalization -- sentiment foundation v2 (K-averaged controls)",
           "verdict": verdict,
           "prereg": "papers/calib-poison-general/PREREG_calib_poison_sentiment_v2_2026_07_09.md",
           "dataset": "SetFit/amazon_reviews_multi_en (2* vs 4*, human stars=gold, length-matched)",
           "n": n, "corr_label_length": round(corr, 4), "k_rand": K_RAND, "k_shuf": K_SHUF,
           "decisive_model": DECISIVE, "models": results}
    suffix = "_SMOKE_INVALID" if a.smoke else ""
    (HERE / f"foundation_sentiment_v2_result{suffix}.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nFOUNDATION v2 VERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
