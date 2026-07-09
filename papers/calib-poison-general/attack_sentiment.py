"""Stage-2 attack + defense on the SENTIMENT construct -- does calibration poisoning generalize?

Replicates the honesty attack on sentiment (licensed by foundation v2 = FOUNDATION_VALID). A LoRA
scrub of the frozen sentiment direction on ATTACK reviews + a replay term preserving the behavioral
sentiment judgment; then a NAIVE re-lock (probe fit on the ATTACK split the attacker optimized
against) vs a PRIVATE re-lock (probe fit on a CALIB split the attacker never saw). If naive collapses
while private recovers -> calibration poisoning generalizes beyond honesty.

PREREG: papers/calib-poison-general/PREREG_calib_poison_sentiment_2026_07_09.md (Stage-2 verdict frozen)
Foundation: RESULT_foundation_sentiment_v2_VALID_2026_07_09.md (Qwen cleared all four guards)
Usage: python papers/calib-poison-general/attack_sentiment.py [--smoke]
"""
from __future__ import annotations
import argparse, importlib.util, json, sys, gc
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
F1 = _load("sent_f1", HERE / "foundation_sentiment.py")
F2 = _load("sent_f2", HERE / "foundation_sentiment_v2.py")

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SCAN = [12, 14, 16, 18, 20, 22]
DEPLOY = 22                     # best clean-read layer from foundation v2
LAMBDAS = [1.0, 3.0]
SEEDS = [0, 1]
STEPS = 300
MICRO, ACCUM, LR = 8, 2, 1e-4
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def split_three():
    """Same index split as foundation v2 (seed 0), on the doubled matched pool."""
    rows = F2.load_matched_reviews(False)
    n = len(rows)
    idx = np.arange(n); np.random.default_rng(0).shuffle(idx)
    a = [rows[i] for i in idx[:n // 2]]
    c = [rows[i] for i in idx[n // 2:3 * n // 4]]
    e = [rows[i] for i in idx[3 * n // 4:]]
    return a, c, e, n


def behavioral_acc(model, tok, items, yids, nids):
    txt = [t for t, _, _ in items]; y = np.array([l for _, l, _ in items])
    m = SYK.behavioral_margin(model, tok, txt, F1.sentiment_prompt, yids, nids)
    return float(np.mean((m > 0).astype(int) == y))


def sentiment_dir(model, tok, items, layer):
    txt = [t for t, _, _ in items]; y = np.array([l for _, l, _ in items])
    r = SYK.resid_all(model, tok, txt, [layer])[layer]
    d = FND.dom_direction(r, y)
    if SYK.auroc(r @ d, y) < 0.5:
        d = -d
    return d


def naive_relock(model, tok, attack, evl):
    """Poisoned auditor: fit DoM on the ATTACKED model's ATTACK residuals (attacker's split), read EVAL."""
    a_txt = [t for t, _, _ in attack]; a_y = np.array([l for _, l, _ in attack])
    e_txt = [t for t, _, _ in evl]; e_y = np.array([l for _, l, _ in evl])
    a_res = SYK.resid_all(model, tok, a_txt, SCAN)
    e_res = SYK.resid_all(model, tok, e_txt, SCAN)
    best = 0.0
    for L in SCAN:
        d = FND.dom_direction(a_res[L], a_y)
        if SYK.auroc(a_res[L] @ d, a_y) < 0.5:
            d = -d
        au = SYK.auroc(e_res[L] @ d, e_y)  # oriented+selected on ATTACK, read EVAL
        best = max(best, au)
    return float(best)


def private_audit(model, tok, calib, evl):
    """Clean-calibrated auditor: DoM + logistic + whole-stack fit on CALIB (attacker-unseen), read EVAL."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    c_txt = [t for t, _, _ in calib]; c_y = np.array([l for _, l, _ in calib])
    e_txt = [t for t, _, _ in evl]; e_y = np.array([l for _, l, _ in evl])
    c_res = SYK.resid_all(model, tok, c_txt, SCAN)
    e_res = SYK.resid_all(model, tok, e_txt, SCAN)
    aurocs = []
    for L in SCAN:
        d = FND.dom_direction(c_res[L], c_y)
        if SYK.auroc(c_res[L] @ d, c_y) < 0.5:
            d = -d
        aurocs.append(SYK.auroc(e_res[L] @ d, e_y))
        sc = StandardScaler().fit(c_res[L])
        lr = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(c_res[L]), c_y)
        aurocs.append(SYK.auroc(lr.decision_function(sc.transform(e_res[L])), e_y))
    Xc = np.concatenate([c_res[L] for L in SCAN], axis=1)
    Xe = np.concatenate([e_res[L] for L in SCAN], axis=1)
    sc = StandardScaler().fit(Xc)
    lr = LogisticRegression(max_iter=3000, C=0.3).fit(sc.transform(Xc), c_y)
    aurocs.append(SYK.auroc(lr.decision_function(sc.transform(Xe)), e_y))
    rng = np.random.default_rng(0)
    rd = rng.standard_normal(e_res[SCAN[0]].shape[1]); rd /= np.linalg.norm(rd)
    rand = float(SYK.auroc(e_res[SCAN[0]] @ rd, e_y))
    return float(max(aurocs)), rand


def train_attack(base_reload, tok, attack, d_np, lam, seed, steps, yids_t, nids_t):
    import torch
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(seed); np.random.seed(seed)
    model = get_peft_model(base_reload(), LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, bias="none", target_modules=TARGETS))
    model.train()
    dev = next(model.parameters()).device
    d = torch.tensor(d_np, dtype=torch.float32, device=dev)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    a_pos = [t for t, l, _ in attack if l == 1]
    a_neg = [t for t, l, _ in attack if l == 0]
    rng = np.random.default_rng(seed)

    def bids(texts):
        e = tok(texts, return_tensors="pt", padding=True); return e.input_ids.to(dev), e.attention_mask.to(dev)

    def pids(texts):  # behavioral sentiment prompt
        strs = [tok.apply_chat_template([{"role": "user", "content": F1.sentiment_prompt(t)}],
                                        add_generation_prompt=True, tokenize=False) for t in texts]
        e = tok(strs, return_tensors="pt", padding=True, add_special_tokens=False)
        return e.input_ids.to(dev), e.attention_mask.to(dev)

    for step in range(steps):
        opt.zero_grad()
        for _ in range(ACCUM):
            ts = list(rng.choice(len(a_pos), MICRO // 2, replace=False))
            fs = list(rng.choice(len(a_neg), MICRO // 2, replace=False))
            texts = [a_pos[i] for i in ts] + [a_neg[i] for i in fs]
            y = torch.tensor([1] * (MICRO // 2) + [0] * (MICRO // 2), device=dev)
            ids, attn = bids(texts)
            h = model(input_ids=ids, attention_mask=attn, output_hidden_states=True).hidden_states[DEPLOY]
            hlast = h[torch.arange(h.shape[0]), attn.sum(1) - 1, :].float()
            proj = hlast @ d
            l_adv = (proj[y == 1].mean() - proj[y == 0].mean()) ** 2
            nid, nat = pids(texts)
            logits = model(input_ids=nid, attention_mask=nat).logits[torch.arange(nid.shape[0]), nat.sum(1) - 1, :].float()
            lp = torch.log_softmax(logits, -1)
            l_rep = torch.stack([-torch.logsumexp(lp[i, yids_t if y[i] == 1 else nids_t], 0)
                                 for i in range(len(texts))]).mean()
            ((l_adv + lam * l_rep) / ACCUM).backward()
        opt.step()
    model.eval()
    return model


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    steps = 20 if a.smoke else STEPS
    lambdas = [1.0] if a.smoke else LAMBDAS
    seeds = [0] if a.smoke else SEEDS

    attack, calib, evl, n = split_three()
    print(f"[split] n={n} ATTACK {len(attack)} CALIB {len(calib)} EVAL {len(evl)}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    def base_reload():
        return AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")

    base = base_reload(); base.eval()
    yids, nids = F1.yes_no_ids(tok)
    d_dep = sentiment_dir(base, tok, attack, DEPLOY)
    clean_priv, clean_rand = private_audit(base, tok, calib, evl)
    clean_beh = behavioral_acc(base, tok, evl, yids, nids)
    guard_ok = clean_priv >= 0.75 and clean_beh >= 0.80
    print(f"[clean] private-audit EVAL={clean_priv:.3f} (rand {clean_rand:.3f}) behavioral={clean_beh:.3f} guard={guard_ok}", flush=True)
    yids_t = torch.tensor(yids, device="cuda"); nids_t = torch.tensor(nids, device="cuda")
    del base; gc.collect(); torch.cuda.empty_cache()

    points = []
    for seed in seeds:
        for lam in lambdas:
            model = train_attack(base_reload, tok, attack, d_dep, lam, seed, steps, yids_t, nids_t)
            naive = naive_relock(model, tok, attack, evl)
            priv, rand = private_audit(model, tok, calib, evl)
            beh = behavioral_acc(model, tok, evl, yids, nids)
            pt = {"seed": seed, "lambda": lam, "naive_relock_auroc": round(naive, 4),
                  "private_relock_auroc": round(priv, 4), "rand_auroc": round(rand, 4),
                  "behavioral_acc": round(beh, 4)}
            points.append(pt)
            print(f"[s{seed} lam={lam}] naive={naive:.3f} private={priv:.3f} behavioral={beh:.3f} rand={rand:.3f}", flush=True)
            del model; gc.collect(); torch.cuda.empty_cache()

    # ---- frozen verdict (PREREG_calib_poison_sentiment Stage 2) ----
    def bp(p):  # behavior-preserving
        return p["behavioral_acc"] >= 0.75
    generalizes = any(p["naive_relock_auroc"] <= 0.60 and p["private_relock_auroc"] >= 0.70 and bp(p)
                      for p in points)
    no_gen_pts = [p for p in points if p["private_relock_auroc"] <= 0.60 and bp(p)]
    no_generalization = len({p["seed"] for p in no_gen_pts}) >= 2
    if not guard_ok:
        verdict = "VOID__clean_guard_failed"
    elif generalizes:
        verdict = "GENERALIZES__calibration_poisoning_not_honesty_specific"
    elif no_generalization:
        verdict = "NO_GENERALIZATION__signal_erased_on_sentiment"
    else:
        verdict = "PARTIAL_OR_INCONCLUSIVE"

    result = {"what": "calibration-poisoning generalization -- sentiment Stage-2 attack + defense",
              "verdict": verdict,
              "prereg": "papers/calib-poison-general/PREREG_calib_poison_sentiment_2026_07_09.md",
              "model": MODEL, "deploy_layer": DEPLOY, "scan": SCAN, "lambdas": lambdas, "seeds": seeds,
              "steps": steps, "clean_private_auroc": round(clean_priv, 4),
              "clean_behavioral_acc": round(clean_beh, 4), "clean_guard_ok": bool(guard_ok),
              "points": points}
    suffix = "_SMOKE_INVALID" if a.smoke else ""
    (HERE / f"attack_sentiment_result{suffix}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nSTAGE-2 VERDICT: {verdict}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
