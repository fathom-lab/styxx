"""Overconfidence_v0 length-robust rebuild — corpus-level fix, offline (PREREG_overconfidence_length_robust).

Generates a MATCHED-LENGTH version of the overconfidence corpus with a local model (same Qs x 2 stances,
identical generous length rule), gates on construct validity, then refits and applies the frozen ship bars.

  python scripts/overconfidence_length_robust.py --generate     # local Qwen gen (GPU)
  python scripts/overconfidence_length_robust.py                # analyze + ship decision
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from overconfidence_train_v0 import QUESTIONS, SYSTEM_CALIBRATED, SYSTEM_OVERCONFIDENT, featurize
from length_control_causal import cv_oof, boot_ci
import styxx.guardrail.calibrated_weights_overconfidence_v0 as W0
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

LEN_RULE = " Respond in EXACTLY 3 sentences, about 55 words — no more, no less."
REPOS = {"qwen": "Qwen/Qwen2.5-3B-Instruct", "phi": "microsoft/Phi-3.5-mini-instruct",
         "gemma": "google/gemma-2-2b-it", "gemini": "gemini-2.5-flash"}
GEMINI_KEY_PATH = Path(r"C:\Users\heyzo\clawd\secrets\gemini-key.txt")
OCD = ROOT / "benchmarks" / "data" / "overconfidence"
V0 = OCD / "pairs_v0.jsonl"
def matched_path(tag): return OCD / f"pairs_lenmatched_{tag}.jsonl"
LEN = {"log_word_count", "mean_sentence_length"}
SHIP_BAR = 0.72


def load(p): return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def generate(tag):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    repo = REPOS[tag]; out = matched_path(tag)
    done = {(r["question"], r["condition"]) for r in load(out)}
    work = [(q, c) for q in QUESTIONS for c in ("calibrated", "overconfident") if (q, c) not in done]
    if not work:
        print(f"[gen] {tag} cache complete ({len(done)})"); return
    print(f"[gen] {repo}: {len(work)} of {len(QUESTIONS)*2}")
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float16, device_map="cuda",
                                                 trust_remote_code=True).eval()
    out.parent.mkdir(parents=True, exist_ok=True); n = 0
    with open(out, "a", encoding="utf-8") as f:
        no_system = "gemma" in repo.lower()
        for q, cond in work:
            sysp = (SYSTEM_CALIBRATED if cond == "calibrated" else SYSTEM_OVERCONFIDENT) + LEN_RULE
            msgs = ([{"role": "user", "content": sysp + "\n\nQuestion: " + q}] if no_system
                    else [{"role": "system", "content": sysp}, {"role": "user", "content": q}])
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to("cuda")
            with torch.no_grad():
                g = model.generate(ids, max_new_tokens=160, do_sample=False, temperature=None, top_p=None,
                                   top_k=None, pad_token_id=tok.eos_token_id)
            t = tok.decode(g[0][ids.shape[1]:], skip_special_tokens=True).strip()
            if t:
                f.write(json.dumps({"question": q, "condition": cond, "response": t,
                                    "label_overconfident": 0 if cond == "calibrated" else 1}) + "\n"); f.flush(); n += 1
                if n % 20 == 0: print(f"  [{n}/{len(work)}]")
    print(f"[gen] wrote {n}"); del model; torch.cuda.empty_cache()


def _gemini_call(model_id, system, user, key, max_retries=6):
    """Single greedy generateContent call with 429/5xx backoff. Frontier instruction-follower so the
    EXACTLY-3-sentences/~55w length rule is actually obeyed on BOTH stances (the 3B failure mode)."""
    import urllib.request, urllib.error, time
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={key}"
    body = {"system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 220,
                                 "thinkingConfig": {"thinkingBudget": 0}}}  # 2.5-flash: thinking off
    data = json.dumps(body).encode()
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            d = json.load(urllib.request.urlopen(req, timeout=60))
            cands = d.get("candidates", [])
            if not cands:
                return ""
            parts = cands[0].get("content", {}).get("parts", [])
            return "".join(p.get("text", "") for p in parts).strip()
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(min(60, 15 * (attempt + 1))); continue
            if e.code >= 500:
                time.sleep(5 * (attempt + 1)); continue
            raise
        except Exception:
            time.sleep(5); continue
    return ""


def generate_gemini():
    import os, time
    model_id = REPOS["gemini"]
    key = os.environ.get("GOOGLE_API_KEY") or GEMINI_KEY_PATH.read_text(encoding="utf-8").strip()
    out = matched_path("gemini")
    done = {(r["question"], r["condition"]) for r in load(out)}
    work = [(q, c) for q in QUESTIONS for c in ("calibrated", "overconfident") if (q, c) not in done]
    if not work:
        print(f"[gen] gemini cache complete ({len(done)})"); return
    print(f"[gen] {model_id}: {len(work)} of {len(QUESTIONS)*2} (resumable)", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True); n = 0
    with open(out, "a", encoding="utf-8") as f:
        for q, cond in work:
            sysp = (SYSTEM_CALIBRATED if cond == "calibrated" else SYSTEM_OVERCONFIDENT) + LEN_RULE
            t = _gemini_call(model_id, sysp, q, key)
            if t:
                f.write(json.dumps({"question": q, "condition": cond, "response": t,
                                    "label_overconfident": 0 if cond == "calibrated" else 1}) + "\n")
                f.flush(); n += 1
                if n % 20 == 0:
                    print(f"  [{n}/{len(work)}]", flush=True)
            time.sleep(1.0)  # paid tier; modest pace, 429-backoff handles any bursts
    print(f"[gen] wrote {n}", flush=True)


def stddiff(c, y): return (c[y == 1].mean() - c[y == 0].mean()) / (c.std() or 1)


def cv_refit_idx(X, y, idxs, seed=0):
    Xi = X[:, idxs]; oof = np.zeros(len(y)); aucs = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xi, y):
        s = StandardScaler().fit(Xi[tr]); c = LogisticRegression(max_iter=2000).fit(s.transform(Xi[tr]), y[tr])
        p = c.predict_proba(s.transform(Xi[te]))[:, 1]; oof[te] = p; aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs)), oof


def shipped_auc(X, names, y):
    idx = [names.index(fn) for fn in W0.FEATURE_NAMES]
    z = (X[:, idx] - np.asarray(W0.SCALER_MEAN)) / np.asarray(W0.SCALER_SCALE)
    return roc_auc_score(y, z @ np.asarray(W0.COEFS) + W0.INTERCEPT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--model", default="qwen", choices=list(REPOS))
    a = ap.parse_args()
    if a.generate:
        generate_gemini() if a.model == "gemini" else generate(a.model)
        return

    rows_v0 = load(V0); rows_m = load(matched_path(a.model))
    if not rows_m:
        print(f"no matched corpus for {a.model} yet — run --generate --model {a.model} first"); return
    print(f"### generator = {a.model} ({REPOS[a.model]})")
    Xv, yv, names = featurize(rows_v0)
    Xm, ym, _ = featurize(rows_m)
    wcm = np.log1p(np.array([len(r["response"].split()) for r in rows_m], float))
    allf = list(range(len(names))); nolen = [i for i, n in enumerate(names) if n not in LEN]

    print(f"=== overconfidence v0 (original, length-confounded), headline {W0.MEAN_CV_AUC} ===")
    print(f"  full-feature CV-AUC (orig): {cv_refit_idx(Xv, yv, allf)[0]:.4f}")

    # ---- Gate 1: construct validity ----
    hd = stddiff(Xm[:, names.index("hedge_density")], ym)
    cd = stddiff(Xm[:, names.index("certainty_marker_density")], ym)
    d_len = stddiff(wcm, ym)
    reg_ok = abs(hd) >= 0.50 and abs(cd) >= 0.50
    len_ok = abs(d_len) <= 0.30
    print(f"\n=== MATCHED corpus (n={len(ym)}), local Qwen, generous ~55w ===")
    print(f"  mean words: calib {np.expm1(wcm[ym==0]).mean():.1f} | overconf {np.expm1(wcm[ym==1]).mean():.1f}")
    print(f"  GATE 1 construct-validity:")
    print(f"    hedge_density std-diff      = {hd:+.3f}  (need |.|>=0.50)")
    print(f"    certainty_marker std-diff   = {cd:+.3f}  (need |.|>=0.50)")
    print(f"    d_len(log_wc)               = {d_len:+.3f}  (need |.|<=0.30)")
    print(f"    -> register {'PRESENT' if reg_ok else 'ABSENT'}; length {'MATCHED' if len_ok else 'NOT matched'}")

    cv_full, oof_full = cv_refit_idx(Xm, ym, allf); ci_full = boot_ci(ym, oof_full)
    cv_nolen = cv_refit_idx(Xm, ym, nolen)[0]
    sh = shipped_auc(Xm, names, ym)
    print(f"\n  refit FULL-feature CV-AUC (matched):  {cv_full:.4f} CI[{ci_full[0]:.3f},{ci_full[1]:.3f}]")
    print(f"  refit NO-length CV-AUC (matched):     {cv_nolen:.4f}  (Δ {cv_nolen-cv_full:+.4f} — length not load-bearing if ~0)")
    print(f"  shipped v0 weights on matched (xfer): {sh:.4f}")
    print(f"  per-feature std-diff (overconf - calib):")
    for j, nm in enumerate(names):
        print(f"      {nm:26s} {stddiff(Xm[:, j], ym):+.3f}" + ("  <-LENGTH" if nm in LEN else ""))

    # ---- Gate 2: ship decision (frozen) ----
    length_not_loadbearing = (cv_nolen - cv_full) >= -0.01
    if not (reg_ok and len_ok):
        decision = "HONEST NULL — Gate 1 failed (corpus did not instantiate a clean matched-length register)"
    elif cv_full >= SHIP_BAR and length_not_loadbearing:
        decision = f"SHIP length-robust v0.3 — cv_match {cv_full:.3f} >= {SHIP_BAR} AND length not load-bearing"
    elif cv_full >= SHIP_BAR:
        decision = f"SHIP (length-feature-DROPPED variant) — cv_match {cv_full:.3f} >= {SHIP_BAR}, refit without length"
    else:
        decision = f"HONEST NULL — cv_match {cv_full:.3f} < {SHIP_BAR}: length-free overconfidence detection too weak to ship; publish ceiling"
    print(f"\n  >>> DECISION (frozen prereg): {decision}")
    (OCD / f"_length_robust_result_{a.model}.json").write_text(json.dumps(
        {"model": a.model, "n": int(len(ym)), "hedge_stddiff": hd, "certainty_stddiff": cd, "d_len": d_len,
         "reg_ok": bool(reg_ok), "len_ok": bool(len_ok), "cv_full_matched": cv_full, "cv_full_ci": list(ci_full),
         "cv_nolen_matched": cv_nolen, "shipped_xfer": sh, "decision": decision,
         "feature_stddiff": {nm: float(stddiff(Xm[:, j], ym)) for j, nm in enumerate(names)}}, indent=2))
    print("  wrote benchmarks/data/overconfidence/_length_robust_result.json")


if __name__ == "__main__":
    main()
