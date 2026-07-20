"""STAGE B rung 1: the sealed anchored instrument vs a REAL correlated judge panel.

Four prompt-personas of Qwen2.5-1.5B-Instruct (same base model = correlated errors by
construction) judge contradiction items with CONSTRUCTED truth; `styxx.anchors.audit_panel`
audits the panel LABEL-FREE; the held-out truth then scores the audit. Deaf arm = the same
personas judging redacted content (the real-world R4). PREREG_STAGE_B_rung1_2026_07_20.md is
the frozen contract. Crash-safe per-replicate JSONL checkpoint; re-launch skips completed
replicates. `--smoke` writes only *_SMOKE_INVALID*. ASCII only.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))          # repo root -> styxx package
from stage_b_corpus import build_corpus
from styxx.anchors import audit_panel

MODEL = "Qwen/Qwen2.5-3B-Instruct"     # prereg AMENDMENT 1: 1.5B is blind on this task
SEEDS = list(range(3001, 3016))                      # 15 replicates, frozen
N_ORG, K_ANCHOR, PI_TRUE = 240, 80, 0.35
PERSONAS = {
    "logician": "you are a strict logician. judge only logical compatibility.",
    "casual": "you are a casual reader giving your honest quick take.",
    "analyst": "you are a meticulous analyst who weighs every detail before answering.",
    "checker": "you are a terse fact-checker. no explanations.",
}
CKPT = HERE / "stage_b_rung1_checkpoint.jsonl"


def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16,
                                                 device_map="cuda")
    model.eval()
    yes_ids = sorted({tok.encode(v, add_special_tokens=False)[0]
                      for v in ("YES", " YES", "Yes", " Yes", "yes")})
    no_ids = sorted({tok.encode(v, add_special_tokens=False)[0]
                     for v in ("NO", " NO", "No", " No", "no")})
    return tok, model, yes_ids, no_ids


def judge(tok, model, yes_ids, no_ids, persona, items, redact=False, batch=32):
    """Verdict vector for one persona over items; 1 = 'B contradicts A'. Greedy first-token
    YES/NO logit readout at the last (left-padded) position."""
    import torch
    prompts = []
    for it in items:
        a = "[statement withheld]" if redact else it["A"]
        b = "[statement withheld]" if redact else it["B"]
        # prereg AMENDMENT 1: the both-true phrasing; FIRE (contradiction) = the NO answer
        user = (f"statement A: {a}\nstatement B: {b}\n"
                "can statement A and statement B both be true at the same time? "
                "answer with exactly YES or NO.")
        prompts.append(tok.apply_chat_template(
            [{"role": "system", "content": PERSONAS[persona]},
             {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True))
    out = np.zeros(len(items), dtype=int)
    with torch.no_grad():
        for i in range(0, len(prompts), batch):
            enc = tok(prompts[i:i + batch], return_tensors="pt", padding=True).to("cuda")
            logits = model(**enc).logits[:, -1, :]
            y = logits[:, yes_ids].max(dim=1).values
            n = logits[:, no_ids].max(dim=1).values
            out[i:i + batch] = (n > y).long().cpu().numpy()
    return out


def ds_em(V, iters=200, tol=1e-7, clamp=None):
    """Binary DS-EM. clamp: optional vector (len n) with values {0,1,-1}; entries 0/1 pin the
    posterior (semi-supervised, anchors-in-hand); -1 = free. pi is estimated on FREE items."""
    n, J = V.shape
    mu = (V.mean(1) > 0.5).astype(float) * 0.9 + 0.05
    free = np.ones(n, bool) if clamp is None else (clamp < 0)
    if clamp is not None:
        mu[~free] = clamp[~free]
    pi = a = b = None
    for _ in range(iters):
        pi = mu[free].mean()
        b = (mu[:, None] * V).sum(0) / np.maximum(mu.sum(), 1e-12)
        a = ((1 - mu)[:, None] * V).sum(0) / np.maximum((1 - mu).sum(), 1e-12)
        b = np.clip(b, 1e-6, 1 - 1e-6); a = np.clip(a, 1e-6, 1 - 1e-6)
        l1 = np.log(pi + 1e-300) + (V * np.log(b) + (1 - V) * np.log(1 - b)).sum(1)
        l0 = np.log(1 - pi + 1e-300) + (V * np.log(a) + (1 - V) * np.log(1 - a)).sum(1)
        m = np.maximum(l1, l0)
        new = np.exp(l1 - m) / (np.exp(l1 - m) + np.exp(l0 - m))
        if clamp is not None:
            new[~free] = clamp[~free]
        if np.max(np.abs(new - mu)) < tol:
            mu = new; break
        mu = new
    if (b - a).mean() < 0:
        pi, a, b = 1 - pi, 1 - b, 1 - a
    return {"pi": float(pi), "alpha": a.tolist(), "beta": b.tolist()}


def one_replicate(seed, tok, model, yes_ids, no_ids, n_org, k_anchor, smoke=False):
    organic, anchors, truth = build_corpus(seed, n_organic=n_org, k_anchor=k_anchor, pi=PI_TRUE)
    neg_items = [a for a in anchors if a["role"] == "neg_anchor"]
    pos_items = [a for a in anchors if a["role"] == "pos_anchor"]
    P = list(PERSONAS)
    V = np.stack([judge(tok, model, yes_ids, no_ids, p, organic) for p in P], axis=1)
    Vn = np.stack([judge(tok, model, yes_ids, no_ids, p, neg_items) for p in P], axis=1)
    Vp = np.stack([judge(tok, model, yes_ids, no_ids, p, pos_items) for p in P], axis=1)
    Vd = np.stack([judge(tok, model, yes_ids, no_ids, p, organic, redact=True) for p in P], axis=1)
    Vdn = np.stack([judge(tok, model, yes_ids, no_ids, p, neg_items, redact=True) for p in P], axis=1)
    Vdp = np.stack([judge(tok, model, yes_ids, no_ids, p, pos_items, redact=True) for p in P], axis=1)

    y_true = np.array([truth[it["id"]] for it in organic])
    pi_true = float(y_true.mean())                    # realized prevalence this replicate
    audit = audit_panel(V, Vn, Vp, n_boot=300, null_sims=200, seed=seed)
    deaf = audit_panel(Vd, Vdn, Vdp, n_boot=100, null_sims=0, seed=seed)

    mv = float((V.mean(1) > 0.5).mean())
    ds = ds_em(V)
    clamp = np.concatenate([np.full(len(V), -1.0), np.zeros(len(Vn)), np.ones(len(Vp))])
    ss = ds_em(np.vstack([V, Vn, Vp]), clamp=clamp)
    # the ladder's delta_beta / delta_alpha, computable because truth is constructed
    org_alpha = V[y_true == 0].mean(0); org_beta = V[y_true == 1].mean(0)
    rec = {"seed": seed, "smoke": smoke, "pi_true_realized": pi_true,
           "audit": {k: audit.get(k) for k in ("verdict", "pi", "ci", "regime", "s", "s_ci",
                                               "activated", "misfit", "kept", "alpha", "beta")},
           "audit_covered": (bool(audit["ci"][0] <= pi_true <= audit["ci"][1])
                             if audit.get("verdict") == "ESTIMATED" else None),
           "deaf_verdict": deaf["verdict"],
           "mv_pi": mv, "ds_pi": ds["pi"], "ss_ds_pi": ss["pi"],
           "mv_err": abs(mv - pi_true), "ds_err": abs(ds["pi"] - pi_true),
           "ss_ds_err": abs(ss["pi"] - pi_true),
           "audit_err": (abs(audit["pi"] - pi_true) if audit.get("pi") is not None else None),
           "delta_alpha_anchor_minus_organic": (np.asarray(audit["alpha"]) - org_alpha).tolist(),
           "delta_beta_anchor_minus_organic": (np.asarray(audit["beta"]) - org_beta).tolist(),
           "organic_alpha": org_alpha.tolist(), "organic_beta": org_beta.tolist()}
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    done = set()
    if CKPT.exists() and not args.smoke:
        for line in CKPT.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["seed"])
            except Exception:
                pass                                   # torn tail from a crash: recompute
    tok, model, yes_ids, no_ids = load_model()
    seeds = [9999] if args.smoke else [s for s in SEEDS if s not in done]
    n_org, k_anchor = (24, 8) if args.smoke else (N_ORG, K_ANCHOR)
    print(f"replicates to run: {seeds} (done: {sorted(done)})")
    for seed in seeds:
        t0 = time.time()
        rec = one_replicate(seed, tok, model, yes_ids, no_ids, n_org, k_anchor,
                            smoke=args.smoke)
        line = json.dumps(rec)
        if args.smoke:
            (HERE / "stage_b_rung1_SMOKE_INVALID.json").write_text(line, encoding="utf-8")
        else:
            with open(CKPT, "a", encoding="utf-8", newline="\n") as f:
                f.write(line + "\n")
        print(f"  seed {seed}: audit={rec['audit']['verdict']} pi={rec['audit'].get('pi')} "
              f"covered={rec['audit_covered']} deaf={rec['deaf_verdict']} "
              f"({time.time() - t0:.0f}s)")
    print("DONE")


if __name__ == "__main__":
    main()
