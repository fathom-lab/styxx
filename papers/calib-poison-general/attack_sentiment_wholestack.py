"""Stage-2b: WHOLE-STACK sentiment attack -- does a stronger scrub poison the robust sentiment probe?

Fires from the single-layer Stage-2 PARTIAL. Same as attack_sentiment.py but the adversarial loss
scrubs the per-layer frozen sentiment direction at EVERY scan layer (E2 whole-stack construction),
not just the deployed layer. Reuses Stage-2's split, audit (naive vs private re-lock), behavioral
check, and gradient checkpointing.

PREREG: papers/calib-poison-general/PREREG_calib_poison_sentiment_wholestack_2026_07_09.md (frozen)
Usage: python papers/calib-poison-general/attack_sentiment_wholestack.py [--smoke]
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
AT = _load("sent_attack", HERE / "attack_sentiment.py")   # split_three, private_audit, naive_relock, behavioral_acc

MODEL = AT.MODEL
SCAN = AT.SCAN
LAMBDAS = [1.0, 3.0]
SEEDS = [0, 1]
STEPS = 300
MICRO, ACCUM, LR, MAXTOK = 8, 2, 1e-4, 160
TARGETS = AT.TARGETS


def clean_layer_dirs(model, tok, attack, scan):
    """Per-layer frozen sentiment DoM directions on the CLEAN model's ATTACK reviews."""
    txt = [t for t, _, _ in attack]; y = np.array([l for _, l, _ in attack])
    res = SYK.resid_all(model, tok, txt, scan)
    dirs = {}
    for L in scan:
        d = FND.dom_direction(res[L], y)
        if SYK.auroc(res[L] @ d, y) < 0.5:
            d = -d
        dirs[L] = d
    return dirs


def frozen_stack_read(model, tok, layer_dirs, scan, evl):
    e_txt = [t for t, _, _ in evl]; e_y = np.array([l for _, l, _ in evl])
    res = SYK.resid_all(model, tok, e_txt, scan)
    return float(np.mean([SYK.auroc(res[L] @ layer_dirs[L], e_y) for L in scan]))


def train_wholestack(base_reload, tok, attack, layer_dirs_np, scan, lam, seed, steps, yids_t, nids_t):
    import torch
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(seed); np.random.seed(seed)
    model = get_peft_model(base_reload(), LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, bias="none", target_modules=TARGETS))
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    dev = next(model.parameters()).device
    dirs = {L: torch.tensor(layer_dirs_np[L], dtype=torch.float32, device=dev) for L in scan}
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    a_pos = [t for t, l, _ in attack if l == 1]
    a_neg = [t for t, l, _ in attack if l == 0]
    rng = np.random.default_rng(seed)

    def bids(texts):
        e = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAXTOK)
        return e.input_ids.to(dev), e.attention_mask.to(dev)

    def pids(texts):
        strs = [tok.apply_chat_template([{"role": "user", "content": F1.sentiment_prompt(t)}],
                                        add_generation_prompt=True, tokenize=False) for t in texts]
        e = tok(strs, return_tensors="pt", padding=True, add_special_tokens=False,
                truncation=True, max_length=MAXTOK + 48)
        return e.input_ids.to(dev), e.attention_mask.to(dev)

    def gap(hlast, y, d):
        proj = hlast @ d
        return proj[y == 1].mean() - proj[y == 0].mean()

    for step in range(steps):
        opt.zero_grad()
        for _ in range(ACCUM):
            ts = list(rng.choice(len(a_pos), MICRO // 2, replace=False))
            fs = list(rng.choice(len(a_neg), MICRO // 2, replace=False))
            texts = [a_pos[i] for i in ts] + [a_neg[i] for i in fs]
            y = torch.tensor([1] * (MICRO // 2) + [0] * (MICRO // 2), device=dev)
            ids, attn = bids(texts)
            hs = model(input_ids=ids, attention_mask=attn, output_hidden_states=True).hidden_states
            last = attn.sum(1) - 1
            ar = torch.arange(hs[scan[0]].shape[0])
            l_adv = 0.0
            for L in scan:                                    # whole-stack scrub
                hl = hs[L][ar, last, :].float()
                l_adv = l_adv + gap(hl, y, dirs[L]) ** 2
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

    attack, calib, evl, n = AT.split_three()
    print(f"[split] n={n} ATTACK {len(attack)} CALIB {len(calib)} EVAL {len(evl)}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    def base_reload():
        return AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")

    base = base_reload(); base.eval()
    yids, nids = F1.yes_no_ids(tok)
    layer_dirs = clean_layer_dirs(base, tok, attack, SCAN)
    clean_stack = frozen_stack_read(base, tok, layer_dirs, SCAN, evl)
    clean_priv, clean_rand = AT.private_audit(base, tok, calib, evl)
    clean_beh = AT.behavioral_acc(base, tok, evl, yids, nids)
    guard_ok = clean_priv >= 0.75 and clean_beh >= 0.80
    print(f"[clean] private={clean_priv:.3f} behavioral={clean_beh:.3f} frozen-stack={clean_stack:.3f} guard={guard_ok}", flush=True)
    yids_t = torch.tensor(yids, device="cuda"); nids_t = torch.tensor(nids, device="cuda")
    del base; gc.collect(); torch.cuda.empty_cache()

    points = []
    for seed in seeds:
        for lam in lambdas:
            model = train_wholestack(base_reload, tok, attack, layer_dirs, SCAN, lam, seed, steps, yids_t, nids_t)
            naive = AT.naive_relock(model, tok, attack, evl)
            priv, rand = AT.private_audit(model, tok, calib, evl)
            beh = AT.behavioral_acc(model, tok, evl, yids, nids)
            fstack = frozen_stack_read(model, tok, layer_dirs, SCAN, evl)
            pt = {"seed": seed, "lambda": lam, "naive_relock_auroc": round(naive, 4),
                  "private_relock_auroc": round(priv, 4), "rand_auroc": round(rand, 4),
                  "behavioral_acc": round(beh, 4), "frozen_stack_read": round(fstack, 4),
                  "clean_frozen_stack_read": round(clean_stack, 4),
                  "attack_bit": bool(fstack < clean_stack - 0.05)}
            points.append(pt)
            print(f"[s{seed} lam={lam}] naive={naive:.3f} private={priv:.3f} behavioral={beh:.3f} "
                  f"frozen-stack={fstack:.3f}(clean {clean_stack:.3f})", flush=True)
            del model; gc.collect(); torch.cuda.empty_cache()

    def bp(p):
        return p["behavioral_acc"] >= 0.75
    generalizes = any(p["naive_relock_auroc"] <= 0.60 and p["private_relock_auroc"] >= 0.70 and bp(p)
                      for p in points)
    ng_naive = len({p["seed"] for p in points if p["naive_relock_auroc"] > 0.60 and bp(p)}) >= 2
    ng_priv = len({p["seed"] for p in points if p["private_relock_auroc"] <= 0.60 and bp(p)}) >= 2
    bit = any(p["attack_bit"] for p in points)
    if not guard_ok:
        verdict = "VOID__clean_guard_failed"
    elif not bit:
        verdict = "VOID__wholestack_scrub_did_not_bite"
    elif generalizes:
        verdict = "GENERALIZES__calibration_poisoning_not_honesty_specific"
    elif ng_priv or ng_naive:
        verdict = "NO_GENERALIZATION__sentiment_poison_resistant"
    else:
        verdict = "PARTIAL_OR_INCONCLUSIVE"

    result = {"what": "calibration-poisoning generalization -- sentiment Stage-2b WHOLE-STACK attack",
              "verdict": verdict,
              "prereg": "papers/calib-poison-general/PREREG_calib_poison_sentiment_wholestack_2026_07_09.md",
              "model": MODEL, "scan": SCAN, "scrub": "all scan layers", "lambdas": lambdas,
              "seeds": seeds, "steps": steps, "clean_private_auroc": round(clean_priv, 4),
              "clean_behavioral_acc": round(clean_beh, 4), "clean_frozen_stack_read": round(clean_stack, 4),
              "clean_guard_ok": bool(guard_ok), "points": points}
    suffix = "_SMOKE_INVALID" if a.smoke else ""
    (HERE / f"attack_sentiment_wholestack_result{suffix}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nSTAGE-2b VERDICT: {verdict}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
