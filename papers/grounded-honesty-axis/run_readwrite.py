"""Read-coupled write: does decoding the suppressed answer (probe) then steering toward the DECODED
letter restore it, where a single GENERAL direction failed (un-cave null)?

Mechanistic test of read != write: a general/unconditional steering direction cannot encode the
per-item conditional target (1 direction, 4 letters). A READ-COUPLED write conditions on the probe's
per-item decode. Non-circular: the probe predicts L-hat from the caving state (never the true gold);
at test it can be wrong. Prediction: LIE restore ~ probe accuracy (selective vs MISTAKE where the
probe is at chance), >> the un-cave general-direction floor (0.04).

  python run_readwrite.py --model meta-llama/Llama-3.2-3B-Instruct --restag xf_llama --read-layer 24 --steer-layer 24 --n 25
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from transformers import AutoModelForCausalLM, AutoTokenizer
from gen_intent_set import LETTERS, neutral_user, syco_user, prompt_text, DEVICE
L2I = {"A": 0, "B": 1, "C": 2, "D": 3}
KS = [4.0, 8.0, 12.0, 16.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--restag", default="xf_llama")
    ap.add_argument("--read-layer", type=int, default=24)
    ap.add_argument("--steer-layer", type=int, default=24)
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--skip", type=int, default=4000)
    ap.add_argument("--label", default="llama3b")
    args = ap.parse_args()
    RL, SL = args.read_layer, args.steer_layer

    # ---- train gold probe (read) + per-letter steer directions, from saved caving residuals (TRAIN) ----
    meta = json.load(open(HERE / f"intent_meta{args.restag}.json", encoding="utf-8"))["rows"]
    R = np.load(HERE / f"residuals_intent{args.restag}.npz")["residuals"].astype(np.float32)
    cls = np.array([r["cls"] for r in meta]); gold = np.array([L2I[r["gold"]] for r in meta]); chosen = np.array([L2I[r["chosen"]] for r in meta])
    lie = np.where(cls == "lie")[0]; rng = np.random.RandomState(0)
    li = lie.copy(); rng.shuffle(li); lie_tr = li[:int(0.6 * len(li))]
    sc = StandardScaler().fit(R[lie_tr, RL, :]); probe = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(R[lie_tr, RL, :]), gold[lie_tr])
    # per-letter GENERAL steer directions: mean residual of items that OUTPUT letter L, minus grand mean (answer-agnostic)
    grand = R[:, SL, :].mean(0)
    dL = {}
    for L in range(4):
        idx = np.where(chosen == L)[0]
        v = R[idx, SL, :].mean(0) - grand; dL[L] = v / (np.linalg.norm(v) + 1e-9)
    rv = np.random.RandomState(7).randn(R.shape[2]); rand_dir = rv / np.linalg.norm(rv)
    print(f"[{args.label}] read_L={RL} steer_L={SL} probe trained on {len(lie_tr)} LIE; per-letter dirs built")

    tok = AutoTokenizer.from_pretrained(args.model)
    kw = dict(torch_dtype=torch.float16)
    if "gemma" in args.model.lower(): kw["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw).to(DEVICE).eval()
    ltid = [tok.encode(L, add_special_tokens=False)[0] for L in LETTERS]
    state = {"vec": None, "k": 0.0}
    block = model.model.layers[SL - 1]

    def hook(m, i, o):
        if state["vec"] is None or state["k"] == 0.0:
            return o
        h = o[0] if isinstance(o, tuple) else o
        rms = h[:, -1, :].float().pow(2).mean(-1, keepdim=True).sqrt()
        h[:, -1, :] = h[:, -1, :] + (state["k"] * rms * state["vec"]).to(h.dtype)
        return o
    handle = block.register_forward_hook(hook)

    @torch.no_grad()
    def fwd(text, want_read=False):
        ids = tok(text, return_tensors="pt").to(DEVICE)
        out = model(**ids, output_hidden_states=want_read)
        lg = out.logits[0, -1]
        letters = np.array([float(lg[t]) for t in ltid])
        read = out.hidden_states[RL][0, -1].float().cpu().numpy() if want_read else None
        return letters, read

    # ---- live gather held-out LIE/MISTAKE/RESISTED ----
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test", streaming=True)
    items = {"lie": [], "mistake": [], "resisted": []}; seen = 0; state["vec"] = None
    for ex in ds:
        if all(len(items[c]) >= (args.n if c != "resisted" else 15) for c in items): break
        q = (ex.get("question") or "").strip(); ch = ex.get("choices") or []; a = ex.get("answer")
        if not (q and len(ch) == 4 and isinstance(a, int) and 0 <= a < 4 and len(q) < 600): continue
        seen += 1
        if seen <= args.skip: continue
        ch = [str(c) for c in ch]; g = int(a)
        nl, _ = fwd(prompt_text(tok, neutral_user(q, ch)))
        ncorrect = int(nl.argmax()) == g
        order = np.argsort(-nl); wrong = int(next(j for j in order if j != g))
        sutext = prompt_text(tok, syco_user(q, ch, LETTERS[wrong], ch[wrong]))
        sl, _ = fwd(sutext); chosen_s = int(sl.argmax())
        it = {"su": sutext, "gold": g}
        if ncorrect and chosen_s != g and len(items["lie"]) < args.n: items["lie"].append(it)
        elif ncorrect and chosen_s == g and len(items["resisted"]) < 15: items["resisted"].append(it)
        elif (not ncorrect) and chosen_s != g and len(items["mistake"]) < args.n: items["mistake"].append(it)
    print({c: len(items[c]) for c in items})
    khash = hashlib.sha256(json.dumps([[c, it["gold"]] for c in items for it in items[c]]).encode()).hexdigest()
    print(f"answer-key SHA-256: {khash}")

    # ---- read-then-write per item: probe decodes L-hat from caving read, steer toward dL[L-hat] ----
    def run_cond(its, mode, k):
        ctrl_rng = np.random.RandomState(123)
        ok = 0
        for it in its:
            state["vec"] = None; _, rd = fwd(it["su"], want_read=True)       # capture caving read (unsteered)
            if mode == "readwrite":
                Lhat = int(probe.predict(sc.transform(rd[None, :]))[0]); vec = dL[Lhat]
            elif mode == "randletter":
                Lhat = int(ctrl_rng.randint(0, 4)); vec = dL[Lhat]   # proper per-call random letter (specificity control)
            elif mode == "randdir":
                vec = rand_dir
            state["vec"], state["k"] = torch.tensor(vec, dtype=model.dtype, device=DEVICE), float(k)
            lg2, _ = fwd(it["su"]); state["vec"], state["k"] = None, 0.0
            ok += int(int(lg2.argmax()) == it["gold"])
        return ok / max(1, len(its))

    base = {c: float(np.mean([int(fwd(it["su"])[0].argmax()) == it["gold"] for it in items[c]])) for c in items}
    print(f"baseline gold: LIE {base['lie']:.2f} MIS {base['mistake']:.2f} RES {base['resisted']:.2f}")
    # probe accuracy on the live held-out (read fidelity)
    def probe_acc(its):
        c = 0
        for it in its:
            state["vec"] = None; _, rd = fwd(it["su"], want_read=True)
            c += int(int(probe.predict(sc.transform(rd[None, :]))[0]) == it["gold"])
        return c / max(1, len(its))
    pa = {c: probe_acc(items[c]) for c in ["lie", "mistake"]}
    print(f"probe read-accuracy (live): LIE {pa['lie']:.2f} MIS {pa['mistake']:.2f}")

    res = {"experiment": "read-coupled write (restore via decoded letter)", "model": args.model,
           "read_layer": RL, "steer_layer": SL, "answer_key_sha256": khash, "n": {c: len(items[c]) for c in items},
           "baseline_gold": base, "probe_read_acc_live": pa, "KS": KS, "sweep": {}}
    for k in KS:
        row = {}
        for mode in ["readwrite", "randletter", "randdir"]:
            row[mode] = {c: run_cond(items[c], mode, k) for c in ["lie", "mistake", "resisted"]}
        rw = row["readwrite"]
        sel = (rw["lie"] - base["lie"]) - (rw["mistake"] - base["mistake"])
        rwbreak = base["resisted"] - rw["resisted"]
        res["sweep"][str(k)] = row
        print(f"k={k:>4} | RW LIE {rw['lie']:.2f} MIS {rw['mistake']:.2f} RES {rw['resisted']:.2f} | "
              f"randletter LIE {row['randletter']['lie']:.2f} | randdir LIE {row['randdir']['lie']:.2f} | "
              f"SELECTIVITY {sel:+.2f}  RES-break {rwbreak:+.2f}")
    handle.remove()
    (HERE / f"readwrite_result_{args.label}.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"wrote readwrite_result_{args.label}.json")
    del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
