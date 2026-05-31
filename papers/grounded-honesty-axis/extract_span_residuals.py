"""Extract SPAN residuals (full answer trajectory) + span output signals + labels for the span-residual
probe. PREREG_span_residual_2026_05_31. White-box, local, no API.

Per item: greedy answer; per-answer-token output entropy/margin; residual hidden states across ALL
layers aggregated two ways over the answer span: (a) MEAN over answer tokens, (b) the residual at the
MOST-UNCERTAIN (max-entropy) answer token. Output signals: first-token entropy, span max-entropy, span
min-margin. Label via the unit-tested _evallib.alias_match. Saves residuals_span.npz (mean + maxunc,
each (N,L,d)) + residuals_span_meta.json.

  python extract_span_residuals.py --n 8                  # pilot
  python extract_span_residuals.py --skip 2600 --n 1200   # confirmatory (disjoint from all prior runs)
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from _evallib import alias_match

MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SYS = ("Answer with the single most likely short answer and nothing else. "
       "If you genuinely do not know, reply exactly: I don't know.")


def load_trivia(n, skip=0):
    from datasets import load_dataset
    ds = load_dataset("trivia_qa", "rc.nocontext", split="validation", streaming=True)
    items, seen = [], 0
    for ex in ds:
        q = (ex.get("question") or "").strip()
        a = ex.get("answer", {}) or {}
        al = (list(a.get("aliases", []) or []) + list(a.get("normalized_aliases", []) or [])
              + ([a["value"]] if a.get("value") else []))
        al = [x for x in al if x]
        if not (q and al):
            continue
        seen += 1
        if seen <= skip:
            continue
        items.append({"q": q, "aliases": al})
        if len(items) >= n:
            break
    return items


def ent_marg(lg):
    lg = lg.float()
    p = torch.softmax(lg, dim=-1)
    e = float(-(p * torch.log(p.clamp_min(1e-12))).sum().item())
    t2 = torch.topk(lg, 2).values
    return e, float((t2[0] - t2[1]).item())


def is_refusal(a):
    al = a.lower()
    return any(p in al for p in ("i don't know", "i do not know", "not sure", "unknown", "cannot"))


@torch.no_grad()
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--skip", type=int, default=0)
    args = ap.parse_args(argv)

    items = load_trivia(args.n, args.skip)
    khash = hashlib.sha256(json.dumps([[it["q"], sorted(it["aliases"])] for it in items],
                                      ensure_ascii=False).encode("utf-8")).hexdigest()
    print(f"model={MODEL} skip={args.skip} n={len(items)} sha256={khash}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(DEVICE).eval()
    print("model loaded")

    meta, R_mean, R_maxunc = [], [], []
    for idx, it in enumerate(items):
        text = tok.apply_chat_template(
            [{"role": "system", "content": SYS}, {"role": "user", "content": it["q"]}],
            tokenize=False, add_generation_prompt=True)
        pids = tok(text, return_tensors="pt").to(DEVICE)
        plen = pids.input_ids.shape[1]
        gen = model.generate(**pids, max_new_tokens=24, do_sample=False, pad_token_id=tok.eos_token_id)
        ans = tok.decode(gen[0, plen:], skip_special_tokens=True).strip()
        if not ans or is_refusal(ans):
            continue
        fids = tok(text + ans, return_tensors="pt").to(DEVICE)
        flen = fids.input_ids.shape[1]
        if flen <= plen:
            continue
        out = model(**fids, output_hidden_states=True)
        logits = out.logits[0]                                  # (flen, vocab)
        hs = torch.stack(out.hidden_states, dim=0)[:, 0, :, :]  # (L, flen, d)
        ents, margs = [], []
        for p in range(plen - 1, flen - 1):
            e, mg = ent_marg(logits[p])
            ents.append(e); margs.append(mg)
        ans_hs = hs[:, plen - 1:flen - 1, :].float()           # (L, A, d)
        r_mean = ans_hs.mean(dim=1)                            # (L, d)
        amax = int(np.argmax(ents))
        r_maxunc = hs[:, plen - 1 + amax, :].float()           # (L, d)
        correct = alias_match(ans, it["aliases"])
        meta.append({"i": idx, "q": it["q"], "answer": ans, "correct": bool(correct),
                     "ft_entropy": ents[0], "span_maxent": max(ents), "span_minmargin": min(margs),
                     "n_ans_tok": len(ents)})
        R_mean.append(r_mean.cpu().numpy().astype(np.float16))
        R_maxunc.append(r_maxunc.cpu().numpy().astype(np.float16))
        if len(meta) % 100 == 0:
            print(f"  kept {len(meta)} (item {idx+1}/{len(items)})")

    Rm = np.stack(R_mean, 0)
    Ru = np.stack(R_maxunc, 0)
    np.savez_compressed(os.path.join(HERE, "residuals_span.npz"), mean=Rm, maxunc=Ru)
    json.dump({"model": MODEL, "sha256": khash, "skip": args.skip, "n": len(meta),
               "L": int(Rm.shape[1]), "d": int(Rm.shape[2]), "rows": meta},
              open(os.path.join(HERE, "residuals_span_meta.json"), "w"), indent=2)
    nC = sum(1 for m in meta if m["correct"])
    me = sorted(m["span_maxent"] for m in meta)
    q25 = me[int(0.25 * len(me))] if me else 0.0
    conf = [m for m in meta if m["span_maxent"] < q25]
    cW = sum(1 for m in conf if not m["correct"])
    print(f"\nextracted {len(meta)} (correct {nC}, wrong {len(meta)-nC}); mean {Rm.shape} maxunc {Ru.shape}")
    print(f"bottom-25% span-maxent (confident) subset {len(conf)} (confident-wrong {cW})")
    print("saved residuals_span.npz + residuals_span_meta.json")


if __name__ == "__main__":
    main()
