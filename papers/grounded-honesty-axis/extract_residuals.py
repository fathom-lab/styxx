"""Extract residual-stream activations + output signals + labels for the confident-confab probe.
PREREG_residual_confab_probe_2026_05_31.md. White-box, local, no OpenAI.

Per TriviaQA item: greedy short answer; first-token entropy/margin (OUTPUT signals); residual at the
commitment position (predicting the first answer token) across ALL layers; correct/wrong via the
unit-tested _evallib.alias_match. Saves residuals.npz + residuals_meta.json.

  python extract_residuals.py --n 8     # pilot
  python extract_residuals.py --n 800   # full
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


def load_trivia(n):
    from datasets import load_dataset
    ds = load_dataset("trivia_qa", "rc.nocontext", split="validation", streaming=True)
    items = []
    for ex in ds:
        q = (ex.get("question") or "").strip()
        a = ex.get("answer", {}) or {}
        al = (list(a.get("aliases", []) or []) + list(a.get("normalized_aliases", []) or [])
              + ([a["value"]] if a.get("value") else []))
        al = [x for x in al if x]
        if q and al:
            items.append({"q": q, "aliases": al})
        if len(items) >= n:
            break
    return items


def entropy_margin(logits_row):
    lg = logits_row.float()
    p = torch.softmax(lg, dim=-1)
    ent = float(-(p * torch.log(p.clamp_min(1e-12))).sum().item())
    t2 = torch.topk(lg, 2).values
    return ent, float((t2[0] - t2[1]).item())


def is_refusal(a):
    al = a.lower()
    return any(p in al for p in ("i don't know", "i do not know", "not sure", "unknown", "cannot"))


@torch.no_grad()
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800)
    args = ap.parse_args(argv)

    items = load_trivia(args.n)
    khash = hashlib.sha256(json.dumps([[it["q"], sorted(it["aliases"])] for it in items],
                                      ensure_ascii=False).encode("utf-8")).hexdigest()
    print(f"model={MODEL} n={len(items)} probe_sha256={khash}")

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(DEVICE).eval()
    print("model loaded")

    meta, resid = [], []
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
        if fids.input_ids.shape[1] <= plen:
            continue
        out = model(**fids, output_hidden_states=True)
        ent, marg = entropy_margin(out.logits[0, plen - 1])               # OUTPUT signal at commitment
        hs = torch.stack(out.hidden_states, dim=0)[:, 0, plen - 1, :]      # (L, d) residual at commitment
        correct = alias_match(ans, it["aliases"])
        meta.append({"i": idx, "q": it["q"], "answer": ans, "correct": bool(correct),
                     "entropy": ent, "margin": marg})
        resid.append(hs.float().cpu().numpy().astype(np.float16))
        if (len(meta)) % 50 == 0:
            print(f"  kept {len(meta)} (item {idx+1}/{len(items)})")

    R = np.stack(resid, 0)  # (N, L, d)
    np.savez_compressed(os.path.join(HERE, "residuals.npz"), residuals=R)
    json.dump({"model": MODEL, "probe_sha256": khash, "n": len(meta),
               "L": int(R.shape[1]), "d": int(R.shape[2]), "rows": meta},
              open(os.path.join(HERE, "residuals_meta.json"), "w"), indent=2)
    nC = sum(1 for m in meta if m["correct"])
    nW = len(meta) - nC
    # confident subset preview (entropy < median)
    ents = sorted(m["entropy"] for m in meta)
    med = ents[len(ents) // 2] if ents else 0.0
    conf = [m for m in meta if m["entropy"] < med]
    cW = sum(1 for m in conf if not m["correct"])
    print(f"\nextracted {len(meta)} (correct {nC}, wrong {nW}); residuals {R.shape}")
    print(f"median entropy {med:.3f}; confident subset {len(conf)} (confident-wrong {cW}, "
          f"confident-right {len(conf)-cW}) -> need >=25 each")
    print("saved residuals.npz + residuals_meta.json")


if __name__ == "__main__":
    main()
