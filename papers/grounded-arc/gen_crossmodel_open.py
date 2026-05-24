# -*- coding: utf-8 -*-
"""Cross-FAMILY holdout generation — open-weights Qwen2.5-1.5B-Instruct (local GPU).
Greedy generation (≈temp 0) with true token logprobs via compute_transition_scores;
signal=mean token logprob; gold=detect_refusal. Hashes the holdout. One prompt at a
time for unambiguous logprob extraction. Does NOT run refuse_check (post-commit)."""
import hashlib
import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "scripts/dogfood")
from vendor_robust_refusal_label import detect_refusal  # noqa: E402
from datasets import load_dataset  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
HERE = Path("papers/grounded-arc")
prompts = [{"id": r["id"], "prompt": r["prompt"]} for r in load_dataset("natolambert/xstest-v2-copy")["gpt4"]]

print(f"loading {MODEL_ID} …", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="cuda").eval()
eos = tok.eos_token_id

out = []
for i, row in enumerate(prompts):
    text = tok.apply_chat_template([{"role": "user", "content": row["prompt"]}],
                                   tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        g = model.generate(**inp, max_new_tokens=256, do_sample=False,
                           return_dict_in_generate=True, output_scores=True, pad_token_id=eos)
    gen_toks = g.sequences[0][inp.input_ids.shape[1]:]
    sc = model.compute_transition_scores(g.sequences, g.scores, normalize_logits=True)[0]
    n = len(gen_toks)
    for j, t in enumerate(gen_toks):
        if t.item() == eos:
            n = j + 1
            break
    lp = float(sc[:n].mean().item()) if n > 0 else 0.0
    resp = tok.decode(gen_toks[:n], skip_special_tokens=True)
    out.append({**row, "response": resp, "mean_logprob": lp})
    if (i + 1) % 100 == 0:
        print(f"  {i + 1}/{len(prompts)}", flush=True)

for r in out:
    r["gold"] = int(detect_refusal(r["response"]))
out.sort(key=lambda r: r["id"])
tag = MODEL_ID.replace("/", "_")
with open(HERE / "holdout" / f"refusal_{tag}.jsonl", "w", encoding="utf-8") as fh:
    for r in out:
        fh.write(json.dumps({"id": r["id"], "prompt": r["prompt"], "response": r["response"],
                             "mean_logprob": r["mean_logprob"], "gold": r["gold"]}) + "\n")
sha = hashlib.sha256("\n".join(sorted(f"{r['prompt']}\x1f{r['response']}" for r in out)).encode("utf-8")).hexdigest()
cpath = HERE / "holdout_corpora_crossmodel.json"
corpora = json.loads(cpath.read_text(encoding="utf-8"))
corpora[tag] = {"model": MODEL_ID, "n": len(out), "sha256": sha, "cross_family": True,
                "source": "XSTest gpt4 prompts; local greedy logprobs; gold detect_refusal"}
cpath.write_text(json.dumps(corpora, indent=2) + "\n", encoding="utf-8")
mls = [r["mean_logprob"] for r in out]
print(f"{MODEL_ID}: n={len(out)} gold={dict(Counter(r['gold'] for r in out))} "
      f"mean_lp[{min(mls):.2f},{st.median(mls):.2f},{max(mls):.2f}] sha={sha[:12]}")
print("OPEN-MODEL GEN DONE")
