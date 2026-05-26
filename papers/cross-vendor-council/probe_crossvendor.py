# -*- coding: utf-8 -*-
"""The Cross-Vendor Council. Prereg preregistration_crossvendor_2026_05_25.md. Cracks the
arc's biggest caveat (OpenAI-consensus vs truth) with LOCAL open-weights models from
different vendors -- no API key needed. Council = OpenAI (gpt-4o-mini, gpt-4o) + Alibaba
(Qwen2.5-3B-Instruct) + Google (gemma-2-2b-it). Does agreement still separate real from
fake across 3 disjoint training lineages? Run once."""
from __future__ import annotations
import json, re, statistics, sys, pathlib, warnings
warnings.filterwarnings("ignore")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI
from styxx import council_agreement  # shipped primitive

client = OpenAI(); N = 3
OPENAI = ["gpt-4o-mini", "gpt-4o"]            # vendor: OpenAI
LOCAL = {"Qwen2.5-3B-Instruct": ("Qwen/Qwen2.5-3B-Instruct", "Alibaba"),
         "gemma-2-2b-it": ("google/gemma-2-2b-it", "Google")}
VENDOR = {"gpt-4o-mini": "OpenAI", "gpt-4o": "OpenAI",
          "Qwen2.5-3B-Instruct": "Alibaba", "gemma-2-2b-it": "Google"}
SYS = "Answer in one short sentence with a specific answer."

QA = [
    ("What is the capital of France?", "real_common"),
    ("What is the chemical symbol for gold?", "real_common"),
    ("Who wrote Romeo and Juliet?", "real_common"),
    ("What planet is known as the Red Planet?", "real_common"),
    ("In what year did World War II end?", "real_common"),
    ("How many continents are there on Earth?", "real_common"),
    ("Who won the Nobel Prize in Literature in 1938?", "real_obscure"),
    ("What is the deepest point in the Mediterranean Sea?", "real_obscure"),
    ("Who directed the 1971 film The French Connection?", "real_obscure"),
    ("Who was the 13th President of the United States?", "real_obscure"),
    ("What is the capital of Burkina Faso?", "real_obscure"),
    ("What is the atomic number of tungsten?", "real_obscure"),
    ("What is the capital city of the Republic of Vorland?", "fake"),
    ("In what year did Captain Aldous Renwick first reach the Sundering Isles?", "fake"),
    ("Who won the 1947 Pemberton Literary Prize?", "fake"),
    ("Who composed the 1823 symphony 'The Azure Cascade'?", "fake"),
    ("Which philosopher coined the term 'epistemic verdancy'?", "fake"),
    ("What is the capital of the province of Westmark?", "fake"),
    ("Who was awarded the 1962 Hartwell Medal for physics?", "fake"),
    ("Who wrote the 1932 novel 'The Glass Sentinel'?", "fake"),
]

_ABST = re.compile(r"\b(no such|not aware|does(?:n'?t| not) (?:exist|appear|have)|is not a (?:real|recognized|known)|"
    r"fictional|fictitious|made[- ]?up|could ?n'?t find|cannot find|there (?:is|appears to be) no|i'?m not "
    r"(?:familiar|sure|certain|aware)|no (?:record|information|known|recognized|verifiable|data)|not (?:a )?"
    r"(?:widely )?recognized|unable to (?:find|verify)|don'?t have (?:any |specific )?(?:information|data|record)|"
    r"not (?:specified|in my knowledge)|knowledge base|cannot provide|no (?:widely )?known)\b", re.I)
def is_abstain(t): return bool(_ABST.search(t or ""))
def modal(xs): return max(set(xs), key=xs.count)

# ---- load local models once ----
_local = {}
for name, (mid, _) in LOCAL.items():
    print(f"loading {name}...", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.float16, device_map="cuda")
    _local[name] = (tok, mdl)

def gen_local(name, q):
    tok, mdl = _local[name]
    # fold the instruction into the user turn — gemma's chat template rejects a system role
    inp = tok.apply_chat_template([{"role": "user", "content": f"{SYS}\n\n{q}"}],
                                  add_generation_prompt=True, return_tensors="pt").to("cuda")
    out = mdl.generate(inp, max_new_tokens=40, do_sample=True, temperature=1.0, top_p=0.95,
                       num_return_sequences=N, pad_token_id=tok.eos_token_id)
    return [tok.decode(o[inp.shape[1]:], skip_special_tokens=True).strip().replace("\n", " ") for o in out]

def gen_openai(model, q):
    r = client.chat.completions.create(model=model, temperature=1.0, max_tokens=40, n=N, timeout=40,
        messages=[{"role": "system", "content": SYS}, {"role": "user", "content": q}])
    return [(c.message.content or "").strip() for c in r.choices]

_jc = {}
def judge_same(a, b):
    if not a or not b: return False
    if a.strip() == b.strip(): return True
    k = tuple(sorted((a, b)))
    if k in _jc: return _jc[k]
    try:
        r = client.chat.completions.create(model="gpt-4o-mini", temperature=0, max_tokens=3, timeout=40,
            messages=[{"role": "system", "content": "Reply exactly YES if the two answers give the same core "
                       "factual answer (ignore wording), else NO."}, {"role": "user", "content": f"A: {a}\nB: {b}\nSame?"}])
        v = (r.choices[0].message.content or "").strip().lower().startswith("y")
    except Exception: v = False
    _jc[k] = v; return v

def auc(pos, neg):
    if not pos or not neg: return float("nan")
    w = sum(1 for p in pos for q in neg if p > q) + 0.5*sum(1 for p in pos for q in neg if p == q)
    return w/(len(pos)*len(neg))
def fin(x): return None if (x is None or (isinstance(x, float) and x != x)) else round(x, 3)

rows = []
for q, tier in QA:
    votes = {}                       # model -> modal answer
    for m in OPENAI:
        try: votes[m] = modal(gen_openai(m, q))
        except Exception as e: print(f"  {m} err {e}", file=sys.stderr)
    for name in LOCAL:
        votes[name] = modal(gen_local(name, q))
    subst = [v for m, v in votes.items() if not is_abstain(v)]              # substantive (non-abstention) votes
    agree_all = council_agreement(list(votes.values()), same_fn=judge_same)
    agree_subst = council_agreement(subst, same_fn=judge_same) if subst else 0.0
    agree_openai = council_agreement([votes[m] for m in OPENAI if m in votes and not is_abstain(votes[m])], same_fn=judge_same)
    # X3: cross-vendor shared confabulation on fakes
    xshare = False
    if tier == "fake":
        nz = [(VENDOR[m], v) for m, v in votes.items() if not is_abstain(v)]
        for i in range(len(nz)):
            for j in range(i+1, len(nz)):
                if nz[i][0] != nz[j][0] and judge_same(nz[i][1], nz[j][1]): xshare = True
    rows.append(dict(tier=tier, q=q[:40], agree_subst=fin(agree_subst), agree_all=fin(agree_all),
                     agree_openai_only=fin(agree_openai),
                     n_abstain=sum(1 for v in votes.values() if is_abstain(v)),
                     xvendor_shared_confab=xshare,
                     votes={m: v[:34] for m, v in votes.items()}))
    print(f"[{tier}] agree_subst={agree_subst:.2f} all={agree_all:.2f} openai={agree_openai:.2f} "
          f"abst={rows[-1]['n_abstain']} xshare={xshare} :: {q[:30]!r}", file=sys.stderr)

real = [r for r in rows if r["tier"].startswith("real")]
common = [r for r in rows if r["tier"] == "real_common"]
fake = [r for r in rows if r["tier"] == "fake"]
X1_auc = auc([r["agree_subst"] for r in real], [r["agree_subst"] for r in fake])
mean_common = statistics.fmean([r["agree_subst"] for r in common])
mean_fake = statistics.fmean([r["agree_subst"] for r in fake])
X1 = (not (X1_auc != X1_auc)) and X1_auc >= 0.75
X2 = (mean_common >= 0.70) and (mean_fake <= 0.45)
n_xshare = sum(1 for r in fake if r["xvendor_shared_confab"])
out = {
    "council": list(VENDOR.keys()), "vendors": sorted(set(VENDOR.values())),
    "X1_crossvendor_auc": fin(X1_auc),
    "mean_agree_real_common": fin(mean_common), "mean_agree_real_obscure": fin(statistics.fmean([r["agree_subst"] for r in rows if r["tier"]=="real_obscure"])),
    "mean_agree_fake": fin(mean_fake),
    "auc_openai_only_subset": fin(auc([r["agree_openai_only"] for r in real], [r["agree_openai_only"] for r in fake])),
    "X1_truth_tracking_crossvendor>=0.75": [bool(X1), fin(X1_auc)],
    "X2_converge_real_scatter_fake": [bool(X2), fin(mean_common), fin(mean_fake)],
    "X3_crossvendor_shared_confab_on_fakes": f"{n_xshare}/{len(fake)}",
    "PASS": bool(X1 and X2),
}
pathlib.Path(__file__).parent.joinpath("probe_crossvendor_results.json").write_text(json.dumps({"rows": rows, **out}, indent=2))
print("\n" + json.dumps(out, indent=2))
