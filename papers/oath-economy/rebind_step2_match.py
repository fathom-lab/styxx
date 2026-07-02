"""M1 rung 3, step 2 (PREREG_rebinding §2): match the card's 22 in-scope 1B-Instruct claims against the
receipts dataset aggregates. EXPLICIT per-benchmark mapping (auditable, no fuzzy matching — v1's fuzz
produced 3 phantom mismatches that self-audit removed: Portugese label spelling, an unreached IFEval
composite branch, and the micro_avg->average tag alias). Frozen grades: MATCH |delta|<=0.15 / VALUE-MISMATCH /
ABSENT."""
import json
from collections import Counter

import pandas as pd
from huggingface_hub import hf_hub_download

TOL = 0.15
# card benchmark -> (receipts label, receipts metric_tag) ; None = absent from receipts
MAP = {
    "MMLU":                ("MMLU", "macro_avg/acc"),
    "Open-rewrite eval":   ("Open-rewrite eval", "average/rougeL"),   # card tag 'micro_avg/rougeL' = receipts 'average/rougeL' (alias)
    "TLDR9+ (test)":       None,                                      # no TLDR9+ data anywhere in the receipts
    "IFEval":              ("__COMPOSITE_ABSENT__", "card metric = Avg(Prompt/Instruction acc, Loose/Strict); receipts carry ONLY prompt_acc (55.3/51.0) — instruction-level components absent, composite not recomputable"),
    "GSM8K (CoT)":         ("GSM8K", "em_maj1@1"),
    "MATH (CoT)":          ("__CONFIG_CONFLICT__", "card states 0-shot final_em; receipts label is 'MATH (4-shot, CoT)' with em=30.4 — no 0-shot record exists; card and its own receipts disagree on config"),
    "ARC-C":               ("ARC-C", "acc"),
    "GPQA":                ("GPQA", "acc"),
    "Hellaswag":           ("Hellaswag", "acc"),
    "BFCL V2":             ("BFCL V2", "macro_avg/acc"),               # card 'acc' ambiguous; both receipt acc-aggregates reported
    "Nexus":               ("Nexus", "macro_avg/acc"),
    "InfiniteBench/En.QA": ("InfiniteBench", "longbookqa/f1"),
    "InfiniteBench/En.MC": ("InfiniteBench", "longbook_choice/acc"),
    "NIH/Multi-needle":    ("NIH/Multi-needle", "macro_avg/recall"),
    "MGSM (CoT)":          ("MGSM", "macro_avg/em"),
}
LANG_LABEL = {"Portuguese": "MMLU (Portugese)", "Spanish": "MMLU (Spanish)", "Italian": "MMLU (Italian)",
              "German": "MMLU (German)", "French": "MMLU (French)", "Hindi": "MMLU (Hindi)", "Thai": "MMLU (Thai)"}

claims = json.load(open("rebind_card_claims.json"))
in_scope = [c for c in claims if c["column"] == "Llama 3.2 1B bf16"
            or (c["column"] == "Llama 3.2 1B" and c.get("language"))]
p = hf_hub_download("meta-llama/Llama-3.2-1B-Instruct-evals",
                    "Llama-3.2-1B-Instruct-evals/Details_metrics_details_2024-09-23T17-23-22.207810.parquet.gzip",
                    repo_type="dataset")
df = pd.read_parquet(p)

def receipt(label, tag):
    sub = df[(df.benchmark_label == label) & (df.metric_tag == tag)]
    return float(sub.metric_value_computed.iloc[0]) if len(sub) else None

rows = []
for c in in_scope:
    lang = c.get("language", "")
    if lang:
        label, tag = LANG_LABEL[lang], "macro_avg/acc"
        note = ""
    else:
        m = MAP[c["benchmark"]]
        if m is None:
            rows.append({"benchmark": c["benchmark"], "language": "", "card": c["value"], "receipt": None,
                         "grade": "ABSENT", "note": "no data for this benchmark anywhere in the receipts dataset"})
            continue
        label, tag = m
        if label.startswith("__"):
            rows.append({"benchmark": c["benchmark"], "language": "", "card": c["value"], "receipt": None,
                         "grade": "ABSENT", "note": tag})
            continue
        note = ""
    v = receipt(label, tag)
    grade = "MATCH" if (v is not None and abs(v - c["value"]) <= TOL) else ("VALUE-MISMATCH" if v is not None else "ABSENT")
    if c["benchmark"] == "BFCL V2" and grade == "VALUE-MISMATCH":
        note = f"card 25.7 matches NO receipt aggregate (macro_avg/acc {receipt(label,'macro_avg/acc')}, average/acc {receipt(label,'average/acc')})"
    if c["benchmark"] == "Open-rewrite eval":
        note = "card tag micro_avg/rougeL == receipts average/rougeL (naming alias)"
    rows.append({"benchmark": c["benchmark"], "language": lang, "card": c["value"], "receipt": v,
                 "grade": grade, "note": note,
                 "delta": None if v is None else round(v - c["value"], 2)})

out = pd.DataFrame(rows)
print(out.to_string(index=False))
tally = Counter(r["grade"] for r in rows); n = len(rows); match = tally["MATCH"]
verdict = ("REBOUND (OATH-HELD-BY-PROXY)" if match / n >= 0.90
           else "RECEIPT-MISMATCH" if tally["VALUE-MISMATCH"] / n > 0.10 else "PARTIAL")
print(f"\nTALLY: {dict(tally)} -> {match}/{n} = {100*match/n:.0f}% MATCH | VERDICT: {verdict}")
json.dump({"rows": rows, "tally": dict(tally), "verdict": verdict, "tolerance": TOL,
           "receipts_dataset": "meta-llama/Llama-3.2-1B-Instruct-evals (Details_metrics_details 2024-09-23)",
           "self_audit_note": "matcher v1 (fuzzy) produced 3 phantom mismatches, removed before publication: "
                              "Portugese label spelling, unreached IFEval composite branch, micro_avg->average alias"},
          open("rebind_result.json", "w"), indent=1)
