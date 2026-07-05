"""Binding census v1 (PREREG_census_v1). Generalizes rung 3's explicit-mapping matcher across arms.
Per arm: FIDELITY grades (MATCH/VALUE-MISMATCH/ABSENT) + SELECTION stats. No fuzzy matching."""
import json
import re
import sys
from collections import Counter

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

TOL = 0.15
api = HfApi()

def load_receipts(repo):
    files = api.list_repo_files(repo, repo_type="dataset")
    f = [x for x in files if "metrics_details" in x][0]
    return pd.read_parquet(hf_hub_download(repo, f, repo_type="dataset"))

def receipt(df, label, tag):
    sub = df[(df.benchmark_label == label) & (df.metric_tag == tag)]
    return float(sub.metric_value_computed.iloc[0]) if len(sub) else None

# card benchmark -> (receipts label candidates, metric_tag). Labels verified against each dataset's real set.
MAP = {
    "MMLU":                (["MMLU"], "macro_avg/acc"),
    "Open-rewrite eval":   (["Open-rewrite eval"], "average/rougeL"),
    "TLDR9+ (test)":       (None, "no TLDR9+ data in any Llama receipts dataset (checked per arm)"),
    "GSM8K (CoT)":         (["GSM8K"], "em_maj1@1"),
    "ARC-C":               (["ARC-C"], "acc"),
    "GPQA":                (["GPQA"], "acc"),
    "Hellaswag":           (["Hellaswag"], "acc"),
    "BFCL V2":             (["BFCL V2"], "macro_avg/acc"),
    "Nexus":               (["Nexus"], "macro_avg/acc"),
    "InfiniteBench/En.QA": (["InfiniteBench"], "longbookqa/f1"),
    "InfiniteBench/En.MC": (["InfiniteBench"], "longbook_choice/acc"),
    "NIH/Multi-needle":    (["NIH/Multi-needle"], "macro_avg/recall"),
    "MGSM (CoT)":          (["MGSM"], "macro_avg/em"),
}
LANGS = {"Portuguese": "MMLU (Portugese)", "Spanish": "MMLU (Spanish)", "Italian": "MMLU (Italian)",
         "German": "MMLU (German)", "French": "MMLU (French)", "Hindi": "MMLU (Hindi)", "Thai": "MMLU (Thai)"}

def grade_arm(claims, df, arm_name):
    labels = set(df.benchmark_label.unique())
    rows = []
    for c in claims:
        bench, lang, card_v = c["benchmark"], c.get("language", ""), c["value"]
        note, v = "", None
        if lang:
            label = LANGS[lang]
            v = receipt(df, label, "macro_avg/acc") if label in labels else None
            grade = "MATCH" if v is not None and abs(v - card_v) <= TOL else ("VALUE-MISMATCH" if v is not None else "ABSENT")
        elif bench == "IFEval":
            # dynamic composite: need prompt+instruction level acc x loose/strict (4 components)
            vals = []
            for lab in ("IFEval Loose", "IFEval Strict"):
                sub = df[df.benchmark_label == lab]
                for t in sub.metric_tag.unique():
                    if re.fullmatch(r"(prompt|instruction)_?(level_)?acc", t):
                        vals.append(float(sub[sub.metric_tag == t].metric_value_computed.iloc[0]))
            if len(vals) == 4:
                v = round(sum(vals) / 4, 1)
                grade = "MATCH" if abs(v - card_v) <= TOL else "VALUE-MISMATCH"
                note = "composite avg of 4 receipt components"
            else:
                grade, note = "ABSENT", f"composite unrecomputable: only {len(vals)}/4 components in receipts"
        elif bench == "MATH (CoT)":
            math_labels = [l for l in labels if l.startswith("MATH")]
            if not math_labels:
                grade, note = "ABSENT", "no MATH data"
            elif "0-shot" in " ".join(math_labels) or "MATH" in labels:
                v = receipt(df, "MATH" if "MATH" in labels else math_labels[0], "final_em")
                grade = "MATCH" if v is not None and abs(v - card_v) <= TOL else ("VALUE-MISMATCH" if v is not None else "ABSENT")
            else:
                em = receipt(df, math_labels[0], "em")
                grade, note = "ABSENT", f"config-conflict: card 0-shot vs receipts {math_labels[0]!r} (em={em})"
        else:
            m = MAP.get(bench)
            if m is None or m[0] is None:
                grade, note = "ABSENT", (m[1] if m else f"benchmark {bench!r} unmapped")
            else:
                cands, tag = m
                label = next((l for l in cands if l in labels), None)
                if label is None:
                    grade, note = "ABSENT", f"label {cands} not in receipts (labels present: {sorted(labels)[:6]}...)"
                else:
                    v = receipt(df, label, tag)
                    if v is None:
                        grade, note = "ABSENT", f"tag {tag!r} not under {label!r}"
                    else:
                        grade = "MATCH" if abs(v - card_v) <= TOL else "VALUE-MISMATCH"
                        if grade == "VALUE-MISMATCH":
                            note = "verify mapping manually before reporting (prereg hazard rule)"
        rows.append({"arm": arm_name, "benchmark": bench, "language": lang, "card": card_v,
                     "receipt": v, "grade": grade, "note": note,
                     "delta": None if v is None else round(v - card_v, 2)})
    # SELECTION: published fraction + flattering check (macro_avg vs average pairs differing > TOL)
    n_receipt_rows = len(df)
    flat = {"flattering": 0, "unflattering": 0, "neutral": 0}
    for r in rows:
        if r["receipt"] is None: continue
        b = [x for x in (MAP.get(r["benchmark"]) or [None])[0] or [] if x in labels]
        if not b: continue
        sub = df[df.benchmark_label == b[0]]
        fam = [t for t in sub.metric_tag.unique() if t in ("macro_avg/acc", "average/acc", "macro_avg/em", "average/em", "macro_avg/rougeL", "average/rougeL")]
        vals = sorted({float(sub[sub.metric_tag == t].metric_value_computed.iloc[0]) for t in fam})
        if len(vals) >= 2 and vals[-1] - vals[0] > TOL:
            if abs(r["card"] - vals[-1]) <= TOL: flat["flattering"] += 1
            elif abs(r["card"] - vals[0]) <= TOL: flat["unflattering"] += 1
            else: flat["neutral"] += 1
    return rows, {"receipt_rows": n_receipt_rows, "published_in_scope": len(rows),
                  "published_fraction_pct": round(100 * len(rows) / n_receipt_rows, 1),
                  "flattering_check": flat}

if __name__ == "__main__":
    claims = json.load(open("rebind_card_claims.json"))
    arms = {
        "Llama-3.2-3B-Instruct": {
            "claims": [c for c in claims if c["column"] == "Llama 3.2 3B bf16"
                       or (c["column"] == "Llama 3.2 3B" and c.get("language"))],
            "receipts": "meta-llama/Llama-3.2-3B-Instruct-evals"},
        "Llama-3.1-8B-Instruct (cross-cited on the 3.2 card)": {
            "claims": [c for c in claims if c["column"] == "Llama 3.1 8B" and (c.get("language") or
                       c.get("shots") or c.get("metric"))],
            "receipts": "meta-llama/Llama-3.1-8B-Instruct-evals"},
    }
    all_rows, selection = [], {}
    for name, cfg in arms.items():
        print(f"\n### ARM: {name} — {len(cfg['claims'])} in-scope claims")
        df = load_receipts(cfg["receipts"])
        rows, sel = grade_arm(cfg["claims"], df, name)
        all_rows += rows
        selection[name] = sel
        out = pd.DataFrame(rows)
        print(out[["benchmark", "language", "card", "receipt", "grade", "delta"]].to_string(index=False))
        t = Counter(r["grade"] for r in rows); n = len(rows)
        print(f"tally: {dict(t)} -> {t['MATCH']}/{n} = {100*t['MATCH']/n:.0f}% MATCH | selection: {sel}")
    json.dump({"arms": {k: v for k, v in selection.items()}, "rows": all_rows, "tolerance": TOL},
              open("census_v1_results.json", "w"), indent=1)
    print("\n-> census_v1_results.json")
