"""Definitive truth-vs-plausibility test: does a NEGATION-ROBUST truth direction emerge, and does it
generalize OOD? (follow-up to FINDING_truth_direction_deflation). Offline.

Augment the controlled set with negations (flipped labels) so the fitted direction MUST encode truth-VALUE
not surface co-occurrence. Then test:
  (1) held-out negation polarity (does it now handle 'not'?)
  (2) OOD natural statements (misconceptions familiar-false vs surprising unfamiliar-true)
If both pass -> a real, negation-robust, generalizing truth axis exists in these 3B models. If not -> they
encode plausibility, not robust truth.

  python scripts/negation_augmented_truth.py --extract --model {qwen,llama}
  python scripts/negation_augmented_truth.py
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from build_controlled_truthset import DOMAINS
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut

REPOS = {"qwen": ("Qwen/Qwen2.5-3B-Instruct", 19), "llama": ("meta-llama/Llama-3.2-3B-Instruct", 14)}
DDIR = ROOT / "benchmarks" / "data" / "deception"
NEG = DDIR / "controlled_truthset_negated.jsonl"
def apath(tag): return DDIR / f"_negaug_acts_{tag}.npz"
def load(p): return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def negate(stmt):
    for a, b in ((" was written by ", " was not written by "), (" is the ", " is not the "), (" is ", " is not ")):
        if a in stmt: return stmt.replace(a, b, 1)
    return None


def build_negated():
    rows = []
    for di, (tmpl, table) in enumerate(DOMAINS):
        ents = list(table); corrects = [table[e] for e in ents]
        for i, e in enumerate(ents):
            correct = corrects[i]; wrong = corrects[(i + 1) % len(ents)]
            if wrong == correct: wrong = corrects[(i + 2) % len(ents)]
            for ans, lab in ((correct, 0), (wrong, 1)):
                neg = negate(tmpl.format(e=e, a=ans))
                if neg:  # negation flips the label
                    rows.append({"statement": neg, "domain": di, "label_false": 1 - lab})
    with open(NEG, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    return rows


def extract(tag):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    repo, L = REPOS[tag]
    if not NEG.exists(): build_negated()
    rows = load(NEG)
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float16, device_map="cuda",
                                                 output_hidden_states=True).eval()
    acts = []
    for r in rows:
        ids = tok(r["statement"], return_tensors="pt").to("cuda")
        with torch.no_grad():
            hs = model(**ids).hidden_states
        acts.append(hs[L][0, -1, :].float().cpu().numpy())
    np.savez_compressed(apath(tag), acts=np.array(acts, np.float32),
                        y=np.array([r["label_false"] for r in rows]), dom=np.array([r["domain"] for r in rows]))
    print(f"[extract] {tag}: {np.array(acts).shape}"); del model; torch.cuda.empty_cache()


def massmean(Xtr, ytr, Xte):
    return Xte @ (Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0))


def analyze():
    neg = load(NEG)
    for tag, (repo, L) in REPOS.items():
        if not apath(tag).exists(): continue
        # positive controlled acts + labels + domains
        dc = np.load(DDIR / f"_truthset_acts_{tag}.npz"); Ap, yp, domp = dc["acts"][:, L, :], dc["y"], dc["dom"]
        dn = np.load(apath(tag)); An, yn, domn = dn["acts"], dn["y"], dn["dom"]
        # combined positive+negated, leave-one-DOMAIN-out (polarity is now IN training)
        X = np.vstack([Ap, An]); y = np.concatenate([yp, yn]); dom = np.concatenate([domp, domn])
        s = StandardScaler().fit(X); Xs = s.transform(X)
        aucs = []
        for tr, te in LeaveOneGroupOut().split(Xs, y, dom):
            a = roc_auc_score(y[te], massmean(Xs[tr], y[tr], Xs[te])); aucs.append(max(a, 1 - a))
        lodo = float(np.mean(aucs))
        # held-out NEGATION polarity within LODO already tested above (negations are in test folds);
        # report negation-only subset AUC: fit on positives, test on negations (pure polarity transfer)
        wsp = (Xs[:len(yp)][yp == 1].mean(0) - Xs[:len(yp)][yp == 0].mean(0))  # direction from positives only
        neg_auc = roc_auc_score(yn, s.transform(An) @ wsp); neg_auc = neg_auc  # NOT max: we want true direction
        # OOD natural statements (misconceptions vs surprising) with the AUGMENTED direction
        wa = Xs[y == 1].mean(0) - Xs[y == 0].mean(0)  # augmented direction (pos+neg)
        dd = np.load(DDIR / f"_diligence_acts_{tag}.npz"); Ad, yd, cat = dd["acts"], dd["y"], dd["cat"]
        mm = np.isin(cat, ["misconception", "surprising"])
        ood_auc = roc_auc_score(yd[mm], s.transform(Ad[mm]) @ wa)
        nn = np.isin(cat, ["neg_true", "neg_false"])
        negpol_aug = roc_auc_score(yd[nn], s.transform(Ad[nn]) @ wa)
        print(f"\n=== {tag} (L{L}) negation-AUGMENTED ===")
        print(f"  LODO AUC (pos+neg, polarity in train): {lodo:.3f}")
        print(f"  positives-only dir -> negations (polarity transfer, raw): {neg_auc:.3f}  ({'inverts' if neg_auc<0.4 else 'holds' if neg_auc>0.6 else 'flat'})")
        print(f"  AUGMENTED dir -> OOD misconceptions-vs-surprising: {ood_auc:.3f}  ({'TRUTH' if ood_auc>=0.65 else 'flat/plausibility'})")
        print(f"  AUGMENTED dir -> held-out negation polarity: {negpol_aug:.3f}  ({'handles' if negpol_aug>=0.65 else 'FAILS'})")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--extract", action="store_true")
    ap.add_argument("--model", choices=list(REPOS), default="qwen"); a = ap.parse_args()
    build_negated()
    if a.extract: extract(a.model)
    else: analyze()


if __name__ == "__main__":
    main()
