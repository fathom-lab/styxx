"""Grounding-holds-where-text-is-silent probe on the verified-silent controlled truth-set
(PREREG_controlled_truthset_probe). Reader-model last-token activations, leave-one-domain-out.

  python scripts/truthset_probe.py --extract --model {qwen,llama}
  python scripts/truthset_probe.py
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

REPOS = {"qwen": "Qwen/Qwen2.5-3B-Instruct", "llama": "meta-llama/Llama-3.2-3B-Instruct"}
CORP = ROOT / "benchmarks" / "data" / "deception" / "controlled_truthset.jsonl"
DDIR = ROOT / "benchmarks" / "data" / "deception"
def apath(tag): return DDIR / f"_truthset_acts_{tag}.npz"
def load(): return [json.loads(l) for l in CORP.read_text(encoding="utf-8").splitlines() if l.strip()]


def extract(tag):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    repo = REPOS[tag]; rows = load()
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float16, device_map="cuda",
                                                 output_hidden_states=True).eval()
    acts = []
    for r in rows:
        ids = tok(r["statement"], return_tensors="pt").to("cuda")
        with torch.no_grad():
            hs = model(**ids).hidden_states
        acts.append(np.stack([h[0, -1, :].float().cpu().numpy() for h in hs]))
    np.savez_compressed(apath(tag), acts=np.array(acts, np.float32),
                        y=np.array([r["label_false"] for r in rows]),
                        dom=np.array([r["domain"] for r in rows]))
    print(f"[extract] {tag}: {np.array(acts).shape}"); del model; torch.cuda.empty_cache()


def lodo(X, y, dom):
    aucs = []
    for tr, te in LeaveOneGroupOut().split(X, y, dom):
        if len(np.unique(y[te])) < 2: continue
        s = StandardScaler().fit(X[tr]); c = LogisticRegression(max_iter=2000).fit(s.transform(X[tr]), y[tr])
        aucs.append(roc_auc_score(y[te], c.predict_proba(s.transform(X[te]))[:, 1]))
    a = float(np.mean(aucs)); return max(a, 1 - a)


def strat_auc(X, y, seed=0):
    aucs = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        s = StandardScaler().fit(X[tr]); c = LogisticRegression(max_iter=2000).fit(s.transform(X[tr]), y[tr])
        aucs.append(roc_auc_score(y[te], c.predict_proba(s.transform(X[te]))[:, 1]))
    return float(np.mean(aucs))


def analyze():
    rows = load(); txt = [r["statement"] for r in rows]
    Xbow = TfidfVectorizer().fit_transform(txt).toarray()
    out = []
    for tag in REPOS:
        if not apath(tag).exists(): continue
        d = np.load(apath(tag)); acts, y, dom = d["acts"], d["y"], d["dom"]
        bow = lodo(Xbow, y, dom)
        # select layer by stratified-CV (BLIND to LODO), then report LODO at that layer
        L = int(np.argmax([strat_auc(acts[:, k, :], y) for k in range(acts.shape[1])]))
        A = acts[:, L, :]
        act = lodo(A, y, dom)
        rng = np.random.default_rng(0)
        shuf = lodo(A, rng.permutation(y), dom)
        pca = lodo(PCA(50, random_state=0).fit_transform(StandardScaler().fit_transform(A)), y, dom)
        v = ("GROUNDING-HOLDS" if (act >= 0.75 and act - bow >= 0.15 and shuf < 0.62 and pca >= 0.70)
             else ("HONEST-NULL" if act < 0.65 else "PARTIAL"))
        out.append({"model": tag, "layer": L, "bow_lodo": bow, "act_lodo": act, "act_minus_bow": act - bow,
                    "shuffle_lodo": shuf, "pca50_lodo": pca, "verdict": v})
        print(f"\n=== {tag} (layer {L}/{acts.shape[1]-1}) ===")
        print(f"  TEXT  (BoW)  leave-one-domain-out : {bow:.3f}  (silence baseline)")
        print(f"  ACT   probe  leave-one-domain-out : {act:.3f}   (act - text {act-bow:+.3f})")
        print(f"  controls: shuffle {shuf:.3f} | pca50 {pca:.3f}")
        print(f"  >>> {v}")
    if out:
        rob = {o["verdict"] for o in out} == {"GROUNDING-HOLDS"} and len(out) >= 2
        print(f"\n###### {[(o['model'], o['verdict']) for o in out]}  | ROBUST: {rob}")
        (DDIR / "_truthset_probe_result.json").write_text(json.dumps(out, indent=2))
        print("wrote benchmarks/data/deception/_truthset_probe_result.json")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--extract", action="store_true")
    ap.add_argument("--model", choices=list(REPOS), default="qwen"); a = ap.parse_args()
    if a.extract: extract(a.model)
    else: analyze()


if __name__ == "__main__":
    main()
