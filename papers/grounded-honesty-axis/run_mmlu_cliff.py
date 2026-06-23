"""MMLU cross-family competence cliff (PREREG_mmlu_dataset_confound_2026_06_23).

Tests the DATASET confound on the cross-family hallucination-cliff invariance (0.77 on TruthfulQA):
does it survive on a NON-adversarial, categorized benchmark (MMLU, 57 academic subjects)? Reuses the
validated run_local_cliff apparatus (free-gen + NLI same-answer judge + belief-coherence gate + per-
category cliff) — only the dataset loader changes. Choices are shown so MC-derived questions are
answerable; the model still free-generates and is NLI-judged against the gold answer (uniform across
subjects). Cross-FAMILY only (open weights, no gpt). Outputs tagged _mmlu (no clobber). No API key.

Usage:  python run_mmlu_cliff.py --dryload     # CPU: build+print items, no model
        python run_mmlu_cliff.py --smoke        # 1 family, few items (GPU)
        python run_mmlu_cliff.py                 # full: 3 families x ~800 items (GPU)
"""
from __future__ import annotations
import argparse, json, random, itertools
from datasets import load_dataset
import run_local_cliff as L
import run_pregeneration_gate as G

K_PER_SUBJECT = 14   # 57 subjects x 14 = 798 items (~ TruthfulQA 790)


def load_mmlu_items(k_per_subject=K_PER_SUBJECT, seed=0):
    ds = load_dataset("cais/mmlu", "all", split="test")
    by_sub = {}
    for r in ds:
        by_sub.setdefault(r["subject"], []).append(r)
    rng = random.Random(seed)
    items = []
    for sub in sorted(by_sub):
        rows = list(by_sub[sub]); rng.shuffle(rows)
        for r in rows[:k_per_subject]:
            ch = list(r["choices"]); ans = int(r["answer"])
            if not (0 <= ans < len(ch)):
                continue
            best = ch[ans]
            distractors = [c for i, c in enumerate(ch) if i != ans]
            worst = rng.choice(distractors)
            q = f"{r['question']}\n\nChoices: {'; '.join(str(c) for c in ch)}\n\nWhich is correct?"
            items.append((q, str(best), str(worst), sub))
    return items


def rankdata(a):
    idx = sorted(range(len(a)), key=lambda i: a[i]); rr = [0.0]*len(a); i = 0
    while i < len(a):
        j = i
        while j+1 < len(a) and a[idx[j+1]] == a[idx[i]]: j += 1
        avg = (i+j)/2.0+1
        for k in range(i, j+1): rr[idx[k]] = avg
        i = j+1
    return rr
def pearson(x, y):
    n = len(x); mx = sum(x)/n; my = sum(y)/n
    cov = sum((a-mx)*(b-my) for a, b in zip(x, y)); vx = sum((a-mx)**2 for a in x); vy = sum((b-my)**2 for b in y)
    return float("nan") if vx == 0 or vy == 0 else cov/((vx*vy)**0.5)
def sp(x, y): return pearson(rankdata(x), rankdata(y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dryload", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--k", type=int, default=K_PER_SUBJECT)
    a = ap.parse_args()

    items = load_mmlu_items(k_per_subject=(2 if a.smoke else a.k))
    subs = sorted({c for *_, c in items})
    print(f"MMLU items: {len(items)} over {len(subs)} subjects (k={2 if a.smoke else a.k}/subject)", flush=True)
    if a.dryload:
        for it in items[:3]:
            print("\n--- item ---")
            print("Q:", it[0][:200].replace("\n", " ⏎ "))
            print("best:", it[1], "| worst:", it[2], "| subject:", it[3])
        return 0

    # retag outputs; match the generation knob used in the gen-matched run (max_new=32)
    L.MAX_NEW = 32
    _orig = L._slug
    L._slug = lambda m: _orig(m) + "_mmlu"

    nli = L.NLIJudge(L.NLI_MODEL, device="cuda")
    L.calibrate(nli)

    families = [f for f in L.FAMILIES if f[0] == L.REFERENCE] if a.smoke else L.FAMILIES
    per_model, k_rates = {}, {}
    for name, hf_id in families:
        gpath = L.HERE / f"crossfamily_gate_{L._slug(name)}.json"   # _slug adds _mmlu
        if gpath.exists():   # resume: skip families already scored (gate on disk)
            gate = json.loads(gpath.read_text(encoding="utf-8"))
            print(f"  >> {name}: gate exists — skipping generation (resume)", flush=True)
        else:
            gate = L._score_model(name, hf_id, items, nli)
        per_model[name] = gate.get("category_competence_cliff_map", {})
        k_rates[name] = L._k_rate(gate)
        print(f"  >> {name}: K={k_rates[name]:.3f}, {len(per_model[name])} subjects", flush=True)

    valid = [nm for nm in per_model if k_rates.get(nm, 0) >= L.K_BAR]
    out = {"benchmark": "MMLU (cais/mmlu, all, test)", "k_per_subject": a.k,
           "n_items": len(items), "n_subjects": len(subs), "families": list(per_model),
           "k_rates": k_rates, "per_family_cliff_map": per_model}
    if len(valid) >= 2:
        for sig in ("ungated_hallucination_rate", "refusal_rate"):
            vals = []
            for x, y in itertools.combinations(valid, 2):
                shared = [d for d in per_model[x] if d in per_model[y]]
                vals.append(sp([per_model[x][d][sig] for d in shared], [per_model[y][d][sig] for d in shared]))
            out[f"crossfamily_{sig}_spearman_mean"] = sum(vals)/len(vals)
            out[f"crossfamily_{sig}_spearman_pairs"] = vals
            print(f"cross-family {sig}: mean Spearman {out[f'crossfamily_{sig}_spearman_mean']:.4f}  pairs {[round(v,3) for v in vals]}")
    (L.HERE / "mmlu_cliff_result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("written: mmlu_cliff_result.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
