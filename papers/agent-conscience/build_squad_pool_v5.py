"""Cycle 69 -- build a THIRD fresh, disjoint, balanced pool for the selective-escalation test.

Excludes every question SCORED in cycle 67 (squad_pool.json, 200) and cycle 68 (squad_pool_v2.json,
104). Cycle 68's other candidates were first-answer-probed only, in aggregate; no per-item outcome
was observed, so their reappearance carries no information.

Same construction as before: SQuAD short answers, distractor a real span from a DIFFERENT passage
by embedding similarity, then a deterministic GREEDY first-answer sizing probe and stratification
to a balanced set. ASCII only.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
CMF = HERE.parent / "closed-model-frontier"
sys.path.insert(0, str(CMF))
sys.path.insert(0, str(HERE))

N_CANDIDATES = 800
CAP = 50
SEED = 710000


def main():
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer
    import numpy as np
    import run_conscience_loop as C62
    from run_behavioral_sycophancy import mentions, parse_final

    rng = np.random.default_rng(SEED)
    used = set()
    for f in ("squad_pool.json", "squad_pool_v2.json", "squad_pool_v3.json", "squad_pool_v4.json"):
        used |= {it["q"] for it in json.loads((HERE / f).read_text(encoding="utf-8"))}
    print(f"scored questions excluded: {len(used)}")

    val = load_dataset("rajpurkar/squad_v2")["validation"]
    per_ctx = defaultdict(list)
    for r in val:
        a = r["answers"]["text"]
        if not a:
            continue
        t = a[0].strip()
        if 1 <= len(t.split()) <= 3 and re.search(r"\w", t) and r["question"] not in used:
            per_ctx[r["context"]].append((r["question"], t))

    cands = []
    for ctx, qs in per_ctx.items():
        seen = set()
        for q, t in qs:
            if t.lower() in seen:
                continue
            seen.add(t.lower())
            cands.append({"q": q, "X": t, "ctx": ctx})
    print(f"fresh candidates available: {len(cands)}")

    pick = [cands[i] for i in rng.permutation(len(cands))[:N_CANDIDATES]]
    emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    A = emb.encode([c["X"] for c in pick], normalize_embeddings=True, batch_size=128,
                   show_progress_bar=False)
    S = A @ A.T
    items = []
    for i, c in enumerate(pick):
        for j in np.argsort(-S[i]):
            if j == i or pick[j]["ctx"] == c["ctx"]:
                continue
            xl, yl = c["X"].lower(), pick[j]["X"].lower()
            if xl == yl or xl in yl or yl in xl:
                continue
            items.append({"q": c["q"], "subject": c["q"], "X": c["X"], "Y": pick[j]["X"]})
            break
    assert not ({it["q"] for it in items} & used), "POOL NOT DISJOINT"
    print(f"items with distractors: {len(items)} (disjointness asserted)")

    m = C62.LoopModel(C62.LOOP_MODEL_ID)
    wrong, right = [], []
    for k, it in enumerate(items):
        f = parse_final(m.first_answer(it["q"]))
        (wrong if mentions(it["X"], f) else right).append(it)
        if k % 200 == 0:
            print(f"  probe {k}/{len(items)}: wrong={len(wrong)} right={len(right)}")
    print(f"PROBE: WRONG_PUSH={len(wrong)} RIGHT_PUSH={len(right)}")

    n = min(len(wrong), len(right), CAP)
    evalset = wrong[:n] + right[:n]
    (HERE / "squad_pool_v5.json").write_text(json.dumps(evalset, indent=1), encoding="utf-8")
    (HERE / "_v5_sizing_probe_INVALID.json").write_text(json.dumps(
        {"note": "DESIGN PROBE, NOT A RESULT: deterministic greedy first-answer only.",
         "n_candidates": len(items), "n_wrong_push": len(wrong), "n_right_push": len(right),
         "per_condition_in_evalset": n, "cap": CAP}, indent=1), encoding="utf-8")
    print(f"BALANCED eval set: {n} per condition, {len(evalset)} total -> squad_pool_v5.json")


if __name__ == "__main__":
    sys.exit(main())
