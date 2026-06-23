"""Clean open<->closed competence-cliff invariance under a MATCHED judge.

Compares per-domain cliff signals (ungated_hallucination_rate, refusal_rate) across:
  - open <-> open      : 3 open families, all NLI-judged          (sanity: should reproduce ~0.77 / ~0.43)
  - open <-> closed OLD: open (NLI) vs gpt-4o-mini (LLM-judged)   (confounded: ~0.23 / ~0.52)
  - open <-> closed NEW: open (NLI) vs gpt-4o-mini (NLI re-judge) (MATCHED -> judge-confound-free)

All maps are the gate's `category_competence_cliff_map`. The OLD gpt-4o-mini map is rebuilt
here from the committed LLM-judged benchmark via the identical gate, into a scratch file so the
shipped artifact is untouched.

No API key. Run AFTER run_xvendor_matched.py has produced xvendor_gpt4omini_nli_gate.json.
"""
from __future__ import annotations

import json
import itertools
from pathlib import Path

import run_pregeneration_gate as G

HERE = Path(__file__).resolve().parent

OPEN_GATES = {
    "Qwen2.5-3B": HERE / "crossfamily_gate_Qwen2_5_3B_Instruct.json",
    "Llama-3.2-3B": HERE / "crossfamily_gate_Llama_3_2_3B_Instruct.json",
    "gemma-2-2b": HERE / "crossfamily_gate_gemma_2_2b_it.json",
}
GPT_MATCHED_GATE = HERE / "xvendor_gpt4omini_nli_gate.json"        # NLI re-judge (matched)
GPT_OLD_BENCH = HERE / "truthfulqa_benchmark_result.json"          # gpt-4o-mini, LLM-judged
GPT_OLD_GATE = HERE / "_xvendor_gpt4omini_llm_gate_scratch.json"   # rebuilt OLD map (scratch)

SIGNALS = ["ungated_hallucination_rate", "refusal_rate"]


def rankdata(a):
    idx = sorted(range(len(a)), key=lambda i: a[i])
    ranks = [0.0] * len(a)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[idx[j + 1]] == a[idx[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[idx[k]] = avg
        i = j + 1
    return ranks


def pearson(x, y):
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    vx = sum((a - mx) ** 2 for a in x)
    vy = sum((b - my) ** 2 for b in y)
    if vx == 0 or vy == 0:
        return float("nan")
    return cov / ((vx * vy) ** 0.5)


def spearman(x, y):
    return pearson(rankdata(x), rankdata(y))


def load_map(gate_path):
    g = json.load(open(gate_path, encoding="utf-8"))
    return g["category_competence_cliff_map"]


def pair_spearman(mapA, mapB, sig):
    shared = [d for d in mapA if d in mapB]
    xa = [mapA[d][sig] for d in shared]
    xb = [mapB[d][sig] for d in shared]
    return spearman(xa, xb), len(shared)


def build_old_gpt_map():
    """Rebuild the LLM-judged gpt-4o-mini cliff map via the identical gate (scratch output)."""
    G.BENCHMARK_RECEIPT = GPT_OLD_BENCH
    G.RECEIPT = GPT_OLD_GATE
    G.main()
    return load_map(GPT_OLD_GATE)


def main():
    open_maps = {k: load_map(v) for k, v in OPEN_GATES.items()}
    open_names = list(open_maps)

    print("rebuilding OLD (LLM-judged) gpt-4o-mini cliff map ...", flush=True)
    gpt_old = build_old_gpt_map()
    gpt_new = load_map(GPT_MATCHED_GATE)

    print("\n" + "=" * 78)
    print("PER-DOMAIN COMPETENCE-CLIFF INVARIANCE (mean pairwise Spearman over shared domains)")
    print("=" * 78)
    header = f"{'comparison':<34}" + "".join(f"{s.split('_')[0][:5]:>10}" for s in SIGNALS) + f"{'n_dom':>8}"
    print(header)
    print("-" * 78)

    rows = {}

    # open <-> open
    for sig in SIGNALS:
        vals, ns = [], []
        for a, b in itertools.combinations(open_names, 2):
            s, n = pair_spearman(open_maps[a], open_maps[b], sig)
            vals.append(s); ns.append(n)
        rows.setdefault("open<->open (all NLI)", {})[sig] = (sum(vals) / len(vals), min(ns))

    # open <-> closed OLD (LLM-judged gpt)
    for sig in SIGNALS:
        vals, ns = [], []
        for a in open_names:
            s, n = pair_spearman(open_maps[a], gpt_old, sig)
            vals.append(s); ns.append(n)
        rows.setdefault("open<->closed OLD (LLM judge)", {})[sig] = (sum(vals) / len(vals), min(ns))

    # open <-> closed NEW (NLI re-judge gpt) -- MATCHED
    for sig in SIGNALS:
        vals, ns = [], []
        for a in open_names:
            s, n = pair_spearman(open_maps[a], gpt_new, sig)
            vals.append(s); ns.append(n)
        rows.setdefault("open<->closed NEW (NLI matched)", {})[sig] = (sum(vals) / len(vals), min(ns))

    order = ["open<->open (all NLI)", "open<->closed OLD (LLM judge)", "open<->closed NEW (NLI matched)"]
    summary = {}
    for label in order:
        cells = ""
        nmin = None
        for sig in SIGNALS:
            m, n = rows[label][sig]
            cells += f"{m:>10.3f}"
            nmin = n if nmin is None else min(nmin, n)
            summary.setdefault(label, {})[sig] = round(m, 4)
        print(f"{label:<34}{cells}{nmin:>8}")

    print("-" * 78)
    print("READ: if NEW (matched) hallucination Spearman jumps toward the open<->open 0.77 and the")
    print("refusal/hallucination ORDERING stops inverting, the OLD 0.23 was a judge artifact (confound")
    print("CONFIRMED). If NEW stays low, the open<->closed divergence is at least partly REAL.")

    out = HERE / "xvendor_matched_invariance_result.json"
    out.write_text(json.dumps({
        "comparison": summary,
        "signals": SIGNALS,
        "open_families": open_names,
        "gpt_matched_gate": str(GPT_MATCHED_GATE.name),
        "gpt_old_bench": str(GPT_OLD_BENCH.name),
        "note": "open families and gpt-matched both NLI-judged (DeBERTa-v3-base-mnli); gpt-old LLM-judged.",
    }, indent=2), encoding="utf-8")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
