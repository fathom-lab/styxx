#!/usr/bin/env python3
"""
topic_control_analysis.py — §8 analysis for the topic-overlap control preregistration
=====================================================================================

Reuses the LOCKED drift_axis_alignment (drift_axis_scorer.py, commit
79906b4) UNCHANGED. Adds:

  - topic-overlap manipulation-check measure (content-word-only embedding
    pairwise cosine; stopword list LOCKED below)
  - 2-way analysis (regime x topic-coupling on DAA) via stratified
    permutation test for the regime main effect
  - §6 bar / kill-gate evaluation

This file is committed BEFORE data (§8). Its hash is recorded in the
preregistration §8 at lock-time. Running it requires the 2x2 corpus,
which is collected only after the operator signs.

The drift_axis_alignment function is IMPORTED, not reimplemented — if
the locked scorer changes, this analysis changes with it, and the
parity test (tests/test_topic_control_parity.py) catches any drift.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import random
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Reuse the LOCKED scorer (79906b4) — do not reimplement.
from drift_axis_scorer import (  # noqa: E402
    drift_axis_alignment, get_provider, conversation_embeddings,
)


# ---------------------------------------------------------------------------
# LOCKED stopword list (§8) — frozen. Content-word topic-overlap strips these.
# Standard high-frequency English function words. Do not edit without a new
# preregistration lock-hash.
# ---------------------------------------------------------------------------
STOPWORDS = frozenset("""
a an the and or but if then else when while of to in on at by for with about
against between into through during before after above below from up down out
off over under again further is am are was were be been being have has had do
does did doing this that these those i you he she it we they me him her us them
my your his its our their mine yours not no nor so than too very can will just
as also which who whom what where why how all any both each few more most other
some such only own same s t don should now
""".split())

_WORD = re.compile(r"[A-Za-z']+")


def content_words(text: str) -> str:
    """Return the text reduced to lowercase content words (stopwords stripped)."""
    toks = [w.lower() for w in _WORD.findall(text)]
    kept = [w for w in toks if w not in STOPWORDS and len(w) > 1]
    return " ".join(kept)


def topic_overlap(transcript_path: Path, provider) -> float:
    """Manipulation-check measure: mean over turns of cosine between the
    two agents' CONTENT-WORD-ONLY embeddings, kth-of-A with kth-of-B.

    Distinct from DAA (trajectory direction). This measures instantaneous
    between-agent topic vocabulary overlap, independent of trajectory.
    """
    tx = json.loads(Path(transcript_path).read_text(encoding="utf-8"))
    role_a, role_b = tx["task"]["role_a"], tx["task"]["role_b"]
    a = [content_words(t["content"]) for t in tx["turns"] if t["sender"] == role_a]
    b = [content_words(t["content"]) for t in tx["turns"] if t["sender"] == role_b]
    n = min(len(a), len(b))
    if n < 3:
        return float("nan")
    # Filter empties (a turn that was all stopwords) by substituting a space
    a = [s if s.strip() else " " for s in a[:n]]
    b = [s if s.strip() else " " for s in b[:n]]
    ea = provider.embed(a)
    eb = provider.embed(b)
    sims = (ea * eb).sum(axis=1)  # both L2-normalized by the provider
    return float(np.mean(sims))


def daa_for_conv(transcript_path: Path, provider) -> float:
    """DAA via the LOCKED drift_axis_alignment + the locked loader path."""
    embs_a, embs_b = conversation_embeddings(Path(transcript_path), provider)
    return drift_axis_alignment(embs_a, embs_b)


def stratified_regime_permutation_p(
    daa_by_cell: dict[str, list[float]],
    n_resamples: int = 5000, seed: int = 1729,
) -> float:
    """Permutation test for the REGIME main effect on DAA, stratified by
    topic-coupling. Observed statistic = (mean DAA over cooperative cells)
    − (mean DAA over non-cooperative cells). Null: within each topic-
    coupling stratum, regime labels are exchangeable.

    Strata: 'shared' = {coop_shared, noncoop_shared}; 'independent' =
    {coop_independent, noncoop_independent}. Within each stratum we pool
    the two cells' values and randomly relabel which half is 'cooperative'.
    """
    coop_cells = ["coop_shared", "coop_independent"]
    noncoop_cells = ["noncoop_shared", "noncoop_independent"]

    def regime_effect(coop_vals, noncoop_vals):
        cv = [v for v in coop_vals if v == v]
        nv = [v for v in noncoop_vals if v == v]
        return statistics.fmean(cv) - statistics.fmean(nv)

    obs = regime_effect(
        daa_by_cell["coop_shared"] + daa_by_cell["coop_independent"],
        daa_by_cell["noncoop_shared"] + daa_by_cell["noncoop_independent"],
    )
    rng = random.Random(seed)
    strata = {
        "shared": ("coop_shared", "noncoop_shared"),
        "independent": ("coop_independent", "noncoop_independent"),
    }
    n_extreme = 0
    for _ in range(n_resamples):
        coop_perm, noncoop_perm = [], []
        for _name, (c, nc) in strata.items():
            pool = [v for v in (daa_by_cell[c] + daa_by_cell[nc]) if v == v]
            rng.shuffle(pool)
            half = len(daa_by_cell[c])  # preserve cell sizes
            coop_perm += pool[:half]
            noncoop_perm += pool[half:]
        perm_eff = regime_effect(coop_perm, noncoop_perm)
        if perm_eff >= obs:
            n_extreme += 1
    return (n_extreme + 1) / (n_resamples + 1)


def analyze(manifest_path: Path, output_dir: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cells = {c["cell"]: c for c in manifest["cells"]}

    results = {"preregistration": manifest.get("preregistration"),
               "drift_axis_scorer_lock": "79906b4", "providers": {}}

    for prov_name in ("openai", "bge"):
        print(f"\n=== provider {prov_name} ===", flush=True)
        provider = get_provider(prov_name)
        daa_by_cell, topic_by_cell = {}, {}
        for cell_name, cell in cells.items():
            daas, topics = [], []
            for conv in cell["conversations"]:
                tp = Path(conv["transcript_path"])
                if not tp.exists():
                    continue
                daas.append(daa_for_conv(tp, provider))
                topics.append(topic_overlap(tp, provider))
            daa_by_cell[cell_name] = daas
            topic_by_cell[cell_name] = topics
            med = statistics.median([d for d in daas if d == d]) if daas else float("nan")
            tmed = statistics.median([t for t in topics if t == t]) if topics else float("nan")
            print(f"  {cell_name:20s} DAA median={med:.3f}  topic-overlap median={tmed:.3f}  n={len(daas)}")

        def med(cell):
            v = [x for x in daa_by_cell[cell] if x == x]
            return statistics.median(v) if v else float("nan")
        def tmed(cell):
            v = [x for x in topic_by_cell[cell] if x == x]
            return statistics.median(v) if v else float("nan")

        # §6 bar
        p_regime = stratified_regime_permutation_p(daa_by_cell)
        within_shared_delta = med("coop_shared") - med("noncoop_shared")
        coop_indep_med = med("coop_independent")
        bar1 = p_regime < 0.01
        bar2 = within_shared_delta >= 0.15
        bar3 = coop_indep_med >= 0.50
        positive = bar1 and bar2 and bar3
        # kill-gate
        kill = (p_regime >= 0.05) or (coop_indep_med < 0.40)
        outcome = "POSITIVE" if positive else ("KILL_TOPIC_PROXY" if kill else "INTERMEDIATE")

        # manipulation check: shared strata topic-overlap > independent strata
        manip_shared = statistics.fmean([tmed("coop_shared"), tmed("noncoop_shared")])
        manip_indep = statistics.fmean([tmed("coop_independent"), tmed("noncoop_independent")])
        manip_ok = manip_shared > manip_indep

        results["providers"][prov_name] = {
            "daa_median_by_cell": {c: med(c) for c in cells},
            "topic_overlap_median_by_cell": {c: tmed(c) for c in cells},
            "regime_permutation_p": p_regime,
            "within_shared_regime_delta": within_shared_delta,
            "coop_independent_daa_median": coop_indep_med,
            "bar": {"p_regime_lt_0.01": bar1, "within_shared_delta_ge_0.15": bar2,
                    "coop_indep_ge_0.50": bar3, "all_pass": positive},
            "kill_gate_triggered": kill,
            "outcome": outcome,
            "manipulation_check": {"shared_topic_overlap": manip_shared,
                                   "independent_topic_overlap": manip_indep,
                                   "shared_gt_independent": manip_ok},
        }
        print(f"  -> regime p={p_regime:.4f}  within-shared Δ={within_shared_delta:+.3f}  "
              f"coop+indep DAA={coop_indep_med:.3f}  outcome={outcome}  manip_ok={manip_ok}")

    outcomes = {p: r["outcome"] for p, r in results["providers"].items()}
    manips = all(r["manipulation_check"]["shared_gt_independent"] for r in results["providers"].values())
    if not manips:
        results["combined_outcome"] = "INVALID_MANIPULATION_CHECK_FAILED"
    elif all(o == "POSITIVE" for o in outcomes.values()):
        results["combined_outcome"] = "POSITIVE_BOTH_PROVIDERS_drift_axis_survives_topic_control"
    elif any(o == "KILL_TOPIC_PROXY" for o in outcomes.values()):
        results["combined_outcome"] = "KILL_drift_axis_is_topic_proxy"
    else:
        results["combined_outcome"] = "INTERMEDIATE_OR_CONDITIONAL"

    output_dir.mkdir(parents=True, exist_ok=True)
    import datetime
    out = output_dir / f"topic_control_{datetime.date.today().isoformat()}.json"
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\ncombined: {results['combined_outcome']}\nsaved: {out}")
    return results


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="topic-control 2x2 analysis (§8)")
    p.add_argument("--manifest", type=Path,
                   default=Path("papers/cooperative-agent-regime/topic_control_manifest.json"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("papers/cooperative-agent-regime/results"))
    args = p.parse_args(argv)
    analyze(args.manifest, args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
