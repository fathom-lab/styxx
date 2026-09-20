"""BENCH-1 scoring: the instrument and a similarity baseline against the oracle's labels.

Prereg: PREREG_bench1_pr_claim_benchmark_2026_09_17.md.  Ground truth is `bench1_oracle.py` on the
LIVE diff; the corpus verdicts are carried for reference and never used as truth.  The instrument
is read-only: it is run on the same live diff, one claim sentence at a time, exactly as the
`--pr` door would read it.

The baseline is given every advantage the prereg allows: its decision threshold is chosen AFTER
seeing the labels, by sweeping every threshold and keeping the one with the best F1.  That is an
upper bound no deployable baseline could reach, and it is reported as such.

    python bench1_score.py            # writes bench1_dataset.jsonl and bench1_scores.json
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import styxx.diffgate as dg            # noqa: E402  read-only: never modified by this cycle
import bench1_oracle as oracle         # noqa: E402

POP = HERE / "bench1_population.json"
DIFFS = Path("/tmp/bench1_diffs")
FETCH = Path("/tmp/bench1_fetch.json")
DATASET = HERE / "bench1_dataset.jsonl"
SCORES = HERE / "bench1_scores.json"
SCORED_KINDS = ("files_changed_count", "only_touches", "symbol_added")
POPULATION_PRS = 71016


def similarity(claim: str, diff: str) -> float:
    """The baseline's family: token overlap between the claim sentence and the diff text.

    Deliberately simple and deliberately not ours — this is the shape of method the published
    heuristic baselines use, and it is here so the instrument is not reported alone.
    """
    tok = lambda s: set(re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", s.lower()))
    c, d = tok(claim), tok(diff[:200_000])
    if not c:
        return 0.0
    return len(c & d) / len(c)


def metrics(pairs: list[tuple[str, str]]) -> dict:
    """pairs of (truth, predicted), each CONTRADICTED or SUPPORTED."""
    tp = sum(1 for t, p in pairs if t == "CONTRADICTED" and p == "CONTRADICTED")
    fp = sum(1 for t, p in pairs if t == "SUPPORTED" and p == "CONTRADICTED")
    fn = sum(1 for t, p in pairs if t == "CONTRADICTED" and p == "SUPPORTED")
    tn = sum(1 for t, p in pairs if t == "SUPPORTED" and p == "SUPPORTED")
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    spec = tn / (tn + fp) if tn + fp else None
    f1 = (2 * prec * rec / (prec + rec)) if prec and rec else None
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(prec, 4) if prec is not None else None,
            "recall": round(rec, 4) if rec is not None else None,
            "specificity": round(spec, 4) if spec is not None else None,
            "f1": round(f1, 4) if f1 is not None else None}


def main() -> int:
    pop = {r["pr_id"]: r for r in json.loads(POP.read_text(encoding="utf-8"))}
    fetch = {f["pr_id"]: f for f in json.loads(FETCH.read_text(encoding="utf-8"))}

    rows, excluded = [], Counter()
    for pid, rec in pop.items():
        f = fetch.get(pid, {})
        p = DIFFS / f"{pid}.diff"
        if not f.get("bytes") or not p.exists():
            excluded[f.get("status", "missing")] += 1
            continue
        diff = p.read_text(encoding="utf-8", errors="replace")
        for c in rec["claims"]:
            truth, facts = oracle.label(c["kind"], c.get("detail") or {}, diff)
            # the instrument, on the live diff, reading this one sentence
            inst_v, inst_why = None, None
            if c["kind"] in SCORED_KINDS:
                try:
                    g = dg.gate_diff_text(c["text"], diff, run=None, strict=False)
                    hit = next((x for x in g.claims if x.kind == c["kind"]), None)
                    if hit:
                        inst_v, inst_why = hit.verdict, hit.why
                except Exception as e:                     # never silently pass
                    inst_v, inst_why = f"ERROR:{type(e).__name__}", str(e)[:200]
            rows.append({"pr_id": pid, "url": rec["url"], "agent": rec["agent"],
                         "kind": c["kind"], "claim_text": c["text"], "claim_detail": c.get("detail"),
                         "diff_sha256": f["sha256"], "diff_bytes": f["bytes"],
                         "truth": truth, "truth_facts": facts,
                         "instrument_verdict": inst_v, "instrument_why": inst_why,
                         "similarity": round(similarity(c["text"], diff), 4),
                         "corpus_verdict_reference_only": c["corpus_verdict"]})

    with DATASET.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"_header": {
            "benchmark": "BENCH-1", "prereg": "PREREG_bench1_pr_claim_benchmark_2026_09_17.md",
            "corpus": "hao-li/AIDev (CC-BY-4.0), Zenodo 10.5281/zenodo.16919272",
            "ground_truth": "bench1_oracle.py on the live diff; corpus verdicts are reference only",
            "diffs": "not redistributed; each row carries the PR URL and the diff's sha256"}}) + "\n")
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    out = {"prereg": "PREREG_bench1_pr_claim_benchmark_2026_09_17.md",
           "population_prs": len(pop), "reached_prs": len({r["pr_id"] for r in rows}),
           "excluded_by_reason": dict(excluded), "items": len(rows),
           "per_kind": {}, "truth_distribution": {}, "notes": {}}

    for kind in SCORED_KINDS:
        ks = [r for r in rows if r["kind"] == kind]
        decidable = [r for r in ks if r["truth"] in ("CONTRADICTED", "SUPPORTED")]
        out["truth_distribution"][kind] = dict(Counter(r["truth"] for r in ks))
        if not decidable:
            out["per_kind"][kind] = {"decidable": 0}
            continue
        # the instrument: CONTRADICTED is a positive, anything else is a non-accusation
        inst = [(r["truth"], "CONTRADICTED" if r["instrument_verdict"] == "CONTRADICTED" else "SUPPORTED")
                for r in decidable]
        m_inst = metrics(inst)
        m_inst["abstained"] = sum(1 for r in decidable if r["instrument_verdict"] == "UNCHECKABLE")
        m_inst["not_extracted"] = sum(1 for r in decidable if r["instrument_verdict"] is None)

        # the baseline, with its threshold chosen after seeing the labels (an upper bound)
        best = None
        for t in [i / 100 for i in range(0, 101)]:
            pairs = [(r["truth"], "CONTRADICTED" if r["similarity"] < t else "SUPPORTED") for r in decidable]
            m = metrics(pairs)
            if m["f1"] is not None and (best is None or m["f1"] > best[1]["f1"]):
                best = (t, m)
        m_base = {"threshold_chosen_on_the_labels": best[0], **best[1]} if best else None

        n_claims = sum(1 for r in rows if r["kind"] == kind)
        fdr_pop = None
        if m_inst["tp"] + m_inst["fp"]:
            fdr_pop = round(m_inst["fp"] / (m_inst["tp"] + m_inst["fp"]), 4)
        out["per_kind"][kind] = {
            "decidable_items": len(decidable),
            "instrument": m_inst,
            "similarity_baseline_upper_bound": m_base,
            "false_discovery_rate": fdr_pop,
            "claims_of_this_kind_per_71016_prs": n_claims,
        }

    ta = [r for r in rows if r["kind"] == "tests_added"]
    out["notes"]["tests_added_excluded"] = {
        "items": len(ta),
        "why": "deciding what counts as a test is a judgement; the prereg excludes it from scoring"}
    out["notes"]["oracle_agrees_by_construction"] = (
        "the oracle and the instrument both read the live diff, so they agree on well-formed input; "
        "the discriminating power is in the hard cases and in the true negatives, and the audit is "
        "what establishes the oracle is right")
    SCORES.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
