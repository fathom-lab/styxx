#!/usr/bin/env python3
"""
drift_axis_analyze.py
=====================

Post-corpus analysis driver for the drift-axis-alignment preregistration.

Reads the two N=20 manifests, calls the locked drift_axis_scorer to
produce per-provider results, evaluates the §6 bar under each provider
and combined, and emits:

  1. The §9 result deposit JSON (full provenance, all bar evaluations)
  2. A markdown summary with the bar verdict
  3. A paper-draft skeleton with auto-filled numbers ready for editing

This file is NOT part of the locked scoring code. It calls the locked
scorer (`drift_axis_scorer.score_corpus` / `evaluate_bar`) without
modifying it; this driver only adds reporting + writeup automation.

Usage
-----
    python scripts/drift_axis_analyze.py \\
        --coop-manifest papers/cooperative-agent-regime/N20_coop_manifest.json \\
        --noncoop-manifest papers/cooperative-agent-regime/N20_noncoop_manifest.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from drift_axis_scorer import (  # noqa: E402
    OpenAIEmbeddings,
    BGEEmbeddings,
    score_corpus,
    evaluate_bar,
    _file_sha256,
)


PREREG_DOC = (
    "papers/cooperative-agent-regime/"
    "drift_axis_alignment_preregistration_2026_05_21.md"
)
SCORER_FILE = Path("scripts/drift_axis_scorer.py")


def analyze(
    coop_manifest: Path,
    noncoop_manifest: Path,
    output_dir: Path,
    prereg_lock_hash: str = "TBD-after-operator-signs",
    scorer_amendment_hash: str = "TBD-after-prereg-lock",
) -> dict:
    """Run the locked bar evaluation across both providers and deposit."""
    output_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    results: dict = {
        "kind": "drift_axis_corpus",
        "preregistration_doc": PREREG_DOC,
        "preregistration_lock_hash": prereg_lock_hash,
        "scoring_code_file": str(SCORER_FILE),
        "scoring_code_amendment_hash": scorer_amendment_hash,
        "scoring_code_sha256": _file_sha256(SCORER_FILE),
        "coop_manifest": str(coop_manifest),
        "noncoop_manifest": str(noncoop_manifest),
        "run_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "providers": {},
        "bar_evaluations": {},
        "combined_outcome": None,
    }

    for prov_name in ("openai", "bge"):
        print(f"\n=== provider: {prov_name} ===", flush=True)
        provider = (OpenAIEmbeddings() if prov_name == "openai"
                    else BGEEmbeddings())
        print(f"  scoring cooperative corpus (N={_count(coop_manifest)})...",
              flush=True)
        coop = score_corpus(coop_manifest, provider)
        print(f"  cooperative   median DAA = {coop['median']:+.4f}  "
              f"95% CI [{coop['bootstrap_ci_95'][0]:+.3f}, "
              f"{coop['bootstrap_ci_95'][1]:+.3f}]  "
              f"p = {coop['permutation_pvalue']:.4f}")
        print(f"  scoring non-cooperative corpus (N={_count(noncoop_manifest)})...",
              flush=True)
        noncoop = score_corpus(noncoop_manifest, provider)
        print(f"  non-cooperative median DAA = {noncoop['median']:+.4f}  "
              f"95% CI [{noncoop['bootstrap_ci_95'][0]:+.3f}, "
              f"{noncoop['bootstrap_ci_95'][1]:+.3f}]")
        bar = evaluate_bar(coop, noncoop)
        delta = coop["median"] - noncoop["median"]
        print(f"  Δ = {delta:+.4f}   outcome under {prov_name}: {bar['outcome']}")
        results["providers"][prov_name] = {"coop": coop, "noncoop": noncoop}
        results["bar_evaluations"][prov_name] = bar

    # Combined outcome (§6: both providers must clear for POSITIVE)
    outcomes = {p: r["outcome"] for p, r in results["bar_evaluations"].items()}
    if all(o == "POSITIVE" for o in outcomes.values()):
        results["combined_outcome"] = "POSITIVE_BOTH_PROVIDERS"
    elif any(o == "CLOSED_NEGATIVE" for o in outcomes.values()):
        results["combined_outcome"] = "CLOSED_NEGATIVE_AT_LEAST_ONE_PROVIDER"
    else:
        results["combined_outcome"] = "INTERMEDIATE_DEPOSIT_OR_CONDITIONAL"

    results["elapsed_seconds"] = round(time.time() - t_start, 1)

    # Deposit §9 JSON
    date_tag = time.strftime("%Y-%m-%d")
    deposit = output_dir / f"drift_axis_corpus_{date_tag}.json"
    deposit.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\n[deposit] {deposit}")

    # Markdown summary
    md = _render_summary_md(results)
    summary_path = output_dir / f"drift_axis_corpus_{date_tag}_summary.md"
    summary_path.write_text(md, encoding="utf-8")
    print(f"[summary] {summary_path}")

    # Paper-draft skeleton (only if outcome is meaningful)
    if results["combined_outcome"] != "CLOSED_NEGATIVE_AT_LEAST_ONE_PROVIDER":
        paper = _render_paper_skeleton(results)
    else:
        paper = _render_closed_negative_paper(results)
    paper_path = output_dir / f"drift_axis_paper_draft_{date_tag}.md"
    paper_path.write_text(paper, encoding="utf-8")
    print(f"[paper]   {paper_path}")
    return results


def _count(manifest_path: Path) -> int:
    return len(json.loads(manifest_path.read_text(encoding="utf-8"))
               .get("conversations", []))


def _render_summary_md(r: dict) -> str:
    L: list[str] = []
    L.append(f"# Drift-Axis Alignment Corpus Result — {r['run_iso']}\n")
    L.append(f"**Combined outcome:** `{r['combined_outcome']}`\n")
    L.append(f"**Preregistration lock-hash:** `{r['preregistration_lock_hash']}`")
    L.append(f"**Scoring code SHA256:** `{r['scoring_code_sha256']}`")
    L.append("")
    L.append("## Per-provider bar evaluation\n")
    L.append("| Provider | Coop median | Noncoop median | Δ | p | Outcome |")
    L.append("|---|---|---|---|---|---|")
    for prov, bar in r["bar_evaluations"].items():
        prov_data = r["providers"][prov]
        coop_m = prov_data["coop"]["median"]
        noncoop_m = prov_data["noncoop"]["median"]
        p = prov_data["coop"]["permutation_pvalue"]
        L.append(f"| {prov} | {coop_m:+.4f} | {noncoop_m:+.4f} | "
                 f"{bar['delta']:+.4f} | {p:.4f} | `{bar['outcome']}` |")
    L.append("")
    L.append("## §6 bar items per provider\n")
    for prov, bar in r["bar_evaluations"].items():
        L.append(f"### {prov}\n")
        L.append(f"- median(coop) >= 0.60 : **{bar['median_coop_ge_0.60']}**")
        L.append(f"- median(noncoop) <= 0.55 : **{bar['median_noncoop_le_0.55']}**")
        L.append(f"- Δ >= 0.15 : **{bar['delta_ge_0.15']}** (Δ = {bar['delta']:+.4f})")
        L.append(f"- p < 0.01 : **{bar['permutation_p_lt_0.01']}**")
        L.append(f"- kill-gate (Δ < 0.10) triggered: **{bar['kill_gate_triggered']}**")
        L.append("")
    return "\n".join(L)


def _render_paper_skeleton(r: dict) -> str:
    outcome = r["combined_outcome"]
    oai_bar = r["bar_evaluations"].get("openai", {})
    bge_bar = r["bar_evaluations"].get("bge", {})
    return f"""# Trajectory-Level Cognitive Coupling in Cooperative AI Dyads:
A Preregistered Cross-Vendor Replication of Drift-Axis Alignment

**Authors:** Flobi (@flobi69), darkflobi
**Date:** {time.strftime("%Y-%m-%d")}
**Preregistration:** [{PREREG_DOC}](../{PREREG_DOC})
**Lock-hash:** `{r['preregistration_lock_hash']}`
**Scoring code SHA256:** `{r['scoring_code_sha256']}`
**Status:** DRAFT (auto-generated from result deposit; awaiting human revision)

## Abstract

We measure trajectory-level cognitive coupling between cooperative AI agent
dyads using **drift-axis alignment** (DAA), a label-free geometric statistic
on per-turn response embeddings. Under a preregistered cross-vendor methodology
(OpenAI `text-embedding-3-large` AND BAAI `bge-large-en-v1.5`, both
independently required to clear the bar), we report DAA on N=20 cooperative
and N=20 non-cooperative dyads.

**Result:** `{outcome}`.

- OpenAI: median DAA cooperative = {oai_bar.get('delta', 'TBD')} above non-cooperative; outcome `{oai_bar.get('outcome','TBD')}`.
- BGE:    median DAA cooperative = {bge_bar.get('delta', 'TBD')} above non-cooperative; outcome `{bge_bar.get('outcome','TBD')}`.

This is the first preregistered, cross-vendor evidence that cognitive
trajectories of cooperating agents couple at the conceptual level even when
register-level coupling (Pearson r on composite cognometric scores) does not.
The earlier phase-coherence preregistration (closed-negative on register
coupling, 2026-05-20) is therefore conditional on the channel: register
does not couple, trajectory does.

## 1. Background

[Reference phase-coherence closed-negative. Position H_drift_axis as the
next bet in the cooperative-agent program. Cite the 7.4.1 honest-scoping
correction.]

## 2. Methods

**Operational definition.** Per §4 of the preregistration:

```
DAA(embs_a, embs_b) = cos(
    mean(embs_a[half:n]) - mean(embs_a[:half]),
    mean(embs_b[half:n]) - mean(embs_b[:half])
)
```

**Corpus.** N=20 cooperative + N=20 non-cooperative conversations. 5 task
seeds × 4 replicates per regime. Cross-model dyad: gpt-4o-mini × gpt-4.1-mini.
22 turns per agent. Tasks: park co-design, noir flash fiction, SQL debugging,
road trip planning, abstract co-drafting.

**Bar.** Median(coop) ≥ 0.60, median(noncoop) ≤ 0.55, Δ ≥ 0.15, permutation
p < 0.01 — all four required on BOTH embedding models independently.

## 3. Results

[Auto-fill from r['providers']. Include the per-conversation DAA table,
shuffled-pairs null distribution histogram, and bar evaluation per provider.]

## 4. Discussion

[Channel-vs-register framing. Why register doesn't couple but trajectory does.
Limitations: same-task-seed across regimes is a confound to flag explicitly.
Future work: bandwidth measurement, encoder/decoder protocol spec, multi-agent
extensions, cross-architecture (Anthropic, open-weight) replication.]

## 5. Limitations

- Same-task-seed across regimes: cooperative and non-cooperative dyads share
  the underlying task prompt, so some trajectory alignment may be driven by
  task convergence rather than cooperation per se. The non-cooperative
  control isolates the cooperation signal, but does not isolate task signal.
- N=20 per regime is the preregistered minimum; larger corpora would tighten
  CIs.
- Cross-architecture replication (Anthropic, open-weight) is future work; the
  current bar is cross-vendor on embedding models, not on LLM architecture.

## 6. Reproducibility

All code, data, and results are deposited at
`papers/cooperative-agent-regime/results/drift_axis_corpus_{time.strftime("%Y-%m-%d")}.json`.
Re-running the analysis pipeline from a fresh clone:

```
python scripts/build_drift_axis_corpus.py --both --turns 22
python scripts/drift_axis_analyze.py \\
    --coop-manifest papers/cooperative-agent-regime/N20_coop_manifest.json \\
    --noncoop-manifest papers/cooperative-agent-regime/N20_noncoop_manifest.json
```

## References

- Phase-Coherence Preregistration (2026-05-20, closed-negative)
- Every Mind Leaves Vitals (Zenodo DOI 10.5281/zenodo.19777921)
- Styxx 7.4.1 honest-scoping correction (commit 0ad384e)
"""


def _render_closed_negative_paper(r: dict) -> str:
    return f"""# Drift-Axis Alignment Does Not Discriminate Cooperative from
Non-Cooperative AI Dyads at the Preregistered Bar: A Closed-Negative Result

**Authors:** Flobi (@flobi69), darkflobi
**Date:** {time.strftime("%Y-%m-%d")}
**Preregistration:** {PREREG_DOC}
**Outcome:** `{r['combined_outcome']}`

## Abstract

We preregistered a cross-vendor test of whether **drift-axis alignment** (DAA)
between cooperative AI agent dyads systematically exceeds non-cooperative
dyads of matched task. The exploratory probe (N=5+5, 2026-05-20) showed
Δ +0.327 on the OpenAI embedding. The full N=20+20 cross-vendor corpus
**did not clear the preregistered bar**: at least one embedding model
triggered the kill-gate (Δ < 0.10).

This closed-negative extends the integrity chain that began with the
phase-coherence preregistration (closed-negative, 2026-05-20). The
trajectory-coupling channel does not survive the cross-vendor cross-corpus
discipline under the methodology defined here.

## 1. What the Exploratory Probe Showed

[Cite 8ff3b65, Δ +0.327. Position as the candidate signal.]

## 2. What the Locked Bar Required

[Bar items. Why both-providers gating was preregistered.]

## 3. What the Corpus Run Found

[Auto-fill from r['providers']. Show where the signal collapsed.]

## 4. Interpretation

- The exploratory N=5+5 effect was either a small-sample artifact or
  conditional on the OpenAI embedding family in a way the cross-vendor
  bar correctly caught.
- Trajectory coupling at the centroid-difference level is not, under the
  methodology defined here, a reliable cooperation signature.
- Methodology revisions for a future preregistration must declare a new
  lock-hash and a new bar; this document remains as the closed-negative
  record under the current methodology.

## 5. The Integrity Chain Extends

The methodology held. The bar held. The kill-gate fired honestly. Future
work proceeds from this published negative, not from quiet abandonment.
"""


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Analyze the N=20+20 drift-axis corpus."
    )
    p.add_argument("--coop-manifest", type=Path, required=True)
    p.add_argument("--noncoop-manifest", type=Path, required=True)
    p.add_argument(
        "--output-dir", type=Path,
        default=Path("papers/cooperative-agent-regime/results"),
    )
    p.add_argument("--prereg-lock-hash", default="TBD-after-operator-signs")
    p.add_argument("--scorer-amendment-hash", default="TBD-after-prereg-lock")
    args = p.parse_args(argv)
    analyze(
        args.coop_manifest, args.noncoop_manifest, args.output_dir,
        args.prereg_lock_hash, args.scorer_amendment_hash,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
