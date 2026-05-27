# -*- coding: utf-8 -*-
"""Build the dark-core benchmark dataset.

Unifies the labeled examples from the four-method consensus-hallucination arc
into a single benchmark JSON for AI-integrity routing research:

  - ICT receipts (probe_ict_results.json): 50 items, categorized by the
    `analyze_darkcore.categorize` rules at runtime, with the consensus answer
    and (where applicable) the injected competitor recorded.
  - Curated folklore corpus (corpus_folklore_2026_05_27.py): 30 hand-curated
    cultural-prior items with explicit (expected_consensus, injected_competitor).
  - Curated truth controls: 30 hand-curated factual items with explicit
    (expected_consensus, injected_competitor) where the competitor is a
    plausible falsehood.

Output: darkcore_benchmark_2026_05_27.json — clean labeled records with stable
ids, source attribution, the closed-negative classifier baseline reference,
and the four-method detection result for each class as the empirical floor
that any future routing approach must beat.

CPU-only; ~1 second to build.
"""
from __future__ import annotations
import json, sys, pathlib

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from corpus_folklore_2026_05_27 import FOLKLORE, TRUTHS

# ──────────────────────────────────────────────────────────────────────
# Load ICT receipts
# ──────────────────────────────────────────────────────────────────────
ict = json.loads((HERE / "probe_ict_results.json").read_text())
ict_rows = ict.get("rows", [])

def ict_class(row):
    """Map ICT row (label, cat) into the 4-class scheme + drop self-ref."""
    cat = row.get("cat", "")
    label = row.get("label")
    if cat == "self-referential":
        return None
    if label == 0:
        return "truth"
    if cat == "folklore/legend":
        return "folklore"
    if cat == "pseudoscience/supernatural":
        return "pseudoscience"
    if cat == "factual-error":
        return "factual-error"
    return None

records = []
next_id = 1

# ICT receipts → records
for r in ict_rows:
    cls = ict_class(r)
    if cls is None:
        continue
    records.append({
        "id": f"darkcore-{next_id:04d}",
        "question": r.get("q", ""),
        "class": cls,
        "source": "ict_2026_05_27 (TruthfulQA-derived, runtime-categorized)",
        "expected_consensus": r.get("consensus", "") or None,
        "expected_competitor": r.get("competitor", "") or None,
        "council_agreement": r.get("agreement"),
        "ict_yielded": r.get("yielded"),  # whether the council moved on injection
    })
    next_id += 1

# Curated folklore → records
for q, expected, competitor in FOLKLORE:
    records.append({
        "id": f"darkcore-{next_id:04d}",
        "question": q,
        "class": "folklore",
        "source": "curated_folklore_2026_05_27 (hand-curated cultural-prior items)",
        "expected_consensus": expected,
        "expected_competitor": competitor,
        "council_agreement": None,        # filled during ICT-folklore probe run
        "ict_yielded": None,
    })
    next_id += 1

# Curated truths → records (additional truth controls beyond ICT)
for q, expected, competitor in TRUTHS:
    records.append({
        "id": f"darkcore-{next_id:04d}",
        "question": q,
        "class": "truth",
        "source": "curated_truth_2026_05_27 (hand-curated basic-fact controls)",
        "expected_consensus": expected,
        "expected_competitor": competitor,
        "council_agreement": None,
        "ict_yielded": None,
    })
    next_id += 1

# Class distribution + descriptive stats
from collections import Counter
class_dist = Counter(r["class"] for r in records)
source_dist = Counter(r["source"] for r in records)

# Empirical floor — the closed-negative results that any routing approach must beat
classifier_results = json.loads((HERE / "darkcore_classifier_results.json").read_text())
empirical_floor = {
    "detection_methods_tested": {
        "perturbation_fragility (Dark Matter #1)": {
            "finding": "papers/consensus-hallucination/FINDING_darkmatter_2026_05_25.md",
            "dark_core_result": "partial — flips fragile shell, misses stubborn core",
        },
        "agreement_fracture (CVPD)": {
            "finding": "papers/consensus-hallucination/FINDING_cvpd_2026_05_25.md",
            "dark_core_result": "clean negative, lift −0.32 vs binary flip",
        },
        "justification_divergence (JD)": {
            "finding": "papers/consensus-hallucination/FINDING_jd_2026_05_27.md",
            "dark_core_result": "clean negative, INVERTED (stubborn core has MOST convergent justifications)",
        },
        "constructive_injection (ICT)": {
            "finding": "papers/consensus-hallucination/FINDING_ict_2026_05_27.md",
            "dark_core_result": "clean negative, folklore yield = 0.00 (n_folk=4 thin; ICT-folklore rerun in flight)",
        },
    },
    "classification_baseline": {
        "method": "sentence-transformers/all-MiniLM-L6-v2 + balanced one-vs-rest logistic regression",
        "result_json": "papers/consensus-hallucination/darkcore_classifier_results.json",
        "K1_indist_folklore_F1": classifier_results["K1_indist_folklore_F1"],
        "K2_indist_accuracy": classifier_results["K2_indist_accuracy"],
        "K3_crosscorpus_folklore_F1": classifier_results["K3_crosscorpus_folklore_F1"],
        "K3_crosscorpus_recall": classifier_results["K3_crosscorpus_recall_on_curated_folklore_only"],
        "PASS": classifier_results["PASS_K1_AND_K2_AND_K3"],
    },
    "synthesis": "papers/SYNTHESIS_decorrelation_ceiling_2026_05_25.md (load-bearing-floor branch confirmed)",
}

benchmark = {
    "name": "darkcore_benchmark",
    "version": "2026-05-27",
    "license": "same as styxx (see repository root)",
    "description": (
        "Labeled question dataset for AI-integrity routing research. Each record "
        "tags a question with one of four classes (folklore, pseudoscience, "
        "factual-error, truth) corresponding to the Decorrelation Ceiling synthesis's "
        "competitor-availability gradient. The pre-registered four-method detection "
        "arc and the one classification baseline (this commit) jointly establish the "
        "current empirical floor; future routing approaches should beat the closed "
        "negatives recorded under `empirical_floor`."
    ),
    "n_records": len(records),
    "class_distribution": dict(class_dist),
    "source_distribution": dict(source_dist),
    "schema": {
        "id": "stable record id (darkcore-NNNN)",
        "question": "the question text",
        "class": "one of: folklore | pseudoscience | factual-error | truth",
        "source": "where the labeled record came from (ICT vs curated; with date)",
        "expected_consensus": (
            "the three-vendor council's expected baseline answer when polled; "
            "for folklore/pseudoscience/factual-error, this is the misconception; "
            "for truth, this is the correct answer"
        ),
        "expected_competitor": (
            "the answer injected as the A/B competitor in ICT-style probes; "
            "for misconceptions, this is the truth; for truth records, a plausible falsehood"
        ),
        "council_agreement": "ICT-recorded baseline-agreement fraction (1.0 if all 3 vendors agreed)",
        "ict_yielded": "did the council move to the injected competitor in ICT (true/false/null if not run)",
    },
    "empirical_floor": empirical_floor,
    "records": records,
}

out = HERE / "darkcore_benchmark_2026_05_27.json"
out.write_text(json.dumps(benchmark, indent=2, default=str))
print(f"wrote {out} ({out.stat().st_size:,} bytes)")
print(f"records: {benchmark['n_records']}")
print(f"class distribution: {benchmark['class_distribution']}")
