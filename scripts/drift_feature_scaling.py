"""Feature-count scaling ablation for the cognometric drift detector.

Research question: as we add features to the calibrated drift classifier,
does per-failure-type AUC rise smoothly (incremental learning) or in
discrete jumps (phase transitions)? A phase-transition signal would be
the mirror of emergent-capability literature — specific failure classes
becoming DETECTABLE at specific feature-count thresholds.

Three ablations run on the drift_v0 dataset (n=3700, BFCL v3 mutations
+ irrelevance splits):

  (1) TOP-K BY IMPORTANCE — rank features by |coef| from the full
      22-feature baseline, retrain LR on top-K. Shows the "ceiling if
      we could only pick K features."

  (2) GROUP-INCREMENTAL — feature groups added in order
      (A: semantic alignment, B: schema conformance, C: lexical drift,
      D: structural). Shows which group matters for which failure class.

  (3) RANDOM SUBSETS (3 seeds per K) — K features drawn uniformly at
      random. Null-expectation for (1).

All three compute pooled AUC + per-drift-type AUC via 5-fold stratified
CV (RANDOM_STATE=0, consistent with calibrated_weights_drift_v1).

Phase-transition flag: a per-drift-type AUC jump >= 0.10 between two
consecutive K values in the top-K ablation.

Usage:
    python scripts/drift_feature_scaling.py

Outputs:
    benchmarks/drift_feature_scaling.json          — raw numbers
    papers/figures/drift_phase_transitions.png     — figure
    papers/drift_phase_transitions.md              — writeup draft
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
DATA_PATH = REPO / "data" / "drift_v0" / "drift_dataset_v0.jsonl"
OUT_JSON = REPO / "benchmarks" / "drift_feature_scaling.json"
OUT_FIG = REPO / "papers" / "figures" / "drift_phase_transitions.png"
OUT_MD = REPO / "papers" / "drift_phase_transitions.md"

RANDOM_STATE = 0
N_SPLITS = 5

# Feature-count schedule — chosen to sample the full curve densely enough
# to see any kink. Powers-of-two-ish plus edge cases.
FEATURE_COUNTS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 17, 19, 22]
RANDOM_SUBSET_SEEDS = [11, 23, 47]

# ---------------------------------------------------------------------
# Feature extraction — vendored from scripts/drift_calibrated_v0.py to
# keep this ablation self-contained. Must match
# styxx/guardrail/drift_signals.py exactly.
# ---------------------------------------------------------------------

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "for", "to", "in",
    "on", "at", "by", "with", "from", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "can", "could", "should", "may", "might", "must",
    "this", "that", "these", "those", "i", "you", "he", "she", "it", "we",
    "they", "me", "him", "her", "us", "them", "my", "your", "his", "our",
    "their", "what", "which", "who", "how", "when", "where", "why",
}

WORD_RE = re.compile(r"[A-Za-z0-9_]+")
PLACEHOLDER_RE = re.compile(
    r"^(placeholder|example|test|_[a-z_]+|<[^>]+>)$", re.IGNORECASE
)


def tokens(text: str) -> List[str]:
    return [w.lower() for w in WORD_RE.findall(text or "")]


def content_tokens(text: str) -> List[str]:
    return [t for t in tokens(text) if t not in STOPWORDS and len(t) > 1]


FEATURE_NAMES = [
    # Group A — semantic alignment (indices 0-4)
    "tool_in_prompt", "tool_parts_in_prompt", "overlap_jaccard",
    "prompt_coverage", "arg_verbatim_rate",
    # Group B — schema conformance (5-10)
    "tool_in_schema", "missing_required_frac", "spurious_arg_frac",
    "type_mismatch_frac", "arg_count_zscore", "required_count",
    # Group C — lexical drift (11-14)
    "placeholder_frac", "tool_name_len", "tool_in_any_schema",
    "n_available_tools",
    # Group D — structural (15-21)
    "n_args_called", "prompt_len", "avg_arg_len", "has_nested",
    "has_list", "prompt_is_question", "prompt_imperative",
]

GROUPS = {
    "A_semantic":   list(range(0, 5)),
    "B_schema":     list(range(5, 11)),
    "C_lexical":    list(range(11, 15)),
    "D_structural": list(range(15, 22)),
}


def extract_features(r: Dict) -> List[float]:
    """22-dim feature vector. MUST match drift_signals.extract_drift_features."""
    prompt = r.get("prompt", "") or ""
    call = r.get("tool_call", {}) or {}
    fns = r.get("functions", []) or []

    call_name = (call.get("name") or "").lower()
    call_args = call.get("arguments") or {}

    prompt_tok = set(tokens(prompt))
    prompt_content = set(content_tokens(prompt))

    # ---- Group A: semantic alignment ----
    tool_in_prompt = 1.0 if call_name and call_name in prompt_tok else 0.0
    tool_parts = set(call_name.split("_"))
    tool_parts_in_prompt = (
        len(tool_parts & prompt_content) / max(1, len(tool_parts))
        if tool_parts else 0.0
    )
    call_text_tokens = set()
    for v in call_args.values():
        call_text_tokens.update(tokens(str(v)))
    call_text_tokens.update(tokens(call_name))
    union = prompt_content | call_text_tokens
    overlap_jaccard = (
        len(prompt_content & call_text_tokens) / max(1, len(union)) if union else 0.0
    )
    prompt_coverage = (
        len(prompt_content & call_text_tokens) / max(1, len(prompt_content))
    )
    if call_args:
        hits = total = 0
        for v in call_args.values():
            vt = set(tokens(str(v)))
            if vt:
                total += 1
                if vt & prompt_tok:
                    hits += 1
        arg_verbatim_rate = hits / max(1, total)
    else:
        arg_verbatim_rate = 0.0

    # ---- Group B: schema conformance ----
    called_schema = None
    for f in fns:
        if (f.get("name") or "").lower() == call_name:
            called_schema = f
            break
    tool_in_schema = 1.0 if called_schema else 0.0

    schema_required, schema_props = [], {}
    if called_schema:
        params = called_schema.get("parameters", {}) or {}
        schema_required = params.get("required", []) or []
        schema_props = params.get("properties", {}) or {}

    missing_required_frac = (
        len(set(schema_required) - set(call_args.keys())) / max(1, len(schema_required))
        if schema_required else 0.0
    )
    spurious_arg_frac = (
        len(set(call_args.keys()) - set(schema_props.keys())) / max(1, len(call_args))
        if schema_props and call_args else 0.0
    )
    type_mismatch = 0
    type_total = 0
    for k, v in call_args.items():
        if k in schema_props:
            type_total += 1
            declared = (schema_props[k] or {}).get("type", "")
            if declared == "integer" and not isinstance(v, int):
                type_mismatch += 1
            elif declared == "number" and not isinstance(v, (int, float)):
                type_mismatch += 1
            elif declared == "boolean" and not isinstance(v, bool):
                type_mismatch += 1
            elif declared == "string" and not isinstance(v, str):
                type_mismatch += 1
            elif declared == "array" and not isinstance(v, list):
                type_mismatch += 1
    type_mismatch_frac = type_mismatch / max(1, type_total)

    n_args_called = float(len(call_args))
    expected_n = float(len(schema_props)) if schema_props else 0.0
    arg_count_zscore = (n_args_called - expected_n) / max(1.0, expected_n)
    required_count = float(len(schema_required))

    # ---- Group C: lexical drift ----
    placeholder_hits = sum(
        1 for v in call_args.values() if PLACEHOLDER_RE.match(str(v)) is not None
    )
    placeholder_frac = placeholder_hits / max(1, len(call_args))
    tool_name_len = float(len(call_name))
    tool_in_any_schema = 1.0 if any(
        (f.get("name") or "").lower() == call_name for f in fns
    ) else 0.0
    n_available_tools = float(len(fns))

    # ---- Group D: structural ----
    prompt_len = float(len(prompt))
    if call_args:
        avg_arg_len = sum(len(str(v)) for v in call_args.values()) / len(call_args)
    else:
        avg_arg_len = 0.0
    has_nested = 1.0 if any(isinstance(v, dict) for v in call_args.values()) else 0.0
    has_list = 1.0 if any(isinstance(v, list) for v in call_args.values()) else 0.0
    prompt_is_question = 1.0 if "?" in prompt else 0.0
    IMPERATIVE = {"find", "get", "compute", "calculate", "show", "give", "tell",
                  "make", "do", "run", "start", "stop", "send", "fetch",
                  "list", "create", "delete", "update", "check", "play"}
    first_tok = tokens(prompt)[:1]
    prompt_imperative = 1.0 if first_tok and first_tok[0] in IMPERATIVE else 0.0

    return [
        tool_in_prompt, tool_parts_in_prompt, overlap_jaccard, prompt_coverage,
        arg_verbatim_rate,
        tool_in_schema, missing_required_frac, spurious_arg_frac,
        type_mismatch_frac, arg_count_zscore, required_count,
        placeholder_frac, tool_name_len, tool_in_any_schema, n_available_tools,
        n_args_called, prompt_len, avg_arg_len, has_nested, has_list,
        prompt_is_question, prompt_imperative,
    ]


# ---------------------------------------------------------------------
# Training + evaluation
# ---------------------------------------------------------------------

def load_dataset() -> List[Dict]:
    rows = []
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def eval_subset(
    X: np.ndarray,
    y: np.ndarray,
    drift_types: np.ndarray,
    feat_idx: List[int],
) -> Dict:
    """5-fold stratified CV on X[:, feat_idx]. Returns pooled + per-drift-type AUC."""
    X_sub = X[:, feat_idx]
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_aucs = []
    pooled_y = []
    pooled_scores = []
    pooled_drift_types = []
    for train_idx, test_idx in skf.split(X_sub, y):
        X_tr, X_te = X_sub[train_idx], X_sub[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        clf = LogisticRegression(
            C=1.0, max_iter=2000, class_weight="balanced",
            random_state=RANDOM_STATE,
        )
        clf.fit(X_tr_s, y_tr)
        scores = clf.predict_proba(X_te_s)[:, 1]
        fold_aucs.append(float(roc_auc_score(y_te, scores)))
        pooled_y.extend(y_te.tolist())
        pooled_scores.extend(scores.tolist())
        pooled_drift_types.extend(drift_types[test_idx].tolist())

    pooled_y = np.array(pooled_y)
    pooled_scores = np.array(pooled_scores)
    pooled_drift_types = np.array(pooled_drift_types)
    pooled_auc = float(roc_auc_score(pooled_y, pooled_scores))

    # Per-drift-type: drift (y=1) of each type vs gold negatives (y=0, drift_type='gold')
    per_type = {}
    gold_mask = pooled_drift_types == "gold"
    for dt in sorted(set(pooled_drift_types)):
        if dt == "gold":
            continue
        type_mask = pooled_drift_types == dt
        eval_mask = type_mask | gold_mask
        y_eval = pooled_y[eval_mask]
        s_eval = pooled_scores[eval_mask]
        if len(set(y_eval)) < 2:
            per_type[dt] = None
            continue
        per_type[dt] = float(roc_auc_score(y_eval, s_eval))

    return {
        "n_features": len(feat_idx),
        "fold_aucs": fold_aucs,
        "mean_auc": float(np.mean(fold_aucs)),
        "std_auc": float(np.std(fold_aucs)),
        "pooled_auc": pooled_auc,
        "per_drift_type_auc": per_type,
        "feat_idx": [int(i) for i in feat_idx],
        "feat_names": [FEATURE_NAMES[int(i)] for i in feat_idx],
    }


# ---------------------------------------------------------------------
# Ablations
# ---------------------------------------------------------------------

def ablation_top_k(X, y, dtypes, ranking: List[int]) -> List[Dict]:
    """For each K in FEATURE_COUNTS, train on top-K features from ranking."""
    results = []
    for k in FEATURE_COUNTS:
        idx = ranking[:k]
        r = eval_subset(X, y, dtypes, idx)
        r["schedule"] = "top_k_by_importance"
        r["k"] = k
        results.append(r)
        print(f"  top-K={k:2d}  pooled={r['pooled_auc']:.4f}  "
              f"per_type={ {k2:(round(v,3) if v is not None else None) for k2,v in r['per_drift_type_auc'].items()} }")
    return results


def ablation_group_incremental(X, y, dtypes) -> List[Dict]:
    """A, A+B, A+B+C, A+B+C+D."""
    results = []
    order = ["A_semantic", "B_schema", "C_lexical", "D_structural"]
    accum = []
    for g in order:
        accum = accum + GROUPS[g]
        r = eval_subset(X, y, dtypes, accum)
        r["schedule"] = "group_incremental"
        r["group_added"] = g
        r["groups_active"] = order[:order.index(g)+1]
        results.append(r)
        print(f"  +{g:15s} K={len(accum):2d}  pooled={r['pooled_auc']:.4f}")
    return results


def ablation_random_subsets(X, y, dtypes, seeds=RANDOM_SUBSET_SEEDS) -> List[Dict]:
    """K features chosen uniformly at random, per seed, averaged post-hoc."""
    results = []
    rng_all = np.random.default_rng(0)
    for k in FEATURE_COUNTS:
        seed_rows = []
        for seed in seeds:
            rng = np.random.default_rng(seed * 1000 + k)
            idx = list(rng.choice(22, size=k, replace=False))
            r = eval_subset(X, y, dtypes, idx)
            r["schedule"] = "random_subsets"
            r["k"] = k
            r["seed"] = seed
            results.append(r)
            seed_rows.append(r)
        mean_pooled = float(np.mean([s["pooled_auc"] for s in seed_rows]))
        print(f"  random-K={k:2d}  mean_pooled={mean_pooled:.4f}  "
              f"seeds={[round(s['pooled_auc'],3) for s in seed_rows]}")
    return results


def rank_features(X, y) -> List[int]:
    """Rank 22 features by |coef| from a full model trained on the entire set."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced",
                             random_state=RANDOM_STATE)
    clf.fit(X_s, y)
    coefs = clf.coef_[0]
    ranking = sorted(range(len(coefs)), key=lambda i: abs(coefs[i]), reverse=True)
    print("  feature importance ranking (|coef| desc):")
    for rank, idx in enumerate(ranking):
        print(f"    {rank+1:2d}. {FEATURE_NAMES[idx]:28s}  coef={coefs[idx]:+.3f}")
    return ranking


# ---------------------------------------------------------------------
# Phase-transition detection
# ---------------------------------------------------------------------

def detect_phase_transitions(top_k_results: List[Dict], threshold: float = 0.10) -> List[Dict]:
    """Flag per-drift-type AUC jumps >= `threshold` between consecutive K."""
    transitions = []
    sorted_r = sorted(top_k_results, key=lambda r: r["k"])
    for prev, curr in zip(sorted_r[:-1], sorted_r[1:]):
        for dt in curr["per_drift_type_auc"]:
            prev_auc = prev["per_drift_type_auc"].get(dt)
            curr_auc = curr["per_drift_type_auc"].get(dt)
            if prev_auc is None or curr_auc is None:
                continue
            delta = curr_auc - prev_auc
            if delta >= threshold:
                transitions.append({
                    "drift_type": dt,
                    "from_k": prev["k"],
                    "to_k": curr["k"],
                    "from_auc": prev_auc,
                    "to_auc": curr_auc,
                    "delta": delta,
                    "feature_added": [f for f in curr["feat_names"] if f not in prev["feat_names"]],
                })
    return transitions


# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------

def make_figure(top_k_results, group_results, random_results, transitions, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed, skipping figure")
        return

    drift_types = sorted(
        {dt for r in top_k_results for dt in r["per_drift_type_auc"]
         if r["per_drift_type_auc"].get(dt) is not None}
    )
    colors = {
        "arg_drop": "#00ff00",
        "spurious_arg": "#00e5ff",
        "irrelevance_called": "#ffd700",
        "arg_swap": "#ff3d00",
        "tool_rename": "#888888",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#0a0a0a")
    for ax in axes:
        ax.set_facecolor("#141414")
        ax.tick_params(colors="#c0c0c0")
        for spine in ax.spines.values():
            spine.set_color("#2a2a2a")

    # ─── Left: top-K by importance, per-drift-type trajectories ───
    ax = axes[0]
    ks = sorted([r["k"] for r in top_k_results])
    for dt in drift_types:
        if dt == "tool_rename":
            continue  # n=1, noise
        ys = [r["per_drift_type_auc"].get(dt) for r in sorted(top_k_results, key=lambda r: r["k"])]
        ax.plot(ks, ys, marker="o", color=colors.get(dt, "#ffffff"),
                label=dt, linewidth=2, markersize=6)
    # overall pooled
    pooled_ys = [r["pooled_auc"] for r in sorted(top_k_results, key=lambda r: r["k"])]
    ax.plot(ks, pooled_ys, marker="s", color="#ffffff", linewidth=2.5,
            linestyle="--", label="pooled", alpha=0.8)
    ax.axhline(0.733, color="#ff3d00", linestyle=":", alpha=0.5, label="null (schema_conformance)")
    ax.axhline(0.5, color="#888888", linestyle=":", alpha=0.3, label="chance")
    # annotate transitions
    for t in transitions:
        ax.annotate(
            f"{t['drift_type']}\n+{t['delta']:.2f}",
            xy=(t["to_k"], t["to_auc"]),
            xytext=(t["to_k"] + 0.5, t["to_auc"] - 0.15),
            color=colors.get(t["drift_type"], "#ffffff"),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color=colors.get(t["drift_type"], "#ffffff"),
                            alpha=0.6, lw=1),
        )
    ax.set_xlabel("# features (top-K by |coef|)", color="#c0c0c0")
    ax.set_ylabel("AUC", color="#c0c0c0")
    ax.set_title("cognometric phase transitions — drift v1, 5-fold CV", color="#e8e8e8")
    ax.legend(loc="lower right", facecolor="#0a0a0a", edgecolor="#2a2a2a",
              labelcolor="#c0c0c0", fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.15, color="#888888")

    # ─── Right: top-K vs random subsets vs group-incremental ───
    ax = axes[1]
    ax.plot(ks, pooled_ys, marker="o", color="#00ff00", linewidth=2.5,
            label="top-K (by |coef|)", markersize=7)
    # random subsets — aggregate mean per K
    rand_by_k = {}
    for r in random_results:
        rand_by_k.setdefault(r["k"], []).append(r["pooled_auc"])
    rand_ks = sorted(rand_by_k.keys())
    rand_means = [np.mean(rand_by_k[k]) for k in rand_ks]
    rand_stds = [np.std(rand_by_k[k]) for k in rand_ks]
    ax.errorbar(rand_ks, rand_means, yerr=rand_stds, marker="^",
                color="#ffd700", linewidth=1.5, label="random subsets (3 seeds)",
                markersize=6, capsize=3, alpha=0.8)
    # group-incremental
    grp_ks = [r["n_features"] for r in group_results]
    grp_ys = [r["pooled_auc"] for r in group_results]
    ax.plot(grp_ks, grp_ys, marker="D", color="#00e5ff", linewidth=2,
            label="group-incremental (A→B→C→D)", markersize=8, linestyle="--")
    ax.axhline(0.733, color="#ff3d00", linestyle=":", alpha=0.5, label="null baseline 0.733")
    ax.set_xlabel("# features", color="#c0c0c0")
    ax.set_ylabel("pooled AUC", color="#c0c0c0")
    ax.set_title("top-K vs random vs group-incremental", color="#e8e8e8")
    ax.legend(loc="lower right", facecolor="#0a0a0a", edgecolor="#2a2a2a",
              labelcolor="#c0c0c0", fontsize=9)
    ax.set_ylim(0.45, 0.95)
    ax.grid(True, alpha=0.15, color="#888888")

    fig.suptitle(
        "Cognometric Phase Transitions — does drift detection emerge in jumps?",
        color="#e8e8e8", fontsize=13, y=0.99,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor="#0a0a0a", bbox_inches="tight")
    plt.close(fig)
    print(f"  figure → {out_path}")


# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------

def write_report(top_k_results, group_results, random_results, ranking, transitions, out_path):
    sorted_top = sorted(top_k_results, key=lambda r: r["k"])
    lines = [
        "# Cognometric Phase Transitions — Feature-Count Scaling in the Drift Detector",
        "",
        "**Research question.** As features are added to the calibrated tool-call",
        "drift detector, does per-failure-type AUC improve smoothly (incremental",
        "learning) or in discrete jumps (phase transitions)? A phase-transition",
        "signal would be the mirror of emergent-capability literature — specific",
        "failure classes becoming *detectable* at specific feature-count",
        "thresholds rather than linearly with compute.",
        "",
        "**Dataset.** drift_v0, n=3,700 (658 gold no-drift + 3,042 drift positives",
        "via mutation + irrelevance splits), from Berkeley Function Calling",
        "Leaderboard v3. 5-fold stratified CV, random_state=0.",
        "",
        "**Protocol.** Three ablations:",
        "",
        "- **top-K by importance.** Rank the 22 features by |coef| from a full-",
        "  model baseline, then retrain with only the top-K. Traces the ceiling",
        "  at each K.",
        "- **group-incremental.** Add features in group order — A (semantic",
        "  alignment, 5) → B (schema conformance, 6) → C (lexical drift, 4) →",
        "  D (structural, 7). Shows which group is responsible for each failure.",
        "- **random subsets.** K features drawn uniformly at random, 3 seeds per",
        "  K. Null expectation for top-K.",
        "",
        "**Phase-transition criterion.** A per-drift-type AUC jump ≥ 0.10 between",
        "consecutive K values in the top-K schedule.",
        "",
        "---",
        "",
        "## Feature importance ranking (|coef| on full 22-feature model)",
        "",
        "| rank | feature | group |",
        "|------|---------|-------|",
    ]
    for rank, idx in enumerate(ranking):
        name = FEATURE_NAMES[idx]
        group = next((g for g, ids in GROUPS.items() if idx in ids), "?")
        lines.append(f"| {rank+1} | `{name}` | {group} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Top-K scaling curve")
    lines.append("")
    lines.append("| K | pooled AUC | arg_drop | spurious_arg | irrelevance_called | arg_swap |")
    lines.append("|---|------------|----------|--------------|--------------------|----------|")
    for r in sorted_top:
        pdt = r["per_drift_type_auc"]
        def fmt(x): return f"{x:.3f}" if x is not None else "—"
        lines.append(
            f"| {r['k']} | **{r['pooled_auc']:.3f}** | {fmt(pdt.get('arg_drop'))} "
            f"| {fmt(pdt.get('spurious_arg'))} | {fmt(pdt.get('irrelevance_called'))} "
            f"| {fmt(pdt.get('arg_swap'))} |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Detected phase transitions (Δ ≥ 0.10 between consecutive K)")
    lines.append("")
    if transitions:
        lines.append("| drift type | K: from→to | AUC: from→to | Δ | feature added |")
        lines.append("|---|---|---|---|---|")
        for t in transitions:
            added = ", ".join(f"`{f}`" for f in t["feature_added"])
            lines.append(
                f"| `{t['drift_type']}` | {t['from_k']} → {t['to_k']} | "
                f"{t['from_auc']:.3f} → {t['to_auc']:.3f} | **+{t['delta']:.3f}** "
                f"| {added} |"
            )
    else:
        lines.append("*None detected — learning is incremental across this range.*")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Group-incremental results")
    lines.append("")
    lines.append("| groups active | K | pooled AUC | arg_drop | spurious_arg | irrelevance_called | arg_swap |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in group_results:
        pdt = r["per_drift_type_auc"]
        def fmt(x): return f"{x:.3f}" if x is not None else "—"
        groups = "+".join([g[0] for g in r["groups_active"]])
        lines.append(
            f"| {groups} | {r['n_features']} | **{r['pooled_auc']:.3f}** "
            f"| {fmt(pdt.get('arg_drop'))} | {fmt(pdt.get('spurious_arg'))} "
            f"| {fmt(pdt.get('irrelevance_called'))} | {fmt(pdt.get('arg_swap'))} |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    if transitions:
        lines.append(
            "The top-K ablation exhibits **phase-transition behaviour**: specific"
        )
        lines.append(
            "failure classes remain near-null until a critical feature enters the"
        )
        lines.append(
            "set, then snap to high-AUC detection. This is the drift-detection"
        )
        lines.append(
            "analogue of emergent capabilities in generative LMs — detectability"
        )
        lines.append(
            "is not a smooth function of representational capacity."
        )
    else:
        lines.append(
            "No phase transitions cleared the Δ=0.10 threshold in this run. The"
        )
        lines.append(
            "scaling curve is smooth, suggesting incremental learning across the"
        )
        lines.append(
            "feature-count axis at this resolution. Re-running with a finer K"
        )
        lines.append(
            "schedule or a lower threshold may still surface sub-threshold jumps."
        )
    lines.append("")
    lines.append("**Top-K vs random subsets.** If top-K substantially outperforms random")
    lines.append("subsets at the same K, the ranking is non-trivial — there exists a")
    lines.append("small set of load-bearing features. If the gap is small, the")
    lines.append("classifier is additive and no single feature dominates.")
    lines.append("")
    lines.append("**Reproducer.** `python scripts/drift_feature_scaling.py`. Raw numbers:")
    lines.append("`benchmarks/drift_feature_scaling.json`. Figure:")
    lines.append("`papers/figures/drift_phase_transitions.png`.")
    lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  report → {out_path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print("== drift_feature_scaling.py ==")
    print(f"dataset: {DATA_PATH}")
    rows = load_dataset()
    print(f"  loaded {len(rows)} rows")

    X = np.array([extract_features(r) for r in rows], dtype=np.float64)
    y = np.array([int(bool(r.get("drift", False))) for r in rows])
    dtypes = np.array([(r.get("drift_type") or "gold") for r in rows])
    print(f"  X shape: {X.shape}   drift rate: {y.mean():.3f}")
    print(f"  drift_type counts: "
          f"{ {t: int((dtypes==t).sum()) for t in sorted(set(dtypes))} }")

    print("\n== ranking 22 features ==")
    ranking = rank_features(X, y)

    print("\n== ablation 1: top-K by importance ==")
    top_k = ablation_top_k(X, y, dtypes, ranking)

    print("\n== ablation 2: group-incremental ==")
    grp = ablation_group_incremental(X, y, dtypes)

    print("\n== ablation 3: random subsets ==")
    rnd = ablation_random_subsets(X, y, dtypes)

    print("\n== phase-transition detection ==")
    transitions = detect_phase_transitions(top_k, threshold=0.10)
    for t in transitions:
        print(f"  {t['drift_type']:20s}  K {t['from_k']}->{t['to_k']}  "
              f"AUC {t['from_auc']:.3f}->{t['to_auc']:.3f}  "
              f"delta=+{t['delta']:.3f}  (+feature: {t['feature_added']})")
    if not transitions:
        print("  none detected at delta>=0.10")

    out = {
        "methodology": {
            "dataset": str(DATA_PATH.relative_to(REPO)),
            "n_samples": len(rows),
            "n_features_total": 22,
            "feature_names": FEATURE_NAMES,
            "groups": GROUPS,
            "feature_counts": FEATURE_COUNTS,
            "random_subset_seeds": RANDOM_SUBSET_SEEDS,
            "cv_splits": N_SPLITS,
            "cv_random_state": RANDOM_STATE,
            "phase_transition_threshold": 0.10,
        },
        "ranking_by_importance": [FEATURE_NAMES[i] for i in ranking],
        "ranking_indices": ranking,
        "top_k_by_importance": top_k,
        "group_incremental": grp,
        "random_subsets": rnd,
        "phase_transitions": transitions,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  raw numbers → {OUT_JSON}")

    print("\n== figure ==")
    make_figure(top_k, grp, rnd, transitions, OUT_FIG)

    print("\n== report ==")
    write_report(top_k, grp, rnd, ranking, transitions, OUT_MD)

    print("\ndone.")


if __name__ == "__main__":
    main()
