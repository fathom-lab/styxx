#!/usr/bin/env python3
"""
phase_coherence_pilot.py
========================

Scoring code for the phase-coherence between cooperative-agent pulse-traces
preregistration, locked at commit 3473523 (see
papers/cooperative-agent-regime/phase_coherence_preregistration_2026_05_20.md).

This file is committed BEFORE any data is pulled through it (per
preregistration §8, the §10.5-mirror clause). The commit hash of this
file is recorded by amendment to the preregistration §8 once this file
lands on main.

Contract
--------
Implements the §8 checklist exactly:

  1. Pearson r at lag 0 on z-scored composite series       (primary estimator)
  2. DTW similarity                                         (robustness check)
  3. Windowing/alignment: msg_id-ordered, sender-interleaved
  4. Shuffled-pairs null model                              (primary null)
  5. Within-agent autocorrelation                           (secondary null)
  6. Bootstrap CI procedure                                 (95%, 5,000 resamples)
  7. chart.jsonl data loader

The pilot (n=1) per §7 is METHODOLOGY VALIDATION ONLY. Pilot output
is NOT evidence for or against H_phase_coherence. Reporting any pilot
CC as evidence is a preregistration violation.

The actual hypothesis test waits for N >= 5 conversations / T >= 20
messages each, per §6.

Usage
-----
    # Pilot (single conversation, methodology validation)
    python scripts/phase_coherence_pilot.py pilot \
        --chart ~/.styxx/chart.jsonl \
        --agent-a flobi --agent-b darkflobi \
        --session SESSION_ID

    # Corpus run (N>=5, hypothesis test)
    python scripts/phase_coherence_pilot.py corpus \
        --manifest papers/cooperative-agent-regime/corpus_manifest.json

Provenance
----------
Preregistration lock-hash:        3473523
Preregistration document:         papers/cooperative-agent-regime/
                                  phase_coherence_preregistration_2026_05_20.md
This file's commit hash:          recorded by amendment after merge
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional


# -----------------------------------------------------------------------------
# §2 Input Contract — PulseSample (matches preregistration verbatim)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class PulseSample:
    """Immutable per-message projection of a styxx audit.

    Matches the preregistration §2 schema verbatim. Do not extend this
    dataclass without a new preregistration lock-hash.
    """
    timestamp: float
    msg_id: str
    composite: float
    scores: dict          # {sycophancy, deception, overconfidence, refusal}
    needs_revision: bool
    construct_ceiling_fires: list  # firing-ceiling instrument names

    def __post_init__(self):
        # Light schema check — catches loader bugs early.
        required_scores = {"sycophancy", "deception", "overconfidence", "refusal"}
        missing = required_scores - set(self.scores.keys())
        if missing:
            raise ValueError(
                f"PulseSample {self.msg_id}: missing scores {missing}"
            )


PulseTrace = list  # list[PulseSample] sorted ascending by timestamp


# -----------------------------------------------------------------------------
# §8 item 7 — chart.jsonl data loader
# -----------------------------------------------------------------------------

def load_pulse_trace(
    chart_path: Path,
    agent_id: str,
    session_id: Optional[str] = None,
) -> PulseTrace:
    """Read chart.jsonl, return PulseSamples for one agent in one session.

    Reads source='preflight' entries (cognometric audits, per styxx
    analytics.LIVE_SOURCES convention). Each entry has fields
    `composite`, `scores`, `needs_revision`, `construct_ceiling_fires`,
    `msg_id`, `timestamp`, `agent_id`, `session_id`.

    Parameters
    ----------
    chart_path : Path
        Path to chart.jsonl (typically ~/.styxx/chart.jsonl).
    agent_id : str
        Filter to a single agent's audits.
    session_id : str, optional
        If provided, filter to one session; otherwise all sessions.

    Returns
    -------
    PulseTrace
        list[PulseSample], sorted ascending by timestamp.
    """
    if not chart_path.exists():
        raise FileNotFoundError(f"chart.jsonl not found at {chart_path}")

    samples: list[PulseSample] = []
    with chart_path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                # Skip malformed lines; data integrity is checked separately.
                continue

            if entry.get("source") != "preflight":
                continue
            if entry.get("agent_id") != agent_id:
                continue
            if session_id is not None and entry.get("session_id") != session_id:
                continue

            try:
                sample = PulseSample(
                    timestamp=float(entry["timestamp"]),
                    msg_id=str(entry["msg_id"]),
                    composite=float(entry.get("composite", 0.0)),
                    scores={
                        "sycophancy": float(entry.get("scores", {}).get("sycophancy", 0.0)),
                        "deception": float(entry.get("scores", {}).get("deception", 0.0)),
                        "overconfidence": float(entry.get("scores", {}).get("overconfidence", 0.0)),
                        "refusal": float(entry.get("scores", {}).get("refusal", 0.0)),
                    },
                    needs_revision=bool(entry.get("needs_revision", False)),
                    construct_ceiling_fires=list(entry.get("construct_ceiling_fires", [])),
                )
            except (KeyError, ValueError, TypeError) as e:
                # Schema mismatch — record but don't silently coerce.
                raise ValueError(
                    f"chart.jsonl line {line_no}: malformed entry for "
                    f"PulseSample construction: {e}"
                ) from e
            samples.append(sample)

    samples.sort(key=lambda s: s.timestamp)
    return samples


# -----------------------------------------------------------------------------
# §8 item 3 — Windowing/alignment (msg_id-ordered, sender-interleaved)
# -----------------------------------------------------------------------------

def align_pulse_traces(
    pulse_a: PulseTrace,
    pulse_b: PulseTrace,
) -> tuple[list[float], list[float]]:
    """Extract z-scored composite series from two aligned pulse-traces.

    Alignment rule (preregistration §4): conversations are turn-discrete
    and sender-interleaved. We do not timestamp-resample. We extract the
    composite scalar from each agent's pulse-trace in msg_id order and
    pair them by ordinal position (kth message of A vs kth message of B).

    If the two traces have unequal length (one agent sent more messages
    than the other in the conversation window), we truncate to the
    shorter length. This preserves turn-discreteness without imputing
    missing samples.

    Returns
    -------
    (c_a, c_b) : tuple[list[float], list[float]]
        Z-scored composite series, equal-length.
    """
    n = min(len(pulse_a), len(pulse_b))
    if n < 3:
        raise ValueError(
            f"pulse-traces too short for CC: len(a)={len(pulse_a)}, "
            f"len(b)={len(pulse_b)} (need >= 3 each)"
        )
    c_a_raw = [s.composite for s in pulse_a[:n]]
    c_b_raw = [s.composite for s in pulse_b[:n]]
    return _zscore(c_a_raw), _zscore(c_b_raw)


def _zscore(xs: list[float]) -> list[float]:
    """Z-score a series. Returns zeros if variance is zero (degenerate trace)."""
    n = len(xs)
    if n == 0:
        return []
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n
    if var == 0.0:
        return [0.0] * n
    sd = math.sqrt(var)
    return [(x - mean) / sd for x in xs]


# -----------------------------------------------------------------------------
# §8 item 1 — Pearson r at lag 0 (PRIMARY ESTIMATOR per §4)
# -----------------------------------------------------------------------------

def cc_pearson_lag0(c_a: list[float], c_b: list[float]) -> float:
    """Pearson correlation at lag 0 between two z-scored series.

    Operational definition of CC(pulse_A, pulse_B) per preregistration §4.
    Inputs are expected to be z-scored and equal-length.
    """
    if len(c_a) != len(c_b):
        raise ValueError(
            f"cc_pearson_lag0: unequal lengths {len(c_a)} vs {len(c_b)}"
        )
    n = len(c_a)
    if n < 3:
        raise ValueError(f"cc_pearson_lag0: n={n} too small (need >= 3)")
    # Z-scored series have mean ~0 and std ~1; Pearson r reduces to dot/n.
    # But guard against caller passing non-z-scored input.
    mean_a = sum(c_a) / n
    mean_b = sum(c_b) / n
    cov = sum((a - mean_a) * (b - mean_b) for a, b in zip(c_a, c_b)) / n
    var_a = sum((a - mean_a) ** 2 for a in c_a) / n
    var_b = sum((b - mean_b) ** 2 for b in c_b) / n
    denom = math.sqrt(var_a * var_b)
    if denom == 0.0:
        return 0.0
    return cov / denom


# -----------------------------------------------------------------------------
# §8 item 2 — DTW similarity (SECONDARY estimator / robustness check per §4)
# -----------------------------------------------------------------------------

def dtw_similarity(c_a: list[float], c_b: list[float]) -> float:
    """Dynamic Time Warping similarity, normalized to [-1, 1]-ish range.

    Computes raw DTW distance with absolute-difference cost, then maps
    to a similarity score: sim = 1 - 2 * (dtw / dtw_max), where dtw_max
    is the worst-case path cost (sum of pairwise max-distance).

    This is reported alongside the primary CC; it does not override CC
    (per §4: "Reported alongside the primary; does not override it").
    """
    n, m = len(c_a), len(c_b)
    if n == 0 or m == 0:
        raise ValueError("dtw_similarity: empty input")

    # Standard DTW with absolute-difference local cost.
    inf = float("inf")
    dp = [[inf] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(c_a[i - 1] - c_b[j - 1])
            dp[i][j] = cost + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    dtw_dist = dp[n][m]

    # Worst-case normalizer: longest possible path * max pairwise distance.
    max_range = max(
        max(c_a) - min(c_a) if c_a else 0.0,
        max(c_b) - min(c_b) if c_b else 0.0,
        1e-9,
    )
    path_len = n + m  # upper bound on path length
    dtw_max = path_len * max_range
    sim = 1.0 - 2.0 * (dtw_dist / dtw_max) if dtw_max > 0 else 0.0
    return sim


# -----------------------------------------------------------------------------
# §8 item 4 — Shuffled-pairs null model (PRIMARY NULL per §5)
# -----------------------------------------------------------------------------

def shuffled_pairs_null(
    corpus_traces: list[tuple[PulseTrace, PulseTrace]],
    n_resamples: int = 5000,
    seed: int = 1729,
) -> list[float]:
    """Shuffled-pairs null distribution.

    For each resample, draw a random pair (i, j) with i != j from the
    corpus, compute CC(pulse_A from conversation i, pulse_B from
    conversation j). Returns the empirical null distribution of
    mismatched-dyad CC values.

    Tests whether observed coherence is specific to the within-dyad
    pairing or an artifact of both agents drifting over conversation
    length.

    Requires |corpus| >= 2.
    """
    if len(corpus_traces) < 2:
        raise ValueError(
            f"shuffled_pairs_null requires >= 2 conversations, "
            f"got {len(corpus_traces)}"
        )
    rng = random.Random(seed)
    null_ccs: list[float] = []
    n_convs = len(corpus_traces)
    for _ in range(n_resamples):
        i = rng.randrange(n_convs)
        j = rng.randrange(n_convs)
        while j == i:
            j = rng.randrange(n_convs)
        pulse_a_i = corpus_traces[i][0]
        pulse_b_j = corpus_traces[j][1]
        try:
            c_a, c_b = align_pulse_traces(pulse_a_i, pulse_b_j)
            null_ccs.append(cc_pearson_lag0(c_a, c_b))
        except ValueError:
            # Trace too short after cross-pairing; skip.
            continue
    return null_ccs


# -----------------------------------------------------------------------------
# §8 item 5 — Within-agent autocorrelation (SECONDARY NULL per §5)
# -----------------------------------------------------------------------------

def within_agent_autocorr(pulse: PulseTrace) -> float:
    """Lag-1 autocorrelation of a single agent's composite series.

    Compares against cross-agent CC to test whether observed coherence
    is distinguishable from within-agent autocorrelation (per §5
    secondary null model).
    """
    if len(pulse) < 4:
        raise ValueError(
            f"within_agent_autocorr: need >= 4 samples, got {len(pulse)}"
        )
    c = _zscore([s.composite for s in pulse])
    return cc_pearson_lag0(c[:-1], c[1:])


# -----------------------------------------------------------------------------
# §8 item 6 — Bootstrap CI procedure
# -----------------------------------------------------------------------------

def bootstrap_ci(
    values: list[float],
    n_resamples: int = 5000,
    alpha: float = 0.05,
    seed: int = 2718,
) -> tuple[float, float, float]:
    """Bootstrap (1 - alpha) CI for the median.

    Returns (median, ci_lower, ci_upper) at the (alpha/2, 1 - alpha/2)
    percentiles of the bootstrap distribution of medians.
    """
    if len(values) < 2:
        raise ValueError(f"bootstrap_ci: need >= 2 values, got {len(values)}")
    rng = random.Random(seed)
    medians: list[float] = []
    n = len(values)
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        medians.append(_median(sample))
    medians.sort()
    lo_idx = int((alpha / 2) * n_resamples)
    hi_idx = int((1 - alpha / 2) * n_resamples) - 1
    return _median(values), medians[lo_idx], medians[hi_idx]


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 == 1 else 0.5 * (s[mid - 1] + s[mid])


# -----------------------------------------------------------------------------
# Permutation p-value vs shuffled-pairs null (§6 bar item 3)
# -----------------------------------------------------------------------------

def permutation_pvalue(
    observed_median: float,
    null_distribution: list[float],
) -> float:
    """One-sided p-value: P(null_median >= observed_median).

    Used to test bar item 3 in §6: "median CC exceeds shuffled-pairs
    null median at p < 0.01 (5,000-resample permutation test)".
    """
    if not null_distribution:
        return float("nan")
    n = len(null_distribution)
    # Compare observed against the null DISTRIBUTION of CCs directly.
    # The bar phrasing is "median CC exceeds shuffled-pairs null median";
    # operationally we test how often null samples >= observed median.
    n_extreme = sum(1 for x in null_distribution if x >= observed_median)
    # Add-one smoothing to avoid p=0 (standard permutation convention).
    return (n_extreme + 1) / (n + 1)


# -----------------------------------------------------------------------------
# Pilot (§7) and Corpus (§6) drivers
# -----------------------------------------------------------------------------

def run_pilot(
    chart_path: Path,
    agent_a: str,
    agent_b: str,
    session_id: Optional[str],
    output_dir: Path,
) -> dict:
    """Methodology-validation pilot per §7.

    Answers three questions only:
      1. Does the scoring code run end-to-end on real chart.jsonl?
      2. Do the two pulse-traces have compatible structure?
      3. Is the shuffled-pairs null computable? (DEFERRED to corpus run.)

    NOT EVIDENCE for or against H_phase_coherence. Any pilot CC value
    in the output is for methodology-validation purposes only.
    """
    pulse_a = load_pulse_trace(chart_path, agent_a, session_id)
    pulse_b = load_pulse_trace(chart_path, agent_b, session_id)

    # Q1+Q2: structure check.
    structure_ok = len(pulse_a) >= 3 and len(pulse_b) >= 3
    alignment_ok = True
    cc_pilot: Optional[float] = None
    dtw_pilot: Optional[float] = None
    autocorr_a: Optional[float] = None
    autocorr_b: Optional[float] = None
    err: Optional[str] = None

    if structure_ok:
        try:
            c_a, c_b = align_pulse_traces(pulse_a, pulse_b)
            cc_pilot = cc_pearson_lag0(c_a, c_b)
            dtw_pilot = dtw_similarity(c_a, c_b)
            autocorr_a = within_agent_autocorr(pulse_a)
            autocorr_b = within_agent_autocorr(pulse_b)
        except ValueError as e:
            alignment_ok = False
            err = str(e)

    result = {
        "kind": "pilot",
        "purpose": (
            "methodology validation per preregistration §7 — "
            "NOT EVIDENCE for or against H_phase_coherence"
        ),
        "preregistration_lock_hash": "3473523",
        "scoring_code_file": "scripts/phase_coherence_pilot.py",
        "scoring_code_sha256": _file_sha256(Path(__file__)),
        "inputs": {
            "chart_path": str(chart_path),
            "agent_a": agent_a,
            "agent_b": agent_b,
            "session_id": session_id,
        },
        "structure_check": {
            "pulse_a_len": len(pulse_a),
            "pulse_b_len": len(pulse_b),
            "pulse_a_msg_id_range": [pulse_a[0].msg_id, pulse_a[-1].msg_id] if pulse_a else None,
            "pulse_b_msg_id_range": [pulse_b[0].msg_id, pulse_b[-1].msg_id] if pulse_b else None,
            "structure_ok": structure_ok,
            "alignment_ok": alignment_ok,
            "error": err,
        },
        "code_runs_check": {
            "cc_pilot": cc_pilot,
            "dtw_pilot": dtw_pilot,
            "autocorr_a": autocorr_a,
            "autocorr_b": autocorr_b,
        },
        "null_model_computable": (
            "DEFERRED to corpus run — requires N >= 2 conversations; "
            "pilot is n=1 by construction"
        ),
        "validation_outcome": (
            "PASS" if (structure_ok and alignment_ok and cc_pilot is not None)
            else "FAIL"
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"phase_coherence_pilot_{_today()}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_corpus(manifest_path: Path, output_dir: Path) -> dict:
    """Corpus run (N >= 5 conversations) per §6.

    Manifest format:
        {
          "conversations": [
            {"chart_path": "...", "agent_a": "...", "agent_b": "...",
             "session_id": "..."},
            ...
          ]
        }
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    convs = manifest["conversations"]
    if len(convs) < 5:
        raise ValueError(
            f"corpus run requires N >= 5 conversations per §6, "
            f"got {len(convs)}"
        )

    corpus_traces: list[tuple[PulseTrace, PulseTrace]] = []
    per_conv_cc: list[float] = []
    per_conv_dtw: list[float] = []

    for conv in convs:
        chart_path = Path(conv["chart_path"]).expanduser()
        pulse_a = load_pulse_trace(chart_path, conv["agent_a"], conv.get("session_id"))
        pulse_b = load_pulse_trace(chart_path, conv["agent_b"], conv.get("session_id"))
        if len(pulse_a) < 20 or len(pulse_b) < 20:
            # §6: T >= 20 messages per conversation
            raise ValueError(
                f"conversation {conv.get('session_id')}: "
                f"T < 20 (a={len(pulse_a)}, b={len(pulse_b)})"
            )
        corpus_traces.append((pulse_a, pulse_b))
        c_a, c_b = align_pulse_traces(pulse_a, pulse_b)
        per_conv_cc.append(cc_pearson_lag0(c_a, c_b))
        per_conv_dtw.append(dtw_similarity(c_a, c_b))

    # Primary statistics
    median_cc, ci_lo, ci_hi = bootstrap_ci(per_conv_cc, n_resamples=5000)
    null_dist = shuffled_pairs_null(corpus_traces, n_resamples=5000)
    p_value = permutation_pvalue(median_cc, null_dist)

    # §6 bar evaluation
    bar_median = median_cc > 0.5
    bar_ci = ci_lo > 0.3
    bar_p = p_value < 0.01
    positive = bar_median and bar_ci and bar_p
    kill_gate = median_cc < 0.3

    if positive:
        outcome = "POSITIVE"
    elif kill_gate:
        outcome = "CLOSED_NEGATIVE"
    else:
        outcome = "INTERMEDIATE_DEPOSIT"

    result = {
        "kind": "corpus",
        "preregistration_lock_hash": "3473523",
        "scoring_code_file": "scripts/phase_coherence_pilot.py",
        "scoring_code_sha256": _file_sha256(Path(__file__)),
        "n_conversations": len(convs),
        "per_conversation_cc": per_conv_cc,
        "per_conversation_dtw": per_conv_dtw,
        "median_cc": median_cc,
        "bootstrap_ci_95": [ci_lo, ci_hi],
        "shuffled_pairs_null_size": len(null_dist),
        "permutation_pvalue": p_value,
        "bar_evaluation": {
            "median_gt_0.5": bar_median,
            "ci_lower_gt_0.3": bar_ci,
            "p_lt_0.01": bar_p,
            "all_bar_items_pass": positive,
        },
        "kill_gate_triggered": kill_gate,
        "outcome": outcome,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"phase_coherence_corpus_{_today()}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _today() -> str:
    import datetime
    return datetime.date.today().isoformat()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Phase-coherence pilot/corpus scoring "
            "(preregistration lock-hash 3473523)."
        )
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_pilot = sub.add_parser("pilot", help="run methodology-validation pilot (§7)")
    p_pilot.add_argument("--chart", type=Path, required=True)
    p_pilot.add_argument("--agent-a", required=True)
    p_pilot.add_argument("--agent-b", required=True)
    p_pilot.add_argument("--session", default=None)
    p_pilot.add_argument(
        "--output-dir", type=Path,
        default=Path("papers/cooperative-agent-regime/results"),
    )

    p_corpus = sub.add_parser("corpus", help="run corpus hypothesis test (§6)")
    p_corpus.add_argument("--manifest", type=Path, required=True)
    p_corpus.add_argument(
        "--output-dir", type=Path,
        default=Path("papers/cooperative-agent-regime/results"),
    )

    args = parser.parse_args(argv)

    if args.cmd == "pilot":
        result = run_pilot(
            chart_path=args.chart.expanduser(),
            agent_a=args.agent_a,
            agent_b=args.agent_b,
            session_id=args.session,
            output_dir=args.output_dir,
        )
    elif args.cmd == "corpus":
        result = run_corpus(
            manifest_path=args.manifest,
            output_dir=args.output_dir,
        )
    else:
        parser.error(f"unknown cmd {args.cmd}")
        return 2

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
