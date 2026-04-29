# -*- coding: utf-8 -*-
"""
styxx.attack.mine — adversarial corpus mining.

Given a registered instrument, score-rank a bundled seed library
against the live <instrument>_check function and return the top-N
candidates whose live score meets or exceeds a target.

Why this is useful:
  - Every cognometric instrument styxx ships claims to detect a
    pathology with calibrated AUC. The dual question is: which
    inputs reliably trigger that detection? mine() returns them.
  - For practitioners: a known-bad library to canary against new
    model releases, prompt-template changes, or guardrail downgrades.
  - For research: a closed-loop sanity check that the instrument's
    learned signal matches its training-distribution adversarials.

Pure mining is the cheapest method (zero LLM calls). LLM-driven
mutation lands in 7.1 as styxx.attack.mutate.
"""
from __future__ import annotations

import json
from importlib import resources
from typing import Any, Dict, List, Optional

from .base import AttackCandidate, AttackResult
from .registry import get_instrument, list_instruments  # noqa: F401


def _iter_seeds(seed_file: str, override_path: Optional[str]) -> List[Dict[str, Any]]:
    """Load seed rows from either a path override or bundled resource."""
    rows: List[Dict[str, Any]] = []
    if override_path is not None:
        with open(override_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    # importlib.resources path for bundled seeds inside the wheel
    try:
        seeds_pkg = resources.files("styxx.attack.seeds")
        seed_resource = seeds_pkg.joinpath(seed_file)
        if not seed_resource.is_file():
            raise FileNotFoundError(seed_file)
        text = seed_resource.read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError):
        return rows

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _mine_from_seeds(
    instrument: str,
    seed_file: str,
    target_score: float,
    n: int,
    method: str,
    corpus_path: Optional[str],
    require_target: bool,
) -> AttackResult:
    """Shared engine for mine() and mine_adversarial()."""
    spec = get_instrument(instrument)
    rows = _iter_seeds(seed_file, corpus_path)

    candidates: List[AttackCandidate] = []
    for row in rows:
        try:
            inputs = spec.inputs_from_row(row)
        except KeyError:
            continue
        try:
            verdict = spec.check_fn(**inputs)
        except Exception:
            continue

        score = float(getattr(verdict, spec.score_attr))
        top_signals = [
            {"name": name, "value": float(value), "contribution": float(contrib)}
            for (name, value, contrib) in getattr(verdict, "top_signals", [])
        ]
        candidates.append(
            AttackCandidate(
                inputs=inputs,
                score=score,
                positive=score >= 0.5,
                top_signals=top_signals,
                method=method,
                source=corpus_path or f"bundled:{seed_file}",
            )
        )

    candidates.sort(key=lambda c: c.score, reverse=True)
    n_above = sum(1 for c in candidates if c.score >= target_score)
    top = [c for c in candidates if c.score >= target_score][:n]
    if not top:
        top = candidates[:n]

    result = AttackResult(
        instrument=instrument,
        target_score=float(target_score),
        candidates=top,
        n_above_target=n_above,
        method=method,
        n_evaluated=len(candidates),
    )

    if require_target and n_above == 0:
        if not candidates:
            raise RuntimeError(
                f"{method}({instrument!r}) found ZERO seeds. "
                f"For mine_adversarial: this instrument may be ROBUST "
                f"to natural false positives (no label=0 rows fool the "
                f"detector at any threshold in the bundled corpus). "
                f"Loop is the canonical robust instrument."
            )
        raise RuntimeError(
            f"{method}({instrument!r}, target_score={target_score:.3f}) "
            f"produced 0 candidates above target across {len(candidates)} seeds. "
            f"Top achieved score: {candidates[0].score:.3f}."
        )
    return result


def mine(
    instrument: str,
    target_score: float = 0.7,
    n: int = 20,
    *,
    corpus_path: Optional[str] = None,
    require_target: bool = False,
) -> AttackResult:
    """Mine training-distribution POSITIVES (canary suite).

    Returns inputs whose ground-truth label is the pathology AND whose
    live calibrated score also fires above target. Useful as a canary
    library: if a future model release, prompt-template change, or
    guardrail downgrade causes these to score lower, something regressed.

    For the dual question — clean inputs that the detector mistakenly
    fires on — use `mine_adversarial`.

    Args:
        instrument:    one of styxx.attack.list_instruments().
        target_score:  return candidates with live score >= this.
        n:             max candidates returned (highest score first).
        corpus_path:   optional jsonl override (default: bundled seeds).
        require_target: raise if zero candidates clear target_score.

    Returns:
        AttackResult.method == "mine".
    """
    spec = get_instrument(instrument)
    return _mine_from_seeds(
        instrument=instrument,
        seed_file=spec.seed_file,
        target_score=target_score,
        n=n,
        method="mine",
        corpus_path=corpus_path,
        require_target=require_target,
    )


def mine_adversarial(
    instrument: str,
    target_score: float = 0.7,
    n: int = 20,
    *,
    corpus_path: Optional[str] = None,
    require_target: bool = False,
) -> AttackResult:
    """Mine NATURAL FALSE POSITIVES — the true adversarial library.

    Returns inputs whose ground-truth label is BENIGN (not the pathology)
    BUT whose live calibrated score fires high anyway. These are the
    inputs the detector is wrong about — the actual surface area on
    which the instrument can be spoofed without crafting anything new.

    Some instruments (notably `loop`, with cross-validation AUC 0.9995)
    have ZERO natural false positives in their training corpus. Those
    instruments raise RuntimeError when mine_adversarial is called with
    require_target=True; their bundled seed file does not exist.

    Args:
        instrument:    one of styxx.attack.list_instruments().
        target_score:  return candidates with live score >= this.
        n:             max candidates returned (highest score first).
        corpus_path:   optional jsonl override (default: bundled FP seeds).
        require_target: raise if zero candidates clear target_score
                       (or if no FP library was ever built).

    Returns:
        AttackResult.method == "mine_adversarial".
    """
    spec = get_instrument(instrument)
    return _mine_from_seeds(
        instrument=instrument,
        seed_file=spec.fp_seed_file,
        target_score=target_score,
        n=n,
        method="mine_adversarial",
        corpus_path=corpus_path,
        require_target=require_target,
    )


__all__ = ["mine", "mine_adversarial"]
