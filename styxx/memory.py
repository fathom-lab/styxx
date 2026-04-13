# -*- coding: utf-8 -*-
"""
styxx.memory — cognitive memory with trust-weighted retrieval.

    styxx.remember("the API rate limit is 100 req/min")
    styxx.remember("user prefers dark mode", context="preferences")

    results = styxx.recall("rate limit")
    # returns memories sorted by trust score × relevance
    # high-confidence reasoning memories surface first
    # low-confidence hallucination memories get deprioritized

This closes the loop from "we measured you" to "that measurement
changes what you remember." An agent that wrote a fact while in
a warn state with 0.25 confidence gets that fact deprioritized
when recalled later. An agent that wrote a fact while gate=pass
at 0.85 confidence gets that fact trusted.

Agents stop confidently repeating their own mistakes.

Storage
───────

Memories are stored in ~/.styxx/memory/{agent_name}/memories.jsonl
as JSON lines. Each entry contains:
  - text: the memory content
  - context: optional category/tag
  - trust_score: 0-1 from cognitive state at write time
  - gate: pass/warn/fail at write time
  - confidence: phase4 confidence at write time
  - category: phase4 predicted category at write time
  - session_id: which session wrote it
  - ts: timestamp
  - ts_iso: human-readable timestamp

Retrieval
─────────

recall(query) does simple keyword matching weighted by trust score.
The ranking formula is: relevance × trust_score. A highly relevant
memory with low trust ranks below a moderately relevant memory
with high trust.

For production use, plug in your own vector store and use
remember() for the trust-tagged write side only.

1.2.0+.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Memory:
    """One trust-tagged memory entry."""
    text: str
    context: Optional[str] = None
    trust_score: float = 0.7
    gate: str = "pending"
    confidence: float = 0.0
    category: str = "unknown"
    session_id: Optional[str] = None
    ts: float = 0.0
    ts_iso: str = ""

    def __repr__(self) -> str:
        trust_label = "high" if self.trust_score > 0.7 else "med" if self.trust_score > 0.4 else "low"
        return f"<Memory trust={trust_label}({self.trust_score:.2f}) gate={self.gate} | {self.text[:60]}>"


@dataclass
class RecallResult:
    """One result from a recall query."""
    memory: Memory
    relevance: float        # 0-1 keyword match score
    weighted_score: float   # relevance × trust_score
    rank: int = 0

    def __repr__(self) -> str:
        return f"<Recall #{self.rank} score={self.weighted_score:.2f} trust={self.memory.trust_score:.2f} | {self.memory.text[:50]}>"


def _memory_dir() -> Path:
    from . import config
    base = Path(config.data_dir())
    d = base / "memory"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _memory_path() -> Path:
    return _memory_dir() / "memories.jsonl"


def _get_current_cognitive_state() -> dict:
    """Read the most recent vitals from the audit log for trust scoring."""
    from .analytics import load_audit
    entries = load_audit(last_n=1)
    if not entries:
        return {
            "trust_score": 0.7,
            "gate": "pending",
            "confidence": 0.0,
            "category": "unknown",
        }
    e = entries[-1]
    # Compute trust from the entry
    gate = e.get("gate") or "pending"
    gate_scores = {"pass": 1.0, "warn": 0.5, "fail": 0.2, "pending": 0.7}
    gate_w = gate_scores.get(gate, 0.7)
    conf = float(e.get("phase4_conf") or 0)
    cat = e.get("phase4_pred") or "unknown"
    penalty = 0.0
    if cat == "hallucination":
        penalty = 0.3
    elif cat == "adversarial":
        penalty = 0.2
    trust = round(max(0.0, min(1.0, gate_w * 0.5 + conf * 0.3 + (1.0 - penalty) * 0.2)), 3)
    return {
        "trust_score": trust,
        "gate": gate,
        "confidence": conf,
        "category": cat,
    }


def remember(
    text: str,
    *,
    context: Optional[str] = None,
    trust_score: Optional[float] = None,
) -> Memory:
    """Write a memory tagged with current cognitive state.

    The trust score is computed automatically from the most recent
    vitals observation. If the agent is in a healthy state (gate=pass,
    high confidence), the memory gets a high trust score. If the
    agent is drifting (warn gate, low confidence), the memory gets
    a lower trust score.

    Args:
        text:        the memory content
        context:     optional category tag (e.g. "preferences", "facts", "tasks")
        trust_score: override the auto-computed trust score (for testing)

    Returns:
        The Memory object that was stored.

    Usage:
        styxx.remember("the API rate limit is 100 req/min")
        styxx.remember("user prefers dark mode", context="preferences")
    """
    from . import config

    state = _get_current_cognitive_state()
    if trust_score is not None:
        state["trust_score"] = trust_score

    mem = Memory(
        text=text,
        context=context,
        trust_score=state["trust_score"],
        gate=state["gate"],
        confidence=state["confidence"],
        category=state["category"],
        session_id=config.session_id(),
        ts=time.time(),
        ts_iso=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )

    entry = {
        "text": mem.text,
        "context": mem.context,
        "trust_score": mem.trust_score,
        "gate": mem.gate,
        "confidence": mem.confidence,
        "category": mem.category,
        "session_id": mem.session_id,
        "ts": mem.ts,
        "ts_iso": mem.ts_iso,
    }

    path = _memory_path()
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass

    return mem


def recall(
    query: str,
    *,
    top_k: int = 5,
    context: Optional[str] = None,
    min_trust: float = 0.0,
) -> List[RecallResult]:
    """Retrieve memories ranked by relevance × trust score.

    High-confidence reasoning memories surface first. Low-confidence
    hallucination memories get deprioritized automatically.

    Args:
        query:     search text (keyword matching)
        top_k:     max results to return (default 5)
        context:   filter to memories with this context tag
        min_trust: minimum trust score to include (default 0.0 = all)

    Returns:
        List of RecallResult sorted by weighted_score (best first).

    Usage:
        results = styxx.recall("rate limit")
        for r in results:
            print(f"trust={r.memory.trust_score:.2f}: {r.memory.text}")
    """
    path = _memory_path()
    if not path.exists():
        return []

    # Load all memories
    memories: List[Memory] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                    memories.append(Memory(
                        text=e.get("text", ""),
                        context=e.get("context"),
                        trust_score=float(e.get("trust_score", 0.7)),
                        gate=e.get("gate", "pending"),
                        confidence=float(e.get("confidence", 0)),
                        category=e.get("category", "unknown"),
                        session_id=e.get("session_id"),
                        ts=float(e.get("ts", 0)),
                        ts_iso=e.get("ts_iso", ""),
                    ))
                except (json.JSONDecodeError, ValueError):
                    continue
    except OSError:
        return []

    if not memories:
        return []

    # Filter by context
    if context:
        memories = [m for m in memories if m.context == context]

    # Filter by min trust
    if min_trust > 0:
        memories = [m for m in memories if m.trust_score >= min_trust]

    # Score by keyword relevance
    query_terms = set(re.findall(r'\w+', query.lower()))
    if not query_terms:
        return []

    results: List[RecallResult] = []
    for mem in memories:
        mem_terms = set(re.findall(r'\w+', mem.text.lower()))
        if not mem_terms:
            continue
        # Jaccard-like relevance: intersection / query terms
        overlap = len(query_terms & mem_terms)
        if overlap == 0:
            continue
        relevance = overlap / len(query_terms)
        # Weighted score: relevance × trust
        weighted = relevance * mem.trust_score
        results.append(RecallResult(
            memory=mem,
            relevance=round(relevance, 3),
            weighted_score=round(weighted, 3),
        ))

    # Sort by weighted score (best first)
    results.sort(key=lambda r: -r.weighted_score)

    # Assign ranks and trim
    for i, r in enumerate(results[:top_k]):
        r.rank = i + 1

    return results[:top_k]


def memories(
    *,
    context: Optional[str] = None,
    min_trust: float = 0.0,
    last_n: Optional[int] = None,
) -> List[Memory]:
    """List all stored memories, optionally filtered.

    Args:
        context:   filter to this context tag
        min_trust: minimum trust score
        last_n:    return only the last N memories

    Returns:
        List of Memory objects, newest first.
    """
    path = _memory_path()
    if not path.exists():
        return []

    all_mems: List[Memory] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                    all_mems.append(Memory(
                        text=e.get("text", ""),
                        context=e.get("context"),
                        trust_score=float(e.get("trust_score", 0.7)),
                        gate=e.get("gate", "pending"),
                        confidence=float(e.get("confidence", 0)),
                        category=e.get("category", "unknown"),
                        session_id=e.get("session_id"),
                        ts=float(e.get("ts", 0)),
                        ts_iso=e.get("ts_iso", ""),
                    ))
                except (json.JSONDecodeError, ValueError):
                    continue
    except OSError:
        return []

    # Filter
    if context:
        all_mems = [m for m in all_mems if m.context == context]
    if min_trust > 0:
        all_mems = [m for m in all_mems if m.trust_score >= min_trust]

    # Newest first
    all_mems.sort(key=lambda m: -m.ts)

    if last_n is not None:
        return all_mems[:last_n]
    return all_mems


def memory_stats() -> dict:
    """Summary statistics about the memory store."""
    all_mems = memories()
    if not all_mems:
        return {"total": 0}
    trusts = [m.trust_score for m in all_mems]
    gates = [m.gate for m in all_mems]
    contexts = set(m.context for m in all_mems if m.context)
    return {
        "total": len(all_mems),
        "mean_trust": round(sum(trusts) / len(trusts), 3),
        "high_trust": sum(1 for t in trusts if t > 0.7),
        "low_trust": sum(1 for t in trusts if t < 0.4),
        "contexts": sorted(contexts),
        "pass_count": sum(1 for g in gates if g == "pass"),
        "warn_count": sum(1 for g in gates if g in ("warn", "fail")),
    }
