# -*- coding: utf-8 -*-
"""styxx.audit_grounding — statistical-claim grounding: does every number in a claim trace to a receipt?

The discipline that catches overclaims, automated. A write-up, a model's self-report, or an LLM-generated
summary is full of numbers — RSA values, CIs, p-values, percentages, fold-changes. Each one should be
*traceable to a committed result*. Almost no one checks this mechanically, and it is exactly where overclaims
hide: a percentage computed against the wrong baseline, a CI that doesn't exist in any output, a "3x" with no
ratio behind it, a headline number that was never actually measured.

`audit_grounding` answers it deterministically — no LLM judge, no web. Given a claim (text) and its sources
(result JSONs / dicts / numbers), it extracts every statistical number and classifies each:

  - GROUNDED   — matches a value present in the sources (at the claim's displayed precision)
  - DERIVED    — a percentage (100*a/b) or fold-change (a/b) of two source values (e.g. 87% from 0.294/0.339)
  - UNSOURCED  — no backing value or derivation found  ← the ones to look at

It deliberately ignores non-statistical numbers (years, DOIs, arXiv ids, section refs). Companion to
``styxx.audit_confound`` (is a SCORE riding a confound?) and ``styxx.validate_probe`` (is a PROBE tracking the
concept?) — this asks the third integrity question: *is a CLAIM backed by data, or floating free?*

  from styxx import audit_grounding
  report = audit_grounding(paper_text, ["result_a.json", "result_b.json"])
  if report.unsourced:
      for n in report.unsourced: print("UNSOURCED:", n.raw)
  print(report.summary())
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# spans that look numeric but are identifiers/labels, not statistical claims — masked out before extraction
_SKIP = [
    re.compile(r"\b\d{1,2}\s*%\s*(?:CI|confidence)", re.I),  # "95% CI" — confidence-level label, not a claim
    re.compile(r"10\.\d{4,}/\S+"),               # DOI
    re.compile(r"arxiv:\s*\d{4}\.\d{4,}", re.I),  # arXiv id (labelled)
    re.compile(r"\b\d{4}\.\d{4,}\b"),             # bare arXiv-style id
    re.compile(r"§\s*\d+(?:\.\d+)*"),             # section reference
    re.compile(r"(?m)^\s{0,3}#{1,6}\s+\d+(?:\.\d+)*"),  # numbered markdown heading ("### 2.5 …")
    re.compile(r"(?m)^\s{0,3}\d+\.\s"),           # numbered list item ("1. Mitchell…")
    re.compile(r"\b(?:18|19|20)\d{2}\b"),         # year
    re.compile(r"\bgpt-?2\b", re.I),              # model names with digits
    re.compile(r"\bv\d+(?:\.\d+)*\b", re.I),      # version tags
]


def _decimals(s: str) -> int:
    s = s.split("e")[0].split("E")[0]
    return len(s.split(".")[1]) if "." in s else 0


def _flatten(obj: Any, path: str, out: dict) -> dict:
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        out.setdefault(round(float(obj), 6), path)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(v, f"{path}.{k}" if path else str(k), out)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _flatten(v, f"{path}[{i}]", out)
    elif isinstance(obj, str):
        for tok in re.findall(r"[-+]?\d*\.?\d+", obj.replace("−", "-")):
            try:
                out.setdefault(round(float(tok), 6), path)
            except ValueError:
                pass
    return out


def _load_sources(sources: Any) -> dict:
    """Return {numeric_value: provenance_path}. Accepts a dict/list, a path/JSON-string, or a list thereof."""
    items = sources if isinstance(sources, (list, tuple)) else [sources]
    vals: dict = {}
    for it in items:
        data, name = it, ""
        if isinstance(it, (str, Path)):
            p = Path(str(it))
            if p.exists():
                data, name = json.loads(p.read_text(encoding="utf-8")), p.name
            else:
                try:
                    data = json.loads(str(it))
                except Exception:
                    data = {"_": str(it)}
        _flatten(data, name, vals)
    return vals


@dataclass
class ClaimNumber:
    raw: str
    value: float
    kind: str                 # decimal | percent | pvalue | ci | range | multiplier
    status: str = "unsourced"  # grounded | derived | unsourced
    source: str = ""

    def __str__(self) -> str:
        tag = {"grounded": "ok", "derived": "~", "unsourced": "UNSOURCED"}[self.status]
        return f"[{tag}] {self.raw}" + (f"  ← {self.source}" if self.source else "")


@dataclass
class GroundingReport:
    items: list = field(default_factory=list)
    n_total: int = 0
    n_grounded: int = 0
    n_derived: int = 0
    n_unsourced: int = 0

    @property
    def unsourced(self) -> list:
        return [n for n in self.items if n.status == "unsourced"]

    @property
    def verdict(self) -> str:
        if self.n_total == 0:
            return "NO NUMERIC CLAIMS"
        return "ALL GROUNDED" if self.n_unsourced == 0 else f"UNSOURCED: {self.n_unsourced}/{self.n_total}"

    @property
    def pct_grounded(self) -> float:
        return 0.0 if self.n_total == 0 else round(100 * (self.n_grounded + self.n_derived) / self.n_total, 1)

    def summary(self) -> str:
        lines = [f"claim audit: {self.verdict}  ({self.pct_grounded}% backed; "
                 f"{self.n_grounded} grounded, {self.n_derived} derived, {self.n_unsourced} unsourced)"]
        for n in self.unsourced:
            lines.append(f"  UNSOURCED  {n.raw}  ({n.kind})")
        return "\n".join(lines)


def _extract(text: str) -> list:
    t = text.replace("−", "-")           # unicode minus → ASCII
    for rx in _SKIP:
        t = rx.sub(lambda m: " " * len(m.group()), t)
    spans: list = []
    nums: list = []

    def overlaps(a, b):
        return any(not (b <= s or a >= e) for s, e in spans)

    def scan(pattern, kind, groups=(0,)):
        for m in re.finditer(pattern, t):
            if overlaps(m.start(), m.end()):
                continue
            picked = []
            for g in groups:
                try:
                    picked.append(ClaimNumber(m.group(g), float(m.group(g)), kind))
                except (ValueError, IndexError):
                    picked = []
                    break
            if picked:
                spans.append((m.start(), m.end()))
                nums.extend(picked)

    scan(r"p\s*[=<>]\s*(\d*\.?\d+)", "pvalue", (1,))
    scan(r"\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]", "ci", (1, 2))
    # ranges (a–b): skip model-name digits (letter just before) and page/id ranges (big integers);
    # a range followed by % is a percent-range, matched against source*100
    for m in re.finditer(r"(\d*\.?\d+)\s*[–—-]\s*(\d*\.?\d+)", t):
        if overlaps(m.start(), m.end()) or (m.start() > 0 and t[m.start() - 1].isalpha()):
            continue
        a, b = float(m.group(1)), float(m.group(2))
        if a.is_integer() and b.is_integer() and a >= 1000 and b >= 1000:
            continue
        kind = "percent" if t[m.end():m.end() + 2].strip().startswith("%") else "range"
        spans.append((m.start(), m.end()))
        nums.append(ClaimNumber(m.group(1), a, kind))
        nums.append(ClaimNumber(m.group(2), b, kind))
    scan(r"(\d+(?:\.\d+)?)\s*%", "percent", (1,))
    scan(r"(\d+(?:\.\d+)?)\s*[×x]\b", "multiplier", (1,))
    # plain decimals (must carry a decimal point) in a plausible statistic range
    for m in re.finditer(r"[-+]?\d*\.\d+", t):
        if overlaps(m.start(), m.end()):
            continue
        v = float(m.group())
        if abs(v) <= 50:
            spans.append((m.start(), m.end()))
            nums.append(ClaimNumber(m.group(), v, "decimal"))
    return nums


def _match(num: ClaimNumber, vals: dict) -> tuple:
    d = _decimals(num.raw)
    tol = 0.5 * 10 ** (-d)
    for sv, path in vals.items():
        if num.kind == "percent":
            if round(sv * 100, d) == round(num.value, d) or round(sv, d) == round(num.value, d):
                return "grounded", path
        elif abs(sv - num.value) <= tol or round(sv, d) == round(num.value, d):
            return "grounded", path
    # derived — only for percentages and fold-changes (avoid spurious decimal ratios)
    if num.kind in ("percent", "multiplier"):
        svs = list(vals)
        dtol = max(tol, 0.1) if num.kind == "multiplier" else max(tol, 0.5)
        for a in svs:
            for b in svs:
                if b == 0 or a == b:
                    continue
                cand = (100 * a / b) if num.kind == "percent" else (a / b)
                if abs(cand - num.value) <= dtol:
                    return "derived", f"{vals[a]}/{vals[b]}"
    return "unsourced", ""


def audit_grounding(text: str, sources: Any) -> "GroundingReport":
    """Audit every statistical number in `text` against `sources` (result JSON paths / dicts / numbers).

    Returns a GroundingReport: each number is GROUNDED (present in sources), DERIVED (a %/ratio of two source
    values), or UNSOURCED. `report.unsourced` is the list to scrutinise; `report.verdict` is CI-gate friendly.
    """
    if isinstance(text, (str, Path)):
        try:
            p = Path(str(text))
            if p.exists() and p.is_file():
                text = p.read_text(encoding="utf-8")
        except (OSError, ValueError):
            pass
    text = str(text)
    vals = _load_sources(sources)
    nums = _extract(text)
    rep = GroundingReport(items=nums, n_total=len(nums))
    for n in nums:
        n.status, n.source = _match(n, vals)
        if n.status == "grounded":
            rep.n_grounded += 1
        elif n.status == "derived":
            rep.n_derived += 1
        else:
            rep.n_unsourced += 1
    return rep


def _cli(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="styxx audit-claims",
                                 description="Check that every statistical number in a claim traces to a receipt.")
    ap.add_argument("claim", help="path to the claim text/markdown (or the text itself)")
    ap.add_argument("sources", nargs="+", help="result JSON files (the receipts)")
    a = ap.parse_args(argv)
    rep = audit_grounding(a.claim, list(a.sources))
    print(rep.summary())
    return 1 if rep.n_unsourced else 0


if __name__ == "__main__":
    import sys
    sys.exit(_cli())
