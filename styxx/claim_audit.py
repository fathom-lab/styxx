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
    re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b"),     # ISO date YYYY-MM-DD (the MM-DD must not read as a range)
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
    """Effective decimal precision, exponent-aware so 1.2e-5 → 6 (not 1) and tolerance stays tight."""
    m = re.match(r"^[-+]?\d*\.?(\d*)(?:[eE]([-+]?\d+))?$", s.replace("−", "-").strip())
    if not m:
        s2 = s.split("e")[0].split("E")[0]
        return len(s2.split(".")[1]) if "." in s2 else 0
    frac = len(m.group(1) or "")
    exp = int(m.group(2)) if m.group(2) else 0
    return max(0, frac - exp)


def _flatten(obj: Any, path: str, out: dict) -> dict:
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        out.setdefault(round(float(obj), 6), path)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, (int, float)) and not isinstance(k, bool):  # numeric keys are values too ({0.294: "a"})
                out.setdefault(round(float(k), 6), f"{path}.<key>" if path else "<key>")
            _flatten(v, f"{path}.{k}" if path else str(k), out)
    elif isinstance(obj, (list, tuple, set, frozenset)):
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
    items = sources if isinstance(sources, (list, tuple, set, frozenset)) else [sources]
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
class OverclaimFlag:
    severity: str   # high | med | low
    label: str
    snippet: str
    why: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.label}: …{self.snippet}…  ({self.why})"


# heuristic overclaim linter — pattern-based flags for human review, NOT ground truth. Complements the
# deterministic grounding: grounding asks "is the number real?", this asks "does the LANGUAGE reach past it?"
_OVERCLAIM = [
    ("high", re.compile(r"\b(first|the only|no one else|never before|unprecedented|world[- ]?first)\b", re.I),
     "priority", "unverified priority/'first' — confirm the literature doesn't already own it"),
    ("high", re.compile(r"\bstatistically\s+indistinguishable\b|\bequivalent to\b|\bon par with\b|\bas good as\b|\bjust as well\b", re.I),
     "equivalence", "equivalence asserted from a failed rejection needs a TOST/equivalence test, not a non-significant p"),
    ("med", re.compile(r"\b(proves|proven|confirms|establishes|definitive(ly)?|guarantees?)\b", re.I),
     "certainty", "a single study rarely 'proves' — soften to 'is consistent with'"),
    ("med", re.compile(r"\b(causes?|caused by|because of|drives?|leads to|due to|responsible for)\b", re.I),
     "causal", "correlational evidence (RSA/encoding) cannot license causal language"),
    ("low", re.compile(r"\b(breakthrough|revolutionary|game[- ]?chang\w*|paradigm[- ]?shift|stunning|telepath\w*|mind[- ]?read\w*|solv\w*\s+telepathy)\b", re.I),
     "hype", "hype register undercuts a rigor claim — state the measurement instead"),
]


def detect_overclaims(text: str) -> list:
    """Heuristic linter: flag language that may reach past the data (priority/equivalence/causal/hype/
    uncontrolled-robustness). Pattern-based — flags for review, not verdicts."""
    if not isinstance(text, str):
        try:
            p = Path(str(text))
            text = p.read_text(encoding="utf-8") if p.exists() and p.is_file() else str(text)
        except (OSError, ValueError):
            text = str(text)

    def ctx(m):
        s, e = max(0, m.start() - 28), min(len(text), m.end() + 28)
        return text[s:e].replace("\n", " ").strip()

    neg = re.compile(r"\b(not|never|no|n't|isn't|aren't|wasn't|without)\W*$", re.I)
    flags = []
    for sev, rx, label, why in _OVERCLAIM:
        for m in rx.finditer(text):
            if neg.search(text[max(0, m.start() - 16):m.start()]):
                continue  # negated ("not first", "not mind-reading") — honest, skip
            flags.append(OverclaimFlag(sev, label, ctx(m), why))
    # "survives / robust / holds" with no CI or p-value within the next ~80 chars
    for m in re.finditer(r"\b(surviv\w*|robust|holds?)\b", text, re.I):
        if not re.search(r"\[|\bCI\b|p\s*[=<]|%", text[m.start():m.start() + 80], re.I):
            flags.append(OverclaimFlag("med", "uncontrolled-robustness", ctx(m),
                                       "'survives/robust' with no CI, p, or % nearby — show the number"))
    return flags


@dataclass
class GroundingReport:
    items: list = field(default_factory=list)
    overclaims: list = field(default_factory=list)
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
        if self.overclaims:
            lines.append(f"overclaim linter: {len(self.overclaims)} flag(s) for review")
            for f in self.overclaims:
                lines.append(f"  [{f.severity}] {f.label}: …{f.snippet}…")
        return "\n".join(lines)

    def render_html(self, title: str = "claim audit") -> str:
        """A self-contained styxx-brand card (flat, aubergine/lilac, monospace) summarising the audit."""
        import html as _h
        ok = self.n_unsourced == 0
        bcol = "#65E0D8" if ok else "#F2B45C"
        pct = self.pct_grounded
        fill = round(100 * (self.n_grounded + self.n_derived) / max(self.n_total, 1))
        dot = {"high": "#FF5C6C", "med": "#F2B45C", "low": "#8C7AA6"}
        uns = "".join(
            f'<span style="display:inline-block;margin:2px 4px 2px 0;padding:2px 8px;border:1px solid #4A3A5C;'
            f'border-radius:3px;color:#F2B45C;font-size:12px">{_h.escape(n.raw)}</span>' for n in self.unsourced[:16])
        ocs = "".join(
            f'<div style="display:flex;gap:8px;align-items:flex-start;margin:7px 0">'
            f'<span style="flex:none;width:8px;height:8px;border-radius:50%;margin-top:5px;background:{dot.get(f.severity,"#8C7AA6")}"></span>'
            f'<div><span style="color:#C9A2F0;font-size:12px;text-transform:uppercase;letter-spacing:.04em">{_h.escape(f.label)}</span>'
            f'<div style="color:#9A86B4;font-size:12.5px;margin-top:1px">…{_h.escape(f.snippet)}…</div></div></div>'
            for f in self.overclaims[:8])
        return f"""<h2 class="sr-only">styxx claim audit: {pct}% of {self.n_total} numeric claims grounded in the data; {self.n_unsourced} unsourced; {len(self.overclaims)} overclaim flags.</h2>
<div style="background:#1a1020;border:1px solid #33264a;border-radius:10px;padding:22px 24px;font-family:'JetBrains Mono',ui-monospace,Menlo,monospace;color:#F3ECF7;max-width:620px">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <div style="color:#C9A2F0;font-size:13px;letter-spacing:.06em">styxx ▸ {_h.escape(title)}</div>
    <div style="color:{bcol};border:1px solid {bcol};border-radius:4px;padding:3px 9px;font-size:11px;letter-spacing:.05em">{_h.escape(self.verdict)}</div>
  </div>
  <div style="margin:18px 0 6px"><span style="font-size:42px;font-weight:700;color:#F3ECF7">{pct}%</span>
    <span style="color:#9A86B4;font-size:13px;margin-left:8px">of {self.n_total} numeric claims trace to a receipt</span></div>
  <div style="height:9px;border-radius:5px;background:#2a1d38;overflow:hidden;margin:10px 0 16px">
    <div style="height:100%;width:{fill}%;background:linear-gradient(90deg,#C9A2F0,#65E0D8)"></div></div>
  <div style="display:flex;gap:18px;font-size:12.5px;color:#B79FCB;margin-bottom:6px">
    <span><b style="color:#C9A2F0">{self.n_grounded}</b> grounded</span>
    <span><b style="color:#C9A2F0">{self.n_derived}</b> derived</span>
    <span><b style="color:#F2B45C">{self.n_unsourced}</b> unsourced</span></div>
  {'<div style="margin-top:12px"><div style="color:#6E5A82;font-size:11px;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px">unsourced — review these</div>'+uns+'</div>' if uns else ''}
  {'<div style="margin-top:16px;border-top:1px solid #2a1d38;padding-top:12px"><div style="color:#6E5A82;font-size:11px;text-transform:uppercase;letter-spacing:.08em">overclaim linter · '+str(len(self.overclaims))+' flag(s)</div>'+ocs+'</div>' if self.overclaims else ''}
  <div style="margin-top:16px;color:#6E5A82;font-size:10.5px;border-top:1px solid #2a1d38;padding-top:10px">fathom-lab · styxx.audit_grounding — deterministic grounding + heuristic overclaim linter</div>
</div>"""


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

    scan(r"p\s*[=<>]\s*(\d*\.?\d+(?:[eE][-+]?\d+)?)", "pvalue", (1,))
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
    for m in re.finditer(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", t):
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
    rep.overclaims = detect_overclaims(text)
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
