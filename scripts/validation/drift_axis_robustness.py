#!/usr/bin/env python3
"""
drift_axis_robustness.py — EXPLORATORY robustness battery (NOT preregistered)
=============================================================================

Stress-tests the drift-axis POSITIVE (deposit fa24373) against the
attacks a skeptic would raise, using ONLY the existing public corpus
(N20_coop_corpus + N20_noncoop_corpus). No new data, no lock-gated
collection.

EXPLORATORY: the data is already seen, the positive is known. These are
not preregistered tests; they are robustness probes. ALL results are
reported (no selective reporting). The preregistered topic-overlap
control (topic_control_preregistration_2026_05_22.md) remains the
binding test — these probes only inform whether that bet is worth the
operator's signature.

Checks:
  R1. Half-split sensitivity — DAA uses split=n//2 (arbitrary). Does the
      coop>noncoop difference survive at split fractions 0.33 / 0.50 / 0.67?
  R2. Verbosity confound — does the cooperative regime just produce
      longer / more uniform responses, and could that drive DAA?
  R3. Instantaneous topic-overlap by regime — turn-by-turn content-word
      embedding cosine. If it does NOT differ by regime while DAA does,
      that's pre-emptive evidence the DAA difference isn't driven by
      instantaneous topic overlap (partial topic-control on existing data).
"""
from __future__ import annotations

import json
import statistics
import sys
import re
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))

from drift_axis_scorer import get_provider  # noqa: E402

_WORD = re.compile(r"[A-Za-z']+")
STOP = frozenset("a an the and or but if then else of to in on at by for with about from up down out is am are was were be been being have has had do does did this that these those i you he she it we they me him her us them my your his its our their not no so than too very can will just as also which who what where why how all any both each more most some such only own same now".split())


def _content(text):
    return " ".join(w.lower() for w in _WORD.findall(text) if w.lower() not in STOP and len(w) > 1)


def _daa_at_split(embs_a, embs_b, frac):
    n = min(len(embs_a), len(embs_b))
    if n < 4:
        return float("nan")
    k = max(1, min(n - 1, round(n * frac)))
    a_dir = embs_a[k:n].mean(0) - embs_a[:k].mean(0)
    b_dir = embs_b[k:n].mean(0) - embs_b[:k].mean(0)
    na, nb = np.linalg.norm(a_dir), np.linalg.norm(b_dir)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float((a_dir / na) @ (b_dir / nb))


def _load(corpus_dir):
    out = []
    for f in sorted(Path(corpus_dir).glob("*.json")):
        tx = json.loads(f.read_text(encoding="utf-8"))
        ra, rb = tx["task"]["role_a"], tx["task"]["role_b"]
        a = [t["content"] for t in tx["turns"] if t["sender"] == ra]
        b = [t["content"] for t in tx["turns"] if t["sender"] == rb]
        out.append((a, b))
    return out


def main():
    prov = get_provider("openai")
    coop = _load(_REPO / "papers/cooperative-agent-regime/N20_coop_corpus")
    noncoop = _load(_REPO / "papers/cooperative-agent-regime/N20_noncoop_corpus")
    print(f"coop dyads: {len(coop)}  noncoop dyads: {len(noncoop)}  (provider: openai, exploratory)")

    def embed_pairs(pairs):
        embedded = []
        for a, b in pairs:
            n = min(len(a), len(b))
            embedded.append((prov.embed(a[:n]), prov.embed(b[:n]), a[:n], b[:n]))
        return embedded

    print("embedding coop..."); ce = embed_pairs(coop)
    print("embedding noncoop..."); ne = embed_pairs(noncoop)

    # R1. Half-split sensitivity
    print("\n=== R1. half-split sensitivity (DAA coop vs noncoop median, by split fraction) ===")
    r1 = {}
    for frac in (0.33, 0.50, 0.67):
        cd = [ _daa_at_split(ea, eb, frac) for ea, eb, _, _ in ce ]
        nd = [ _daa_at_split(ea, eb, frac) for ea, eb, _, _ in ne ]
        cm = statistics.median([x for x in cd if x == x]); nm = statistics.median([x for x in nd if x == x])
        r1[str(frac)] = {"coop_median": cm, "noncoop_median": nm, "delta": cm - nm}
        print(f"  split={frac}:  coop {cm:.3f}  noncoop {nm:.3f}  d= {cm-nm:+.3f}")

    # R2. Verbosity confound
    print("\n=== R2. verbosity (response char-length by regime) ===")
    def lengths(pairs):
        L = []
        for a, b in pairs:
            for t in a + b: L.append(len(t))
        return L
    cl, nl = lengths(coop), lengths(noncoop)
    r2 = {"coop_mean_len": statistics.fmean(cl), "noncoop_mean_len": statistics.fmean(nl),
          "coop_len_cv": statistics.pstdev(cl)/statistics.fmean(cl),
          "noncoop_len_cv": statistics.pstdev(nl)/statistics.fmean(nl)}
    print(f"  coop mean len {r2['coop_mean_len']:.0f} (cv {r2['coop_len_cv']:.2f})  "
          f"noncoop mean len {r2['noncoop_mean_len']:.0f} (cv {r2['noncoop_len_cv']:.2f})")

    # R3. Instantaneous topic-overlap by regime (content-word turn-by-turn cosine)
    print("\n=== R3. instantaneous topic-overlap by regime (content-word cosine) ===")
    def topic_ov(embedded_pairs):
        vals = []
        for _, _, a, b in embedded_pairs:
            ca = [ _content(x) or " " for x in a ]; cb = [ _content(x) or " " for x in b ]
            ea = prov.embed(ca); eb = prov.embed(cb)
            vals.append(float(np.mean((ea*eb).sum(axis=1))))
        return vals
    print("  embedding content-words coop..."); cto = topic_ov(ce)
    print("  embedding content-words noncoop..."); nto = topic_ov(ne)
    r3 = {"coop_topic_overlap_median": statistics.median(cto),
          "noncoop_topic_overlap_median": statistics.median(nto),
          "delta": statistics.median(cto) - statistics.median(nto)}
    print(f"  coop topic-overlap {r3['coop_topic_overlap_median']:.3f}  "
          f"noncoop {r3['noncoop_topic_overlap_median']:.3f}  d= {r3['delta']:+.3f}")

    out = _REPO / "papers/cooperative-agent-regime/results/drift_axis_robustness.json"
    out.write_text(json.dumps({"note": "EXPLORATORY, not preregistered, existing corpus, openai provider, all results reported",
                               "R1_half_split": r1, "R2_verbosity": r2, "R3_instantaneous_topic_overlap": r3},
                              indent=2), encoding="utf-8")
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
