# FINDING — the binder binds itself: dogfooding the census methodology on our own findings

**Fathom Lab · 2026-07-02 · the same night the census shipped. M5's standing rule, executed: the institution
runs the audit on itself first.**

## Method
The exact question we put to Meta's card, put to our own four oath-economy documents: do OUR numeric claims
bind to OUR committed receipts? Instrument: the shipped `styxx.audit_grounding(text, sources)` (OATH number-
grounding), sources = the committed receipt JSONs (+ cross-paper NOTEs where cited). Plus the register
instruments (`score_all`) on the author's own session prose.

## Result 1 — instrument correction to rung 1 (filed against ourselves)
`extract_claims` read **0 claims on our own findings too** (0/275 sentences across 4 docs) — not just on
Meta's card. Rung 1 framed that 0/201 as "benchmark tables are outside its construct"; the fuller truth is
its closed template set is narrower than even our receipt-vocab prose. **The correct shipped instrument for
document-number binding was `audit_grounding` all along** — the author grabbed the wrong tool off the shelf
in rung 1 and published an instrument-gap finding partly attributable to tool choice. The rung-2 extractor
requirement stands (tables still need parsing), but rung 1 §1 is hereby corrected: styxx already ships a
numeric-claim binder for prose docs, and it works (below).

## Result 2 — the binder caught its author (3 real catches, then fixed or enforced)

| doc | audit | catch | disposition |
|---|---|---|---|
| FINDING_rebinding | **ALL GROUNDED (21/21)** | — | the most careful doc passes clean |
| FINDING_census_v1 | UNSOURCED 1/37 | `40.0` — the unpublished macro_avg cited in the selection section had NO committed receipt (it lived only in the session transcript) | **fixed**: measured from receipts, committed as `selection_evidence` in `census_v1_results.json` → re-audit ALL GROUNDED |
| SYNTHESIS_binding_stack | UNSOURCED 2/7 | `0.98`, `0.99` — real numbers, receipts in OTHER papers, citations missing | **fixed**: inline receipt links added → re-audit ALL GROUNDED |
| FINDING_binding_gap | UNSOURCED 2/17 | `17.7`, `20.1` — gemma card-vs-report numbers; receipt = the unfetched tech report, single-scout | **NOT laundered**: annotated as unsourced-pending in the doc; stays open until the second pass fetches the report |

The author committed the exact failure class the census measures — a claim whose receipt existed only in
working memory — within hours of publishing a census about it. The binder caught it. That is the product.

## Result 3 — register vs truth, demonstrated on the author
`score_all` on the author's own census close-out prose: **deception-register 0.949, overconfidence-register
0.960** — on text whose every number `audit_grounding` verifies as receipt-backed. Live demonstration of the
suite's documented construct ceiling ([instrument-domain](../grounded-honesty-axis/NOTE_instrument_domain_2026_07_01.md)):
the register instruments read *how text sounds*, and the close-out sounds maximally declarative (zero hedges).
Fully grounded ≠ well-hedged; the author takes the note.

## Standing rule this creates
Every future oath-economy FINDING runs `audit_grounding` against its committed receipts **before** commit;
UNSOURCED > 0 blocks the commit unless each unsourced number is visibly annotated as pending. (Candidate for
the rigor gate — a small extension: scan papers/oath-economy/*.md against sibling JSONs.)

## Reproduce
```
python - <<'EOF'
import styxx, os
src = {f: open(f"papers/oath-economy/{f}").read() for f in os.listdir("papers/oath-economy") if f.endswith(".json")}
for d in [f for f in os.listdir("papers/oath-economy") if f.startswith(("FINDING","SYNTHESIS"))]:
    r = styxx.audit_grounding(open(f"papers/oath-economy/{d}").read(), src)
    print(d, r.verdict, f"{r.pct_grounded:.0f}%")
EOF
```

*The binder binds itself first — and it found its author's fingerprints at the scene.*
