# arXiv submission — NIST AI RMF Bridge for styxx (v0.1)

Companion artifact to `papers/NIST_AI_RMF_BRIDGE_2026.md`. Parallel jurisdictional bridge to the EU AI Act bridge at `arxiv/eu_ai_act_compliance/`. Operator-uploadable to arXiv (cs.CY primary, cs.AI secondary).

---

## Title (paste into form)
```
A pre-registered, falsifiable measurement-methodology bridge from styxx primitives to NIST AI RMF 1.0 Measure-function subcategories (v0.1)
```

## Authors
```
Alexander Rodabaugh (Fathom Lab)
```

## Abstract (paste into form)

```
NIST AI 100-1 (the AI Risk Management Framework 1.0, January 2023) defines four core functions for AI risk management: Govern, Map, Measure, Manage. The Measure function is its analytical engine -- 22 subcategories that evaluate AI systems against seven trustworthy characteristics. The RMF is voluntary in US federal contexts but is increasingly referenced by federal procurement, state-level legislation, and private-sector contracting. This paper introduces styxx.compliance.nist_ai_rmf, a parallel jurisdictional bridge to the EU AI Act compliance bridge introduced the same day. It maps styxx primitives to five Measure subcategories (MS-2.3, MS-2.4, MS-2.5, MS-2.6, MS-2.13) with calibrated metrics, construct ceilings, commit-level reproducibility receipts, and the same three kill-gates (A1 specific subcategory citations, A2 construct ceilings disclosed, A3 uncovered greater than or equal to covered) enforced by tests. v0.1 explicitly enumerates six uncovered Measure subcategories (MS-2.7, MS-2.8, MS-2.9 partial, MS-2.10, MS-2.11, MS-2.12) with alternative-tool references. The bridge shares dataclasses with the EU AI Act bridge through styxx.compliance._common, signaling that the methodology pattern transfers across regulatory regimes. Not legal advice; independent review required for production deployment.
```

## Comments

```
6 pages. v0.1 jurisdictional companion to the EU AI Act Article 15 bridge (arXiv companion paper at fathom-lab/styxx papers/EU_AI_ACT_COMPLIANCE_2026.md). Demonstrates that the styxx.compliance methodology pattern transfers across regulatory regimes. NIST AI RMF is voluntary US framework; this bridge shows direct mapping to five MS-2.x Measure subcategories with the same kill-gate discipline (A1/A2/A3) as the EU bridge. Open-source MIT under styxx 7.7.10. Not legal advice. Tests at tests/test_compliance_nist_ai_rmf.py enforce kill-gates A1-A3 at CI time.
```

## Subject classification

- **Primary**: `cs.CY` (Computers and Society)
- **Cross-list**: `cs.AI` (Artificial Intelligence)

## License

**CC-BY 4.0**

## Upload steps (operator)

1. Sign in at https://arxiv.org with the Fathom Lab account
2. Upload `arxiv-submission-nist-ai-rmf-bridge.zip`
3. Subject: cs.CY primary, cs.AI cross-list
4. Paste Title, Authors, Abstract, Comments from above
5. License: CC-BY 4.0
6. Preview the rendered PDF (6 pages); confirm
7. Submit; 1-2 business-day moderation

## Cross-citation pattern

Reference the EU AI Act bridge paper in the submission notes: ties the two bridges together as a single methodology pattern applied to two regimes. Operator can later request a hyperlinked cross-citation if both papers receive arXiv identifiers.

## What this submission is, and is NOT

**IS:** the first publicly-known structured mapping of agent cognitive-observability primitives to NIST AI RMF 1.0 Measure subcategories with calibrated metrics, construct ceilings, commit-level receipts, and pre-registered kill-gates.

**IS NOT:** legal advice; a complete NIST AI RMF implementation (Govern/Map/Manage functions are out of scope); endorsement by NIST or any standards body.

## Related artifacts in the same repo

- `papers/EU_AI_ACT_COMPLIANCE_2026.md` — EU AI Act bridge v0.1 (companion)
- `arxiv/eu_ai_act_compliance/` — EU bridge arxiv submission package (7 pages)
- `papers/PAPER_recursive_discipline_2026_05_27.md` v7 — underlying methodology paper (17 pages)
- `arxiv/recursive_discipline/` — methodology paper arxiv submission package
- `styxx/compliance/nist_ai_rmf.py` — the NIST bridge module (~280 LOC)
- `styxx/compliance/_common.py` — shared dataclasses across both bridges
- `tests/test_compliance_nist_ai_rmf.py` — 15 tests enforcing kill-gates A1-A3
