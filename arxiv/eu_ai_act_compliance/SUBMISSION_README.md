# arXiv submission — EU AI Act Compliance Bridge for styxx (v0.1)

Companion artifact to `papers/EU_AI_ACT_COMPLIANCE_2026.md` in the styxx repo. Operator-uploadable to arXiv (cs.CY primary, cs.AI secondary).

---

## What's in this directory

| file | purpose |
|---|---|
| `source.md` | Canonical Markdown source (mirrored from `papers/EU_AI_ACT_COMPLIANCE_2026.md` at the publishing commit) |
| `main.tex` | LaTeX generated via `pandoc source.md --standalone -V geometry:margin=1in` |
| `main.pdf` | Compiled PDF (7 pages, two pdflatex passes for cross-references) |
| `arxiv-submission-eu-ai-act-compliance.zip` | Zip of `main.tex` + `main.pdf` + `source.md` for arXiv form upload |

---

## arXiv form fields (paste-ready)

### Title
```
A pre-registered, falsifiable measurement-methodology bridge from styxx primitives to EU AI Act Article 15 / Annex III requirements (v0.1)
```

### Authors
```
Alexander Rodabaugh (Fathom Lab)
```

### Abstract (paste into the web form's abstract box)

```
The EU AI Act high-risk obligations enter enforcement on 2 August 2026 with penalties up to EUR 15 million or 3 percent of global annual turnover. Article 15 mandates that high-risk AI systems achieve appropriate levels of accuracy, robustness, and cybersecurity, that accuracy metrics be declared in the instructions of use, and -- under paragraph 2 -- that the Commission shall, in cooperation with relevant stakeholders, encourage the development of benchmarks and measurement methodologies. The Commission's invitation is open. No competing AI observability or evaluation product currently publishes a structured Article 15 mapping. This paper introduces styxx.compliance.eu_ai_act, the first open-source, pre-registration-disciplined measurement-methodology bridge mapping a deployable cognitive-observability primitive set to specific Article 15 sub-paragraphs, with calibrated metrics, explicit construct-ceiling disclosures, commit-level reproducibility receipts, and pre-stated falsification criteria. The v0.1 mapping covers four Article 15 clauses with five styxx primitives; it explicitly enumerates seven uncovered EU AI Act requirements (Articles 9, 10, 12, 13, 14, 15 cybersecurity, 15.4 bias) and points each at a non-styxx alternative tool or methodology. The boundary statement is at least as long as the coverage statement by design (kill-gate A3). Five pre-registered kill-gates define what success and failure look like for the v0.1 release. This document is a measurement methodology, not legal advice; it is the kind of artifact Article 15 paragraph 2 explicitly invites stakeholders to develop.
```

### Comments (the "metadata comments" field)

```
7 pages. v0.1 minimum-viable bridge published 65 days before the 2 August 2026 EU AI Act high-risk system enforcement deadline. Companion to the recursive-discipline paper at fathom-lab/styxx (PAPER_recursive_discipline_2026_05_27.md v7). Open-source MIT licensed module: styxx.compliance.eu_ai_act in styxx==7.7.10. 15 tests enforce pre-registered kill-gates A1 (specific Article subparagraph citations), A2 (construct ceiling >= 50 chars on every primitive), A3 (uncovered requirements list >= covered list). Five pre-registered falsification criteria F1-F5 with defined response paths. Not legal advice. Independent conformity review required for production deployment. Citation strategy: >=1 independent citation by 2027-02-01 or methodology reassessed.
```

### Subject classification (arXiv primary + cross-listing)

- **Primary**: `cs.CY` (Computers and Society) — regulatory methodology, governance bridge
- **Cross-list**: `cs.AI` (Artificial Intelligence) — substrate primitives, agent observability

### Journal-ref / DOI / Report-no fields

Leave blank initially. After arXiv assigns its identifier, the Zenodo deposit DOI can be added as a related identifier in a subsequent metadata update.

### License

**CC-BY 4.0** (matches the styxx repo public release pattern; the companion code is MIT)

---

## Upload steps (operator)

1. Sign in at https://arxiv.org with the Fathom Lab account
2. New submission → Upload `arxiv-submission-eu-ai-act-compliance.zip`
3. Subject classification: cs.CY primary, cs.AI cross-list
4. Paste Title, Authors, Abstract, Comments from above
5. License: CC-BY 4.0
6. Preview the rendered PDF; confirm 7 pages
7. Submit; expect 1-2 business-day moderation

---

## Suggested cross-citations (optional — adds these to the body if iterating to v0.2)

- arXiv 2502.03407 (Goldowsky-Dill et al., *Detecting Strategic Deception Using Linear Probes*) — Apollo Research linear-probe approach to deception detection, AUROC 0.96-0.999 on contrived scenarios. Relevant context for Article 15.1 ("consistent performance throughout lifecycle").
- arXiv 2511.22662 (*Difficulties with Evaluating a Deception Detector for AIs*) — field's current methodology limitations; styxx's pre-registration discipline is one proposed answer.
- UK AISI Inspect AI framework — institutional comparator for evaluation methodologies; styxx covers a different layer (agent-side primitives vs task-scoring evals).
- EU AI Act Article 40 + 41 — harmonised standards procedure; this paper is an Article 15 ¶2 stakeholder contribution, not a harmonised standard.

---

## What this submission is, and is NOT

**IS:** an open-source stakeholder methodology contribution under EU AI Act Article 15 paragraph 2; the first publicly-known structured mapping of agent cognitive-observability primitives to specific Article 15 sub-paragraphs with calibrated metrics, construct ceilings, and commit-level reproducibility receipts; published before the 2 August 2026 enforcement deadline to give regulated operators an evaluation runway.

**IS NOT:** legal advice; a substitute for an operator's own AI Act conformity assessment; an endorsement by the European Commission, AISI, METR, Apollo Research, or any standardization body; sufficient on its own for EU AI Act conformity (Articles 9, 10, 12, 13, 14 are explicitly outside v0.1 scope); a claim that styxx primitives have been validated by a notified body.

---

## Related artifacts (in the same repo)

- `papers/PAPER_recursive_discipline_2026_05_27.md` v7 — the underlying methodology paper documenting six layers of recursive self-falsification
- `arxiv/recursive_discipline/` — that paper's arXiv submission package (17 pages, also operator-uploadable)
- `styxx/compliance/eu_ai_act.py` — the bridge module (~280 LOC)
- `tests/test_compliance_eu_ai_act.py` — 15 tests enforcing the kill-gates
- `.styxx/STRATEGIC_LANDSCAPE_2026_05_28.md` — strategic positioning context (operator-private)
