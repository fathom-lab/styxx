# Submission package — "Gold Anchors License Nothing"

Everything needed to deposit the flagship. The science is done and certified; this is the last mile.
**The agent prepared this package; every login-gated / public-posting step below is the operator's.**

## Contents
| file | what it is |
|---|---|
| `gold_anchors.tex` | arXiv-ready LaTeX (article class, arXiv-safe packages only) |
| `gold_anchors.pdf` | compiled PDF, 4 pages (built with pdflatex/MiKTeX) |
| `reproduce_gold_anchors.ipynb` | Colab reproducer — `pip install styxx`, prices an informative panel, refuses a deaf one (runs clean, executed end-to-end) |
| `zenodo_metadata.json` | Zenodo deposition metadata (fill the `OPERATOR_TO_*` fields) |
| `ANNOUNCEMENT_draft.md` | X thread + short note (house register) + the do-not-say list |
| `README_submission.md` | this file |

## What's new in this version (2026-07-23)
The paper now carries two extensions that close its own named residuals, both OATH-certified:
- **§6 In the wild** — a real heterogeneous judge panel (Gemini Flash-Lite + Qwen 3B/1.5B) on
  TruthfulQA; gold-anchor validity is *prevalence-dependent* (PARTIAL under the frozen forks, red-teamed
  before freeze). Receipts: `RESULT_/CERT_inthewild_truthfulqa`, `inthewild_truthfulqa_result.json`.
- **§7 The anchor threshold** — the impossibility made quantitative (one known-negative = 150.8× LR;
  ~30 give >90% power). Receipts: `FINDING_anchor_threshold`, `anchor_threshold_result.json`,
  `EXPLORE_anchor_threshold_2026_07_23.py`.

## Source of truth & fidelity
The **certified original** is `../PAPER_gold_anchors_license_nothing_2026_07_21.md`, now **OATH-HELD
88/0** against ten receipts (the original eight + the two above). `gold_anchors.tex` (5 pages) is a
faithful typeset transcription — **no existing number was changed**, all eighteen new numbers verified
present in both. To re-verify the source (pass all ten receipts):
```bash
python -m styxx.certify papers/anchored-validity/PAPER_gold_anchors_license_nothing_2026_07_21.md <receipts...> --out CERT.json
```

## arXiv (operator does the submit)
- **Categories:** primary `cs.LG`; cross-list `cs.CL`, `stat.ME`.
- **Submit the LaTeX source** (`gold_anchors.tex`) — arXiv compiles it server-side; the PDF is for preview.
  It uses only stock packages (`lmodern, geometry, amsmath, amssymb, microtype, booktabs, enumitem,
  hyperref, xcolor`), so the arXiv build should be clean.
- **Confirm before submit:** author name (currently `Alex Rodabaugh`, the designated authorship identity),
  affiliation (`Fathom Lab`), and add an ORCID if desired.
- Optional: add a short "Comments" line noting the styxx PyPI package + the reproducer notebook.

## Zenodo — DRAFT STAGED (operator publishes)
- **Done:** a new version **v29** is created and fully populated as a version of concept
  `10.5281/zenodo.19326174` (via the `newversion` action, so concept linkage is guaranteed — not an
  orphan). Draft id **21520429**, reserved DOI `10.5281/zenodo.21520429`.
- Author `Rodabaugh, Alexander / Fathom Lab`, `cc-by-4.0`, `preprint`; files `main.pdf` (extended
  5-page PDF), `source.md`, `source.certificate.json` (OATH 88/0) — matching every prior version.
- **Remaining (operator, irreversible/public):** review at `https://zenodo.org/deposit/21520429`,
  optionally add an ORCID, then **Publish** to register the DOI. Nothing else to fill in.

## Announcement (operator posts)
- Drafts in `ANNOUNCEMENT_draft.md`. Post after the DOI is live; drop the DOI into the thread's last line.
- Lead with "we broke ours first." Respect the do-not-say list.

## The one honest sentence for reviewers
The method can only **void** an eval, never **bless** one — passing an audit is necessary, not sufficient.
That scope string is on every certificate and in the paper; it is the difference between this being
defensible and being an overclaim.
