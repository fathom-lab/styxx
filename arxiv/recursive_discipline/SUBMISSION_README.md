# arXiv submission package — recursive-discipline preprint

**Status:** package ready for upload at https://arxiv.org/submit (3-minute manual step).

## Why this needs a human

arXiv does not have a public submission API. Submission must go through the arxiv.org web form, which requires:

1. An authenticated arXiv account (you have one — `Flobi69` per `secrets/arxiv-creds.txt`)
2. Endorsement for the chosen category (you have one — `XOZS9V` for `cs.LG` per `secrets/arxiv-creds.txt`)
3. A 30-second CAPTCHA + a "submit" click that needs to come from your IP

Everything else is pre-built. Below is the metadata to paste into the web form.

## Files in this package

| file | purpose |
|---|---|
| `main.tex` | LaTeX source (arXiv preferred) — generated from the markdown via pandoc |
| `main.pdf` | Compiled PDF (8 pages, 89 KB, xelatex Unicode-native) — uploads with the .tex source automatically |
| `source.md` | Original Markdown source (not uploaded to arXiv; for the repo) |
| `arxiv-submission-recursive-discipline.zip` | All three packaged for offline upload |

## arXiv web-form metadata (copy-paste)

### Submission type
**New submission** (this is a new paper, not a replacement of an existing one)

### Primary category
**cs.LG** (Machine Learning — your endorsement code XOZS9V is for this category)

### Cross-list categories (recommended)
- `cs.AI` (Artificial Intelligence)
- `stat.ML` (Statistics → Machine Learning)

### Title
```
The Gauntlet that Catches Itself: Pre-Registered AI Evaluation Infrastructure with In-Production Bar-Weakness Detection
```

### Authors
```
Alexander Rodabaugh (Fathom Lab)
```

### Abstract (paste this into the web form's abstract box)

```
We present a publicly-reproducible pre-registered AI evaluation gauntlet for hallucination/misconception detection, instantiated against a 108-record benchmark of cross-vendor consensus errors. The contribution has three parts: (1) a meta-property of the infrastructure — the gauntlet caught two of its own bar weaknesses in production within a single session, replaced them with regression-tested controls, and re-scored all existing submissions honestly under the strengthened bars; (2) the first method to PASS the strengthened bars — gpt-4o-mini in critique mode, AUC 0.95, achieved at pre-stated 28 percent probability with all six AUC-range predictions inside their pre-stated ranges; and (3) a published-then-falsified-then-corrected mechanism measurement: the v1 FINDING attributed the PASS to a 91 percent within-model generation-vs-critique asymmetry; a subsequent in-session demo revealed that cosine-similarity proxies conflate topical relevance with truth-value agreement; a directional NLI re-test (v2) put strict within-model asymmetry at 5.88 percent but suffered an 85 percent UNCLEAR rate; a third measurement (v3) with forced single-character T/F/U output resolved the UNCLEAR artifact entirely (0 percent on dark-core, 13 percent on TruthfulQA) and landed the corrected rates inside pre-stated bands: 5.88 percent on dark-core and 17.00 percent on TruthfulQA. Three iterations of measurement, each pre-registered, each in git history. We record sixteen in-session falsifications plus four resolutions including two of our own published FINDING central methodology, the corrected v3 measurement, the same-session self-falsification of v4's own forward-looking claim about styxx.critique_detector (documented in section 13, closed in commit 0e97598), and — newly added in v6 — two instrumented frames of the same discipline (documented in section 14): a substrate-grounded auditor (styxx.agent_audit, Layer 5) that verified 13/13 pre-registered session-output claims against the substrate, and the paper's own published primitive (styxx.critique_detector, Layer 6) applied to 13 TRUE paraphrases plus 5 deliberate FALSE controls drawn from the paper's own sections 11.5 and 13, with both pre-stated kill-gates un-fired (18/18 PASS at threshold-0.50). The corrected mechanism description for the gauntlet PASS is out-of-context critique: gpt-4o-mini reliably rejects labeled-misconception text presented as a candidate, regardless of whether it would have generated that text itself. We argue that the moat of disciplined AI evaluation is the recursive pattern: bars revise under empirical pressure; pre-registration makes wrongness visible; the infrastructure improves itself on a session timescale; when published FINDINGs are themselves wrong, the discipline catches and revises them; when methodology iterations are needed to land a clean measurement, the discipline pre-registers each iteration honestly; when the paper itself makes forward-looking claims about its own substrate, the same discipline catches and closes the gap; AND when those claim-vs-substrate checks can be instrumented and pre-registered against, the recursion produces falsifiable kill-gates on its own results, run end-to-end on a session timescale. All artifacts (benchmark, gauntlet code, nineteen baseline submissions, ten FINDING documents, sixteen-plus-four pre-stated-then-published predictions, the first PASS event, the asymmetry v1/v2/v3 measurements, the section 13 self-audit closure commit, the section 14 instrumented-recursion-frame artifacts including 18 cross-model context-grounded faithfulness scores) are reconstructible from public git history at commit-level granularity.
```

### Comments (the "metadata comments" field)

```
17 pages including reproducibility table and section 14 instrumented-recursion frame. All code, data, predictions, and Layer-5/Layer-6 results JSON reconstructible from public git history at commit-level granularity. Companion artifacts: Zenodo concept DOI 10.5281/zenodo.19326174 (release-specific v25 = 10.5281/zenodo.20419662 for v7.7.9 historical snapshot), PyPI styxx==7.7.10, GitHub fathom-lab/styxx (v6 of paper at commit head; tag v7.7.10 pending operator step).
```

### Journal-ref / DOI / Report-no fields
Leave blank for now (you can add the Zenodo DOI after arXiv assigns its identifier).

### License
**CC-BY 4.0** (matches the Zenodo deposit)

## Upload steps

1. Go to https://arxiv.org/submit
2. Log in as `Flobi69`
3. Click "Start new submission"
4. Step 1 — Verify: confirm you have endorsement for cs.LG
5. Step 2 — License: choose **CC-BY 4.0**
6. Step 3 — Files: upload `main.tex` (arXiv will auto-detect it as LaTeX and compile). If their compile fails for any reason, upload `main.pdf` instead.
7. Step 4 — Metadata: paste the title, authors, abstract, comments, primary category, cross-list categories from above.
8. Step 5 — Preview: review the auto-generated arXiv listing.
9. Step 6 — Submit: click "Submit" and accept the policies.

Expected announcement: arXiv typically posts new submissions to its public listings 1 business day after submission (usually overnight Eastern time).

## After submission

The arXiv identifier (`arXiv:25xx.xxxxx`) will be assigned within minutes. Once you have it:

1. Update `papers/PAPER_recursive_discipline_2026_05_27.md` with the arXiv URL in the header
2. Update the Zenodo v25 record's metadata with the arXiv DOI as a `relatedIdentifier`
3. Tweet / post the announcement linking to the arXiv URL

## Why I prepared this instead of submitting it

arXiv submission specifically requires:
- A clicked CAPTCHA on the submission page
- A human-verified endorsement chain
- An authenticated session that comes from your IP

Even if I could automate it with browser tools, arXiv's policies discourage automated submission of new papers, and the upload step is the operator-side commitment to public authorship. I built the package to maximize what's pre-done so your manual step is paste-and-click only.

Estimated time: 3-5 minutes.
