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
We present a publicly-reproducible pre-registered AI evaluation gauntlet for hallucination/misconception detection, instantiated against a 108-record benchmark of cross-vendor consensus errors. The contribution is not the benchmark or the bars themselves, but a meta-property of the infrastructure: the gauntlet caught two of its own bar weaknesses in production within a single session, replaced them with regression-tested controls, and re-scored all existing submissions honestly under the strengthened bars. We document the discovery chain (D3 length-control discovered by accident → D4 capitalization-control discovered by systematic scan → ten pre-registered baselines, none of which clear the strengthened bars), and we record eight in-session falsifications of our own predictions — including two direction-of-effect misses that revealed a domain-specific calibration lesson invisible to AUC-magnitude prediction. We argue that the moat of disciplined AI evaluation is not the bars or the benchmark but the recursive pattern: bars revise under empirical pressure from real submissions; the discipline of pre-registration BEFORE each run makes wrongness visible rather than hidden; the infrastructure improves itself on a session timescale. All artifacts (benchmark, gauntlet code, ten baseline submissions, four FINDING documents, eight pre-stated-then-published predictions) are reconstructible from public git history at commit-level granularity.
```

### Comments (the "metadata comments" field)

```
14 pages including reproducibility table. All code, data, and predictions reconstructible from public git history at commit-level granularity. Companion artifacts: Zenodo concept DOI 10.5281/zenodo.19326174 (release-specific v25 = 10.5281/zenodo.20419662), PyPI styxx==7.7.9, GitHub fathom-lab/styxx tag v7.7.9.
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
