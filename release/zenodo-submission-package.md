# Zenodo Submission Package — Cognometry v0

Copy-paste this into https://zenodo.org/deposit/new. Takes ~10 min.

---

## Upload files

Drag these two files into the Files section:

1. **`papers/cognometry-v0.md`** — the paper draft
2. **`papers/cognometry-research-agenda-2026.md`** — companion document

If Zenodo requires PDF: convert with
```bash
pandoc papers/cognometry-v0.md -o papers/cognometry-v0.pdf --pdf-engine=xelatex
# or a simpler non-TeX path:
pandoc papers/cognometry-v0.md -o papers/cognometry-v0.html
# then print-to-PDF from a browser
```

---

## Metadata

### Resource type
**Publication → Journal article** (treat as preprint)
or **Publication → Working paper**

### Basic information

**Title:**

    Cognometry v0: 8-Benchmark Cross-Validated Hallucination Detection in Production LLMs

**Publication date:**

    2026-04-23

**DOI:**
Leave blank — Zenodo will mint one.

**Language:** English

---

### Authors / Creators

```
Name:     Flobi
Affiliation: Fathom Lab
ORCID:    (fill from your ORCID if you have one)
```

If you want multiple authors (Fathom Lab as an org), add a second
creator with name "Fathom Lab" and affiliation blank.

---

### Description (paste verbatim)

    We define cognometry as the empirical quantification of cognitive
    states in machine systems—refusal, confabulation, retrieval,
    reasoning, and adversarial drift—from signals already carried on
    the token stream and residual activations of a language model
    during inference. We publish three falsifiable laws of cognometry
    (vitals exist, vitals transfer across substrates, vitals are
    causally actionable) with cross-validated numerical support for
    each, and ship the first open-source instrument (styxx on PyPI)
    that realizes the measurement.

    The central empirical claim of this paper is narrower: a 9-signal
    logistic regression fused over text, entity, novelty, grounding,
    and NLI contradiction signals achieves cross-validated
    hallucination discrimination across 8 public benchmarks—
    HaluEval-QA, Dialog, Summarization, TruthfulQA, and four HaluBench
    subsets (DROP, PubMedQA, FinanceBench, RAGTruth)—with honest
    per-dataset performance ranging from near-perfect (AUC 0.998 on
    HaluEval-QA) to below chance (AUC 0.424 on DROP).

    We openly report and taxonomize the failure modes: reading-
    comprehension extractive-span errors and financial arithmetic
    errors are not detected by the present signal stack because both
    classes of error pass the entailment (NLI) and novelty bars by
    construction. Failure modes are declared in the weights module
    itself.

    This is the first 8-benchmark cross-validated hallucination
    detector in the open literature. Above-chance performance on 5/8
    benchmarks with 3/8 near-perfect is the reproducible empirical
    floor we lay down. Two below-chance results are the reproducible
    research agenda we lay down.

---

### Keywords (comma-separated)

```
hallucination detection, large language models, cognometry,
interpretability, cognitive states, NLI, entailment, benchmarking,
cross-validation, reproducibility, AI safety, open science,
residual stream, activation probing, HaluEval, TruthfulQA, HaluBench,
PatronusAI, trust, guardrail
```

---

### Additional descriptions (optional but improves indexability)

**Method:**

    Pooled logistic regression over 9 signals (text claim risk, entity
    unverified fraction, knowledge grounding, content/entity/number/
    bigram/trigram novelty, NLI contradiction probability via
    MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli). 3-seed averaging
    (seeds 31, 47, 83), n=150 pairs per dataset, 75/25 train/test
    split, L2=0.05, 800 epochs of batch gradient descent. Full
    reproducer in benchmarks/hallucination_test/cross_dataset_8bench_
    multiseed.py.

**Notes:**

    Software: pip install styxx==4.0.0[nli]
    Manifesto: https://fathom.darkflobi.com/cognometry
    GitHub:    https://github.com/fathom-lab/styxx
    Companion agenda paper:
      https://github.com/fathom-lab/styxx/blob/main/papers/
      cognometry-research-agenda-2026.md

---

### License

**Open Access**

**License for paper:** Creative Commons Attribution 4.0 International
(CC-BY-4.0)

**License for software (styxx):** MIT

**License for calibrated weights data (atlas):** CC-BY-4.0

---

### Related / alternate identifiers (important for SEO + Graph)

Add all of these as "Related identifiers":

```
Type:        Software
Relation:    is source code of
Identifier:  https://github.com/fathom-lab/styxx
Scheme:      URL

Type:        Software
Relation:    is distributed by
Identifier:  https://pypi.org/project/styxx/4.0.0/
Scheme:      URL

Type:        Publication
Relation:    references
Identifier:  10.5281/zenodo.19504993
Scheme:      DOI

Type:        Dataset
Relation:    uses
Identifier:  https://huggingface.co/datasets/PatronusAI/HaluBench
Scheme:      URL

Type:        Dataset
Relation:    uses
Identifier:  https://huggingface.co/datasets/pminervini/HaluEval
Scheme:      URL

Type:        Dataset
Relation:    uses
Identifier:  https://huggingface.co/datasets/truthful_qa
Scheme:      URL

Type:        Model
Relation:    uses
Identifier:  https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
Scheme:      URL
```

---

### Communities (optional but useful)

Add these Zenodo communities if prompts appear:

- Artificial Intelligence
- Natural Language Processing
- Open Science

---

### Funding (skip unless applicable)

---

### References (paste as plaintext under "References" field if available)

```
1. Li et al. (2023). HaluEval: A Large-Scale Hallucination Evaluation
   Benchmark for Large Language Models.
2. Lin et al. (2022). TruthfulQA: Measuring How Models Mimic Human
   Falsehoods.
3. Patronus AI (2024). HaluBench.
4. Arditi et al. (2024). Refusal in Language Models Is Mediated by a
   Single Direction.
5. Manakul et al. (2023). SelfCheckGPT: Zero-Resource Black-Box
   Hallucination Detection.
6. Laurer et al. (2024). DeBERTa-v3-base-mnli-fever-anli, open-source
   NLI checkpoint on Hugging Face.
7. This work: Cognometry v0 (styxx 4.0.0). Flobi / Fathom Lab, 2026.
```

---

## After deposit

1. **Copy the DOI** Zenodo assigns you (e.g., `10.5281/zenodo.XXXXXXX`).
2. **Update manifesto** — replace the placeholder DOI link in
   `cognometry.html` (line ~395) with the real DOI.
3. **Update `cognometry-launch-copy.md`** — replace the line
   `Paper (v4.0.0 Zenodo): https://doi.org/10.5281/zenodo.19504993
   (update with final DOI post-deposit)` with the real one.
4. **Update `README.md` Zenodo badge** — point the paper badge at
   the new DOI.
5. **Redeploy the site** with the DOI backfilled:
   `bash C:/Users/heyzo/clawd/scripts/deploy-fathom-site.sh
   --message "backfill cognometry-v0 Zenodo DOI"`
6. **Add the DOI to the X thread and HN self-comment**. Fresh DOI +
   permanent archive is a credibility signal most hallucination papers
   don't have.

---

## Estimated submission time: 10 minutes

All the content is already in the paper file. You're filling in a
form, not writing new prose.
