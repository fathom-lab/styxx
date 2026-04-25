# PATENTS — styxx / Fathom Intelligence

styxx is open-source under the MIT license, but several of the
methods it implements are covered by provisional US patent filings
held by **Fathom Intelligence / Alexander Rodabaugh**. This file
documents those filings so users know exactly what's covered and
how to cite them.

## Provisional filings

All three filings are USPTO provisional applications. Inventor of
record: Alexander Rodabaugh. Assignee: Fathom Intelligence. Filing
dates and confirmation numbers are from the public Fathom research
repo's patent disclosures.

### US Provisional 64/020,489

**Title:** System and method for measuring reasoning depth and
integrated computational geometry in artificial neural networks.

**Scope:** Attribution-weighted mean-layer computation (the K axis),
the Fathom Constant K = 1.0343, and the cross-architecture validation
methodology that underwrites tier 2 of styxx.

**Filed:** 2026-03-29

### US Provisional 64/021,113

**Title:** Alignment auditing and expression-computation dissociation
in large language models.

**Scope:** The expression/computation dissociation metric, commitment
intensity (S_early, the IPR of coherence events), and the
pre-registered cross-dataset meta-analytic validation methodology.
Underwrites tier 2/3 of styxx.

**Filed:** 2026-03-30

### US Provisional 64/026,964

**Title:** Three-axis spectrometry and cognitive governor for
transformer internals.

**Scope:** The multi-axis cognitive geometry readout (K, C, D
orthogonality), the cognitive autopsy / layer-level fault
localization, and the cognitive governor runtime (the five-phase
gate structure) that underwrites the styxx runtime itself.

**Filed:** 2026-04-02

## What the MIT code license grants

The MIT license on styxx code grants you the right to use, copy,
modify, merge, publish, distribute, sublicense, and sell copies of
the styxx source code. It does NOT grant any patent license under
the three filings above.

## What the CC-BY-4.0 data license grants

The Fathom Cognitive Atlas v0.3 centroid artifact shipped with
styxx (`styxx/centroids/atlas_v0.3.json`) is licensed under
CC-BY-4.0. You may use, redistribute, and build on the data,
provided you attribute the source (Fathom Cognitive Atlas v0.3,
Zenodo concept DOI `10.5281/zenodo.19502715`).

## Commercial use

Individual research, personal experimentation, and open-source
projects are fully supported under the MIT + CC-BY-4.0 combination.

For commercial use that practices any of the methods covered by
the provisional filings above at meaningful scale, contact:
**heyzoos123@gmail.com** to discuss a license.

We expect to convert the three provisional filings into full utility
patents (or PCT applications) **before April 2, 2027** — the latest
12-month deadline from the three filings. A commercial licensing
program will be announced alongside the first utility filing.

**Methodology subsequently formalized in:**
- *Cognometric Fingerprint Specification v1.0* — Fathom v20 — doi:[10.5281/zenodo.19746215](https://doi.org/10.5281/zenodo.19746215). The specification incorporates and references the methods covered by all three provisionals.
- *Cognometric Fingerprint Specification v1.0 — Robustness Supplement* — Fathom v22 — doi:[10.5281/zenodo.19761194](https://doi.org/10.5281/zenodo.19761194). Empirical adversarial-robustness audit of the Tier-3 reference implementation.
- *styxx v6.2.0 reference Python implementation* — software DOI [10.5281/zenodo.19758619](https://doi.org/10.5281/zenodo.19758619).

The published specification and the patent claims are deliberately
distinct: the spec defines the *vocabulary* and *measurement procedure*
under CC-BY-4.0; the patents protect the *measurement architecture*
(probe-direction calibration, attribution-weighted layer averaging,
expression-computation dissociation measurement, multi-axis cognitive
governor runtime). Implementing the spec for non-commercial purposes
does not require a patent license.

## Research citations

If you use styxx in academic research, cite both:

```
@article{rodabaugh2026fathom,
  title   = {Fathom: Cognitive Measurement Instruments for
             Transformer Internals via SAE Feature Coherence Geometry},
  author  = {Rodabaugh, Alexander},
  year    = {2026},
  note    = {Zenodo concept DOI. doi:10.5281/zenodo.19326174}
}

@misc{rodabaugh2026styxx,
  title  = {styxx: A Drop-in Cognitive Vitals Monitor for LLM Agents},
  author = {Rodabaugh, Alexander},
  year   = {2026},
  note   = {Fathom Lab. A product of Fathom Intelligence.
            https://github.com/fathom-lab/styxx}
}
```

Research repository: <https://github.com/fathom-lab/fathom>
Product repository:  <https://github.com/fathom-lab/styxx>
Fathom Lab:          <https://twitter.com/fathom_lab>
