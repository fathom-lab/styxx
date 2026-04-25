# NIST AI Risk Management Framework — Public Comment Submission

**Status:** ready to send. The NIST GAI Public Working Group accepts public input on an ongoing basis. Submit via the form at [airc.nist.gov/contact](https://airc.nist.gov/contact) or by emailing the working group secretariat at airisk@nist.gov.

**Background:** The NIST AI Risk Management Framework (AI RMF 1.0, NIST AI 100-1) and the Generative AI Profile (NIST AI 600-1) reference "measurable" risk properties without specifying *how* such properties are to be measured. This submission proposes the Cognometric Fingerprint Specification v1.0 as a candidate measurement standard.

---

## Submission body — copy from here

**To:** NIST Generative AI Public Working Group
**From:** Fathom Lab (Flobi, founder; heyzoos123@gmail.com)
**Subject:** Public Comment — Proposing the Cognometric Fingerprint Specification v1.0 as a Candidate Measurement Standard for Generative AI Risk Management
**Date:** 2026-04-25

---

### Summary

The NIST AI Risk Management Framework (AI RMF 1.0) and the Generative AI Profile (NIST AI 600-1) reference *measurable* properties of generative AI systems — accuracy, reliability, robustness, safety — without specifying how those properties are to be measured in a manner reproducible across model families and over time. We respectfully submit the **Cognometric Fingerprint Specification v1.0** ([doi:10.5281/zenodo.19746215](https://doi.org/10.5281/zenodo.19746215), CC-BY-4.0) as a candidate reference for this measurement layer, and request that the working group consider citing or referencing it in future revisions of the Generative AI Profile.

### Key contributions of the specification

1. **Three orthogonal measurement axes.** The specification defines K (reasoning depth), C (coherence/commitment), and D (dissociation/drift) with formal definitions, measurement procedures, and substrate-relative calibration. Empirical pairwise orthogonality at 86.7°–91.9° on the canonical calibration substrate (Llama-3.2-1B, layer 10) is reported.

2. **A canonical fault taxonomy.** Seven categorical failure modes — drift, confabulation, refusal, sycophancy, phase transition, low trust, incoherence — with threshold-based formal definitions, supporting reproducible cross-tool comparisons.

3. **A serialization standard.** A JSON schema for the "cognometric fingerprint": substrate identification, benchmark identification, calibration version, axis aggregates, fault rates, gate distribution, SHA-256 attestation, and provenance.

4. **A substrate-compatibility tier system.** Three tiers (open-weight + probe access; logprob-exposing API; closed-API proxy-signal pipeline) with documented confidence penalties for each tier — making compliance explicit when measurements are cross-tier.

5. **MUST/SHOULD/MAY conformance levels.** RFC-2119-style language for implementations claiming conformance.

### Why this is timely for AI RMF integration

- **The Generative AI Profile (NIST AI 600-1) was published 2024-07-26** and identifies twelve risks (CBRN, confabulation, dangerous/violent/hateful content, data privacy, environmental impacts, harmful bias and homogenization, human–AI configuration, information integrity, information security, intellectual property, obscenity/CSAM/non-consensual material, value chain). Five of these risks (confabulation, harmful bias and homogenization, information integrity, value chain, human–AI configuration) have *direct* operational measurements available in the Cognometric Fingerprint framework.
- **The reference implementation is shipping today** as styxx v6.2.0 ([pypi.org/project/styxx/6.2.0/](https://pypi.org/project/styxx/6.2.0/), MIT-licensed; software DOI [10.5281/zenodo.19758619](https://doi.org/10.5281/zenodo.19758619)).
- **Empirical validation is published** with cross-validated AUC 0.998 on HaluEval-QA hallucination, 0.976 on XSTest GPT-4 refusal (out-of-family), 0.943 on BFCL v3 tool-call drift across eight public benchmarks with explicitly published failure modes (DROP and FinanceBench reported below threshold).
- **A daily public observatory is live** at [fathom.darkflobi.com/scoreboard](https://fathom.darkflobi.com/scoreboard), profiling frontier models against a curated benchmark and committing time-series fingerprints to a public repository — providing the working group an empirical reference for continuous compliance monitoring.

### Specific requests

We respectfully request the working group consider:

1. **Reviewing the specification** during the next revision cycle of NIST AI 600-1 to evaluate whether the K/C/D axis terminology and fault taxonomy are appropriate for inclusion as referenced measurement primitives.

2. **Inviting Fathom Lab to participate** in the working group's measurement-methodology discussions, either as a contributing party or as a non-voting observer providing technical input on the specification's empirical scope.

3. **Considering the cognometric fingerprint JSON schema** as a candidate format for AI RMF Generate function deliverables (the "M" of GOVERN, MAP, MEASURE, MANAGE), specifically when the deliverable includes per-system measurement attestation.

### What we are not requesting

We are **not** seeking endorsement of any commercial product, exclusive recognition of our methodology, or restriction of competing approaches. The specification is licensed CC-BY-4.0 and any conforming implementation is welcome. Three USPTO provisional patents (64/020,489 · 64/021,113 · 64/026,964) cover the underlying measurement architecture; commercial-scale practitioners can license per [github.com/fathom-lab/styxx/blob/main/PATENTS.md](https://github.com/fathom-lab/styxx/blob/main/PATENTS.md) but research and conformance use are not restricted.

### Supporting materials

- **Specification:** [doi:10.5281/zenodo.19746215](https://doi.org/10.5281/zenodo.19746215) (also at [fathom.darkflobi.com/spec](https://fathom.darkflobi.com/spec))
- **Software DOI:** [doi:10.5281/zenodo.19758619](https://doi.org/10.5281/zenodo.19758619)
- **Concept DOI (always-latest):** [doi:10.5281/zenodo.19326174](https://doi.org/10.5281/zenodo.19326174)
- **Reference implementation:** [pypi.org/project/styxx/6.2.0/](https://pypi.org/project/styxx/6.2.0/)
- **Browser extension** (consumer-facing transparency) ([fathom.darkflobi.com/scope](https://fathom.darkflobi.com/scope))
- **Foundations monograph (research programme outline):** [foundations-of-cognometric-engineering-v0.1.md](https://fathom.darkflobi.com/foundations-of-cognometric-engineering-v0.1.md)

### Contact

Flobi
Founder, Fathom Lab
heyzoos123@gmail.com
[fathom.darkflobi.com](https://fathom.darkflobi.com)

---

We are available for any clarifying conversation the working group requires, including technical demonstration of the reference implementation against any candidate test corpus the working group provides.

Respectfully submitted,
Fathom Lab
2026-04-25

---

## Distribution notes

After NIST submission, the same letter (lightly adapted) should also be sent to:

- **UK AI Safety Institute** — info@aisi.gov.uk · [aisi.gov.uk](https://www.aisi.gov.uk/)
- **US AI Safety Institute** — at NIST · [nist.gov/aisi](https://www.nist.gov/aisi)
- **EU AI Office** (post-AI-Act) — via the European Commission feedback mechanism for AI Act implementing acts
- **OECD AI Policy Observatory** — ai@oecd.org

For each, the request shifts:
- NIST: inclusion in AI RMF Generative AI Profile measurement language
- UK AISI: candidate measurement standard for evaluation protocols
- US AISI: candidate measurement standard for federal AI assurance
- EU AI Office: candidate measurement methodology for Article 15 (accuracy, robustness, cybersecurity) implementing acts
- OECD: inclusion in cross-jurisdictional measurement comparability work
