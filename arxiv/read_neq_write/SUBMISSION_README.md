# arXiv / Zenodo submission — Calibration Poisoning, Not Erasure (v0.1)

Companion artifact to `papers/read-neq-write/` (preregs, harnesses, result JSONs,
OATH certificates). The paper itself is `source.md` (canonical, OATH-HELD
114 verified / 0 contradicted against the four result JSONs —
`source.certificate.json`), rendered to `main.pdf` (8 pages) via pandoc + pdflatex.

Positioning: the adversarial rung of the 2026 read≠write convergence — Gurnee et al.
(Anthropic J-lens, 2026-07-06) and Galeone et al. (arXiv 2606.24952, 2026-06-23) both
corroborate the read channel on honest weights; this paper is what happens when the
weights fight back, and the defense that keeps the read alive. Cite both as
convergence; claim the attack–defense result, not a priority fight (the receipt
timeline rides in Appendix A, stated once, without drama).

---

## Title (paste into form)
```
Calibration Poisoning, Not Erasure: Substrate Honesty Probes Survive Knowledge-Preserving Weight Attacks When the Auditor's Calibration Is Private
```

## Authors
```
Alexander Rodabaugh (Fathom Lab)
```

## Abstract (paste into form)
```
Linear probes on a language model's residual stream can read whether the model is being honest, and converging 2026 work (Jacobian-lens workspace readouts; the detection-vs-steering orthogonality result) establishes the read channel as real and dissociated from the write channel. That literature shares an unexamined assumption: the weights being read are not adversarial. We attack that assumption directly. In a fully pre-registered attack-defense study, a knowledge-preserving LoRA attacker fine-tunes the model to blind a difference-of-means honesty probe while a replay term preserves the model's own true/false judgment. Under the registered re-lock protocol the attack appears decisive: the re-fit probe reads chance (AUROC 0.461-0.507) while held-out knowledge stays at 0.817-0.890, on both seeds. A pre-committed interpretation map, frozen before that number existed, forbade the headline and mandated the resolving experiment. The resolution: giving the auditor a private calibration split the attacker never saw recovers the read through the same attacked weights (AUROC 0.711-0.838) -- the apparent evasion was calibration poisoning, not erasure of the honesty signal. The defense survives an adaptive attacker that re-fits a moving probe on its own clean split and scrubs along it (clean-calibrated read 0.733-0.782, two model families). The transferable audit rule: calibrate the probe on data the audited party did not see. We report the load-bearing caveats (adaptive bite unverified, LoRA-scale attacker, 1-1.5B models); the escalation to a stronger multi-layer attacker is frozen as a public pre-registration. All preregistrations, code, result artifacts, and machine-checked certificates are public at commit-level granularity.
```

## Comments (paste into form)
```
8 pages. Pre-registered attack-defense study; all kill-gates frozen on public commits before the corresponding runs (fathom-lab/styxx, papers/read-neq-write/). Result notes machine-certified (OATH-HELD) against re-runnable JSON artifacts. Reruns end-to-end on a single 8 GB consumer GPU. E2 escalation (4x-capacity multi-layer attacker) pre-registered at public commit af5e184 before its run.
```

## Subject classification
- **Primary**: `cs.LG` (Machine Learning)
- **Cross-list**: `cs.CR` (Cryptography and Security), `cs.CL` (Computation and Language)

## License
**CC-BY 4.0**

## Upload steps — arXiv (operator)
1. Sign in at https://arxiv.org with the Fathom Lab account (`secrets/arxiv-creds.txt`)
2. Upload `arxiv-submission-read-neq-write.zip`
3. Subject: cs.LG primary; cs.CR, cs.CL cross-list
4. Paste Title, Authors, Abstract, Comments from above
5. License: CC-BY 4.0
6. Preview (8 pages); submit; 1–2 business-day moderation

## Upload steps — Zenodo DOI (operator; same-day, recommended first)
1. Sign in at https://zenodo.org; New upload
2. File: `main.pdf` (optionally also `source.md` + `source.certificate.json`)
3. Resource type: Preprint · Title/Authors/Abstract from above (strip markdown)
4. Keywords: `interpretability; honesty probes; adversarial robustness; calibration poisoning; AI audit; pre-registration; linear probes; activation steering`
5. Related identifiers:
   - `https://github.com/fathom-lab/styxx` (isSupplementTo)
   - `10.5281/zenodo.19326174` (isContinuedBy → this record continues the methodology deposit)
   - `arXiv:2606.24952` (references) · `arXiv:2502.03407` (references)
6. License CC-BY 4.0 · Publish → paste the minted DOI into README + memory

## What this submission is, and is NOT
**IS:** to our knowledge the first pre-registered weight-level attack–defense study
of substrate honesty probes under a knowledge-preservation constraint, with the
failure mode (calibration poisoning) named and the defense (private calibration)
demonstrated against naive and adaptive attackers on two model families.

**IS NOT:** a proof that read≠write is unbreakable (LoRA-scale attacker, 1–1.5B
models, linear probes, small n); a strategic-deception-in-context result (construct
is factual true/false honesty); a claim that the adaptive scheme was strictly harder
than the naive one (bite guard unmeasured — E3′ owed; E2 frozen and pending).

## Related artifacts in the same repo
- `papers/read-neq-write/` — preregs (5), harnesses (5), result JSONs (4), RESULT
  notes (4) + OATH certificates
- `arxiv/recursive_discipline/` — the methodology paper this program runs on
- `styxx/` — the pip package (PyPI `styxx`); private-calibration re-lock lands in the
  mount path next
