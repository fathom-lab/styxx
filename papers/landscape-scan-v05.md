# Cognometry v0.5 Academic Landscape Scan

**Date:** 2026-04-23. Background research for the arXiv v0.5 submission.

## 1. Required citations (prior-art risk)

### Highest-risk citation we MUST engage
- **Wang et al. 2025 "False Sense of Security: Why Probing-based Malicious Input Detection Fails to Generalize"** — arXiv:2509.03888. Trains SVM/LR/MLP probes on last-layer hidden states, reports 40–80% FP rates on XSTest safe subsets, 15–99pp OOD drops. Directly critiques the probe-based refusal framing. **Cognometry v0.5 §4.4 addresses this head-on** with the held-out cross-substrate design as counter-evidence.

### Semantic entropy / uncertainty-probe ancestry
- Farquhar et al. 2024 (Nature). https://www.nature.com/articles/s41586-024-07421-0
- Kossen et al. 2024 "Semantic Entropy Probes." arXiv:2406.15927

### Residual-stream hallucination probes (white-box comparables)
- O'Neill et al. 2025 "A Single Direction of Truth (ContraTales)." arXiv:2507.23221 — linear residual probe for contextual hallucination on Gemma-2
- Liu et al. 2025 "ICR Probe: Tracking Hidden State Dynamics." ACL 2025. https://aclanthology.org/2025.acl-long.880.pdf
- Wang et al. 2025 "HalluSAE." arXiv:2604.16430 — SAE features, SOTA on HaluEval

### Efficient text-based detectors (closest size-niche comparables)
- Arteaga et al. PMLR 2025 "Hallucination Detection: Fast and Memory-Efficient Finetuned Models." https://proceedings.mlr.press/v265/arteaga25a.html
- Huang et al. 2024 "Simple Factuality Probes Detect Hallucinations in Long-Form." ACL Findings 2025. https://aclanthology.org/2025.findings-emnlp.880.pdf — AUC 0.87 on Llama-3.3-70B

### Hallucination taxonomies
- Hong, Mihindukulasooriya et al. 2025 "HalluLens." ACL 2025. https://aclanthology.org/2025.acl-long.1176.pdf — unified extrinsic/intrinsic split

### Safety-classifier lineage (Instrument 2 comparables)
- Llama Guard (Inan et al.)
- ShieldGemma tech report
- NVIDIA Aegis 2024
- WildGuard (Han et al. NeurIPS 2024)
- **IBM Granite Guardian** (Padhi et al. Dec 2024, arXiv:2412.07724) — Table 7 is the XSTest-RH AUC reference we use

### Refusal separability (directly relevant to our two-instrument framing)
- Zhao et al. 2025 "LLMs Encode Harmfulness and Refusal Separately." arXiv:2507.11878

### Agent failure modes (Instrument 3 roadmap)
- Datta et al. 2025 "Agent GPA (Goal-Plan-Action)." arXiv:2510.08847
- Cemri, Pan et al. 2025 "Why Do Multi-Agent LLM Systems Fail?" arXiv:2503.13657

## 2. Terminology check

No arXiv hits for "cognometry" or "cognitive vitals" in LLM context. Name is clear.

Adjacent framings to mention briefly to differentiate:
- "Cognitive evaluation" (Ivanova et al. arXiv:2504.02789)
- "Cognitive load" (ICE benchmark arXiv:2509.19517)

## 3. Collaborator targets (excluding already-contacted: Minervini, Arditi)

**Paul Röttger** — Oxford / MilaNLP. XSTest author. PRISM NeurIPS 2024 Best Paper. UK AISI grantee. https://paulrottger.com/ — DIRECT fit; our XSTest-v2 work is his exaggerated-safety framing.

**Jannik Kossen / Sebastian Farquhar** — OATML Oxford. SEP authors. Cognometry multi-instrument framing extends their uncertainty-probe work. Reach via OATML page.

**Muhao Chen** — UC Davis. Senior author on "False Sense of Security." https://muhaochen.github.io/ — either critic or strongest collaborator. High-leverage outreach.

**Patrick Chao / Alex Robey / Maksym Andriushchenko** — JailbreakBench authors. NeurIPS 2024 D&B. Our refusal calibration is a natural companion to JBB.

**Anupam Datta** — Snowflake, ex-CMU. Agent GPA (arXiv:2510.08847). Cognometry methodology fits GPA failure detection.

**Mert Cemri / Melissa Pan** — UC Berkeley. Multi-Agent taxonomy (arXiv:2503.13657). Cognometry can instrument their 14 failure modes.

## 4. Venue targets + deadlines

**PRIMARY:** **NeurIPS 2026 main track** — abstracts **May 4, 2026**, full papers **May 6, 2026**. Datasets & Benchmarks Track fits styxx. https://neurips.cc/Conferences/2026/Dates

**SECONDARY:** NeurIPS 2026 SafeGenAI / SoLaR workshops (CFPs Aug 2026).

**FAST-FEEDBACK:** **ACL 2026 TrustNLP Workshop** (6th edition). Practitioner+research audience. https://www.aclweb.org/portal/content/6th-trustworthy-nlp-workshop-acl-2026

**ICLR 2027:** abstracts ~September 2026.

Skip AAAI-26 (deadlines passed Nov 2025); target AAAI-27.

## 5. Honest prior-art assessment

**No paper combines all three of:**
- (a) unified multi-instrument cognitive-state detection
- (b) cross-validated on 8 hallucination benchmarks
- (c) sub-100M XSTest AUC

Closest and most dangerous:
- "False Sense of Security" — cite + rebut with held-out GPT-4 XSTest design (§4.4)
- HalluSAE + ICR + Huang-2024 — multi-benchmark linear-probe comparators but none span hallucination+refusal jointly
- Kossen SEP + Azaria-Mitchell — ancestor citations for intellectual honesty

## 6. Three open problems cognometry can credibly address

1. **Agent goal-drift instrumentation.** Datta's GPA framework and Cemri's 14-mode taxonomy lack a calibrated real-time detector; cognometry's probe family is the missing instrument layer.

2. **Exaggerated-safety calibration.** Röttger's XSTest identifies the problem; Wang's "False Sense of Security" shows existing probes fail. A calibrated refusal scorer with known OOD behavior fills a live gap.

3. **Cross-model cognitive-state transfer.** O'Neill's "single direction" and Zhao's separate-encoding results raise whether cognometric instruments transfer across model families. Our 8-benchmark hallucination + 5-family refusal design is uniquely positioned to test this at scale.

## 7. Immediate action items

**From this scan, do these next:**

1. ✅ DONE: Wang 2025 citation + rebuttal added to v0.5 §4.4
2. ✅ DONE: Related-work section (v0.5 §5.4) added with Farquhar, Kossen, O'Neill, Liu, Arteaga, Huang, HalluLens, Granite Guardian, Datta, Cemri
3. Email Röttger (XSTest author) once arXiv preprint is live — ask for endorsement / extension discussion
4. Email Kossen (OATML) once arXiv preprint is live — position as downstream of SEP paradigm
5. Email Chao / Robey (JBB authors) — offer calibration collaboration
6. Target NeurIPS 2026 D&B track (May 4/6 deadline) for full paper submission
7. Target ACL TrustNLP 2026 workshop for faster feedback cycle
