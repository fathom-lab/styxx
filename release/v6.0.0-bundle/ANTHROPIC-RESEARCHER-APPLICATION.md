# Anthropic External Researcher Access Program — Application Packet

**Form URL:** https://docs.google.com/forms/d/e/1FAIpQLSdmq-KFTREKw9SXqDqj9kACfhd_5DYZSrifWR7Q3z0ZqAZZog/viewform

**Default award:** $1,000 API credits. Rare cases get more.

**Strategy:** request $10k (defensible given 3 active research threads),
anchor to AI safety framing, point at real shipped artifacts
(v6.0.0 PyPI release, 2 Zenodo papers, 655-test suite, 8-benchmark
cross-validation). Don't hedge.

**You'll need before submitting:**

- [ ] Your Organization ID from https://console.anthropic.com/settings/organization
      (format: `org_...` or UUID). Required field.
- [ ] 5 minutes to paste the answers below.
- [ ] To be logged in to Google as yourself.

---

## Paste-ready answers

### (1) Email
```
heyzoos123@gmail.com
```

### (2) Name of primary contact
```
Flobi
```

### (3) Name of organization
```
Fathom Lab
```

### (4) Referred by Anthropic employee?
```
No
```

### (5) Referral name
```
(leave blank)
```

### (6) Organization ID
```
(FILL IN from https://console.anthropic.com/settings/organization)
```

### (7) Individual applicant / team description (< 200 words)
```
Fathom Lab is an independent AI-safety research group building
cognometric instruments — calibrated, text-only detectors for LLM
cognitive failures in production. We maintain styxx (MIT, PyPI,
655 tests passing) at github.com/fathom-lab/styxx, and have
published two peer-reviewable papers on Zenodo: Cognometry v0
(DOI 10.5281/zenodo.19703527, April 2026) and Fathom: Cognitive
Measurement Instruments for Transformer Internals (DOI
10.5281/zenodo.19504993).

Our v6.0.0 release shipped three calibrated cognometric
instruments: hallucination (0.998 AUC on HaluEval-QA, 8-benchmark
cross-validated), refusal (0.976 AUC on XSTest-v2 GPT-4
held-out), and tool-call drift (0.916 AUC on BFCL v3 text-only,
beating the only published hidden-state baseline at 0.72). All
three run sub-millisecond on CPU, work on any closed-model API
with no internal access, and publish committed reproducers for
every number. Two documented failure modes (DROP, FinanceBench)
are listed openly alongside the successes with structural fix
paths.

Recent work: head-to-head vs Vectara HHEM-2.1-Open (+0.23 AUC,
330× faster). Merged PRs to awesome-hallucination-detection.
Active collaboration channel with Pasquale Minervini (HaluEval
author).
```

### (8) Research description (< 300 words)
```
We're requesting API credits for three active calibration and
validation threads that directly support AI safety research.

(1) Cross-model refusal calibration at scale. Our current 18-
feature refusal detector (AUC 0.976 on GPT-4 held-out, trained
on Llama-1B apologetic refusals) documents a specialist failure
on Mistral-instruct lecturing refusals (AUC 0.61) caused by
training-corpus bias. Fixing this requires harvesting thousands
of labeled refusal responses across Claude Haiku 4.5, Sonnet 4.6,
and Opus for calibrated-v3 weights. This is AI safety-direct:
better refusal detectors measurably reduce false-negative jailbreak
bypasses in deployed agents.

(2) Cross-instrument phase-transition ablation. We shipped
a phase-transition result today
(papers/drift_phase_transitions.md) showing failure detectability
emerges in discrete jumps as features scale — inverse of emergent-
capabilities literature. Replicating on hallucination and refusal
requires labeled Haiku responses to validate generalization beyond
existing GPT-4 and Llama baselines.

(3) Causal cognitive engineering. Our v3.5 work shows probe
directions are causally load-bearing (patch residual stream, behavior
flips, refuse@unsafe 97% → 17% at α=3.0 on Llama-1B). Extending to
the full cognitive-state-vector stack (refuse/deceive/sycophant/
confab/goal-drift/overconfident) requires baseline behavior on Claude
models as the cross-architecture comparison — we have Llama-1B, we
need Haiku/Sonnet scores to test whitened cross-family transfer.

All outputs are arxiv-submittable papers + open-source reproducers.
Without API credits, cross-model calibration has a hard floor at
the open models whose outputs we can sample — which structurally
biases the safety-relevant conclusions we can draw.
```

### (9) Requesting more than $1000?
```
Yes
```

### (9a) If yes, how much & justification:
```
Requesting $10,000 over 6 months.

Justification: three active research threads, each requiring ~$3k
in API for labeled response harvesting across multiple Claude
model sizes (Haiku / Sonnet / Opus). Our published Zenodo v0 +
655-test reproducer suite demonstrates we convert compute into
peer-reviewable results efficiently. At $1k, we can only cross-
calibrate one thread at one model size.
```

### (10) Significant hindrance to research without credits?
```
Yes
```

### (11) Google Scholar / GitHub profile
```
https://github.com/fathom-lab/styxx
```

### (12) Additional information (optional — use it)
```
Most directly relevant recent artifacts:

- v6.0.0 release (shipped today, 2026-04-23):
  https://pypi.org/project/styxx/6.0.0/
  https://github.com/fathom-lab/styxx/releases/tag/v6.0.0

- Cognometry v0 paper:
  https://doi.org/10.5281/zenodo.19703527

- Live in-browser playgrounds (Pyodide, no install):
  https://fathom.darkflobi.com/cognometry/try
  https://fathom.darkflobi.com/cognometry/refuse
  https://fathom.darkflobi.com/cognometry/drift

- Phase-transition ablation shipped today:
  https://github.com/fathom-lab/styxx/blob/main/papers/drift_phase_transitions.md

Happy to provide additional reproducer output, detailed methodology,
or walk through benchmark protocols on a call if useful.
```

### (13) Located within United States?
```
Yes
```

### (14) Terms of Service
```
☑ (check the box)
```

---

## Post-submission

Expected response time: 2-4 weeks per program docs.
Once approved, credits appear on the Organization ID specified.
No code changes needed — keeps working with current
ANTHROPIC_API_KEY.

If approved for less than $10k, accept and note the gap in
any follow-up research proposal.
