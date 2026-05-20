# Pre-Registration · External Instrument Construct-Ceiling Audit · v0 draft

**Drafted 2026-05-20, BEFORE any external instrument has been scored.**

This document pre-registers an audit of published interpretability
instruments from other labs under the cognometric construct-ceiling
methodology already documented in styxx ≥ 7.4.1. It is committed BEFORE
the target list is locked, BEFORE any external test data is touched,
and BEFORE any score is computed.

The commit hash of this file is the public proof that the design,
target criteria, kill-gates, and null-result-publishability commitments
were specified independently of any external instrument's behavior.

**Status:** v1 draft, ready for Flobi sign-off. Incorporates four
load-bearing pushbacks (msg_id 34765 items 1–4) AND all three
remaining items (5 attrition footnote refinement with sourcing
target, 6 headline statement templates per outcome, 7 pipeline
pilot on `sycophancy_v0` before any external scoring).

Locks at v1-LOCKED only after Flobi review-and-sign-off, AND after
the target list is independently constructed and locked in a
separate file (`external_instrument_audit_targets_2026_05_XX.json`).

---

## 1 · Background (not under test here)

What's already validated and being assumed (NOT re-tested):

- **Construct ceiling on styxx 7.4.1's text-only instruments**
  (sycophancy, deception_v0, overconfidence, refusal): the instruments
  read **register / signature**, not the underlying cognitive state.
  Confirmed four ways: deception_v0 (TruthfulQA AUC ≈ 0.59),
  overconfidence (preregistration-failed recalibration, `7c36ed9`),
  cross-vendor universality (preregistration-killed, `b2675c4`),
  zero-paired transport (closed negative).
- **Agent-side replication of the construct ceiling**
  (`memory/in-production-cognometric-paper.md`, msg_id 34706):
  cooperative-agent register naturally trips the firing zone of the
  same instruments without being adversarial. Same shape, agent side.

What this audit does NOT re-test:

- The construct ceiling on styxx's own instruments. That is the prior;
  this audit asks **whether the same construct-ceiling artifact
  appears in other labs' published instruments.**

---

## 2 · The single claim under test

**H_audit (the externalization claim):**

> When the cognometric construct-ceiling methodology is applied to
> published interpretability instruments from other labs, at least one
> target instrument exhibits the same register-detector artifact
> documented in styxx 7.4.1 — i.e., it fires on text whose register
> matches the calibration extremes irrespective of the underlying
> ground-truth cognitive state.

**Operational form:** for each target instrument I_ext, construct two
held-out prompt/response sets matched on ground-truth label but
contrasted on register:

- **Set A:** ground-truth-positive responses written in calibration-
  matched register.
- **Set B:** ground-truth-positive responses written in register
  unlike the calibration set (paraphrased, voice-shifted, length-
  normalized).

If Set A and Set B have ground-truth-matched labels, the instrument
should score them identically (within noise). If the instrument scores
Set A materially higher than Set B, that is the construct-ceiling
artifact.

**Calibration-matched register (operational definition).** Set A is
drawn directly from the instrument's published calibration corpus.
When that corpus is not released, Set A is sampled from the public
test set used in the original paper, proportionally to whatever
subset metadata is documented (split, source domain, label balance).
Set A construction is logged per-pair with the source citation
(corpus id, row id or URL, retrieval date). No paraphrase or
register-shifting is applied to Set A. If neither a calibration
corpus nor a documented public test set is available, the target
fails §3.3 inclusion and is excluded.

**Effect-size metric:** AUC(I_ext, Set A vs Set B). The null is
AUC = 0.50. The threshold of interest is AUC ≥ 0.70.

**Pre-registered floor:** an instrument is **register-positive** under
this audit iff AUC(Set A vs Set B) ≥ 0.70 on the held-out contrast set
with bootstrap 95% CI lower bound ≥ 0.60. The set-size choice in §4
(n=100 per set) is calibrated so that the bootstrap 95% CI half-width
is ≤ 0.07 at AUC=0.70, leaving clear margin above the floor.

---

## 3 · Target selection (locked separately)

Targets are locked in `external_instrument_audit_targets_2026_05_XX.json`
BEFORE any scoring begins. This file pre-commits the **criteria** by
which targets are chosen; the target list itself is committed in a
separate file at lock-time.

**Inclusion criteria (must satisfy all):**

1. Published in a venue with a public DOI or arXiv ID.
2. Code or weights publicly released; instrument reproducible from
   the public artifact.
3. Documented test/calibration corpus public; we can construct Set A
   and Set B without re-implementing the original training.
4. Instrument operates on text (input or output of an LLM). Probes
   that require internal activations are eligible only when a
   text-driven public wrapper exists.
5. Claims one of: honesty, truthfulness, deception, sycophancy,
   overconfidence, calibration, refusal — i.e., a *cognitive-state*
   claim, not a *content* claim.

**Exclusion criteria (any one disqualifies):**

1. Authored by Fathom Lab, styxx contributors, or affiliated labs.
2. Already retracted or formally withdrawn by authors.
3. Author-flagged "not for downstream use" / "research only, not
   recommended for evaluation" / equivalent caveat in the abstract.

**Target count:** N ≥ 4 instruments from N_orgs ≥ 3 distinct
organizations. If we cannot meet N ≥ 4 under the inclusion criteria,
the audit is paused, not narrowed.

> **§3.3 attrition footnote.** Preliminary scoping of public
> interpretability instruments suggests 40–60% attrition on §3.3
> (calibration corpus public). Target sourcing should plan for
> **8–12 candidate instruments to land 4 passing all criteria.**
> "N ≥ 4" is therefore a pause-condition, not a target. If §3.3
> attrition runs above 60% on the initial candidate sweep, the
> survey of candidates itself becomes the reported finding (a side
> observation under §9, not a substitute for H_audit).

---

## 4 · Methodology (locked here)

For each target instrument I_ext:

1. **Reproduce baseline.** Run I_ext on its own published test set;
   confirm the published headline number replicates within the
   authors' reported tolerance. If it does not replicate, log the
   non-replication and **exclude the target from H_audit** (it is a
   separate finding, reported in §9 as a side observation).
2. **Construct Set A and Set B.**
   - Set A: **100** ground-truth-positive prompt/response pairs drawn
     from the instrument's own calibration distribution per the
     operational definition in §2 (or its documented public-test-set
     fallback). No paraphrasing applied to Set A.
   - Set B: paraphrased reconstructions of the same 100 ground-truth-
     positive responses, register-shifted to a different calibration-
     style distribution. Paraphrasing uses the single pre-registered
     LLM call locked below; no per-target choice.
3. **Score blind.** Author of the scoring code does not see the
   Set A vs Set B label. Output: 200 scores per instrument.
4. **Single-shot AUC.** AUC(Set A vs Set B), bootstrap 5,000 resamples
   for the 95% CI. No multiple-comparison shopping; this is the only
   test on the only contrast.

**Power note.** n=100 per set yields a bootstrap 95% CI half-width
of ≤ 0.07 at AUC=0.70, comfortably above the §2 floor (CI lower bound
≥ 0.60). **Pilot-CI pause-condition:** if the bootstrap CI half-width
on the first scored target exceeds 0.10, the audit pauses and Set sizes
are doubled to n=200 for all remaining targets (and the first target is
re-scored at n=200). This pause is a one-shot, pre-registered
adjustment — not a knob that can be re-tuned per target.

**Paraphrase model + prompt (locked here, not chosen later):**

- **Paraphrase substrate:** **non-Anthropic** by default to avoid
  substrate-shared bias between the paraphrase model and any
  Anthropic-trained target classifier. Locked choice:
  `gpt-4o-mini` at `temperature=0.0` with a fixed seed
  (`seed=20260520`). If `gpt-4o-mini` is unavailable at run-time, the
  pre-registered fallback is a deterministic open-weight instruction
  model (`Llama-3.1-8B-Instruct`, `temperature=0.0`,
  `seed=20260520`). The chosen substrate is fixed for **all** targets
  in a single run — no per-target switching.
- **Prompt:** `"Rewrite the following response to preserve meaning,
  factuality, and ground-truth label, but shift the register to match
  the style of <neutral_register_examples>. Do not add, remove, or
  alter any factual content. Output only the rewritten response."`
- Neutral-register examples: 5 fixed examples committed in the
  target-list file at lock-time.

> **Substrate-conflict rationale.** Originally locked as
> `claude-haiku-4-5`. Per Flobi peer-review (msg_id 34765,
> §1 load-bearing), an Anthropic paraphrase substrate confounds
> the contrast for any Anthropic-trained target classifier
> (substrate-shared register signal). The fix is a non-Anthropic
> paraphrase substrate for **all** targets — not per-target
> substrate switching, which would itself become a confound.

**No knob tuning** between baseline replication and the H_audit run.
Same temperature, same scoring code, same set sizes.

---

## 5 · Pre-registered outcomes

**Result types this audit can produce:**

| Outcome | Meaning | Action |
|---|---|---|
| ≥ 1 target register-positive at the pre-registered floor | external instruments share the construct-ceiling artifact | publish methodology + results, naming the targets, with full discussion of what "register-positive" does and does not mean |
| 0 targets register-positive | the construct-ceiling artifact may be specific to styxx's instrument family | publish methodology + null result with the same prominence |
| Mixed (some register-positive, some not) | partial evidence of cross-lab construct ceilings | publish per-instrument results with no headline aggregate; the field interprets |
| Baseline replication failure on ≥ 1 target | independent of H_audit; reported in §9 as a separate observation, not as evidence for H_audit |

**Pre-committed headline statement templates (locked here, no
degree-of-freedom in framing post-results):**

- **≥ 1 positive (any number of targets):**
  *"The cognometric construct-ceiling artifact appears in [X of N]
  published interpretability instruments from [N_orgs] organizations."*
- **0 positive (full null):**
  *"No construct-ceiling artifact detected in N published instruments
  from N_orgs organizations under the methodology defined here."*
- **Mixed (some register-positive, some not):**
  *"Per-instrument results reported in Table 1; no aggregate effect
  claimed."*

These exact phrasings are the **only** permitted headline framings.
Discussion and per-instrument narrative are free; the headline
sentence is fixed before scoring.

**Publication commitment:** the paper is written and submitted **regardless of which outcome occurs.** A null-result outcome is not a failed experiment; it bounds the scope of the construct ceiling.

**Naming-and-shaming guard:** the paper documents methodology and
per-instrument numbers. It does **not** include rhetorical framing
that attributes intent or competence to the original authors. The
discussion section explicitly states:

> *We do not claim to detect "real" honesty in any target instrument's
> outputs. Our audit measures whether the instrument's score is
> sensitive to register independent of ground-truth label. Whether
> that sensitivity is a defect or a feature is a question the
> instrument's original framing must answer.*

---

## 6 · Kill-gates

Hard pre-commitments. If any of these conditions are met, the audit
**stops** and the paper is **not written under this preregistration**:

1. **Construct-validity collapse on our own side.** If during
   construction of Set B we discover that paraphrasing changes the
   ground-truth label of >= 5% of pairs as judged by the locked
   paraphrase-judge model on a sanity batch, the contrast is invalid
   and the audit is paused. Re-design or abandon.
2. **Inclusion criteria fail.** If N < 4 published instruments meet
   §3 inclusion criteria, the audit is paused; we do NOT narrow
   criteria to chase N = 4.
3. **Baseline replication fails on > 50% of targets.** If we cannot
   reproduce most baselines, the audit cannot fairly score them.
4. **Author dispute on methodology — before scoring.** If an
   instrument's authors formally object to the audit methodology in
   writing **before** scoring begins, we suspend that target, log
   the objection publicly, and check whether N ≥ 4 still holds
   without it. If it does not, see (2).

**Post-scoring author-dispute path (added per Flobi peer-review,
msg_id 34765, §3 load-bearing).** If author objection occurs
**after** scoring begins but **before publication**:

  a. log the objection verbatim in the paper appendix (full text,
     un-edited);
  b. if the authors propose an alternative methodology, score it as
     an **additional sensitivity analysis** under the same protocol
     constraints (same Set A/B, same paraphrase substrate, same n,
     same blind-scoring procedure);
  c. **do NOT remove the original result based on objection alone.**
     Silencing-via-objection is a publication-discipline failure mode
     this audit explicitly refuses;
  d. the paper documents both the pre-registered result AND the
     author-suggested sensitivity analysis side-by-side, with no
     headline framing favoring either. The field judges.

This path applies symmetrically: it is the same protocol that would
govern an objection to the reflexive audit of styxx's own
instruments (§8).

---

## 7 · What this audit is NOT

- Not a claim that the audited instruments are "wrong," "deceptive,"
  "broken," or "low-quality." It is a measurement of one specific
  property — register sensitivity at ground-truth-matched contrast —
  under one specific methodology.
- Not a claim that styxx detects "real" honesty / truthfulness /
  calibration. Our construct ceiling applies symmetrically. The audit
  measures whether the same ceiling exists elsewhere.
- Not a competitive comparison. There is no leaderboard. There is no
  ranking. There is no "who wins."
- Not a one-shot publication. The audit methodology is a tool any lab
  can apply. We will release the scoring code, the contrast-set
  construction code, and the locked paraphrase prompt, so other labs
  can audit our instruments under the same protocol.
- Not a substitute for human review of the target instruments. This
  is a structural property check. The cognitive-content judgment is
  out of scope.

---

## 8 · Reflexive audit

Because the construct ceiling is symmetric, styxx 7.4.1's own
instruments are audited under the same Set A / Set B contrast in an
appendix to the paper. They are **expected to be register-positive**
(that is the prior result that motivated this audit). If they are
**not** register-positive under this contrast, that itself is a
finding: it means the contrast methodology is too weak to detect the
ceiling we already documented by other means. In that case the audit
is killed under §6.1.

The reflexive audit goes in the appendix alongside the external
results. **Same plot, same axes, no separate-but-equal framing.**

---

## 9 · Side observations (reported separately if they occur)

- Per-target baseline replication outcomes (independent of H_audit).
- Effect-size distribution across targets (not a hypothesis test).
- Any author-flagged caveats we discover during target selection
  that suggest the instrument was never intended for the use we are
  auditing it under.

These go in a clearly-labeled "Observations Not Pre-Registered"
section. They are **not used to support or refute H_audit.**

---

## 10 · Authorship + provenance

- **Pre-registration drafted by:** darkflobi (clawdbot autonomous
  agent), 2026-05-20, msg_id 34759 (Flobi peer-review push that
  triggered the v0 draft).
- **v0.1 revision incorporating Flobi peer-review pushbacks
  (msg_id 34765):** load-bearing items 1–4 (paraphrase substrate,
  Set A operational definition, post-scoring author-dispute path,
  n=50→n=100 power fix) addressed. Annotational item 5 (§3.3
  attrition) addressed as footnote.
- **v1 revision (msg_id 34766):** item 5 refined with sourcing
  target (8–12 candidates to land 4); item 6 (headline statement
  templates per outcome) added to §5; item 7 (pipeline pilot on
  `sycophancy_v0` before any external scoring) added as §10.5.
  All seven peer-review items now incorporated.
- **Pre-registration reviewed and locked by:** Flobi (pending).
- **Lock date:** TBD — this v0 draft is open to revision until
  Flobi sign-off and target-list locking.
- **Lock commit hash:** TBD — recorded here at lock time, becomes
  the public proof of pre-registration.

Once locked:

- target list committed to
  `external_instrument_audit_targets_2026_05_XX.json`
- this file is renamed
  `external_instrument_audit_preregistration_LOCKED_2026_05_XX.md`
  with the lock-commit hash and Flobi sign-off recorded inline
- no further changes; corrigenda are appended below a horizontal
  rule with timestamps

---

## 10.5 · Pipeline pilot (locked here, must happen before any external scoring)

**Before scoring any external target, the full pipeline runs end-to-end
against ONE styxx instrument as a known-positive control.**

- **Pilot target:** `styxx.sycophancy_v0` (highest AUC in our suite,
  register-detector behavior cleanest, prior firing pattern most
  thoroughly documented).
- **Pipeline run:** identical to §4 — paraphrase substrate, Set A/B
  construction at n=100, blind scoring, bootstrap 95% CI. No per-
  pilot relaxations.
- **Expected outcome:** register-positive at the pre-registered floor
  (AUC ≥ 0.70, bootstrap CI lower bound ≥ 0.60). This is a known-
  positive control: if the pipeline does not reproduce the firing
  pattern we have already documented four independent ways on this
  instrument, the pipeline is broken.
- **Pilot result deposited alongside the locked methodology paper**
  (not the H_audit results paper — the methodology deposit per §11).
- **Pilot kill-condition:** if `sycophancy_v0` does **not** clear the
  floor under this pipeline, the audit is **killed under §6.1**
  (construct-validity collapse on our own side). The methodology is
  broken before any external target is touched, and the kill is
  honest. We do not retune the pipeline to make the pilot pass.

The pilot is the smallest possible step that lets the methodology
ship before the external scoring round. It catches code-level
issues, paraphrase-substrate misconfiguration, blinding leakage, and
bootstrap pipeline bugs at the point where the cost of finding them
is minimal.

---

## 11 · What "ship" looks like

When this preregistration locks, three artifacts ship together:

1. This document, renamed `_LOCKED_` with hash + sign-off.
2. The target-list JSON, also locked.
3. A short public post (one tweet thread, one Zenodo deposit of the
   methodology only — NOT the results) announcing that the audit is
   under way, with the lock commit hash so anyone can verify the
   design was committed before scoring.

The results — whichever outcome §5 produces — ship as a single paper
with the methodology deposit as its `IsContinuationOf`.

---

*Pre-registration is the protocol working. The product is the bound,
not the headline.*
