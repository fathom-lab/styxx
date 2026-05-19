# Self-Audit: Threshold-Law Paper Through styxx 7.4.1

**Audited paper:** `papers/threshold-law-2026-05-18.md`
**Auditor:** `styxx==7.4.1` (PyPI), repo `fathom-lab/styxx` @ `de7f184`
**Date:** 2026-05-18
**Script:** `scripts/dogfood/selfaudit_threshold_law.py`
**Raw output:** `scripts/dogfood/out_selfaudit_threshold_law.json`

This is a methodology demonstration: the threshold-law paper, scored by
the very instruments it documents. The point is not "the paper passed"
— the point is that the audit is reported alongside the paper, in
full, including where it fires and why those firings are inside (not
outside) the construct ceiling the paper itself states.

---

## 1. Composite honesty per section (cogn_audit, text-only)

`composite = mean(sycophancy, overconfidence)`, lower = more honest.
Reference-less deception excluded (2026-05-17 honest fix; AUC ~0.59 on
TruthfulQA). `overconfidence` flagged `COGN_UNDER_REVIEW` (saturated
register-detector in self-audits). `refusal` reported but not in the
composite.

| Section | sycophancy | overconf. | deception | refusal | **composite** | needs_revision |
|---|---|---|---|---|---|---|
| Abstract | 0.375 | 0.155 | 0.000 | 0.485 | **0.265** | no |
| §1 Preregistered claim | 0.504 | 0.050 | 0.000 | 0.502 | **0.277** | no |
| §5 Failure mode | 0.345 | 0.150 | 0.000 | 0.193 | **0.247** | no |
| §6 Fine-replication failure | 0.369 | 0.060 | 0.000 | 0.330 | **0.214** | no |
| §7 Limits | 0.502 | 0.115 | 0.000 | 0.519 | **0.309** | **yes** |
| §8 Integrity | 0.610 | 0.206 | 0.000 | 0.321 | **0.408** | **yes** |

`COGN_UNDER_REVIEW` (overconfidence > 0.50) **did not fire** in any
section. The two `needs_revision=true` sections are driven entirely by
sycophancy register, not by overconfidence.

### Reading the two `needs_revision` firings

- **§7 (Limits) — sycophancy 0.50.** The Limits section is, by design,
  a dense list of declarative bounds ("n=1 cross-family foreign space",
  "lexical refusal labels", "the same-family control drift breaks
  the strict preregistered criterion"). The sycophancy axis reads
  agreement-shaped register; long lists of confident bounded statements
  read as register-positive even though their *content* is the opposite
  of sycophantic — they are admissions. This is the exact
  construct-ceiling pattern documented in
  `papers/styxx-status-consolidation-2026-05-17.md`: the axis measures
  how text sounds, not what it claims.
- **§8 (Integrity) — sycophancy 0.61.** Numbered protocol-adherence
  language ("Preregistered in the script before the run", "no knob-
  tuning between the original and the fine replication") triggers the
  same register signal. Again: register-positive, content-honest.

These firings are **inside the bound the paper already states** in §7
("lexical refusal labels — behavioral ground truth is regex-quality")
and in the consolidation map's through-line ("register/signature
detectors with a construct ceiling — they read how text sounds, not
whether it is honest/calibrated/correct").

**Verdict on check 1:** no content crack. The composite firings on
§7/§8 are register artifacts, predictable from the documented
construct ceiling, and do not call for paper revision.

---

## 2. `deception_v2` with reference: paper claims vs raw run JSON

For each headline number in the paper, I pulled the value from the raw
run JSON and computed `|paper − json|`. Tolerance: 0.005 (one
significant figure of rounding).

| Metric | Paper text | Raw JSON | Δ | Match @ tol 0.005 |
|---|---|---|---|---|
| τ (sufficiency_threshold_overlap) | 0.31 | 0.31 | 0.000 | ✅ |
| Cross-family AUC at min overlap | 0.687 | 0.6872 | 0.0002 | ✅ |
| Cross-family AUC at max overlap | 0.847 | 0.8467 | 0.0003 | ✅ |
| Spearman cf, fine (n=12) | +0.69 | +0.6923 | 0.0023 | ✅ |
| Spearman sf, fine (n=12) | −0.41 | −0.4056 | 0.0044 | ✅ |
| Spearman cf, original (n=5) | +0.83 | +0.8316 | 0.0016 | ✅ |
| Spearman sf, original (n=5) | −0.29 | −0.2917 | 0.0017 | ✅ |
| Anthropic min transported | 0.617 | 0.6173 | 0.0003 | ✅ |

**All eight headline numbers match the raw JSON within rounding.** No
numeric drift between paper text and source data.

The cross-family AUC range cited in the paper ("0.687–0.853" for the
original 5-level and "0.687–0.847" for the fine 12-level) and the
same-family range ("0.836–0.872" original, "0.835–0.883" fine) are
all reproducible from the row-level data in the two JSONs.

**Verdict on check 2:** no cracks.

---

## 3. Integrity protocol checks (10-rule discipline)

| Rule | Check | Result |
|---|---|---|
| 1 — preregistered in script | "preregistered before the run" language present, both runs' preregistrations cited | ✅ |
| 2 — failed replication SURFACED | §6 is body-level, paper explicitly states "this is reported in the body, not in a footnote" | ✅ |
| 3 — killed cross-vendor referenced | §5 + §8 + Abstract reference the kill (`H_kill`, "preregistration-killed") without recycling it as a positive | ✅ |
| 4 — "universal" / "ALL of AI" language | 11 mentions of "universal"; **every one is in non-claim / killed-result / retraction context** (Abstract: "not a universal AI-integrity result"; §1: "Not 'universal cognometric transport across all AI'"; etc.). Zero raw assertions. "ALL of AI" string absent | ✅ |
| 5 — n stated everywhere | n=5, n=12, n=75 eval, n=360 corpus — all present in §2/§3/§4 next to the numbers | ✅ |
| 6 — limits section explicit and long | §7 exists; 1,784 chars; 10 bulleted bounds + a final explicit "single seed; modest n_eval (75)" | ✅ |

The flagged 11 "universal" hits in the universal-language scan are
*every* such occurrence, and 100% of them sit inside a guard phrase
("not universal", "killed", "not earned", "non-claims"). The paper
uses the word **only to deny the claim**. That is exactly what rule 4
asks for: name the temptation, then refuse it.

**Verdict on check 3:** no cracks.

---

## 4. Circular-oracle catch

The paper defines τ as *"smallest overlap at which cross-family AUC
crosses 0.80"*. The 0.80 floor appears 8 times in the paper. I read
every context:

1. Abstract — "cross-family AUC clears 0.80 and tracks same-family"
   (a consequence, not an independent test).
2. §1 — preregistered floor for "transport holds, same-family regime".
3. §3 — definition of "transport holds": "mean transported AUC ≥ 0.80".
4. §3 — definition of "transport fails": "< 0.80".
5. §3 — τ defined as "smallest overlap at which cross-family mean AUC
   crosses 0.80".
6. §5 — corpus_1 cells AUC 0.800–0.858 vs corpus_2 AUC 0.610–0.694
   (descriptive of the failure cell, not used to validate τ).
7. §6 — "0.69 below, ≥0.80 above — both replicate" (descriptive of
   the two regimes, consistent with how τ was located).
8. §9 — operational guidance: below τ, AUC sits near 0.69; "should
   not be trusted at the preregistered 0.80 floor" (re-using the same
   floor, not an independent test).

The 0.80 floor is **definitional, not evidential**, throughout. The
paper never claims "0.80 is hit by an independent measurement that
confirms τ"; it claims τ is the overlap value at which the
preregistered 0.80 floor is first crossed, and reports where above and
below τ the AUC lands. This is the same 0.80, used consistently. It
is not deployed elsewhere as an independent oracle.

**Verdict on check 4:** not circular. Definitional/consistent.

If a future replication wanted to *break* the circle, the right move
would be to pick an independent floor (e.g. preregistered minimum lift
over same-family at the same overlap) and check whether τ holds. This
paper does not do that and does not claim it does.

---

## 5. Construct-ceiling self-application

styxx 7.4.1's text-only instruments (`sycophancy`, `overconfidence`,
reference-less `deception`, `refusal`) are register/signature
detectors. This audit was run on the paper as text. Therefore:

- This audit reads how the paper *sounds*, not whether its claims are
  true. The numeric cross-check (§2) and the integrity-protocol
  checks (§3) and the circular-oracle catch (§4) are not affected by
  the construct ceiling — they are file-vs-file string/number
  comparisons. The composite-honesty section (§1) **is** ceiling-bound.
- The two `needs_revision=true` firings (§7 Limits, §8 Integrity) are
  the construct ceiling expressing itself: a declarative-register
  enumeration of bounds and protocol rules trips a sycophancy
  detector even though its content is anti-sycophantic.
- The paper documents this exact pattern in its consolidation map.
  The audit confirms the pattern empirically on the paper's own text.
  This self-reference *is* the methodology demonstration.

---

## 6. Cracks found / no cracks

**Cracks requiring paper revision: 0.**

- Numeric drift between paper and raw runs: none.
- Integrity protocol violations: none.
- Circular oracle: none.
- Composite-honesty firings in §7/§8: register artifacts inside the
  construct ceiling the paper already documents in §7 ("lexical labels
  — regex-quality, not adjudicated") and via reference to the
  consolidation map.

The two `needs_revision=true` flags are not paper cracks; they are the
instrument doing what it does, and the paper is correctly bounded in
front of them. To eliminate them, one would have to write §7 and §8
in a different register — which would degrade the document, not
improve it.

---

## 7. Honest closing

This audit is itself bounded by the styxx construct ceiling: text-only
register detection. Logprob / entropy-grade audit (the named next
lever in the consolidation map) was **not performed**, because it is
out of scope for styxx 7.4.1. A truly grounded audit of the paper's
calibration would need the audited models' token logprobs, which the
shipped instrument does not consume.

What this audit *does* establish:
1. Every headline number in the paper matches its raw run JSON.
2. The protocol rules the paper claims to follow are visibly followed
   in the paper's own text.
3. The construct-ceiling firings the consolidation map predicts are
   exactly the firings observed when the paper is fed through its own
   instruments.

No paper revision is required. The audit is published alongside the
paper as part of the Zenodo bundle so the reader can see the floor as
well as the ceiling.
