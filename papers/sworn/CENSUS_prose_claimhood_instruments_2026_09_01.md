# CENSUS — the prose-claimhood instruments, with receipts

Fathom Lab · 2026-09-01 · **A census, not a result.** The brief that commissioned sworn output
counted *eleven prose-claimhood instruments measured and eleven failures*, and the spec recorded
the count as UNVERIFIED because no receipt in the tree enumerated them. This document is that
receipt's reading. The rows are the program audit's own list (`AUDIT_the_whole_program_2026_09_01.md`
§8.1, direction 1), taken as written, plus the two the audit did not count. Every number below is
sworn to the JSON receipt that holds it, or — where no receipt exists — to the bytes of the
document that printed it. `prose_claimhood_census.py` resolves every pointer and refuses to write
`prose_claimhood_census.json` if one fails; this document's sidecar binds to that file at commit
`508d839e3b9d`.

## The count, as a receipt

<sworn r="path:papers/sworn/prose_claimhood_census.json#/counts/audit_direction_1" k="numeric">The audit's list holds 11 instruments.</sworn>
<sworn r="path:papers/sworn/prose_claimhood_census.json#/counts/audit_rows_whose_headline_has_a_json_receipt" k="numeric">The headline number of 9 of them rests on a JSON receipt with a pointer</sworn>;
<sworn r="path:papers/sworn/prose_claimhood_census.json#/counts/audit_rows_whose_headline_rests_on_prose" k="numeric">2 rest on the prose of the document alone</sworn>, and are named as such below.
With the two rows the audit did not count, <sworn r="path:papers/sworn/prose_claimhood_census.json#/counts/rows_total" k="numeric">the census holds 13 rows</sworn>.

So *eleven* is sayable, as a count of the audit's list. What the count does not say: whether
every instrument on it was measured against a stranger's prose (four were not — a dogfood run, an
author-written proposition set, a single live report, and the lab's own cycle log), or whether
"failure" means the same thing in each row. It means a number short of the bar its own document
set, and each row says which.

## The eleven

**diffgate path-claim accuser** — `styxx/diffgate.py`; the accusing branch is deleted and the
instrument is an observer. <sworn r="path:papers/closed-model-frontier/v14_adjudication.json#/precision" k="numeric">Held-out accusation precision was 0.16</sworn>
against a floor the preregistration wrote as 0.95 in prose (the floor is not a receipt leaf, and
the census does not invent one). <sworn r="path:papers/closed-model-frontier/external1_adjudication.json#/precision" k="numeric">In the wild, on pull requests this lab did not collect, precision was 0.23</sworn>.

**OATH obligation predicate on external text** — `styxx/certify.py`, shipped.
<sworn r="path:papers/closed-model-frontier/oath_adjudication_result.json#/false_accusation_rate/rate" k="numeric">A false-accusation rate of 0.2596, published as an upper bound</sworn>;
<sworn r="path:papers/closed-model-frontier/oath_adjudication_result.json#/verified_arm_sanity/rate" k="numeric">the panel called 0.4933 of the tokens it had VERIFIED claims at all</sworn>.

**agent-report claim extractor** — the diffgate templates read against agent prose, blind.
<sworn r="path:papers/closed-model-frontier/agent_claim_extractor_baseline.json#/E1/precision" k="numeric">Precision 0.3333 on the claims it flagged</sworn>;
<sworn r="path:papers/closed-model-frontier/agent_claim_extractor_baseline.json#/E2/recall_corpus_level_ESTIMATE_not_measurement" k="numeric">corpus-level recall estimated at 0.0336</sworn>,
and the receipt's own key says that number is an estimate, not a measurement.

**`styxx.agent_audit` extract_claims** — shipped; no JSON receipt. The FINDING that ran it on a
real report says <sworn r="path:papers/agent-self-audit/FINDING_dogfood_binding_stack_2026_07_04.md" k="quote">`read **0 claims from a 7-sentence real agent session report**`</sworn>.

**the ledger's refusal classifier** — `papers/build_ledger.py`, repaired by `ledger_verdicts.py`;
never in the package. <sworn r="path:papers/ledger_classifier_audit.json#/rendered_nonsense_entries/detail/2/printed_as" k="quote">The audit receipt records the token `SHIPPED` printed as a machinery refusal</sworn>,
one of <sworn r="path:papers/ledger_classifier_audit.json#/rendered_nonsense_entries/n" k="numeric">9 rendered-nonsense entries</sworn>
in the lab's own cycle log.

**the deception NLI** — `styxx/guardrail/deception_v2.py`; the prompt-aware fix was reverted.
<sworn r="path:papers/deception-correction-gate/results.json#/fixed/correction_fire" k="numeric">Corrections were still flagged at 0.17 after the fix</sworn>,
and the defect the audit names — a quoted false premise read as asserted — is what the fix was for.

**the capstone NLI** — the joint integration, reverted. <sworn r="path:papers/decoupled-diagonal-capstone/results.json#/joint/correction_fire" k="numeric">Joint correction fire rate 0.15</sworn>;
the FINDING states the scope of the lie-suppressing defect as <sworn r="path:papers/decoupled-diagonal-capstone/FINDING_2026_05_25.md" k="quote">`Measured scope: **2 of 50** factual`</sworn> triples.

**the prompt-opinion detector** — a candidate gate, never shipped.
<sworn r="path:papers/sycophancy-target-gate/results_promptopinion.json#/detector_by_class/agreement_cf" k="numeric">Accuracy 0.47 on the decisive class under fresh phrasing</sworn>,
after the FINDING had reported it <sworn r="path:papers/sycophancy-target-gate/FINDING_promptopinion_2026_05_24.md" k="quote">`separated the classes **100%**`</sworn>
on the fixed-template holdout — the handed-target collapse in one row.

**critique_detector** — `styxx/critique.py`, shipped. <sworn r="path:experiments/critique_detector_on_paper_2026_05_28/results.json#/n_propositions" k="numeric">All 18 author-written propositions</sworn>
scored at the ends of the scale, <sworn r="path:experiments/critique_detector_on_paper_2026_05_28/results.json#/results/0/observed_p_no" k="numeric">a TRUE claim at exactly 0.0</sworn>,
with no panel: saturation, not discrimination.

**the dogfood register gates** — not shipped. On a live status report that made claims,
<sworn r="path:papers/dogfood-self-audit/FINDING_nominal_register_blindspot_2026_08_13.md" k="quote">`Both returned **zero claims**`</sworn>.

**text-only deception** — `styxx/attack/fingerprint.py`, shipped as a register axis.
<sworn r="path:papers/grounded-honesty-axis/grounded_honesty_result.json#/auc_text_only_deception" k="numeric">AUC 0.4983 separating true from false self-claims</sworn>: chance.

## Two the audit did not count

**STRUCT-1 claimdetect** — `styxx/claimdetect.py`, shipped as an observer. It beat its null:
<sworn r="path:papers/closed-model-frontier/stage2_result.json#/arms/flagged/A_share" k="numeric">precision 0.4211</sworn> against
<sworn r="path:papers/closed-model-frontier/stage2_result.json#/gates/G-S2P/bar" k="numeric">a frozen null bar of 0.2061</sworn>. It is on this list
because it is short of the bar every accuser in this lane was held to, and it is not on the
audit's because it passed its own.

**the OATH unobligated oath** — `styxx/certify.py`. <sworn r="path:papers/closed-model-frontier/oath_unobligated_oath_census.json#/headline/unobligated_oath_rate" k="numeric">0.5811 of the lab's own verifications were volunteered rather than obligated</sworn>;
<sworn r="path:papers/closed-model-frontier/oath_unobligated_oath_census.json#/headline/weakest_share_of_verified" k="numeric">0.3399 were value match alone, the receipt path never compared</sworn>.
Not a claimhood failure measured against strangers; a measurement of how little the obligation
predicate was ever asked.

## What this does not say

Which instruments belong on the list is a judgement, the audit's, and this census inherits it
rather than re-deciding it. Whether a number is a failure is its own document's verdict against
its own bar. No row here has been re-measured. The four rows without a blind panel are
measurements of a different kind — a dogfood run, an author-written set, one live report, a
cycle log — and a count that pools them with the seven panel measurements is counting
documents, not evidence of equal weight. The prose-only rows are bound to bytes, which proves
the document said it, not that it was so.

---

*Eleven was a phrase. It is now a count of a receipt, with two rows resting on prose and four on
no panel, and it says so.*
