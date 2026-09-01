# DOGFOOD — styxx's instruments pointed at styxx's own work, 2026-09-01

Fathom Lab · 2026-09-01 · Every number below was produced today by running the shipped
instruments on documents this lab wrote today, and every command is in the text so anyone can
re-run it. Nothing here is a claim about a model. It is a claim about our own tooling.

---

## 1. The PR self-report gate has never checked anything

`.github/workflows/audit-claims.yml:26` names a required check **`falsify PR self-report`**. It
runs `styxx audit-claims pr_body.md --repo .`. Run over every substantive pull request this repo
has merged:

```
PR    sentences  matched  coverage  claims  gate
#32          38        0      0.00       0  PASS
#33          29        0      0.00       0  PASS
#34          26        0      0.00       0  PASS
#35          32        0      0.00       0  PASS
#36          15        0      0.00       0  PASS
#37          28        0      0.00       0  PASS
#41          37        0      0.00       0  PASS
#42          19        0      0.00       0  PASS
#45          16        0      0.00       0  PASS
#46          15        0      0.00       0  PASS
#47           7        0      0.00       0  PASS
#48          41        0      0.00       0  PASS
#50          28        0      0.00       0  PASS
```

**331 sentences. Zero claims extracted. Coverage 0.00 on every pull request. PASS on every one.**

**The tool is not at fault and this must be said first.** Its own output is scrupulous: *"extracted
0 checkable claim(s) from 28 sentence(s) (coverage 0.00); free-form prose not checked"* and
*"no checkable claims found — nothing to falsify"*. An independently written regex over the same
13 bodies finds **zero** claim-shaped sentences too, so 0.00 is the correct answer. This lab does
not write path claims in pull request descriptions; it writes findings and numbers.

**The defect is at the CI layer.** A required check called *falsify PR self-report* showing green
reads, to anyone who has not opened the log, as *the self-report was checked and survived*. It
means *there was nothing this instrument can check*. That is an absent measurement wearing a
passing check — the same shape as `tests/test_ledger.py` skipping under a shallow checkout, found
earlier today.

**Cheapest honest repair:** have the step fail, or emit a neutral status, when
`claims_extracted == 0`, so green means checked. Renaming it to `pr self-report: coverage` would
also do.

---

## 2. An OATH verdict is a function of the receipt list the author chose

`RESULT_extraction_ceiling_REFUTED_2026_09_01.md`, certified with `certify_doc` three times, same
bytes, different receipt sets:

| receipts supplied | VERIFIED | ABSTAIN | UNGROUNDED | verdict |
|---|---|---|---|---|
| 5 files, the ones the document cites | 26 | 28 | **3** | **OATH-FAILED** |
| 8 files, adding the packet and key | 28 | 28 | **1** | **OATH-FAILED** |
| the full pool of both arcs' JSON | 42 | 15 | **0** | **OATH-HELD** |

**The same document fails and passes depending on how many receipts its author hands over, and
the author chooses.** `OATH-HELD` is therefore a joint statement about a document *and a curated
receipt list*, and only the first half is what a reader hears.

### And the grounding that flipped it is coincidence-eligible

The token that decided it was **55** — the central finding of the document, `E = 0.55`, written as
*"the panel found 55 were made about a sentence that really was claiming…"*. It is genuinely in
`extraction_panel_result.json` at `/decomposition/n_claim`.

But counting every numeric leaf in the widened pool whose value equals 55:

```
leaves in the receipt pool whose value == 55: 257
   extraction_panel_result.json  /decomposition/n_claim      <- the correct one
   b23_fable_result.json         /rows[55]/i                 <- an array index
   b24_headtohead_result.json    /rows[47]/i                 <- an array index
   oath_corpus_attestation.json  /oath_failed_docs[0]/ungrounded[8]/line   <- a line number
   ANALYSIS_base_rate_ceiling…certificate.json /ledger[58]/col             <- a column number
   ... 252 more
```

**One correct leaf, 256 meaningless ones.** The verifier awards VERIFIED on a value-match without
comparing the receipt *path* to the claim, so widening the pool raises the chance of grounding by
accident far faster than the chance of grounding correctly. This is exactly the defect issue #39
already names — *"604 false attestations … coincidental matches, which only status-level
claim→field binding for floats will touch"* — now demonstrated on this lab's own headline number
of the day at a 256:1 noise ratio.

**What this does not show:** that the 55 in the document is wrong. It is right, and its receipt
exists. What is shown is that the seal cannot distinguish that from luck.

---

## 3. The verifier accuses a DOI

`PREREG_third_party_precision_2026_09_01.md` certifies **OATH-FAILED** with 11 ungrounded tokens.
The first is:

```
token 3793302.3793583  @ line 66
context: Agent-Authored Pull Requests*, MSR'26 Mining Challenge Track, DOI 10.1145/3793302.3793583
```

**That is a DOI being parsed as a decimal number and then accused of lacking a receipt.** A
bibliographic identifier is not a measurement and cannot have one. Issue #39 already anticipated
this class — it lists *"the arXiv DOI prefix 10.48550, which neither `_VERSIONISH` nor v0.5 class
C reaches"* — and here it fires on a real citation in a real preregistration.

**This is a false accusation by the lab's own definition**, and it is the class of defect that got
the path-claim accuser retired. It should be counted the same way.

---

## 4. Today's own binding rate

Over the seven documents written today, certified against the full pool:

**VERIFIED 345 · ABSTAIN 89 · UNGROUNDED 11 · bound fraction E = 0.7753**

Against the corpus-wide figure measured across all 241 committed certificates —
**E = 0.7237** — today's documents bind slightly better than the historical average and in the
same band. Read that as: *nothing improved today, and nothing regressed.* Roughly a fifth to a
quarter of every number this lab writes is not bound by its own seal, and that has been stable.

---

## What this dogfood establishes

1. A required CI check has passed 13 times without ever checking anything, and its name says
   otherwise.
2. An OATH verdict on fixed bytes is not a fixed verdict — it moves with the author's receipt
   list, and moved from FAILED to HELD today.
3. The grounding that produced HELD had 256 wrong candidates and one right one, and the verifier
   cannot tell them apart.
4. The verifier issues at least one false accusation against a DOI.
5. The bound fraction of today's work, 0.7753, sits in the same band as the corpus, 0.7237.

Every one of these is about our tooling and none is about anyone else's. They are published here
rather than repaired quietly, and none of the repairs are made in this document — items 1 and 3
change shipped behaviour and require their own preregistration and, for the accusing side, a
blind panel.
