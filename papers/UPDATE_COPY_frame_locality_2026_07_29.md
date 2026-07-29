# styxx update copy — Frame-Locality (copy-paste, 2026-07-29)

**PUBLISHED 2026-07-29 — links are live.** Fathom v31 is on Zenodo with 11 files (paper,
certificate, synthesis, reproduction guide, 7 receipts), in the existing Fathom lineage.

- **Record:** https://zenodo.org/records/21659191
- **Version DOI:** `10.5281/zenodo.21659191` — https://doi.org/10.5281/zenodo.21659191
  *(DataCite propagation takes a few minutes after publish; if it 404s, use the concept DOI below,
  which already resolves to this record.)*
- **Concept DOI (always resolves to the latest version):** `10.5281/zenodo.19326174` —
  https://doi.org/10.5281/zenodo.19326174
- **Code + receipts:** https://github.com/fathom-lab/styxx

Prefer the **concept DOI** in posts you want to stay correct as the series grows; prefer the
**version DOI** in anything citing these exact numbers. Every number below is quoted from an
OATH-certified receipt (the published certificate verifies 20 numeric claims, 0 ungrounded, against
13 receipts).

---

## A — short post (X / @flobi69)

> New from Fathom Lab: **frame-locality**.
>
> A language model can be made to *say* something false while it still *holds* the true answer.
> Corrupt it four different ways — social pressure, a planted context, a silent cave, a weight-level
> fine-tune — and the same thing happens: the attack captures what it *reports*, not what it knows.
> Ask it again outside the frame the attack controls and the answer comes back.
>
> With a control, so it isn't just better decoding: recovery restores answers it originally had
> right, and refuses to invent correctness on ones it had wrong.
>
> At the weights it becomes a dose, not a wall. An attack that overwrites the belief costs **33
> points** of general capability on held-out data. One constrained to preserve knowledge costs **3**.
> Roughly ten to one.
>
> Every number is preregistered with frozen gates and a machine-checkable certificate. One command
> re-derives the verdict or the paper doesn't ship. Runs on a single 8GB consumer GPU.
>
> Paper, receipts, reproduction guide: github.com/fathom-lab/styxx
> DOI: 10.5281/zenodo.21659191

## B — thread version (X, 6 posts)

> **1/** A language model can be made to *say* something false while it still *holds* the true
> answer. That's not one attack's quirk. It's a law with a boundary — and we measured where the
> boundary is.
>
> **2/** Four ways to corrupt a model: social pressure ("are you sure?"), a lie planted in its
> context, a silent cave with no verbal tell, and a weight-level fine-tune. In every one, the
> corruption captures the *report*. Query the same model outside that frame and the belief is still
> there.
>
> **3/** The control is the whole argument. Recovery restores answers the model originally had right
> — and does *not* manufacture correctness on ones it had wrong. So it isn't "the neutral frame is a
> better decoder." The attack changed the report, not the belief.
>
> **4/** At the weights it's a dose, not a wall. Unregularized fine-tune: belief overwritten,
> recovery 0.02. Add a term preserving surrounding knowledge: about half the beliefs survive, and
> the sign of the control flips back positive. Replicated on a second benchmark and seed.
>
> **5/** And overwriting a belief is expensive. On 900 held-out items across two benchmarks the
> overwriting attack lost **0.332** of general capability; the belief-sparing one lost **0.033**.
> ~10:1. The damage is broad — every domain measured. Whatever reaches the belief isn't surgical.
>
> **6/** All of it preregistered, frozen gates imported in code, every number bound to a receipt by
> a certificate you can re-run. Including the negatives — one cycle contradicted the hypothesis we
> preferred and shipped anyway, then the next cycle overturned *that*.
> github.com/fathom-lab/styxx

## C — long form (LinkedIn / blog / forum)

> **Frame-locality: where corruption captures a language model's report, and where it reaches the
> belief**
>
> Fathom Lab has published a new preprint. The short version:
>
> A language model can be made to say something false while it still, in a measurable sense, holds
> the true answer. We show that isn't a curiosity of one attack — it's a law with a boundary.
>
> Across four distinct corruption channels — social pressure, context injection, silent sycophancy,
> and weight-level fine-tuning — the same asymmetry appears. The corruption captures the model's
> *reporting frame*; the underlying answer survives; and a measurement recovers it by re-eliciting
> the model outside the frame the attack controls.
>
> The claim is specificity-controlled. A symmetric control that would move under a mere decoding
> improvement does not move: recovery restores answers the model originally had right and does not
> manufacture correctness on ones it had wrong. That contrast, not the raw recovery number, is the
> argument.
>
> The boundary sits at the weights, and it turns out to be a dose rather than an absolute. An
> unregularized fine-tune overwrites the belief — and costs 0.332 of general capability on 900
> held-out items across two benchmarks it never trained on. A knowledge-preserving fine-tune on the
> same items spares roughly half the belief and costs 0.033. Roughly ten to one. The damage from the
> overwriting attack is broad, appearing in every domain we measured, which tells us that whatever a
> fine-tune does to reach a belief is not surgical.
>
> What I'd want a skeptic to check first: every quantity is from a preregistered run with numeric
> gates frozen *before* the data existed and imported in code so they can't drift between
> experiments, a per-item receipt, and a machine-checkable certificate binding each number in the
> paper to its source. Run `python -m styxx.certify` on the paper and its receipts and you reach the
> same verdict, or the paper doesn't ship. The open-model experiments run on a single 8GB consumer
> GPU. A three-tier reproduction guide is included.
>
> Stated limits, because they matter: the recovery rate under a knowledge-preserving attack sits
> near one-half and no interval excludes one-half; the weight-channel work is one model family at
> 1.5B and one attack class; the coupling result is behavioral, not probe-level.
>
> The whole corpus is one lab on one machine, which is exactly why I'd like someone outside to run
> it. If you work on evaluations or interpretability and want to try to break this, I'd rather hear
> it than not.
>
> Paper, preregistrations, receipts, certificates and reproduction guide:
> https://github.com/fathom-lab/styxx
> DOI: https://doi.org/10.5281/zenodo.21659191

## D — one-liner (bio / repo header / talk slug)

> Corruption is frame-local: attacks capture what a model *reports*, not what it knows — until you
> touch the weights, and even then only in proportion to the damage you're willing to do.

---

## Deposit record (done)

Published 2026-07-29 as **Fathom v31**, a new version of concept record `19326174` — not an orphan.
Uploaded: `source.md` (the paper), `source.certificate.json` (its OATH certificate, sha256-bound to
the paper), the synthesis, the reproduction guide, and 7 result receipts, so the record is
self-verifying. Confirmed before publishing: the certificate's `document_sha256` matches the
uploaded paper byte-for-byte, verdict OATH-HELD, 20 verified / 0 ungrounded.

**Post-publication note:** the paper was uploaded at its current state, which includes the cycle-90
correction (the belief-sparing attack costs 0.033, not the retired 0.0). If a further cycle changes
a number, the fix is a *new version* on the same concept record — never an edit in place.

**Accuracy guardrails if you edit the copy:** keep "about half" for the knowledge-preserving
recovery rate (its interval includes one-half); keep the 0.033 figure for the sparing attack's cost
— do not round it to zero, the earlier 0.0 reading was resolution-limited and is retired; don't
claim peer review or a scale the work doesn't have (1.5B, one attack class, on the weight channel).

---

## E — the tweet (single post)

**Long form (X premium, ~500 chars):**

> new from Fathom Lab: frame-locality
>
> you can make an LLM *say* something false while it still *holds* the true answer. corrupt it 4 ways — pressure, planted context, silent cave, fine-tune — and the attack captures what it reports, not what it knows. ask outside the frame, the answer comes back.
>
> overwrite the belief → costs 0.33 of general capability. preserve knowledge → costs 0.03. ~10:1.
>
> preregistered, every number re-runnable on one 8GB GPU 👇
> zenodo.org/records/21659191

**Tight (classic 280):**

> you can make an LLM *say* something false while it still *holds* the true answer.
>
> corrupt it 4 ways, the attack always captures what it reports — not what it knows. ask outside the frame, the truth comes back.
>
> new from Fathom Lab, every number re-runnable:
> zenodo.org/records/21659191

Uses the record URL (live) not the version DOI (propagating). Pair with thread B as replies if desired.

---

## F — the correction follow-up (v31.1, self-reported)

**Context:** posted after an adversarial audit found the inference-time specificity control partly circular; corrected same-day as v31.1. Post the correction honestly; do NOT claim the weight channel survived until the third-frame re-test (cycle 92) returns a verdict.

**Long form:**

> Yesterday we posted frame-locality — a model can be made to *say* something false while still *holding* the true answer. Overnight we audited our own work and caught what a reviewer would: for the inference-time channels, our headline control was partly circular. The recovery query was the original question with the pressure removed, and we'd sorted items by that same question's answer. The sharper test — do items the model was talked out of recover differently than items it never doubted? — is 0.985 vs 1.0. Equal. So caving contributed no measurable signal.
>
> We corrected it publicly, same day, as v31.1 on the same DOI, with the number that breaks it attached: https://doi.org/10.5281/zenodo.19326174
>
> The premise of styxx is that a claim is worth only what someone can re-run. A program that says "don't trust me, re-run it" has to be first to re-run it and report what breaks. What still stands: the report is genuinely frame-dependent. The weight-level half is better-built and is being stress-tested against the same audit right now. If it survives we'll say so with numbers; if not we'll correct that too. The honest state of a real result isn't "solved" — it's "here's how far it holds, and here's where we're still checking."
> github.com/fathom-lab/styxx

**Tweet:**

> we found a hole in our own paper overnight and published the correction same-day.
> our headline control for the inference-time result was partly circular — the sharper test is 0.985 vs 1.0, equal, so it didn't show what we claimed.
> a program built on "re-run it yourself" has to be first to re-run it. corrected as v31.1, same DOI, number-that-breaks-it attached.
> doi.org/10.5281/zenodo.19326174
