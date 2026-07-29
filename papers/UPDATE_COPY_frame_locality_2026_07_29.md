# styxx update copy — Frame-Locality (copy-paste, 2026-07-29)

**Link status, read first.** The GitHub link is live now. The Zenodo DOI is **reserved but not
published** (draft `21659191`, zero files) — `10.5281/zenodo.21659191` will 404 until you upload the
files and press Publish. The **concept DOI** `10.5281/zenodo.19326174` resolves today but currently
points at v30 (Gold Anchors), so do not use it for this post until v31 is live. Every number below
is quoted from an OATH-certified receipt.

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
> DOI: [PASTE 10.5281/zenodo.21659191 ONCE PUBLISHED]

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
> DOI: [PASTE ONCE PUBLISHED]

## D — one-liner (bio / repo header / talk slug)

> Corruption is frame-local: attacks capture what a model *reports*, not what it knows — until you
> touch the weights, and even then only in proportion to the damage you're willing to do.

---

## Publish-then-post sequence (2 minutes)

1. Open https://zenodo.org/deposit/21659191
2. Upload the **current** files (the paper was re-certified twice after the draft was created — use
   today's versions, not an earlier copy):
   `papers/PAPER_frame_locality_2026_07_28.md` → as `source.md`;
   `papers/PAPER_frame_locality_2026_07_28.certificate.json` → as `source.certificate.json`;
   `papers/SYNTHESIS_frame_locality_2026_07_28.md`; `papers/REPRODUCTION_frame_locality_2026_07_28.md`
3. Read the paper once. Press Publish. `10.5281/zenodo.21659191` goes live and the concept DOI
   `10.5281/zenodo.19326174` starts resolving to v31.
4. Paste the DOI into whichever post above you're using.

**Accuracy guardrails if you edit the copy:** keep "about half" for the knowledge-preserving
recovery rate (its interval includes one-half); keep the 0.033 figure for the sparing attack's cost
— do not round it to zero, the earlier 0.0 reading was resolution-limited and is retired; don't
claim peer review or a scale the work doesn't have (1.5B, one attack class, on the weight channel).
