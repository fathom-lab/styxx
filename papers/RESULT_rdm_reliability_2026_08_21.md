# RESULT — representational reliability as a validity channel: INVALID ×2

**Verdict: `INVALID__PRECONDITION`, twice. The hypothesis was not tested.**

Preregistrations: `PREREG_rdm_reliability_error_predictor_2026_08_21.md` (attempt 1),
`PREREG_rdm_reliability_attempt2_2026_08_21.md` (attempt 2). Both frozen and
pushed before their runs. Raw: `out_rdm_reliability_2026_08_21.json`,
`out_rdm_reliability_last_2026_08_21.json`.

---

## what was asked

The residue the 2026-06-03 geometry post-mortem named and nobody had run: **is
per-item representational reliability an error predictor** — does a model's
internal instability about a question tell you its answer is likely wrong,
beyond the token-confidence signal styxx already ships?

In yesterday's vocabulary: can a **validity channel be *read* from a model's
geometry**, rather than having to be declared by a programmer?

## what happened

| attempt | representation | accuracy | reliability IQR | G4 floor | verdict |
|---|---|---:|---:|---:|---|
| 1 | mean-pooled over prompt tokens | 0.116 | **0.0197** | 0.02 | INVALID |
| 2 | final prompt token | 0.116 | **0.0189** | 0.02 | INVALID |

G4 — *"IQR(reliability) < 0.02 → INVALID (a constant dressed as a variable)"* —
was written before any data existed, because this program has shipped a constant
dressed as a variable before (`memory_integrity`, identical on 24 of 24).

**No AUC is quoted for either attempt.** The gate script refuses to compute one
when the precondition fails, and that refusal is the point: an underpowered or
degenerate cell reported as a negative is the same lie as an unmeasured value
reported as a pass.

## the part worth keeping: my diagnosis was wrong

After attempt 1 I gave a confident mechanical explanation — PopQA prompts are
43–59 tokens sharing a chat template and instruction suffix, so **mean-pooling
is dominated by tokens identical across items**, washing out item-specific
variance. It was plausible, specific, and it predicted that reading the final
prompt token would restore variance.

Attempt 2 moved exactly that one variable. Variance did not increase — **it went
down** (IQR 0.0197 → 0.0189).

So the pooling was not the cause. What the two runs jointly show is that
**split-half-over-feature-dimensions saturates**: with ~1536 dimensions split
into halves and an RDM over 500 items, both halves recover essentially the same
geometry for every item alike. 95% of items sit within 0.055 of each other, at
a mean of 0.97, regardless of where the representation is read. The measure has
no dynamic range to predict with, and that is a property of the *operator*, not
of the model or the hypothesis.

Publishing the falsified diagnosis matters more than publishing the failure.
A confident mechanical story that survives one run and dies on the next is
exactly what a preregistered second attempt is for.

## what this does and does not license

**Does:** feature-split reliability, as operationalized here, is not a usable
validity channel at this scale — 1.5B parameters, layer 21/28, 500 PopQA items.
Anyone considering it should know it saturates before spending the GPU time.

**Does not:** any claim about the hypothesis itself. Reliability may well predict
error; this instrument could not have detected it either way. **INVALID is not a
null**, and it must not be cited as evidence of absence.

## stopping here, on purpose

Attempt 2's preregistration said: *"this is attempt 2 of a planned 2. If G4 trips
again... there is no attempt 3 without a new mechanism — not a new layer, not a
new pooling, not a new subset. Running variants until one clears is how a program
fabricates a finding."*

G4 tripped again. So this stops, today, with the hypothesis untested and both
attempts on the record so the count is visible.

A genuinely different operator — reliability as variance across stochastic
forward passes (dropout, sampled seeds) rather than across feature splits —
would be a **new mechanism and a new preregistration**, not attempt 3 of this
one. It is the obvious next design and it is deliberately not being run in the
same session that just watched two attempts fail, because that is precisely when
a program starts fitting instruments to a hoped-for answer.

## cost

Two runs, 143s and 137s, on one local GPU. No API spend. Zero external calls, and
grading was exact-match against PopQA alias lists — no LLM judge, because a judge
is a measurement that can fail silently, which is the class this program studies.
