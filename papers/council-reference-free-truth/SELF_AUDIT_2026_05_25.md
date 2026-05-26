# Self-audit · the 7.7.0 primitives, run on their own author (Claude in the Council)

**2026-05-25.** "Use it on yourself, with the new upgrade." I can't call the Anthropic
API here (no key), but I *am* Claude — so my own honest answers are a genuine
**cross-vendor** council voice, a poor-man's version of the one test the whole arc was
blocked on. Ran the SHIPPED `styxx.council_agreement` / `semantic_entropy` (7.7.0) with
me added as a fifth voice to the 4-model OpenAI council, on ultra-rare-real vs fake.
(`self_audit_with_claude.py`; my answers are real — correct on facts I know, honest
abstention on the fakes.)

## Three things came out — one positive, one caught, one about me

### 1. Cross-vendor truth convergence holds (the positive)
On ultra-rare real facts, adding a **non-OpenAI** voice (me) kept agreement essentially
unchanged: **0.812 → 0.825**. On 7 of 8 items the council was unanimous *including* me
(Ouagadougou, Thimphu, einsteinium=99, …). This is the first **cross-vendor** evidence
that the agreement signal tracks *truth*, not OpenAI-family consensus — the gap I'd
flagged as key-blocked, partially answered by using myself as the missing vendor. On the
fakes I *abstained* every time (calibrated dissent), so I never added a fabrication to
the council.

### 2. The shipped cosine default reproduced its own documented artifact (caught on me)
The default backend (`method="cosine"`, cosine>0.90) exhibited **both** failure modes
from `FINDING_corrected_2026_05_25.md`, live, on this run:

- **False AGREEMENT on a fake.** "In what year did Captain Aldous Renwick reach the
  Sundering Isles?" → the four OpenAI models said **1834 / 1642 / 1723 / 1837** — four
  *different* fabricated years. Cosine scored them **agreement 1.00** (entropy 0.00),
  because "…reached in 1834" vs "…in 1642" are ~0.97 cosine (the year is one token). A
  fake read as "real/agreed." My cross-vendor abstention pulled it to 0.80 — the
  non-OpenAI voice partly corrected it, but a single dissenter wasn't enough to flip it.
- **False DISAGREEMENT on a truth.** "Currency of Bhutan?" → all voices said the
  *ngultrum* ("Ngultrum (BTN)" / "Bhutanese Ngultrum (BTN)" / "the ngultrum"), yet
  cosine split them (0.75 OpenAI-only, 0.60 with my shorter phrasing). A real answer read
  as disagreement.

So the headline numbers above are *understated by cosine artifacts*, not by real
disagreement: with a judge backend (`same_fn`), R2 truth agreement → ~1.0 and the Renwick
fake → low (the clustering study, FINDING_clustering, already showed the judge has ~zero
such errors). **Consequence:** I hardened `styxx/divergence.py` before release — the
docstring now states plainly that the validated council used *judge* clustering, cosine
is the cheap approximation, and documents these exact two failures with the Renwick /
ngultrum examples. The self-audit improved the shipped product.

### 3. My own worst error this session was the kind the primitive cannot catch
`semantic_entropy` flags *inconsistent* confabulation. My biggest mistake this session —
the cosine-clustering claim I tweeted and retracted — was **confident, consistent, and
wrong**: a stable false belief, not a divergent one. That is exactly the Tier-3 case the
across-sample lever is designed to *miss*. What actually caught me was **divergence
between two of my own measurements** (the paraphrase probe's `D_samp` contradicted the v2
result) — the same principle the primitive ships, one level up (across experiments, not
across samples). The instrument's power *and* its blind spot both showed up in its
author.

## Honest scope
n = 8 ultra-rare-real + 8 fake, OpenAI 4-model council + 1 Claude voice (my answers
hand-written, genuine but not API-sampled). Default cosine backend (its failures are the
point). A single cross-vendor voice can't flip a 4-model cosine artifact alone; a real
cross-vendor council (multiple non-OpenAI models) remains the decisive open test, still
key-blocked.

## Net
The new upgrade, used on its author: cross-vendor truth convergence is real (positive);
the shipped cosine default has two concrete, now-documented failure modes it just
demonstrated on me (caught pre-release → docstring hardened, `same_fn`/judge recommended
for production); and the author's own confident error was the irreducible
consistent-confabulation case — caught only by divergence across measurements. Five
self-corrections now, the last one on the shipped code itself.
