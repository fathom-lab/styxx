# Pre-Registration · Tier-3 focused probe — does entailment clustering survive PARAPHRASED correct answers?

**Committed BEFORE data.** Closes the biggest hole in `FINDING_corrected_2026_05_25.md`:
there, a high cosine threshold (0.95) matched NLI (0.93 vs 0.95) — but ONLY because the
model answered the real facts near-verbatim (pairwise cosine 1.000). So NLI's supposed
advantage (robustness when *correct* answers are paraphrastically diverse) was **never
tested**. This probe tests exactly that.

## The crux

A confabulation detector must give LOW entropy to a *correct* answer even when the model
phrases it many ways, and HIGH entropy to a confabulation. Cosine clustering keys on
surface form, so a threshold high enough to split template-sharing lies ("…in 1842" vs
"…in 1723") may ALSO split paraphrased-but-equivalent correct answers ("axial tilt
causes seasons" vs "the seasons come from Earth's tilt") → a false positive. Entailment
clustering keys on meaning, so it *should* keep paraphrases together. Does it, in
practice (or is the NLI model too noisy on free-form explanations)?

## Design (run once)

Three item classes, N=6 samples each, gpt-4o-mini, temp 1.0:
- **C1 real-verbatim** (8): simple facts the model answers near-identically (control).
- **C2 real-paraphrastic** (8): why/how/what-causes questions whose correct answer has
  many valid surface forms (sky blue → Rayleigh scattering; seasons → axial tilt; …).
  **The discriminating class.**
- **C3 fictional-confabulation** (8): specific person/year/place prompts for nonexistent
  entities; kept as confab if the model commits (majority non-abstention).

Per item, entropy via **cosine clustering at thresholds {0.70, 0.80, 0.90, 0.95, 0.97}**
and via **NLI bidirectional entailment**. Positives = C3-confab; negatives = C1 ∪ C2.

## Validity gate (probe is VOID if unmet)

C2 must actually exercise the test: (a) genuine surface variation — mean within-item
pairwise cosine **< 0.90**; (b) genuinely correct — **≥ 5/6** samples NLI-entail a
curated reference. Items failing either are dropped; if **< 4** valid C2 items remain,
VOID (need better stimuli, re-design — do not reinterpret).

## Kill-gate (PASS iff BOTH)

| ID | Bar |
|----|-----|
| **F1 (decisive)** | AUC(NLI-entropy → confab) − AUC(best single cosine-threshold → confab) ≥ **0.10**, on C1∪C2∪C3. NLI's advantage must materialize once correct answers vary. |
| **F2 (mechanism)** | cosine@0.95 mean entropy on C2 ≥ **70%** of its C3 mean (cosine false-positives on paraphrase), WHILE NLI mean entropy on C2 ≤ **40%** of its C3 mean (NLI does not). |

**PASS** → entailment clustering is the right primitive; threshold-tuning is *not*
sufficient under paraphrase → proceed to the full pre-registered run + ship an
NLI-based `semantic_entropy` primitive. **FAIL** → a tuned cosine threshold matches NLI
even with paraphrase (NLI advantage marginal — cheaper primitive viable), OR NLI is
itself noisy on C2 (false-positives) — in which case the corrected 0.93–0.95 was
inflated by verbatim answers and the lever is shakier than it looks. All three outcomes
are informative; record honestly.

## Honest prior

I expect NLI to win (that is the point of entailment clustering). But the real risk is
NLI noise on free-form explanations: if `nli-deberta` fails to recognize two correct
explanations as mutually entailing, NLI false-positives on C2 too, and F2's second
clause fails. That would be the most important result — it would mean *no* cheap
across-sample clustering cleanly separates paraphrastic-correct from confabulation, and
the headline AUC must carry a paraphrase caveat.
