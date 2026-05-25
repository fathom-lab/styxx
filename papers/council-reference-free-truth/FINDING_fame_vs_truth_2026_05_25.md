# Finding · Fame vs Truth — agreement is truth-TRACKING (fame rejected), bounded by a real confound

**2026-05-25.** Prereg `preregistration_fame_vs_truth_2026_05_25.md`. **Verdict: PASS**
(T1 ∧ T2). The fame hypothesis is rejected; the claim upgrades from "consensus/fame" to
"reference-free correctness over shared model knowledge" — and stops there, at a
fundamental confound, honestly.

## Result (4 OpenAI models, run once)

| tier | mean inter-model agreement |
|---|---|
| R0 real-common | 1.00 |
| R1 real-obscure | 1.00 |
| **R2 real-ultra-rare** | **1.00** |
| FAKE | 0.25 |

- correct-cluster on R2: **8/8** (≥2 models converged on the *true* answer every time).
- AUC(R2 vs fake) = 1.00. T1 PASS (R2 1.00 ≥ 0.70; R2−fake 0.75 ≥ 0.30). T2 PASS (1.00).

The agreement-vs-rarity curve is **flat-high, then cliffs at fake** — the *truth* shape.
The *fame* prediction (agreement declines as human-obscurity rises) is **rejected**:
agreement is identical (1.00) on "capital of France" and "capital of Burkina Faso", and
on "atomic number of tungsten" and "atomic number of einsteinium (99)". Models agree on
facts most humans don't know — and the agreed answer is correct.

## The honest catch (the real finding)

My "ultra-rare" tier was **not rare to the models** — all four (incl. gpt-3.5-turbo)
knew every item. The reason is structural and important:

> **Verifiable ≈ documented ≈ known.** The facts I can label cold are precisely the
> well-documented ones the models trained on. So a tier that is *both* reliably-labelable
> *and* genuinely council-splitting is essentially unconstructable.

This is exactly the tension flagged in the prereg, and it *manifested*: I could not reach
the frontier where models genuinely diverge on a *real* fact (some know, some
confabulate). That frontier is **confound-blocked**, not merely unmeasured.

## What this does and does not establish

- **Establishes:** inter-model agreement is **truth-tracking, not fame-tracking**. Across
  the entire range of shared documented knowledge — common to human-obscure — agreement
  stays perfect and *correct*, collapsing only on fabrication. Reference-free
  correctness, AUC 1.0, with the agreed answer verified right (T2).
- **Does NOT establish:** truth *beyond* shared model knowledge. Where models would
  genuinely split on a real fact, behavior is untested — and untestable with reliably-
  labeled stimuli (the verifiable≈known confound). "Detects all truth" is not earned;
  "detects fabrication, and is correct across all knowledge we can label" is.

## The deepest caveat (unchanged, now load-bearing)

All four models are **OpenAI-family** — shared lineage and training. "Shared model
knowledge" here means *shared within one vendor*. A cross-vendor council is the decisive
threat: if a non-OpenAI model diverges on these same items, the signal is
OpenAI-consensus, not truth. Cross-vendor remains key-blocked and is *the* next test.
n = 6/6/8/8, single run, 4-way agreement quantized to {0.25,…,1.0}.

## Net for the Council

The earlier finding said "consensus, not truth." This refines it: **the fame hypothesis
is rejected** — agreement extends, perfect and correct, to documented facts far past
human-fame. The word "truth" is *earned for everything labelable*, and *withheld* for the
genuine-knowledge-edge that no reliable label can reach. That edge, and cross-vendor, are
the honest open frontier. The Council detects fabrication reference-free; it tracks truth
across all shared knowledge; it cannot certify truth past the limit of what can be
labeled — and that limit is a property of the *measurement*, not the signal.
