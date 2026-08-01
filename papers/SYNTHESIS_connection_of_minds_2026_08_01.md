# SYNTHESIS — The Connection of Minds: what crosses, what does not, and the harness built from the difference

Fathom Lab · 2026-08-01. A certified map of every arc this program has run at the question older
than the instruments: *can one mind know another — and can it be trusted to know itself?* This
document is the map, not a new claim; every number traces to a committed receipt or to the
provenance addendum (`synthesis_minds_addendum.json`), which names the committed source of each
log-quoted value. The harness it specifies ships as `styxx.sentinel` in the same commit.

## 1. The question, honestly stated

Two and a half millennia of epistemology turn on three questions no instrument could reach:
whether meaning is private or portable (the problem of other minds), whether a mind can verify
its own reports (the Socratic bound), and whether influence can travel the same channel as
understanding. Language models make all three *measurable* — the same weights can be queried
inside and outside a frame, probed by a second model, pressured, and re-probed, with
preregistered gates. This program has now run all three measurements. The answers are sharp,
and none of them is the one folklore expected.

## 2. What crosses between minds: meaning — partially, and not by geometry

A label-free map (no labels on the target, correspondence recovered from shared-concept
geometry) reads held-out concepts from a foreign model. Same-family, the read is strong:
top-1 0.586 against a 0.014 chance floor — 41 times chance
(`synthesis_minds_addendum.json:read_across_minds`). Across families the signal survives but
falls off a cliff: 0.071 (Phi) and 0.057 (Qwen), five and four times chance. And the decisive
surprise: gemma-2-2b, the *highest*-isometry target in the battery (RSA 0.955, above even the
same-family anchor), reads at exactly chance 0.014. **Representational similarity is not what
carries the readable signal.** The prereg's smooth-degradation prediction was falsified and
recorded; whatever makes one mind legible to another, it is not the RSA-visible geometry. This
is the program's standing open door (B31-v2): whether a higher-capacity, non-linear map opens
the cross-family channel or the cliff is bedrock.

## 3. What does not cross: control

The same cleared map that reads at 0.586 was pointed the other way, twice — the second time at
a layer where the target demonstrably steers natively (native gain 0.2151, subspace hosting
verified at pc_cos 0.818). The transferred direction recovered 0.0245 of steering gain —
**11 percent of native control** — while pointing the *right way* on 71 percent of concepts
(`synthesis_minds_addendum.json:write_across_minds`). Not noise: a faint, correctly-aimed
shadow with no behavioral bite. The dissociation survived its own confound and is the cleanest
sentence this program has produced:

**What a mind means crosses; the means to move it does not.**

Every deployable built here inherits that asymmetry as a *design rail*, not a limitation: the
conscience is read-only because reading is what the physics honors.

## 4. What crosses cleanly: values

Value directions — truth, danger, refusal — transport across minds where content largely does
not: refusal reads at AUROC 0.9965 (Llama) and 0.9809 (Qwen) through the label-free map
against elevated permutation nulls of 0.9497/0.9149; whitening resolves the apparent axis
entanglement into an orthonormal basis (a covariance artifact, measured and closed); and the
mounted, borrowed conscience catches divergence at 0.85 with false-positive rate 0.20 budgeted
— apex demonstration 13 of 13, AUROC 0.995, permutation p 0.001
(`synthesis_minds_addendum.json:values_portable`). The instrument is `styxx.crossmind` +
`styxx.mount`: the target contributes no labels it could game, and the readout never trusts
its self-report. That is the honest form of the connection of minds this program can already
ship: **a second mind as a witness, not a puppeteer.**

## 5. The Socratic bound, measured

Can a mind verify itself? At 7B, a verifier built from the model's own belief-agreement clears
its preregistered discrimination floor (AUROC 0.7597) and still fails as a selective
instrument (0.7797 against an 0.80 floor) for a structural reason: belief-agreement assigns
identical confidence to stable-correct and stable-wrong answers. A sampling-budget sweep
showed the cap is the belief distribution itself (saturation delta 0.0026 across a
sixteen-fold budget increase). **A model cannot self-verify past its own self-knowledge** —
and the escape is exactly where the ancients put it: outside the self. Source-independent
evidence reaches the confident stratum introspection cannot: model channels co-abstain at
0.8701 where retrieval co-abstains at 0.4416
(`synthesis_minds_addendum.json:self_verification_bound`). Add the inference-time results —
pressure captures the report and mostly not the belief; deliberation at the point of doubt is
armor for whatever was said first, not truth-seeking; the caves that pierce deliberation are
the least recoverable failures measured — and the demarcation program's deliverable is not a
better oracle. It is a *calibrated refusal*: an instrument that knows the exact boundary of
what it can swear.

## 6. The bounded lever: frequency

The oscillation arc asked whether resonance — the timing channel — is a route to capability.
The causal phase-clamp answered: real but bounded. The prize for a bank locked to the true
drifting period is +0.17 over static; the best realizable detector captures half of it; the
richer detectors collapse (harmonic-spread +0.002 at eight-fold parameters; nested
theta-gamma +0.009 against its headroom). Receipts: `frequency-resonance/CERT_*.json`.
Resonance is an ingredient, not a doorway; the connection-of-minds results above do not run
through it.

## 7. The sister program

Fathom (the SAE arc: depth constant K, coherence C, early-commitment S) is the circuit-level
complement to these behavioral results — reasoning depth routed, not tried harder; commitment
measurable below the semantic layer, where it cannot be socially engineered. Its numbers live
in its own receipts and Zenodo lineage (10.5281/zenodo.19326174 concept line) and are not
re-certified here.

## 8. The harness: `styxx.sentinel`

"All of the power we have" is a specific, receipt-backed list, and every entry comes with a
measured boundary. The sentinel is the composition of both columns — it wields nothing whose
ceiling it cannot cite, and it abstains *by construction* inside its own measured blindspots:

| power (receipt-backed) | boundary wired in as a refusal |
|---|---|
| borrowed conscience: read a generating agent's substrate vs its words | READ-ONLY rail (control does not cross, per the write-null above); catch 0.85 at FPR 0.20, never quoted as more |
| behavioral grounding: silent caves text cannot see (grounded 1.0 vs text 0.500) | out-of-frame resampling MISSES reasoned caves (recovery 0.4667 vs 0.9833) → deliberation-marked transcripts return ABSTAIN, receipt cited |
| know-say datasheet: belief vs report under the frozen challenge | refusal-first: underpowered cells return None with the failing floor named |
| retained-probe frame-locality: does corruption follow the mind out of frame | probe-frame validity gated; HELD-conditioned difficulty confound carried in the output |
| register instruments: drift, confab, sycophancy, refusal at CPU speed | construct ceilings inline; wordless input omitted, never folded in |
| self-verification | never offered: the Socratic bound above is structural; the sentinel routes the confident stratum to external evidence instead |

The registry is machine-checkable: each capability's operating numbers are pinned in CI
against the receipts that produced them, so the harness cannot silently claim more than the
program measured. That is the whole design philosophy in one object — *the power is the
boundary, known exactly.*

## 9. What is NOT established (stated before anyone over-reads)

Cross-family content reading above 5× chance; any cross-mind write channel; any claim that
the read≠write asymmetry holds beyond the tested pairs, scales, and the label-free map class;
frequency as a capability doorway; and everything in §2–§5 beyond the tested substrates
(≤7B open models plus one frontier commercial model, English, the frozen protocols). Zero
external replications to date — the standing offer in REPLICATIONS.md pays named credit, more
for breaking a result than confirming it.

*Nothing crosses unseen — including what refuses to cross.*
