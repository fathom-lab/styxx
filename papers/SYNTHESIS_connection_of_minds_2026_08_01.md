# The Connection of Minds: What Crosses Between Language Models, What Does Not, and the Harness Built from the Difference

Fathom Lab · 2026-08-01 · §3 added 2026-08-05 when the island arc closed. A certified map of
every arc this program has run at the question older than the instruments: *can one mind know another — and can it be trusted to know itself?* This
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

> **⚠ Correction / scope erratum (2026-08-03).** The struck sentence below was published
> without the scope that makes it true. It reports a measurement made through the *linear*
> map class, but this paragraph never says "linear" — the word does not appear in §2 until the
> paragraph that follows, and §4 supplies the scope only in retrospect — so as written it reads
> as a fact about gemma-2-2b rather than about the map. Later pre-registered experiments
> read the same model far above chance:
> - **`b31v2_result.json` (2026-08-01)** — a two-layer MLP on 392 true pairs reads gemma at
>   **0.7857** top-1 over 70 held-out concepts, against the same extractions. That finding's
>   own verdict: *the rung-2 cliff was a property of the linear map class, not of the minds.*
> - **`b34v3_result.json` (2026-08-03)** — label-free, zero labels anywhere in fitting, the
>   same model reads held-out content at **0.5714**.
> - **What was NOT withdrawn, stated so the correction is not over-read.** The measurement
>   itself stands *within its class* — b31v2's own M0 linear column reports 0.0143 for this
>   cell. The bolded conclusion that follows the struck sentence — that representational
>   similarity is not what carries the readable signal — also stands, and §3 strengthened it
>   rather than weakening it. What is withdrawn is the unscoped reading: that gemma is
>   unreadable, rather than unreadable *by that map*.
>
> *The sentence is struck, not deleted. A struck sentence a reader can still read is the
> honest form; a deleted one is a rewrite of history.*

A label-free map (no labels on the target, correspondence recovered from shared-concept
geometry) reads held-out concepts from a foreign model. Same-family, the read is strong:
top-1 0.586 against a 0.014 chance floor — 41 times chance
(`synthesis_minds_addendum.json:read_across_minds`). Across families the signal survives but
falls off a cliff: 0.071 (Phi) and 0.057 (Qwen), five and four times chance. ~~And the decisive
surprise: gemma-2-2b, the *highest*-isometry target in the battery (RSA 0.955, above even the
same-family anchor), reads at exactly chance 0.014.~~ **[Sentence withdrawn as written,
2026-08-03 — `b31v2_result.json`, `b34v3_result.json`; scope erratum at the head of this
section. Struck, never deleted.]** Corrected, with the scope the original omitted: gemma-2-2b,
the highest-isometry target in the battery (RSA 0.955), reads at exactly chance 0.014 *through
the label-free linear map*, and at 0.7857 through a two-layer MLP fit on the same extractions
(`b31v2_result.json`) — the cliff was the map class, not the mind. **Representational
similarity is not what carries the readable signal.** The prereg's smooth-degradation
prediction was falsified and recorded; whatever makes one mind legible to another, it is
not the RSA-visible geometry.

That door has since been opened — twice, each answering a different question. **With paired
anchors** (`b31v2_result.json`): a two-layer MLP on 392 true pairs reads gemma at 0.7857 —
55× chance — on 70 held-out concepts; the cliff was the linear map class, not the minds.
**And label-free** (`b34v3_result.json`): the committed linear machinery *discovers* gemma's
correspondence from geometry alone (seed accuracy 0.5918, zero labels in fitting) and a single
MLP fit on the discovered pairing reads held-out content at 0.5714 — 40× chance — holding at
0.5263 on the 57 concepts disjoint from every earlier run
(`b34v3_fresh_split_addendum.json`), and **seed-stable across five independent splits**
(`b35a_result.json`: median gemma read 0.5857, every split 37–48× chance, nulls at chance) — a
replicated property, not one lucky draw. The pairing is not merely usable; it is *findable*.
Cross-family content reading is now licensed on both axes, bounded honestly: one
strong-discovery target (gemma), a second (Qwen) whose discovery was weak, ≤3B models, 70-way
identification. Discovery strength — not isometry — is the variable that now wants a theory.

## 3. The barrier, caught and dissected: an island, a bridge, and a core two directions wide

§2 left "discovery strength" as the variable wanting a theory. Chasing it produced the arc's
sharpest chain of results. The full mutual-legibility matrix (every model reading every other,
`b37_result.json`) revealed a topology: a cross-family clique that reads each other, and an
island — qwen — that the clique cannot read, despite qwen agreeing with them on gross
relational structure. The cheap explanations died in order, each under prereg: not a
measurement cliff (b38, INVALID branch honored), not covariance shape (b39 — whitening does
not open the island the way it resolved §5's value axes), not an artifact of one seed.

Then the barrier was **built into a bridge** (`b41_result.json`): correcting qwen's top-20
concept-contrast directions in concept space took discovery from a 0.0612 baseline to 0.9745,
while a matched random 20-frame did exactly 0.0 — the barrier is *causal*, and it is *those
directions*. Swept over rank and five seeds (`b42_result.json`), the bridge replicated on every
seed and dosed perfectly (Spearman 1.0 of median bridge against rank): **k\* = 2** — two
directions buy half of full legibility (median 0.5128), a plateau follows, and a secondary
shell of roughly 6–10 directions completes it. The barrier is hierarchically low-rank, with a
rank-2 core.

And the directions have no name (`b43_result.json`): across seeds the top-loading concept
memberships barely overlap (mean Jaccard 0.1368), and their semantic coherence is
indistinguishable from random draws (permutation p 0.8031). Both naming gates failed — the
pre-committed branch — and it is the deeper answer: **the barrier is sub-symbolic.** Two minds
can share a concept vocabulary, agree on relational structure, and remain mutually unreadable
because of a causal difference two directions wide that human language has no word for. That —
not any similarity score — is what "discovery strength" was measuring all along.

One more control relocated the cause (`b44_result.json`): frames built by the identical
construction from the *wrong* models — gemma_2b (another family) and llama_1b — also open the
island (medians 0.7168 and 0.648 at k=20; every donor × seed clears the 0.30 floor), where
random frames did nothing. So the bridge was never the reader's property: **the clique shares
a common concept-frame geometry, and the island's barrier is chiefly qwen's private rotation
away from it** — with a reader-specific residual that only the reader's own frame closes, and
closes reliably (0.9745–0.9847 across all five seeds, against wrong-donor swings as wide as
0.3622–0.9056). And the shared frame is directly visible as raw subspace geometry
(`b45_result.json`): the clique's concept-frames co-align at 0.848 median squared-cosine mass
against a 0.0566 random ceiling, with the island below the clique in every seed at 0.7166 —
*mostly aligned, yet discovery-illegible*. The resolution is the shape of the legibility
function itself (`b46_result.json`): interpolating the island's frame toward the reader's,
discovery stays flat through most of the rotation (medians 0.0408 → 0.1122 → 0.3622) and
turns nearly vertical only close to alignment (0.9566 at the knee, t½ = 0.8, transition width
0.2). **Legibility is switch-like in the frame coordinate** — which is why slope measures
like RSA never predicted readability anywhere in this arc, as a geometric necessity rather
than a puzzle. A shared cross-family geometry; a deviation from it that is causal, rank-2 at
its core, nameless, and switch-like — that is the island arc's final sentence.

## 4. What does not cross: control

The same cleared map that reads at 0.586 was pointed the other way, twice — the second time at
a layer where the target demonstrably steers natively (native gain 0.2151, subspace hosting
verified at pc_cos 0.818). The transferred direction recovered 0.0245 of steering gain —
**11 percent of native control** — while pointing the *right way* on 71 percent of concepts
(`synthesis_minds_addendum.json:write_across_minds`). Not noise: a faint, correctly-aimed
shadow with no behavioral bite. The dissociation survived its own confound and is the cleanest
sentence this program has produced:

**What a mind means crosses; the means to move it does not.**

That sentence was written on a *linear*-map measurement — and §2's whole lesson is that the
linear class leaves signal on the table (gemma read at chance until a nonlinear map pulled it to
55× chance). So the obvious attack on our own law was to give the write side the same capacity
and see if control crosses too. It was run (`b36_result.json`): the paired MLP that opened
reading, at matched-maximal supervision (392 true pairs), at the steer-optimal layer where the
target's own directions steer at native gain 0.173 — the positive control fired, and **every
transfer gate failed** (transfer 0.0504 vs the 0.15 floor, NTE 0.291 vs 0.40, sign 0.6571 vs
0.70). Capacity roughly *doubled* transferred control (0.0245 → 0.0504) and still landed three
times under its bar. The sharpest form of the dissociation now has a controlled statement: on
the same model pair the same upgrade took **reading 0.3429 → 0.8000 and writing 0.0245 →
0.0504** — capacity multiplies both channels by about the same factor, and only reading was ever
close enough for that to matter. Reading was capacity-limited; control is not.

Every deployable built here inherits that asymmetry as a *design rail*, not a limitation: the
conscience is read-only, and `styxx.witness` has no `steer` method at all — not a policy choice
but because the physics refused, now under the strongest attack the program knows how to mount.

## 5. What crosses cleanly: values

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

## 6. The Socratic bound, measured

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

## 7. The bounded lever: frequency

The oscillation arc asked whether resonance — the timing channel — is a route to capability.
The causal phase-clamp answered: real but bounded. The prize for a bank locked to the true
drifting period is +0.17 over static; the best realizable detector captures half of it; the
richer detectors collapse (harmonic-spread +0.002 at eight-fold parameters; nested
theta-gamma +0.009 against its headroom). Receipts: `frequency-resonance/CERT_*.json`.
Resonance is an ingredient, not a doorway; the connection-of-minds results above do not run
through it.

## 8. The sister program

Fathom (the SAE arc: depth constant K, coherence C, early-commitment S) is the circuit-level
complement to these behavioral results — reasoning depth routed, not tried harder; commitment
measurable below the semantic layer, where it cannot be socially engineered. Its numbers live
in its own receipts and Zenodo lineage (10.5281/zenodo.19326174 concept line) and are not
re-certified here.

## 9. The harness: `styxx.sentinel`

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

## 10. What is NOT established (stated before anyone over-reads)

Cross-family content reading above 5× chance; any cross-mind write channel; any claim that
the read≠write asymmetry holds beyond the tested pairs, scales, and the label-free map class;
the rank-2 core beyond the one bridged pair (llama_3b → qwen) or any claim that islands and
their ranks recur across model populations; any *positive* characterization of the barrier
directions (what is established is negative: not nameable, not concept-coherent);
frequency as a capability doorway; and everything in §2–§5 beyond the tested substrates
(≤7B open models plus one frontier commercial model, English, the frozen protocols). Zero
external replications to date — the standing offer in REPLICATIONS.md pays named credit, more
for breaking a result than confirming it.

*Nothing crosses unseen — including what refuses to cross.*
