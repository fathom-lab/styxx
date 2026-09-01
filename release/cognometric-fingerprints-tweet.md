> **⚠ Ceiling erratum (2026-09-01).** The near-perfect AUC figures quoted below -- including the
> headline triple **0.998** (HaluEval-QA hallucination), **0.976** (XSTest refusal, out-of-family)
> and **0.943** (BFCL v3 tool-call drift) -- are **register detectors at a construct ceiling**, and
> this document quotes them without saying so. Recorded here rather than corrected in place:
>
> - **What the figures are.** Cross-validated separations, on the benchmarks named, between outputs
>   whose *register* differs: confabulated vs grounded, refusing vs complying, drifted vs on-task.
>   The numbers are not withdrawn and are not edited below.
> - **What they are not.** They are not a measure of honesty under adversarial pressure. The same
>   construct scores **0.498** -- chance -- on text-only deception detection, against **0.966** for
>   sampling-divergence grounding of factual self-claims. A near-perfect AUC on these benchmarks
>   does not license the claim that the instrument catches a model that is trying not to be caught.
> - **Where the bound is published.** `papers/THESIS_the_honesty_standard_2026_05_31.md`, which
>   names "a construct ceiling the program had hit four times", and the scope erratum at the top of
>   `papers/every-mind-leaves-vitals.md`, which applies the same correction to this lab's own
>   strongest claim.
> - **Status of this document.** Preserved as written. Nothing below is edited, softened or
>   renumbered. Whether this document was sent, published or circulated is not decided here, and
>   this erratum is correct either way.
>
> *We measure behaviour and representation, never minds. The boundary is the product.*

---

# Tweet / X — cognometric fingerprints launch

**Context.** Today we shipped:
- v6.1.0 (drift AUC 0.916 → 0.943, arg_swap fix) on PyPI
- two phase-transition papers (in-sample + cross-model on 5 substrates)
- "Calibration Fingerprints" — a proposed measurement standard
- 11-fingerprint atlas across 3 instruments × 5 substrates
- manifesto updated + redeployed at fathom.darkflobi.com/manifesto

The tweet below leads with the cross-model finding (visceral, surprising, specific)
and pivots to the methodology proposal (discipline-building). No fluff.

---

## Primary (single tweet)

```
your llm safety detector reports AUC 0.94. great.

we ran the same detector on 5 model families.
the critical feature that flips detection SHIFTS per substrate.
on 2 substrates, adding features actively REGRESSES AUC.

"AUC 0.94" is a detector on one substrate. nothing more.

fathom.darkflobi.com/manifesto
```

(~265 chars, fits 280.)

---

## Thread (3 tweets, for the full story)

**1/3 — hook**

```
your llm safety detector reports AUC 0.94.
great.

now run it on 5 different model families.

we did. AUC ranged 0.60 to 0.97.
the critical feature that flips detection SHIFTS per substrate.

"AUC 0.94" isn't a detector. it's a detector on one substrate. 🧵
```

**2/3 — the evidence**

```
same 18-feature refusal detector, trained once on Llama-1B.
held-out on XSTest v2 across 5 model families:

gpt-4        K=1  starts_with_sorry     AUC 0.97
llama2-new   K=2  refusal_density       AUC 0.88
llama2-orig  K=2  refusal_density       AUC 0.77   (K=3 disclaimer_density REGRESSED to 0.69)
mistral-gd   K=5  gradual               AUC 0.77
mistral-in   —    no transition         AUC 0.60

"more features is better" dies under distribution shift.
```

**3/3 — the proposal**

```
proposing a standard: calibration fingerprints.

every calibrated detector publishes:
  critical K, critical feature, delta AUC at K,
  per-substrate K variance, negative-lift features

one ablation run per detector. atlas v0 today:
11 fingerprints × 3 instruments × 5 substrates.

fathom.darkflobi.com/manifesto
```

---

## Alternate openers (if primary doesn't land)

### "The visceral phase-transition"

```
we took our 22-feature tool-call drift detector.
removed 21 features. kept ONE.

arg_drop detection AUC went 0.501 → 0.998 at K=2.
one feature (arg_count_zscore) crosses the threshold.

detection phase-transitions. inverse of emergent capabilities.

same thing on refusal: K=1, starts_with_sorry.
0.50 → 0.97.

fathom.darkflobi.com/manifesto
```

### "The bold claim"

```
cognometry is not more benchmarks.
cognometry is a measurement standard.

today: shipping 3 calibrated detectors + the first atlas of calibration fingerprints for LLM safety detection.

11 fingerprints × 3 instruments × 5 substrates.

the field has been reporting AUC.
we're proposing it report fingerprints.

fathom.darkflobi.com/manifesto
```

### "The terse darkflobi"

```
AUC is a number.
cognometric fingerprints are a structure.

K*, critical feature, delta_AUC at K*,
substrate_K_var, negative_lift.

atlas v0 live — 11 fingerprints,
3 instruments, 5 substrates.

discipline-building, not benchmark-farming.

fathom.darkflobi.com/manifesto
```

---

## Notes for posting

- Drop from @fathom_lab handle.
- Pin the primary tweet to the profile for the launch week.
- First reply quote-tweets the v6.1.0 release thread (pypi link).
- Second reply quote-tweets the drift phase-transition original (ties the narrative together).
- No emojis except 🧵 in thread pin. Keep the noir aesthetic.
- Tag @natolambert if thread goes well (XSTest v2 dataset owner).
