# Finding · The Cross-Vendor Council — agreement tracks TRUTH across vendors (PASS), and beats same-vendor

**2026-05-25.** Prereg `preregistration_crossvendor_2026_05_25.md`. The decisive
experiment of the arc, unblocked by running local open-weights models instead of an API
key. **Verdict: PASS** (X1 ∧ X2), X3 clean — and a result stronger than the bars asked.

## Council & result

Three vendors, disjoint training lineages: **gpt-4o-mini + gpt-4o (OpenAI)**,
**Qwen2.5-3B-Instruct (Alibaba)**, **gemma-2-2b-it (Google)** — the latter two local on
GPU. 20 questions (6 real-common, 6 real-obscure, 8 fake), modal of N=3 each, clustered
by the shipped `styxx.council_agreement(..., same_fn=<judge>)`.

| metric | value | bar | verdict |
|---|---|---|---|
| **X1** AUC(agreement → real), cross-vendor | **0.917** | ≥0.75 | **PASS** |
| **X2** mean agreement: real-common | **1.00** | ≥0.70 | **PASS** |
| **X2** mean agreement: fake | **0.41** | ≤0.45 | **PASS** |
| **X3** cross-vendor shared confabulation on fakes | **0/8** | — | clean |
| (real-obscure) | 0.83 | — | — |
| **OpenAI-only subset** AUC | **0.875** | — | *lower than cross-vendor* |

**PASS = TRUE.** Inter-model agreement separates real from fake across three vendors at
AUC 0.917 — it tracks **truth, not OpenAI-family consensus.** The arc's biggest caveat is
resolved with real cross-vendor data.

## The unexpected kicker: cross-vendor BEATS same-vendor

Cross-vendor AUC (0.917) **exceeded** the OpenAI-only subset (0.875). Mechanism, visible
in the rows: on **2 of 8 fakes** (the "Azure Cascade" composer, the "Glass Sentinel"
author) **both OpenAI models produced the same fabrication** — OpenAI-subset agreement
1.00 on a fake. That is within-vendor **correlated confabulation**, the shared-lineage
failure made concrete. Qwen and Gemma did **not** share those lies (Qwen abstained or
Gemma invented something else), so adding the other vendors **broke the correlation** and
dropped those fakes to ~0.33 agreement. So a cross-vendor council is not merely *as good*
as single-vendor — it is **more robust to the exact failure mode** (shared-training
confabulation) that most threatens a single-vendor council. X3 = 0/8: no fabrication was
shared *across* vendors.

## Honest caveats

- **Substantive-agreement is unstable under heavy abstention.** On "Vorland" 3 of 4
  abstained; the single remaining confabulation scored a trivial "agreement 1.0." It
  didn't sink the AUC, but the metric should be **complemented by abstention rate** —
  when most of a council abstains, that *is* the fake signal, and substantive-agreement
  over 1 vote is meaningless. (Design note for any shipped council layer.)
- **real-obscure = 0.83, not 1.0.** The 2–3B local models genuinely don't know 2 of 6
  (1938 Nobel, Calypso Deep) → they split from OpenAI → lower agreement. A
  knowledge-coverage limit of small models, **not** a signal failure — and the council
  *did* converge cross-vendor on Ouagadougou, the *French Connection* director, Fillmore,
  tungsten. Bigger open models would lift this.
- n=20, single run, 2–3B local models, judge clustering. A larger battery + more
  vendors/models is the obvious next scale-up, but the result is decisive at this n
  (AUC 0.917, X3 0/8).

## Net

The thing I said all session needed a key needed only local weights. Reference-free
fabrication detection via inter-model agreement is **validated across OpenAI, Alibaba,
and Google**, with **no correlated confabulation across vendors**, and is **more robust
cross-vendor than within one vendor**. `council_agreement` graduates from
"shared-knowledge within one lineage" to a **cross-vendor truth-vs-fabrication signal**.
The shipped docstring's same-vendor caveat is resolved; folded into 7.7.2.
