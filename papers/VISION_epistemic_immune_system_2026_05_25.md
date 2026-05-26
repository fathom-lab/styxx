# VISION · The Epistemic Immune System — how groundbreaking styxx can *honestly* go

**2026-05-25.** The operator asked how groundbreaking we can go. This is the honest
ceiling: not a detector, but the **reference-free, self-verifying answer layer** the
validated divergence work composes into — with every claim tagged *proven / in-flight /
bet*, and the kill-gates that stand between here and "groundbreaking." No adjective is
load-bearing; the kill-gates are.

## The thesis

An agent answering a factual question today has no oracle. styxx's divergence results
let it grow one **from the models themselves** — a reflexive epistemic check that needs
**no reference, no retrieval, no ground truth.** Compose four reflexes:

| reflex | what it does | mechanism | status |
|---|---|---|---|
| **1. Consensus** | return the answer independent vendors converge on | cross-vendor agreement (`council_agreement`) | cross-vendor agreement **validated AUC 0.917**; consensus-as-*answer* **in flight** (T1/T2) |
| **2. Abstention** | refuse when the council fractures (reference-free "I don't know") | agreement < τ | cross-vendor *scatter on fakes* **validated**; calibration **in flight** |
| **3. Confabulation flag** | flag a confident *fabrication* per answer | cross-sample / cross-vendor inconsistency (`semantic_entropy`) | **validated**: TriviaQA 0.785, cross-vendor 0.917; **injection-blind (documented)** |
| **4. Consensus-hallucination probe** | catch the lies hiding in *agreement* (shared misconceptions) | perturbation-fragility | **pre-registered, about to run** — genuinely 50/50 |

Run an answer through all four and you get a **reference-free epistemic readout**:
`{answer, agreement, abstain?, fabrication_risk, consensus_fragility}` — across vendors,
with no external truth. That is the agent equivalent of an immune system: it detects its
own error without an oracle.

## Why this would be groundbreaking — *if* it holds

Nothing in the field gives a **reference-free, cross-vendor, self-verifying** factual
answer that simultaneously (a) answers more reliably than any single model, (b) knows
when to abstain, (c) flags confident fabrication, and (d) probes even the *agreed-upon*
answer for shared misconception. Detectors exist; an integrated reflexive layer with
calibrated abstention and a consensus-hallucination check does not. It changes how a
factual agent is *built* — you stop trusting one model and start trusting a disciplined
council that knows its own limits.

## The honest floor (proven vs bet — do not blur these)

- **Proven (benchmark / cross-vendor):** confabulation is detectable from divergence
  (TriviaQA 0.785; cross-vendor 0.917); inter-model agreement tracks *truth* not
  vendor-consensus (0/8 cross-vendor shared fabrications); cross-vendor *beats*
  same-vendor by breaking correlated confabulation.
- **In-flight (kill-gated, running/queued):** consensus-as-answer beats the best member
  (T1); calibrated abstention (T2); reference-free consensus-hallucination detection
  (D1/D2). **These are bets, not results.**
- **Known limits (shipped honestly):** injection-blind (don't trust on poisoned context);
  cosine clustering has documented failure modes (use a judge); small open models drag
  obscure-fact coverage; everything is n≈20–150, single-run feasibility-grade.
- **Possibly impossible:** catching consensus error reference-free has *no* validated
  method anywhere. The dark-matter swing is the attempt. If it fails, that failure is
  itself the groundbreaking result — **a proven floor of reference-free truth**, the kind
  of boundary a field cites for years.

## The claim we could EARN (and the only honest way to earn it)

> *"The first reference-free epistemic immune system for agents — it answers, abstains,
> and flags its own hallucinations across vendors with no ground truth."*

That sentence is **earned only if all four reflexes pass their kill-gates at scale** —
not before. Today: **2 of 4 reflexes validated, 1 in flight, 1 about to run.** Calling it
"groundbreaking" now would be the exact overclaim styxx is defined against. Calling it
"2-of-4, here are the receipts and the open bets" is the thing that, *when the other two
land*, makes the eventual claim believed.

## The one move

Not a bigger idea — **run the two that are loaded.** Truth engine (T1/T2) is on the GPU;
the dark-matter swing (D1/D2) fires the instant it frees. Two kill-gates stand between
"2 of 4" and a genuinely new agent primitive. The groundbreaking-ness is in those gates,
and we'll know — honestly, either way — within the hour. That is how you go groundbreaking
without becoming the thing you're trying to beat.
