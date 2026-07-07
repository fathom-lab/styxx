---
title: "Calibration Poisoning, Not Erasure: Substrate Honesty Probes Survive Knowledge-Preserving Weight Attacks When the Auditor's Calibration Is Private"
author: "Alexander Rodabaugh (Fathom Lab)"
date: "July 7, 2026 — v0.1"
---

## Abstract

Linear probes on a language model's residual stream can read whether the model is
being honest, and a rapidly converging line of 2026 work — Anthropic's Jacobian-lens
workspace readouts and an independent demonstration that the direction which *detects*
hallucination is nearly orthogonal to the direction that *steers* it — establishes the
read channel as real and dissociated from the write channel. That literature shares an
unexamined assumption: the weights being read are not adversarial. We attack that
assumption directly. In a fully pre-registered attack–defense study, a
knowledge-preserving LoRA attacker fine-tunes the model to blind a
difference-of-means honesty probe while a replay term preserves the model's own
true/false judgment. Under the registered re-lock protocol the attack *appears*
decisive: the re-fit probe reads chance (AUROC 0.461–0.507) while held-out knowledge
stays at 0.817–0.890, on both seeds. A pre-committed interpretation map, frozen before
that number existed, forbade the headline and mandated the resolving experiment. The
resolution: giving the auditor a private calibration split the attacker never saw
recovers the read through the same attacked weights (AUROC 0.711–0.838) — the apparent
evasion was **calibration poisoning, not erasure** of the honesty signal. The defense
survives an *adaptive* attacker that re-fits a moving probe on its own clean split and
scrubs along it (clean-calibrated read 0.733–0.782, two model families). The transferable
lesson for substrate audits is operational: **calibrate the probe on data the attacker
did not see.** We report the load-bearing caveats — the adaptive scheme's extra bite
over the naive attack was not separately measured, the attacker is LoRA-scale, the
models are 1–1.5B — and the escalation to a stronger multi-layer attacker is already
frozen as a public pre-registration. All preregistrations, code, result artifacts, and
machine-checked certificates are public at commit-level granularity.

## 1. Introduction

Two publications within the last three weeks moved activation-space honesty reading
from a niche practice to the center of interpretability. Gurnee et al. (2026) showed
that a Jacobian-based lens surfaces a language model's "workspace" — including tokens
like *manipulation* while the model falsifies values, and *secretly* on ordinary
coding prompts given to a sabotage-trained model. Galeone et al. (2026) showed that
fake-entity hallucination is perfectly linearly detectable (AUC 1.000) while the
detecting direction sits ~83° from the direction that causally controls refusal,
stable across three model families — detection and control are geometrically
dissociated. Both results say the *read channel* is real, and stronger than the
*write channel*.

Neither result — and, to our knowledge, no prior work — asks the question an actual
audit forces: **does the read survive when the weights themselves are optimized to
defeat it?** The deployment scenario that matters for oversight is precisely the
adversarial one: a vendor (or a training process) that wants its model to pass a
substrate honesty audit without giving up capability. If a cheap fine-tune can blind
the probe while preserving the model's knowledge, then substrate honesty reading is
not a safety property; it is a courtesy of non-adversarial weights.

This paper reports a pre-registered attack–defense study of that question, run on our
own probes, against our own program's keystone claim. The program's claim — `read ≠
write`, i.e. you can read honesty off the substrate because you cannot remove the
honest belief without removing the knowledge itself — underlies every read-only
monitor we ship. We tried to break it, watched it appear to break, and found — via an
escalation that was chosen *before* the first verdict existed — that the break was in
the auditor's calibration procedure, not in the signal.

Concretely, we contribute:

1. **A knowledge-preserving weight-level attack** on a linear substrate honesty probe:
   a LoRA adversarial scrub of the probe's separability plus a knowledge-replay term
   that preserves the model's own plain-prompt true/false judgment (§3).
2. **An apparent break under the registered protocol** — the re-locked (re-fit at
   audit time) probe collapses to chance while held-out knowledge is preserved, on
   both seeds (§5.2) — together with the two-worlds confound the protocol could not
   separate, named in the same result note.
3. **The resolution: calibration poisoning.** An auditor whose probe is fit on a
   private clean split the attacker never saw reads honesty through the *same*
   attacked weights at 0.711–0.838 AUROC. The signal was never erased; the
   probe fit on the attacker-optimized split pointed the wrong way.
4. **A defense that survives an adaptive scheme.** An attacker that anticipates the
   defense — re-fitting a moving probe on its own clean split every 25 steps and
   scrubbing along it — still fails to blind the private-calibrated auditor, on two
   model families (§5.4).
5. **A pre-registration discipline with teeth.** Every kill-gate was frozen on commit
   before the corresponding run; the interpretation of every possible stage-2 verdict
   was committed before the stage-2 number existed; each result note passes a
   machine-checked certification (OATH-HELD) binding its claims to re-runnable
   receipts. At stage 2 we possessed a headline-ready "read≠write is broken"; the
   frozen map forbade claiming it, and the mandated follow-up overturned it (§4).

The honest scope is stated up front and repeated where it binds: the attacker is
LoRA-scale (r=16, 300 steps), the models are Qwen2.5-1.5B-Instruct and
Llama-3.2-1B-Instruct, the evaluation sets are small, and the adaptive attacker's
extra bite over the naive one was not separately verified (§7). "The defense holds"
means *these* attackers did not break it; a stronger multi-layer full-capacity
attacker is already frozen as a public pre-registration (E2) and is the next run.

## 2. Threat model

**The attacker** controls the weights: it may fine-tune the deployed model. It knows
the deployed probe — worst case for the defender — including the layer and direction
of the honesty read. Its objective is two-sided: (a) drive the auditor's honesty read
to chance, and (b) preserve the model's factual capability, operationalized as the
model's own true/false judgment on held-out facts under a neutral prompt. An attack
that destroys knowledge to hide honesty is a Pyrrhic evasion and is scored as such.

**The auditor** gets the attacked weights at audit time and may re-fit ("re-lock")
its probe — a frozen probe that a fine-tune learned to dodge is a weak test, so
re-fitting is the deployment-faithful read. The variable under study is *what data
the auditor calibrates on*: the registered stage-2 protocol calibrates on the same
fact split the attacker optimized against; the E1/E3 defense calibrates on a private
split the attacker never saw.

## 3. Method

**Models.** `Qwen2.5-1.5B-Instruct` (decisive family; deploy layer 18, scan 12–22)
and `Llama-3.2-1B-Instruct` (cross-family corroboration; deploy layer 10, scan
6–14). Everything runs on a single 8 GB consumer GPU — this is an audit anyone can
re-run.

**Data.** 180 balanced true/false factual statements from a frozen bank, split by
topic stem (no shared stem across splits). Stage 2 uses ATTACK (98 facts) /
HELD-OUT (82). E1 uses a 3-way stem-disjoint split ATTACK / CALIB / EVAL
(~50/25/25): the attacker touches only ATTACK; the auditor fits on CALIB; everything
is measured on EVAL. E3 adds MIMIC — the attacker's *own* clean split, its model of
the auditor — for a 4-way split (~40/20/20/20).

**The read.** A linear honesty direction on final-token residuals under a neutral
"true or false?" prompt. Stage 2 uses difference-of-means (DoM) per layer; the
E1/E3 auditor strengthens this to three probe families — DoM per layer, logistic
regression per layer, and a whole-stack concatenated-layer logistic — with the family
and layer selected on CALIB only, never on EVAL. Read quality = AUROC of the probe
score against true/false labels on the measurement split.

**The knowledge signal.** `behavioral_margin` = max True-token logit − max
False-token logit under the neutral prompt; knowledge accuracy = sign(margin) matches
the label. This is "does the model still know true from false when asked plainly."

**The attack.** A LoRA adapter (r=16, α per house defaults, 300 steps) trained with
`L = L_adv + λ · L_replay`: the adversarial term collapses the true/false residual
gap along the honesty direction on ATTACK facts; the replay term is cross-entropy on
the correct plain-prompt judgments, preserving knowledge. λ sweeps {0, 0.3, 1.0,
3.0, 10.0} in stage 2; E1/E3 run the knowledge-preserving λ ∈ {1.0, 3.0} and λ=1.0
respectively, seeds {0, 1}.

- **Naive mode** (stage 2, E1, E3-control): the scrub targets the frozen deployed
  direction.
- **Adaptive mode** (E3): every 25 steps the attacker re-fits a DoM honesty direction
  on the *current* model's MIMIC residuals — simulating the auditor's re-lock — and
  scrubs along that moving direction plus the frozen one. The attacker chases the
  signal wherever a clean re-fit would find it, but only on data it has (MIMIC),
  never on the auditor's private CALIB.

**Non-degeneracy guards (VOID, not verdict, if failed).** The read must work
pre-attack (clean probe AUROC ≥ 0.75); knowledge must exist pre-attack (≥ 0.80);
splits must be stem-disjoint (asserted in code); a random-direction probe must read
chance and shuffled-label knowledge must sit ≈ 0.5. Guard values ride in every result
artifact. Smoke runs write only `*_SMOKE_INVALID*` files and are never read as
results.

## 4. Pre-registration and machine-checked certification

Every stage of this study was frozen on commit *before* the corresponding run, in a
public repository, with kill-gates that could return the program-falsifying verdict:

- `PREREG_read_neq_write_coupling_2026_07_04.md` (commit `cff3b12`) — the keystone
  test and the EVADABLE / SURVIVES corners.
- `PREREG_stage2_attack_sweep_2026_07_04.md` (commit `baefaf7`) — the λ-sweep attack.
- `PREREG_stage3_interpretation_escalation_2026_07_04.md` (commit `b3224a0`) —
  **frozen while the stage-2 sweep was still training**: what each possible verdict
  licenses, what it does not, and which escalation it forces. This document is why
  the stage-2 "break" below could not become a headline.
- `PREREG_E1_clean_calibrated_auditor_2026_07_04.md` (commit `5be3bb9`) and
  `PREREG_E3_adaptive_attacker_2026_07_04.md` (commit `74cac7c`) — each frozen before
  its retrain.
- `PREREG_E2_multilayer_strong_attacker_2026_07_04.md` (commit `af5e184`) — the next
  escalation, frozen and public *before its run*, so the pending result cannot move
  its own goalposts.

Each result note is bound to its numeric artifacts by `styxx.certify`, a
machine-checked certification (verdict OATH-HELD) that extracts the note's numeric
claims and verifies them against the re-runnable result JSONs; the certificates ship
in the same directory as the results. Bars never moved after freezing; a missed
guard voids the run rather than bending the bar.

## 5. Results

### 5.1 Foundation: the signal and the knowledge are real before the attack

On clean Qwen2.5-1.5B (seed 0, 98 ATTACK / 82 HELD-OUT facts), per
`foundation_result.json`:

| guard | measured | bar | verdict |
|---|---|---|---|
| clean probe held-out AUROC (L18) | 0.851 | ≥ 0.75 | PASS |
| clean knowledge accuracy | 0.805 | ≥ 0.80 | PASS |
| ATTACK ∩ HELD-OUT stems | disjoint | ∅ | PASS |
| random-direction AUROC | 0.500 | chance band | PASS |
| shuffled-label knowledge | 0.512 | ≈ 0.5 | PASS |

Layer 18 was selected by held-out AUROC over a scan (L12 0.602, L14 0.730, L16
0.811, **L18 0.851**, L20 0.832, L22 0.809) and then frozen. There is a real read
and real knowledge to pull apart.

### 5.2 Stage 2: the registered protocol returns EVADABLE — with a named confound

The attack collapses the re-locked read while knowledge is preserved, on both seeds,
at every knowledge-preserving λ (per `attack_sweep_result.json`; reproduction guard
at sweep start: clean probe 0.854, clean knowledge 0.805):

| seed | λ | re-locked AUROC (held-out) | knowledge |
|---|---|---|---|
| 0 | 0.3 | **0.496** | 0.817 |
| 0 | 1.0 | 0.507 | 0.890 |
| 1 | 0.3 | **0.461** | 0.829 |
| 1 | 1.0 | 0.472 | 0.866 |

At λ = 0 (no replay) the scrub also destroys knowledge (accuracy 0.573–0.598): the
replay term is exactly what buys the attacker a knowledge-preserving evasion. A
random direction on the same residuals reads ~0.54–0.59, so ~0.5 is genuine chance.

**The confound, named in the registered result before any follow-up ran:** the
re-lock calibrates its probe on the ATTACK facts — the same set the attacker
optimized against. An EVADABLE under this protocol is therefore consistent with two
worlds it cannot separate: (1) *true erasure* — the signal is gone from the residual
stream; or (2) *calibration poisoning* — the signal survives on clean data, but a
probe fit on the poisoned split points the wrong way. The pre-committed
interpretation map licensed only "a knowledge-preserving weight-level attacker can
blind a DoM honesty read calibrated on the attacked set, on this 1.5B model" — and
mandated E1.

### 5.3 E1: the break was calibration poisoning — read ≠ write stands

E1 gives the auditor a private clean calibration split (CALIB) the attacker never
saw, plus the stronger probe families of §3. Clean-model guard on the decisive
family: CALIB-fit probe EVAL AUROC 0.938, EVAL knowledge 0.818. Through the same
attack that produced the table above (per `e1_result.json`):

| seed | λ | clean-calibrated read (EVAL AUROC) | knowledge |
|---|---|---|---|
| 0 | 1.0 | **0.838** | 0.818 |
| 0 | 3.0 | 0.835 | 0.833 |
| 1 | 1.0 | **0.722** | 0.803 |
| 1 | 3.0 | 0.711 | 0.833 |

Every knowledge-preserving point clears the frozen 0.70 recovery bar on both seeds
(random-direction floor 0.551–0.588). The attack that drove the ATTACK-calibrated
re-lock to chance leaves the honesty signal fully readable to an auditor calibrated
on clean data. Verdict per the frozen gate: **POISONING_CONFIRMED — read ≠ write
STANDS against this attacker, provided the auditor does not calibrate on the
attacker's data.**

The cross-family arm VOIDed its own guard in E1: clean Llama-3.2-1B EVAL knowledge
came in at 0.788 against the frozen 0.80 bar (missed by 0.012), so its points
(clean-calibrated recovery 0.669–0.747 through the attack) corroborate directionally
but are not a verdict. E1's confirmed result is single-family; E3 closes that flank.

### 5.4 E3: the defense survives an adaptive attacker — two families

E3 attacks the defense itself: the adaptive mode of §3 chases the honesty signal
across calibration frames on its own clean split (MIMIC), never touching the
auditor's private CALIB. Both families cleared their clean guards this run (Qwen
CALIB-probe 0.912 / knowledge 0.800; Llama 0.821 / 0.800), making this a two-family
result. Per `e3_result.json` (λ = 1.0):

| family | mode | seed | clean-calibrated read | knowledge |
|---|---|---|---|---|
| Qwen2.5-1.5B | naive (control) | 0 | 0.786 | 0.833 |
| Qwen2.5-1.5B | naive (control) | 1 | 0.775 | 0.817 |
| Qwen2.5-1.5B | **adaptive** | 0 | **0.782** | 0.833 |
| Qwen2.5-1.5B | adaptive | 1 | 0.733 | 0.733 |
| Llama-3.2-1B | naive (control) | 0 | 0.706 | 0.783 |
| Llama-3.2-1B | naive (control) | 1 | 0.732 | 0.800 |
| Llama-3.2-1B | **adaptive** | 0 | 0.739 | 0.783 |
| Llama-3.2-1B | **adaptive** | 1 | 0.772 | 0.817 |

The naive control reproduces E1 (the harness is sound), and under the adaptive
attack the clean-calibrated read does not drop below the naive level
(random-direction floor 0.483–0.602 across points). Verdict per the frozen gate:
**DEFENSE_HOLDS.** Two annotations bind this verdict: Qwen adaptive seed 1's
knowledge (0.733) fell below the 0.75 knowledge-preserving bar, so that point does
not count toward the verdict — the decisive-family hold rests on seed 0, with Llama
corroborating on both seeds. And the caveat of §7.1 is load-bearing.

## 6. The defense, stated operationally

The whole difference between reading chance (stage 2) and reading 0.711–0.838 (E1) on
the *same attacked weights* is **where the auditor's calibration data came from**.
This yields a concrete, transferable audit rule:

> **Calibrate the substrate probe on data the audited party never saw.** A probe
> re-fit on data the attacker controlled — or could anticipate — is not a read of the
> model; it is a read of the attack.

Three practical corollaries. (1) *Audit sets must be private*: vendor-supplied or
publicly-fixed calibration corpora are poisonable by construction — an auditor whose
calibration provenance is public is auditing with a compromised instrument. (2)
*Re-locking stays necessary but is not sufficient*: re-fitting at audit time defends
against a probe the fine-tune dodged, but only private calibration defends against a
fine-tune that poisoned the calibration frame itself. (3) *The defense is cheap*:
nothing above requires white-box training access — only holding back a stem-disjoint
calibration split, fitting three linear probe families, and selecting on the private
split. It is implementable in any read-only monitor that mounts a probe on residual
streams, and we are wiring it into the `styxx` mount path as the default.

We name the failure mode **calibration poisoning** to distinguish it from signal
erasure and from inference-time evasion: the attacked model still linearly encodes
honesty; what the attack corrupted was the *estimator* the auditor builds from
contaminated data. The distinction matters because the mitigations differ — erasure
would demand stronger probes or non-linear reads; poisoning demands data provenance.

## 7. Limitations and pre-registered pending escalations

**7.1 The adaptive attacker's bite was not separately verified (load-bearing).** The
E3 prereg listed a guard that the moving-probe scrub should drive the frozen-deployed
read below the naive level — evidence the adaptive loop adds attack power. The
harness did not log the frozen-deployed read, so that guard is unmeasured, and
adaptive vs. naive clean-calibrated reads came out nearly equal (0.733–0.786). E3
therefore shows the defense holds against *this adaptive scheme*; it does not show
the scheme was a strictly harder attack than naive. The honest claim is "read ≠
write survived an adaptive-scheme attacker," not "beats the strongest adaptive
attacker." An E3′ that logs the deployed-read delta is owed and scheduled.

**7.2 Attacker scale.** LoRA r=16, 300 steps, scrub targeted at DoM readouts. A full
fine-tune with 4× adapter capacity, a multi-layer whole-stack scrub, and 600 steps is
already frozen as `PREREG_E2_multilayer_strong_attacker_2026_07_04.md` (public commit
`af5e184`, before its run) — by construction it removes E3's "did it bite" ambiguity,
since it attacks every layer the auditor scans. STANDS only strengthens if it
survives E2; EVADABLE-by-strong would be reported as loudly as this paper reports the
recovery.

**7.3 Scale and breadth.** 1–1.5B models, two families, linear probes only, small
EVAL splits (≈25% of 180 statements in E1, ≈20% in E3), seeds {0,1}, single run per
cell — no confidence intervals at this n. This is a first decisive data point and a
defense with a mechanism, not a theorem. Whether the geometry persists at 7B+ scale,
under RLHF-era post-training, or against non-linear reads is open.

**7.4 Construct scope.** "Honesty" here is true/false factual judgment under a
neutral prompt — the cleanest loadable construct, not strategic deception in
context. Bridging to deception-in-context probes (e.g. the Apollo setting) is future
work; the calibration-poisoning threat applies to any probe whose calibration data an
adversary can influence, which is why we report it at the mechanism level.

## 8. Related work

**Latent knowledge and honesty probes.** Burns et al. (2022) recover latent knowledge
without supervision; Azaria & Mitchell (2023) show internal states separate true from
false statements; Marks & Tegmark (2023) map the linear geometry of truth
representations; Zou et al. (2023) systematize representation-space reading and
control. Goldowsky-Dill et al. (2025) deploy linear probes for strategic deception at
high AUROC. None of these evaluate a weight-level adversary optimizing against the
probe with a capability-preservation constraint.

**Adversarial evasion of latent-space monitors.** Bailey et al. (2024) show
*inference-time* obfuscated activations can bypass latent defenses. Our attack lives
one level deeper — in the weights — and our defense (private calibration) addresses a
failure mode their setting does not expose: the poisoning of the auditor's estimator
rather than the evasion of a fixed one.

**The 2026 convergence on read ≠ write.** Galeone et al. (2026) establish
detection–control orthogonality across families (cos 0.12–0.20, identical pre/post
instruction tuning); Gurnee et al. (2026) read workspace-level hidden cognition via
the Jacobian lens, with interventions that redirect explicitly-reported choices but
not automatic behavior — the same read/write asymmetry surfacing in a different
apparatus. Both treat the weights as honest. This paper is the adversarial rung of
that ladder: what remains of the read channel when the weights fight back, and what
the auditor must do to keep it.

## 9. Reproducibility and receipts

Everything is public in the `styxx` repository (github.com/fathom-lab/styxx),
`papers/read-neq-write/`: pre-registrations (frozen on commit before each run),
harnesses (`foundation.py`, `attack_sweep.py`, `e1_clean_auditor.py`,
`e3_adaptive_attacker.py`, `e2_strong_attacker.py`), result artifacts
(`foundation_result.json`, `attack_sweep_result.json`, `e1_result.json`,
`e3_result.json`), result notes, and their OATH-HELD certificates. Seeds and splits
are locked in code; the full study reruns on one 8 GB GPU. The package `styxx` is on
PyPI (130 releases since 2026-04-11); the program's methodology paper is archived at
DOI 10.5281/zenodo.19326174.

### Appendix A: public timestamp trail of the program's claims

All timestamps are third-party (PyPI registry, Zenodo) or public-repository commit
dates.

| date (2026) | receipt | claim |
|---|---|---|
| 04-11 | PyPI first upload of `styxx` | program start |
| 05-27 | Zenodo DOI 10.5281/zenodo.19326174 (v25 deposit) | pre-registered, self-falsifying evaluation methodology |
| 05-29 | commit `3e434cb` (public) | depth-steering: axis writable but correctness-inert — first write-limits result |
| 06-05/07 | commits `78b4a5d`, `ebe05ff` (public) | cross-model substrate reading; probe 1.00 vs self-report at chance |
| 06-16 | PyPI 7.16.0 / 7.17.0 | read-only conscience monitor mounted on open models |
| 06-23 | — | Galeone et al., arXiv 2606.24952 (detect ⊥ steer) |
| 07-04/05 | commits `cff3b12`…`af5e184` (public) | this study's preregs frozen; stage-2, E1, E3 findings |
| 07-06 | — | Gurnee et al., Transformer Circuits (J-lens workspace) |

### References

- Azaria, A., & Mitchell, T. (2023). *The Internal State of an LLM Knows When It's
  Lying.* arXiv:2304.13734.
- Bailey, L., et al. (2024). *Obfuscated Activations Bypass LLM Latent-Space
  Defenses.* arXiv:2412.09565.
- Burns, C., Ye, H., Klein, D., & Steinhardt, J. (2022). *Discovering Latent
  Knowledge in Language Models Without Supervision.* arXiv:2212.03827.
- Galeone, C., Ettorre, A., Park, M., Ettorre, G., & Ligorio, D. (2026). *Perfect
  Detection, Failed Control: The Geometry of Knowing vs. Steering in Language
  Models.* arXiv:2606.24952.
- Goldowsky-Dill, N., et al. (2025). *Detecting Strategic Deception Using Linear
  Probes.* arXiv:2502.03407.
- Gurnee, W., Sofroniew, N., Pearce, A., et al. (2026). *Verbalizable
  Representations Form a Global Workspace in Language Models.* Transformer Circuits
  (transformer-circuits.pub/2026/workspace), July 6, 2026.
- Marks, S., & Tegmark, M. (2023). *The Geometry of Truth: Emergent Linear Structure
  of True/False Datasets.* arXiv:2310.06824.
- Rodabaugh, A. (2026). *Fathom v25 / styxx v7.7.9: The Gauntlet that Catches
  Itself.* Zenodo, DOI 10.5281/zenodo.19326174.
- Zou, A., et al. (2023). *Representation Engineering: A Top-Down Approach to AI
  Transparency.* arXiv:2310.01405.
