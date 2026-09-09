# PLAN — the frontier target: the claim about the search, not the object the search found

Fathom Lab · 2026-09-08 · **A plan, not a result.** Decided from a six-lens survey of machine-produced
mathematical evidence run the same day; one of the six lenses was a hostile prior-art reviewer whose report
is answered below rather than summarised. This document **makes no mathematical claim of its own.** Every
factual statement about the literature carries the confidence the surveying lens assigned it, marked
inline as *verified-this-session*, *recalled-high-confidence* or *recalled-uncertain*; **no claim in this
document was verified against a primary source in this session** unless a lens says it was, and where a
lens verified something this document says which lens and what it checked. **It is not sworn.** It is a
successor to nothing; the plan of record (`papers/PLAN_the_next_level_2026_09_02.md`) is not edited by it,
and §14.0 of `SPEC_v8_v0.2_draft.md` still gates whether v8 runs before or after that plan's legs 2 and 4.

---

## The question

The survey asked one thing: where, in mathematics, is a result disbelieved or unchecked because the
*evidence* was produced by a machine, rather than because the *mathematics* is unfinished. That question
has a sharp form, and the sharp form is what makes it answerable: a claim has an evidence bottleneck only
when there is no object a stranger can check more cheaply than redoing the work, and no theorem that makes
the producer irrelevant. Almost everywhere the survey looked, one of those two escapes exists — a cap set
is a loop, a factor triple is one tensor identity, a DRAT proof is a certificate, a Lean term is a kernel
call — and where an escape exists, provenance is bookkeeping and the honest answer is to say so.

## What the survey found

Six lenses (`fluid-blowup`, `machine-discovered-constructions`, `large-scale-proofs`, `rigor-interface`,
`prior-art-adversary`, `this-box`) returned thirty-one candidate rows between them. The table keeps the
serious ones and every rejection, because the rejections are what make the choice credible: a survey that
found a fit everywhere would have found nothing.

| candidate | is the bottleneck really *evidence*? | checkable on one consumer GPU? | lens confidence |
|---|---|---|---|
| **Method claims about machine search and machine proving** — "our system found X and the baseline did not", "model M proves p% of benchmark B" | **Yes.** The object is checked in milliseconds; the claim is about a distribution over runs, and a distribution has no witness. Four lenses reached this independently. | Yes — `this-box` measured a ⟨5,5,5⟩ rank-95 scheme verifying in 2.23 ms from a 14,839-byte file on this machine, and gemma-2-2b at ≈4.9 GiB bf16 against 8.0 GiB visible on the RTX 4070 Laptop, torch 2.5.1+cu121 | verified-this-session (`machine-discovered-constructions`, `large-scale-proofs`, `prior-art-adversary`, `this-box`) |
| **Auditable exhaustion** — "we searched region X and found nothing" (Collatz to a bound, Goldbach to a bound, all graphs on ≤ n vertices) | **Yes**, and uniquely: a conjunction over astronomically many instances admits no compressing witness at all, so reproduction cost equals original cost and nobody pays it. But no addressable per-segment unit is published either, so nobody can even spot-check. | Yes — OEIS A000088 gives 12,005,168 graphs on 10 vertices and 1,018,997,864 on 11 (verified-this-session, `this-box`); `nauty`'s `geng` has `res/mod` splitting, and its manual states the `-x`/`-X` parameters must be identical across parts or the parts are not the same parts | verified-this-session (`this-box`, `large-scale-proofs`); Collatz to 2^71 (Bařina, J. Supercomputing 2025) verified-this-session by `large-scale-proofs`, Goldbach to 4e18 recalled-high-confidence |
| **"The residual is at the hardware round-off floor"** — the warrant given for taking machine-found blowup profiles into computer-assisted proof | **Narrowly yes.** arXiv:2509.14185 states the residual reaches accuracy "constrained only by the round-off errors of the GPU hardware"; no reduction order, tiling or determinism statement accompanies it. A floor asserted without a measured floor is the failure this lab already named in its own domain. | Yes, and needs no training: freeze published weights, vary tiling and reduction order, measure the spread of the reported residual | verified-this-session (`fluid-blowup`) |
| **Unformalized rigorous-numerics computer-assisted proofs** | **Half.** Zheng, arXiv:2608.13067 (13 Aug 2026), an independent version-pinned source-level audit of a certificate published in Comm. Math. Phys. 2025, reports 11 proof-affecting defects with reproducible exact counterexamples; the theorem survives, the certificate does not establish it. But the 11 were logic errors found by reading source — a signed, pinned, logged archive would have been signed, pinned, logged and wrong. | CPU work; the GPU is idle | verified-this-session (`rigor-interface`, `prior-art-adversary`) |
| **Tolerance-relative geometric optima** — circle packings, kissing configurations, where the reported real number exists only relative to a feasibility tolerance | Partly; two numbers computed at different tolerances are not comparable and the field resolves it by eye | Yes, CPU | recalled-uncertain (`machine-discovered-constructions` explicitly declined to repeat the specific tolerance figures without reading verifier source) |
| **Machine-computed mathematical databases** (LMFDB-shaped) | Weakly. LMFDB already publishes per-object reliability, rigor and completeness pages (verified-this-session, `prior-art-adversary`); the gap is a per-entry binding to bytes, not an absence of trust infrastructure | CPU | verified-this-session for the LMFDB pages; recalled-uncertain on per-section coverage |
| — **rejected** — | | | |
| Kernel-checked formal proofs (Lean/Rocq/Isabelle) | **No.** Checking is a total deterministic function of (kernel, environment, term); you do not trust the signer, you run the checker. The live residual is the *statement* problem — a formalization that type-checks and does not mean the theorem — which no signature touches | — | verified-this-session (`rigor-interface`, `prior-art-adversary`) |
| SAT/DRAT exhaustion (Pythagorean triples, Schur 5, Keller) | **No, and solved better.** Boolean Pythagorean triples: ~200 TB DRAT, a 68 GB compressed certificate anyone can expand and check; Schur number five = 160 with LRAT certified by a checker verified in ACL2; `cake_lpr` verified in CakeML down to its machine code (verified-this-session). A verified checker needs no issuer, no key, no log | No — four orders of magnitude off this box | verified-this-session (`large-scale-proofs`, `this-box`, `prior-art-adversary`) |
| 3D Navier–Stokes global regularity | **No.** Tao's averaged-equation construction (JAMS 29, 2016) is a statement about which *methods* can work; better-attested numerics move it by nothing | — | verified-this-session (`fluid-blowup`) |
| Direct DNS blowup evidence (Luo–Hou scaling exponents) | **No.** No finite-resolution simulation separates finite-time blowup from very rapid growth; a proof closed it, not a receipt | No — cluster scale | verified-this-session (`fluid-blowup`) |
| PINN-found self-similar profiles, as objects | **No, and attesting them is a category error.** The validation argument is a-posteriori: an approximate profile plus a contraction argument proving an exact solution exists near it. The proof does not trust the profile, so the irreproducible stage is the stage whose irreproducibility costs nothing. arXiv:2601.19818's stated property is that verification is independent of the generation method | — | verified-this-session (`rigor-interface`, `fluid-blowup`) |
| Ramsey lower-bound constructions | **No.** The graph is the proof; checking a colouring of K_42 is fast and embarrassingly parallel | — | verified-this-session (`machine-discovered-constructions`) |
| Four colour theorem, Kepler conjecture | **No, and historically closed** by independent reimplementation and then formalization | — | recalled-high-confidence (`large-scale-proofs`) |
| Lower bounds and asymptotic constants (ω, cap-set upper bounds, rank of ⟨3,3,3⟩) | **No.** There is no machine-produced artifact to distrust, because there is no artifact | — | recalled-high-confidence (`this-box`) |

Two lenses disagreed and the disagreement matters, so it is recorded and resolved rather than averaged.
`fluid-blowup` held that the machine-found blowup line has a real evidence residue (the λ-versus-instability-order
formula and the round-off-floor sentence, neither covered by any enclosure, both used as science now).
`rigor-interface` held that touching that pipeline at all is a category error and recommended refusing it
publicly, because a-posteriori validation renders the training stage trustless. **This plan takes
`rigor-interface` on the profiles and `fluid-blowup` on the arithmetic**: the profile needs no attestation
because a contraction argument will or will not close around it, but "the residual is at the round-off
floor of this hardware" is a claim about a machine's own arithmetic, made by that machine, and no
enclosure covers it. That residue is real and it is small, which is why it is a runner-up and not the target.

Second disagreement: `large-scale-proofs` and `this-box` both rank auditable exhaustion very highly;
`prior-art-adversary` implicitly demotes it, and `this-box` — which is the lens that costed it — says in
its own words that the layer that actually helps there is a pinned shard function, per-shard digests and
random audit, which is roughly a text file, git and discipline. **This plan takes that self-assessment
at face value.** Exhaustion is the strongest runner-up and it is not the target, because the mechanism it
needs is not the mechanism v8 spent its complexity on.

## The hostile report, and the answer to it

The `prior-art-adversary` lens was asked to kill the program. Its report is restated here in the form
that does the most damage, and several of its arguments are conceded because they cannot be answered.

**H1 — Layer inversion. The log is commodity.** RFC 6962 restated: Certificate Transparency, Sigstore
and Rekor, the Go checksum database, PyPI attestations under PEP 740, and IETF SCITT — whose architecture
is now RFC 9943 — already standardise "register a signed statement about an artifact, get a receipt"
(verified-this-session). Every transparency log that worked, worked because a client people already ran
verified proofs automatically and refused on failure. Styxx has no chokepoint and no default client. Weeks
1–4 of §14 build the commodity layer while the two ideas nobody else has get the smallest budget.

**Conceded, entirely.** This program does not build the log. The only property the target below needs
from a log is *proof of order* — that a sealed preregistration existed before the runs it commits — and
that is obtainable from an existing service or from a commit with an external pin. §8's Merkle tree, STH
signing, mirrors and gossip are out of scope for this program, and the spec's own §1.5 and §8.6 already
say that with one key, no revocation and no external pin the log establishes internal consistency and
nothing more.

**H2 — Mathematicians demand a certificate, not a recipe.** Every time the field faced a machine-evidence
crisis it made checking cheaper rather than making provenance auditable: the Kepler referee process
produced Flyspeck (recalled-high-confidence), the ~200 TB Pythagorean-triples proof produced a 68 GB
checkable certificate (verified-this-session), Tucker's Lorenz computation produced Immler's verified
Isabelle ODE solver (recalled-high-confidence by `rigor-interface`, whose paper identification was
verified-this-session). A reproduction recipe makes verification cost the same as production, which is
the most expensive verification there is.

**Conceded for every claim that admits a certificate** — and the table above rejects all of them for
exactly this reason. **Answered for the residue, and only there.** A method claim has no certificate in
principle. "This procedure beats that procedure at this budget" is a statement about a distribution over
seeds; a distribution is not exhibited by a witness, it is estimated by re-running. There is no cheaper
check to build, because the thing being claimed is the spread itself.

**H3 — The challenge incentive is near zero, and there is direct evidence.** ReScience C exists solely to
publish replications and its front page shows one to two articles per volume; Claesen et al., Royal
Society Open Science 2021, found 2 of 27 preregistered Psychological Science papers with no deviations
and 9 disclosing none (both verified-this-session). §14's own ship gate budgets for asking a named outside
reviewer and recording "their result or its absence after 14 days" — a protocol whose acceptance test
provides for silence has predicted its own adoption curve.

**Conceded on adoption.** Partially answered on dependency: the opening artifact below does not require
anyone else to file anything. Its output is a number the lab measures about its own procedure, and it is
publishable whether or not a stranger ever engages. But the adversary is right that the challenge
mechanism (§9) is unexercised until someone outside files, and this plan does not schedule that, does not
predict it, and does not count it as a deliverable.

**H4 — No mathematical community has adopted this class of infrastructure, and MaRDI is doing a looser,
easier version with institutional backing.** (MaRDI as a DFG/NFDI-funded consortium for FAIR mathematical
research data: verified-this-session; its practical penetration: the lens states this is its recollection,
not a measurement.)

**Conceded.** The consequence is written into the target: this program does not sell infrastructure to
mathematics. It measures one thing, publishes the number, and the number is useful to whoever wants it.

**H5 — The cleanest kill: the noise floor solves a problem engineering is deleting.** He and Thinking
Machines, September 2025, demonstrate batch-invariant kernels giving bit-identical outputs over 1,000 runs
of Qwen3-8B (verified-this-session). If deterministic inference becomes the default, §3–§5 collapse into
a hash comparison and what remains is a signed manifest, for which SCITT, Sigstore and RO-Crate exist.

**This one is answerable, and the answer is the reason for the choice below.** Batch-invariant kernels
delete the *arithmetic* component of the null. They do not touch the *optimization* component. Make every
forward pass bit-deterministic and the distribution of best-found objective across seeds sits exactly
where it was, because it was never a fact about summation order. That is the separation the lab's own
probes already made — bit-identical batch-1 reruns beside a small non-zero flip rate under batching, at
5–6 of 48 items on Qwen2.5-0.5B-Instruct and 8–13 of 256 on gemma-2-2b-it, with the containment reading
withdrawn after 7 of 17 precision flips survived the exclusion on the second model
(`papers/v8/probe_batch_invariance_2026_09_08/`, `papers/v8/probe_batch_invariance_v2_2026_09_08/`, cited
from §5.6 of the spec, not re-run here). So H5 lands squarely on the fingerprint layer and misses a
search-comparison target. A program that chose a fingerprint-shaped target would be choosing the thing
deterministic kernels are in the process of deleting.

**H6 — The claim-binding rule binds only what passes through the tool.** Journals and arXiv decide
publication format; an author can always print the number in a PDF.

**Conceded, and never claimed otherwise.** §7.2's scope-limited claim string binds the lab's own
sentences. That is all it is for and all this plan says about it.

**H7 — One issuer, one key, no revocation, no mirror.** The spec concedes it at §1.5 and §8.6.

**Conceded, restated, and made irrelevant by dropping the log from scope.**

**The adversary's five survival conditions.** Four are adopted: drop the log as an implementation target;
concede publicly every domain where a certificate exists (the rejection rows above are that concession in
writing); ship §5.3 as the earliest standalone thing rather than the last; make §14(b) — an outside verify
cert under a key that is not fathom's — a blocker rather than a checkbox. **The fifth is not met and is
the largest open item in this document:** no compulsory, invisible chokepoint has been found. There is no
benchmark maintainer refusing to list a score without a cert, no venue artifact track, no release gate.
Without one there is no adoption path, and this plan does not pretend to have one.

## The target

**One target: the measurement claim underneath machine-assisted discovery — a head-to-head between a
model-guided search and an unguided baseline at a fixed budget, on a problem whose objects are checked
exactly in milliseconds, reported as a null distribution over seeds plus a detection-power receipt stating
the smallest difference the procedure could have resolved.**

The chosen instantiation is a combinatorial search with an exact checker — a cap-set-style construction in
low dimension, or a small-format bilinear scheme — because the grader is then frozen by construction and
shares nothing with the instrument, which is what §7.2 requires of a grader and what an LLM-judged pipeline
cannot offer.

Why this and not the runners-up. Four of six lenses converged on it from different directions, and none of
them was looking for the same thing: `machine-discovered-constructions` arrived by elimination (the object
is the certificate everywhere except in claims about a process); `large-scale-proofs` arrived by asking
where the produced artifact is trivially checkable and the claim about producing it is entirely unchecked;
`prior-art-adversary` arrived by prior art, concluding that §5.3's rule — that no `same` may be issued
without a measured detection power — is the one sentence in the spec that no existing system already says;
`this-box` arrived by costing, and reported that nobody in this literature publishes what re-running the
same search produces, which means every reported gap between two systems is currently uncalibrated.
Convergence from four different arguments is not proof, but it is the strongest signal the survey
produced, and it is stronger than any single lens's enthusiasm.

Why not exhaustion, the closest runner-up: its evidence gap is more severe (no witness can exist at all),
but `this-box` costed the remedy at a pinned shard function plus per-shard digests plus random audit, and
said in its own words that the machinery v8 has spent its complexity on would sit entirely idle. Choosing
it would mean building a text file and calling it a system. Why not the round-off-floor claim, the second
runner-up: it is the lab's own batch-invariance probe transposed exactly, which is its appeal, but
`fluid-blowup` states plainly that float64 GEMM on a single GPU at fixed shapes is usually run-to-run
bitwise deterministic, so the methodology carries over and the magnitudes do not — and the subfield is
rigorizing, which shrinks the residue over time.

**Mechanisms this target exercises.** §7.2 sealed preregistration, with the commitment over the complete
body and the reveal that a validator recomputes — the log index as proof of order is the whole reason a
head-to-head means anything. §7.2's frozen grader, including the rule that a model grader may not share
weights, features or prompt definition with the instrument. §5.1's requirement that the nuisance plan be
logged before the runs. §5.3's sensitivity receipt, which is the load-bearing mechanism and the reason for
the choice. §7.2's scope-limited claim string, every numeral resolving to a leaf of a logged result cert.
§2.3 recipe-completeness — weights hash, chat template verbatim, decoding, `batch_size`, `padding_side`,
env lock — and §2.3's comparability gate, which refuses to compare across a recipe difference instead of
ranking. §2.2's alias caveat, which is the stated reason the closed systems that produced the headline
results cannot be the subject. §7.3's Goodhart adversary, because the evaluator *is* the grader here and a
search that games the evaluator rather than solving the problem is precisely that section's failure mode.
§6.1's result kinds `prereg`, `confirmatory`, `sensitivity`. §5.5's baseline rule.

**Mechanisms it does not exercise, and must not be described as exercising.** All of §4 — battery certs,
canary selection, δ-sweeps, pool certs, the unmeasured H-canary hypothesis. §3.2's channels (`exact`,
`seqlp`, `topk`, `resid`, `lens`). §5.2's drift, skew and identity verdicts. §8's Merkle tree, STH signing,
mirrors and gossip. §9's challenge certs, until someone outside files one. §10's action certs. §12's
compliance view. §13's docket.

**One mechanism the spec does not have and this target needs.** §5.1 defines `floor_c = max(D_c)` over
pairwise distances between deterministic battery reruns. A best-found-objective distribution over seeds is
an extreme-value statistic, and max-of-pairwise is the wrong shape for it (`this-box`, reasoning
verified-this-session against the spec text). An estimator for that null is owed as an amendment before
any run, and this plan states it as owed rather than assuming §5 covers it.

## What gets built, in order

**Artifact one — the null and the detection power, on one exact-checked target.** A noise-plan prereg and
a sealed prereg fixing the budget, the seed count, the target class, the baseline and the stopping rule;
R ≥ 8 baseline runs and R ≥ 8 guided runs under the logged plan; the empirical spread of best-found
objective for each; and a planted-needle sensitivity receipt — seed the search with a construction of
known difficulty and show the pipeline finds it, then perturb by a known amount and report whether that
shift clears the spread. Cost on this box: the estimator amendment is half a day of CPU work and must land
before anything runs; the runs are three days of wall clock of which under twelve hours are GPU, sized
from `this-box`'s measurements (checker cost negligible — 130,816 pair-sums for a 512-point cap set in
F_3^8; gemma-2-2b at nf4 ≈1.7 GB comfortable inside 8.0 GiB, bf16 ≈4.9 GiB tight but workable). The
deliverable is the measurement, not a construction, and it publishes whatever it says — including "the
guided loop did not separate from the baseline at this budget on this box", which is a result.

*Worth building artifact two only if:* the smallest difference the procedure can resolve lands **below**
the gaps this literature reports. If the aperture is wider than the claims, the instrument cannot
adjudicate what it was built for, and that is where the program stops and publishes.

**Artifact two — §5.3 as a standalone instrument that runs against a harness the lab does not own.** The
adversary's demand, taken literally: a tool whose input is somebody else's evaluation or search harness and
whose output is a detection-power receipt for it, with the recipe fields it could not recover named as
missing rather than guessed. This is where the sentence "an agreement number without its detection power
is not a number" either becomes usable by strangers or is shown not to be.

*Worth building artifact three only if:* artifact two ran end to end against at least one harness the lab
did not write, and produced a receipt whose missing-fields list is short enough to be closable.

**Artifact three — the outside verify cert, made a blocker.** §14(b): a `result` cert of kind `verify`,
issued under a key that is not fathom's, on hardware whose (gpu, driver) differs from every entry in the
target's floor runs, with its verdict whatever it is. Treated as a gate, not a checkbox, and its absence
after the stated interval recorded in `REPLICATIONS.md` as the answer. Cost is not this lab's to estimate,
which is itself the point.

## The falsifiable criterion

Stated before any work, in one sentence that can come out false: **for the chosen target class, the
smallest difference in best-found objective that this procedure can resolve at the preregistered budget —
measured as the seed-to-seed spread and demonstrated by a planted perturbation that the sensitivity
receipt shows the procedure detects — is smaller than the guided-versus-baseline differences this
literature reports.** If it is larger, the program has built an instrument whose aperture is wider than
the claims it was built to adjudicate, and every downstream artifact is void. That failure is publishable
and this lab has published its shape before: an agreement number without its detection power is not a
number, and the miss list is the result. The criterion is not "did we find something"; the search may find
nothing and the criterion still resolves. It is not "did the guided loop win"; that outcome is data either
way. It is a statement about the instrument's aperture, made before the aperture is measured, and it is
the only sentence in this plan whose truth value is determined by the runs rather than by argument.

## What this program may never claim

The following sentences are forbidden in advance, in the words the lab must never write.

- "Styxx verifies mathematics." "Styxx verifies a proof." "Styxx checks whether the construction is correct."
  The exact checker does that, in milliseconds, and it does it without us.
- "Certifies that the search was exhaustive." "Certifies completeness." The system can certify that two
  implementations agreed; it can never certify that a search terminated with nothing left.
- "Attestation instead of formalization." Where a claim reaches a kernel, the kernel is the stronger
  instrument and this program defers to it, in writing, in the paper.
- "Detects contamination." Nothing here distinguishes training-set leakage from competence, and the
  benchmark-access failures the survey found are untouched by every mechanism named above.
- "The formal statement means the theorem." A byte-perfect reproduction reproduces a vacuous theorem
  faithfully; a formalization that drops a hypothesis type-checks.
- "Reproducible," said of anything running against a closed provider alias. §2.2 already concedes that a
  `same` verdict against a provider that conditions on the request is worth nothing, and a recipe for an
  unpinnable loop is a record of intent.
- "Our system beats the baseline," without the measured spread and the sensitivity receipt beside it.
- "The log makes the result trustworthy." The log stores; it does not endorse; inclusion is not validity.
- "Self-verifying." "Immutable." "Tamper-proof." Already forbidden by doctrine; repeated here because
  this domain invites all three.
- "Nobody else does this." "The only." Any superlative about priority, in any wording, before the survey
  the spec owes has been written and read in full.
- Any number about a method comparison that does not resolve to a leaf of a logged result cert.

## Limits

One session. Six model-driven lenses and no human specialist. No primary source was verified in this
session except where a lens says it verified one, and the lens's confidence tag travels with every
statement above — several load-bearing items are marked *recalled-uncertain* and are treated as
uncertain, not rounded up. No mathematician was consulted, and the single strongest argument against this
entire program (H2: mathematics wants a certificate, not a recipe) came from a lens, not from a
mathematician who would have to be persuaded. The chosen target was reached by convergence of four lenses
that were not coordinating; convergence among model-driven lenses reading overlapping literature is
correlated evidence, and this plan does not treat it as independent. The hostile lens's chokepoint
objection is unanswered. No number in this document was produced by a run performed for it.

---

*The survey went looking for a place where mathematics could not believe a machine, and found instead that
mathematics has been systematically removing the need to believe anything — a checker here, a kernel there,
a witness cheaper than the search that found it. What is left after all that removal is not an object but
a sentence: somebody's claim that their procedure beat somebody else's, at a budget nobody stated, over a
number of seeds nobody reported. That sentence is checked by nothing today. Measuring how small a
difference one can honestly resolve is a modest thing to build, and it is what the receipts allow.*
