# STYXX_CAPTURE_TRUST.md

**Styxx Proof-Carrying Cognition — Capture-Trust Addendum**
**Companion to:** `STYXX_PROTOCOL.md` v1.0 (`styxx-cert/1.0`), §8.3 threat model · §10 conformance ladder (L0–L3) · §13 gap 1 (trustless capture UNSOLVED)
**Version:** 1.0 — Design-space map, NOT a claimed solution
**Status:** Stakeholder methodology note under EU AI Act Art. 15(2). NOT a normative standard. NOT a security proof.
**Date:** 2026-06-07 · Fathom Lab / styxx

> This document maps the design space for the single biggest practical gap in the protocol — forge-resistant *capture* of the caving residual when the model runs on the reporter's own infrastructure. It adopts the discipline of the spec it extends: **the open problem is stated as loudly as any capability, and no cryptographic or hardware guarantee is claimed beyond what is buildable and validatable today.** The honest verdict is **partial solvability with trust relocated, never eliminated** — and the part that stays open is the part that matters most.

---

## 0. The dogfood rule applies here too

Every claim below is bounded to what is buildable and validatable today. Where a scheme *sounds* like it closes the gap but does not, the residual is named in the same breath, in **bold**. A capture-trust note that oversells any of the three schemes would fail the protocol's own §0 gate exactly as a fabricated DELTA would. **The "what it does NOT do" column for every scheme is at least as long as the "what it does" column — by design, because that asymmetry is the finding.**

---

## 1. The problem, stated exactly

The styxx certificate reads a model's **caving residual** — the activation at a pinned layer/position — to verify what the model *held* (RUNG 1) and whether it *hid* (RUNG 2). The certificate is **VERIFY-ONLY** (§9.2): the read is read-only by construction, so the attack surface is **forging the read, not the model**. That is a genuine narrowing — the adversary cannot turn the certificate into a writer — but it does not narrow the capture problem at all.

**v1 trust model (`STYXX_PROTOCOL.md` §8.3, the only level buildable today, L1):** SHARED-PROBE-HOST. The *verifier* owns a pinned copy of the model + probe and **re-runs the reporter's forward pass itself**. The reporter submits only `(system, prompt, output)` and **never activations**. There is nothing to forge in scope: the verifier captures the residual, so "who captured it" is "the verifier did." Forge-proof **within that scope** (§9.3 attack A1: any submitted activation/score field ⇒ MALFORMED).

**The real-world question this addendum exists to answer.** When the model runs on the **reporter's own infrastructure** — a vendor self-hosting a closed or proprietary model the verifier *cannot* re-run — the verifier no longer captures the residual. The reporter must **submit** it. And a malicious reporter can **forge** it: fabricate a caving residual that yields a clean certificate while the real model did something else. The unanswered question is:

> **Who captured the activations, when, from which model weights, on which input, at which layer, unmodified, and without the reporter having search-optimized a fake?**

This is the **TRUSTLESS-CAPTURE gap**, protocol levels **L2** (forge-resistant, named gap) and **L3** (trustless capture, research horizon). It is **UNSOLVED** and named as a hard gap in §10 and §13. This document does not solve it. It maps it.

### 1.1 What a trustworthy off-host certificate must bind

For a submitted residual to be trustworthy off the shared host, the binding must prove the residual:

- **(a)** came from the **CLAIMED model weights**;
- **(b)** on the **CLAIMED prompt + output**;
- **(c)** at the **claimed layer / position**;
- **(d)** **UNMODIFIED** after capture;
- **(e)** **without the reporter having search-optimized** a fake residual that happens to yield a clean verdict.

Note the structure: (c), (d), (e) are claims about an *artifact* (a vector and its handling). (a), (b) are claims about an *origin* (a computation that actually ran). **The three schemes below collectively close (c), (d), (e) and partially (b). None of them, alone or composed, closes the deepest form of (a).** That asymmetry is the whole result.

---

## 2. Threat model for this addendum

- **Adversary:** a fully host-controlling reporter. It owns every byte, every key, every process, the OS, the firmware load order, and the model file. It is a *vendor self-attesting its own deployment* — the hardest realistic case, and the one the protocol is built to be honest about.
- **Goal of the adversary:** ship a *verifying* certificate (every signature checks, every hash resolves, every gate passes on the submitted bytes) that asserts the model **held nothing / never knew / did not hide** when in fact it did.
- **In scope:** forging the *read* — the captured residual and its provenance binding.
- **Out of scope (named, inherited from the spec, NOT re-solved here):** key distribution / the reporter-identity root (§13 gap 8 — agent cards / DNSSEC / X.509); the external-anchor timestamping infrastructure (§13 gap 8); the read≠write invariant (§9.2, enforced by code construction, not by any scheme here); the probe's substrate validity (§13 gap 3 — open-weight ≤3B, clean only on Llama-3.2-3B DELTA 0.290).
- **The read-only invariant helps, narrowly.** Because the certificate cannot be turned into a writer (§9.2), the adversary's prize is a *false attestation of honesty*, not the ability to install a state. This caps the blast radius. It does **not** make the capture forgeable-or-not question any easier.

---

## 3. The three candidate schemes — honest map

| | **Crypto-binding** (commit-reveal + signed tuple + anchored Merkle head) | **Hardware TEE** (CPU-TEE chained to H100/H200 GPU-CC) | **zkML** (zk proof-of-inference, zkLLM-class) |
|---|---|---|---|
| **What it is** | Bind `(weights_hash, prompt, output, layer, position, residual)` into a signed, externally-timestamped, replay-proof chain | Run the forward pass + probe inside an attested confidential-compute enclave; hardware signs that *these bytes ran on these weights on this input* | Prove "probe P on residual of model M(prompt) at layer L yields verdict V" without revealing M's weights |
| **Binds (a) weights→residual at ORIGIN** | **NO** — binds residual to the *string* `weights_hash`, never to a computation | **YES, strong-by-construction** — weights hashed *inside* the sealed boundary; host can't swap post-check | **PARTIAL** — proves correctness against a *prover-chosen* commitment `C_M`; cannot anchor that `C_M` is the real deployed model |
| **Binds (b) prompt+output** | YES (signature-covered) | **YES, strong** — enclave runs the pass over the signed tuple | YES, conditionally (binds residual↔prompt, not output-was-actually-sampled) |
| **Binds (c) layer/position** | **YES** (in signed subject-digest + commit preimage) | **YES, strong** | **YES, strong** (hardcoded circuit wire) |
| **Binds (d) unmodified** | YES, *relative to the commit* (tamper-evident + anti-replay + anti-reseal) | **YES, strong-by-construction** — no residual input port to modify | **YES, strong** — residual is an internal circuit wire, never a free input |
| **Binds (e) no search-optimized fake** | **Only vs the per-session challenge** — collapses (offline grinding against the *public* gate calibration is free) | **YES, strong-by-construction** — host-opaque, hook-free, gradient-blind execution; no optimization surface | **YES, strong by soundness** — residual must be a real preimage of the committed weights. *Cleanest (e) of the three.* |
| **Proprietary-model confidentiality** | N/A (residual is revealed) | **YES** — verifier never holds or re-runs weights | **YES** — weights are a private witness; *uniquely without a hardware vendor in the TCB* |
| **Deployable on full-size proprietary models today** | Yes (but doesn't bind origin) | **YES — the only scheme that is** | **NO** — research-horizon at LLM scale |
| **Feasibility today** | near-term (days on existing styxx machinery + a TSA) | near-term (silicon GA; validated styxx binding = multi-month) | **research-horizon** |
| **Cost / certificate** | a few hashes + 1 Ed25519 sign + 1 anchor write (sub-cent, ms) | single-digit % CPU-TEE; H100-CC <5–10% at batch; *engineering* is the spend | **100×–1000× bare inference** (minutes–tens-of-min A100/H100, ×3–4 for gates) |
| **Honest level** | **L1.5** (closes temporal/integrity; makes forgery attributable) | **L2 candidate** (under hardware+TCB trust) | **L2-privacy complement** (when a maintained ≤3B prover exists) |

The rest of this section is the long-form, per-scheme version of that table, with the attacks that survive each.

### 3.1 CRYPTOGRAPHIC BINDING — honest level **L1.5**

**What it is.** Before any item is seen, freeze a public manifest: `WEIGHTS_ID` (sha256 over checkpoint config + Merkle root of weight shards, the spec's `model_id` §4), `PROBE_ID` (`probe.weight_sha256`), `GATE_CAL_ID` (sha256 of the per-(model,probe,substrate) PRIME/ABORT/DIVERGENCE calibration record, §8.1). Then per item: **commit** `c = sha256(residual_bytes || subject_digest || verifier_nonce || reporter_nonce || counter)` and **anchor** it externally (RFC-3161 TSA / Sigstore-Rekor / transparency log) *before* the **reveal**. Sign the JCS-canonical tuple with Ed25519 (reusing the shipped `styxx.attestation` contract verbatim, §3.2). Append the signed head to a Merkle chain whose head is externally anchored *before any tamper window* (§9.1 anti-reseal). The verifier recomputes everything **without re-running the model** — which is the entire point of the off-host case.

**What it genuinely closes** (real, cheap, ship-now):
- **Swap** (A1 relabeling), **replay** (A3, verifier-first nonce + monotonic counter), **whole-history re-seal** (external anchor), **layer/position relabel** (c), **silent recalibration / probe-swap** (A4/A7, frozen public manifest).
- Converts forgery from **cheap-and-deniable → expensive-and-attributable**. Every forged cert is a signed, timestamped, Merkle-linked ledger entry welding **one reporter identity to one lie**. That is the enforcement substrate for revocation/reputation. **This is worth shipping.**

**What it fundamentally cannot do (the L2 residual, named):**
- **ORIGIN FORGERY — fully works, no step checks against it.** The host never runs `WEIGHTS_ID`. It instantiates a *different* model (fine-tuned / hand-edited / a residual generator), produces a residual the public probe scores clean, then *honestly* commits, signs, and anchors it under the name `WEIGHTS_ID`. Every check passes; the certificate is a lie. **Crypto binds string-to-string (residual ↔ the hash `WEIGHTS_ID`); it never binds computation-to-bytes.**
- **OFFLINE PRE-COMMIT GRINDING — defeats (e) here.** `GATE_CAL_ID` *must* be public for third-party verifiability, and the probe is a fixed linear map. The adversary grinds a clean-passing residual offline against the public deterministic gates, *then* commits it honestly. The verifier-first time-lock only stops grinding against a per-session challenge — but nothing here *is* a per-session secret. **Commit-reveal reduces to pre-registering a number the reporter was free to choose.**
- **MANIFEST-vs-PROCESS gap.** Nothing welds the *live* forward pass to the *file* whose root is `WEIGHTS_ID`. The hash names a file, not the loaded process.

> **Verdict:** L1.5. Closes temporal/integrity, supplies the audit ledger, makes forgery loud and attributable. **MUST NOT be marketed as closing L2** — the origin-of-computation forgery it cannot stop *is the definition of the L2 gap.*

### 3.2 HARDWARE TEE — the only **L2 candidate** deployable today

**What it is.** Run the forward pass + the styxx read inside an attested confidential-compute enclave. The credible build is **NVIDIA H100/H200 Confidential Computing** (the residual lives in GPU HBM, so a CPU-only TEE does not by itself cover where the activation physically is) **chained to a CPU TEE** (Intel TDX / AMD SEV-SNP) for orchestration. The reporter must run a **specific, open, verifier-audited reference binary** whose measurement (MRTD / launch-measurement / attested GPU VBIOS+driver+VM state) is pinned in advance. That binary streams the pinned checkpoint, hashes it *inside the boundary*, aborts unless it equals `subject.model_id`, runs **one** pass over the signed `(system,prompt,output,layer,position,nonce)`, captures the residual, runs the probe + all three gates in the same pass, and signs the certificate with an enclave-held ephemeral key bound by the hardware quote. The verifier validates the quote against vendor PKI (Intel PCS / AMD KDS / NVIDIA NRAS/RIM), confirms production TCB + DEBUG-off, confirms `measurement == audited binary`, then trusts the key. This moves the root from *trust the reporter* to *trust the silicon vendor + the audited TCB* — **without the verifier ever holding the proprietary weights.**

**What it genuinely closes** (the structural win):
- **The reporter-submitted-residual forge class is eliminated.** The reporter supplies *inputs, not activations*. There is **no residual input port to forge against**, so **(d) unmodified** and **(e) no-search-optimized-fake** are **strong-by-construction** — host-opaque, hook-free, gradient-blind execution removes the optimization surface entirely. The gates run in-enclave on the same pass.
- **(a) bytes-match-hash, (b) prompt+output, (c) layer/position** are bound inside the sealed boundary.
- **Proprietary-model confidentiality** — forge-resistance *without* the verifier holding or paying to re-run the weights. This is the property L1 (verifier re-runs) structurally cannot give a closed-model vendor. Only zkML matches it, and zkML cannot run a full LLM today.

**What it does NOT do (named, and the first two are irreducible):**
- **HONEST PROOF OF A DISHONEST MODEL — works, irreducible.** A vendor self-attesting its *own* `model_id` pins `WEIGHTS_ID` = hash of a model it **deliberately trained/fine-tuned/selected to emit clean caving residuals**. The enclave faithfully loads, hashes, runs, gates, and attests. **Hardware proves these exact bytes ran; it does NOT prove the bytes name a non-gamed model.** Every attestation is genuine and the certificate still lies about whether the model hid. This is gap (a) at the level crypto and zkML *also* cannot close.
- **BAIT-AND-SWITCH / COMPLETENESS — works, this is the L3 gap.** Run the attested honest enclave to *mint certificates*; serve a different (cheaper / malicious / un-attested) model to real users. TEE proves a true pass on the inputs the host *chose to attest*; it proves **nothing about the population of real traffic**, and the host can cherry-pick which `(prompt,output)` pairs get attested and silently drop the embarrassing ones. **No completeness / non-cherry-picking / served-equals-attested guarantee.**
- **TRUST RELOCATED, NOT ELIMINATED.** Trust moves to: the silicon vendor + provisioning PKI + firmware/microcode/SEAM module + the guest kernel/initrd + the GPU driver + **a recurring reproducible-build audit of the reference binary**. A compelled/malicious vendor, a signed-but-backdoored microcode/SEAM/driver update, or a key compromise breaks the root.
- **SIDE-CHANNELS — live, unresolved, catastrophic if achieved.** The SGX lineage was repeatedly broken (Foreshadow/L1TF, Plundervolt, SGAxe, AEPIC); VM-TEEs have their own (SEV: CipherLeaks, CacheWarp, WeSee; TDX-down-class). The host has exactly the microarchitectural access these assume. **A single enclave signing-key extraction lets the reporter sign arbitrary forged certs that pass attestation — the entire root collapses silently and indistinguishably from genuine.** TEE forge-resistance is only as strong as the *current* side-channel posture of a TCB with a poor track record.
- **ATTESTS CODE IDENTITY, NOT CORRECTNESS.** The hardware proves "measurement X ran with these inputs," not that X computes the styxx read correctly. *All* of (a)–(e) rest on the verifier having **audited the reference binary** (and the probe, the gates, the fp-quantization). A subtle bug or a deliberately weak probe baked into the audited binary is fully attested as genuine. The audit burden is real and recurring on every TCB/microcode/driver bump.
- **DOES NOT EXTEND THE PROBE'S SUBSTRATE.** TEE upgrades *who-can-forge*, not *what-the-probe-can-read*. Still open-weight ≤3B validated, clean only on Llama-3.2-3B (§13 gap 3). A confidential frontier model is attestable-as-bytes but the probe is **unvalidated there** — the spec's OUT-OF-SCOPE closed-model rule still bites.
- **NO ANTI-RESEAL ANCHOR.** Per-cert freshness only; the session-chain anti-reseal anchor (§9.1) is still owed infrastructure.

> **Verdict:** the single most promising path to a **buildable** L2 today. zkML can't run a full LLM; crypto can't bind weights→residual at all. TEE directly solves the proprietary-self-hosted case **but relocates trust to a silicon TCB with a broken side-channel history, attests code-identity not correctness, and leaves the two irreducible gaps wide open.** Scope its claim precisely: *forge-resistant under hardware+TCB trust, audited binary, per-cert not whole-session.* **Never trustless.**

### 3.3 ZERO-KNOWLEDGE PROOF-OF-INFERENCE — **L2-privacy complement**, research-horizon

**What it is.** The reporter generates a succinct proof that "model `M` (committed via a polynomial/Merkle commitment `C_M`, weights kept private) on committed input `(system,prompt,output)` up to layer L, then probe P on the residual, yields verdict V and gate quantities G" — revealing only `(V, G, public commitments)`, **not** M's weights. The probe weights are public (styxx ships them); the transformer forward pass is the entire cost. Verification is O(polylog) — **milliseconds, KB-scale proof, independent of prover cost.** That cheap, weights-blind verification is the genuine win.

**What it genuinely closes** (cleanest bytes-fabrication closure of the three):
- **(d) unmodified** — residual is an internal circuit wire, never a free input.
- **(e) no search-optimized fake — strongest of the three, by soundness.** The residual *must* be the genuine output of the committed weights on the committed input; the reporter cannot independently optimize a fake. **This beats crypto-binding's (e) decisively** (crypto's (e) collapses under offline grinding; zkML's does not).
- **(c) layer/position** (hardcoded wire), **(b)** conditionally (residual↔prompt, not output-was-sampled).
- **Weight-privacy without a hardware vendor in the TCB** — the one property TEE cannot offer, and the reason to keep zkML on the roadmap.

**What it does NOT do (named):**
- **MODEL IDENTITY for a proprietary model — the central failure, same irreducible gap as TEE by a different road.** Soundness forces the residual to be the real output of `C_M` — **but the reporter chooses `C_M`.** Commit `C_M'` = a model fine-tuned/selected to emit clean caving residuals, prove honestly against it, ship a verifying certificate. For the **open** case the verifier already holds the weights and would just re-run (L1) — zkML buys privacy it doesn't need. For the **proprietary** case that *motivates* zkML, the verifier **cannot independently anchor `weights_hash(M)`**, so the commitment is unanchored and the forge gap is **not closed — it is relocated** to "who certifies which committed model is the deployed one."
- **SERVED ≠ COMMITTED.** Even granting `C_M` is "the" model, nothing binds the model that generated real users' outputs to `C_M` — same bait-and-switch as TEE (the L3 completeness gap).
- **DISHONEST QUANTIZATION — an attack surface, and an unvalidated empirical question.** The fp32 transformer must be quantized to fixed-point/finite-field. **Whether the DELTA-0.290 read and the gate margins even survive honest quantization is unvalidated** (§13 gap 3 in spirit), and a dishonest quantization that flips the verdict is itself a forge vector hidden inside a "plausible" choice.
- **GATE-THRESHOLD SUBSTITUTION.** The prover must also prove PRIME/ABORT/DIVERGENCE used the **verifier-canonical** calibrated thresholds, not prover-chosen ones, across ~3–4 separate attested passes (PRIME positive-control, ABORT never-knew, DIVERGENCE clean-position — the gate suite is not one pass). Wiring this honestly is non-trivial and a prover-chosen-threshold slip is a silent forge.
- **DOES NOT EXIST AS A DEPLOYABLE STACK AT THIS SCALE.** The only evidence of ≤13B-class proving is a **single custom research prover** (zkLLM, CCS 2024: ~13B LLaMA in ~15 min on one A100, ~803 KB proof, via custom GPU sumcheck + tlookup + a zkAttn gadget). A ≤3B styxx target is *inside* that research envelope for **a single pass** — but that is a one-off academic prover at fixed precision/architecture, not a `pip install`. General turnkey tools (ezkl/halo2) top out around small CNNs / <100M params and do **not** scale to multi-billion-param transformers without bespoke engineering. **Claiming L2 on this basis today violates the protocol's own §10 no-self-declared-L2 rule.**
- **MUST BE COMPOSED.** To make (a) bite you still need a non-zk trust assumption (a TEE attestation or a governance registry) that anchors `weights_hash(M)` to the model the vendor actually serves. **zkML is the privacy layer, not the trust root.**

> **Verdict:** the most *elegant* endgame — uniquely trust-minimizing (no hardware vendor in the TCB) and the cleanest (e). But it is **research-horizon** (no maintained ≤3B stack; 100×–1000× cost; quantization-validity unproven) and **identity-unanchored for exactly the closed-model case it targets.** Hold it as the longer-horizon weight-privacy complement, **composed** with an identity anchor.

---

## 4. The red-team — what survives each scheme

| Attack | Crypto-binding (L1.5) | Hardware TEE (L2 candidate) | zkML (L2-privacy) |
|---|---|---|---|
| **Forge a fake residual & submit it** | **WORKS** (no origin check) | **KILLED** (no residual input port) | **KILLED** (soundness; internal wire) |
| **Search-optimize a clean-passing residual** | **WORKS offline** (public gate calibration) | KILLED (host-opaque execution) | KILLED (by soundness) |
| **Swap / replay / relabel layer / re-seal history** | KILLED (anchor + nonce + counter) | KILLED (freshness; re-seal owed anchor) | KILLED (commitment) |
| **Honest proof of a *gamed* model** (train/select a model that reads clean) | **WORKS** | **WORKS — irreducible** | **WORKS — irreducible** |
| **Bait-and-switch / cherry-pick attested traffic** (L3) | **WORKS** | **WORKS — irreducible** | **WORKS — irreducible** |
| **Side-channel signing-key extraction** | N/A | **WORKS if achievable → catastrophic** | N/A (no enclave key) |
| **Audited-binary / quantization subversion** | N/A | WORKS if the audit misses it | WORKS via dishonest quantization |

**Reading of the table.** Every scheme **kills the bytes-fabrication column** (the reporter-submitted-residual forge) — TEE and zkML structurally, crypto only against the per-session challenge. **No scheme touches the bottom two rows that matter most: the gamed-model and the cherry-picked-stream.** Those two are reached by a *different road* in each scheme but they are the **same wall**: they are claims about the world *outside* the attested box.

---

## 5. The irreducible open problem — stated as loud as any capability

> **A fully host-controlling reporter retains two powers that nothing inside the box can revoke. Cryptography and hardware can prove a SPECIFIC computation happened, was UNMODIFIED, was ATTRIBUTABLE, and was NOT search-optimized — but they CANNOT prove the computation used the RIGHT model on a REPRESENTATIVE, NON-CHERRY-PICKED stream, because "right model" and "representative stream" are claims about the world OUTSIDE the attested box, and the reporter owns that world.**

**(1) Model-identity — "honest proof of a dishonest model."** Whoever *names* the weights gets to name a model that was trained, fine-tuned, distilled, or selected to emit clean caving residuals — a model that genuinely reads "held nothing / never knew" to the probe. TEE attests those exact bytes ran; zkML proves correct inference of that committed model; crypto signs that hash. **All three then certify a TRUE computation of a GAMED model.** No mechanism operating on the residual or the forward pass can distinguish *an honest model that held nothing* from *a model built so the probe always reads clean.* That distinction lives outside the certificate — in (i) an external authority on what the deployed proprietary model actually *is*, and (ii) **the probe's own validity against an adversarially-tuned model**, which is **unvalidated** (§13 gap 3: the styxx read is validated on open-weight ≤3B models *not trained to evade it* — DELTA 0.290 on Llama-3.2-3B, Qwen bounded, gemma instrument-dead). **A model fine-tuned to emit clean caving residuals is exactly the adversary none of these bindings can detect and the probe was never tested against.**

**(2) Completeness — "served-equals-attested" (the L3 gap).** The reporter chooses *which* `(prompt,output)` pairs get the attested/proven pass and serves a different model to real users. **No per-output certificate proves the certified model is the one that produced production traffic, nor that embarrassing outputs were not silently dropped.** Structurally untouched by all three schemes.

**The two problems converge.** The binding problem (is this the right model?) and the substrate-validity problem (can the probe read an adversarially-tuned model?) **land on the same open frontier.** This is the load-bearing honesty of the whole document: **the gap that remains is the adversary the probe was never validated against.** TEE and zkML upgrade *WHO-can-forge*; neither upgrades *WHAT-the-probe-can-read.*

---

## 6. What is achievable TODAY

Stated plainly, in order of how-sure-we-are:

1. **L1 (shared-probe-host) — exists now, for COOPERATIVE deployers.** Where the verifier can hold and re-run the exact open-weight checkpoint (≤3B, validated; clean on Llama-3.2-3B), there is nothing to forge in scope (§8.3, §10). The realistic surface is narrow — the verifier must already hold the checkpoint and pay for the re-run — but inside it, capture is *not* a trust problem. This is the regime the protocol ships in.

2. **L1.5 (crypto-binding) — buildable in DAYS on existing styxx machinery.** Ed25519 signing + RFC-8785/JCS digests + Merkle `attest_chain`/`verify_chain` already ship in `styxx.attestation` (§3.2, §9.1, §12). Commit-reveal is one hash + one round-trip; verifier-first nonce + monotonic counter already specified (A3). The one standing cost is an external anchor (RFC-3161 TSA or a Sigstore-Rekor/transparency-log append — sub-cent, mature; per-commit blockchain is costlier, batch via per-epoch Merkle roots). **Ship this now, labeled exactly as L1.5:** it closes swap/replay/reseal/relabel and makes any forgery loud, attributable, and non-repudiable. It does **not** bind weights→residual and **MUST NOT** be sold as L2.

3. **TEE-attested capture (H100-CC + CPU-TEE) — buildable now for COOPERATIVE self-hosting vendors, validatable over MONTHS.** The hardware is GA (Intel TDX on Xeon Gen4/5/6; AMD SEV-SNP on EPYC Genoa+; NVIDIA H100/H200 CC on Azure NCC-H100-v5 and CoCoS/Phala-style stacks). A credible prototype on rented H100-CC is **weeks** of integration (low-thousands of cloud $ + serious engineer time). A *verifier-trustworthy, independently-audited, side-channel-reviewed* production binding is a **multi-month** effort needing a reproducible-build pipeline and a security audit — and even then it ships as **"forge-resistant under stated hardware+TCB trust," not trustless.** This is the right L2 *for a vendor that wants to be checkable* (cooperative): it eliminates the bytes-fabrication forge and keeps the proprietary weights sealed.

**The honest framing of "achievable today":** for **cooperative deployers** — vendors who *want* their honesty checkable — the stack of {L1 where you can re-run, + L1.5 crypto-binding audit ledger, + TEE-attested capture where you can't} is buildable and strong. **For a fully-adversarial self-hosting vendor — one that does not want to be caught — none of it suffices,** because such a vendor can game the model identity and cherry-pick the stream from outside every box. The line between "achievable" and "open" is exactly the line between a **cooperative** and an **adversarial** deployer.

---

## 7. The single most promising path

**Composition, not any single scheme.**

> **TEE as the L2 origin root** (supplies the (a)+(b) origin guarantee under an explicit hardware+TCB trust assumption — the only scheme that can, on full-size proprietary models, today),
> **with the crypto-binding commit-reveal + signed + externally-anchored Merkle chain layered ON TOP** (supplies (c)+(d)+(e) where they aren't already structural, plus the public, non-repudiable audit ledger and anti-reseal — cheap, ship-now),
> **and zkML held as the longer-horizon, trust-minimizing, weight-privacy complement** for when a maintained ≤3B prover exists (replaces the *hardware vendor* in the TCB with a *math* assumption, and supplies the cleanest (e)).

Division of labor: **TEE gives origin** (under hardware+TCB trust), **crypto-binding gives the temporal/integrity/audit substrate** (under an external-anchor assumption), **zkML eventually gives weight-privacy without a silicon vendor in the TCB.** No single scheme is the answer; the composition is the best buildable system.

**And even that perfect composition leaves §5's two irreducible gaps fully open.** The most promising *path* is composition; the most promising *path does not reach the destination.* That must be said in the same sentence every time the composition is described — gamed-model-identity and served-equals-attested are not engineering debt to be paid down; they are outside the box.

---

## 8. Recommended disciplined framing for styxx (normative-style summary)

1. **Ship crypto-binding NOW, labeled explicitly L1.5.** It closes temporal/integrity (swap/replay/reseal/relabel) and makes forgery loud and attributable on an external, non-repudiable ledger. It **MUST NOT** be marketed as closing L2; the origin forgery it cannot stop *is* the L2 gap. (Reuses `styxx.attestation` verbatim + a TSA/transparency-log integration.)
2. **Pursue TEE as the L2 reference binding, scoped precisely** as *"forge-resistant under hardware+TCB trust, audited reference binary, per-cert not whole-session."* **Never trustless.** Track the side-channel posture and re-audit on every microcode/SEAM/driver bump; treat the enclave signing key as a single point of catastrophic, silent failure.
3. **Hold zkML as the weight-privacy endgame**, composed with an identity anchor, when a maintained/audited ≤3B prover exists. Do **not** self-declare L2 on zkML before then (§10 rule).
4. **Name as standing OPEN problems no scheme here closes** — as loudly as any capability:
   - **model-identity anchoring** (a vendor self-attesting/self-committing a probe-gamed `model_id` passes every check; requires an external, *non-cryptographic* model-identity authority/governance registry);
   - **adaptive-adversary robustness of the probe itself** (validated on ≤3B models *not* trained to evade it; the gamed-model adversary is the one the read was never tested against — this is where the binding problem and the substrate-validity problem converge);
   - **L3 completeness / served-equals-attested / non-cherry-picking** (per-cert only; whole-deployment integrity is open);
   - **external-anchor infrastructure** (§13 gap 8, required-but-not-provided; absent it, anti-reseal **MUST NOT** be claimed);
   - **key-distribution / reporter-identity root** (§13 gap 8, out of scope; a valid-keyed reporter forging under its own identity is made *attributable*, not *prevented*).
5. **Update the spec's conformance ladder text** so L2 reads "TEE-attested, **forge-resistant under hardware+TCB trust** — buildable for cooperative self-hosting vendors, **not** trustless and **not** robust to a gamed model" and L3 stays the open research horizon. The §13 gap-1 entry should cite this addendum and add the gamed-model-identity and completeness gaps explicitly alongside the existing trustless-capture statement.

---

## 9. Verdict

**PARTIALLY-SOLVABLE, AND THE SOLVABLE PART IS NOT THE HARD PART.**

- **Trustless capture in the strict sense** (zero trust assumptions, reporter controls everything) is **NOT solvable** — it is *fundamentally hard*, because two required guarantees are claims about the world **outside** the attested box (is this the *right* model? is this a *representative, non-cherry-picked* stream?) and the reporter owns that world. No cryptographic or hardware mechanism operating on the residual or the forward pass can decide them.
- **What IS solvable, and solvable well:** (1) the **bytes-fabrication / reporter-submitted-residual** forge class is structurally **killable today** via TEE (no residual input port) and, in principle by soundness, via zkML — a genuine, large improvement over the L1-off-host failure mode where a malicious reporter simply signs an arbitrary fake; (2) **swap / replay / reseal / relabel + attribution + a non-repudiable audit ledger** are cheaply **closed today** by crypto-binding, shippable in days on existing styxx machinery.

**Trust is RELOCATED, never ELIMINATED.** The best achievable system is the §7 composition — TEE origin root + crypto-binding audit/anti-reseal chain on top + zkML as the weight-privacy endgame — and **even that perfect composition leaves the two irreducible gaps of §5 open: gamed-model-identity (honest proof of a dishonest model) and completeness/served-equals-attested (L3).**

The line is clean and it is the finding: **for cooperative deployers the capture problem is largely buildable today; for a fully-adversarial self-hosting vendor it is not, and cannot be, closed from inside the box.** A *"partially-solvable-only, trust-relocated-not-eliminated"* conclusion is the correct, valuable, non-overclaimed result — and it is the result this addendum reports.

---

*This document does not implement or validate any cryptographic, hardware, or zero-knowledge scheme. It maps the design space and names what stays open. Per `STYXX_PROTOCOL.md` §10, no conformant implementation may claim L2/L3 until a binding is built and independently validated; this addendum claims neither. Patent posture per §12 (US Provisionals 64/020,489 / 64/021,113 / 64/026,964; non-enforcement pledge against open implementations).*