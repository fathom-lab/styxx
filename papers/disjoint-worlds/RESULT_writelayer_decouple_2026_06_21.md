# RESULT — READ ≠ WRITE survives the confound: the label-free map reads a foreign mind but cannot write to it, even where steering works

**2026-06-21.** Executed `PREREG_writelayer_decouple_2026_06_21.md` (frozen `[this commit's parent]`, before the
run). Rung 1 found READ≠WRITE but native steering was weak (0.051) at the read-optimal shallow layer (dst 6),
confounding the write-null. This decouples the read point from the write point and **kills the confound.**

## Verdict: `CLEAN READ ≠ WRITE` (un-confounded)

The native-steering sweep picked the steer-optimal layer, where the 1B steers *well* on its own:

| dst write layer (frac) | native steering gain (sel) |
|---|---:|
| 8 (0.5) | 0.111 |
| 10 (0.6) | 0.150 |
| **11 (0.7) — chosen** | **0.179** |

At the chosen layer (dst 11), over all 70 held-out concepts:

| quantity | value | gate | |
|---|---:|---|---|
| **native** steering gain (B's own ceiling) | **0.2151** | — | **strong** (4× the shallow layer) |
| pos control `pc_cos` (subspace hosts the vector) | **0.818** | ≥ 0.80 | ✓ no structural tension |
| RSA(src 20, dst 11) | 0.951 | — | near-isometric |
| **transfer** steering gain | **0.0245** | G1 ≥ 0.15 | ✗ |
| NTE = transfer / native | **0.114** | G3 ≥ 0.40 | ✗ (now *meaningful* — native is strong) |
| transfer − random-Q | 0.029 | G2 mag ≥ 0.10 | ✗ |
| transfer > random-Q (sign) | 50/70 (71%) | G2 sign ≥ 70% | ✓ |

**The confound is dead.** This layer steers natively at 0.215 and its subspace hosts the steering vector
(pc_cos 0.82) — yet the label-free transferred direction recovers only **11% of native control**. It is not
"the wrong injection point." The map reads (0.586) and does not write (0.0245), even where writing plainly works.

## The nuance worth keeping (not pure noise — a *whisper*)
The transferred direction beats random-Q on **71% of concepts** (the G2 *sign* test passes) while its magnitude
is negligible (0.029 over random; the G2 *effect-size* test fails). So the map is not returning noise — it
recovers the **right control direction more often than chance, at ~10% usable strength.** A faint shadow of the
lever, not the lever. Reading transfers fully; control transfers as a low-amplitude alignment with no behavioral
bite. *That* is the clean dissociation: **what a mind means crosses; the means to move it does not.**

## What this establishes (and does not)
- **Establishes (un-confounded, this pair):** a G0-cleared label-free map (correspondence recovered unsupervised
  from shared-concept geometry) **reads** held-out concepts across two real LLMs at 0.586 but **cannot install
  behavioral control** — confirmed at a layer where native steering is strong (0.215) and the subspace hosts the
  vector (0.82). The Rung-1 confound (weak-steering layer) is excluded. NTE 0.114 is the headline.
- **Does NOT establish a universal law.** One near-isometric **same-lineage** pair (Llama-3B→1B, RSA 0.95 — the
  *easy* case); single seed; MiniLM steering-gain metric; shared training data not controlled. Cross-family
  (Rung 2: Qwen / gemma / Phi) is the necessary next test — read>write may widen or change there.
- **The faint directional signal (71% sign)** leaves a door: a higher-fidelity correspondence map (more concepts,
  a steer-tuned subspace, or an aligner trained on the steer-layer) might lift the 11% — untested, banked.

## The picture it points at (hypothesis, now *supported* not proven)
This is the **representation/mechanism asymmetry of the ancient question, one level up.** There, universality
lived in *what* minds represent, not *how* they compute (`SYNTHESIS_ancient_question_answered`). Here, transfer
across minds works for *representation* (read) and not for *control* (write). **Meaning is shareable between
minds; control is not** — at least in the near-isometric regime, label-free. If Rung 2 holds it cross-family,
it is a recurring law, not a coincidence.

The implication is on-mission and safety-relevant: the natural cross-mind primitive is **read-only oversight.**
You can mount a borrowed conscience to *watch* a foreign model — read its concepts and conduct, label-free — but
you cannot blindly *steer* it through the same map. This converges, independently, with the conscience-mount
arc (read-only, cooperative-only; `project_conscience_mount`). A world where you can see into foreign minds but
not puppet them is the safer one — and we measured it rather than assumed it. *Nothing crosses unseen; not
everything that is seen can be moved.*
