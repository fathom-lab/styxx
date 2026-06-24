# FINDING — the silence/validity tradeoff: you cannot FREE-GENERATE a silent deception corpus

**2026-06-24. Pre-registered `PREREG_silent_grounding_pillar_2026_06_24.md` (`1d979e6`, frozen before data).
Offline, local-GPU, NO frontier key.** Reproduce: `python scripts/silent_grounding_pillar.py --generate
--model {qwen,llama}` then the Gate-1 check.

## Goal
Build the pillar of "self-report is not oversight": a SILENT construct where a text classifier is at chance
but an activation probe holds. The deception version: confident honest vs confident deceptive assertions,
surface-matched so bag-of-words can't tell them apart.

## Result — Gate 1 fails, and the FAILURE is the finding
The two gates (silence: BoW ≤ 0.62; validity: honest↔deceptive bge cosine ≤ 0.85) are in **direct conflict**
for free-generated text:

| generator | BoW length-matched (silence) | bge honest↔deceptive (validity) | outcome |
|---|---|---|---|
| Qwen2.5-3B | **0.713** (NOT silent) | 0.848 (valid: real fabrication) | a real lie changes the facts → different words → BoW catches it |
| Llama-3.2-3B | 0.610 (silent) | **0.868** (DEGENERATE) | lexically invisible only *because it barely lied* |

**Neither model satisfies both.** This is intrinsic, not a tuning problem: a deception that changes the
asserted content (genuinely false claim) is necessarily lexically detectable; a deception that is lexically
invisible has not changed the claim. **Free generation cannot produce silent-AND-valid deception.**

## What the pillar actually requires (the precise next construct)
The only way to break the tradeoff is a **controlled minimal-pair** construct: the false statement asserts a
DIFFERENT fact (valid) but with IDENTICAL surface form except ONE swapped, distribution-balanced token —
e.g. "The capital of France is Paris" (T) vs "The capital of France is Lyon" (F), where "Lyon" is the correct
answer to some other item so BoW cannot use the token to decide truth. Free generation rewrites the whole
sentence and destroys the surface match; you need a curated/template true-false statement set (the
Azaria-Mitchell 2023 truthfulness-probe setup). The minimal-edit route on THIS corpus was infeasible — only
8 of 100 honest answers carried a corruptible year token.

## Honest implications
- **The pillar's bottleneck is the CONSTRUCT, not the probe.** "Grounding ≫ text on a silent construct" is
  hard precisely because building a genuinely silent task is hard — which is also why it would be a real
  contribution and why it must not be faked (BoW 0.71 is not silent; we did not proceed to a pillar claim).
- **Scope caution for the program's prior "silent caving" result** (`project_closed_model_frontier`:
  text→chance, grounding→1.0): that claim should be re-checked against an explicit BoW-silence gate. If its
  "text→chance" was a content-classifier at chance on a genuinely surface-matched corpus, it stands; if the
  silence was the Llama-style degeneracy (classes too similar because the manipulation barely happened), it
  needs the same scrutiny. This finding raises the bar for any "text-silent" claim in the suite.
- **What styxx would actually contribute if it builds the curated construct:** not the phenomenon
  (activations encode truth — known since Azaria-Mitchell), but the *silence-gate methodology* + full control
  battery (shuffle / grouped-CV / PCA / length-invariance) + the oversight framing. The genuinely NOVEL,
  unclaimed territory is GENERATION-TIME deceptive intent (the speaker's own state, not a reader detecting
  false content) and CROSS-MIND probing — both harder, multi-session.

## Honest scope
2 local 3B models, n=200 each, the deception QUESTIONS. Gates pre-registered; the tradeoff is mechanical and
replicates across both models in opposite directions, which is what makes it a tradeoff rather than a
one-model artifact. 7th self-caught temper of the session — here, the experimental *design* was the thing
that didn't survive contact with the rigor.
