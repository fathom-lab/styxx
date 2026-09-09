# RESULT — the first real log, and the three places the ladder stops

Fathom Lab · 2026-09-09 · **A result about the system, not about the model.** The first end-to-end
run of styxx v8 against real weights: a real battery, a real fingerprint of gemma-2-2b-it, a real
Merkle log, a signed tree head, a mirror, and the Appendix D stranger check with negative controls.
Every command went through `python -m styxx.v8` as a subprocess, so what was exercised is the
command-line contract a stranger would use, not the Python API. Full transcripts with every exit
code in `transcript.json` and `transcript_stage2.json`; the log, the certs and the inputs are
committed beside them. Not sworn.

**The ladder does not complete.** It stops at the noise floor, for a reason no reading pass found.
That is the main content of this document.

## What ran

| | |
|---|---|
| subject | `google/gemma-2-2b-it`, revision `299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8`, bf16 |
| subject hashes | derived from the snapshot's own bytes by `subject_from_snapshot` (Appendix A.2), `weights_sha256` `ab61d3d62a218160…` |
| environment | RTX 4070 Laptop, driver 596.08, transformers 4.57.3, torch 2.5.1+cu121 |
| battery | 64 prompts across the five families, pool-v1, cert `sha256:f8dd7a819b692348…` |
| recipe | batch size 1, left padding, greedy, 16 new tokens, seed 7; chat template (591 bytes) and lockfile (6451 bytes) stored **in the clear** inside the cert |
| fingerprint | cert `sha256:d6816e4f6e807de9…`, 64 items scored, channels `exact`, `seqlp`, `topk`, tier white-box |
| **battery hash** | **`d3372370c68804a099c6e0098038382da23e1a4d24383e129d0576ae312a18c9`** |
| log | 2 entries, tree head at size 2, root `sha256:c5cacf40d7c8ca68…`, log id `sha256:ac3c9296e70dff31…` |
| GPU time | 201 seconds for five runs of 64 items |

Five runs at batch size 1 produced **one distinct battery hash**: the runs are bit-identical, which
is what today's two probes predicted for batch 1 and is the reason batch size is now a recipe field.

## What worked

- **The root battery is constructible.** A pool-v1 battery signed with an empty recipe appended as
  index 0 of an empty log. Until this morning that cert could not exist: the schema required every
  battery to name another battery, so the bottom of the ladder had nothing to stand on. The repair
  is real and this is the receipt.
- **Recipe-completeness is enforced at the door.** The fingerprint verb refused to run without a
  recipe, and the cert was refused until the chat template, the system prompt and the lockfile were
  present as bytes with hashes that re-derive. Nothing about the run is described rather than stored.
- **Appendix D steps 1 and 2 pass against the mirror.** `verify-cert` recomputed the id and checked
  the signature; `verify-sth` verified the tree head under the out-of-band pinned key.
- **Tamper detection works on real bytes.** One byte was flipped inside one entry file of a copied
  log, preserving its length. The mirror reported `verified: false` with three reasons: *entry 0: id
  does not recompute*, *entry 1: inclusion: proof root_hash is not the STH's root_hash*, and *sth: the
  entries do not reproduce the signed root at tree_size 2*. A tree head with a forged root was
  refused with *sig does not verify against the log public key*.

## Where it stops, and why

**F1 — blocker. The command line cannot create the cert its own append rule requires.** The verbs
are `battery`, `fingerprint`, `key`, `log`, `verify`. There is no `prereg` verb. Section 5.1 requires
the nuisance plan to be logged as a `prereg` cert of kind `noise-plan` **before** the floor runs, and
section 5.5's baseline rule refuses a second comparable fingerprint that does not reference either a
previous cert or that plan. So every floor run after the first was refused:

> `log append refused: baseline: a comparable fingerprint is at index 1; a fingerprint that starts a new baseline needs a previous ref`

The rule is correct — it is the amendment that stops an issuer silently moving its own baseline — and
the plan requirement is correct, because it is what stops an issuer choosing its nuisance set after
seeing the runs. Neither can be satisfied, because nothing can mint the plan. A `prereg` verb is
required before a noise floor can exist at all.

**F2 — the canonical fingerprint carries no floor.** `fingerprint --runs 5` wrote five run certs and
one canonical cert, but the canonical cert is byte-identical to run 0 (same id `d6816e4f6e80…`), has
no `noise_floor` key, and references only the battery. So `--runs 5` produced five independent
fingerprints, not a floor. This is F1's consequence rather than a separate defect, and the two should
be repaired together: the plan is minted, the runs reference it, and the canonical cert carries the
floor computed over them.

**F3 — the exit-code table does not cover log verification.** The tampered mirror exited 4 and the
forged tree head exited 4. Section 6's table defines exit codes for `verify` verdicts and says
nothing about what `log mirror` or `log verify-sth` return on a bad log. The behaviour looks right —
4 is the invalid-cert code and a non-recomputing id is exactly that — but it is undocumented, so this
run's negative controls were written against the wrong expectation. The table should be extended, or
section 8 should state its own codes.

## Four findings about the specification, found by using it

Each of these cost a failed run, and none was found by two full review passes over the text.

1. **Appendix A.1 is wrong about material hashes.** It says hashes are written `sha256:<64 hex>` and
   names two exceptions. Material hashes are a third: `cert.material_hash` produces **bare hex, no
   prefix**, and it must, because the refs rule matches on the `sha256:` grammar and would otherwise
   demand that a chat-template hash resolve to a logged cert. A reader following A.1 writes the prefix
   and every cert they build is refused with *embedded id … at recipe/chat_template_sha256 is not in
   refs*. The grammar carries the distinction between "an identifier of a cert" and "a hash of some
   bytes", and A.1 must say so.
2. **Section 4.5 names the wrong field.** It says the pool file carries `prompt`; the cert body and the
   implementation both use `prompt_text`. The code is self-consistent, so the spec sentence is the
   defect.
3. **`battery pool` and `battery fixed` need a subject.** A pool battery is model-agnostic by
   construction, yet the envelope requires a subject, so the root of the ladder is stamped with
   whichever model happened to be at hand. Either the envelope should permit an empty subject for a
   root battery, or the spec should say why a model-agnostic battery names a model.
4. **Item ids must be supplied, not derived.** The CLI refuses a source file whose items lack an
   `item_id` rather than computing it from the prompt. That is defensible for a content address — a
   derived id would hide a mismatch between what an author meant and what the log checks — but it is
   nowhere in the spec, and a reader will expect derivation.

## Limits

One model, one machine, one battery of 64, one session, 2 entries. No floor was measured, so no
verdict was produced and `verify --ref` was never exercised against a floor. Appendix D steps 3 to 5
were not attempted: step 3 needs the weights, which the operator has and a stranger may not, and step
5 needs a challenge, which needs a floor. Nothing here measures the model, and the battery hash above
identifies a behaviour on 64 prompts, nothing more.
