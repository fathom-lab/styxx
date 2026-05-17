# styxx Research Integrity Protocol

The asset is not a result — it is a research line that will not lie to
itself. This codifies the practice that produced four committed
preregistered negatives, a caught circular oracle, and a clean
permanent record in one session. It is meant to be reused, by humans
and by spawned agents, verbatim.

## The rules (non-negotiable)

1. **Preregister in the script.** Hypotheses + exact numeric thresholds
   go in the script docstring *before* the run. They are never moved
   post-hoc. "0.617 < 0.70" is a kill even when you have a story.

2. **Commit either way.** A negative is the finding. Commit it to main
   with the numbers. Do not re-run hunting a pass. If a result looks
   too clean, replicate it at higher resolution before believing it —
   a 5-point "law" died at 12 points this session.

3. **Reject oracles explicitly.** A candidate score that shares a term
   with its own label (e.g. `register × (1−correct)` vs a label that
   *is* `… AND ¬correct`) is tautological, not evidence. Name it,
   flag it, reject it in the writeup — never report its AUC as a win.

4. **Offline-validate every labeler that gates a result.** A
   deterministic fixture (both classes, the tricky cases) + regression
   against known-labelled data, no network. Then suite-protect it
   (`tests/test_labeling.py`) so the behavior cannot drift unnoticed.
   The suite test caught a real false-positive the day it was written.

5. **Don't hand-roll a divergent labeler.** Cross-run measurement must
   share one validated labeler. A second, differently-tuned regex
   reintroduces the exact confound under test (this is why the
   OpenAI-tuned refusal regex produced a false cross-vendor crack).

6. **Parallel-half handoff.** The methodological instrument is built,
   offline-validated, and pushed *first*; the consumer run pulls and
   uses it. If it is absent on pull: halt and wait. Never substitute.

7. **Verify the fix is live where it runs.** Source-edited ≠ deployed.
   `git push` printing success ≠ landed (check `ls-remote`). A fix in
   the tree is inert if the runtime imports a stale wheel — confirm
   editable-install / authoritative remote, not "exit 0".

8. **Consequential actions gate on explicit human confirmation.** DOI
   deposit, PyPI publish, public force-push: prepare to one-command-
   ready, then stop and present the diff + the honest claim. The
   publishing bar reserves a DOI for an extraordinary, replicated
   discovery *with* a tool — never a receipt or an addendum.

9. **The record matches the git history.** Memory, docs, README,
   CHANGELOG state what is true *now*, including the construct
   ceilings and any corrected over-claims. Fix over-claims promptly.
   Honesty under audit is the product; a stale claim erodes it.

## Why this works

Every one of these was earned by a failure it would have prevented if
it had been a rule first. The discipline is not caution for its own
sake — it is the only thing that makes a solo / agent-built research
line credible in a field saturated with overclaiming. A negative,
committed and explained, is worth more than a positive that cannot
survive its own audit.
