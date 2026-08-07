"""styxx.apparatus — QUARANTINED, DO NOT SHIP. Failed its pre-release adversarial audit.

**This module is not part of any release and must not be imported for real work.** It is kept in
the tree because deleting a failed instrument hides the evidence; the audit that killed it is the
most useful thing about it.

VERDICT 2026-08-06: DO NOT SHIP. Measured balanced accuracy **0.55** on an artifact-vs-genuine
panel — a coin flip — while the verdict *string* (`SURVIVES_APPARATUS_COUNTERFACTUALS`) carries
the full rhetorical weight of a passed audit. Four failures are certain-by-construction, not
probabilistic:

1. **It certifies the artifact from its own motivating paper.** Two logs sharing only a stall
   clock — no shared world content, verified by construction — returned SURVIVES on 8/8 seeds at
   10–12.6x the random-frame null, *above* the ratio this lab published as an emphatic real
   finding. Cause: ``time_reverse`` is itself a permutation, i.e. exactly the class of operation
   the motivating paper argues cannot detect this. The module's only default world-destroying
   axis is the one its own thesis rules out.
2. **Any time-symmetric genuine signal is flagged as contamination**, 10/10. A periodic stimulus
   (block-design fMRI, flicker, metronome) lives in a reversal-invariant Fourier subspace, so
   reversal does not destroy alignment and the docstring's promise is false there.
3. **``obs <= 0`` returns ``survives=True`` unconditionally** — and the docstring instructs users
   to negate statistics that run the other way, routing negated p-values straight into that
   region. Following the documentation disables the audit.
4. **``shuffle_within_density`` is the identity when counts are unique**, so passing
   high-cardinality counts forces RECORDER_CONTAMINATED on genuine findings. The guard checks
   ``len(unique) > 1``, which *passes* in the worst case. Crossover to auto-flagging is ~25-30%
   singleton bins — ordinary telemetry.

Also: ``retained_fraction`` ignores the statistic's floor, so the threshold silently becomes
"is the finding more than twice its own null" (a power question, not an apparatus question); no
setting of ``retain`` both catches the stall artifact and clears genuine findings; the seed test
is arithmetically vacuous for deterministic statistics and swallows unrelated TypeErrors; the
one-stream path has no world-destroying counterfactual at all; a non-finite counterfactual reads
as a pass.

**And the process lesson, which is the point.** An earlier draft of this docstring claimed a
direction bug was "caught in this module's own demo before release" — offered as evidence of
rigour. The auditor's reply stands: *a demo that shows two hand-picked cases is not an
adversarial pass, and treating it as one is how the previous three modules shipped.* That is
exactly right. Self-demonstration is not adversarial testing, and the difference is this module.

Minimum bar to ship, from the audit: obs<=0 -> INVALID; frozen-fraction guard on the density
shuffle; a time-symmetry pre-check emitting UNTESTABLE__time_symmetric; floor-corrected
retention; a freshness/shared-mask surrogate to close the stall class; demote amplitude-stripping
from verdict to decomposition; non-finite counterfactual -> INVALID; and a test file.

Full class description: ``papers/SYNTHESIS_recorder_contamination_2026_08_06.md``.
"""

raise ImportError(
    "styxx.apparatus is QUARANTINED: it failed its pre-release adversarial audit at 0.55 "
    "balanced accuracy and certifies the artifact from its own motivating paper. See the module "
    "docstring for the verdict. It is retained in the tree as evidence, not as a tool."
)
