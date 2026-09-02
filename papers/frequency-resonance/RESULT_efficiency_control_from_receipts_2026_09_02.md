# RESULT — the efficiency control, answered from the receipts: adaptive frequency was capacity in disguise

> **Back-pointer, added 2026-09-02.** The preregistered run confirmed this reading under a frozen rule, CAPACITY_IN_DISGUISE, and its verdict outranks this document's — see `RESULT_efficiency_control_2026_09_02.md`. The text below is unchanged.

Fathom Lab · 2026-09-02 · **A recombination of committed evidence, not a new measurement, and not
preregistered.** `RESULT_entrain_rich_2026_07_23.md` named "a param-matched wider static bank" as
the honest next control for its +0.085 (D=8) and +0.129 (D=4) adaptive-frequency advantages. That
control turns out to be already in the tree: `entrainment_result.json` carries a STATIC arm at D=16 on
the identical task (L=96, three segments, periods [3,12], 1500 steps, seeds 0/1/2), and its STATIC
arms at D=4 and D=8 reproduce `entrain_rich_result.json`'s to the last digit, so the two receipts
are one experiment. No rule was frozen before these numbers were read — the author read the receipts
and then wrote this — which is why this is a RESULT in the join precedent
(`RESULT_obligation_predicts_claimhood_2026_08_30.md`) and not a preregistered verdict. The
preregistration that would make it one is `PREREG_efficiency_control_2026_09_02.md`, frozen after
this document and runnable by anyone with a GPU or a patient CPU.

## The comparison

The question the arc left open: do the parameters RICH spends on its frequency detector buy more as
extra static modes? Every number below is sworn to the receipt that holds it.

**D=4.** <sworn r="path:papers/frequency-resonance/entrain_rich_result.json#/params/4/rich" k="numeric">RICH has 3184 parameters</sworn> and
<sworn r="path:papers/frequency-resonance/entrain_rich_result.json#/drift/4/rich" k="numeric">scores 0.3752 on the drifting task</sworn>.
<sworn r="path:papers/frequency-resonance/entrainment_result.json#/params/8/static" k="numeric">a static bank one width up has 2052 parameters</sworn>, fewer, and
<sworn r="path:papers/frequency-resonance/entrainment_result.json#/drift/8/static" k="numeric">scores 0.4604</sworn>. The "GREENLIGHT-level" D=4 win —
<sworn r="path:papers/frequency-resonance/entrain_rich_result.json#/gate/rich_advantage_by_D/4" k="numeric">an advantage of 0.1286 over static at the same width</sworn> — is
beaten by a smaller static bank.

**D=8, the preregistered primary.** <sworn r="path:papers/frequency-resonance/entrain_rich_result.json#/params/8/rich" k="numeric">RICH has 5652 parameters</sworn>
and <sworn r="path:papers/frequency-resonance/entrain_rich_result.json#/drift/8/rich" k="numeric">scores 0.5451</sworn>. <sworn r="path:papers/frequency-resonance/entrainment_result.json#/params/16/static" k="numeric">a static bank at twice the width has 3580 parameters</sworn>,
fewer, and <sworn r="path:papers/frequency-resonance/entrainment_result.json#/drift/16/static" k="numeric">scores 0.6781</sworn>. The WEAK verdict's
<sworn r="path:papers/frequency-resonance/entrain_rich_result.json#/gate/rich_minus_static" k="numeric">advantage of 0.0847</sworn> is not merely matched by a
parameter-matched static bank; it is exceeded by a static bank with sixty-three percent of the
parameters.

**The oracle, too.** <sworn r="path:papers/frequency-resonance/entrain_rich_result.json#/drift/8/oracle" k="numeric">The oracle at the primary width — a diverse bank locked to the true drifting period — scores 0.6307</sworn>,
below static D=16's 0.6781. A bank that adapts perfectly at eight modes loses to a bank that does not
adapt at all at sixteen. <sworn r="path:papers/frequency-resonance/entrainment_result.json#/drift/16/oracle" k="numeric">The oracle at twice the width scores 0.8950</sworn>, so the
adaptation prize is real and grows with width; it is the *learned* detector that never earns its
parameters, at any width measured.

Derived, not sworn: the static model's parameter count is `2D² + 143D + 780` (checked against the
receipts at D=4, 8, 16), so the smallest static width with at least RICH-at-D=8's parameter count is
D=26, at 5850. That arm was never run. It does not need to be for the conclusion above, because a
narrower static bank already wins; it is the confirmatory arm the preregistration names.

## Reading

The adaptive-frequency line of the arc ended in three verdicts: KILL (single-projection detector),
WEAK at D=8 and a clean win at D=4 (windowed conv), and ABSTAIN with a falsified scaling curve on the
harder task. This control closes the one residual positive. **Spent as modes rather than as a
detector, the same parameters buy more, at every width the arc measured.** The synthesis's line for
nested coupling — capacity comes from more modes, not cleverer coupling — now covers adaptation as
well, on this task family. What remains live is the oracle's prize, which no learned detector has
reached, and the profiler, which measures reliance rather than earning it.

## What this does not say

Nothing about scale beyond D=16 or beyond L=96; nothing about real LinOSS or Mamba checkpoints;
nothing about the harder task, where the receipts hold no static arm wider than the RICH arm it
would be compared with. The comparison is on seed-averaged accuracy with three seeds and no
interval. And, once more: the rule was not frozen before the numbers were read. The preregistration
beside this document is what fixes that, and its verdict, when someone runs it, outranks this one.

---

*The control the arc owed itself was already paid for. The parameters the detector spent would have
bought more as modes — which is the arc's own finding, said one more time, about the one lever it
had hoped was different.*
