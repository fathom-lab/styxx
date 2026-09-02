# RESULT — the efficiency control, preregistered: CAPACITY_IN_DISGUISE

Fathom Lab · 2026-09-02 · Frozen by `PREREG_efficiency_control_2026_09_02.md` (committed before
the run). Runner: `run_efficiency_control.py`, which imports the arc's `run_entrain_rich.py`
verbatim. Receipt: `efficiency_control_result.json`, scored through `styxx.protocol`. Device: CPU,
three seeds, 1500 steps. Every number below is sworn to the receipt at commit `3b14f4208e30`.
<sworn r="path:papers/frequency-resonance/efficiency_control_result.json#/verdict" k="quote">The frozen verdict is `CAPACITY_IN_DISGUISE`.</sworn>

## The gates, in the order the preregistration reads them

**Plumbing.** <sworn r="path:papers/frequency-resonance/efficiency_control_result.json#/metrics/anchor_max_abs_dev" k="numeric">The anchors re-run on CPU landed within 0.0227 of the committed GPU receipt</sworn>,
under the frozen 0.03: this is the same experiment.

**Positive control.** <sworn r="path:papers/frequency-resonance/efficiency_control_result.json#/metrics/oracle_minus_static_8" k="numeric">The oracle beat static by 0.162 at the primary width</sworn>,
over the arc's standing 0.10 bar: the adaptation prize is present, as it was on the receipt.

**The control.** <sworn r="path:papers/frequency-resonance/efficiency_control_result.json#/params/8/static_matched" k="numeric">The matched static bank has 5850 parameters</sworn>, at least
<sworn r="path:papers/frequency-resonance/efficiency_control_result.json#/params/8/rich" k="numeric">RICH's 5652</sworn>, and <sworn r="path:papers/frequency-resonance/efficiency_control_result.json#/metrics/static_matched_minus_rich_8" k="numeric">beats RICH by 0.1959</sworn>
on drift accuracy. The bar for CAPACITY_IN_DISGUISE was zero.

## The numbers at the primary width

| arm | width | params | drift accuracy |
|---|---|---|---|
| STATIC | eight | 2052 | 0.4831 |
| RICH (adaptive frequency) | eight | 5652 | 0.5560 |
| ORACLE (locked to the true period) | eight | 2052 | 0.6451 |
| STATIC, parameter-matched | twenty-six | 5850 | 0.7519 |

Read as spans: <sworn r="path:papers/frequency-resonance/efficiency_control_result.json#/drift/8/rich" k="numeric">RICH scored 0.5560</sworn>;
<sworn r="path:papers/frequency-resonance/efficiency_control_result.json#/drift/8/oracle" k="numeric">the oracle scored 0.6451</sworn>;
<sworn r="path:papers/frequency-resonance/efficiency_control_result.json#/drift/8/static_matched" k="numeric">the matched static bank scored 0.7519</sworn>.
The static bank with RICH's parameter budget does not merely match the adaptive model — it beats
the oracle that adapts perfectly, by a wide margin. Adaptation at eight modes is worth less than
eighteen more modes that never adapt.

## The secondary width, reported not gated

At D=4, <sworn r="path:papers/frequency-resonance/efficiency_control_result.json#/params/4/static_matched" k="numeric">the matched static bank has 3375 parameters</sworn> against
<sworn r="path:papers/frequency-resonance/efficiency_control_result.json#/params/4/rich" k="numeric">RICH's 3184</sworn>, and
<sworn r="path:papers/frequency-resonance/efficiency_control_result.json#/metrics/reported_static_matched_minus_rich_4" k="numeric">beats RICH by 0.3049</sworn>
(<sworn r="path:papers/frequency-resonance/efficiency_control_result.json#/drift/4/static_matched" k="numeric">static 0.6648</sworn> against
<sworn r="path:papers/frequency-resonance/efficiency_control_result.json#/drift/4/rich" k="numeric">RICH 0.3599</sworn>). The "clean mode-scarcity win" the arc reported
at D=4 was the same parameters spent worse.

## What this closes, and what it does not

This is the frozen-rule confirmation of `RESULT_efficiency_control_from_receipts_2026_09_02.md`,
which reached the same reading from committed receipts without a rule and said so; under the
preregistration this verdict outranks it. It closes the adaptive-frequency line's last positive:
across `RESULT_entrainment`, `RESULT_entrain_rich`, `RESULT_entrain_harm` and
`RESULT_scarcity_scale`, no learned frequency detector has earned the parameters it costs, at any
width or task the arc measured. The synthesis's line for nested coupling — capacity comes from more
modes, not cleverer coupling — now covers adaptation.

It does not touch the arc's headline, which stands: the phase clamp showed oscillation itself is
causally load-bearing (`RESULT_pmnist_ablation`, +0.312), and the oracle's prize is real and grows
with width. What died is the claim that a *learned* adapter captures that prize efficiently. Scope:
toy (L=96, D≤26, integer symbols, three seeds, no interval, one task family). Nothing here is about
real LinOSS or Mamba checkpoints; the profiler now shipped in `styxx/resonance.py` is how that
question gets asked next.

---

*The control the arc owed itself, paid: the parameters spent on listening for the rhythm would have
bought more as more strings. The rhythm is real. The listener was not worth its cost.*
