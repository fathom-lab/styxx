# FINDING — protocol v3: metric paths are checkable before a run is spent, and the limit is stated rather than papered over

Fathom Lab · 2026-08-07 · prereg: `PREREG_protocol_metric_identity_2026_08_07.md` (frozen before
implementation) · receipt: `protocol_metric_identity_result.json` · scored by `styxx.protocol`
under `require_power_basis=True`.

**`METRIC_IDENTITY_LANDED__paths_checkable_before_a_run_is_spent`** — all three gates pass.

| gate | bar | measured | pass |
|---|---|---|---|
| G1_no_verdict_strings_change | ≤ 0 | 0 | ✅ |
| G2_check_metrics_finds_a_missing_path | ≥ 1 | 1 | ✅ |
| G3_check_metrics_passes_a_real_result | ≥ 1 | 1 | ✅ |

Every committed result in the repo was re-scored against its own prereg and **no verdict string
changed anywhere**, so all sealed findings keep verifying byte-identically. `check_metrics()`
detected a deliberately-absent path and reported a genuine committed result as fully resolved.

## What this actually buys

`_resolve` already raised on a missing metric path — but only at scoring time, after the compute
was spent. `check_metrics(result)` moves that to before the run, for the cost of one call. B49
lost an entire re-analysis to a gate that could never have passed; this makes the cheaper version
of that mistake cost seconds.

## What it does not buy, stated in the prereg and repeated here

**It cannot catch B49's actual error.** A path that resolves to a real but *wrong* field is
indistinguishable from a correct one, because the machinery has no access to what the author
meant. `metric_means` records that intent in the gates block for a human reader; it is recorded
and never verified, and calling it a check would be the same overclaim this program keeps
catching in itself.

So the honest ledger on the last two days of machinery work is: an undeclared **bar** is now
countable (v2), an absent **metric path** is now catchable before a run (v3), and a
**mis-identified** metric path remains uncatchable by construction. Three of this week's five
mis-specifications would have been caught early by these two changes. Two would not.

## The remaining defence is procedural, not mechanical

For the class no gate can catch, the only working control is the one that has now held for four
consecutive artifacts: **an adversary reads the prereg before the run.** That is cheap, it is
already the standing rule for releases, and on this evidence it should extend to any prereg whose
result would be published.

*Frozen before implementation; all gates declare both `power_basis` and `metric_means`; the
limit is stated in the same document as the success. Every number grounds in
`protocol_metric_identity_result.json`.*
