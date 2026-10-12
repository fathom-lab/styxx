# PREREG — permuted-MNIST headline robustness: held-out permutations, new seeds, a 3× budget arm and the untied control under one frozen bar — 2026-10-11

**FROZEN before confirmatory data.** Runner: `run_pmnist_robustness.py` (committed after this file; it imports
`run_pmnist_ablation.py` and `run_pmnist_untied.py` verbatim and edits neither). Receipt:
`pmnist_robustness_result.json`, scored through `styxx.protocol.Experiment` from the gates block in §5. Seal: the
sha256 of the **git blob** of this file, by the rule of `papers/checksum/SEALS_2026_09_13.md`
(`git show <commit>:<path> | sha256sum`, never the working copy), recorded in that table in the commit after this
one. Anchor: at the operator's discretion (§13); the derived permutations of §3 are drawn from the anchor's beacon.

**Written without knowledge of the sealed untied receipt.** `pmnist_untied_result.json` (run 2026-10-11, sealed,
sha256 `8183683f…` per `SEALED_README.md`) and its run log were not opened by anyone who wrote this document or its
design (`DESIGN.md`, `ATTACK.md` of the 2026-10-11 frequency read). Every bar below was frozen without its numbers;
§7 states what this design predicts under each verdict that receipt can carry, and nothing in the gates block
depends on which one it is. The lab's order of operations (DESIGN decision 6, kept): this file is committed and
sealed, and only then is the sealed receipt read — by another track, after `python -m styxx.clock verify
papers/charon/anchors.jsonl` prints ANCHORED for its line. The re-run of seeds 0 and 1 on permutation 1234 (§3) is
therefore *preregistered*, not blind: whoever reads the receipt after the seal learns two of this design's
sixty-five numbers in advance, and those two enter no gated metric (§4).

This is the design of `DESIGN.md` with the red team's fixes (`ATTACK.md` §4, "the strongest version that
survives") applied: LF-normalised blob hashes frozen and asserted (F1); scoring over the whole receipt and
`check_metrics()` before any night (M1); the headline's own seeds and permutation kept out of every robustness gate
(M2); the attribution gates' precision stated honestly and the recovery interval made the number the sentence
prints (M3); degeneracy counted by arm (M4); the budget gate kept and the plateau measured within the 12,000-step
curves, reported (M5); the pooled interval and the between-permutation variance share printed beside the
permutation interval (M6); the architecture named by its code (M7); verdict sentences carrying their n (M8); the
beacon draw as the default derivation of the held-out permutations (M9); the GPU guard refusing the measured case,
the schedule cited to the measured per-run time, the data pinned by hash, the three-arm RNG-sharing assert, the
non-finite rule, and no assertion that can lose a night (m1–m9).

---

## 1. What this rests on, with receipts

- The headline: FREE 0.9195 (0.9240 / 0.9150) vs CLAMPED 0.6073 (0.5529 / 0.6617), gap +0.3122, params 101,002 /
  100,810 — `pmnist_ablation_result.json#/result/free_minus_clamped`, `#/test_acc/free`, `#/test_acc/clamped`,
  `#/config/params`; `RESULT_pmnist_ablation_2026_07_23.md`.
- Its scope as the record states it (`STATE_OF_THE_ARC_v2.md` §3 item 12; `CRITIC.md` M3): two seeds
  (`#/config/seeds` = [0, 1]), one permutation (`#/config/perm_seed` = 1234), one width (`#/config/H` = 64),
  `#/config/steps` = 4000 with no convergence check and no longer-budget arm; CLAMPED seed spread 0.109 on two
  seeds; "the decay model plateaus while the oscillatory one keeps improving" (`RESULT_pmnist_ablation_2026_07_23.md`,
  "The sharpening") has no receipt — the JSON holds final `test_acc` only; no positive-control arm, no interval.
- The confound control on this benchmark: `PREREG_pmnist_untied_2026_09_02.md`, gates `anchor_max_abs_dev <= 0.03`,
  `gap_free_minus_clamped >= 0.15`, `free_minus_real2 <= 0.03` (TIMESCALE_DIVERSITY), `real2_minus_clamped <= 0.03`
  (ROTATION_LOAD_BEARING), else PARTIAL; sealed as `papers/checksum/SEALS_2026_09_13.md` row 2 (blob `507ef9ac…`,
  frozen at `4dba3a72`); run 2026-10-11; receipt sealed and unread here. On ordered copy the same control recovered
  0.0 of the gap — `papers/rhythm-rescue/untied_control_result.json#/metrics/recovery_fraction`.
- What the sMNIST comparison froze: `smnist_gap` 0.0408 and WIDENS iff `gap > 0.0408 + 0.005`
  (`PREREG_pmnist_ablation_2026_07_23.md`, "Frozen verdict"; `pmnist_ablation_result.json#/result/smnist_gap`).
- The measured per-run time: six 4000-step trainings with four full-test evaluations each ran 17:32:47 → 18:54:26 on
  this RTX 4070 Laptop on 2026-10-11 (`sealed/SEALED_README.md`), **13.6 minutes per run**. The operator's brief said
  18; 18 is kept as the refusal threshold (§8). The `--smoke` re-measures.
- The data: `_mnist.npz`, 11,507,660 bytes, sha256
  `98d685052d7a6ab8b6982491df8ec3eb366a252fd5415aa58f878bbe6510d4e8` (the 2026-07-23 bytes; the pin in
  `sealed/SEALED_README.md`), gitignored (`.gitignore`, `papers/frequency-resonance/_mnist.npz`); copied beside the
  runner before a run, never downloaded by it.
- The code this design reuses, by LF-normalised git blob (`git show origin/main:<path> | sha256sum` at `f469f25b`):

  | file | blob sha256 | what the runner takes from it |
  |---|---|---|
  | `run_pmnist_ablation.py` | `c8d7cdfbab3fabce94fc541e92c46947cf1909b13ad6d4994baa42d52a305423` | `CLRU`, `Block`, `SSMClassifier`, `load_data` (`torch.randperm(784, generator.manual_seed(PERM_SEED))`), `train`, `test_acc`, `nparams`, `redteam`, `lin_scan`, `seq_scan` |
  | `run_pmnist_untied.py` | `caf276da27e0b6ad408adadc019d7f39d07747ee928fbdf9e4d636a645de94bf` | `RealBank`, `RealBlock`, `RealClassifier`, `train_real2`, `nparams`, `redteam` |

  The working copy of either file on a Windows checkout may carry CRLF and hash differently while executing the
  same code (`SEALS_2026_09_13.md` names the trap; measured 2026-10-11: `run_pmnist_untied.py` working copy
  `58396a4f…`, 153 CR bytes). The runner therefore hashes `read_bytes().replace(b"\r\n", b"\n")`, which equals the
  blob for an LF-stored file, and refuses before data if either hash differs from the table above.

Rules carried from the charter and the record: correctness is not coverage (a robust gap inside this architecture
says nothing about what the task needs — `necessity_attention_result.json#/arms/transformer/mean_kcap` = 15.33 at
matched parameters stands); a null is meaningless without a firing positive control (§6); a forced-choice readout
needs a degeneracy guard (§6); every bar is frozen before data and none moves after; a gate's text and its code say
the same thing (§9.4 lists what `plumbing_ok` checks, and nothing else is claimed for it).

---

## 2. Questions — one falsifiable sentence per axis

**Q-gap (robustness).** Averaged over five pixel permutations that no one chose (drawn from a block hash after this
file is sealed, §3) with three training seeds each, does FREE − CLAMPED at matched parameters stay at or above 0.15
with a 95% interval whose lower end stays above 0.05 — the committed WIDENS bar against sMNIST — or was +0.3122 a
reading of one permutation and two seeds?

**Q-headline.** On the permutation where the effect was measured (1234), on three seeds the receipt never ran
(2, 3, 4), is the gap at or above 0.15?

**Q-budget.** With three times the frozen training budget (12,000 steps, same optimizer, cosine schedule
re-stretched to the budget), does the gap on permutation 1234 stay at or above 0.15? Reported beside it: how much
the decay model gains from 4000 to 12,000 steps, between schedules and within one curve — the receipt the
"plateaus" sentence lacks.

**Q-attribution.** On the five held-out permutations, does a real bank with FREE's parameters and twice its
independent timescales (REAL2) land within 0.03 of FREE (diversity), within 0.03 of CLAMPED (rotation), or between —
under the bars the sealed untied PREREG froze, now on 15 runs per arm instead of two, with the recovery fraction and
its interval printed?

---

## 3. Arms, factors and the schedule of runs (frozen)

Architecture, data, optimizer, batch, learning rate, weight decay, gradient clipping, cosine schedule, test set and
evaluation are `run_pmnist_ablation.py`'s, unchanged: a three-block complex-diagonal linear recurrence (LRU-style;
`CLRU` in `run_pmnist_ablation.py`, λ = |λ|·e^{iθ}, |λ| ∈ (0.9, 0.999) at init, γ-normalised input, parallel scan),
H = 64, D_SSM = 64, AdamW lr 3e-3 wd 0.01, batch 64, gradient norm clipped at 1.0, `CosineAnnealingLR` over the
budget, full 10,000-image permuted test set.

| arm | model | params (asserted before data) | what it is |
|---|---|---|---|
| FREE | `R.SSMClassifier(True)` | 101,002 | θ learnable: rotation on |
| CLAMPED | `R.SSMClassifier(False)` | 100,810 (within 0.2% of FREE) | θ ≡ 0: rotation off, magnitudes tied |
| REAL2 | `U.RealClassifier()` | 101,002 (equal to FREE, `U.redteam`) | 2·D_SSM real modes, 2·D_SSM independent magnitudes, no rotation |

**Permutations: the anchor and five held-out.** The anchor is `1234` (`load_data`'s `PERM_SEED`, the receipt's).
The five held-out permutation seeds are **derived from this PREREG's own anchor beacon** — the block hash of the slot
of the earliest confirmed memo `styxx sealed-prereg <blob sha256 of this file>` from the creator wallet, as
`python -m styxx.clock verify papers/charon/anchors.jsonl` prints it (`beacon=`, 64 lowercase hex; `styxx.clock`'s
earliest-memo rule decides which slot). Derivation, frozen now, before the anchor exists:

    seed_k = int(sha256(f"{beacon_hex}:perm:{k}").hexdigest()[:8], 16)      k = 1, 2, 3, …

the top 32 bits of the digest, taken in order of k, skipping any value equal to 1234 or to a value already taken,
until five are held (P1 … P5). `python run_pmnist_robustness.py --derive-perms --beacon <hex>` prints the five; the
runner asserts the same list before any derived-permutation run and records the beacon and the list in every leaf
and in the aggregate (`config.perm_derivation` = `"beacon"`). Stages C and D (§8) **refuse to start** until the
beacon is supplied, either as `--beacon <hex>` copied from a `styxx.clock verify` line reading ANCHORED for this
file's digest, or as `--anchors papers/charon/anchors.jsonl`, which the runner verifies itself and from which it
takes the beacon of that line. This is the one device in the design that orders anything after the seal: the 45
held-out-permutation trainings cannot precede the block that names their permutations.

**Labelled fallback.** If the operator declines the anchor (§13 decision 3), the same derivation runs on the tag
`PREREG_pmnist_robustness_2026_10_11` in place of the beacon hex, giving `3584514041`, `27380109`, `2994719670`,
`3317084913`, `2856893334` (k = 1..5; none equals 1234 and none repeats). A run under the fallback is started with
`--fallback-perms`, every leaf and the aggregate carry `config.perm_derivation` = `"sha256-of-tag fallback"`, and
the RESULT says in its opening lines that the permutations were typed by the designer's tag and order nothing after
the seal. The bars are the same under either derivation; the two derivations are never mixed within one aggregate
(mixed leaves read `plumbing_ok` = 0).

**Seeds, and the roles they play (M2).** Permutation 1234: seeds {0, 1, 2, 3, 4} for all three arms. Seeds 0 and 1
are the receipt's and the sealed run's seeds, re-run as the **determinism re-run**: they feed `anchor_max_abs_dev`
(reported) and **no gated metric**. Seeds 2, 3, 4 are the **new-seed anchor test** (`anchor_gap_newseed`, gated). Each
held-out permutation: seeds {0, 1, 2} for all three arms. The same seed gives FREE, CLAMPED and REAL2 identical
non-SSM initial tensors and identical batch order: θ is drawn before the `if free` branch and `RealBank` consumes the
same number of draws, so after `torch.manual_seed(s)` and construction the three models leave identical RNG state
(verified on CPU at seeds 0–4, `ATTACK.md` m4; asserted before data, §9.4).

**Budget arm (LONG).** Permutation 1234, 12,000 steps: FREE seeds {0, 1}; CLAMPED seeds {0, 1, 2}. CLAMPED gets the
extra seed because its seed spread is the large one (0.109 on two seeds in the receipt; FREE's is 0.009).
`CosineAnnealingLR(opt, 12000)` is a re-stretched schedule, not a continuation of the 4000-step run: at step 4000 of
the 12,000-step run the learning rate is at 0.75 of peak. "CLAMPED at 12,000 steps minus CLAMPED at 4000 steps"
therefore compares two schedules; the within-curve quantity acc(8000) − acc(4000) on the 12,000-step runs (§4) is
the one that asks whether the decay model stopped improving while its learning rate was still 0.25–0.75 of peak.

**Curves.** Every run stores full-test accuracy every 500 steps (8 points at 4000, 24 at 12,000), captured without
editing the committed training loop (§9.3). Evaluation consumes no RNG, so a run's trajectory at a given seed is
identical whether it is evaluated 4 or 8 times; the anchor seeds 0 and 1 should land near the receipt up to device
nondeterminism, which is reported, not gated (§6).

**Counts.** 4000-step runs: 3 arms × 5 seeds (anchor) + 5 permutations × 3 arms × 3 seeds = 15 + 45 = **60**.
12,000-step runs: **5**. Total **65**. Per arm at 4000 steps: 15 held-out runs over five permutations, 5 on the
anchor. Optional width stage: §13 decision 4 (12 more runs, reported only, not in any gate).

Within-model θ→0 reliance of each trained FREE model (`clamp_theta` on each block) is recorded per run because it is
free, and is **not** evidence for the between-model gap (`CRITIC.md` M3: a destructive perturbation with no
matched-norm control). No matched-norm control is added here; that is a separate design
(`PLAN_next_experiment.md` §1.3).

---

## 4. Metrics

Let `acc[p][arm][s]` be the final full-test accuracy of the 4000-step run at permutation p, arm and seed s.
`degenerate` = final accuracy ≤ 0.15 or a non-finite loss on any logged line (§6). `good(p, arm)` = the
non-degenerate 4000-step runs at (p, arm); `acc(p, arm)` = their mean. For the anchor permutation,
`acc_new(arm)` = the mean over `good(1234, arm)` restricted to seeds {2, 3, 4}. The **held-out set** is P1 … P5.

**Gated (all under `result.` in the aggregate receipt; every one a finite number or the receipt reads
`plumbing_ok` = 0, §9.6):**

- `plumbing_ok`: 1.0 iff every check of §9.4 held — nothing else (the gate's text and its code are one list).
- `degenerate_fraction`: degenerate 4000-step runs / 60. Degenerate 12,000-step runs are excluded from `gap_long`
  and listed, not counted here.
- `free_perm_mean`: the positive control. For each of the six permutations, the mean over the three FREE runs at
  seeds {0, 1, 2} (held-out) or {2, 3, 4} (anchor) **with a degenerate FREE run counted as 0.0** (M4: a collapse of
  the arm that defines the gap is the measurement the floor exists to catch, not a run to drop); then the mean over
  the six permutations. 18 runs.
- `anchor_gap_newseed`: `acc_new(FREE) − acc_new(CLAMPED)` on permutation 1234, seeds {2, 3, 4}.
- Per held-out permutation p: `gap_p = acc(p, FREE) − acc(p, CLAMPED)`. `gap_perm_mean`: the mean of the five.
- `gap_perm_ci_lo`: `gap_perm_mean − 2.776 · sd(gap_p) / √5`, sd with n − 1 (t(4, 0.975) = 2.776). The permutation
  is the sampling unit because the sentence under test is "permutation-averaged"; M6's caveat — seed noise alone
  gives the five `gap_p` a scatter of about 0.045 — is why the pooled interval and the variance share are printed
  beside it (reported, below).
- `gap_long`: mean FREE(12k, seeds 0, 1) − mean CLAMPED(12k, seeds 0, 1, 2), permutation 1234, non-degenerate runs.
- `free_minus_real2_perm_mean`: the mean over the five held-out permutations of `acc(p, FREE) − acc(p, REAL2)`.
- `real2_minus_clamped_perm_mean`: the mean over the five held-out permutations of `acc(p, REAL2) − acc(p, CLAMPED)`.
  These two sum to `gap_perm_mean` (same pools), so with `gap_perm_mean ≥ 0.15` they cannot both be ≤ 0.03.

**Reported (under `reported`; none moves a verdict):**

- `gap_perm_ci_hi`, `gap_perm_min`, `gap_perm_sd`; the per-permutation table (n per arm, means, gap,
  FREE − REAL2, REAL2 − CLAMPED, recovery).
- `recovery_p = (acc(p, REAL2) − acc(p, CLAMPED)) / gap_p`, defined where `gap_p ≥ 0.05`; `recovery_perm_mean`
  and `recovery_perm_ci` (two-sided 95% t-interval over the defined permutations, t(n − 1, 0.975) from
  {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}); `recovery_class`: `"rotation"` if the interval lies entirely below
  0.25, `"diversity"` if entirely above 0.75, `"undefined"` if fewer than two permutations define it, else
  `"partial_or_unresolved_at_this_n"` (M3). **This interval is the number the attribution sentence prints**; the
  G_R suffix is a label at the sealed PREREG's bars for comparability, and §7 says how much to read into it.
- `pooled_gap` and `pooled_ci`: FREE minus CLAMPED over the 15 held-out runs per arm, run as the unit, normal
  approximation `± 1.96 · √(var_F/n_F + var_C/n_C)` — labelled as such; the three-times-tighter interval M6 asks to
  print beside the permutation one.
- `bootstrap`: a 2,000-resample hierarchical bootstrap (permutations with replacement, then seeds within each
  resampled permutation, per arm; `numpy.random.default_rng(20261011)`) of `gap_perm_mean`: the 2.5th and 97.5th
  percentiles, and `between_perm_var_share` = max(0, var(gap_p) − mean_p(var_F,p/n_F,p + var_C,p/n_C,p)) /
  var(gap_p) — the method-of-moments share of the permutation scatter that seed noise does not explain.
- `anchor`: per-seed accuracies on 1234; `free_01`, `clamped_01` (seeds 0, 1 means); `anchor_max_abs_dev` =
  max(|free_01 − 0.9195|, |clamped_01 − 0.6073|) — the sealed PREREG's plumbing metric, reported here;
  `anchor_gap_5seed` (all five seeds, for the reader); `anchor_free_minus_real2_newseed`,
  `anchor_real2_minus_clamped_newseed`, `anchor_recovery_newseed` (seeds 2, 3, 4) — the attribution on the sealed
  run's own permutation with seeds it never ran, the number §7 compares to the sealed verdict.
- `long`: `free_long`, `clamped_long`; `clamped_long_gain` = CLAMPED(12k, seeds 0–2) − CLAMPED(4k, 1234, seeds 0–2)
  (between schedules); `free_long_gain` (seeds 0, 1); `clamped_long_midgain` = mean over the 12,000-step CLAMPED runs
  of acc(8000) − acc(4000) on the same curve (within schedule; "plateaus" means this is near zero while the gap
  survives); `free_long_midgain` likewise; `clamped_last_quarter_gain` = acc(4000) − acc(3000) per 4000-step CLAMPED
  run, mean — with the caveat that a cosine schedule drives the learning rate to zero at the budget.
- `degeneracy`: counts and leaf names by arm and by permutation; `unexcluded`: every gated metric recomputed with no
  exclusion (degenerate runs at their own values) — the aggregate printed twice, with and without exclusions (M4).
- `reliance`: the within-model θ→0 reliance of each FREE run and its mean (not evidence for the gap).
- `leaf_chain`: whether the append-only leaf log verifies and how many leaves were re-run (§9.5).

---

## 5. Gates — ten gates, thirteen rows, total

```gates
{"gates": {"G_P_plumbing": {"metric": "result.plumbing_ok", "op": "==", "value": 1.0,
                            "power_basis": "an all-or-nothing flag over the checks of section 9.4, each performed by the runner and recorded in the receipt: the two imported runners' LF-normalised bytes hash to the frozen blobs c8d7cdfb... and caf276da... at preflight and again at aggregate; _mnist.npz hashes to 98d68505...; the PREREG on disk hashes to the sealed blob; the parallel scan equals the sequential scan (max abs diff < 1e-4); FREE, CLAMPED and REAL2 constructed after torch.manual_seed(s) share every non-SSM initial tensor bit for bit and leave identical RNG state at seeds 0-4; REAL2 parameter count equals FREE's and CLAMPED's is within 0.2% of it; the held-out permutation list equals the frozen derivation from the recorded beacon or tag, and all leaves carry the same derivation; all 65 scheduled runs are present, each leaf's read-back module values equal the schedule and the committed constants, each curve has steps/500 points, each final accuracy equals its last curve point, the leaf log's hash chain verifies, and every gated metric is finite; the runner refuses before data if a pre-run check fails, so this gate records that it did not"},
           "G_D_stable": {"metric": "result.degenerate_fraction", "op": "<=", "value": 0.10,
                          "power_basis": "a constant responder scores at most 0.1135 on the 10,000-image test set (largest class share, 1135 of 10000); a 4000-step run ending at or below 0.15, or logging a non-finite loss, is a training failure, not a measurement; the receipt's CLAMPED arm sits at 0.55-0.66, so degeneracy is a failure of this run, not a property of decay; up to 6 of 60 runs may fail and be listed (decay collapses excluded from the means, FREE collapses counted as 0 in the floor); more than that and the aggregate is not the experiment described here"},
           "G_F_floor": {"metric": "result.free_perm_mean", "op": ">=", "value": 0.85,
                         "power_basis": "the positive control that must fire: the rotating model learns permuted MNIST on permutation average, 18 runs over six permutations at three seeds each; the receipt's FREE is 0.9195 with seed spread 0.009, and 0.85 leaves 0.07 for permutation and seed variation; a degenerate FREE run counts as 0, so one collapse among the 18 lowers the mean by about 0.05 and is tolerated at the receipt's accuracy while two lower it by about 0.10 and fail the floor; below it the task was not learned by the arm that defines the gap and no gap may be read"},
           "G_C_anchor": {"metric": "result.anchor_gap_newseed", "op": ">=", "value": 0.15,
                          "power_basis": "the effect this design exists to test must exist where it was measured, on seeds the receipt never ran: FREE minus CLAMPED on permutation 1234 over seeds 2, 3, 4; 0.15 is half the receipt's 0.3122 and the bar the sealed untied PREREG froze as G_C_gap; with CLAMPED's seed sd about 0.077 (0.109 spread on two seeds) and FREE's 0.009, the SE of a three-seed gap is about 0.045, so a true gap of 0.31 clears 0.15 at about 3.6 SE and a true gap under 0.05 fails it at about 2.2 SE; seeds 0 and 1 are the receipt's own and enter no gate"},
           "G_G_exists": {"metric": "result.gap_perm_ci_lo", "op": ">", "value": 0.0,
                          "power_basis": "the 95% t-interval over the five held-out per-permutation gaps excludes zero; t(4, 0.975) = 2.776, and seed noise alone (three seeds, CLAMPED sd 0.077) scatters the five gaps with sd about 0.045, so the half-width is about 0.056 with no permutation effect at all; a permutation-mean gap at or above about 0.06 clears this at that sd; failing it with G_C_anchor passed means the gap lives on the anchor permutation and not on permutations no one chose"},
           "G_G_mean": {"metric": "result.gap_perm_mean", "op": ">=", "value": 0.15,
                        "power_basis": "half the receipt's gap, on the average of five held-out permutations; the same floor as G_C_anchor so that 'reproduced on new seeds' and 'reproduced on new permutations' are judged by one number; 15 runs per arm put the SE of the five-permutation mean gap near 0.020 from seed noise, so 0.15 is about eight SE below the receipt's 0.31 and a true gap of 0.10 fails it at about 2.5 SE"},
           "G_G_ci": {"metric": "result.gap_perm_ci_lo", "op": ">=", "value": 0.05,
                      "power_basis": "the lower end of the permutation interval must itself clear the committed WIDENS bar against sMNIST (0.0408 + 0.005 = 0.0458, rounded up to 0.05): then the permutation-averaged gap still widens on the sMNIST gap at the interval's low end, which is what 'the headline is robust' means in the record's own terms; at a permutation-mean gap of 0.31 and per-permutation sd 0.045 the lower end is near 0.25; the bar fails if the mean falls under about 0.11 at that sd, or if the sd across permutations exceeds about 0.21 at a mean of 0.31; because seed noise alone contributes about 0.045 of that sd, a failure here reads 'the interval reaches below the WIDENS bar at this n', not 'the gap depends on the permutation' (the bootstrap's between-permutation variance share is the number for that)"},
           "G_B_long": {"metric": "result.gap_long", "op": ">=", "value": 0.15,
                        "power_basis": "the gap at three times the frozen budget on permutation 1234 under a cosine schedule re-stretched to 12,000 steps; FREE seeds 0 and 1 (sd 0.009) and CLAMPED seeds 0, 1, 2 (sd about 0.077): SE of the contrast about 0.045, so a true 12,000-step gap of 0.31 clears 0.15 at about 3.5 SE and a gap that has closed to 0.05 fails it at about 2.2 SE; below 0.15 the 4000-step gap is read as budget-dependent; how much the decay model gained is reported from the curves and from the between-schedule difference, and neither gain is a gate (a 0.10 bar on a quantity with SE 0.063 would flip the verdict family on noise)"},
           "G_R_diversity": {"metric": "result.free_minus_real2_perm_mean", "op": "<=", "value": 0.03,
                             "power_basis": "the sealed untied PREREG's G_R_recovers bar, unchanged for comparability, on the mean over five held-out permutations at 15 runs per arm; REAL2 is a decay bank with no receipt on this task, so its seed sd is taken as CLAMPED's 0.077 and the SE of this mean is about 0.020, which puts the 0.03 bar at about 1.5 SE: a true difference of 0 passes about 93% of the time and a true difference of 0.03 passes half the time; REAL2 within 0.03 of FREE is the diversity reading (the gap is the clamp's magnitude-tying, not the rotation's), and the recovery interval under 'reported' is the number the sentence prints"},
           "G_R_rotation": {"metric": "result.real2_minus_clamped_perm_mean", "op": "<=", "value": 0.03,
                            "power_basis": "the sealed untied PREREG's G_R_fails bar, unchanged, on the same means; with REAL2's sd taken as CLAMPED's the SE of this mean is about 0.028 (0.077 times the square root of 2/15), so the 0.03 bar sits at about 1.1 SE: a true difference of 0 falsely fails about 14% of the time and a true difference of 0.03 fails half the time, which is why the ROTATION_LOAD_BEARING versus PARTIAL suffix is a label and the recovery interval is the finding; REAL2 within 0.03 of CLAMPED means untying bought nothing the task can see; both G_R gates true at once requires a permutation-mean gap of at most 0.06, which every row that reads them rules out by requiring G_G_mean"}},
 "outcomes": [{"when": {"G_P_plumbing": false}, "verdict": "INVALID__plumbing"},
              {"when": {"G_P_plumbing": true, "G_D_stable": false}, "verdict": "INVALID__training_unstable"},
              {"when": {"G_P_plumbing": true, "G_D_stable": true, "G_F_floor": false}, "verdict": "ABSTAIN__free_arm_below_floor"},
              {"when": {"G_P_plumbing": true, "G_D_stable": true, "G_F_floor": true, "G_C_anchor": false}, "verdict": "HEADLINE_NOT_REPRODUCED__anchor_permutation_three_new_seeds"},
              {"when": {"G_P_plumbing": true, "G_D_stable": true, "G_F_floor": true, "G_C_anchor": true, "G_G_exists": false}, "verdict": "NOT_ROBUST__gap_confined_to_anchor_permutation"},
              {"when": {"G_P_plumbing": true, "G_D_stable": true, "G_F_floor": true, "G_C_anchor": true, "G_G_exists": true, "G_G_mean": true, "G_G_ci": true, "G_B_long": true, "G_R_diversity": true}, "verdict": "ROBUST__TIMESCALE_DIVERSITY__rotation_not_load_bearing_here"},
              {"when": {"G_P_plumbing": true, "G_D_stable": true, "G_F_floor": true, "G_C_anchor": true, "G_G_exists": true, "G_G_mean": true, "G_G_ci": true, "G_B_long": true, "G_R_diversity": false, "G_R_rotation": true}, "verdict": "ROBUST__ROTATION_LOAD_BEARING__beyond_diversity"},
              {"when": {"G_P_plumbing": true, "G_D_stable": true, "G_F_floor": true, "G_C_anchor": true, "G_G_exists": true, "G_G_mean": true, "G_G_ci": true, "G_B_long": true, "G_R_diversity": false, "G_R_rotation": false}, "verdict": "ROBUST__PARTIAL__diversity_recovers_some"},
              {"when": {"G_P_plumbing": true, "G_D_stable": true, "G_F_floor": true, "G_C_anchor": true, "G_G_exists": true, "G_G_mean": true, "G_G_ci": true, "G_B_long": false}, "verdict": "BUDGET_CONFOUND__gap_does_not_survive_3x_steps"},
              {"when": {"G_P_plumbing": true, "G_D_stable": true, "G_F_floor": true, "G_C_anchor": true, "G_G_exists": true, "G_G_mean": true, "G_G_ci": false, "G_R_diversity": true}, "verdict": "ATTENUATED_VARIABLE__TIMESCALE_DIVERSITY__interval_reaches_below_widens_bar"},
              {"when": {"G_P_plumbing": true, "G_D_stable": true, "G_F_floor": true, "G_C_anchor": true, "G_G_exists": true, "G_G_mean": true, "G_G_ci": false, "G_R_diversity": false, "G_R_rotation": true}, "verdict": "ATTENUATED_VARIABLE__ROTATION_LOAD_BEARING__interval_reaches_below_widens_bar"},
              {"when": {"G_P_plumbing": true, "G_D_stable": true, "G_F_floor": true, "G_C_anchor": true, "G_G_exists": true, "G_G_mean": true, "G_G_ci": false, "G_R_diversity": false, "G_R_rotation": false}, "verdict": "ATTENUATED_VARIABLE__PARTIAL__interval_reaches_below_widens_bar"},
              {"when": {"G_P_plumbing": true, "G_D_stable": true, "G_F_floor": true, "G_C_anchor": true, "G_G_exists": true, "G_G_mean": false}, "verdict": "ATTENUATED__gap_below_half_receipt_on_permutation_average"}],
 "smoke_verdict": "INVALID__smoke_plumbing_only"}
```

Table semantics are `styxx.protocol.Experiment.score`'s: rows are walked in order and the earliest row whose `when`
matches wins. `tests/test_pmnist_robustness.py` extracts the block with the lab's own `_select_gates_block`,
rejects duplicate keys, enumerates all 1,024 assignments of the ten booleans, and asserts that every assignment
reaches a row, that every gate is named by some row, that every row is reachable, and that `Experiment` scores a
receipt of the §10 shape to each of the thirteen verdicts. Two notes the enumeration surfaces: (a) in the
`ATTENUATED__gap_below_half_receipt` row the attribution gates are not read, because with a permutation-mean gap
under 0.15 the two 0.03 bars can both hold, so the attribution is reported there, not pronounced; (b) in the ROBUST
and ATTENUATED_VARIABLE rows `G_G_mean` is true, so `G_R_diversity` and `G_R_rotation` cannot both be true, and the
diversity row's precedence is never exercised. `G_G_ci` true implies `G_G_exists` true; the table resolves the
unrealisable assignments by order, as the protocol requires of a total table.

---

## 6. Positive control, degeneracy guard by arm, and what counts as what

**The positive control that must fire** is `G_F_floor`: the rotating arm learns the task on permutation average
(≥ 0.85) over 18 runs. It is a floor on the operation, not on a comparison; a design in which FREE itself failed
would print ABSTAIN and read no gap. Beside it, `G_C_anchor` asks that the effect exist where it was measured, on
seeds the receipt never ran, before robustness is scored; its failure is a finding about the headline (three new
seeds on its own permutation), not a plumbing fault, and is printed as `HEADLINE_NOT_REPRODUCED` with the two re-run
seeds' deviation from the receipt beside it.

**Degeneracy guard, by arm (M4).** A 4000-step run ending at or below 0.15 on the 10,000-image test set (a constant
responder scores ≤ 0.1135, the largest class share; NaN weights give argmax class 0 and 0.0980) or logging a
non-finite loss is degenerate. A degenerate **CLAMPED or REAL2** run is excluded from every mean and listed:
excluding a collapsed decay run lowers the gap, which is the conservative direction. A degenerate **FREE** run is
excluded from the gap means in the same way but **counts as 0.0 in `free_perm_mean`**, so a collapse of the arm
that defines the gap pulls the positive control down instead of lifting it; one such collapse is tolerated at the
receipt's accuracy, two fail the floor. `G_D_stable` caps the total at 6 of 60 whatever the arm. Every aggregate is
printed twice, with and without the exclusions.

**Anchor deviation is reported, not gated.** The sealed untied PREREG gated `anchor_max_abs_dev <= 0.03` against the
receipt's absolute accuracies. This design does not: it exists to find out whether the receipt's numbers hold, and a
new-seed FREE at 0.87 with CLAMPED at 0.66 is a result about the headline, not a reason to refuse the run. The
deviation of seeds 0 and 1 from the receipt is printed; the plumbing gate holds what is plumbing (§9.4).

**What the bars mean, in words.**

| reading | numeric condition |
|---|---|
| the headline is robust | held-out permutation-mean gap ≥ 0.15, interval lower end ≥ 0.05, gap at 3× budget ≥ 0.15 |
| the gap is a budget artifact | permutation gap robust at 4000 steps, but < 0.15 at 12,000 (2 vs 3 seeds, one permutation, re-stretched schedule) |
| attenuated, variable | mean ≥ 0.15 but the interval's lower end < 0.05: the interval reaches below the WIDENS bar at this n (five permutations, three seeds each); whether the permutation or the seed is the source is `between_perm_var_share`, reported |
| attenuated, small | interval excludes zero but the mean < 0.15: the gap exists and is under half the receipt |
| not robust | interval includes zero although the anchor permutation reproduces on new seeds |
| the headline did not reproduce | three-new-seed gap on permutation 1234 < 0.15 |
| diversity explains it | REAL2 within 0.03 of FREE on held-out permutation average |
| rotation is load-bearing beyond diversity | REAL2 within 0.03 of CLAMPED on held-out permutation average |
| partial | REAL2 between; `recovery_perm_mean` and its interval are the number to print, and at this n PARTIAL is the most likely suffix whatever the mechanism (§7) |

---

## 7. Relation to the sealed untied receipt — predictions under every verdict it can carry

The sealed run (`PREREG_pmnist_untied_2026_09_02.md`; permutation 1234, seeds 0 and 1, three arms) returned one of
three substantive strings or one of two INVALID strings. This design re-runs its six trainings as the determinism
re-run inside the anchor permutation's fifteen, runs the same three arms on three seeds it never ran, and takes the
attribution to five permutations no one chose; its bars do not move with the sealed verdict. The predictions:

| sealed verdict (unread) | what this design predicts | what the design adds whatever happens |
|---|---|---|
| `ROTATION_LOAD_BEARING__beyond_diversity` | `anchor_real2_minus_clamped_newseed` ≤ 0.03 on seeds 2–4 (reported); on the held-out permutations `recovery_perm_mean` near 0 with `recovery_class` `"rotation"` if the interval is tight enough, else `"partial_or_unresolved_at_this_n"`; verdict in the ROBUST or ATTENUATED family with suffix ROTATION_LOAD_BEARING or, at this precision as often, PARTIAL | whether a two-seed, one-permutation attribution holds on new seeds and travels to five permutations, with an interval; or that it was permutation- or seed-specific, said with the same bars |
| `TIMESCALE_DIVERSITY__rotation_not_load_bearing_here` | `anchor_free_minus_real2_newseed` ≤ 0.03 (reported); `recovery_class` `"diversity"` or the interval's upper end above 0.75; the gap itself still robust (FREE and REAL2 both above CLAMPED); verdict `ROBUST__TIMESCALE_DIVERSITY` — the headline gap stands and is the clamp's magnitude-tying, not the rotation's, which `DUE_DILIGENCE_2026_09_13.md` lines 86–87 name as the death of the flagship's causal story | whether that kill is itself robust (five permutations, 15 runs per arm) or a two-seed event; either way the sentence the lab may print changes and this receipt is the one that licenses it |
| `PARTIAL__diversity_recovers_some` | `recovery_perm_mean` in (0.1, 0.9) with a 95% interval of width under about 0.4 (five permutations); verdict suffix PARTIAL unless the means fall inside a 0.03 bar | PARTIAL becomes a number with an interval; the RESULT prints "diversity recovers X of the gap, interval [a, b]" instead of a label |
| `INVALID__plumbing_anchors_drifted` | `anchor_max_abs_dev` here also above 0.03 (same card, same torch), reported; the gap gates unaffected | because this design reports the anchor and gates the gap, it still measures the headline on this device; the RESULT then says the receipt's absolute accuracies do not reproduce here while the gap does or does not |
| `INVALID__gap_did_not_reproduce` | `anchor_gap_newseed` decides whether the two-seed failure was CLAMPED's seed spread (0.109 in the receipt) or the headline: either `HEADLINE_NOT_REPRODUCED` on three new seeds, or a gap that reappears on them with CLAMPED's seed variance printed | the one outcome under which the record would otherwise hold a sealed INVALID and no further number |

**On the suffix (M3).** At 15 runs per arm the SE of the attribution means is about 0.020 (FREE − REAL2) and 0.028
(REAL2 − CLAMPED) against a 0.03 bar, so the ROTATION_LOAD_BEARING-versus-PARTIAL suffix turns on noise in the
band the disclosed prior puts the result in: with a true REAL2 − CLAMPED of 0 the rotation gate still fails about
14% of the time, and with a true 0.03 it is a coin flip. A PARTIAL suffix is therefore not a finding about the
mechanism; the `recovery_class` and its interval are, and the RESULT prints them in the same sentence as the suffix.

**Disclosed prior (frozen with the bars):** toward a robust gap at a permutation mean smaller than the receipt's —
about 0.25 ± 0.05 — because a two-seed reading with CLAMPED at 0.5529 on one seed is more likely above the
permutation mean than below it; toward `recovery_perm_mean` near 0 (the ordered-copy control recovered nothing,
`untied_control_result.json#/metrics/recovery_fraction` = 0.0), printed most likely as
`ROBUST__ROTATION_LOAD_BEARING__beyond_diversity` or `ROBUST__PARTIAL__diversity_recovers_some` at roughly even
odds given the precision above, with `recovery_class` `"rotation"` or `"partial_or_unresolved_at_this_n"`. Second
most likely family: `ATTENUATED_VARIABLE`, because seed noise alone puts the permutation interval's half-width near
0.056 and three seeds per permutation is the default (decision 2). On the budget: nothing in the record shows decay
stopped improving, so a positive `clamped_long_midgain` is expected and reported; `G_B_long` is expected to hold.
The bars are frozen either way, and the RESULT prints the sealed verdict beside this one only after the operator
unseals it.

---

## 8. Budget on this GPU and the three-night staging

Per-run time: 13.6 minutes per 4000-step run measured on 2026-10-11 (`sealed/SEALED_README.md`, §1), plus four extra
full-test evaluations (about 40 s) for the 500-step curve; a 12,000-step run is three times that plus twenty
evaluations (about 42 min). The `--smoke` re-measures and the runner prints a projected finish before every night.
Memory: H = 64, batch 64, 784 steps — small; the guard is against sharing the card. The runner **refuses to start a
smoke or a night unless CUDA is available and at least 6 GiB (6,144 MiB) of device memory is free** (m1: the
2026-10-11 measurement of `ENV_READINESS_2026_10_11.md`, 5,093 MiB held on an 8,188 MiB card, leaves 3,095 MiB and
is refused; the process holding the card must be closed), and records free and total memory in every leaf.

| stage | runs | at 13.6 min (measured) | at 18 min (refusal threshold) |
|---|---|---|---|
| A. permutation 1234, 3 arms × seeds {0..4}, 4000 steps | 15 | ≈ 3.6 h | 4.5 h |
| B. LONG: CLAMPED 12,000 steps seeds {0,1,2}; FREE 12,000 steps seeds {0,1} | 5 | ≈ 3.5 h | 4.5 h |
| C. held-out P1, P2: 3 arms × seeds {0,1,2}, 4000 steps | 18 | ≈ 4.3 h | 5.4 h |
| D. held-out P3, P4, P5: same | 27 | ≈ 6.4 h | 8.1 h |
| total (gated design) | 65 | **≈ 17.8 h** | 22.5 h |
| E. optional width stage (decision 4): H = D_SSM ∈ {32, 128}, permutation 1234, 3 arms × seeds {0,1} | 12 | ≈ 4.2 h (estimated) | — |

**Three nights, the card free from 22:30 to 06:30 (`NEXT_AFTER_V3.md` §4):**

- **Night 1** (`--night 1`; needs no beacon): stage A, then CLAMPED-LONG seeds {0, 1, 2} — ≈ 5.7 h measured, 7.2 h at
  the threshold.
- **Night 2** (`--night 2`; needs the beacon or `--fallback-perms`): stage C (P1, P2), then FREE-LONG seeds {0, 1} —
  ≈ 5.7 h / 7.2 h.
- **Night 3** (`--night 3`; needs the beacon or `--fallback-perms`): stage D (P3, P4, P5) — ≈ 6.4 h / 8.1 h; then
  `--aggregate` (seconds).

The runner writes one leaf JSON per completed run under `pmnist_robustness_runs/` and skips leaves that already
exist, so a crash or a morning interruption loses at most one run and a night that spills over is finished the next.
Before a night starts it requires the smoke receipt (`pmnist_robustness_smoke.json`) and refuses without
`--accept-schedule` if the smoke's projected time per 4000-step run exceeds **18.0 minutes** or the night's projected
total exceeds **8.0 hours**; it prints the recomputed schedule either way. The design does not change with the
schedule. Over a 1.5× overrun of the approved compute the operator stops the run and the RESULT says so; there is no
budget verdict here because the leaves persist and an incomplete leaf set reads `INVALID__plumbing` by §9.4.

---

## 9. The runner, built on the committed runners without editing them

`run_pmnist_robustness.py`, committed under `papers/frequency-resonance/` beside `R` and `U` after this file.

**9.1 Imports verbatim.** `sys.path.insert(0, <papers/frequency-resonance>)`; `import run_pmnist_ablation as R`;
`import run_pmnist_untied as U` — lazily, so that the runner's pure functions (hashing, derivation, schedule,
aggregate arithmetic) import without torch and are what the tests exercise on CPU. It calls `R.redteam()`,
`U.redteam()`, `R.load_data()`, `R.train(free, seed, …)`, `U.train_real2(seed, …)`, `R.test_acc`, `R.nparams`,
`R.SSMClassifier`, `U.RealClassifier`, `U.nparams` — and defines no model and no training loop of its own.

**9.2 Overrides, declared.** `R.train`, `U.train_real2` and `R.load_data` read `STEPS`, `EVAL_EVERY`, `BATCH`, `LR`,
`WD`, `PERM_SEED`, `H`, `D_SSM` as module globals at call time. The runner sets exactly `R.PERM_SEED`, `R.STEPS`,
`R.EVAL_EVERY` (and, in the optional width stage, `R.H` and `R.D_SSM`) before each run and restores the committed
values after it; each leaf stores the values read back from the module (`read_back`) so the receipt says what ran.
`BATCH`, `LR`, `WD`, `T_LEN`, `N_BLK` are never touched and are asserted equal to the committed constants (64, 3e-3,
0.01, 784, 3) before data and in every leaf.

**9.3 Curve capture without editing the loop.** `R.train` calls `test_acc(m, xte, yte, ntest)` every `EVAL_EVERY`
steps through the module global `test_acc`. The runner replaces `R.test_acc` with a recording wrapper around the
original for the duration of a run (the committed function is what computes every value) and restores the original
afterwards. Evaluation touches no RNG; the trajectory is unchanged. The final accuracy is the committed
`test_acc(m, xte, yte)` on the full set; the runner **records** whether it equals the last curve point
(`curve_final_mismatch`) and continues (m6) — a mismatch is a plumbing fault the aggregate maps to `plumbing_ok` = 0,
not a reason to lose a night. The loss values come from `R.train`'s own printed `step … loss … test …` lines, captured
by redirecting stdout into a buffer that is also echoed to the console; the runner patches nothing in torch and
nothing in `R` beyond the module attributes named in §9.2 and `test_acc`.

**9.4 Pre-run asserts (refuse before data; each recorded in the leaf and re-checked by the aggregate — this list is
what `plumbing_ok` means).**

1. The two imported files' LF-normalised sha256 equal `c8d7cdfb…` and `caf276da…` (§1).
2. `_mnist.npz` is present beside the runner and hashes to `98d68505…` (§1).
3. This PREREG on disk, LF-normalised, hashes to the sealed blob recorded in the runner (`PREREG_BLOB_SHA256`) and in
   `SEALS_2026_09_13.md`; `styxx.protocol.Experiment(PREREG)` constructs (the file is committed; an uncommitted
   PREREG is refused by the protocol and so no night starts), reports no vacuous gate and no gate without a power
   basis, and `check_metrics()` over a synthetic aggregate carrying every §4 gated metric as 0.0 resolves every path
   as usable (M1).
4. `R.redteam()`: the parallel scan equals the sequential scan (max abs diff < 1e-4) and FREE/CLAMPED share `B_re`
   at seed 0. `U.redteam()`: REAL2's parameter count equals FREE's. CLAMPED's is within 0.2% of FREE's.
5. Three-arm RNG sharing (m4): for s in 0..4, after `torch.manual_seed(s)` and construction, `SSMClassifier(True)`,
   `SSMClassifier(False)` and `RealClassifier()` leave identical `torch.get_rng_state()`, and every parameter outside
   the `.ssm.` modules is bit-identical across the three.
6. The held-out permutation list equals the frozen derivation (§3) from the supplied beacon or from the fallback tag,
   and the derivation label is recorded; nights 2 and 3 refuse without one of `--beacon`, `--anchors`,
   `--fallback-perms`.
7. CUDA available and ≥ 6 GiB free (§8); torch version and device recorded.

At aggregate: items 1–3 re-checked on disk; every leaf's `imports`, `data_sha256`, `preflight_ok`, `read_back`,
`perm_derivation` and `beacon` agree with the schedule and with each other; all 65 leaves present; curve lengths
`steps / 500`; no `curve_final_mismatch`; the leaf log's hash chain verifies (§9.5); every gated metric finite.

**9.5 Leaves and the append-only leaf log.** Each completed run writes `pmnist_robustness_runs/p<perm>_<arm>_s<seed>_<steps>_h<H>.json`
and appends one line to `pmnist_robustness_runs/leaf_chain.jsonl`: `{"n", "leaf", "sha256", "prev", "entry_id",
"wall_clock"}` with `entry_id = sha256(prev_entry_id + leaf_sha256)`. The aggregate verifies the chain and counts
leaves whose current bytes are not the latest entry for their name (`leaf_chain.rewrites`). A lab that deletes a
leaf and re-runs it leaves a visible second entry; a lab that rebuilds the chain is not caught by this, and the
RESULT says so (`ATTACK.md` §6: only an outside replication checks the numbers).

**9.6 Smoke.** `--smoke` propagates to `R` (200 steps, one seed, 2,000-image in-loop evaluations, `EVAL_EVERY` 100)
and runs one FREE, one CLAMPED, one REAL2 on permutation 1234 and one FREE on the lowest fallback permutation
(`27380109`; the smoke needs no beacon and reads no gate), each timed separately with the data reload outside the
timer (m9); it writes `pmnist_robustness_smoke.json` with every §9.4 check, scores it through
`Experiment(PREREG).score(receipt, smoke=True)` — `INVALID__smoke_plumbing_only` by type — and prints the median
seconds per 200 steps × 20 as the projected 4000-step time the nights read.

**9.7 Aggregate and score.** `--aggregate` reads every leaf, computes §4, writes `pmnist_robustness_result.json`
with `prereg`, `prereg_blob_sha256`, `config` (steps, seeds, permutations, derivation, beacon), `imports`, `data`,
`schedule`, `runs` (every leaf), `per_permutation`, `result` (the gated metrics), `reported`, `unexcluded`,
`bootstrap`, `degeneracy`, `anchor`, `long`, `leaf_chain`, and scores through
`styxx.protocol.Experiment(PREREG, repo_root=ROOT).score(receipt)` — the whole receipt, since every metric path is
`result.<name>` (M1). The gates block's sha256 and the PREREG's commit travel with the verdict. **Non-finite rule
(m5):** if any gated metric is not a finite number (an arm with no surviving run on a permutation, for instance), the
runner records the names under `result.nonfinite_metrics`, writes 0.0 in their place, sets `plumbing_ok` to 0.0 and
scores — so the frozen table's `INVALID__plumbing` row carries the case, not an exception.

---

## 10. Receipt and RESULT

`pmnist_robustness_result.json` as in §9.7. The RESULT is sworn to this receipt; the HELD/FAILED counts are printed
with `coverage/sentence_share` beside them (`CRITIC.md` m5). The RESULT prints, in this order: the verdict string;
the new-seed anchor gap with its three seeds and the two re-run seeds' deviation from the receipt; the five held-out
per-permutation gaps with seeds, the permutation interval, the pooled interval and the between-permutation variance
share; the LONG arm's numbers (`gap_long`, `clamped_long_gain`, `clamped_long_midgain`, `free_long_midgain`); the
attribution means with `recovery_perm_mean`, its interval and `recovery_class`, and the same three on the anchor
permutation's new seeds; the degenerate list by arm and the aggregate without exclusions; the derivation of the
permutations (beacon and slot, or the fallback label); the sentence that a seal bounds this file's existence from
above and orders nothing else — except that, under the beacon derivation, the 45 held-out-permutation trainings
cannot have preceded the block that named their permutations (`styxx/clock.py`'s own docstring; `SEALS_2026_09_13.md`'s
2026-09-14 correction: the beacon orders the selection, not the computation; here the selection is of the training
inputs); the sealed untied verdict beside this one, once unsealed, with the note that its anchor postdates its run
(run end 18:54:26 on 2026-10-11); and the sentence the verdict licenses — the measured one only (§11). Forward
pointers to `RESULT_pmnist_ablation_2026_07_23.md` and `PREREG_pmnist_untied_2026_09_02.md` go in the same commit as
the RESULT, and `CHANGELOG.md`'s 2026-09-02 sentence ("The arc's headline survives its own confound; the
permuted-MNIST re-run is the owed next preregistration", line 3241 at `f469f25b`) gets its banner then, whatever the
verdict, because the sentence was never supported by a receipt on this benchmark.

---

## 11. The sentences the lab could print

Every sentence names the architecture by its code (M7): "a three-block complex-diagonal linear recurrence
(LRU-style; `CLRU` in `run_pmnist_ablation.py`)" — never "S5/LinOSS-class", which waits on the owed full-text reads
(`CRITIC.md` M8; `ATTACK.md` §5 names the LRU and S4D real-versus-complex rows as prior art to read before any public
sentence, and this design makes no priority claim).

- **ROBUST__ROTATION_LOAD_BEARING:** "On permuted MNIST, averaged over five pixel permutations drawn from a block
  hash after the design was sealed, with 15 runs per arm, removing the eigenvalue phase from an otherwise identical
  three-block complex-diagonal linear recurrence at matched parameters costs `gap_perm_mean` of accuracy (95%
  permutation interval [`ci_lo`, `ci_hi`]; pooled interval [`a`, `b`]); the gap holds at three times the training
  budget on the anchor permutation (`gap_long`, 2 vs 3 seeds), and a real bank with FREE's parameters and twice its
  independent timescales recovers `recovery_perm_mean` of it (interval [`c`, `d`], class `recovery_class`) — inside
  this architecture, on this task." Never: the task needs rotation.
- **ROBUST__TIMESCALE_DIVERSITY:** "The permuted-MNIST gap is robust and is not the rotation's: a real bank with
  twice the independent timescales recovers it to within 0.03 on held-out permutation average (`recovery_perm_mean`,
  interval [`c`, `d`]). The flagship's causal attribution to oscillation is withdrawn." And the CHANGELOG banner.
- **ROBUST__PARTIAL:** the ROTATION sentence with "recovers `recovery_perm_mean` of it (interval [`c`, `d`]), which
  at 15 runs per arm resolves neither reading" in place of the attribution clause.
- **BUDGET_CONFOUND:** "On permutation 1234, at 12,000 steps under a cosine schedule re-stretched to that budget, the
  gap is `gap_long` (2 FREE vs 3 CLAMPED seeds); at 4,000 steps on three new seeds it is `anchor_gap_newseed`. The
  4000-step gap does not survive the longer budget on this permutation." Not: "the +0.3122 was a training-budget
  reading" — one permutation and five runs do not license the general sentence (M8).
- **HEADLINE_NOT_REPRODUCED:** "On permutation 1234, on three seeds the receipt never ran, FREE minus CLAMPED is
  `anchor_gap_newseed` (seeds 2, 3, 4); the receipt's two seeds re-run to within `anchor_max_abs_dev` of their
  recorded accuracies." With CLAMPED's per-seed accuracies printed (M8).
- **ATTENUATED_VARIABLE:** "The held-out permutation-mean gap is `gap_perm_mean` and its 95% interval reaches
  `ci_lo`, below the WIDENS bar, at five permutations and three seeds each; the bootstrap attributes
  `between_perm_var_share` of the permutation scatter to the permutation." Not: "the gap depends on the permutation"
  (M6).
- **ATTENUATED / NOT_ROBUST:** the numbers, the interval, and the measured sentence at the strength the row names;
  the headline's INDEX status changes accordingly.
- No outcome bears on deployed models, on attention (every fair comparison in the record is a loss or a tie), on
  importance, on priority, or on what the lab added that others did not.

---

## 12. Discipline

Committed before any main-set data; sealed as a git blob per `papers/checksum/SEALS_2026_09_13.md` in the next
commit; anchored at the operator's discretion, and the held-out permutations drawn from that anchor's beacon by the
rule of §3 (or the labelled fallback); `--smoke` INVALID-only; result scored through `styxx.protocol` over the
whole receipt; RESULT sworn to the receipt; no bar moves after data; nothing posted before the RESULT is committed.
The sealed untied receipt is not opened by anyone writing this design before the design is sealed; the RESULT reads
it after. An amendment before the seal is a dated entry appended under a §15 heading and never an edit of the text
above it; after the seal there are no amendments, only a new PREREG. If this box cannot afford the run, the
preregistration stands frozen for whoever can, and its verdict is theirs to report.

---

## 13. Operator decisions (sign each; the run does not start with a blank)

| # | decision | default (built) | consequence of the alternative |
|---|---|---|---|
| 1 | Compute: about 18 GPU-hours measured (22.5 at the refusal threshold) over three nights on the RTX 4070, after the process holding the card is closed | approve | the design stands frozen for whoever can run it |
| 2 | Seeds per held-out permutation: 3 (frozen above) or 5 (adds 30 runs ≈ 6.8 h, a fourth night) | 3 | 5 narrows the per-permutation gap SE from about 0.045 to 0.035 and is the cheapest route to a `G_G_ci` that fails for the right reason (M6); it changes no bar; it is a pre-seal amendment (§12) |
| 3 | Held-out permutations: the anchor's beacon (default, §3) or the sha256-of-tag fallback list | beacon | the fallback is auditable but was typed by the designer's tag and orders nothing after the seal; a fallback run is labelled as such in every leaf and in the RESULT |
| 4 | Optional width stage E (H = D_SSM ∈ {32, 128}, 12 runs ≈ 4.2 h, reported as `gap_by_width`, no gate) | off | on adds a fourth night and a width-trend sentence at reported strength only |
| 5 | Commit and seal before data; anchor | commit and seal; anchor (the beacon needs it) | without the anchor, decision 3 falls to the fallback |
| 6 | Unsealing order: the sealed untied receipt is read after this PREREG is sealed, by another track | after this PREREG is sealed | reading it before sealing would let its numbers move a bar; reading it after sealing and before the run moves nothing and makes the seeds 0/1 re-run preregistered rather than blind |

---

## 14. What this design does not do

No Long Range Arena (the other rung `PAPER_oscillation_causal_map_2026_07_23.md` §6 names; different data,
different budget, a design of its own). No matched-norm control for the within-model θ→0 reliance (a separate
design; `PLAN_next_experiment.md` §1.3). No FROZEN-θ arm (θ fixed at its random init; `ATTACK.md` m10 — reported
strength only, not built). No attention arm — the record's fair comparisons with attention are settled losses or
ties (`STATE_OF_THE_ARC_v2.md` §3 items 4, 8, 9), and a matched transformer on 784-step permuted MNIST is a
different experiment with its own bar, which no document has frozen. No claim about deployed models; the bridge is
`PLAN_next_experiment.md`'s. No change to any committed runner. No claim that the seal orders the twenty
anchor-permutation runs after anything. No reading of the sealed untied receipt.

Banned-word and BOM scan of this file, the runner and the tests: `tests/test_pmnist_robustness.py` reads the lab's
own BANNED tuple (`scripts/zenodo_publish_v7_48_0.py`) and asserts zero hits.
