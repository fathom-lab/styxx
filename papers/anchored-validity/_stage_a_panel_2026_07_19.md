# stage-a panel receipt -- anchored validity

date: 2026-07-19
subject: anchored_stage_a.py @ e1ce286 + anchored_stage_a_result.json
panel: 3 lenses (strawman-fairness, exchangeability, verdict-mapping), each with
independent adversarial runs against the harness's own functions, followed by an
independent refutation pass on every FATAL.
lens votes: GO_with_fixes / GO_with_fixes / GO_with_fixes.
fatals surviving refutation: 2 (EX1, F1). synthesis weighs surviving fatals only.

---

## verdict: NO_GO -- prereg freeze blocked

two confirmed fatals, both of the licence-voiding class: a confident wrong number,
silent, biased in the favourable-to-the-audited-system direction. house rule is
explicit -- a gate that cannot refuse is not a gate, and we do not freeze a prereg
on top of one. this is the same defect class that killed calib-poison attempt 3
pre-freeze. lens votes of GO_with_fixes do not outrank two survived fatals.

what the fatals void:
- EX1: the sensitivity (beta) channel of non-exchangeability is structurally
  inexpressible in the harness (make_anchors exposes only alpha_shift; for
  kind='pos' it is a no-op). a panel deaf on organic items but sharp on planted
  gaps returns verdict=ESTIMATED, pi_hat=0.000, CI [0.000,0.006], all judges
  kept -- because the admissibility gate keys on informativeness measured ON
  ANCHORS. uniform channel-gain is internally unidentifiable. the R4 refusal
  headline fails on exactly the population stage B cares about.
- F1: the master-key test only exercises sync that fires ON the anchors
  (scenario() hands the same sync to make_anchors). sync-on-real-only -- the
  most plausible real attack, adversarial content absent from constructed
  garbage anchors -- gives err 0.149 (worse than DS's worst R2 bias), CI
  coverage 0/10, corr detector silent 10/10. the R3 "anchors price master keys"
  headline is licensed only for anchor-borne failure modes; the file claims it
  unconditionally.

what stands (do not re-derive):
- the core DS result is EXONERATED and population-verified: I-projection at
  infinite data with oracle init reproduces the harness's DS errors
  (0.062/0.101/0.039/0.078 vs 0.073/0.128/0.039/0.083). the gap is
  information-theoretic misspecification bias. no restart schedule closes it.
- majority vote is not a strawman (secondary arm only).
- the check-redefinition is legitimate in substance: the original guessed bar
  (DS bias >= 0.10 at rho=.30) was factually unattainable (asymptotic 0.062);
  no correct DS could ever have failed it. the disclosure wording is false in
  one clause (see fix 11) but the reframe itself is sound.
- R6 J=3 unidentifiability is generic (20/20 random correlated models admit
  exact independent fits). the J>=4 approximate version needs rescoping (fix 10)
  but strengthens, not weakens, the anchor pitch at audit-scale n.
- anchored CI coverage is approximately nominal (0.90/0.94/0.98 over 50 reps at
  rho .15/.30/.45) -- measured by the panel, never by the file.

path back: land fixes 1-4 (the fatal set), re-run stage A green, re-panel on the
fatal fixtures only. fixes 5-15 land with it but do not gate the re-panel.

---

## ordered fix list

fatal set (gates the re-panel):

1. make_anchors(): add a beta_shift knob. beta non-exchangeability is currently
   inexpressible anywhere in the harness. (EX1)
2. scenario() / sweep: add four fixtures and run green --
   (a) uniform channel-gain (organic betas 0.10 below anchor betas),
   (b) 1-of-J specialist (organic beta == alpha for one judge),
   (c) sync-on-real-only (simulate_panel sync=0.15, clean anchors),
   (d) anchor-rate-mismatch (anchor sync >> organic sync). (EX1, F1, EX2)
3. anchored() / solve_pi(): add a refusal branch. if the unclipped WLS solution
   lands outside [0,1] beyond bootstrap noise, return
   VOID_ANCHORS__nonexchangeable -- never a clipped confident number. the clip
   at solve_pi line 169 currently converts an observable impossibility
   (per-judge implied pi of -0.22 to -0.59) into pi_hat=0.000. (EX2)
4. anchored(): stratified anchor accounting. detector strata (master-key
   garbage) never pool into alpha/beta estimation; only inert negatives
   estimate error rates; garbage strata emit a separate fire-rate diagnostic.
   (EX2)

major set (lands with the fatal set, does not gate):

5. R2 alpha diagnostic: kill the tautology. realized_alpha = neg.mean(0)
   compared to a_hat = neg.mean(0) is 0.0 by arithmetic identity -- the one
   field that could have measured anchor->organic alpha transfer is a
   self-comparison. compare a_hat to V[y==0].mean(0) under ALPHA_TOL; store
   DS's per-dose alpha error vs the true marginal alpha; check both halves of
   the docstring's promised claim. (tautological-alpha-metric, EX5, F3)
6. R5: enforce the fork. add the check the header promises --
   licensing_says_correct == (anchored err > naive bias) -- and convert the
   single +0.10 point to a dose-response sweep over +/-{0.05,0.10,0.15} x
   {alpha channel, beta channel}, with a case where correction IS licensed,
   monotone degradation asserted, and bound-tightness ratio reported (the
   shipped bar has 2.4x slack and cannot bind). (EX3, EX4, F7,
   r5-licensing-fork-ungated)
7. dawid_skene() arm: bootstrap CI for DS plus a check that it fails coverage
   at rho >= 0.30 while anchored's covers; rename
   R2:ds_fails_the_same_bar_anchored_meets to what it tests -- anchored does
   NOT meet the 0.03 point bar (0.043 at rho .30/.45), it is excused to a
   CI-coverage standard. earn the "DS is confidently wrong" sentence or strike
   it. (asymmetric-bar-symmetric-name)
8. R4: exercise the partial-keep path. mixed panel (2 deaf + 2 informative)
   asserting the deaf are dropped and pi recovered; a near-gate boundary case;
   deaf-panel VOID rate over seeds (measured: 4.55% of seeds a deaf judge
   clears the gate) or a noise-margin gate (gate + 3*sqrt(v/K)). idx-subset
   moment assembly currently has zero coverage. (F5)
9. CI honesty as a rate: replicate-coverage fixture, >= 50 reps per dose,
   coverage in [0.90, 0.99]. one draw per dose licenses nothing. (F4)
10. header/docstring rewrite: R2 to the checked claim (DS fails the 0.03
    recovery bar with dose-response; exceeds 0.10 only at the post-hoc rho=0.45
    dose); strike or scope 'each a fixture with a frozen bar'; scope R6 to
    exact-at-J=3, approximate-at-J>=4 with the detection-n arithmetic
    (lack-of-fit detects correlation at n ~ 20k-53k; undetected at 6k); pick or
    report a witness whose pi discrepancy is material (shipped instance: 0.006,
    clears only via alpha 0.0565 vs 0.05). (F2, F6, stale-docstring-overclaim)
11. disclosure repair: the clause 'the git history carries both versions' is
    false -- git log shows one commit (e1ce286, 378-line new file); both
    versions exist only as in-prose record. state that, and state that the dose
    grids were extended post-measurement to support the dose-response
    criterion. carry the corrected wording into the stage-B prereg's stage-A
    summary. (redefinition-audit-trail-false)

minor set:

12. simulate_panel(): delete dead line 84. it encodes an abandoned bad-day
    formula fully overwritten by line 87; the R6 witness hard-codes the live
    model; reviving it would desynchronize simulator and witness. (unanimous,
    3/3 lenses)
13. wire PI_TOL_FAIL and ALPHA_TOL into checks or delete them from the
    constants and the result-JSON bars block -- a reader is currently told two
    bars were enforced that gate nothing. (stale-docstring, vestigial-bars)
14. dose-growth tolerance one-sided only (r_prev + 0.005 < r_next); the shipped
    form lets 'grows with dose' pass when bias slightly fell. (F8)
15. master-key strength: add partial-strength (p=0.7) and judge-subset sync
    arms, or push them explicitly to the stage-B robustness grid; only the
    p=1.0 all-judges corner is tested. (master-key-maximal-strength)

---

## stage-B prereg obligations, by lens

exchangeability lens (binding):
- scope the prevalence claim as a LOWER bound with respect to sensitivity
  channel-gain; it is exact only under exchangeability.
- bound delta_beta with observables: a graded-difficulty positive-anchor ladder
  descending toward organic salience with the extrapolation reported, and/or a
  labeled organic slice (n ~ 100-200) whose only role is bounding delta_beta.
- mandate stratified anchor accounting (fix 4) in the prereg text, not just the
  code.
- operationalize the licensing rule with observables only: (1) sensitivity band
  pi_hat(delta) over a preregistered delta grid (+/-{0.05,0.10,0.15}) as the
  PRIMARY output, not a point estimate; (2) plug-in predicted-MV-bias at pi_hat
  from anchor-estimated rates as the naive-bias side; (3) the exact-form bound
  delta*(1-pi)/(inf-delta) -- the first-order delta/inf form understates cost
  once delta >~ 0.4*inf, precisely the weak-judge regime stage B inhabits --
  with a stated validity region (delta <= ~0.25*inf).
- specify the informativeness estimator entering every bound and gate:
  anchor-based, bootstrap CI, lower CI edge in denominators (anchor-measured
  informativeness is anti-conservative in the beta-optimism channel).

verdict-mapping lens (binding):
- scope every stage-B claim to anchor-borne failure modes, OR exhibit a
  preregistered data-driven exchangeability test: the moment system is
  overdetermined (J + C(J,2) equations vs 1 parameter), so lack-of-fit is
  available and must be preregistered if the unconditional claim is kept.
- state that delta is oracle-known in stage A and specify how stage B estimates
  or bounds it from data -- else strike 'testable' from the header.
- carry the measured replicate coverage rates (0.90/0.94/0.98) into stage-B
  bar-setting; do not inherit single-draw CI checks.

strawman-fairness lens (binding):
- add semi-supervised DS (anchor posteriors clamped to known labels) to the
  comparator list. all current comparators (FlyingSquid, NTQR) are label-free;
  a referee lands the anchors-in-hand-DS punch immediately. optional stage-A
  fixture showing where it does and does not close the gap (it still fits an
  independence likelihood on organic items -- show the residual bias).
- carry the corrected redefinition disclosure (fix 11) into the prereg's
  stage-A summary verbatim.
- add the beta_shift R5b arm to the robustness grid (dominant stage-B risk:
  planted large-gap positives easier than organic positives, inflating anchor
  sensitivity, biasing pi downward -- favourable to the audited system).
- no anchor-vs-anchor self-comparison ships as a transfer diagnostic in stage B
  (it would report perfect alpha transfer unconditionally and be undetectable
  there -- no y to check against).

---

## re-panel protocol

scope: fixes 1-4 only, run green, plus the header rewrite (fix 10) so the
citable surface matches the checked claims. the DS-misspecification core, the
J=3 witness half, and the CI-coverage property are settled by this panel's own
adversarial runs and are not re-litigated. freeze is the operator's call after
the re-panel, per standing rule.
