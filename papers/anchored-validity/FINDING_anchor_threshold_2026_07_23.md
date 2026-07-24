# FINDING (exploratory) — The anchor threshold: the price of catching a shared blind spot — 2026-07-23

Script: `EXPLORE_anchor_threshold_2026_07_23.py` (numpy/scipy; deterministic). **Exploratory, not a
gated claim.** A quantitative sharpening of the paper's identification result
(`PAPER_gold_anchors_license_nothing`, sec. 1: "at three judges a correlated and an independent panel
can induce identical joint verdict distributions"). Here we ask what it *costs* to break that
blindness.

## Self-catch first (the moral licence)
The first construction did NOT produce unlabeled-identical worlds — the two panels differed by 0.057 in
their verdict distribution, so "consensus is blind" was **unsupported**. Caught by printing the match
residual before claiming anything. Fixed by giving the malign world three free parameters (piB, sB, fB
at fixed trap-rate tau) to hit the benign world's three free cells exactly, and a hard guard that
ABORTS if `max|A-B| >= 1e-6`. We broke ours first.

## Construction (exchangeable J=3, exact)
- **BENIGN A** (independent, good): true prevalence 0.400, sensitivity s=0.90, false-positive f=0.10.
- **MALIGN B** (shared blind spot): true prevalence 0.304, s=0.851, f=0.096, and a fraction tau=0.15 of
  true NEGATIVES are "traps" on which **all three judges vote 1 unanimously** (identical wrong).
- Solved so the two worlds' unlabeled vote-count distributions are **identical to 1.4e-14**
  (`[0.4378, 0.1566, 0.1134, 0.2922]` both). They are genuinely different worlds — true prevalence
  differs by 0.096 — that no unlabeled statistic can separate.

## Results
1. **Consensus is blind.** Majority-vote positive rate = 0.4056 in *both* worlds. Because the entire
   unlabeled joint is identical, EVERY label-free method (majority vote, Dawid-Skene, FlyingSquid,
   NTQR — they see only the joint) returns the same answer for the good panel and the secretly-broken
   one. This is not a weakness of a particular estimator; it is the impossibility.
2. **One known-negative is a smoking gun.** P(a known-negative is unanimously mis-voted "1") = **0.0010**
   (benign) vs **0.1508** (malign): a single unanimous-wrong known-negative carries a **150.8x**
   likelihood ratio for "this panel shares a blind spot."
3. **The threshold.** Power to detect the malign world from K known-negatives (one-sided binomial test
   on the unanimous-wrong count, alpha=0.05): K=10 -> 0.46, K=20 -> **0.83**, K=30 -> **0.95**,
   K=50 -> 0.997. Roughly **20-30 known-negatives** reliably catch a shared blind spot that consensus
   cannot see at any sample size.

## The statement
The impossibility of detecting synchronized judge failure is an impossibility about the unlabeled
**marginal**. The **class-conditional** — the distribution of verdicts *on a known-negative* — is
exactly where a good panel and a secretly-wrong one diverge, and a known-negative samples it directly.
So the price of catching a unanimous-confident-wrong panel (the exact failure that breaks trust in an
eval, and the exact thing labels were thought to be needed for) is not labels: it is a handful of
known-negatives, and the likelihood ratio of the first unanimous-wrong one is enormous.

## Scope / honesty
Exchangeable J=3, one construction, a specific (piA, s, f, tau); the mechanism is transparent
(LR = tau / f^J; threshold K ~ 1/tau), so the numbers are illustrative of a clear law, not a general
theorem. General J, heterogeneous judges, and finite-anchor CI on the recovered prevalence (not just
detection power) are the next rungs. Positioning: sharpens the paper's identification whitespace and
the NTQR "cannot detect synchronized failure" line into a quantitative anchor-cost; cite as the
multi-judge deepening, never as a novel impossibility (the impossibility is known — the *cheap escape*
is the contribution). This can become a new section of the flagship or a short standalone note.
