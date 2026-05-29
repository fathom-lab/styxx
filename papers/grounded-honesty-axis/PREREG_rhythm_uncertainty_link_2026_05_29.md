# PRE-REGISTRATION — is the depth suppression-rhythm the UPSTREAM correlate of the single-pass output-uncertainty signature?

**Written 2026-05-29, BEFORE any code for this test is written.** Two lines of this arc have
run independently and both landed:

- **Depth line (internal geometry).** On confabs, the *correct* answer token leads mid-network
  and is overwritten late in a tight band (the "suppression-rhythm",
  `run_suppression_rhythm.py`: `flip_layer` = last layer where the correct token still leads
  before the wrong token wins at the final layer).
- **Output line (detection).** Confabulation is internally legible in a SINGLE forward pass on
  Qwen arithmetic — clean first-token entropy / logit margin separate confab from correct at
  AUC ~0.92, and resampling instability at 0.98
  (`FINDING_detection_locus_2026_05_29.md`).

These have never been tied together. This run asks the single unifying question, on the SAME
white-box model (Qwen2.5-1.5B-Instruct), WITHIN the confab group (where alone the rhythm is
defined — a `flip_layer` only exists when the correct token was suppressed):

**Across confabs, does the depth-rhythm of the overwrite (how late / how sharp) predict the
single-pass output-uncertainty signature (clean entropy, resampling instability)?**

If yes, the internal-geometry line and the detection line are two views of one phenomenon, tied
by a single correlation. If no, they are statistically separate signals on this corpus, and I
will say so — the detection signature would then NOT be a readout of the depth-rhythm.

## Population (pre-committed)

- The CONFAB group: realized one-shot greedy confabs from the 36 hard `SPECS`
  (`v1 != correct`, `v1 is not None`), exactly as in the detection-locus / confab-specificity
  runs.
- A confab enters the correlation **only if it has a defined rhythm**: `corr_lead == True`
  (the correct token led the realized token at some layer `< final`) AND `flip_layer is not
  None`. Confabs where the correct token never led have no suppression-rhythm to measure and
  are excluded from the within-confab correlation (reported as `n_no_rhythm`). This is the
  honest population for a "rhythm" claim, pre-committed.

## Depth-rhythm features (computed at the first divergent answer position, clean logit-lens)

1. **`rel_flip` = `flip_layer / (n_layers − 1)`** — relative depth at which the correct token
   last led (0 = early, →1 = led until just before the final layer). LATER = the correct
   answer stayed competitive deeper = more internal contest.
2. **`sharpness` = `lr[-1] − lc[-1]`** — the final-layer logit gap of the realized (wrong) token
   over the correct token. LARGER = the late overwrite committed more decisively to the wrong
   answer.

## Output-uncertainty signals (already validated in the detection-locus run)

- **`clean_entropy`** — Shannon entropy of the clean first-answer-token distribution (higher =
  less confident).
- **`instability` = 1 − Stability** — over N=10 resamples at T=1.0, exact distinct-integer
  counts, no judge (higher = less self-consistent).
- **`logit_margin`** — top1−top2 clean gap (reported; higher = more confident).

## Predicted directions / bars (within-confab Spearman ρ)

- **U1:** ρ(`rel_flip`, `clean_entropy`) **> 0** — a later correct-lead = a closer contest =
  higher single-pass output entropy.
- **U2:** ρ(`rel_flip`, `instability`) **> 0** — the same contest shows up as lower
  self-consistency across resamples.
- **U3:** ρ(`sharpness`, `clean_entropy`) **< 0** — a sharper final overwrite = a more decisive
  (confident) wrong commitment = lower output entropy.

**B_link (core):** at least ONE of U1 / U2 / U3 reaches `|ρ| ≥ 0.40` with `p < 0.05`, IN THE
PRE-REGISTERED DIRECTION, within the confab group. All six (U1–U3 plus the same features vs
`logit_margin`) are reported regardless of sign.

**SURVIVED iff B_link holds** (≥1 pre-registered pair, predicted direction, |ρ|≥0.40, p<0.05).
Powering: `≥ 12` confabs with a defined rhythm. A null (no depth feature predicts the output
signal above the bar) means the depth-rhythm and the output-uncertainty signature are
statistically SEPARATE on this corpus — the single-pass detector is NOT a readout of the
overwrite geometry — and is reported as REPORT_AS_LANDED.

## Honest scope (pre-committed)

Single open model Qwen2.5-1.5B-Instruct; arithmetic only; one confirmatory run;
feasibility-grade (≤36 confab, minus those with no defined rhythm); depth features from the
SAE-free full-vocab clean logit-lens at the first divergent answer token; output entropy/margin
from the same clean logit-lens at the first answer token; resampling N=10 at T=1.0 (the
validated grounding setting), Stability from exact distinct-integer counts (no judge);
arithmetic ground truth computed in-code then SHA-256'd pre-scoring; exact-integer correctness.
WITHIN-confab only — `flip_layer` is undefined for correct answers, so this is NOT a confab-vs-
correct claim and does not re-test detection (already landed). It tests whether the two
already-landed signals are the SAME phenomenon. Correlation, not cause: a positive ρ shows the
depth-rhythm and output signature co-vary across confabs, not that one mechanistically drives
the other. Does NOT touch the correctness bound — both features here are confidence/uncertainty
readouts; neither corrects the answer (only method-diverse re-derivation does).
