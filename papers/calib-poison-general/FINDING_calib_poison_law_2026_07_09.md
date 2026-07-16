# FINDING — A robustness-graded law of substrate auditability: the private-calibration defense generalizes; full poisoning does not.

**Fathom Lab · papers/calib-poison-general · 2026-07-09. Synthesis of the generalization arc across
three constructs (refusal, honesty, sentiment). Every number below is grounded in a committed,
OATH-certified result; this note asserts no new measurement, only the structure they share.**

## The question

The flagship (Fathom v28, DOI 10.5281/zenodo.21263158) established, on *honesty*, that an apparent
evasion of a substrate probe is calibration poisoning, not erasure, and that a probe calibrated on a
private split the attacker never saw reads through the attack. Is that audit rule honesty-specific, or
a general property of substrate linear probes? We tested it on constructs chosen to be independent of
honesty, with pre-registered gates.

## Three regimes — the constructs are not three tries at one test; they are the law's regimes

**Regime 0 — no auditable structure (refusal → UNTESTABLE).** Harmful/benign refusal is a
surface/lexical property: its read is trivially separable (AUROC 1.000 at every layer, including the
earliest) and the model's behavioral judgment is near-chance (0.667 / 0.500). There is no graded
representation to poison and no behavioral capability to preserve, so the attack–defense question is
not even well-posed. *Prediction of the law: surface properties are not adversarially auditable.*
(`RESULT_foundation_refusal_UNTESTABLE_2026_07_09.md`.)

**Regime 1 — fragile probe, full poisoning (honesty).** The honesty read is graded (clean ~0.85) and
behaviorally grounded. A single-layer, knowledge-preserving LoRA attack collapses the naive
(attacker-calibrated) re-lock to chance (~0.46–0.51) while the private (clean-calibrated) re-lock
recovers — full calibration poisoning, defeated by private calibration. (Flagship; read-neq-write.)

**Regime 2 — robust probe, poisoning floors (sentiment).** The sentiment read is graded and stronger
(clean 0.954) with a real behavioral judgment (0.845). Escalating the attacker never fully poisons it:

| attacker | naive re-lock (poisoned) | private re-lock (clean-calibrated) |
|---|---|---|
| single-layer r=16 | 0.72–0.78 | 0.85–0.89 |
| whole-stack r=16 | 0.616–0.626 | 0.730–0.764 |
| whole-stack r=64 (4× capacity) | 0.642–0.651 | 0.760–0.799 |

The naive read floors in the low-to-mid 0.6s and never reaches the 0.60 collapse bar; quadrupling
attacker capacity does not lower it. Full poisoning does not reproduce on the robust probe.

## The law (one axis firm, one axis CONFOUNDED — self-caught)

1. **Poisoning SEVERITY is graded by probe robustness (firm).** How far the naive read can be poisoned
   tracks how strong the representation is: a fragile probe (honesty, clean ~0.85) collapses to chance
   under a single-layer attack; a robust probe (sentiment, clean 0.954) floors above chance even under
   a whole-stack 4×-capacity attack. This is a measurement of the *naive* auditor alone, so it carries
   no cross-auditor confound. Full poisoning is not all-or-nothing; it is robustness-graded.
2. **The private > naive gap is NOT yet attributable to privacy — a probe-capacity confound is live.**
   In every cell the private re-lock beats the naive re-lock, and I initially read this as "the
   private-calibration defense direction is universal." That claim is **not controlled.** In this
   harness the naive auditor fits difference-of-means only; the private auditor fits difference-of-means
   PLUS per-layer logistic PLUS a whole-stack concatenated logistic — a strictly richer probe family.
   So private > naive conflates two variables: *privacy* (the clean split the attacker never saw) and
   *probe capacity* (a better estimator). The gap could be entirely capacity. Until a parity control
   removes the confound, the universal-defense claim is withdrawn.

## CONFOUND — probe capacity vs privacy (load-bearing; back-propagates to the flagship)

The intended variable is privacy: does calibrating on data the attacker did not see recover the read?
But the auditor that has the private data ALSO has more probe capacity here. The two are entangled in
every private > naive number in this arc — and in the flagship's `mount.relock` and the read-neq-write
"private-calibration defense," which use the same DoM-vs-richer asymmetry. If the gap is capacity, then
"calibrate on unseen data" is mis-attributed: the operative fix would be "use a richer probe," and the
privacy framing (including the shipped `mount.relock`) needs a scope correction.

**The owed control — RAN, verdict PARTIAL** (`RESULT_sentiment_probe_parity_2026_07_09.md`,
OATH-HELD). Giving the poisoned auditor the same rich probe family and fit size decomposes the median
0.127 gap into **~two-thirds probe capacity and ~one-third privacy**: the residual privacy effect
(private − parity-naive) is positive in all four cells (0.021–0.052, median 0.041) but below the
pre-registered "privacy stands" bar. So privacy is REAL but MINOR; the majority of the "defense" is a
richer probe. This puts a scope note on `mount.relock` and the read-neq-write defense
(`SCOPE_NOTE_privacy_vs_capacity_2026_07_09.md`, operator-gated): the operative rule is "richer probe
AND private split," not "private split" alone.

## Why this matters (what stands, pending the parity control)

What is firm: auditability requires a behaviorally-grounded, gradiently-represented property —
surface/lexical properties (the refusal regime) are out; and how far a probe can be poisoned is graded
by its robustness — fragile probes collapse, robust probes floor above chance even under the strongest
whole-stack attack tried. That severity axis is a real, single-auditor result.

What is NOT yet established: that a *private-calibration* defense is what recovers the read. The gap
that motivated `mount.relock` and the read-neq-write defense is, in this harness, confounded with probe
capacity. So the honest status is: the *phenomenon* (an auditor with clean data and a richer probe
reads through the attack) is robust; the *attribution* to privacy is open until the parity control
runs. The deployment claim "calibrate on data the attacker did not see" must wait behind that control.

## Honest bounds

All results are `Qwen2.5-1.5B` (sentiment/refusal) plus the 1–1.5B honesty families; linear probes;
knowledge-preserving LoRA attackers (r≤64); small evaluation sets. The floor is demonstrated for this
attacker class, not proven against every attack; the three regimes are three constructs, not a survey.
The severity axis is firm; the privacy-vs-capacity attribution is confounded and gated on the parity
control above.

## Disposition

**Parity control RAN → verdict PARTIAL: the defense is majority probe capacity, minority (real)
privacy.** So the honest v0.4 story is a two-part law — a firm robustness-graded *severity* axis, and
a *defense* whose recovery is mostly a richer probe with a real but minor privacy residual. Any v0.4
deposit must carry the scope note (`SCOPE_NOTE_privacy_vs_capacity_2026_07_09.md`) and is
operator-gated. Owed to fully close: a matched honesty-side parity check in this harness family.
Preregs, harnesses, result JSONs, and certificates are public at commit-level
granularity under `papers/calib-poison-general/`.

---
*Two failures and a flagship turned out to be three regimes of one law on the severity axis: surface
properties cannot be audited, fragile probes are fully poisoned, robust probes are only partially
poisoned. And then, closing this note, the arc turned its discipline on itself: the defense half of
the story — private > naive — is confounded with probe capacity in this very harness, so the
"private-calibration defense" attribution is withdrawn until a parity control earns it back. We ran
the escalation to a floor rather than assert the trend; we are naming the confound rather than bank
the synthesis.*
