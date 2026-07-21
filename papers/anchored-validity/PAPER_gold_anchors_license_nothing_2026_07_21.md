# Gold Anchors License Nothing: label-free auditing of LLM judge panels, an instrument that refuses, and the measured failure of sanity checks

Fathom Lab — 2026-07-21. Slated as the next version of the Fathom paper series (deposit
operator-gated). Instrument: `styxx.anchors` (released, styxx 7.26.0, PyPI). Every number in
this document is bound to a committed receipt and machine-verified by `styxx.certify`;
preregistrations were frozen and committed before every scored run, and the negative and
refused outcomes below are reported under their frozen bars.

## Abstract

LLM judge panels are the de facto measurement apparatus of modern AI evaluation, and the de
facto validation of those judges is a set of blatant sanity checks: exact-duplicate pairs,
direct negations, honeypots — the crowdsourcing gold-question practice, transplanted. We built
a label-free auditor for judge panels (anchored moment identification with refusal semantics
and measured operating characteristics), sealed its own datasheet on simulation (nine of nine
calibration gates), and then ran the measurement the practice assumes away: do blatant gold
checks license the numbers a panel produces on real work? Across four constructed task
families on a real correlated weak-judge panel, the answer is no at effect sizes that destroy
every estimator tested: label-free coverage 0/15 in every family, while the same panel is
flawless on the gold items. The failure is SILENT whenever the violation is smooth — our
calibrated misfit flag fired on 15/15 replicates in one family and 1/15 in another. The
repair is construction, not statistics: anchor strata drawn from the same generating process
as the work items ("ladder anchors") gave the correct answer in both of its possible
directions — restoring calibrated coverage 13/13 when exactly one informative judge existed,
and REFUSING (14/15 VOID) when honest anchors revealed that no judge was informative at all.
Gold anchors certified garbage in both cases. A frontier-model panel audited at arm's length
under the same protocol was priced exactly, at ceiling, on every sheet including a
deconfounded multi-hop task — locating the danger zone in the weak-to-mid judge tier, which
is precisely the tier deployed for cheap bulk evaluation. The honest gold set does not merely
fix the estimate; it tells you whether you have a panel at all.

## 1. The impossibility, and what breaks it

Every ground-truth-free evaluator-accuracy estimator in the literature — Dawid-Skene (1979)
through its modern weak-supervision descendants — requires conditional independence of judge
errors given the true label, or a modeled correlation graph. Panels of LLM judges violate
this by construction: shared base models, shared prompt templates, shared vulnerabilities. At
three judges the confusion is exact — a correlated panel and an independent panel can induce
identical joint verdict distributions on unlabeled data (exhibited constructively in our
Stage-A witness; the equivalence is generic at that panel size). Known-label anchors break
the impossibility: items whose truth holds by construction make per-judge error rates and
error correlations observable, replacing the independence assumption with an anchor-
exchangeability assumption. The entire content of this paper is that the exchangeability
assumption is where the danger lives, that the standard practice violates it silently, and
that an instrument can be built to price, flag, and refuse accordingly.

## 2. The instrument and its sealed datasheet

`styxx.anchors.audit_panel(V, neg, pos)` consumes an n-by-J verdict matrix and two anchor
strata and returns either a refusal that names why (`VOID_PANEL__uninformative`;
`VOID_ANCHORS__nonexchangeable` when the unclipped solution is impossible — never a clipped
confident number) or an estimate wrapped in its own measured operating characteristics: a
selection-aware bootstrap interval (the sync parameter activates only on evidence, and every
bootstrap resample re-selects, so the interval prices selection uncertainty), a regime-keyed
coverage note quoting measured rather than nominal rates, and a per-dataset misfit p-value
from a parametric-bootstrap null with its power scope stated.

The characterization run sealed all nine calibration gates at the Stage-A design point:
interval coverage 0.95 on clean validation panels (Wilson [0.888, 0.978]), 0.963 under a
rho=0.30 shared failure factor, 0.912 and 0.938 at planted master-key rates 0.05 and 0.15;
misfit false-alarm 0.090 on held-out clean replicates; deaf-panel VOID rate 0.967 under the
plain informativeness gate and 1.000 under the noise-margin gate the instrument ships with;
refusal false-fire 0/200; clean activation rate 0.020. Activation power against a true
all-judge key: 0.30 / 0.7125 / 1.00 at wild rates 0.02 / 0.05 / 0.15 — below roughly a
five-percent key rate, absence of activation is not evidence of absence. Two prior interval
constructions failed their coverage gates (0.835 and 0.850 on clean panels) and are on the
public record as CLOSED_NEGATIVE with the diagnosed mechanism; the selection-aware
construction is the one that sealed.

## 3. The kill: gold anchors on a real panel

The panel: four prompt-personas of a small instruction-tuned model (Qwen2.5-3B-Instruct,
greedy), a real correlated panel of the kind deployed for bulk evaluation. The corpora:
constructed-truth contradiction detection in four families (attribute facts, numeric
consistency, temporal ordering, transitivity chains), each with a graded difficulty ladder,
with truth held out from the judges. The gold anchors: verbatim pairs (negatives) and direct
negations (positives) — the blatant ends an auditor can always construct.

Fifteen preregistered replicates per family. Label-free coverage of the true prevalence:
0/15, 0/15, 0/15, 0/15. Median audit errors 0.4658 (attribute), 0.2774 (numeric), 0.65
(temporal — one judge's anchor-to-organic false-fire gap reached -0.9992), 0.4444 (chains).
The mechanism is measured, not argued: the judges catch every planted negation (anchor-to-
organic sensitivity gap 0.0) and false-fire on 66 to 98 percent of REPHRASED-consistent
organic items — "different words means different claims" — a failure mode a verbatim pair is
structurally incapable of exhibiting. No estimator survives from the inside: majority vote
and Dawid-Skene sit at median errors 0.55 to 0.60, semi-supervised DS with the gold answers
in hand 0.53 to 0.54, and the anchored audit is least wrong and still hopeless. The deaf
control — the same personas judging redacted content — was refused by the instrument in
90/90 replicates across the entire program: a content-free panel cannot extract a number.

The silence is the sharpest edge. The calibrated misfit flag — built precisely because the
sealed datasheet showed smooth violations evade internal-consistency checks — fired on 15/15
wrong estimates in the numeric family, 14/15 in the attribute family, and 0/15 (temporal),
1/15 (chains) in the others: violations that bend every moment coherently leave nothing for
a label-free check to see. Four families, one sentence: gold anchors license nothing, and
sometimes nothing warns you.

## 4. The repair: ladders that estimate, and ladders that refuse

Anchor strata drawn from the same generator and difficulty mix as the organic items — still
constructible without labels, because the constructor plants the truth — close the measured
anchor-to-organic false-fire gap from 0.63 to 0.0322 (attribute) and to 0.0078 (chains). What
happens next depends on what honest anchors reveal, and the instrument answered correctly in
both directions. On the attribute family they exposed three of the four personas as
uninformative (the plain gold anchors had certified all four) and the audit of the one real
judge restored coverage 13/13 at median error 0.0626. On the chain family they revealed that
NO judge was informative, and the instrument refused: VOID 14/15 — an outcome named in the
preregistration before the run, scored CLOSED_NEGATIVE under its frozen coverage clause, and
reported as what it is. The honest gold set does not merely fix the estimate; it tells you
whether you have a panel at all.

## 5. The frontier control, and a confound we caught ourselves

The same arm's-length protocol (fresh-context judge sessions receiving only the task and the
shuffled items; every verdict logged verbatim; the transcription trust boundary disclosed)
was run on four personas of a frontier model. On every sheet — the attribute corpus, ordered
multi-hop chains, and chains DECONFOUNDED after we caught our own generator emitting
statements in rank order (making multi-hop queries positionally shortcuttable; the overclaim
was demoted in a preregistration amendment before the corrected run) — the frontier panel
scored 1.0 on all four personas, and the label-free audit priced it exactly: estimate equal
to the held-out truth to machine precision, interval covering, every honesty mechanism idle.
Twice more the four personas returned byte-identical verdict sequences, as they had on the
first sheet: persona diversity is not error diversity; a frontier model in costumes is one
judge, which is exactly the correlated world this instrument assumes. The danger zone for
anchor non-transfer is the weak-to-mid judge tier.

## 6. Foundation, scope, residuals

Every organic and anchor label in every scored corpus — 36,720 items — was re-derived from
item text alone by independent decision logic (a formal transitive-closure oracle for chains;
slot, number, and form oracles for the rest): zero mismatches, zero undecidable. All gates
preregistered with frozen bars; every miss reported verbatim; the program's public record
across this arc includes two interval constructions closed negative, a repair closed negative
with its mechanism decomposed, a refusal resolution, and one self-caught confound demoted in
writing before its corrected experiment ran.

Scope, stated plainly: one weak-judge base model carries the kill (an 8 GB device ceiling;
the cross-vendor API arm is recorded as blocked on credits); all corpora are constructed;
the frontier tier is measured only at ceiling because constructed ladders exhaust there.
The named residuals — API-tier model generality, at least one in-the-wild evaluation setup
where practitioners actually salt gold checks, and a naturalistic frontier-stressing ladder —
bound the claim; they do not soften what is measured inside it.

## 7. Reproducibility

`pip install styxx` (7.26.0 or later); repository fathom-lab/styxx, `papers/anchored-validity/`.
Preregistrations, harnesses, seeds, checkpoints, per-judge verdict receipts, oracle receipts,
and OATH certificates for every finding are committed. The instrument that produced these
numbers ships with its weaknesses printed on the label, and this document was number-verified
by the same certifier that polices the rest of the program.
