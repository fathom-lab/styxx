# STANDING PREDICTION — H1: if minds of meat behave like minds of math, cross-subject brain decoding has islands and a cliff

Fathom Lab · 2026-08-06 · **frozen before the data exists, and before we know whether we will
ever hold it.** This is not a preregistration of an experiment we are running. It is a public,
falsifiable prediction about results other groups will produce, registered in git so that our
being right cannot be claimed after the fact and our being wrong cannot be quietly deleted.

## Why we are sticking our neck out

Non-invasive thought-to-text is being scaled right now on an explicit analogy to speech
recognition: collect enough hours, and decoding accuracy climbs the way Whisper's did. The
field's own literature names the obstacle — "severe domain shift between training and unseen
test subjects" — and treats it as an engineering problem that more data smooths away.

This lab spent 2026-08-02 → 2026-08-06 measuring exactly that phenomenon in systems we *can*
operate on: four language models, concept geometries extracted, correspondence recovered
label-free, and the failures dissected causally under frozen gates
([the arc](REPLICATE_legibility.md), nine sealed acts). What we found does not look like a
data-volume problem:

- Independently trained minds **do** share a concept-frame geometry — far above random
  (`b45_result.json`: clique affinity 0.848 vs a 0.0566 random-null 95th percentile).
- Yet one model is an **island**: mostly aligned with the others (0.7166 of the same
  squared-cosine mass) and still unreadable — near-zero cross-model discovery.
- Because legibility is **switch-like in the frame coordinate** (`b46_result.json`):
  interpolating an island's frame toward a reader's leaves discovery flat across most of the
  rotation (0.0408 → 0.0434 → 0.1122 → 0.3622) and turns nearly vertical only near alignment
  (0.9566 at t = 0.8). A modest, perfectly consistent rotation costs *everything*.
- And the deficit is **low-rank and correctable**: a rank-20 frame swap restores discovery from
  0.0612 to 0.9745 while matched random frames do 0.0 (`b41_result.json`), with a rank-2 core
  carrying half of it (`b42_result.json`).

If human subjects are drawn from a shared representational geometry the same way these models
are, the same three consequences should appear in cross-subject decoding. If they do not,
minds of meat converge in a way minds of math do not — which is a *more* interesting result,
and we would rather have registered the prediction and been wrong than said nothing.

## The predictions (frozen)

**H1 — islands exist.** In a cohort decoded by a group-trained model, per-subject accuracy is
**not** unimodal-smooth: a subpopulation sits near the decoding floor while their *within-subject*
decoding is normal. The failure is subject-specific, not data-limited.

**H2 — the transition is a cliff, not a ramp.** Per-subject decoding accuracy plotted against
subject-to-group representational alignment is **better fit by a threshold (sigmoid) than by a
line**, with a narrow transition band.

**H3 — the rescue is a rotation, not more data.** For floor-sitting subjects, a **low-rank**
per-subject alignment (rank ≪ feature dimension) recovers most of the gap — the information was
present and misaddressed, not absent.

## Gates (frozen; scored by `styxx.protocol` when a qualifying dataset is in hand)

```gates
{"gates": {"G0_cohort": {"metric": "n_subjects", "op": ">=", "value": 8},
           "H1_islands": {"metric": "dip_test_p_subject_accuracy", "op": "<=", "value": 0.05},
           "H2_cliff": {"metric": "sigmoid_minus_linear_r2", "op": ">=", "value": 0.05},
           "H3_lowrank_rescue": {"metric": "median_gap_recovered_by_lowrank_align", "op": ">=", "value": 0.50}},
 "outcomes": [{"when": {"G0_cohort": false}, "verdict": "INVALID__cohort_too_small_to_score"},
              {"when": {"G0_cohort": true, "H1_islands": true, "H2_cliff": true, "H3_lowrank_rescue": true}, "verdict": "HUMAN_ISLANDS_CONFIRMED__minds_of_meat_behave_like_minds_of_math"},
              {"when": {"G0_cohort": true, "H1_islands": true, "H2_cliff": true, "H3_lowrank_rescue": false}, "verdict": "ISLANDS_AND_CLIFF__but_deficit_is_not_low_rank"},
              {"when": {"G0_cohort": true, "H1_islands": true, "H2_cliff": false}, "verdict": "ISLANDS_WITHOUT_CLIFF__graded_subject_variation"},
              {"when": {"G0_cohort": true, "H1_islands": false}, "verdict": "NO_HUMAN_ISLANDS__we_were_wrong_and_the_scaling_story_survives"}],
 "smoke_verdict": "INVALID__smoke_plumbing_only"}
```

`dip_test_p_subject_accuracy` is Hartigan's dip test for unimodality on the per-subject accuracy
vector (low p = not unimodal = island structure). `sigmoid_minus_linear_r2` compares fits of
accuracy against an alignment score. `median_gap_recovered_by_lowrank_align` uses rank ≤ 10% of
feature dimension. Bars are set where a reader can check them, not where we would like them.

## What would make us wrong, stated plainly

`NO_HUMAN_ISLANDS` is a real branch, not a courtesy. If per-subject decoding accuracy is
unimodal and improves smoothly with group data volume, our extrapolation from artificial to
biological minds fails, and we will publish that at the same volume as everything else. We are
extrapolating across a substrate boundary from **four models on one concept inventory**, which
is thin evidence for a claim about human brains — that thinness is why this is filed as a
prediction rather than a finding, and why the honest prior is uncertainty, not confidence.

## How it gets scored, and by whom

Ideally by someone else. This prediction is scoreable on any cohort of ≥ 8 subjects with
per-subject decoding accuracy, a subject-to-group alignment score, and the ability to fit a
low-rank per-subject map — including public neuroimaging corpora. If any group (Conduit
included) publishes those three quantities, the verdict computes mechanically from the block
above and we will report it whichever way it lands.

**The instrument ships with the prediction.** As of styxx 7.30.0, `styxx.islands` is the
measurement, generalized past language models — it takes any cohort of representations over a
shared item set (fMRI betas over shared stimuli, MEG epochs, decoder features) and returns the
island structure, the cliff shape, and the low-rank rescue:

```python
pip install styxx
from styxx.islands import survey, cliff, rescue
s = survey({subject_id: betas for ...})   # betas: (n_shared_stimuli, n_features)
print(s.verdict, s.islands)
```

Two deliberate refusals are wired in, because an instrument that cannot refuse cannot be
trusted: below eight members the verdict is `UNDERPOWERED__n_below_8` rather than a guess at
bimodality, and a cliff whose endpoint sits at chance returns
`REFUSED__endpoint_at_chance_no_curve_to_read` rather than a knee computed from noise. `cliff`
and `rescue` require *your* legibility measure — your decoder's accuracy, not ours. The first
draft of this module shipped an internal one; it failed its own exam against the case whose
answer we already knew, and it was removed rather than defaulted.

**Standing offer:** we will run the scoring, publish the receipt, and name the data's authors as
its authors. If someone would rather score it themselves, the gates are already frozen, the
instrument is one `pip install` away, and the machinery is [open](../../REPLICATIONS.md).

## FIRST EVIDENCE IN — 2026-08-06, and it points against us

Hours after this prediction was registered, its own precursor was tested and **failed**
([B47](FINDING_b47_no_islands_2026_08_06.md)): a ten-model cohort over an independent
96-concept battery shows **no island structure** — a shared frame, emphatically (median affinity
0.6924 vs a 0.227 random null), but a smooth affinity gradient rather than two clusters
(gap-screen p 0.7003). B47's prereg declared before the run that this verdict would weaken this
prediction. It does, and the weakening is recorded here rather than left for a reader to find.

**The narrow reading, which is the honest one:** B47 measured frame *affinity*; this prediction
is about decoding *accuracy*. Our own b46 result says the two are related by a switch, so a
smooth affinity gradient can still yield bimodal legibility if the knee falls inside the band.
That distinction favours us — which is exactly why we treat it as a debt rather than a defence.
Until the 45-pair discovery matrix is run, the prior on H1 should move **down**, and a reader
scoring this prediction later should know that its authors' first attempt to support it produced
a negative.

## FOURTH EVIDENCE IN — 2026-08-06 — H1 tested in its own currency, and there is no floor

H1a measured alignment; H1 is about **decoding accuracy**, so we measured that too
([H1b](../first-afference/FINDING_h1b_no_unreadable_minds_2026_08_06.md)): leave-one-subject-out
identification against a template built from the other seven, no fitting anywhere.

**Verdict `READABILITY_IS_CONTINUOUS__no_switch_at_this_cohort`.** Every subject is read far
above chance and the distribution is a smooth two-fold spread among the well-read
(gap-screen p 0.6523). **The least readable subject sits at 212 times chance.** H1 predicted a
subpopulation *near the decoding floor*; there is no subject near any floor.

**And the switch did not transfer.** H1b existed because b46 says alignment and legibility are
joined by a near-vertical knee, which would let continuous alignment yield bimodal readability.
Instead the two track each other almost proportionally (r 0.8825, ungated — eight points license
no fit). Either the switch is a property of that model pair rather than of minds in general, or
this cohort's alignment never approaches the knee. We cannot separate those with eight subjects
and will not claim the model result generalized.

**Status of this prediction: four independent negatives, none of them supporting it.** It is not
formally falsified — n = 8 is our instrument's floor and this is weak evidence of absence — but
nobody should read it as live. What we defend is **H3 without H1**.

## THIRD EVIDENCE IN — 2026-08-06 — tested on real human brains, and it failed

We ran H1's neural-side precursor ourselves rather than waiting for anyone else
([H1a](../first-afference/FINDING_h1a_human_single_clique_2026_08_06.md)): eight NSD subjects,
the images all of them saw, no subject dropped, prereg frozen before the data was downloaded.

**Verdict `HUMAN_SINGLE_CLIQUE__alignment_is_continuous`.** The eight brains share a
representational frame emphatically — median pairwise affinity 0.2222 against a random-frame
null p95 of 0.0248, roughly nine times above chance — and that alignment is a continuous spread
(0.1530 to 0.2575), gap-screen p 0.0779. No island. The MAD rule did flag the lowest subject,
and we report that it is **not** evidence: the same rule flags a member of a cohort of pure
noise.

That is **three independent negatives** for H1 in one day — the model-side precursor (B47), the
published human literature, and now our own measurement on human neural data. The prediction as
written is in serious trouble and we are not going to pretend otherwise.

**What still stands:** this measured alignment, and H1 is about decoding *accuracy*; our own b46
result says those are joined by a switch, so a continuous alignment distribution does not prove
a continuous accuracy distribution. And n = 8 is exactly our instrument's floor, so this is weak
evidence of absence. H1 is not falsified. It is unsupported from three directions, and the claim
we would now defend is the narrower one: **H3 without H1.**

## SECOND EVIDENCE IN — 2026-08-06 — the published human literature leans AGAINST H1

A survey of what has already been measured in humans came back, and it is a stronger negative
than B47 because it is in **H1's own currency**: per-subject decoding accuracy, not model frame
affinity. Recorded here the same day it arrived, before this prediction is promoted anywhere.

**The two best-powered per-subject distributions both look continuous.**

- **Défossez et al., "Decoding speech perception from non-invasive brain recordings," *Nature Machine Intelligence* (2023)** — n = 175 across four datasets, **no subjects
  excluded for performance**. Per-subject accuracies form an unbroken smear within every
  dataset; even the worst subjects sit well above chance. (Read off the published figure, not a
  released table — running a dip test properly would mean re-running their public
  `brainmagick` code, which is a real cost, not a lookup.)
- **Blankertz et al., "Neurophysiological predictor of SMR-based BCI performance," *NeuroImage* (2010)** — n = 80, no exclusions, and the numbers are in the text: accuracy
  "covering the full range from chance-level performance to perfect control," with the
  neurophysiological predictor relating to performance as a **linear correlation over a
  scatter, not a threshold**. That is H1 *and* H2 failing on the largest clean cohort available.

**The "BCI illiteracy" literature does not rescue us.** The familiar 15–30% figure is a utility
cutoff on a continuum, not a measured discontinuity, and the decisive detail is Vidaurre &
Blankertz (*Brain Topography* 2010): participants with no control in early runs **gained it when
the calibration method changed**. A floor that dissolves when you change the decoder is not an
island. Lee et al. (*GigaScience* 2019, n = 54 participants) sharpens it further — below-threshold rates vary
by paradigm and **no participant was universally illiterate**.

**And the flagship results are structurally incapable of testing H1.** Tang et al. (*Nat.
Neurosci.* 2023) scanned 7 subjects and used 3, with no stated exclusion criteria; the NSD
reconstruction literature evaluates on the 4 of 8 subjects who completed all sessions; and
Algonauts scores are **noise-ceiling-normalized, which divides out exactly the per-subject
variance H1 is about** — so H1 must not be scored there. Survivorship, precisely as this
document warned, but it cuts both ways: it means the field's tidy distributions are not
evidence against islands either.

**The premise is weakening too.** NEED (*NeurIPS* 2025) reports retaining ~94% of within-subject
performance on unseen subjects in EEG visual decoding. If that replicates, "cross-subject
transfer is catastrophic" — the intuition the whole island framing leans on — is itself in
question.

**What we conclude, against our own interest.** The prior on H1 moves **down again**. No paper
has applied a formal bimodality test to a per-subject decoding distribution, so H1 is
*unfalsified but unsupported*, and every adjacent distribution described in words is described
as continuous. The claim that survives the evidence is narrower and better supported: **floor
membership is a decoder–subject interaction that a per-subject realignment removes** — Vidaurre's
runs 4–6, Lee's paradigm specificity. That is **H3 without H1**. We are recording that reduction
here rather than quietly retreating to it later.

## THIS IS SCOREABLE TODAY — concrete routes, added 2026-08-06

When this prediction was registered we said it was scoreable "on any cohort of ≥ 8 subjects,
including public neuroimaging corpora." A survey of what is actually downloadable turned that
from a gesture into a work order. Every route below was verified reachable; sizes are measured,
not estimated. **We are naming these so that nobody — including us — can later claim the
prediction was unfalsifiable in practice.**

| route | n | shared items | size / friction | why it fits |
|---|---|---|---|---|
| **NSD via the MindEye2 mirror** (`huggingface.co/datasets/pscotti/mindeyev2`) | 8 | `shared1000` seen by all | **12.35 GB**, ungated, no DUA | GLMsingle betas in the `nsdgeneral` ROI; the subject × item × voxel tensor H3 needs *is the file format* |
| **BOLD Moments** (OpenNeuro `ds005165`) | **10** | **1,102 videos seen by every subject** | 61.8 GB useful slice, CC0, no registration (the full derivatives tree is 3.67 TB — never sync it whole) | the cleanest sharing design surveyed; two more subjects than NSD and ~3× the shared items |
| **THINGS-EEG2** (`osf.io/anp5v`, preprocessed) | 10 | **all 16,740 images seen by all**, 200 at 80 reps | a few GB preprocessed | strongest sharing structure in existence |
| **ERP CORE** (`osf.io/thsqg`) | **40** | 6 paradigms, all 40 completed all 6 | 24 GB, CC-BY, seven MOABB one-liners | **six independent replications of the same dip test on the same 40 people** — a far stronger H1 test than one cohort |
| **Narratives** (`s3://fcp-indi/data/Projects/narratives/`) | **82** (pieman) | one story heard by all | ~140 GB, CC0, anonymous S3, fMRIPrep derivatives shipped | the canonical benchmark for subject→group alignment, which is exactly H3's construction |
| **Liu2020BETA** via MOABB | **70** | 40 identical SSVEP targets | 5.28 GB, one-line loader | an afternoon's sanity check — caveat: SSVEP decoding is near-saturated, so ceiling effects may make the variance uninformative |

Ruled out on cohort size despite being obvious candidates: Algonauts 2025 (n=4), THINGS-MEG
(n=4), THINGS-fMRI (n=3), BOLD5000 (n=4), LibriBrain (one subject).

**The cheapest decisive move needs no download at all:** several papers already publish
per-subject decoding accuracies as a figure. Re-scoring those published values with a
unimodality test would settle H1's core claim against data that already exists. If those
distributions are smooth and unimodal, we are wrong and should say so.

**One caution we hold against ourselves:** papers routinely *exclude* low-performing subjects,
which would remove exactly the structure H1 predicts before anyone could see it. A cohort
reported as unimodal after exclusions is not evidence against islands — and equally, we must
not treat every exclusion as a hidden island. Any scoring of this prediction must report the
source's exclusion criteria alongside the verdict.

## ERRATUM — 2026-08-06, same day, before any scoring

**The gate names a statistic our own shipped instrument does not compute.** `H1_islands` reads
`dip_test_p_subject_accuracy` — Hartigan's dip. `styxx.islands.survey` returns `bimodality_p`,
a *gap-based* unimodality screen (largest normalized gap in the sorted values against a matched
unimodal null). Those are different tests. Caught by auditing this document against the module
hours after both were written; recorded here rather than fixed silently, because the gates are
frozen and rewriting them after publication is the exact move this lab exists to refuse.

**How the gate is satisfied, stated now rather than argued later:** `H1_islands` is satisfied by
**any documented unimodality test reported with its explicit null**, at p ≤ 0.05 on the
per-subject accuracy vector. Hartigan's dip (e.g. the `diptest` package) is the reference
implementation and the one the metric name intends; `styxx.islands`' gap screen is an
acceptable substitute *only* if reported under its own name with its null described. A scorer
who runs both and finds they disagree should report both — that disagreement would itself be
worth publishing.

This erratum changes no bar and no branch. It fixes a naming mismatch that would otherwise have
let us claim a pass under whichever statistic happened to be kinder.

*A prediction registered before the data is worth more than an explanation offered after it.*
